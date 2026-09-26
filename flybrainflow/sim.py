"""The sim loop: ties the map, odour/wind, agent lifecycle, brains and physics into one runnable
tick, per-cohort where it matters and shared where it must be.

Every piece here already exists and is already tested on its own -- `agents.py`, `brains/`,
`world/`. This is the "in this order, every tick" wiring each of those modules' own docstrings
already describe, actually written down and run end to end for the first time, rather than left
implicit as an example in three separate places that have never been run together.

    sim = Sim.from_scenario(scenario)
    for _ in range(n_ticks):
        sim.tick(dt)
    sim.cohorts["brain"].agents         # current fly-brained agents
    sim.cohorts["baseline"].agents      # current baseline agents (only if [baseline] enabled)
    sim.move_target(0, (12.0, 4.0))     # relocate a target mid-run; also fires automatically from
                                         # the scenario's own `[[target_moves]]` schedule, if it has one

One `physics.step()` call per tick, over every walking agent from every cohort combined -- see
`agents.py`'s own docstring for why that sharing (also target slots, also agent ids) has to happen
at all: a fly-brained agent and a baseline agent genuinely need to see and avoid each other, not
run as two simulations layered on top of one another that happen to share a picture. Each cohort's
own brain only ever sees and steers its own agents; physics is the only place the two cohorts
actually interact.
"""

from __future__ import annotations

import numpy as np

from .agents import Population, build_cohorts
from .brains import Baseline, ToyBrain
from .world import load_map_for_scenario, odor_field_for_scenario, physics_step
from .world.airflow import AirflowField
from .world.map import WalkableMap


class Sim:
    """Owns one scenario's map, odour field, cohorts and brains, and knows how to advance all of
    it by one `dt`. Every piece can be supplied instead of built (tests do this to swap in a
    smaller map, a fixed rng, or a stub brain) -- left as `None`, `Sim` builds the same thing
    `Sim.from_scenario` would."""

    def __init__(
        self,
        scenario,
        walkable_map: WalkableMap | None = None,
        odor_field=None,
        cohorts: dict[str, Population] | None = None,
        brains: dict[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.scenario = scenario
        self.map = walkable_map if walkable_map is not None else load_map_for_scenario(scenario)
        if odor_field is None:
            airflow = AirflowField.build(self.map)
            odor_field = odor_field_for_scenario(scenario, walkable_map=self.map, airflow=airflow)
        self.odor_field = odor_field
        self.cohorts = cohorts if cohorts is not None else build_cohorts(scenario, rng=rng)
        if brains is None:
            brains = {"brain": ToyBrain(self.map, scenario.targets, scenario_seed=scenario.meta.seed)}
            if "baseline" in self.cohorts:
                brains["baseline"] = Baseline(self.map, target_positions=[t.position for t in scenario.targets])
        self.brains = brains
        self.t = 0.0
        # Sparse log of every `move_target()` call, in the shape `viewer_export.export_for_viewer`'s
        # own `target_moves` parameter expects -- (t, index, position) triples, in call order (not
        # necessarily sorted; that function sorts them itself). Kept here rather than inside
        # `Recorder` since a move is a rare, scheduled event, not a per-tick sample -- there's
        # nothing to "capture" every tick the way agent positions are.
        self.target_moves: list[tuple[float, int, tuple[float, float]]] = []
        # `scenario.target_moves` (an optional `[[target_moves]]` TOML schedule) is static config;
        # this is the live, mutable queue of what's left to fire, sorted once up front and popped
        # from the front as `tick()` catches up to each one's `at_s`.
        self._pending_moves = sorted(scenario.target_moves, key=lambda m: m.at_s)

    def move_target(self, index: int, new_xy) -> None:
        """Move target `index` to `new_xy`, live, mid-run -- the one operation everything that
        cares where a target is has to agree on, so a move can't leave odour, routing, capture and
        the viewer marker looking at different positions (see docs/M1_PLAN.md's Step 0 for the bug
        this fixes: each of those used to read the scenario's original, frozen config and never
        looked again).

        Three things happen, in order, and all three matter:
          1. The `OdorSource` actually moves (`OdorField.sample`/`.gradient` read `.position` fresh
             every call, and `OdorField`'s own geodesic cache keys on it too, so smell and
             wind-occlusion routing both pick this up for free, no separate invalidation needed).
          2. Every brain in `self.brains` gets told, not just whichever cohort is on screen --
             `Baseline` and `ToyBrain` (and the real connectome brain later) each hold their own
             `TargetAssignment`, so a target moving has to reach both cohorts' copies or one of
             them would keep routing/capturing at the stale position while the other one updated,
             quietly breaking the fly-brain-vs-baseline comparison the whole project measures.
             `move_target` is looped generically, the same "one call shape, no isinstance check"
             rule already used for `desired_velocities`/`assigned_targets`.
          3. The move is logged to `self.target_moves`, so a recording actually shows it -- see
             `viewer_export.export_for_viewer`'s own `target_moves` parameter.

        Deliberately does not touch `self.scenario.targets` -- `Scenario`/`TargetConfig` are frozen
        on purpose (the static, as-configured venue), so "where a target currently is" lives only
        in `self.odor_field.sources[index]` and each brain's own `TargetAssignment`, never in a
        second copy that could drift out of sync with them.
        """
        xy = (float(new_xy[0]), float(new_xy[1]))
        self.odor_field.sources[index].move_to(xy)
        for brain in self.brains.values():
            brain.move_target(index, xy)
        self.target_moves.append((self.t, index, xy))

    @classmethod
    def from_scenario(cls, scenario, rng: np.random.Generator | None = None) -> "Sim":
        return cls(scenario, rng=rng)

    def tick(self, dt: float) -> None:
        """Advance the whole sim by `dt` seconds: fire any scheduled target moves, spawn, steer
        (each cohort through its own brain), move (one shared physics call), feed, leave -- the
        same five-step order `agents.py`'s own docstring lays out (now covering every cohort at
        once instead of one), with the scheduled-move check ahead of all of it.

        Moves fire before `spawn`/`_steer` in the *same* tick they become due (`self.t` is this
        tick's start time, not yet advanced) specifically so an already-walking agent re-routes
        using the new position this tick, not one tick late -- matters for the "visibly re-routes"
        behaviour M1's own done-when line and the calibration gate both test for."""
        sc = self.scenario
        while self._pending_moves and self._pending_moves[0].at_s <= self.t:
            move = self._pending_moves.pop(0)
            self.move_target(move.target, move.to)

        for pop in self.cohorts.values():
            pop.spawn(dt)

        walking = {tag: pop.walking_ids() for tag, pop in self.cohorts.items()}
        active = [tag for tag in self.cohorts if walking[tag]]
        if active:
            positions = np.concatenate([self.cohorts[tag].positions(walking[tag]) for tag in active], axis=0)
            radii = np.full(len(positions), sc.agents.radius_m)
            # Slice this cohort's positions back out of the array just built, rather than querying
            # `Population.positions()` a second time for the same ids -- same offset-bookkeeping
            # pattern used below for `new_positions`, so there's one technique for "this cohort's
            # slice", not two.
            offset = 0
            velocity_parts = []
            for tag in active:
                n = len(walking[tag])
                velocity_parts.append(self._steer(tag, walking[tag], positions[offset : offset + n], dt))
                offset += n
            velocities = np.concatenate(velocity_parts, axis=0)
            new_positions, _ = physics_step(
                positions, velocities, radii, dt, self.map, max_speed_mps=sc.agents.max_speed_mps
            )
            offset = 0
            for tag in active:
                n = len(walking[tag])
                self.cohorts[tag].set_positions(walking[tag], new_positions[offset : offset + n])
                offset += n

        for tag, pop in self.cohorts.items():
            # `assigned_targets()` (id -> committed target index) reflects whatever `TargetAssignment.
            # assign` decided inside this tick's `_steer` call -- passing it to `feed` means an
            # agent can only be captured at the target it's actually walking to, not any target
            # that happens to be within capture radius (see `Population.feed`'s own docstring for
            # the bug this fixes: a fly spawned right next to the *other* stream's target used to
            # be scooped up on tick one, before it ever took a step toward its own).
            assigned = self.brains[tag].assigned_targets()
            # `target_positions` -- current positions, in `scenario.targets` order -- comes from
            # `self.odor_field.sources` rather than `sc.targets` directly: `Scenario`/`TargetConfig`
            # are frozen (the static, as-configured venue), so a moved target's live position exists
            # only on its `OdorSource` (see `move_target`'s own docstring). Without this, `feed`
            # falls back to `sc.targets[ti].position` and a moved target could never be captured at
            # its new spot, or could still be captured at its old one.
            live_positions = [s.position for s in self.odor_field.sources]
            pop.feed(dt, assigned_targets=assigned, target_positions=live_positions)  # default capture radius
            left = pop.leave()
            if left:
                self.brains[tag].forget(left)

        self.t += dt

    # -- internals ------------------------------------------------------

    def _steer(self, tag: str, ids: list[int], positions: np.ndarray, dt: float) -> np.ndarray:
        """One cohort's desired velocities. Every brain shares one call signature (see
        `Baseline.desired_velocities`/`ToyBrain.desired_velocities`'s own docstrings) specifically
        so this method can call whichever brain it's holding the same way, without checking its
        type -- a brain that doesn't use `personalities`/`odor_field`/`dt` (like `Baseline`) just
        ignores them. Adding a third brain shape means giving it the same signature, not adding a
        branch here."""
        sc = self.scenario
        brain = self.brains[tag]
        pop = self.cohorts[tag]
        radii = np.full(len(ids), sc.agents.radius_m)
        # A source's `source_targets` pin (see `agents.SpawnConfig`) travels with the agent as
        # `Agent.preferred_target`; only ids that actually have one need to be in this dict, and
        # an id with no preference is simply absent -- `TargetAssignment.assign` treats absent and
        # None the same way (fall back to nearest).
        preferred = {i: pop.agents[i].preferred_target for i in ids if pop.agents[i].preferred_target is not None}
        personalities = [pop.agents[i].personality for i in ids]
        return brain.desired_velocities(
            ids, positions, radii, sc.agents.max_speed_mps, preferred,
            personalities=personalities, odor_field=self.odor_field, dt=dt,
        )
