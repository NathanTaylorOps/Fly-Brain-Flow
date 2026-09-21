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

    @classmethod
    def from_scenario(cls, scenario, rng: np.random.Generator | None = None) -> "Sim":
        return cls(scenario, rng=rng)

    def tick(self, dt: float) -> None:
        """Advance the whole sim by `dt` seconds: spawn, steer (each cohort through its own
        brain), move (one shared physics call), feed, leave -- the same five-step order
        `agents.py`'s own docstring lays out, now covering every cohort at once instead of one."""
        sc = self.scenario
        for pop in self.cohorts.values():
            pop.spawn(dt)

        walking = {tag: pop.walking_ids() for tag, pop in self.cohorts.items()}
        active = [tag for tag in self.cohorts if walking[tag]]
        if active:
            positions = np.concatenate([self.cohorts[tag].positions(walking[tag]) for tag in active], axis=0)
            radii = np.full(len(positions), sc.agents.radius_m)
            velocities = np.concatenate(
                [
                    self._steer(tag, walking[tag], self.cohorts[tag].positions(walking[tag]), dt)
                    for tag in active
                ],
                axis=0,
            )
            new_positions, _ = physics_step(
                positions, velocities, radii, dt, self.map, max_speed_mps=sc.agents.max_speed_mps
            )
            offset = 0
            for tag in active:
                n = len(walking[tag])
                self.cohorts[tag].set_positions(walking[tag], new_positions[offset : offset + n])
                offset += n

        for tag, pop in self.cohorts.items():
            # `_targets.assigned` (id -> committed target index) is the same dict `_steer` just
            # used this tick to route this cohort's flies -- passing it to `feed` means an agent
            # can only be captured at the target it's actually walking to, not any target that
            # happens to be within capture radius (see `Population.feed`'s own docstring for the
            # bug this fixes: a fly spawned right next to the *other* stream's target used to be
            # scooped up on tick one, before it ever took a step toward its own).
            assigned = self.brains[tag]._targets.assigned
            pop.feed(dt, assigned_targets=assigned)  # default capture radius -- no scenario dial yet
            left = pop.leave()
            if left:
                self.brains[tag].forget(left)

        self.t += dt

    # -- internals ------------------------------------------------------

    def _steer(self, tag: str, ids: list[int], positions: np.ndarray, dt: float) -> np.ndarray:
        """One cohort's desired velocities. `ToyBrain` and `Baseline` don't share a call
        signature -- the toy (and eventually real) brain needs personalities, the odour field and
        `dt` for its own leaky activity update; `Baseline` is a stateless steering formula with
        none of that. Dispatching on type is a small, honest shim for there being exactly two
        brain shapes right now; if a third ever needs its own signature, this is the one place
        that grows a branch, not every caller of `Sim`."""
        sc = self.scenario
        brain = self.brains[tag]
        pop = self.cohorts[tag]
        radii = np.full(len(ids), sc.agents.radius_m)
        # A source's `source_targets` pin (see `agents.SpawnConfig`) travels with the agent as
        # `Agent.preferred_target`; only ids that actually have one need to be in this dict, and
        # an id with no preference is simply absent -- `TargetAssignment.assign` treats absent and
        # None the same way (fall back to nearest).
        preferred = {i: pop.agents[i].preferred_target for i in ids if pop.agents[i].preferred_target is not None}
        if isinstance(brain, ToyBrain):
            personalities = [pop.agents[i].personality for i in ids]
            return brain.desired_velocities(
                ids, positions, radii, sc.agents.max_speed_mps, personalities, self.odor_field, dt, preferred
            )
        return brain.desired_velocities(ids, positions, radii, sc.agents.max_speed_mps, preferred)
