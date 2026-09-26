"""The baseline: what a standard crowd model does, for comparison against the real fly brain.

The plan names PySocialForce (Helbing & Molnár's social-force model) as the reference algorithm
-- but the plan also makes the *same* physics solver a hard requirement for every agent type, so
the comparison is of steering decisions, not of two different simulators (see
`world.physics`'s own docstring). Importing PySocialForce wholesale would mean baseline agents
move through *its* collision/movement engine, not ours -- so what's here is a small, self-written
reimplementation of just the steering equations (goal-seeking plus exponential pedestrian and
wall repulsion, the same shapes PySocialForce uses), producing `desired_velocities` for our own
`physics.step()` to actually move people with. Credited as the algorithmic source in
ATTRIBUTION.md, not vendored as a dependency.

Goal-seeking follows the wall-aware geodesic field from `world.geodesic`, not a straight line to
the target -- a straight-line bearing walks an agent straight into the first wall of a stadium or
train-station venue and leaves it stuck there (physics.step's wall-push holds it just off the
wall forever, going nowhere), which is exactly the failure this project's odour routing already
had to solve once. Reusing that same machinery here means a baseline agent actually finds its way
around a barrier instead of stalling at it.

    brain = Baseline(walkable_map, target_positions=[[5.0, 1.0]])
    vel = brain.desired_velocities(ids, positions, radii, max_speed_mps=1.3)
    brain.forget(pop.leave())   # call this with whatever `Population.leave()` returns each tick

Target commitment (nearest-first, sticks once decided, deliberately doesn't re-route around a busy
target) is shared logic, not reimplemented here -- see `TargetAssignment` in `brains/targeting.py`
for the full rationale.
"""

from __future__ import annotations

import numpy as np

from ..world.geodesic import gradient_direction
from ..world.map import WalkableMap
from .repulsion import agent_repulsion, wall_repulsion
from .targeting import TargetAssignment


class Baseline:
    """A standard social-force-flavoured steering model: no smell, no brain, just geometry --
    walk toward your target along the shortest walkable path, don't walk through people or
    walls. This is what fly-brained agents are measured against."""

    def __init__(self, walkable_map: WalkableMap, target_positions):
        self.map = walkable_map
        try:
            self._targets = TargetAssignment(walkable_map, target_positions)
        except ValueError:
            raise ValueError("Baseline needs at least one target position to steer toward") from None
        self.target_positions = self._targets.target_positions
        # Same dict objects as `self._targets.{assigned,fields}`, not copies -- kept as direct
        # attributes because existing tests (and any external code) already reach in as
        # `brain._assigned`; `self._targets` is the actual implementation underneath.
        self._assigned = self._targets.assigned
        self._fields = self._targets.fields

    def desired_velocities(
        self,
        ids,
        positions,
        radii,
        max_speed_mps,
        preferred_targets: dict | None = None,
        *,
        personalities=None,
        odor_field=None,
        dt: float | None = None,
    ) -> np.ndarray:
        """One steering vector per id in `ids`, in the same order. `positions`/`radii` line up
        with `ids` the same way `world.physics.step()` expects them.

        `preferred_targets` (id -> target index) overrides nearest-distance assignment for an id's
        *first* assignment only -- see `TargetAssignment.assign`'s own docstring for why nearest
        alone isn't always enough. Omit it (the default) for ordinary nearest-target behaviour.

        `personalities`/`odor_field`/`dt` are accepted and ignored -- `Baseline` has no smell and
        no persistent internal state, so none of the three apply to it. They exist here only so
        `Baseline` and `ToyBrain` (and whatever the real connectome-driven brain ends up being)
        share one call signature: `Sim._steer` calls every brain the same way, by keyword, rather
        than checking which brain type it's holding and calling it differently -- a dispatch that
        would only grow branches as more brain types are added, not shrink."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        n = len(ids)
        if n == 0:
            return np.zeros((0, 2))
        radii = np.broadcast_to(np.asarray(radii, float), (n,))
        max_speed = np.broadcast_to(np.asarray(max_speed_mps, float), (n,))

        target_idx = self._assign(ids, positions, preferred_targets)
        goal_dir = self._goal_direction(target_idx, positions)
        social = agent_repulsion(positions, radii)
        wall = wall_repulsion(self.map, positions)

        raw = goal_dir * max_speed[:, None] + social + wall
        speed = np.linalg.norm(raw, axis=1)
        too_fast = speed > max_speed
        scale = np.ones(n)
        scale[too_fast] = max_speed[too_fast] / np.maximum(speed[too_fast], 1e-12)
        return raw * scale[:, None]

    def assigned_targets(self) -> dict[int, int]:
        """The live id -> committed-target-index mapping, for callers (namely `Sim.tick`, to
        restrict `Population.feed`'s capture to an agent's own assigned target) that need to know
        what this brain decided without reaching into `self._targets` directly."""
        return self._targets.assigned

    def forget(self, ids) -> None:
        """Drop per-agent state (its committed target) for ids that are gone for good -- fed and
        left, in the usual case. Without this, `_assigned` grows for as long as the process runs:
        every id that was ever spawned stays in it forever, even once only a handful of agents are
        actually alive at once, which adds up over a long-running or live session. Call this with
        whatever `Population.leave()` (or any other removal) returns, each tick; forgetting an id
        that's still walking is harmless -- it's just reassigned the next time it's seen."""
        self._targets.forget(ids)

    def move_target(self, index: int, new_xy) -> None:
        """Relocate target `index` to `new_xy` -- see `TargetAssignment.move_target` for the full
        rationale (already-assigned agents stay committed to `index` and just get a correct field
        for its new position; nothing needs reassigning). `self.target_positions` is the same list
        object as `self._targets.target_positions` (see `__init__`), so it reflects this move
        immediately with no further action here."""
        self._targets.move_target(index, new_xy)

    # -- internals ------------------------------------------------------

    def _assign(self, ids, positions, preferred_targets: dict | None = None) -> np.ndarray:
        return self._targets.assign(ids, positions, preferred_targets)

    def _goal_direction(self, target_idx: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Unit vector at each position pointing toward decreasing geodesic distance to its
        assigned target -- "downhill" on the shortest-walking-distance field, so it bends around
        whatever's in the way instead of aiming through it."""
        out = np.zeros_like(positions)
        for ti in np.unique(target_idx):
            rows = target_idx == ti
            field = self._targets.field_for(int(ti))
            # Distance decreases toward the target, so head the opposite way from the field's own
            # gradient (which points toward increasing distance, i.e. away from the target).
            out[rows] = -gradient_direction(self.map, field, positions[rows])
        return out
