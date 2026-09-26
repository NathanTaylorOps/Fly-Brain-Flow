"""Shared plumbing every brain needs to walk toward a fixed target, so `Baseline` and `ToyBrain`
(and whatever the real connectome-driven brain ends up being) don't each keep their own copy of
the same three things: commit to the nearest target once and stick with it, cache each target's
wall-aware distance field instead of recomputing it every tick, and release both once an agent is
gone for good. Extracted after the baseline steering model and the toy brain turned out to need
the exact same three pieces almost verbatim.
"""

from __future__ import annotations

import numpy as np

from ..world.geodesic import geodesic_distance_field
from ..world.map import WalkableMap


class TargetAssignment:
    """Nearest-target commitment (straight-line, "which one looks closest from here"), kept once
    decided -- an agent does not re-route away from a target that turns out to be busy, since
    suppressing that would suppress the queuing/choke-point behaviour this project exists to
    measure. Also caches each target's geodesic distance field, built once and reused by every
    agent that ends up assigned to it."""

    def __init__(self, walkable_map: WalkableMap, target_positions, connectivity: int = 8):
        self.map = walkable_map
        self.target_positions = [np.array(t, float) for t in target_positions]
        if not self.target_positions:
            raise ValueError("need at least one target position to assign agents to")
        self.connectivity = connectivity
        self.assigned: dict[int, int] = {}
        self.fields: dict[int, np.ndarray] = {}

    def assign(self, ids, positions: np.ndarray, preferred: dict | None = None) -> np.ndarray:
        """Target index per id, in the same order as `ids`. Deciding is a one-time event per id;
        already-assigned ids just get their existing choice looked back up.

        `preferred` (id -> target index, or id absent/None meaning "no preference") overrides the
        nearest-distance pick for that id's first assignment. This exists because "nearest" alone
        silently breaks a scenario where a source sits right next to a target meant for a
        *different* stream to walk to -- see `agents.SpawnConfig.source_targets`'s own docstring
        for the full story (the two-stream corridor test is exactly this case). Once an id is
        assigned, `preferred` has no further effect on it, same as nearest-distance assignment."""
        for i, p in zip(ids, positions):
            if i not in self.assigned:
                pref = None if preferred is None else preferred.get(i)
                if pref is not None:
                    self.assigned[i] = int(pref)
                else:
                    dists = [np.linalg.norm(p - t) for t in self.target_positions]
                    self.assigned[i] = int(np.argmin(dists))
        return np.array([self.assigned[i] for i in ids], dtype=int)

    def field_for(self, target_index: int) -> np.ndarray:
        if target_index not in self.fields:
            self.fields[target_index] = geodesic_distance_field(
                self.map, self.target_positions[target_index], connectivity=self.connectivity
            )
        return self.fields[target_index]

    def forget(self, ids) -> None:
        """Release ids that are gone for good. Without this, `assigned` grows for as long as the
        process runs -- every id ever spawned stays in it forever, even once only a handful of
        agents are actually alive at once (first caught as an unbounded-memory-growth bug in
        `Baseline`, before this bookkeeping was pulled out into shared code)."""
        for i in ids:
            self.assigned.pop(i, None)

    def move_target(self, index: int, new_xy) -> None:
        """Relocate an existing target in place -- update `target_positions[index]` and drop that
        index's cached field, without touching who's assigned to it.

        This is item assignment on the existing `target_positions` list, not a rebind of
        `self.target_positions` to a new list object: `Baseline`/`ToyBrain` (and anything else
        holding a reference to this same list, e.g. as `brain.target_positions`) must see the
        update through their own reference, which only works if the list object itself never
        changes identity.

        `self.assigned` is deliberately untouched. Commitment is sticky by target *index*, not by
        the position that index happened to occupy at assignment time -- see `assign`'s own
        docstring for why an agent doesn't get to re-route once committed. An agent already
        assigned to `index` is still committed to *that target*, wherever it now is; it just needs
        a correct distance field to walk toward it, which is exactly what dropping the cached
        field below forces on the next `field_for(index)` call. Every *other* target's cached
        field is left alone -- nothing about a different target changed.

        New assignments made after this call are unaffected by the stale-cache problem in the
        first place: `assign` reads `self.target_positions[idx]` fresh, on every call, for every
        id that isn't already in `self.assigned` -- so an id assigned for the first time after
        `move_target` naturally compares its distance against the new position, not the old one."""
        self.target_positions[index] = np.array(new_xy, float)
        self.fields.pop(index, None)
