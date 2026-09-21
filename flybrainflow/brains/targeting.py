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

    def assign(self, ids, positions: np.ndarray) -> np.ndarray:
        """Target index per id, in the same order as `ids`. Deciding is a one-time event per id;
        already-assigned ids just get their existing choice looked back up."""
        for i, p in zip(ids, positions):
            if i not in self.assigned:
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
        agents are actually alive at once (see docs/JOURNAL.md for the run that caught this the
        first time, in `Baseline` before this was pulled out into shared code)."""
        for i in ids:
            self.assigned.pop(i, None)
