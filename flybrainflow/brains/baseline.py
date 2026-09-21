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

Each agent commits to the nearest target (straight-line, "which one looks closest from here") the
first time it's seen, and sticks with that choice -- it does not re-route away from a target that
turns out to be busy. That's deliberate: re-routing around congestion would suppress the very
queuing/choke-point behaviour this project exists to measure. An agent that commits to a full
target simply queues near it, held back by the pedestrian-repulsion term and, as a last resort,
`physics.step()`'s own overlap correction, until a slot opens.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..world.geodesic import geodesic_distance_field, sample_field
from ..world.map import WalkableMap

# Helbing-style exponential repulsion constants: A is the strength at contact, B the range over
# which it decays. Not fit to any dataset -- reasonable defaults for human-scale agents, tunable
# once real crowd footage or the M4 lane-formation test gives something to calibrate against.
AGENT_REPULSION_A = 2.0
AGENT_REPULSION_B = 0.3
AGENT_REPULSION_CUTOFF_M = 3.0
WALL_REPULSION_A = 2.0
WALL_REPULSION_B = 0.3
WALL_REPULSION_CUTOFF_M = 1.5


class Baseline:
    """A standard social-force-flavoured steering model: no smell, no brain, just geometry --
    walk toward your target along the shortest walkable path, don't walk through people or
    walls. This is what fly-brained agents are measured against."""

    def __init__(self, walkable_map: WalkableMap, target_positions):
        self.map = walkable_map
        self.target_positions = [np.array(t, float) for t in target_positions]
        if not self.target_positions:
            raise ValueError("Baseline needs at least one target position to steer toward")
        self._fields: dict[int, np.ndarray] = {}
        self._assigned: dict[int, int] = {}  # agent id -> target index; released by `forget()`

    def desired_velocities(self, ids, positions, radii, max_speed_mps) -> np.ndarray:
        """One steering vector per id in `ids`, in the same order. `positions`/`radii` line up
        with `ids` the same way `world.physics.step()` expects them."""
        positions = np.asarray(positions, float).reshape(-1, 2)
        n = len(ids)
        if n == 0:
            return np.zeros((0, 2))
        radii = np.broadcast_to(np.asarray(radii, float), (n,))
        max_speed = np.broadcast_to(np.asarray(max_speed_mps, float), (n,))

        target_idx = self._assign(ids, positions)
        goal_dir = self._goal_direction(target_idx, positions)
        social = _agent_repulsion(positions, radii)
        wall = _wall_repulsion(self.map, positions)

        raw = goal_dir * max_speed[:, None] + social + wall
        speed = np.linalg.norm(raw, axis=1)
        too_fast = speed > max_speed
        scale = np.ones(n)
        scale[too_fast] = max_speed[too_fast] / np.maximum(speed[too_fast], 1e-12)
        return raw * scale[:, None]

    def forget(self, ids) -> None:
        """Drop per-agent state (its committed target) for ids that are gone for good -- fed and
        left, in the usual case. Without this, `_assigned` grows for as long as the process runs:
        every id that was ever spawned stays in it forever, even once only a handful of agents are
        actually alive at once, which adds up over a long-running or live session. Call this with
        whatever `Population.leave()` (or any other removal) returns, each tick; forgetting an id
        that's still walking is harmless -- it's just reassigned the next time it's seen."""
        for i in ids:
            self._assigned.pop(i, None)

    # -- internals ------------------------------------------------------

    def _assign(self, ids, positions) -> np.ndarray:
        """Nearest target by straight-line distance, decided once per id and kept from then on."""
        for i, p in zip(ids, positions):
            if i not in self._assigned:
                dists = [np.linalg.norm(p - t) for t in self.target_positions]
                self._assigned[i] = int(np.argmin(dists))
        return np.array([self._assigned[i] for i in ids], dtype=int)

    def _field_for(self, target_index: int) -> np.ndarray:
        if target_index not in self._fields:
            self._fields[target_index] = geodesic_distance_field(
                self.map, self.target_positions[target_index]
            )
        return self._fields[target_index]

    def _goal_direction(self, target_idx: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Unit vector at each position pointing toward decreasing geodesic distance to its
        assigned target -- "downhill" on the shortest-walking-distance field, so it bends around
        whatever's in the way instead of aiming through it."""
        eps = self.map.resolution
        out = np.zeros_like(positions)
        for ti in np.unique(target_idx):
            rows = target_idx == ti
            field = self._field_for(int(ti))
            p = positions[rows]
            here = sample_field(self.map, field, p)
            gx = sample_field(self.map, field, p + [eps, 0.0]) - sample_field(self.map, field, p - [eps, 0.0])
            gy = sample_field(self.map, field, p + [0.0, eps]) - sample_field(self.map, field, p - [0.0, eps])
            grad = np.c_[gx, gy] / (2 * eps)
            # Distance decreases toward the target, so head the opposite way from its gradient.
            direction = -grad
            norm = np.linalg.norm(direction, axis=1)
            unreachable = ~np.isfinite(here) | (norm < 1e-9)
            safe_norm = np.where(unreachable, 1.0, norm)
            unit = direction / safe_norm[:, None]
            unit[unreachable] = 0.0  # already at the target, or no walkable path to it at all
            out[rows] = unit
        return out


def _agent_repulsion(positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Helbing-style exponential push away from nearby agents: `A * exp((r_ij - d_ij) / B)`,
    strongest at contact and fading smoothly with distance rather than switching on at a hard
    boundary -- the softer, earlier nudge that produces lane-splitting behaviour, with
    `physics.step()`'s own overlap correction as the hard backstop if this isn't enough."""
    n = len(positions)
    force = np.zeros((n, 2))
    if n < 2:
        return force
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=AGENT_REPULSION_CUTOFF_M, output_type="ndarray")
    if len(pairs) == 0:
        return force
    i, j = pairs[:, 0], pairs[:, 1]
    delta = positions[i] - positions[j]
    dist = np.linalg.norm(delta, axis=1)
    zero = dist < 1e-9
    dist_safe = np.where(zero, 1e-6, dist)
    n_ij = delta / dist_safe[:, None]
    n_ij[zero] = np.array([1.0, 0.0])
    r_ij = radii[i] + radii[j]
    magnitude = AGENT_REPULSION_A * np.exp((r_ij - dist) / AGENT_REPULSION_B)
    f = n_ij * magnitude[:, None]
    np.add.at(force, i, f)
    np.add.at(force, j, -f)
    return force


def _wall_repulsion(m: WalkableMap, positions: np.ndarray) -> np.ndarray:
    """Same shape of repulsion as `_agent_repulsion`, but away from the nearest wall, using the
    map's own distance-to-wall field (the same one `world.physics` uses for its hard wall push)."""
    n = len(positions)
    force = np.zeros((n, 2))
    d = m.distance_at(positions)
    near = d < WALL_REPULSION_CUTOFF_M
    if not near.any():
        return force
    eps = m.resolution
    p = positions[near]
    gx = (m.distance_at(p + [eps, 0.0]) - m.distance_at(p - [eps, 0.0])) / (2 * eps)
    gy = (m.distance_at(p + [0.0, eps]) - m.distance_at(p - [0.0, eps])) / (2 * eps)
    grad = np.c_[gx, gy]  # points away from the wall, toward increasing clearance
    norm = np.linalg.norm(grad, axis=1)
    zero = norm < 1e-9
    direction = np.zeros_like(grad)
    direction[~zero] = grad[~zero] / norm[~zero, None]
    magnitude = WALL_REPULSION_A * np.exp(-d[near] / WALL_REPULSION_B)
    force[near] = direction * magnitude[:, None]
    return force
