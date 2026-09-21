"""Shared Helbing-style exponential repulsion, away from other agents and away from walls.

Pulled out of `baseline.py` when `toy.py` turned out to need the exact same "how urgent, and
which way" signal -- for `Baseline` it becomes part of the steering velocity directly; for
`ToyBrain` it becomes a sensory input the brain reacts to, same shape as the plan's own LC16
("something's coming at me, left or right") entry. One implementation, two uses.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..world.map import WalkableMap

# A is the strength at contact, B the range over which it decays. Not fit to any dataset --
# reasonable defaults for human-scale agents, tunable once real crowd footage or the M4
# lane-formation test gives something to calibrate against.
AGENT_REPULSION_A = 2.0
AGENT_REPULSION_B = 0.3
AGENT_REPULSION_CUTOFF_M = 3.0
WALL_REPULSION_A = 2.0
WALL_REPULSION_B = 0.3
WALL_REPULSION_CUTOFF_M = 1.5


def agent_repulsion(
    positions: np.ndarray,
    radii: np.ndarray,
    A: float = AGENT_REPULSION_A,
    B: float = AGENT_REPULSION_B,
    cutoff: float = AGENT_REPULSION_CUTOFF_M,
) -> np.ndarray:
    """Push away from nearby agents: `A * exp((r_ij - d_ij) / B)`, strongest at contact and
    fading smoothly with distance rather than switching on at a hard boundary."""
    n = len(positions)
    force = np.zeros((n, 2))
    if n < 2:
        return force
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
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
    magnitude = A * np.exp((r_ij - dist) / B)
    f = n_ij * magnitude[:, None]
    np.add.at(force, i, f)
    np.add.at(force, j, -f)
    return force


def wall_repulsion(
    m: WalkableMap,
    positions: np.ndarray,
    A: float = WALL_REPULSION_A,
    B: float = WALL_REPULSION_B,
    cutoff: float = WALL_REPULSION_CUTOFF_M,
) -> np.ndarray:
    """Same shape of repulsion as `agent_repulsion`, but away from the nearest wall, using the
    map's own distance-to-wall field (the same one `world.physics` uses for its hard wall push)."""
    n = len(positions)
    force = np.zeros((n, 2))
    d = m.distance_at(positions)
    near = d < cutoff
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
    magnitude = A * np.exp(-d[near] / B)
    force[near] = direction * magnitude[:, None]
    return force
