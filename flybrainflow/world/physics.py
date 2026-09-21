"""2D circle-collision physics: how agents actually move, once something (a brain, or the
baseline model) has decided which way they want to go.

Deliberately dumb: point-ish circles, no legs, no articulated body. One shared solver for every
agent type — fly-brained or baseline — so comparing the two is a comparison of steering
decisions, not of two different movement engines (this is the "same solver for a fair
comparison" line from the plan, implemented literally: `step()` doesn't know or care what
produced `desired_velocities`).

    new_pos, actual_vel = step(positions, desired_velocities, radii, dt, walkable_map,
                                max_speed_mps=1.3)

Each step:
  1. Clamp each agent's desired velocity to its max speed, and move it there.
  2. Push apart any agents whose circles now overlap — an iterative position correction, not a
     force. Cheap, stable, and the standard technique for real-time crowd sims at this fidelity;
     it does NOT model contact forces or momentum transfer, just "don't occupy the same space".
  3. Push any agent whose circle now overlaps a wall back out, along whichever direction gets it
     clear of the wall fastest (the gradient of the map's own distance-to-wall field) — repeated
     up to `push_iterations` times, same as step 2, since one push isn't always enough to fully
     clear a large overlap (an agent driven hard into a wall for a whole timestep, say).
  4. A hard fallback: anything still not walkable after that gets snapped to the nearest walkable
     point outright, so an agent can never end a step inside a wall, whatever went wrong upstream.

The agent-agent pass uses a KD-tree (`scipy.spatial.cKDTree`) so its cost scales with how many
agents are actually near each other, not with the square of the total population. Still, at the
extreme densities this project specifically wants to study (a clogged bottleneck), a handful of
relaxation iterations per step may not fully separate every overlap in one frame — agents can
end a step slightly interpenetrating rather than being perfectly incompressible. That's a real
approximation, not swept under the rug: worth watching for at M2 when the bottleneck/Sugiyama
tests actually run at density, and easy to check for directly (see `max_overlap` below).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .map import WalkableMap


def step(
    positions,
    desired_velocities,
    radii,
    dt: float,
    walkable_map: WalkableMap,
    max_speed_mps=None,
    push_iterations: int = 3,
    agent_agent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every agent by one step of `dt` seconds. Returns (new_positions, actual_velocities).

    `actual_velocities` is the resulting displacement over `dt`, after collision correction — not
    the same as `desired_velocities` whenever a wall or another agent got in the way.
    """
    pos = np.asarray(positions, float).reshape(-1, 2).copy()
    n = len(pos)
    vel = np.asarray(desired_velocities, float).reshape(-1, 2).copy()
    r = np.broadcast_to(np.asarray(radii, float), (n,)).copy()

    if max_speed_mps is not None:
        cap = np.broadcast_to(np.asarray(max_speed_mps, float), (n,))
        speed = np.linalg.norm(vel, axis=1)
        too_fast = speed > cap
        if too_fast.any():
            scale = np.ones(n)
            scale[too_fast] = cap[too_fast] / np.maximum(speed[too_fast], 1e-12)
            vel = vel * scale[:, None]

    new_pos = pos + vel * dt

    if agent_agent and n > 1:
        new_pos = _resolve_agent_collisions(new_pos, r, iterations=push_iterations)

    new_pos = _iterate_wall_push(walkable_map, new_pos, r, push_iterations)

    # An agent whose desired move overshoots the map entirely (a big velocity for a long dt,
    # tunnelling straight through a wall) has no usable gradient out there for the push above to
    # follow -- it just gets stuck. `nearest_walkable` brings it back onto the map, but only
    # knows about the wall, not the agent's radius, so it can land right back inside the radius
    # deficit it started with. Re-running the push once it's back in valid territory, where the
    # gradient means something again, actually resolves that deficit instead of leaving it.
    bad = ~walkable_map.is_walkable(new_pos)
    if bad.any():
        new_pos[bad] = walkable_map.nearest_walkable(new_pos[bad])
        new_pos = _iterate_wall_push(walkable_map, new_pos, r, push_iterations)

    actual_vel = (new_pos - pos) / dt
    return new_pos, actual_vel


def max_overlap(positions, radii) -> float:
    """How far the worst-overlapping pair of agents currently interpenetrates (metres, 0 if none).
    A cheap way to check, after a step, whether `push_iterations` was actually enough."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)
    if n < 2:
        return 0.0
    r = np.broadcast_to(np.asarray(radii, float), (n,))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=2 * float(np.max(r)), output_type="ndarray")
    if len(pairs) == 0:
        return 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(pos[i] - pos[j], axis=1)
    overlap = (r[i] + r[j]) - dist
    return float(max(overlap.max(), 0.0))


# ---------------------------------------------------------------------------


def _iterate_wall_push(m: WalkableMap, pos: np.ndarray, r: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        moved = _resolve_wall_collisions(m, pos, r)
        if np.allclose(moved, pos):
            return moved
        pos = moved
    return pos


def _resolve_agent_collisions(pos: np.ndarray, r: np.ndarray, iterations: int) -> np.ndarray:
    n = len(pos)
    for _ in range(iterations):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=2 * float(np.max(r)), output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        delta = pos[i] - pos[j]
        dist = np.linalg.norm(delta, axis=1)
        overlap = (r[i] + r[j]) - dist
        colliding = overlap > 0
        if not colliding.any():
            break
        i_c, j_c, overlap_c = i[colliding], j[colliding], overlap[colliding]
        delta_c = delta[colliding].copy()
        dist_c = dist[colliding].copy()
        # Exactly-coincident agents have no separation direction to push along; pick a
        # deterministic (if arbitrary) one rather than dividing by zero.
        zero = dist_c < 1e-9
        if zero.any():
            delta_c[zero] = np.array([1e-6, 0.0])
            dist_c[zero] = 1e-6
        push = (delta_c / dist_c[:, None]) * (overlap_c[:, None] / 2)
        np.add.at(pos, i_c, push)
        np.add.at(pos, j_c, -push)
    return pos


def _resolve_wall_collisions(m: WalkableMap, pos: np.ndarray, r: np.ndarray) -> np.ndarray:
    eps = m.resolution
    d = m.distance_at(pos)
    deficit = r - d
    need = deficit > 0
    if not need.any():
        return pos
    p = pos[need]
    gx = (m.distance_at(p + [eps, 0.0]) - m.distance_at(p - [eps, 0.0])) / (2 * eps)
    gy = (m.distance_at(p + [0.0, eps]) - m.distance_at(p - [0.0, eps])) / (2 * eps)
    grad = np.c_[gx, gy]
    norm = np.linalg.norm(grad, axis=1)
    zero = norm < 1e-9
    dirn = np.zeros_like(grad)
    dirn[~zero] = grad[~zero] / norm[~zero, None]
    pos = pos.copy()
    pos[need] = p + dirn * deficit[need][:, None]
    return pos
