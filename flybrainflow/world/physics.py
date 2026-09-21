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
  2. Relax agent-agent and agent-wall overlap together, one round at a time, up to
     `push_iterations` rounds: push apart any agents whose circles now overlap (an iterative
     position correction, not a force — cheap, stable, the standard technique for real-time crowd
     sims at this fidelity; it does NOT model contact forces or momentum transfer, just "don't
     occupy the same space"), *then*, in the same round, push any agent whose circle now overlaps
     a wall back out along the map's own distance-to-wall gradient.
  3. A hard fallback: anything still not walkable after that gets snapped to the nearest walkable
     point outright, so an agent can never end a step inside a wall, whatever went wrong upstream.

Real bug, found and fixed: steps 2 and 3 above used to be two separate passes, each run to its
*own* full convergence before the other started (agent-agent fully resolved, then wall overlap
fully resolved). That's wrong whenever the two constraints fight each other: pushing two
overlapping agents apart can push one of them into a wall, and the wall-push pass that runs after
has no way to signal "now go re-check agent-agent again" -- so a wall-adjacent bottleneck could
converge to a *permanent* agent-agent overlap that no number of `push_iterations` ever fixed
(confirmed directly: two 0.3 m-radius agents stacked 5 cm apart, 15 cm from a wall, held a stable
0.45 m overlap at push_iterations = 3, 10, 20 and 50 alike). Exactly the "M2 bottleneck/Sugiyama"
density this project is built to study is where two agents squeezed against a wall is the *normal*
case, not an edge case -- so this wasn't a rare corner, it was going to be the common case at the
scenarios that matter most. Fixed by interleaving: each round now does one agent-push pass
followed by one wall-push pass, so a wall-push that reintroduces agent overlap gets a chance to be
corrected by the very next round's agent-push, and vice versa, instead of each constraint getting
exactly one uncontested say.

The agent-agent pass uses a KD-tree (`scipy.spatial.cKDTree`) so its cost scales with how many
agents are actually near each other, not with the square of the total population. Still, at the
extreme densities this project specifically wants to study (a clogged bottleneck), a handful of
relaxation rounds per step may not fully separate every overlap in one frame — agents can end a
step slightly interpenetrating rather than being perfectly incompressible. That's a real
approximation, not swept under the rug: worth watching for at M2 when the bottleneck/Sugiyama
tests actually run at density, and easy to check for directly (see `max_overlap` below) -- now a
meaningfully more accurate check than before this fix, since it's no longer measuring an overlap
that was structurally guaranteed to never improve.
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

    new_pos = _relax(walkable_map, new_pos, r, push_iterations, agent_agent=agent_agent and n > 1)

    # An agent whose desired move overshoots the map entirely (a big velocity for a long dt,
    # tunnelling straight through a wall) has no usable gradient out there for the push above to
    # follow -- it just gets stuck. `nearest_walkable` brings it back onto the map, but only
    # knows about the wall, not the agent's radius, so it can land right back inside the radius
    # deficit it started with. Re-running the relax once it's back in valid territory, where the
    # gradient means something again, actually resolves that deficit instead of leaving it.
    bad = ~walkable_map.is_walkable(new_pos)
    if bad.any():
        new_pos[bad] = walkable_map.nearest_walkable(new_pos[bad])
        new_pos = _relax(walkable_map, new_pos, r, push_iterations, agent_agent=agent_agent and n > 1)

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


def _relax(m: WalkableMap, pos: np.ndarray, r: np.ndarray, iterations: int, agent_agent: bool) -> np.ndarray:
    """One round = one agent-push pass (if any agents to push apart) followed by one wall-push
    pass, repeated up to `iterations` times, stopping early once a round changes nothing. See the
    module docstring's "real bug, found and fixed" note for why this has to interleave the two
    passes round-by-round rather than running each to its own separate convergence -- a wall-push
    that reintroduces agent overlap needs the *next* round's agent-push to see and fix it, not a
    pass that already finished and never runs again."""
    for _ in range(iterations):
        before = pos
        if agent_agent:
            pos = _push_apart_agents_once(pos, r)
        pos = _resolve_wall_collisions(m, pos, r)
        if np.allclose(pos, before):
            break
    return pos


def _push_apart_agents_once(pos: np.ndarray, r: np.ndarray) -> np.ndarray:
    """One position-correction pass: every currently-overlapping pair gets pushed apart by half
    its overlap each, along the line between their centres. Not run to its own convergence here --
    see `_relax`, which interleaves this with the wall push so the two constraints can't
    permanently fight each other."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=2 * float(np.max(r)), output_type="ndarray")
    if len(pairs) == 0:
        return pos
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    dist = np.linalg.norm(delta, axis=1)
    overlap = (r[i] + r[j]) - dist
    colliding = overlap > 0
    if not colliding.any():
        return pos
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
    pos = pos.copy()
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
