"""Geodesic (wall-respecting) distance on the walkable grid.

Straight-line distance is fine in an open room; it's wrong the moment
there's a wall, a barrier, or a turnstile between two points — which is
exactly what a stadium or train-station venue is full of. This gives the
actual shortest walking distance around obstacles, from one source cell to
every walkable cell, via Dijkstra over the grid graph (8-connected;
corner-cutting diagonally between two walls is disallowed, same rule most
grid pathfinders use).

    from flybrainflow.world.geodesic import geodesic_distance_field, sample_field
    field = geodesic_distance_field(walkable_map, source_xy)   # metres, np.inf where unreachable
    sample_field(walkable_map, field, agent_positions)          # look the field up at arbitrary points

This is plain Dijkstra: O(cells log cells), correct, not fast. Fine for M0's
test venues and a prototype stadium section; if a full stadium map with
targets that get dragged around live turns out to need this recomputed every
frame, this is the file to replace with a vectorized wavefront/fast-marching
approximation. Flagged as a risk in docs/PLAN.md, not solved here.
"""

from __future__ import annotations

import heapq

import numpy as np

from .map import WalkableMap

_STEPS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_STEPS_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SQRT2 = 1.4142135623730951


def geodesic_distance_field(m: WalkableMap, source_xy, connectivity: int = 8) -> np.ndarray:
    """Shortest walking distance (metres) from `source_xy` to every cell of `m`, respecting walls.

    Cells with no walkable path to the source (sealed behind walls) are `np.inf`.
    `source_xy` is snapped to the nearest walkable cell first, same as the rest of the map API.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    snapped = m.nearest_walkable([source_xy])[0]
    six_arr, siy_arr = m.world_to_cell([snapped])
    six, siy = int(six_arr[0]), int(siy_arr[0])

    ny, nx = m.ny, m.nx
    mask = m.mask
    res = m.resolution
    dist = np.full((ny, nx), np.inf)
    if not (0 <= six < nx and 0 <= siy < ny) or not mask[siy, six]:
        return dist  # defensive; nearest_walkable should always land on a walkable cell

    dist[siy, six] = 0.0
    steps = _STEPS_4 + _STEPS_DIAG if connectivity == 8 else _STEPS_4
    heap: list[tuple[float, int, int]] = [(0.0, six, siy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy in steps:
            nx_, ny_ = x + dx, y + dy
            if not (0 <= nx_ < nx and 0 <= ny_ < ny) or not mask[ny_, nx_]:
                continue
            if dx != 0 and dy != 0:
                if not mask[y, nx_] or not mask[ny_, x]:
                    continue  # don't cut the corner between two walls
                step_cost = res * _SQRT2
            else:
                step_cost = res
            nd = d + step_cost
            if nd < dist[ny_, nx_]:
                dist[ny_, nx_] = nd
                heapq.heappush(heap, (nd, nx_, ny_))
    return dist


def sample_field(m: WalkableMap, field: np.ndarray, xy) -> np.ndarray:
    """Look up a per-cell field (such as the one above) at arbitrary world points, nearest-cell."""
    p = m.nearest_walkable(xy)
    ix, iy = m.world_to_cell(p)
    ix = np.clip(ix, 0, m.nx - 1)
    iy = np.clip(iy, 0, m.ny - 1)
    return field[iy, ix]
