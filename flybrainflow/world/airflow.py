"""Wind that bends around obstacles: 2D potential (irrotational) flow on the walkable grid.

Not full CFD. No turbulence, no eddies, no wake recirculation behind a bluff
object — a real pillar sheds a churning vortex behind it; this model can't
represent that at all. What it does give you is real, standard potential-flow
behaviour: streamlines deflect around solid obstacles and speed up through
narrow gaps (a genuine Venturi effect), computed by solving Laplace's
equation for the velocity potential on the same grid the map already uses.
Obstacles (anything non-walkable) are no-flow (Neumann) boundaries; the
literal outer edge of the map's raster is treated as still connected to the
open sky (Dirichlet, fixed at the free-stream value).

    field = AirflowField.build(walkable_map)
    field.velocity_at(agent_positions, direction_deg=45, speed_mps=2.0)   -> (n, 2) local wind vectors

Solved once per map, for two reference directions (unit wind in +x, unit wind
in +y). Laplace's equation is linear, so any actual wind direction/speed
afterwards is just a weighted sum of those two solutions — changing the wind
dial at runtime costs a dot product, not a resolve.

**This only produces real airflow where the venue has an actual opening
reaching the map's edge.** A fully enclosed box — walls on every side, which
is what every map generator's padding margin produces by default — is
genuinely, physically sealed: no amount of clever padding classification
changes that, and an earlier version of this file tried exactly that (treat
some depth of padding as "secretly open") and got it wrong twice, once
sealing every venue solid and once making the walls fully porous, both
resolution-dependent accidents rather than real physics. There's no way to
tell "a real wall" from "harmless padding" from a zero-thickness polygon
boundary alone, so this file doesn't try: everything non-walkable blocks air
except the literal last row/column of the raster. A venue like `corridor` or
`bottleneck` needs `open_ends=True` (see `map.py`) — which removes the
padding at the two ends specifically, so the walkable area actually reaches
the map edge there — before wind has anywhere to enter or leave. Without
that, correctly computed airflow is legitimately all zero, which is the
right physical answer for a sealed room, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from .map import WalkableMap

# Tiny diagonal regularisation so a walkable pocket fully sealed off from any
# open (Dirichlet) cell -- a room with literally no way out -- still solves
# instead of hitting a singular matrix. Physically that pocket should have
# zero velocity throughout; this nudges it there instead of crashing.
_REGULARISATION = 1e-6


@dataclass
class AirflowField:
    map: WalkableMap
    solid: np.ndarray  # (ny, nx) bool -- real, air-blocking obstacle cells
    grad_x: np.ndarray  # (ny, nx, 2) -- velocity per cell for unit wind toward +x
    grad_y: np.ndarray  # (ny, nx, 2) -- velocity per cell for unit wind toward +y

    @classmethod
    def build(cls, m: WalkableMap) -> "AirflowField":
        solid, open_ = _classify(m)
        phi_x = _solve_potential(m, solid, open_, unit=(1.0, 0.0))
        phi_y = _solve_potential(m, solid, open_, unit=(0.0, 1.0))
        grad_x = _velocity_grid(phi_x, m.resolution, solid)
        grad_y = _velocity_grid(phi_y, m.resolution, solid)
        return cls(m, solid, grad_x, grad_y)

    def velocity_at(self, xy, direction_deg: float, speed_mps: float) -> np.ndarray:
        """(n, 2) local wind vector at each point, for this wind-dial setting."""
        p = np.asarray(xy, float).reshape(-1, 2)
        if speed_mps <= 0:
            return np.zeros_like(p)
        snapped = self.map.nearest_walkable(p)
        ix, iy = self.map.world_to_cell(snapped)
        ix = np.clip(ix, 0, self.map.nx - 1)
        iy = np.clip(iy, 0, self.map.ny - 1)
        r = np.deg2rad(direction_deg)
        cx, cy = np.cos(r), np.sin(r)
        return speed_mps * (cx * self.grad_x[iy, ix] + cy * self.grad_y[iy, ix])


# ---------------------------------------------------------------------------


def _classify(m: WalkableMap) -> tuple[np.ndarray, np.ndarray]:
    """Split non-walkable cells into `solid` (blocks air) and `open` (connected to the sky).

    Uses `m.air_open` when the map declares it (an explicit doorway, from `open_ends=True` on a
    generator, say) — that's the reliable source of truth. Without it, falls back to treating the
    raster's literal outer edge as open, which is coarser and can be wrong when a map's padding
    happens to be only a cell or two deep at its resolution (every side would then read as "the
    edge", not just the intended opening).
    """
    not_walkable = ~m.mask
    if m.air_open is not None:
        open_ = not_walkable & m.air_open
    else:
        border = np.zeros_like(not_walkable)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        open_ = not_walkable & border
    solid = not_walkable & ~open_
    return solid, open_


def _solve_potential(m: WalkableMap, solid: np.ndarray, open_: np.ndarray, unit: tuple[float, float]) -> np.ndarray:
    """Solve Laplace's equation for the velocity potential, for a unit free-stream in `unit`."""
    ny, nx = m.ny, m.nx
    walkable = m.mask
    idx = -np.ones((ny, nx), dtype=np.int64)
    coords = np.argwhere(walkable)
    idx[walkable] = np.arange(len(coords))
    n = len(coords)
    phi = np.zeros((ny, nx))
    if n == 0:
        return phi

    ux, uy = unit
    centres = m.cell_centres().reshape(ny, nx, 2)
    phi_open = ux * centres[..., 0] + uy * centres[..., 1]  # the imposed free-stream potential

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n)
    for k, (iy, ix) in enumerate(coords):
        diag = -4.0 - _REGULARISATION
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxp, nyp = ix + dx, iy + dy
            if 0 <= nxp < nx and 0 <= nyp < ny and walkable[nyp, nxp]:
                rows.append(k)
                cols.append(idx[nyp, nxp])
                vals.append(1.0)
            elif 0 <= nxp < nx and 0 <= nyp < ny and open_[nyp, nxp]:
                b[k] -= phi_open[nyp, nxp]
            else:
                diag += 1.0  # mirror: a solid wall or the map edge reflects, contributing nothing new
        rows.append(k)
        cols.append(k)
        vals.append(diag)

    a = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    phi_vals = spsolve(a, b)
    phi[walkable] = phi_vals
    phi[open_] = phi_open[open_]
    return phi


def _velocity_grid(phi: np.ndarray, resolution: float, solid: np.ndarray) -> np.ndarray:
    """velocity = grad(phi), with solid cells filled from their nearest open/walkable neighbour
    first so the finite difference at a wall doesn't see an artificial jump to an unset value."""
    filled = phi
    if solid.any():
        filled = phi.copy()
        _, (ry, rx) = ndimage.distance_transform_edt(solid, return_indices=True)
        filled[solid] = phi[ry[solid], rx[solid]]
    dphi_dy, dphi_dx = np.gradient(filled, resolution)
    u = np.where(solid, 0.0, dphi_dx)
    v = np.where(solid, 0.0, dphi_dy)
    return np.stack([u, v], axis=-1)
