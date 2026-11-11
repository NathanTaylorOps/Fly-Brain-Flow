"""Three-gate maze and an evolving, wall-respecting odor transport model.

This is a 2D advection-diffusion approximation, not turbulent CFD. Odor travels
on open grid faces only. The displayed wind vectors and behavioral wind sensors
come from the same potential-flow field used by the transport solver.
"""
from __future__ import annotations

import math
import numpy as np

from .map import WalkableMap
from .airflow import AirflowField
from .fields import OdorSource, Wind

MAZE_SPAWN = (1.5, 6.0)
MAZE_FOOD = (14.5, 7.0)


def _rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def maze_map(resolution: float = 0.25) -> WalkableMap:
    """16 × 12 m outdoor maze with three alternating gates and side branches."""
    if not math.isfinite(resolution) or not 0.1 <= resolution <= 0.5:
        raise ValueError('Maze resolution must be between 0.1 and 0.5 m')
    obstacles = []
    for x, low, high in ((4.0, 2.0, 4.0), (8.0, 8.0, 10.0), (12.0, 3.0, 5.0)):
        obstacles.extend([_rectangle(x - .25, 0, x + .25, low),
                          _rectangle(x - .25, high, x + .25, 12)])
    # Short walls create dead-end branches but preserve the main winding route.
    obstacles += [_rectangle(4.25, 5.75, 6.0, 6.25),
                  _rectangle(10.0, 6.75, 11.75, 7.25),
                  _rectangle(12.25, 9.0, 14.0, 9.5)]
    m = WalkableMap.from_rings([_rectangle(0, 0, 16, 12)], obstacles,
                              resolution=resolution, padding=resolution,
                              name='Three-gate maze',
                              meta={'spawn': MAZE_SPAWN, 'food': MAZE_FOOD,
                                    'gates': [[4, 3], [8, 9], [12, 4]],
                                    'air_boundary': 'Outdoor openings on all four sides'})
    # Explicit ambient air outside the walking boundary. Internal barriers remain
    # solid. This permits arbitrary compass directions without hidden wall leaks.
    m.air_open = np.zeros_like(m.mask)
    m.air_open[0, :] = m.air_open[-1, :] = True
    m.air_open[:, 0] = m.air_open[:, -1] = True
    return m


class DynamicOdorField:
    """Finite-volume upwind transport with reflecting solid walls.

    Each source has an independent concentration grid. Moving a source changes
    future emission while old odor dissipates naturally. ``reset`` clears it.
    Concentration units, diffusion and emission are illustrative model settings.
    A world loop must call ``advance(dt)`` exactly once per simulated time step.
    """
    def __init__(self, wind: Wind, sources: list[OdorSource],
                 walkable_map: WalkableMap, airflow: AirflowField | None = None,
                 *, diffusion: float = 0.35, decay: float = 0.025,
                 emission: float = 3.0):
        if any(not math.isfinite(v) or v < 0 for v in (diffusion, decay, emission)):
            raise ValueError('Transport parameters must be finite and nonnegative')
        self.wind = wind
        self.sources = sources
        self.walkable_map = walkable_map
        self.airflow = airflow or AirflowField.build(walkable_map)
        self.diffusion, self.decay, self.emission = diffusion, decay, emission
        self._fields = [np.zeros_like(walkable_map.mask, dtype=float) for _ in sources]
        self.time_s = 0.0
        self._wind_key = None
        self._sync_wind()

    def reset(self):
        for c in self._fields:
            c.fill(0)
        self.time_s = 0.0

    def _sync_wind(self):
        key = (float(self.wind.direction_deg), float(self.wind.speed_mps))
        if self._wind_key == key:
            return
        if not all(math.isfinite(x) for x in key) or key[1] < 0:
            raise ValueError('Wind direction/speed must be finite; speed nonnegative')
        m, a = self.walkable_map, self.airflow
        theta = np.deg2rad(key[0])
        velocity = key[1] * (np.cos(theta) * a.grad_x + np.sin(theta) * a.grad_y)
        # Faces connect adjacent walkable cells or the declared ambient border.
        open_cells = m.mask | (m.air_open if m.air_open is not None else ~a.solid)
        self._open_x = open_cells[:, :-1] & open_cells[:, 1:]
        self._open_y = open_cells[:-1, :] & open_cells[1:, :]
        self._ux = .5 * (velocity[:, :-1, 0] + velocity[:, 1:, 0]) * self._open_x
        self._vy = .5 * (velocity[:-1, :, 1] + velocity[1:, :, 1]) * self._open_y
        # Same face velocities averaged back to centers for particles and sensors.
        self._velocity = np.zeros((*m.mask.shape, 2), dtype=float)
        self._velocity[:, :-1, 0] += self._ux * .5
        self._velocity[:, 1:, 0] += self._ux * .5
        self._velocity[:-1, :, 1] += self._vy * .5
        self._velocity[1:, :, 1] += self._vy * .5
        self._velocity[~m.mask] = 0
        dx = m.resolution
        out = np.zeros_like(m.mask, dtype=float)
        out[:, :-1] += np.maximum(self._ux, 0) / dx + self.diffusion * self._open_x / dx**2
        out[:, 1:] += np.maximum(-self._ux, 0) / dx + self.diffusion * self._open_x / dx**2
        out[:-1, :] += np.maximum(self._vy, 0) / dx + self.diffusion * self._open_y / dx**2
        out[1:, :] += np.maximum(-self._vy, 0) / dx + self.diffusion * self._open_y / dx**2
        max_out = float(out[m.mask].max(initial=0)) + self.decay
        self._stable_dt = .85 / max_out if max_out > 0 else 1.0
        self._wind_key = key

    def advance(self, dt: float):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        if dt == 0:
            return
        self._sync_wind()
        if len(self._fields) != len(self.sources):
            raise ValueError('Changing source count requires constructing a new field')
        m, dx = self.walkable_map, self.walkable_map.resolution
        count = max(1, math.ceil(dt / self._stable_dt))
        h = dt / count
        source_cells = []
        for source in self.sources:
            if not math.isfinite(source.strength) or source.strength < 0:
                raise ValueError('Source strength must be finite and nonnegative')
            if not m.is_walkable([source.position])[0]:
                raise ValueError('Odor source must be inside the walkable maze')
            ix, iy = m.world_to_cell([source.position])
            source_cells.append((int(iy[0]), int(ix[0])))
        for _ in range(count):
            for c, source, (row, col) in zip(self._fields, self.sources, source_cells):
                fx = (self._ux * np.where(self._ux >= 0, c[:, :-1], c[:, 1:])
                      - self.diffusion / dx * (c[:, 1:] - c[:, :-1]) * self._open_x)
                fy = (self._vy * np.where(self._vy >= 0, c[:-1, :], c[1:, :])
                      - self.diffusion / dx * (c[1:, :] - c[:-1, :]) * self._open_y)
                change = -self.decay * c
                change[:, :-1] -= fx / dx
                change[:, 1:] += fx / dx
                change[:-1, :] -= fy / dx
                change[1:, :] += fy / dx
                c += h * change
                c[row, col] += h * source.strength * self.emission / dx**2
                c[~m.mask] = 0  # ambient is odor-free; outgoing odor leaves domain
                np.maximum(c, 0, out=c)  # only roundoff under the positivity CFL
        self.time_s += dt

    def _selected_grid(self, kind=None, source=None):
        total = np.zeros_like(self.walkable_map.mask, dtype=float)
        for s, c in zip(self.sources, self._fields):
            if (source is not None and s is source) or (source is None and (kind is None or s.kind == kind)):
                total += c
        return total

    def _sample_grid(self, grid, xy):
        p = np.asarray(xy, dtype=float).reshape(-1, 2)
        m = self.walkable_map
        ix, iy = m.world_to_cell(p)
        valid = m.is_walkable(p)
        result = np.zeros((len(p), *grid.shape[2:]), dtype=float)
        result[valid] = grid[iy[valid], ix[valid]]
        return result

    def sample(self, xy, kind=None, source=None):
        return self._sample_grid(self._selected_grid(kind, source), xy)

    def sample_by_kind(self, xy):
        return {kind: self.sample(xy, kind=kind) for kind in self.kinds()}

    def kinds(self):
        return {s.kind for s in self.sources}

    def gradient(self, xy, kind=None, eps=None, source=None):
        # Mirror concentration at solid faces, so a wall itself is never a
        # spurious downhill signal. Compute local gradient, not a target vector.
        m = self.walkable_map
        c = self._selected_grid(kind, source)
        gx, gy = np.zeros_like(c), np.zeros_like(c)
        valid_x = m.mask[:, :-1] & m.mask[:, 1:]
        valid_y = m.mask[:-1, :] & m.mask[1:, :]
        dx = (c[:, 1:] - c[:, :-1]) * valid_x / (2 * m.resolution)
        dy = (c[1:, :] - c[:-1, :]) * valid_y / (2 * m.resolution)
        gx[:, :-1] += dx
        gx[:, 1:] += dx
        gy[:-1, :] += dy
        gy[1:, :] += dy
        return self._sample_grid(np.stack([gx, gy], axis=-1), xy)

    def local_wind(self, xy):
        self._sync_wind()
        return self._sample_grid(self._velocity, xy)

    def environment_snapshot(self):
        """JSON-compatible exact grid; x is fastest, y increases upward."""
        self._sync_wind()
        m = self.walkable_map
        return {'origin': list(m.origin), 'resolution': m.resolution,
                'width': m.nx, 'height': m.ny,
                'mask': m.mask.astype(np.uint8).ravel().tolist(),
                'concentration': self._selected_grid().ravel().tolist(),
                'velocity_x': self._velocity[..., 0].ravel().tolist(),
                'velocity_y': self._velocity[..., 1].ravel().tolist(),
                'direction_deg': self.wind.direction_deg,
                'speed_mps': self.wind.speed_mps, 'time_s': self.time_s,
                'model': '2D potential wind + upwind advection/diffusion; illustrative, no turbulence',
                'source_positions': [list(s.position) for s in self.sources]}
