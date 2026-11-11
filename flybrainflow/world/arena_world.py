"""Deterministic venue geometry, exit routing and an illustrative odor hazard.

Routing is explicitly an engineered shortest-cost field, not a connectome
result. Exits are crossing lines with outside corridors, not capture targets.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .map import WalkableMap
from .fields import OdorSource, Wind
from .live_maze import DynamicOdorField


def _rect(x, y, w, h):
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=float)


class ArenaWorld:
    width = 64.0
    height = 48.0
    venue_bounds = (6.0, 6.0, 58.0, 42.0)
    routing_clearance = .65

    def __init__(self, resolution: float = .5):
        if not math.isfinite(resolution) or not .25 <= resolution <= .5:
            raise ValueError('Resolution must be between .25 and .5 m')
        self.resolution = resolution
        self.obstacles = []
        specifications = [
            ('stage', 'Event stage', 26, 33, 12, 5),
            ('seats_sw', 'South-west seating', 12, 12, 10, 6),
            ('seats_nw', 'North-west seating', 12, 30, 10, 6),
            ('seats_se', 'South-east seating', 42, 12, 10, 6),
            ('seats_ne', 'North-east seating', 42, 30, 10, 6),
            ('kiosk_w', 'West service island', 9, 22, 5, 4),
            ('kiosk_e', 'East service island', 50, 22, 5, 4),
        ]
        for ident, label, x, y, w, h in specifications:
            self.obstacles.append({'id': ident, 'label': label, 'x': x, 'y': y,
                                   'width': w, 'height': h,
                                   'polygon': _rect(x, y, w, h).tolist()})
        outer = [_rect(6, 6, 52, 36), _rect(0, 21, 8, 6),
                 _rect(56, 21, 8, 6), _rect(29, 0, 6, 8), _rect(29, 40, 6, 8)]
        self.map = WalkableMap.from_rings(outer,
            [np.array(o['polygon']) for o in self.obstacles],
            resolution=resolution, padding=resolution, name='Four-exit event arena')
        self.walkable_map = self.map
        self.map.air_open = np.zeros_like(self.map.mask)
        self.map.air_open[0, :] = self.map.air_open[-1, :] = True
        self.map.air_open[:, 0] = self.map.air_open[:, -1] = True
        self.exits = [
            self._exit('north', 'North gate', 32, 44, 1, 1, 44, [29, 35]),
            self._exit('east', 'East gate', 60, 24, 0, 1, 60, [21, 27]),
            self._exit('south', 'South gate', 32, 4, 1, -1, 4, [29, 35]),
            self._exit('west', 'West gate', 4, 24, 0, -1, 4, [21, 27]),
        ]
        self._by_id = {e['id']: e for e in self.exits}
        self._safe = self.map.mask & (self.map.distance_to_wall >= self.routing_clearance)
        self._node = np.full(self._safe.shape, -1, dtype=np.int32)
        self._node[self._safe] = np.arange(self._safe.sum(), dtype=np.int32)
        self._cells = self.map.cell_centres().reshape(*self._safe.shape, 2)[self._safe]
        self._prepare_edges()
        self._routes = {}
        self._route_key = None
        self.time_s = 0.0
        self.hazard = DynamicOdorField(Wind(direction_deg=0, speed_mps=1),
            [OdorSource('repellent', (32, 24), strength=0)], self.map,
            diffusion=.5, decay=.045, emission=5)
        self.hazard_enabled = False

    @staticmethod
    def _exit(ident, label, x, y, axis, sign, crossing, span):
        return {'id': ident, 'label': label, 'x': x, 'y': y, 'width': 6.0,
                'orientation': ident, 'axis': axis, 'sign': sign,
                'crossing': crossing, 'span': span, 'open': True}

    def _prepare_edges(self):
        nrows, ncols = self._safe.shape
        rows, cols, lengths = [], [], []
        yy, xx = np.nonzero(self._safe)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            tx, ty = xx + dx, yy + dy
            valid = (tx >= 0) & (tx < ncols) & (ty >= 0) & (ty < nrows)
            k = np.flatnonzero(valid)
            valid[k] &= self._safe[ty[k], tx[k]]
            if dx and dy:
                k = np.flatnonzero(valid)
                valid[k] &= self._safe[yy[k], tx[k]] & self._safe[ty[k], xx[k]]
            k = np.flatnonzero(valid)
            rows.append(self._node[yy[k], xx[k]])
            cols.append(self._node[ty[k], tx[k]])
            lengths.append(np.full(len(k), self.resolution * math.hypot(dx,dy)))
        self._row, self._col, self._length = map(np.concatenate, (rows, cols, lengths))
        self._clearance_cost = 1 + .15 / np.maximum(self.map.distance_to_wall[self._safe], .5)

    def spawn_points(self, n: int, seed: int = 42, radius: float = .24) -> np.ndarray:
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError('n must be a nonnegative integer')
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be positive')
        rng = np.random.default_rng(seed)
        spacing = max(.8, 2 * radius + .15)
        x = np.arange(7, 57, spacing)
        y = np.arange(7, 41, spacing)
        xx, yy = np.meshgrid(x, y)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        candidates += rng.uniform(-.025, .025, candidates.shape)
        safe = self.map.is_walkable(candidates) & (self.map.distance_at(candidates) >= max(radius+.35, self.routing_clearance))
        candidates = candidates[safe]
        if n > len(candidates):
            raise ValueError(f'Only {len(candidates)} separated spawn locations fit at radius {radius}')
        return candidates[rng.permutation(len(candidates))[:n]].copy()

    def set_exit(self, exit_id: str, opened: bool):
        if exit_id not in self._by_id:
            raise ValueError(f'Unknown exit {exit_id!r}')
        self._by_id[exit_id]['open'] = bool(opened)
        self._route_key = None

    def set_hazard(self, position=None, *, enabled=None, strength=None,
                   direction_deg=None, speed_mps=None):
        if position is not None:
            p = np.asarray(position, float)
            if p.shape != (2,) or not np.isfinite(p).all() or not self.map.is_walkable([p])[0]:
                raise ValueError('Hazard position must be inside walkable space')
            self.hazard.sources[0].move_to(tuple(p))
        if enabled is not None:
            self.hazard_enabled = bool(enabled)
        if strength is not None and (not math.isfinite(strength) or strength < 0 or strength > 20):
            raise ValueError('Hazard strength must be in [0,20]')
        if strength is not None:
            self._hazard_strength = float(strength)
            if enabled is None:
                self.hazard_enabled = strength > 0
        self.hazard.sources[0].strength = getattr(self, '_hazard_strength', 1.0) if self.hazard_enabled else 0.0
        self.hazard.wind.set(direction_deg, speed_mps)
        self._route_key = None

    def set_wind(self, direction_deg=None, speed_mps=None):
        if direction_deg is not None and not math.isfinite(direction_deg):
            raise ValueError('Wind direction must be finite')
        if speed_mps is not None and (not math.isfinite(speed_mps) or not 0 <= speed_mps <= 10):
            raise ValueError('Wind speed must be in [0,10]')
        self.hazard.wind.set(direction_deg, speed_mps)
        self._route_key = None

    def advance(self, dt: float):
        self.hazard.advance(dt)
        self.time_s += dt

    def _open_ids(self, open_exit_ids):
        ids = [e['id'] for e in self.exits if e['open']] if open_exit_ids is None else list(open_exit_ids)
        if any(i not in self._by_id for i in ids):
            raise ValueError('Unknown exit ID')
        # Passing an explicit set selects only those exits; closed world exits
        # remain closed regardless of a stale caller selection.
        return tuple(sorted(i for i in set(ids) if self._by_id[i]['open']))

    def _update_routes(self, ids, hazard_weight):
        if not math.isfinite(hazard_weight) or hazard_weight < 0:
            raise ValueError('hazard_weight must be finite and nonnegative')
        key = (ids, float(hazard_weight), int(self.time_s) if hazard_weight else 0)
        if key == self._route_key:
            return
        cost = self._clearance_cost.copy()
        if hazard_weight:
            c = self.hazard._selected_grid()[self._safe]
            cost += hazard_weight * c / (1 + c)
        values = self._length * .5 * (cost[self._row] + cost[self._col])
        graph = sparse.csr_matrix((values, (self._row, self._col)), shape=(len(cost), len(cost)))
        self._routes = {}
        for ident in ids:
            exit_ = self._by_id[ident]
            axis, sign, crossing = exit_['axis'], exit_['sign'], exit_['crossing']
            span = exit_['span']
            terminal = ((self._cells[:, 1-axis] >= span[0]+self.routing_clearance)
                        & (self._cells[:, 1-axis] <= span[1]-self.routing_clearance)
                        & (sign*(self._cells[:,axis]-crossing) >= .75)
                        & (sign*(self._cells[:,axis]-crossing) <= 1.25))
            source = np.flatnonzero(terminal).astype(np.int32)
            if not len(source):
                raise ValueError(f'Exit {ident} has no safe terminal cells')
            distance, predecessor, _ = dijkstra(graph, directed=False, indices=source,
                                               min_only=True, return_predecessors=True)
            self._routes[ident] = (distance, predecessor)
        self._route_key = key

    def navigation(self, positions, open_exit_ids=None, hazard_weight=0.0):
        """Return local next-cell directions and chosen exit IDs for each point.

        The shortest-cost field is recomputed at most once per simulated second
        under changing hazard. Closed exits are excluded immediately.
        """
        p = np.asarray(positions, float).reshape(-1, 2)
        ids = self._open_ids(open_exit_ids)
        direction = np.zeros_like(p)
        chosen = np.full(len(p), '', dtype='<U8')
        if not ids or not len(p):
            return direction, chosen
        self._update_routes(ids, hazard_weight)
        ix, iy = self.map.world_to_cell(p)
        ix, iy = np.clip(ix,0,self.map.nx-1), np.clip(iy,0,self.map.ny-1)
        nodes = self._node[iy, ix]
        if np.any(nodes < 0):
            # Route callers outside conservative clearance to a nearby safe cell.
            from scipy.spatial import cKDTree
            if not hasattr(self, '_safe_tree'):
                self._safe_tree = cKDTree(self._cells)
            nodes[nodes < 0] = self._safe_tree.query(p[nodes < 0])[1]
        costs = np.stack([self._routes[ident][0][nodes] for ident in ids])
        best = costs.argmin(axis=0)
        reachable = np.isfinite(costs[best, np.arange(len(p))])
        for k, ident in enumerate(ids):
            subset = np.flatnonzero((best == k) & reachable)
            if not len(subset):
                continue
            chosen[subset] = ident
            predecessor = self._routes[ident][1][nodes[subset]]
            follow = predecessor >= 0
            direction[subset[follow]] = self._cells[predecessor[follow]] - p[subset[follow]]
            ex = self._by_id[ident]
            # At a terminal the outward direction remains meaningful until the
            # crowd engine removes the agent for crossing the exit line.
            direction[subset[~follow], ex['axis']] = ex['sign']
        norm = np.linalg.norm(direction,axis=1)
        direction /= np.maximum(norm[:,None], 1e-12)
        return direction, chosen

    def crossed_exit(self, positions, open_exit_ids=None):
        """Per-position exit id, or empty string; requires crossing its actual line."""
        p = np.asarray(positions, float).reshape(-1,2)
        result = np.full(len(p), '', dtype='<U8')
        for ident in self._open_ids(open_exit_ids):
            e = self._by_id[ident]
            axis, sign = e['axis'], e['sign']
            crossed = (sign*(p[:,axis]-e['crossing']) >= 0)
            crossed &= (p[:,1-axis] >= e['span'][0]) & (p[:,1-axis] <= e['span'][1])
            result[crossed] = ident
        return result

    def geometry_snapshot(self):
        m = self.map
        return {'width': self.width, 'height': self.height,
                'bounds': list(m.bounds), 'venue_bounds': list(self.venue_bounds),
                'outer': [r.tolist() for r in m.outer], 'obstacles': self.obstacles,
                'exits': [dict(e) for e in self.exits], 'resolution': self.resolution,
                'origin': list(m.origin), 'grid_width': m.nx, 'grid_height': m.ny,
                'walkable': m.mask.astype(np.uint8).ravel().tolist(),
                'routing_clearance_m': self.routing_clearance}

    def environment_snapshot(self):
        result = self.hazard.environment_snapshot()
        result.update({'kind': 'repellent', 'enabled': self.hazard_enabled,
                       'strength': self.hazard.sources[0].strength})
        return result

    def environment(self):
        return self.environment_snapshot()
