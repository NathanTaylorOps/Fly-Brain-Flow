"""Seeded behavioral crowd arena with measured contacts and evacuation metrics.

This baseline uses engineered exit routes and local avoidance. It does not run
fruit-fly brains, biological vision, pain, or neural learning. Mass, inertia and
contact friction are illustrative 2D mechanics, not a calibrated human model.
"""
from __future__ import annotations

from collections import deque
import copy
import math
import time

import numpy as np
from scipy.spatial import cKDTree

from .world.arena_world import ArenaWorld


DEFAULTS = dict(population=200, seed=7, preset='open', wind_direction_deg=180.,
                wind_speed_mps=1., hazard_strength=0., hazard_position=[32., 24.])


def _number(value, name, minimum, maximum, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be numeric')
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    if integer and int(value) != value:
        raise ValueError(f'{name} must be an integer')
    return int(value) if integer else float(value)


class ArenaSim:
    model = {'label': 'Behavioral crowd baseline',
             'description': 'Independent seeded behavioral agents; engineered exit routing, reaction delays, local repellent-gradient avoidance, inertia and contacts. Before alarm, a virtual boundary holds the initial crowd inside for comparison. No real-connectome navigation, biological vision or learning.',
             'physics': 'Adaptive swept grid checks; mass-weighted contact projection and tangential friction. Illustrative model units.'}

    def __init__(self, config=None, *, population=None, seed=None, scenario=None):
        overrides = dict(config or {})
        for key, value in [('population', population), ('seed', seed), ('preset', scenario)]:
            if value is not None:
                overrides[key] = value
        self.reset(overrides)

    @staticmethod
    def _config(config):
        c = copy.deepcopy(DEFAULTS)
        c.update(config)
        unknown = set(c) - set(DEFAULTS)
        if unknown:
            raise ValueError('Unknown configuration: ' + ', '.join(sorted(unknown)))
        c['population'] = _number(c['population'], 'population', 50, 500, True)
        c['seed'] = _number(c['seed'], 'seed', 0, 2**32-1, True)
        if c['preset'] not in ('open', 'blocked', 'surge'):
            raise ValueError('preset must be open, blocked or surge')
        for key, high in [('wind_direction_deg', 360), ('wind_speed_mps', 3), ('hazard_strength', 5)]:
            c[key] = _number(c[key], key, 0, high)
        xy = c['hazard_position']
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError('hazard_position must contain x and y')
        c['hazard_position'] = [_number(x, 'hazard_position', -1e5, 1e5) for x in xy]
        return c

    def reset(self, config=None):
        c = self._config({**getattr(self, 'config', {}), **(config or {})})
        world = ArenaWorld()
        if not world.map.is_walkable([c['hazard_position']])[0]:
            raise ValueError('Hazard must be in walkable space')
        world.set_wind(c['wind_direction_deg'], c['wind_speed_mps'])
        world.set_hazard(c['hazard_position'], strength=c['hazard_strength'])
        if c['preset'] == 'blocked':
            for e in world.exits[2:]:
                world.set_exit(e['id'], False)
        rng = np.random.default_rng(c['seed'])
        n = c['population']
        positions = world.spawn_points(n, seed=c['seed'], radius=.24)
        self.world, self.config, self.rng = world, c, rng
        self.positions = np.asarray(positions, float)
        self.velocities = np.zeros((n, 2))
        self.ids = np.arange(n, dtype=int)
        self.active = np.ones(n, dtype=bool)
        self.mass = rng.uniform(.8, 1.2, n)
        self.radius = .24
        self.preferred_speed = rng.uniform(1.15, 1.65, n) * (1.25 if c['preset'] == 'surge' else 1.)
        self.reaction_delay = rng.uniform(.2, 2.5, n) * (.35 if c['preset'] == 'surge' else 1.)
        self.headings = rng.uniform(-np.pi, np.pi, n)
        self.wander_phase = rng.uniform(0, 2*np.pi, n)
        self.exit_ids = np.full(n, '', dtype=object)
        self.exposure = np.zeros(n)
        self.agent_status = np.full(n, 'wandering', dtype=object)
        self.time_s, self.alarm_time = 0., None
        self.evacuation_times = []
        self.evacuated_by_exit = {e['id']: 0 for e in world.exits}
        self.history = deque(maxlen=1200)
        self._events = deque(maxlen=2000)
        self._next_history = 0.
        self._wall_corrections = 0
        self._overlap_pairs, self._max_overlap = 0, 0.
        self._step_wall_s = 0.
        self._safe_clearance = self.radius + world.map.resolution * .75
        return self.state()

    def _safe(self, positions):
        positions = np.asarray(positions).reshape(-1, 2)
        safe = self.world.map.distance_at(positions) >= self._safe_clearance
        for e in self.world.exits:
            if not e['open'] or self.alarm_time is None:
                axis = e['axis']
                at_gate = (positions[:, 1-axis] >= e['span'][0]-self.radius) & (positions[:, 1-axis] <= e['span'][1]+self.radius)
                outside = e['sign']*(positions[:, axis]-e['crossing']) > -self.radius
                safe &= ~(at_gate & outside)
        return safe

    def _swept_move(self, old, proposed):
        """Stop or slide if any part of the intended segment leaves safe cells."""
        if not len(old):
            return proposed
        delta = proposed-old
        sample_count = max(2, int(np.ceil(np.max(np.linalg.norm(delta, axis=1)) /
                                        (self.world.map.resolution*.2))) + 1)
        fractions = np.linspace(0, 1, sample_count)
        samples = old[:, None, :] + delta[:, None, :] * fractions[None, :, None]
        legal = self._safe(samples.reshape(-1, 2)).reshape(len(old), sample_count).all(axis=1)
        result = proposed.copy()
        bad = np.flatnonzero(~legal)
        self._wall_corrections += len(bad)
        for k in bad:
            # Try tangential axis slides; neither path can cross a closed cell.
            choices = [old[k] + [delta[k, 0], 0], old[k] + [0, delta[k, 1]]]
            choices.sort(key=lambda p: -np.linalg.norm(p-old[k]))
            result[k] = old[k]
            for candidate in choices:
                path = old[k] + fractions[:, None] * (candidate-old[k])
                if self._safe(path).all():
                    result[k] = candidate
                    break
        return result

    def _contacts(self, pos, vel, mass):
        """Project disk overlap without allowing the correction through a wall."""
        if len(pos) < 2:
            return pos, vel
        diameter = 2*self.radius
        for iteration in range(12):
            pairs = cKDTree(pos).query_pairs(diameter, output_type='ndarray')
            if not len(pairs):
                break
            a, b = pairs[:, 0], pairs[:, 1]
            delta = pos[b]-pos[a]
            distances = np.linalg.norm(delta, axis=1)
            overlap = diameter-distances
            if overlap.max(initial=0) < .001:
                break
            normals = delta / np.maximum(distances[:, None], 1e-9)
            coincident = distances < 1e-9
            normals[coincident] = [1., 0.]
            wa = mass[b]/(mass[a]+mass[b])
            wb = 1-wa
            correction = np.zeros_like(pos)
            np.add.at(correction, a, -normals*overlap[:, None]*wa[:, None]*.7)
            np.add.at(correction, b, normals*overlap[:, None]*wb[:, None]*.7)
            # A bounded projection prevents a dense neighbor sum from jumping walls.
            sizes = np.linalg.norm(correction, axis=1)
            correction *= np.minimum(1, self.radius*.4/np.maximum(sizes, 1e-9))[:, None]
            pos = self._swept_move(pos, pos+correction)
            # Inelastic normal impulse and mild tangential contact friction.
            rel = vel[b]-vel[a]
            vn = np.sum(rel*normals, axis=1)
            impulse = np.minimum(vn, 0)[:, None]*normals
            changes = np.zeros_like(vel)
            np.add.at(changes, a, impulse*wa[:, None])
            np.add.at(changes, b, -impulse*wb[:, None])
            tangent = rel-vn[:, None]*normals
            np.add.at(changes, a, tangent*wa[:, None]*.04)
            np.add.at(changes, b, -tangent*wb[:, None]*.04)
            vel += changes*.5
        return pos, vel

    def _advance(self, dt):
        self.world.advance(dt)
        idx = np.flatnonzero(self.active)
        if not len(idx):
            self.time_s += dt
            return
        pos, vel = self.positions[idx].copy(), self.velocities[idx].copy()
        reacting = np.zeros(len(idx), dtype=bool)
        desired = np.zeros_like(pos)
        if self.alarm_time is not None:
            reacting = self.time_s-self.alarm_time >= self.reaction_delay[idx]
            if reacting.any():
                directions, exits = self.world.navigation(pos[reacting], hazard_weight=2.5)
                desired[reacting] = directions*self.preferred_speed[idx[reacting], None]
                self.exit_ids[idx[reacting]] = exits
        calm = ~reacting
        wander_angle = self.headings[idx[calm]] + .35*np.sin(self.time_s*.6+self.wander_phase[idx[calm]])
        desired[calm] = .20*np.c_[np.cos(wander_angle), np.sin(wander_angle)]
        concentration = np.asarray(self.world.hazard.sample(pos))
        sensing_repellent = calm & (concentration > 1e-8)
        if sensing_repellent.any():
            # Explicit engineered local response: walk down the measured scalar
            # concentration gradient. This is neither vision nor neural learning.
            gradient = self.world.hazard.gradient(pos[sensing_repellent])
            magnitude = np.linalg.norm(gradient, axis=1)
            response = concentration[sensing_repellent] / (.2+concentration[sensing_repellent])
            desired[sensing_repellent] -= .75 * response[:, None] * gradient / np.maximum(magnitude[:, None], 1e-9)
        # Anticipatory local avoidance, based on neighboring disks only.
        pairs = cKDTree(pos).query_pairs(1.1, output_type='ndarray')
        avoid = np.zeros_like(pos)
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            delta = pos[b]-pos[a]
            distance = np.linalg.norm(delta, axis=1)
            norm = delta/np.maximum(distance[:, None], 1e-8)
            strength = np.clip((1.0-distance)/.55, 0, 1.5)**2
            np.add.at(avoid, a, -norm*strength[:, None])
            np.add.at(avoid, b, norm*strength[:, None])
        acceleration = (desired-vel)/.35 + 2.*avoid/self.mass[idx, None]
        size = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(1, 4./np.maximum(size, 1e-9))[:, None]
        # Substeps bound traveled distance relative to body radius, even under pressure.
        maxspeed = max(float(np.linalg.norm(vel, axis=1).max(initial=0)), 2.2)
        substeps = max(1, math.ceil((maxspeed+4*dt)*dt/(self.radius*.35)))
        h = dt/substeps
        for _ in range(substeps):
            vel += acceleration*h
            speed = np.linalg.norm(vel, axis=1)
            vel *= np.minimum(1, 2.2/np.maximum(speed, 1e-9))[:, None]
            candidate = self._swept_move(pos, pos+vel*h)
            blocked = np.linalg.norm(candidate-(pos+vel*h), axis=1) > 1e-9
            vel[blocked] = (candidate[blocked]-pos[blocked])/h
            pos, vel = self._contacts(candidate, vel, self.mass[idx])
        self.positions[idx], self.velocities[idx] = pos, vel
        speed = np.linalg.norm(vel, axis=1)
        moving = speed > .03
        self.headings[idx[moving]] = np.arctan2(vel[moving, 1], vel[moving, 0])
        self.agent_status[idx] = np.where(reacting, np.where(speed>.15, 'evacuating', 'waiting'), 'wandering')
        self.agent_status[idx[sensing_repellent]] = 'avoiding'
        self.exposure[idx] += np.asarray(self.world.hazard.sample(pos))*dt
        self.time_s += dt
        for e in self.world.exits:
            if self.alarm_time is None:
                break  # Gathering never increments evacuation or loses agents.
            if not e['open']:
                continue
            axis, sign, crossing = e['axis'], e['sign'], e['crossing']
            lo, hi = e['span']
            crossed = (sign*(pos[:, axis]-crossing) > 0) & (pos[:, 1-axis]>=lo) & (pos[:, 1-axis]<=hi)
            crossed &= self.active[idx]
            gone = idx[crossed]
            if len(gone):
                self.active[gone] = False
                self.agent_status[gone] = 'evacuated'
                self.exit_ids[gone] = e['id']
                self.evacuated_by_exit[e['id']] += len(gone)
                self.evacuation_times.extend([self.time_s]*len(gone))
        remaining = self.positions[self.active]
        pairs = cKDTree(remaining).query_pairs(2*self.radius-.001, output_type='ndarray') if len(remaining)>1 else []
        self._overlap_pairs = len(pairs)
        self._max_overlap = (float((2*self.radius-np.linalg.norm(remaining[pairs[:, 1]]-remaining[pairs[:, 0]], axis=1)).max()) if len(pairs) else 0.)
        if self.time_s >= self._next_history:
            self.history.append({'time': round(self.time_s, 3), 'remaining': int(self.active.sum()),
                                 'evacuated': int((~self.active).sum()), 'max_overlap': self._max_overlap})
            self._next_history = self.time_s+1.

    def step(self, steps=1, dt=.05):
        steps = _number(steps, 'steps', 1, 100, True)
        dt = _number(dt, 'dt', .001, .1)
        started = time.perf_counter()
        for _ in range(steps):
            self._advance(dt)
        self._step_wall_s = time.perf_counter()-started
        return self.state()

    def control(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError('Controls must be a nonempty object')
        allowed = {'alarm','exit','preset','wind_direction_deg','wind_speed_mps','hazard_strength','hazard_position'}
        if set(values)-allowed:
            raise ValueError('Unsupported control')
        if 'preset' in values and len(values) != 1:
            raise ValueError('Preset reset must be a separate control operation')
        if 'alarm' in values and not isinstance(values['alarm'], bool):
            raise ValueError('alarm must be boolean')
        changes = {k: v for k, v in values.items() if k in DEFAULTS}
        checked = self._config({**self.config, **changes})
        if not self.world.map.is_walkable([checked['hazard_position']])[0]:
            raise ValueError('Hazard must be in walkable space')
        if 'exit' in values:
            e = values['exit']
            if not isinstance(e, dict) or set(e) != {'id','open'} or not isinstance(e['open'], bool):
                raise ValueError('exit must contain id and boolean open')
            if e['id'] not in {x['id'] for x in self.world.exits}:
                raise ValueError('Unknown exit id')
        if 'preset' in values:
            return self.reset(checked)
        if values.get('alarm') and self.alarm_time is None:
            self.alarm_time = self.time_s
        if 'alarm' in values and not values['alarm']:
            self.alarm_time = None
            # Restore the experimental gathering boundary if alarm is cancelled
            # while a body is touching an otherwise open crossing line.
            for gate in self.world.exits:
                axis, sign = gate['axis'], gate['sign']
                at_gate = self.active & (self.positions[:, 1-axis] >= gate['span'][0]-self.radius) & (self.positions[:, 1-axis] <= gate['span'][1]+self.radius)
                clipped = at_gate & (sign*(self.positions[:, axis]-gate['crossing']) > -self.radius-.001)
                self.positions[clipped, axis] = gate['crossing']-sign*(self.radius+.001)
                self.velocities[clipped, axis] = 0.
        if 'exit' in values:
            self.world.set_exit(values['exit']['id'], values['exit']['open'])
            if not values['exit']['open']:
                gate = next(e for e in self.world.exits if e['id'] == values['exit']['id'])
                axis, sign = gate['axis'], gate['sign']
                at_gate = self.active & (self.positions[:, 1-axis] >= gate['span'][0]-self.radius) & (self.positions[:, 1-axis] <= gate['span'][1]+self.radius)
                clipped = at_gate & (sign*(self.positions[:, axis]-gate['crossing']) > -self.radius-.001)
                self.positions[clipped, axis] = gate['crossing']-sign*(self.radius+.001)
                self.velocities[clipped, axis] = 0.
        self.world.set_wind(checked['wind_direction_deg'], checked['wind_speed_mps'])
        self.world.set_hazard(checked['hazard_position'], strength=checked['hazard_strength'])
        self.config = checked
        self._events.append({'time': self.time_s, 'control': copy.deepcopy(values)})
        return self.state()

    def state(self):
        idx = np.flatnonzero(self.active)
        speeds = np.linalg.norm(self.velocities[idx], axis=1)
        evacuated = int((~self.active).sum())
        phase = 'complete' if not len(idx) else ('evacuating' if self.alarm_time is not None else 'calm')
        recent = sum(t > self.time_s-5 for t in self.evacuation_times)
        peak_density = (float(max(cKDTree(self.positions[idx]).query_ball_point(self.positions[idx], 2., return_length=True))) / (np.pi*4) if len(idx) else 0.)
        agents = [{'id': int(i), 'x': float(self.positions[i, 0]), 'y': float(self.positions[i, 1]),
                   'position': self.positions[i].tolist(), 'vx': float(self.velocities[i, 0]),
                   'vy': float(self.velocities[i, 1]), 'heading': float(self.headings[i]),
                   'speed': float(np.linalg.norm(self.velocities[i])), 'state': str(self.agent_status[i]),
                   'status': str(self.agent_status[i]), 'exit_id': str(self.exit_ids[i]),
                   'exposure': float(self.exposure[i]), 'radius': self.radius} for i in idx]
        metrics = {'evacuated': evacuated, 'remaining': len(idx), 'overlap_pairs': self._overlap_pairs,
                   'max_overlap': self._max_overlap, 'stalled': int(np.sum(speeds < .1)) if self.alarm_time is not None else 0,
                   'mean_speed': float(speeds.mean()) if len(speeds) else 0.,
                   'density': len(idx)/self.world.map.walkable_area_m2,
                   'throughput_per_s': recent/min(5., max(self.time_s, .05)),
                   'peak_local_density': peak_density,
                   'last_evacuation_time': max(self.evacuation_times) if self.evacuation_times else None,
                   'evacuation_elapsed': ((max(self.evacuation_times) if not len(idx) and self.evacuation_times else self.time_s)-self.alarm_time) if self.alarm_time is not None else 0.,
                   'evacuated_by_exit': dict(self.evacuated_by_exit),
                   'wall_corrections': self._wall_corrections, 'last_step_wall_s': self._step_wall_s}
        return {'time': self.time_s, 'time_s': self.time_s, 'phase': phase, 'alarm_time': self.alarm_time,
                'population': self.config['population'], 'remaining': len(idx), 'evacuated': evacuated,
                'agents': agents, 'metrics': metrics, 'exits': copy.deepcopy(self.world.exits),
                'config': copy.deepcopy(self.config), 'history': list(self.history), 'model': self.model,
                'capabilities': {'alarm': True, 'wind': True, 'hazard': True, 'exit_control': True,
                                 'neural_simulation': False, 'learning': False}}

    def environment(self):
        geometry = self.world.geometry_snapshot()
        return {**geometry, 'arena': geometry, 'hazard': self.world.environment()}

    def export(self):
        return {'schema': 'flybrainflow-arena-observation-v1', 'state': self.state(),
                'environment': self.environment(), 'events': list(self._events),
                'note': 'Behavioral-model observations, not neural data or exact physics replay.'}
