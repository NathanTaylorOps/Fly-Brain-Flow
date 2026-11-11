"""Explicit behavioral comparison: local odor search and obstacle avoidance.

This is hand-written behavior, NOT an output of the connectome. The controller
never reads food positions, a goal-vector, or a shortest-path route. It samples
odor at nearby points, chooses a safe direction, and remembers visited places.
Wind is a weak upwind preference while odor is present. The plume model itself
can convey information around barriers; this is an environmental assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@dataclass
class _State:
    heading: float
    rng: np.random.Generator
    elapsed: float = 0.
    next_pause: float = 6.
    pause_until: float = 0.
    odor: float = 0.
    anchor_position: np.ndarray | None = None
    anchor_time: float = 0.
    escape_until: float = 0.
    visited: dict = field(default_factory=dict)
    behavior: str = 'casting'


class ForagerBrain:
    """A documented behavioral baseline that can move without a neural model."""
    control_capabilities = {'odor': True, 'stimulus': False, 'silence': False}
    def __init__(self, walkable_map, *, seed=7):
        self.map, self.seed = walkable_map, int(seed)
        self.states, self.headings, self.last_sense = {}, {}, {}
        self._snapshot = {}
        self.time_s = 0.
        self.odor_enabled = True
        self._angles = np.linspace(-np.pi, np.pi, 32, endpoint=False)

    def assigned_targets(self):
        # Lifecycle bookkeeping only; the controller never reads target locations.
        return {i: 0 for i in self.states}

    def move_target(self, index, new_xy):
        # No goal memory: the changed environmental plume is sensed next tick.
        pass

    def forget(self, ids):
        for i in ids:
            for collection in (self.states, self.headings, self.last_sense):
                collection.pop(i, None)
            self._snapshot.pop(str(i), None)

    def set_controls(self, *, odor_enabled=None):
        if odor_enabled is not None:
            self.odor_enabled = bool(odor_enabled)

    def behavioral_snapshot(self):
        return {
            'controller': 'behavioral', 'time_s': self.time_s,
            'description': 'Hand-written local odor search, upwind preference, obstacle avoidance and exploration memory. Not connectome-driven navigation.',
            'odor_enabled': self.odor_enabled,
            'agents': dict(self._snapshot),
        }

    def _sample(self, odor_field, points):
        if not self.odor_enabled:
            return np.zeros(len(points))
        values = np.asarray(odor_field.sample(points), float).reshape(len(points), -1).sum(axis=1)
        return np.maximum(np.nan_to_num(values, nan=0., posinf=0., neginf=0.), 0.)

    def desired_velocities(self, ids, positions, radii, max_speed_mps,
                           preferred_targets=None, *, personalities, odor_field, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        positions = np.asarray(positions, float).reshape(-1, 2)
        radii = np.broadcast_to(radii, (len(ids),))
        max_speeds = np.broadcast_to(max_speed_mps, (len(ids),))
        velocities = np.zeros_like(positions)
        self.time_s += dt
        for row, (agent_id, pos) in enumerate(zip(ids, positions)):
            if agent_id not in self.states:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(agent_id)]))
                self.states[agent_id] = _State(float(rng.uniform(-np.pi, np.pi)), rng)
            state = self.states[agent_id]
            state.elapsed += dt
            if state.anchor_position is None:
                state.anchor_position = pos.copy()
            if state.elapsed - state.anchor_time >= 2.5:
                if np.linalg.norm(pos - state.anchor_position) < .45:
                    # Local plume maxima and cul-de-sacs require leaving the
                    # attractive patch. This uses only remembered motion.
                    state.escape_until = state.elapsed + 4.
                state.anchor_position = pos.copy()
                state.anchor_time = state.elapsed
            radius = float(radii[row])
            # Both antennae and the directional fan are local measurements.
            reach = max(.65, 3 * radius, 4 * self.map.resolution)
            directions = np.c_[np.cos(self._angles), np.sin(self._angles)]
            probes = pos + reach * directions
            # A small radial sensor fan; no access to terrain beyond this radius.
            samples = pos + directions[:, None, :] * np.linspace(.03, reach, 24)[None, :, None]
            clearance = self.map.distance_at(samples.reshape(-1, 2)).reshape(32, 24)
            clearance_limit = radius + self.map.resolution * .75
            safe = np.min(clearance, axis=1) >= clearance_limit
            probe_odor = self._sample(odor_field, probes)
            lateral = np.array([-np.sin(state.heading), np.cos(state.heading)]) * max(radius, .12)
            antennae = self._sample(odor_field, np.array([pos + lateral, pos - lateral]))
            center = float(self._sample(odor_field, [pos])[0])
            delta = probe_odor - center
            delta[~safe] = 0
            # Normalizing the spatial difference preserves weak distant gradients.
            spread = max(float(np.max(np.abs(delta))), 1e-12)
            uphill = float(np.max(delta[safe])) if np.any(safe) else 0.
            odor_signal = (center > 1e-9 and uphill > max(1e-12, center * 1e-6)
                           and state.elapsed >= state.escape_until)
            odor_score = delta / spread if odor_signal else np.zeros(32)
            angles_from_heading = _angle(self._angles - state.heading)
            continuity = (.10 if odor_signal else .35) * np.cos(angles_from_heading)
            wall_margin = .12 * np.minimum(np.min(clearance, axis=1) / reach, 1)
            keys = [tuple(np.floor(p / .6).astype(int)) for p in probes]
            visits = np.array([state.visited.get(key, 0.) for key in keys])
            novelty = -.30 * np.minimum(visits, 5)
            # When smell rises reliably, prioritize following it over exploration.
            if odor_signal:
                novelty *= .12
            wind = np.asarray(odor_field.local_wind([pos]), float).reshape(-1, 2)[0]
            wind_norm = float(np.linalg.norm(wind))
            upwind = -.06 * directions @ (wind / wind_norm) if wind_norm > 1e-9 and center > 1e-9 else 0.
            scores = 2.5 * odor_score + continuity + wall_margin + novelty + upwind
            if not odor_signal:
                scores += .08 * np.sin(self._angles + state.elapsed * .18 + (agent_id % 11))
            scores[~safe] = -np.inf
            if not np.any(safe):
                # Shorten the wall sensor in tight corners; physics still constrains motion.
                probes = pos + directions * max(radius * 1.5, self.map.resolution)
                local_clearance = self.map.distance_at(probes)
                desired_angle = self._angles[int(np.argmax(local_clearance))]
                chosen_clearance = float(np.max(local_clearance))
                state.behavior = 'turning'
            else:
                desired_angle = self._angles[int(np.argmax(scores))]
                chosen_clearance = float(np.min(clearance[np.argmax(scores)]))
                state.behavior = 'surging' if odor_signal else 'casting'
            error = float(_angle(desired_angle - state.heading))
            state.heading = float(_angle(state.heading + np.clip(error, -4.5 * dt, 4.5 * dt)))
            if abs(error) > .7:
                state.behavior = 'turning'
            # Fly turns on the spot for a sharp change, then resumes walking.
            alignment = max(0., np.cos(error))
            speed = float(max_speeds[row]) * (.70 + .16 * np.sin(state.elapsed * 2.1 + agent_id)) * alignment
            if chosen_clearance < clearance_limit:
                speed = 0.
            # Brief investigative pauses, followed by a new walking bout.
            if state.elapsed >= state.next_pause:
                state.pause_until = state.elapsed + float(state.rng.uniform(.12, .30))
                state.next_pause = state.elapsed + float(state.rng.uniform(4., 8.))
            if state.elapsed < state.pause_until:
                speed, state.behavior = 0., 'sniffing'
            direction = np.array([np.cos(state.heading), np.sin(state.heading)])
            # Heading changes are gradual, so the current heading may differ
            # from the safe chosen ray. Check the actual intended swept path.
            forward_points = pos + direction * (speed * dt) * np.linspace(.25, 1., 4)[:, None]
            forward_clearance = self.map.distance_at(forward_points)
            current_clearance = float(self.map.distance_at([pos])[0])
            if np.any(forward_clearance < min(clearance_limit, current_clearance)):
                speed = 0.
                if state.behavior != 'sniffing':
                    state.behavior = 'turning'
            velocities[row] = speed * direction
            key = tuple(np.floor(pos / .6).astype(int))
            state.visited[key] = state.visited.get(key, 0.) + dt
            # Only observed visits are remembered; never a target route.
            if len(state.visited) > 2000:
                state.visited = {k: v*.5 for k, v in state.visited.items() if v > .1}
            state.odor = center
            self.headings[agent_id] = state.heading
            self.last_sense[agent_id] = np.array([center, antennae[0], antennae[1], wind[0], wind[1], speed])
            self._snapshot[str(agent_id)] = {
                'heading': state.heading, 'state': state.behavior, 'odor': center,
                'odor_left': float(antennae[0]), 'odor_right': float(antennae[1]),
                'speed_mps': speed,
            }
        return velocities
