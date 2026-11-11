"""Experimental real-connectome controller with an explicit 2D movement decoder.

No goal-vector or baseline steering enters the output. A bilateral general ORN
input can be enabled to test odor feedback; it is not receptor-specific sensing.
DNp09 supplies forward drive, DNp01 supplies an illustrative escape impulse,
and DNa02 left-minus-right supplies yaw. These mappings and scales are declared
engineering assumptions, not validated locomotor physiology.
"""
from __future__ import annotations

import numpy as np
from ..spiking.live_engine import LiveLIFEngine


class LiveConnectomeBrain:
    def __init__(self, graph, *, seed=7, params=None, odor_enabled=False):
        self.graph = graph
        self.template = LiveLIFEngine(graph.indptr, graph.indices, graph.weights,
                                      params=params, seed=seed)
        self.seed = seed
        self.engines, self.headings, self.last_sense = {}, {}, {}
        self.stimulus_hz, self.stimulus_group, self.silenced = 0., 'loom', False
        self.odor_enabled = bool(odor_enabled)
        self._readouts = ['MN9', 'DNp01', 'DNp09', 'DNa02_L', 'DNa02_R']
        self._rates = {name: 0. for name in self._readouts}
        self._activity, self._total_spikes, self._time_ms = [], 0, 0.
        required = ['sugar', 'loom', 'ORN_L', 'ORN_R', *self._readouts]
        missing = [name for name in required if not len(graph.populations.get(name, []))]
        if missing:
            raise ValueError('Missing annotated populations: ' + ', '.join(missing))

    def set_controls(self, *, stimulus_hz=None, silenced=None, stimulus_group=None,
                     odor_enabled=None):
        if stimulus_hz is not None:
            if not np.isfinite(stimulus_hz) or not 0 <= stimulus_hz <= 1000:
                raise ValueError('stimulus_hz must be 0..1000')
            self.stimulus_hz = float(stimulus_hz)
        if stimulus_group is not None:
            if stimulus_group not in ('sugar', 'loom'):
                raise ValueError('stimulus_group must be sugar or loom')
            self.stimulus_group = stimulus_group
        if silenced is not None:
            self.silenced = bool(silenced)
        if odor_enabled is not None:
            self.odor_enabled = bool(odor_enabled)

    def assigned_targets(self):
        # One target is used only for lifecycle capture, not steering.
        return {i: 0 for i in self.engines}

    def move_target(self, index, new_xy):
        pass  # OdorField owns the current location; no cached goal vector.

    def forget(self, ids):
        for i in ids:
            self.engines.pop(i, None)
            self.headings.pop(i, None)
            self.last_sense.pop(i, None)

    def desired_velocities(self, ids, positions, radii, max_speed_mps,
                           preferred_targets=None, *, personalities, odor_field, dt):
        result = np.zeros((len(ids), 2))
        groups = self.graph.populations
        lesion = np.unique(np.concatenate([groups[x] for x in self._readouts]))
        snapshots = []
        for row, (agent_id, pos) in enumerate(zip(ids, positions)):
            if agent_id not in self.engines:
                self.engines[agent_id] = self.template.fork(seed=np.random.SeedSequence(
                    [self.seed, int(agent_id)]).generate_state(1)[0])
                self.headings[agent_id] = 0.
            engine = self.engines[agent_id]
            heading = self.headings[agent_id]
            sensory = np.zeros(6)
            drive = np.zeros(engine.n)
            drive[groups[self.stimulus_group]] = self.stimulus_hz
            if self.odor_enabled:
                lateral = np.array([-np.sin(heading), np.cos(heading)]) * .1
                antennae = np.asarray([pos + lateral, pos - lateral])
                # Sum all odor sources, with an explicitly declared saturating encoder.
                concentration = np.asarray(odor_field.sample(antennae)).reshape(2, -1).sum(axis=1)
                rates = 100 * np.maximum(concentration, 0) / (1 + np.maximum(concentration, 0))
                for name, rate in zip(('ORN_L', 'ORN_R'), rates):
                    drive[groups[name]] = rate
                sensory[:2] = rates
            sensory[2] = self.stimulus_hz
            self.last_sense[agent_id] = sensory
            active = np.flatnonzero(drive)
            engine.set_poisson(active, drive[active])
            engine.set_silenced(lesion if self.silenced else [])
            spikes = engine.advance(dt * 1000)
            rates = {name: spikes.rate_hz(groups[name]) for name in self._readouts}
            # Declared transducer: zero motor spikes => zero translation/yaw.
            speed = max_speed_mps * np.clip((rates['DNp09'] + rates['DNp01']) / 100., 0, 1)
            yaw = np.clip((rates['DNa02_L'] - rates['DNa02_R']) / 50., -1, 1) * 2.
            heading += yaw * dt
            self.headings[agent_id] = heading
            result[row] = speed * np.array([np.cos(heading), np.sin(heading)])
            snapshots.append(rates)
            self._total_spikes = int(spikes.counts.sum())
            self._time_ms = engine.time_ms
            top = np.flatnonzero(spikes.counts)
            top = top[np.argsort(spikes.counts[top])[-12:][::-1]]
            self._activity = [{'body_id': str(self.graph.body_ids[i]),
                              'type': str(self.graph.types[i]),
                              'spikes': int(spikes.counts[i]),
                              'position': (self.graph.positions[i].tolist()
                                           if np.isfinite(self.graph.positions[i]).all() else None)} for i in top]
        if snapshots:
            self._rates = {name: float(np.mean([r[name] for r in snapshots])) for name in self._readouts}
        return result

    def neural_snapshot(self):
        return {'rates_hz': self._rates, 'last_step_spikes': self._total_spikes,
                'time_ms': self._time_ms, 'active_brains': len(self.engines),
                'top_active': self._activity,
                'stimulus_description': ('LC4/LPLC2 visual pathway' if self.stimulus_group == 'loom'
                                          else 'LB3b/LB3c taste pathway') + ': direct Poisson input.',
                'odor_enabled': self.odor_enabled,
                'decoder': 'DNp09 forward + DNp01 escape; DNa02 L-R yaw. Engineered 2D transducer.',
                'parameters': vars(self.template.params)}

    def anatomy_snapshot(self):
        valid = np.flatnonzero(np.isfinite(self.graph.positions).all(axis=1))
        ids = valid[::max(1, int(np.ceil(len(valid) / 6000)))]
        return {'positions': self.graph.positions[ids].tolist(),
                'note': 'Downsampled measured soma coordinates; all loaded neurons still simulate.',
                'populations': {name: [self.graph.positions[i].tolist() for i in self.graph.populations[name]
                                      if np.isfinite(self.graph.positions[i]).all()]
                                for name in ('sugar', 'loom', 'MN9', 'DNp01')}}
