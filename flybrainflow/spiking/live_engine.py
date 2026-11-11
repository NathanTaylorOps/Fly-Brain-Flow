"""Incremental CPU LIF simulation (mV, ms, Hz), independent of any body/UI.

The active-set loop is adapted from NewYorkImperialist/fly-simulator at
ab8d479c524ac31710a78b76244bf5e5a5bbc44d, brain/engine.py. Its equations follow
Shiu et al. (2024). Numerical agreement is not evidence of biological validity.
Only synaptic contacts from spiking cells are traversed. State persists across
advance() calls, including the RNG and delayed-spike queue. No edges are pruned.

MIT License -- Copyright (c) 2026 Jayden Lin
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise ImportError("Live neural simulation requires numba; install the live extra.") from exc


@dataclass(frozen=True)
class LIFParams:
    dt_ms: float = 0.1
    rest_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    poisson_scale: float = 250.0
    rest_epsilon_mv: float = 1e-6

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("All LIF parameters must be finite")
        if min(self.dt_ms, self.membrane_tau_ms, self.synapse_tau_ms) <= 0:
            raise ValueError("Time step and time constants must be positive")
        if min(self.refractory_ms, self.delay_ms, self.w_syn_mv,
               self.poisson_scale, self.rest_epsilon_mv) < 0:
            raise ValueError("Delays, gains and epsilon cannot be negative")
        if self.threshold_mv <= max(self.reset_mv, self.rest_mv):
            raise ValueError("Threshold must exceed resting and reset voltage")
        for value in (self.refractory_ms, self.delay_ms):
            if not math.isclose(value / self.dt_ms, round(value / self.dt_ms), abs_tol=1e-8):
                raise ValueError("Delay and refractory period must be multiples of dt_ms")

    def propagator(self):
        tm, ts, dt = self.membrane_tau_ms, self.synapse_tau_ms, self.dt_ms
        a, c = math.exp(-dt / tm), math.exp(-dt / ts)
        b = dt / tm * a if abs(ts - tm) < 1e-12 else ts / (ts - tm) * (c - a)
        return a, b, c


@dataclass(frozen=True)
class SpikeResult:
    counts: np.ndarray  # per-neuron spikes during this call (not cumulative)
    duration_ms: float
    spike_times_ms: np.ndarray  # absolute simulation time; optional trace
    spike_indices: np.ndarray
    trace_truncated: bool = False

    def rate_hz(self, indices) -> float:
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices) or self.duration_ms <= 0:
            return 0.0
        return float(self.counts[indices].mean() * 1000.0 / self.duration_ms)


@njit(cache=True, fastmath=False)
def _advance(n_steps, step0, state, ptr, idx, weights, ring, ring_count,
             delay, pois_idx, pois_p, pois_w, a, b, c, rest, threshold, reset,
             eps, active, is_active, n_active, silenced, counts, rng,
             trace_steps, trace_indices):
    n_trace, total_spikes = 0, 0
    n_slots = len(ring_count)
    for s in range(n_steps):
        t = step0 + s
        slot = t % n_slots
        count = 0
        k = n_active - 1
        while k >= 0:
            i = active[k]
            if silenced[i]:
                state[i, 0], state[i, 1] = rest, 0.0
                is_active[i] = 0
                n_active -= 1
                active[k] = active[n_active]
            elif t >= state[i, 2]:
                g = state[i, 1]
                v = rest + a * (state[i, 0] - rest) + b * g
                g *= c
                state[i, 0], state[i, 1] = v, g
                if v > threshold:
                    ring[slot, count] = i
                    count += 1
                    state[i, 2] = t + max(state[i, 3], 1.0)
                elif abs(g) < eps and abs(v - rest) < eps:
                    state[i, 0], state[i, 1] = rest, 0.0
                    is_active[i] = 0
                    n_active -= 1
                    active[k] = active[n_active]
            k -= 1
        ring_count[slot] = count
        # Brian2 schedule: integration/threshold -> delivery -> reset.
        delivery_slot = (t - delay) % n_slots
        for k in range(ring_count[delivery_slot]):
            pre = ring[delivery_slot, k]
            if silenced[pre]:
                continue
            for edge in range(ptr[pre], ptr[pre + 1]):
                post = idx[edge]
                if silenced[post] or t < state[post, 2]:
                    continue
                state[post, 1] += weights[edge]
                if not is_active[post]:
                    is_active[post] = 1
                    active[n_active] = post
                    n_active += 1
        for k in range(len(pois_idx)):
            event = rng.random() < pois_p[k]
            i = pois_idx[k]
            if event and not silenced[i] and t >= state[i, 2]:
                state[i, 0] += pois_w
                if not is_active[i]:
                    is_active[i] = 1
                    active[n_active] = i
                    n_active += 1
        for k in range(count):
            i = ring[slot, k]
            state[i, 0], state[i, 1] = reset, 0.0
            counts[i] += 1
            total_spikes += 1
            if n_trace < len(trace_steps):
                trace_steps[n_trace], trace_indices[n_trace] = t, i
                n_trace += 1
    return n_active, n_trace, total_spikes


def csr_from_edges(pre, post, signed_counts, n_neurons):
    """Pre-major CSR; duplicates are retained and have additive effects."""
    pre, post = np.asarray(pre), np.asarray(post)
    weights = np.asarray(signed_counts, dtype=np.float64)
    if not (pre.ndim == post.ndim == weights.ndim == 1 and len(pre) == len(post) == len(weights)):
        raise ValueError("Edge arrays must be one-dimensional and equally sized")
    if not isinstance(n_neurons, (int, np.integer)) or n_neurons < 1:
        raise ValueError("n_neurons must be a positive integer")
    for ids in (pre, post):
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= n_neurons):
            raise ValueError("Neuron indices must be valid integers")
    order = np.argsort(pre, kind="stable")
    ptr = np.zeros(n_neurons + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(pre.astype(np.int64), minlength=n_neurons))
    return ptr, post[order].astype(np.int32), weights[order]


class LiveLIFEngine:
    """Independent neural state over signed, pre-major CSR connectivity.

    Driven neurons use zero refractory time as in Shiu's Poisson-drive setup.
    Silencing clamps neurons to rest and blocks queued outgoing delivery while
    silenced. This is a declared intervention, not a model of pharmacology.
    """

    def __init__(self, indptr, indices, signed_synapse_counts, *, params=None, seed=0):
        self.params = params or LIFParams()
        ptr, idx = np.asarray(indptr), np.asarray(indices)
        w = np.asarray(signed_synapse_counts)
        if ptr.ndim != 1 or len(ptr) < 2 or not np.issubdtype(ptr.dtype, np.integer):
            raise ValueError("indptr must be an integer CSR pointer array")
        self.n = len(ptr) - 1
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("indices must be an integer array")
        if w.ndim != 1 or len(w) != len(idx) or not np.all(np.isfinite(w)):
            raise ValueError("Weights must match indices and be finite")
        if ptr[0] != 0 or ptr[-1] != len(idx) or np.any(np.diff(ptr) < 0):
            raise ValueError("Invalid CSR pointers")
        if np.any(idx < 0) or np.any(idx >= self.n):
            raise ValueError("CSR target outside neuron range")
        self.indptr = np.ascontiguousarray(ptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(idx, dtype=np.int32)
        self.weights = np.ascontiguousarray(w, dtype=np.float64) * self.params.w_syn_mv
        self._seed = int(seed)
        self._allocate_state()

    def _allocate_state(self):
        p = self.params
        self._state = np.zeros((self.n, 4), dtype=np.float64)
        self.v, self.g = self._state[:, 0], self._state[:, 1]
        self._ref = round(p.refractory_ms / p.dt_ms)
        self._delay = round(p.delay_ms / p.dt_ms)
        self._ring = np.zeros((self._delay + 1, self.n), dtype=np.int32)
        self._ring_count = np.zeros(self._delay + 1, dtype=np.int64)
        self._active = np.empty(self.n, dtype=np.int32)
        self._is_active = np.zeros(self.n, dtype=np.uint8)
        self.counts = np.zeros(self.n, dtype=np.int64)
        self._silenced = np.zeros(self.n, dtype=np.bool_)
        self._pois_idx = np.empty(0, dtype=np.int32)
        self._pois_p = np.empty(0, dtype=np.float64)
        self.reset()

    @property
    def time_ms(self):
        return self.step_count * self.params.dt_ms

    def _indices(self, indices):
        idx = np.asarray(indices)
        if idx.size == 0:
            return np.empty(0, dtype=np.int32)
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or np.any(idx >= self.n):
            raise ValueError("Indices must be one-dimensional valid neuron integers")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("Duplicate neuron indices are not allowed")
        return np.ascontiguousarray(idx, dtype=np.int32)

    def set_poisson(self, indices, rates_hz):
        idx = self._indices(indices)
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=float), idx.shape)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0) or np.any(rates * self.params.dt_ms > 1000):
            raise ValueError("Rates must be finite and between 0 and 1000/dt_ms Hz")
        self._state[self._pois_idx, 3] = self._ref
        keep = rates > 0
        self._pois_idx = idx[keep]
        self._pois_p = np.ascontiguousarray(rates[keep] * self.params.dt_ms / 1000)
        self._state[self._pois_idx, 3] = 0

    def set_silenced(self, indices):
        idx = self._indices(indices)
        self._silenced[:] = False
        self._silenced[idx] = True
        self.v[idx], self.g[idx] = self.params.rest_mv, 0.0

    def kick(self, indices, delta_mv):
        """Explicit instantaneous voltage injection, for controlled experiments."""
        idx = self._indices(indices)
        if not np.isfinite(delta_mv):
            raise ValueError("delta_mv must be finite")
        idx = idx[~self._silenced[idx] & (self.step_count >= self._state[idx, 2])]
        self.v[idx] += delta_mv
        fresh = idx[self._is_active[idx] == 0]
        self._active[self.n_active:self.n_active + len(fresh)] = fresh
        self._is_active[fresh] = 1
        self.n_active += len(fresh)

    def reset(self, seed=None):
        """Reset neural state and RNG, retaining stimulation and lesion settings."""
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.v[:] = self.params.rest_mv
        self.g[:] = 0
        self._state[:, 2] = 0
        self._state[:, 3] = self._ref
        self._state[self._pois_idx, 3] = 0
        self._ring_count[:] = 0
        self._is_active[:] = 0
        self.counts[:] = 0
        self.n_active, self.step_count = 0, 0

    def fork(self, seed=0):
        """New independent state, sharing only immutable graph arrays/parameters."""
        other = object.__new__(type(self))
        other.params, other.n = self.params, self.n
        other.indptr, other.indices, other.weights = self.indptr, self.indices, self.weights
        other._seed = int(seed)
        other._allocate_state()
        return other

    def advance(self, duration_ms, *, record_spikes=False, trace_capacity=100_000):
        if not np.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("duration_ms must be finite and nonnegative")
        n_steps = round(duration_ms / self.params.dt_ms)
        if not math.isclose(n_steps * self.params.dt_ms, duration_ms, abs_tol=1e-8):
            raise ValueError("duration_ms must be an integer multiple of dt_ms")
        if not isinstance(trace_capacity, int) or trace_capacity < 0:
            raise ValueError("trace_capacity must be a nonnegative integer")
        before = self.counts.copy()
        capacity = trace_capacity if record_spikes else 0
        steps = np.empty(capacity, dtype=np.int64)
        ids = np.empty(capacity, dtype=np.int32)
        p = self.params
        self.n_active, n_trace, total = _advance(
            n_steps, self.step_count, self._state, self.indptr, self.indices, self.weights,
            self._ring, self._ring_count, self._delay, self._pois_idx, self._pois_p,
            p.w_syn_mv * p.poisson_scale, *p.propagator(), p.rest_mv, p.threshold_mv,
            p.reset_mv, p.rest_epsilon_mv, self._active, self._is_active, self.n_active,
            self._silenced, self.counts, self._rng, steps, ids)
        self.step_count += n_steps
        return SpikeResult(self.counts - before, n_steps * p.dt_ms,
                           steps[:n_trace] * p.dt_ms, ids[:n_trace].copy(),
                           bool(record_spikes and total > capacity))
