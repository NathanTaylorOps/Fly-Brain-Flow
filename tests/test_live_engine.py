import numpy as np
import pytest

pytest.importorskip("numba")
from flybrainflow.spiking.live_engine import LIFParams, LiveLIFEngine, csr_from_edges


def make_engine(pre=(0,), post=(1,), w=(400,), n=2, **kwargs):
    return LiveLIFEngine(*csr_from_edges(np.array(pre, int), np.array(post, int), w, n), **kwargs)


def test_propagator_matches_matrix_exponential():
    from scipy.linalg import expm
    for ts in (5., 20.):
        p = LIFParams(synapse_tau_ms=ts)
        matrix = expm(np.array([[-1/20, 1/20], [0, -1/ts]]) * p.dt_ms)
        np.testing.assert_allclose(p.propagator(), [matrix[0, 0], matrix[0, 1], matrix[1, 1]], rtol=1e-12)


def test_quiet_graph_and_synaptic_delay():
    e = make_engine()
    assert e.advance(10).counts.sum() == 0
    e.kick([0], 60)
    e.advance(1.8)
    assert e.g[1] == 0
    e.advance(.1)
    assert e.g[1] == pytest.approx(110)
    result = e.advance(10)
    assert result.counts[1] >= 1


def test_inhibition_and_refractory_discard():
    inhibitory = make_engine(w=(-400,))
    inhibitory.kick([0], 60)
    result = inhibitory.advance(20)
    assert result.counts.tolist() == [1, 0]
    assert inhibitory.v[1] < -52
    refractory = make_engine()
    refractory.kick([0, 1], 60)
    assert refractory.advance(20).counts.tolist() == [1, 1]
    assert refractory.g[1] == 0


def test_chunking_and_reset_preserve_seeded_experiment():
    a, b = make_engine(seed=42), make_engine(seed=42)
    for engine in (a, b):
        engine.set_poisson([0], 150)
    one = a.advance(500, record_spikes=True)
    chunks = [b.advance(t, record_spikes=True) for t in (0.1, 9.9, 90, 400)]
    np.testing.assert_array_equal(one.counts, sum(x.counts for x in chunks))
    np.testing.assert_array_equal(one.spike_times_ms, np.concatenate([x.spike_times_ms for x in chunks]))
    np.testing.assert_array_equal(one.spike_indices, np.concatenate([x.spike_indices for x in chunks]))
    assert one.counts[1] > 0
    a.reset()
    repeated = a.advance(500, record_spikes=True)
    np.testing.assert_array_equal(one.spike_times_ms, repeated.spike_times_ms)
    np.testing.assert_array_equal(one.counts, repeated.counts)


def test_silencing_changes_readout_and_fork_state_is_independent():
    a = make_engine(seed=42)
    b = a.fork(seed=42)
    assert a.weights is b.weights
    assert not np.shares_memory(a.v, b.v)
    for engine in (a, b):
        engine.set_poisson([0], 150)
    b.set_silenced([1])
    assert a.advance(500).counts[1] > 0
    assert b.advance(500).counts[1] == 0
    assert b.v[1] == -52


def test_trace_overflow_never_discards_simulation_spikes():
    a = make_engine()
    a.set_poisson([0], 10000)
    result = a.advance(10, record_spikes=True, trace_capacity=2)
    assert len(result.spike_indices) == 2
    assert result.trace_truncated
    assert result.counts[0] > 2


def test_bad_data_and_inputs_are_rejected():
    with pytest.raises(ValueError):
        LiveLIFEngine([0, 2, 1], [1], [1])
    with pytest.raises(ValueError):
        LIFParams(dt_ms=0)
    engine = make_engine()
    for ids, rate in (([2], 1), ([0, 0], 1), ([0], -1), ([0], float("nan"))):
        with pytest.raises(ValueError):
            engine.set_poisson(ids, rate)
    with pytest.raises(ValueError):
        engine.advance(.05)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_matches_brian2_spike_for_spike():
    b2 = pytest.importorskip("brian2")
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = .1 * b2.ms
    rng = np.random.default_rng(11)
    n = 15
    pre, post, weights = [0, 0, 0], [0, 1, 2], [1000, 80, 60]
    for i in range(1, n):
        for j in range(1, n):
            if i != j and rng.random() < .3:
                pre.append(i); post.append(j); weights.append(int(rng.integers(-50, 150)))
    neurons = b2.NeuronGroup(n, '''dv/dt = (-52*mV-v+g)/(20*ms) : volt (unless refractory)
dg/dt = -g/(5*ms) : volt (unless refractory)''', threshold="v > -45*mV",
                            reset="v=-52*mV; g=0*mV", refractory=1*b2.ms, method="exact")
    neurons.v = -52*b2.mV
    synapses = b2.Synapses(neurons, neurons, "w : volt", on_pre="g_post += w", delay=1.8*b2.ms)
    synapses.connect(i=pre, j=post)
    synapses.w = np.array(weights)*.275*b2.mV
    monitor = b2.SpikeMonitor(neurons)
    neurons.v[0] = 0*b2.mV
    b2.Network(neurons, synapses, monitor).run(100*b2.ms)
    expected = sorted(zip(np.rint(np.asarray(monitor.t/b2.ms)*10).astype(int), np.asarray(monitor.i)))
    engine = make_engine(pre, post, weights, n, params=LIFParams(refractory_ms=1))
    engine.kick([0], 52)
    result = engine.advance(100, record_spikes=True)
    actual = sorted(zip(np.rint(result.spike_times_ms*10).astype(int), result.spike_indices))
    assert len(expected) > 50
    assert actual == expected
