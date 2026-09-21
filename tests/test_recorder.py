"""Recorder tests: what gets captured each tick, what round-trips through disk, and the actual
point of recording anything -- that a brain-cohort fly's exact sensory input for a tick is there
to recompute its brain from later, per the plan's "replay debugger for agents" design.
"""

import numpy as np

from flybrainflow.recorder import Recorder
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim


def _scenario(**overrides):
    raw = {
        "scenario": {"name": overrides.pop("name", "t"), "boundary_mode": "open", "seed": overrides.pop("seed", 0)},
        "map": {"source": overrides.pop("map", "gen:corridor?length=15&width=5")},
        "spawn": {
            "sources": overrides.pop("sources", [[1.0, 2.5]]),
            "rate_per_s": overrides.pop("rate_per_s", 2.0),
            "population_cap": overrides.pop("population_cap", 4),
        },
        "targets": overrides.pop(
            "targets", [{"kind": "sugar", "position": [13.0, 2.5], "slots": 2, "feeding_time_s": 1.0}]
        ),
        "baseline": overrides.pop("baseline", {"enabled": True}),
    }
    return Scenario.from_dict(raw)


def test_capturing_nothing_yields_correctly_shaped_empty_arrays():
    sc = _scenario()
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    rec = Recorder(sim)
    arrs = rec.to_arrays()
    assert len(arrs["t"]) == 0
    assert arrs["sense"].shape == (0, 6)
    assert arrs["seed"].item() == sc.meta.seed


def test_one_row_per_alive_agent_per_tick_across_both_cohorts():
    sc = _scenario(population_cap=3, rate_per_s=100.0)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    rec = Recorder(sim)
    sim.tick(1.0)  # spawns up to population_cap in each cohort
    rec.capture()
    arrs = rec.to_arrays()
    n_alive = len(sim.cohorts["brain"].agents) + len(sim.cohorts["baseline"].agents)
    assert len(arrs["t"]) == n_alive
    assert set(arrs["brain"]) == {"brain", "baseline"}


def test_brain_cohort_rows_carry_a_real_sense_vector_baseline_rows_dont():
    sc = _scenario(population_cap=2, rate_per_s=100.0)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    rec = Recorder(sim)
    sim.tick(1.0)
    rec.capture()
    arrs = rec.to_arrays()
    brain_rows = arrs["sense"][arrs["brain"] == "brain"]
    baseline_rows = arrs["sense"][arrs["brain"] == "baseline"]
    assert len(brain_rows) > 0 and len(baseline_rows) > 0
    assert np.isfinite(brain_rows).all()
    assert np.isnan(baseline_rows).all()


def test_save_and_load_round_trips_exactly(tmp_path):
    sc = _scenario(population_cap=3, rate_per_s=3.0)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(1))
    rec = Recorder(sim)
    for _ in range(80):
        sim.tick(0.1)
        rec.capture()
    before = rec.to_arrays()

    path = tmp_path / "run.npz"
    rec.save(path)
    after = Recorder.load(path)

    assert set(after) == set(before)
    for key in before:
        if before[key].dtype.kind in "SU":
            assert np.array_equal(before[key], after[key])
        else:
            assert np.allclose(before[key], after[key], equal_nan=True)


def test_a_flys_sense_vector_can_be_pulled_back_out_and_matches_what_the_brain_actually_used():
    # The actual point of recording anything, per the plan: recompute a brain later from just its
    # recorded sensory input, without re-running the rest of the crowd. This doesn't build the
    # full replay-and-recompute machinery (that's the inspector, an M3 item) -- it just proves the
    # recorded number really is the number the brain saw, not something reconstructed after the
    # fact that only looks right.
    sc = _scenario(population_cap=1, rate_per_s=100.0, baseline={"enabled": False})
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(2))
    rec = Recorder(sim)
    sim.tick(1.0)
    fly_id = next(iter(sim.cohorts["brain"].agents))
    live_sense = sim.brains["brain"].last_sense[fly_id].copy()
    rec.capture()
    arrs = rec.to_arrays()
    row = arrs["id"] == fly_id
    assert np.allclose(arrs["sense"][row][0], live_sense)


def test_capture_after_forget_records_nothing_for_a_departed_agent():
    sc = _scenario(
        population_cap=1,
        rate_per_s=100.0,
        map="gen:corridor?length=10&width=5",
        sources=[[1.0, 2.5]],
        targets=[{"kind": "sugar", "position": [9.0, 2.5], "slots": 1, "feeding_time_s": 0.5}],
        baseline={"enabled": False},
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    rec = Recorder(sim)
    fly_id = None
    for _ in range(600):
        sim.tick(0.1)
        rec.capture()
        if fly_id is None and sim.cohorts["brain"].agents:
            fly_id = next(iter(sim.cohorts["brain"].agents))
        if fly_id is not None and fly_id not in sim.cohorts["brain"].agents:
            break
    assert fly_id is not None, "the fly never spawned"
    arrs = rec.to_arrays()
    rows_for_it = arrs["t"][arrs["id"] == fly_id]
    assert len(rows_for_it) > 0  # it was recorded while alive...
    assert fly_id not in sim.cohorts["brain"].agents  # ...and is gone from the live population
    assert fly_id not in sim.brains["brain"].last_sense  # ...and its brain state was released too
