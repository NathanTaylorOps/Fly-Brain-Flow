"""viewer_export.py tests -- the Python-side half of the M0 viewer. What actually renders it
(`viewer/index.html`, a plain canvas playback) isn't something pytest can check; that's been
verified by hand with a headless browser (see docs/JOURNAL.md) rather than left unchecked. What's
tested here is the one real piece of logic this module has: reshaping a flat, columnar recording
into the per-tick, per-agent frames the viewer actually consumes, plus the JSON stays valid.
"""

import json

import numpy as np

from flybrainflow.recorder import Recorder
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim
from flybrainflow.viewer_export import export_for_viewer


def _recorded_run(**overrides):
    raw = {
        "scenario": {"name": "t", "boundary_mode": "open", "seed": overrides.pop("seed", 0)},
        "map": {"source": overrides.pop("map", "gen:corridor?length=15&width=5")},
        "spawn": {
            "sources": overrides.pop("sources", [[1.0, 2.5]]),
            "rate_per_s": overrides.pop("rate_per_s", 3.0),
            "population_cap": overrides.pop("population_cap", 4),
        },
        "targets": overrides.pop(
            "targets", [{"kind": "sugar", "position": [13.0, 2.5], "slots": 2, "feeding_time_s": 1.0}]
        ),
        "baseline": overrides.pop("baseline", {"enabled": True}),
    }
    sc = Scenario.from_dict(raw)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    rec = Recorder(sim)
    for _ in range(overrides.pop("ticks", 100)):
        sim.tick(0.1)
        rec.capture()
    return rec.to_arrays(), sim, sc


def test_exported_json_is_valid_and_matches_recorded_frame_count(tmp_path):
    recording, sim, sc = _recorded_run()
    path = tmp_path / "run.json"
    data = export_for_viewer(recording, sim.map, sc, path)

    reloaded = json.loads(path.read_text())
    assert reloaded == data  # round-trips cleanly through actual disk, not just in memory

    n_distinct_ticks = len(np.unique(recording["t"]))
    assert len(data["frames"]) == n_distinct_ticks


def test_every_row_in_the_recording_lands_in_exactly_one_frame():
    recording, sim, sc = _recorded_run(ticks=50)
    from flybrainflow.viewer_export import _build_payload

    payload = _build_payload(recording, sim.map, sc)
    total_agents_across_frames = sum(len(f["agents"]) for f in payload["frames"])
    assert total_agents_across_frames == len(recording["t"])


def test_frame_agents_carry_both_cohorts_and_the_right_fields():
    recording, sim, sc = _recorded_run(ticks=30, population_cap=3)
    from flybrainflow.viewer_export import _build_payload

    payload = _build_payload(recording, sim.map, sc)
    last_frame = payload["frames"][-1]
    tags = {a["brain"] for f in payload["frames"] for a in f["agents"]}
    assert tags <= {"brain", "baseline"}
    for a in last_frame["agents"]:
        assert set(a) == {"id", "brain", "x", "y", "status"}
        assert a["status"] in ("walking", "feeding")


def test_walls_and_targets_come_from_the_map_and_scenario_not_the_recording():
    recording, sim, sc = _recorded_run(map="gen:bottleneck?length=20&width=8&gap_width=1.2&gap_length=2")
    from flybrainflow.viewer_export import _build_payload

    payload = _build_payload(recording, sim.map, sc)
    assert len(payload["walls"]["outer"]) == len(sim.map.outer)
    assert len(payload["walls"]["holes"]) == len(sim.map.holes) == 2  # the bottleneck's two jaws
    assert payload["targets"] == [{"kind": t.kind, "position": [float(t.position[0]), float(t.position[1])]} for t in sc.targets]
    assert payload["bounds"] == list(sim.map.bounds)


def test_an_empty_recording_produces_no_frames_but_still_valid_geometry(tmp_path):
    recording, sim, sc = _recorded_run(ticks=0)
    assert len(recording["t"]) == 0
    path = tmp_path / "empty.json"
    data = export_for_viewer(recording, sim.map, sc, path)
    assert data["frames"] == []
    assert len(data["walls"]["outer"]) >= 1
    json.loads(path.read_text())  # still valid JSON
