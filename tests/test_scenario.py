"""Scenario loader tests."""

import tempfile
from pathlib import Path

from flybrainflow.scenario import Scenario, ScenarioError
from flybrainflow.world import load_map_for_scenario

ROOT = Path(__file__).resolve().parents[1]


def test_example_scenarios_load():
    sc = Scenario.from_toml(ROOT / "scenarios" / "corridor_bidirectional.toml")
    assert sc.meta.name == "corridor_bidirectional"
    assert sc.meta.boundary_mode == "open" and sc.meta.scale_mode == "pedestrian"
    assert len(sc.targets) == 2 and sc.targets[0].slots == 6
    assert abs(sc.max_exit_rate_per_s - 6.0) < 1e-9  # 2 targets x 6 slots / 2 s
    assert sc.spawn.sources == ((1.0, 2.5), (39.0, 2.5))
    # Each source is pinned to the FAR target (see the scenario file's own comment on this --
    # real bug, fixed: nearest-distance assignment alone always picks a stream's own end).
    assert sc.spawn.source_targets == (0, 1)

    ring = Scenario.from_toml(ROOT / "scenarios" / "ring_road.toml")
    assert ring.meta.boundary_mode == "closed" and ring.meta.scale_mode == "car"
    assert ring.targets == () and ring.spawn.population_cap == 22


def test_scenario_builds_its_map():
    sc = Scenario.from_toml(ROOT / "scenarios" / "corridor_bidirectional.toml")
    m = load_map_for_scenario(sc)
    assert m.name == "corridor"
    assert m.is_walkable([list(s) for s in sc.spawn.sources]).all()
    assert m.is_walkable([list(t.position) for t in sc.targets]).all()

    ring = Scenario.from_toml(ROOT / "scenarios" / "ring_road.toml")
    rm = load_map_for_scenario(ring)
    assert rm.name == "ring" and abs(rm.meta["circumference"] - 230) < 0.1


def _write(toml: str) -> Path:
    d = tempfile.mkdtemp()
    p = Path(d) / "s.toml"
    p.write_text(toml)
    return p


MINIMAL = """
[scenario]
name = "t"
[map]
source = "gen:corridor"
[spawn]
sources = [[1, 1]]
[[targets]]
position = [5, 1]
"""


def test_minimal_scenario_gets_defaults():
    sc = Scenario.from_toml(_write(MINIMAL))
    assert sc.meta.duration_s == 300.0 and sc.map.walkable_grid_m == 0.1
    assert sc.targets[0].kind == "sugar" and sc.targets[0].feeding_time_s == 2.0
    assert sc.agents.max_speed_mps == 1.3 and sc.baseline.enabled


def test_validation_names_the_bad_key():
    cases = {
        MINIMAL.replace('name = "t"', 'name = "t"\nscale_mode = "boat"'): "scale_mode",
        MINIMAL.replace("sources = [[1, 1]]", "sources = []"): "sources",
        MINIMAL.replace("position = [5, 1]", "position = [5, 1]\nslots = 0"): "slots",
        MINIMAL.replace("[map]\nsource = \"gen:corridor\"", "[map]"): "source",
        MINIMAL + "\n[agents]\nfield_of_view_deg = 400\n": "field_of_view_deg",
    }
    for toml, key in cases.items():
        try:
            Scenario.from_toml(_write(toml))
        except ScenarioError as e:
            assert key in str(e), (key, str(e))
            continue
        raise AssertionError(f"expected a ScenarioError mentioning {key}")


DICT_SOURCE = """
[scenario]
name = "t"
[map]
source = "gen:corridor"
[spawn]
sources = [{ position = [1, 1], target = 1 }, [9, 1]]
[[targets]]
position = [5, 1]
[[targets]]
position = [9, 1]
"""


def test_dict_form_spawn_source_pins_a_preferred_target():
    # `{position=..., target=i}` is the fix for nearest-distance assignment silently breaking a
    # source co-located with the "wrong" target -- see scenarios/corridor_bidirectional.toml's own
    # comment and SpawnConfig.source_targets's docstring. A plain [x, y] source (the second entry
    # here) must still work exactly as before: no preference, i.e. None.
    sc = Scenario.from_toml(_write(DICT_SOURCE))
    assert sc.spawn.sources == ((1.0, 1.0), (9.0, 1.0))
    assert sc.spawn.source_targets == (1, None)


def test_out_of_range_target_index_is_a_named_scenario_error():
    bad = DICT_SOURCE.replace("target = 1", "target = 5")  # only targets 0 and 1 exist
    try:
        Scenario.from_toml(_write(bad))
    except ScenarioError as e:
        assert "target" in str(e)
        return
    raise AssertionError("expected a ScenarioError for an out-of-range spawn source target index")


def test_closed_mode_needs_no_sources_or_targets():
    toml = """
[scenario]
name = "ring"
boundary_mode = "closed"
scale_mode = "car"
[map]
source = "gen:ring"
[spawn]
population_cap = 22
"""
    sc = Scenario.from_toml(_write(toml))
    assert sc.spawn.sources == () and sc.targets == ()
