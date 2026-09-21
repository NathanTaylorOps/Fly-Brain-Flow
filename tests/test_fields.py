"""Odour/wind field tests."""

import numpy as np

from flybrainflow.scenario import Scenario
from flybrainflow.world import AirflowField, OdorField, OdorSource, Wind, load_map_for_scenario, odor_field_for_scenario
from flybrainflow.world.map import bottleneck, room_with_exit


def test_concentration_decays_with_distance():
    field = OdorField(Wind(), [OdorSource(kind="sugar", position=(0, 0), strength=1.0, range_m=10)])
    near, mid, far = field.sample([[1, 0], [5, 0], [100, 0]], kind="sugar")
    assert near > mid > far > 0
    assert far < 1e-3


def test_concentration_at_source_equals_strength():
    field = OdorField(Wind(), [OdorSource(kind="sugar", position=(3, 4), strength=2.0, range_m=10)])
    c = field.sample([[3, 4]], kind="sugar")
    assert abs(c[0] - 2.0) < 1e-9


def test_zero_wind_is_isotropic():
    field = OdorField(Wind(direction_deg=0, speed_mps=0), [OdorSource(kind="sugar", position=(0, 0), range_m=10)])
    east, west, north, south = field.sample([[5, 0], [-5, 0], [0, 5], [0, -5]])
    assert abs(east - west) < 1e-9
    assert abs(east - north) < 1e-9
    assert abs(north - south) < 1e-9


def test_downwind_carries_further_than_upwind():
    # Wind blows toward +x, so a point east of the source is downwind and should read stronger
    # than an equally-distant point west of it (upwind).
    field = OdorField(Wind(direction_deg=0, speed_mps=2.0), [OdorSource(kind="sugar", position=(0, 0), range_m=10)])
    downwind, upwind = field.sample([[8, 0], [-8, 0]])
    assert downwind > upwind


def test_kind_channels_are_independent():
    field = OdorField(
        Wind(),
        [
            OdorSource(kind="sugar", position=(0, 0), strength=1.0, range_m=10),
            OdorSource(kind="mate", position=(0, 0), strength=5.0, range_m=10),
        ],
    )
    by_kind = field.sample_by_kind([[2, 0]])
    assert set(by_kind) == {"sugar", "mate"}
    assert by_kind["mate"][0] > by_kind["sugar"][0]  # stronger source, same distance
    assert abs(field.sample([[2, 0]], kind="sugar")[0] - by_kind["sugar"][0]) < 1e-9


def test_gradient_points_toward_source():
    field = OdorField(Wind(), [OdorSource(kind="sugar", position=(10, 10), range_m=20)])
    g = field.gradient([[0, 10]], kind="sugar")[0]
    # standing due west of the source, the gradient should point mostly east (+x)
    assert g[0] > 0
    assert abs(g[1]) < g[0]


def test_moving_a_source_is_picked_up_immediately():
    src = OdorSource(kind="sugar", position=(0, 0), range_m=10)
    field = OdorField(Wind(), [src])
    before = field.sample([[5, 0]], kind="sugar")[0]
    src.move_to((5, 0))
    after = field.sample([[5, 0]], kind="sugar")[0]
    assert after > before


def test_wind_vector_matches_bearing_convention():
    w = Wind(direction_deg=90, speed_mps=2.0)
    assert abs(w.vector[0]) < 1e-9  # 0 = +x, 90 = +y, so due-90 has no x component
    assert abs(w.vector[1] - 2.0) < 1e-9


def test_wind_set_rejects_negative_speed():
    w = Wind()
    try:
        w.set(speed_mps=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_walkable_map_makes_odor_route_around_a_wall():
    # A target just past a bottleneck wall segment should smell much weaker to a fly standing
    # directly on the far side of that wall (blocked) than to one an equal straight-line distance
    # away but out on the open corridor centreline (clear line to the gap).
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.2)
    gap_x = sum(m.meta["gap_x"]) / 2
    target_pos = (gap_x + 6, m.meta["width"] / 2)  # past the gap, on the centreline

    open_field = OdorField(Wind(), [OdorSource(kind="sugar", position=target_pos, range_m=20)])
    walled_field = OdorField(Wind(), [OdorSource(kind="sugar", position=target_pos, range_m=20)], walkable_map=m)

    blocked_point = [[gap_x - 6, 0.5]]  # behind the lower wall segment, same-ish straight-line distance
    c_open = open_field.sample(blocked_point, kind="sugar")[0]
    c_walled = walled_field.sample(blocked_point, kind="sugar")[0]
    assert c_walled < c_open  # wall-aware version correctly discounts the point the wall blocks


def test_occluded_field_matches_open_field_with_clear_line_of_sight():
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.2)
    src = (2.0, m.meta["width"] / 2)
    query = [[6.0, m.meta["width"] / 2]]  # same side of the map, nothing in between
    open_c = OdorField(Wind(), [OdorSource(kind="sugar", position=src, range_m=20)]).sample(query, "sugar")[0]
    walled_c = OdorField(Wind(), [OdorSource(kind="sugar", position=src, range_m=20)], walkable_map=m).sample(
        query, "sugar"
    )[0]
    assert abs(open_c - walled_c) < 0.05  # clear line of sight -> walking distance ~= straight-line


def test_moving_source_invalidates_the_geodesic_cache():
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.2)
    src = OdorSource(kind="sugar", position=(2.0, m.meta["width"] / 2), range_m=20)
    field = OdorField(Wind(), [src], walkable_map=m)
    query = [[6.0, m.meta["width"] / 2]]
    before = field.sample(query, "sugar")[0]
    src.move_to((5.9, m.meta["width"] / 2))  # right next to the query point now
    after = field.sample(query, "sugar")[0]
    assert after > before


def test_gradient_still_works_in_occluded_mode():
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.2)
    gap_x = sum(m.meta["gap_x"]) / 2
    field = OdorField(
        Wind(), [OdorSource(kind="sugar", position=(gap_x + 5, m.meta["width"] / 2), range_m=20)], walkable_map=m
    )
    g = field.gradient([[gap_x - 5, m.meta["width"] / 2]], kind="sugar")[0]
    assert g[0] > 0  # target is further along +x; gradient should point that way, not straight up/down
    assert np.linalg.norm(g) > 0


def test_odor_field_for_scenario_accepts_a_map_for_occlusion():
    import tempfile
    from pathlib import Path

    toml = """
[scenario]
name = "t"
[map]
source = "gen:bottleneck"
[spawn]
sources = [[2, 4]]
[[targets]]
position = [38, 4]
"""
    d = tempfile.mkdtemp()
    p = Path(d) / "s.toml"
    p.write_text(toml)
    sc = Scenario.from_toml(p)
    m = load_map_for_scenario(sc)
    field = odor_field_for_scenario(sc, walkable_map=m)
    assert field.walkable_map is m


def test_odor_uses_the_bent_local_wind_when_an_airflow_field_is_given():
    # Right next to a bottleneck wall segment, the real (bent) wind has a strong sideways
    # component -- deflecting around the wall -- that a single global wind vector doesn't have.
    # That should measurably change the odour's downwind stretch versus the naive global-wind
    # version, at a point where the two models actually disagree about which way the air moves.
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.25, open_ends=True)
    gap_x = sum(m.meta["gap_x"]) / 2
    wind = Wind(direction_deg=0, speed_mps=2.0)
    src = OdorSource(kind="sugar", position=(gap_x + 3, 1.0), range_m=20)  # past the gap, low against the wall line
    query = [[gap_x - 0.5, 1.0]]  # tight against the wall segment upstream of the gap

    global_field = OdorField(wind, [OdorSource(**src.__dict__)], walkable_map=m)
    bent_field = OdorField(wind, [OdorSource(**src.__dict__)], walkable_map=m, airflow=AirflowField.build(m))

    c_global = global_field.sample(query, "sugar")[0]
    c_bent = bent_field.sample(query, "sugar")[0]
    assert abs(c_global - c_bent) > 1e-9  # the two wind models genuinely disagree here


def test_wind_alignment_uses_the_bent_path_not_a_straight_line():
    # Real gap found on review: the "is this point downwind or upwind" question used a
    # straight-line bearing from source to point even in occluded mode, which can point straight
    # through the very wall the scent has to detour around. A point deep in the exit stub, with
    # the source in the room's far corner well below the door, makes the two disagree clearly: a
    # straight line from the corner to the stub cuts through the wall beside the door, but the
    # real walking path bends through the doorway first, then straight down the stub.
    stub_length = 6.0
    m = room_with_exit(width=20, height=15, door_width=1.5, stub_length=stub_length)
    y0, y1 = m.meta["door_y"]
    door_y_mid = (y0 + y1) / 2
    src_pos = (2.0, 1.0)  # far corner, well below the door
    query = np.array([[m.meta["width"] + stub_length - 1.0, door_y_mid]])  # deep in the stub

    field = OdorField(Wind(), [OdorSource(kind="sugar", position=src_pos, range_m=30)], walkable_map=m)
    geo_dir = field._geodesic_direction(query, field.sources[0])[0]

    straight = query[0] - np.array(src_pos)
    straight_unit = straight / np.linalg.norm(straight)

    assert geo_dir[0] > 0.9  # deep in the straight stub, walking away from the source means +x
    assert abs(geo_dir[1] - straight_unit[1]) > 0.1  # meaningfully different from the naive bearing


def test_source_param_isolates_one_plume_from_a_same_kind_sum():
    # Real bug, fixed: two same-kind sources used to always sum into one combined plume for
    # `kind=`, so a fly that had committed to the far one still felt (and was pulled off course
    # by) the near one's smell purely because they're the same kind -- see `gradient`'s own
    # docstring and `ToyBrain._sense` for the full story (this is the root cause behind
    # scenarios/corridor_bidirectional.toml's "spawning right on the sugar" symptom persisting
    # even after target *assignment* was fixed). `source` must isolate exactly one plume.
    near = OdorSource(kind="sugar", position=(0, 0), strength=1.0, range_m=30)
    far = OdorSource(kind="sugar", position=(38, 0), strength=1.0, range_m=30)
    field = OdorField(Wind(), [far, near])  # order shouldn't matter -- identity picks the source
    query = [[1.0, 0.0]]  # right next to `near`, far from `far`

    combined = field.sample(query, kind="sugar")[0]
    far_only = field.sample(query, source=far)[0]
    near_only = field.sample(query, source=near)[0]
    assert abs(combined - (far_only + near_only)) < 1e-9  # kind-wide sum is genuinely the two added
    assert far_only < near_only  # confirms `near` is the one dominating the combined reading
    assert far_only < 0.3  # this point is 38m from `far` with only a 30m range -- weak on its own
    assert near_only > 0.9  # ...compared to `near`, which this point is standing right next to

    # The gradient near `near` is completely dominated by `near` itself in the kind-wide sum, but
    # isolating `far` gives the gradient that actually, correctly, points toward `far` (+x) instead.
    g_far_only = field.gradient(query, source=far)[0]
    assert g_far_only[0] > 0  # +x, correctly toward the far source
    g_combined = field.gradient(query, kind="sugar")[0]
    assert g_combined[0] < 0  # dominated by `near`: pulls back toward x=0, the wrong direction


def test_from_scenario_builds_matching_sources():
    import tempfile
    from pathlib import Path

    toml = """
[scenario]
name = "t"
[map]
source = "gen:corridor"
[wind]
direction_deg = 45
speed_mps = 1.5
[spawn]
sources = [[1, 1]]
[[targets]]
kind = "sugar"
position = [5, 1]
odor_strength = 2.0
odor_range_m = 12
"""
    d = tempfile.mkdtemp()
    p = Path(d) / "s.toml"
    p.write_text(toml)
    sc = Scenario.from_toml(p)
    field = odor_field_for_scenario(sc)
    assert field.wind.direction_deg == 45 and field.wind.speed_mps == 1.5
    assert len(field.sources) == 1
    s = field.sources[0]
    assert s.kind == "sugar" and s.position == (5.0, 1.0) and s.strength == 2.0 and s.range_m == 12.0
