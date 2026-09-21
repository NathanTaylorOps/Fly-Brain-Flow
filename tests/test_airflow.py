"""Potential-flow airflow field tests."""

import numpy as np

from flybrainflow.world.airflow import AirflowField
from flybrainflow.world.map import bottleneck, corridor


def test_zero_speed_gives_zero_velocity():
    m = corridor(length=10, width=4, resolution=0.5, open_ends=True)
    field = AirflowField.build(m)
    v = field.velocity_at([[5, 2]], direction_deg=0, speed_mps=0)
    assert np.allclose(v, 0.0)


def test_sealed_box_has_no_airflow():
    # No open_ends: every side is walled (padding included). A real sealed room has no wind,
    # regardless of what the dial outside says -- this is the correct physical answer, not a bug.
    # Resolution needs to be fine enough that the default 0.5 m padding is more than one cell
    # deep -- otherwise the wall and the map's outer edge are the same cell, which is a genuinely
    # separate (and separately tested) ambiguity, not what this test is checking.
    m = corridor(length=10, width=4, resolution=0.1, open_ends=False)
    field = AirflowField.build(m)
    v = field.velocity_at([[5, 2]], direction_deg=0, speed_mps=3.0)[0]
    assert np.allclose(v, 0.0, atol=1e-6)


def test_thin_padding_without_declared_openings_is_a_known_edge_case():
    # When a map's padding is only one cell deep and it hasn't declared explicit air_open cells,
    # the border-is-open fallback can't tell "the intended doorway" from "an ordinary side wall
    # that just happens to be thin" -- both touch the raster's outer edge. Documenting the actual
    # (imperfect) behaviour here so it's a known, tested limitation, not a silent surprise.
    m = corridor(length=10, width=4, resolution=0.5, open_ends=False)
    field = AirflowField.build(m)
    v = field.velocity_at([[5, 2]], direction_deg=0, speed_mps=3.0)[0]
    assert v[0] > 1.0  # wrongly porous on every side, not zero -- the known limitation


def test_open_ended_corridor_flow_matches_the_dial_away_from_side_walls():
    # On the centreline of a wide-enough, open-ended corridor, flow should point roughly the way
    # the wind dial says, at roughly the dial's speed -- nothing nearby to deflect it much.
    m = corridor(length=20, width=10, resolution=0.5, open_ends=True)
    field = AirflowField.build(m)
    v = field.velocity_at([[10, 5]], direction_deg=0, speed_mps=2.0)[0]
    assert v[0] > 1.0
    assert abs(v[1]) < v[0]


def test_direction_and_speed_are_free_after_build():
    m = corridor(length=10, width=4, resolution=0.5, open_ends=True)
    field = AirflowField.build(m)
    # Two very different dial settings off the same built field -- both must work and differ.
    v0 = field.velocity_at([[5, 2]], direction_deg=0, speed_mps=1.0)[0]
    v90 = field.velocity_at([[5, 2]], direction_deg=90, speed_mps=1.0)[0]
    assert not np.allclose(v0, v90)
    v_fast = field.velocity_at([[5, 2]], direction_deg=0, speed_mps=5.0)[0]
    assert np.allclose(v_fast, v0 * 5.0, atol=1e-6)  # linear in speed, exactly


def test_flow_speeds_up_through_the_gap():
    # Classic Venturi check: the gap of an open-ended bottleneck should carry faster flow (in the
    # wind direction) than the wide corridor upstream of it, for wind blowing straight through.
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.25, open_ends=True)
    field = AirflowField.build(m)
    gap_x = sum(m.meta["gap_x"]) / 2
    centre_y = m.meta["width"] / 2
    v_upstream = field.velocity_at([[gap_x - 15, centre_y]], direction_deg=0, speed_mps=1.0)[0]
    v_gap = field.velocity_at([[gap_x, centre_y]], direction_deg=0, speed_mps=1.0)[0]
    assert v_gap[0] > v_upstream[0]


def test_flow_deflects_around_a_wall_segment():
    # Right next to a bottleneck's wall segment (not in the gap), flow can't go straight through;
    # it must pick up a sideways (y) component to get around, unlike far upstream in the open
    # corridor where flow runs straight and true (negligible y component).
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.25, open_ends=True)
    field = AirflowField.build(m)
    gap_x = sum(m.meta["gap_x"]) / 2
    near_wall_point = [[gap_x - 0.5, 1.0]]  # tight against the lower wall segment, off centreline
    far_upstream_point = [[gap_x - 15, 1.0]]  # same y, well upstream, open corridor
    v_near = field.velocity_at(near_wall_point, direction_deg=0, speed_mps=1.0)[0]
    v_far = field.velocity_at(far_upstream_point, direction_deg=0, speed_mps=1.0)[0]
    assert abs(v_near[1]) > abs(v_far[1])


def test_sealed_pocket_does_not_crash_and_settles_near_zero():
    # A walkable cell with no route at all to open air (boxed in on every side) has no physical
    # airflow; the regularisation should keep the solve from blowing up or crashing.
    from flybrainflow.world.map import WalkableMap

    big = np.zeros((20, 20), bool)
    big[5:15, 5:15] = True  # sealed room, nothing connects it to the map border
    m = WalkableMap.from_mask(big, resolution=0.5)
    field = AirflowField.build(m)
    v = field.velocity_at([[5.0, 5.0]], direction_deg=0, speed_mps=3.0)[0]
    assert np.all(np.isfinite(v))
    assert np.linalg.norm(v) < 0.5  # sealed room: should settle near zero, not blow up
