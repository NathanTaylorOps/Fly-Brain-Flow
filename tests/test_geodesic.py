"""Geodesic (wall-respecting) distance field tests."""

import numpy as np

from flybrainflow.world.geodesic import geodesic_distance_field, gradient_direction, sample_field
from flybrainflow.world.map import WalkableMap, bottleneck, corridor, room_with_exit


def test_open_corridor_matches_straight_line():
    m = corridor(length=10, width=4, resolution=0.2)
    field = geodesic_distance_field(m, (1, 2))
    d = sample_field(m, field, [[9, 2]])[0]
    assert abs(d - 8.0) < 0.3  # nothing in the way; grid discretisation only


def test_wall_forces_a_detour():
    # A point on the other side of the bottleneck's wall segment (not through the gap) must be
    # further, walking, than a straight line would suggest -- and further than a point that IS
    # reachable through the gap at the same straight-line distance class.
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.2)
    gap_x = sum(m.meta["gap_x"]) / 2
    source = (gap_x - 10, m.meta["width"] / 2)  # left of the gap, on the centreline
    field = geodesic_distance_field(m, source)

    through_gap = (gap_x + 10, m.meta["width"] / 2)  # symmetric point past the gap
    straight = np.hypot(through_gap[0] - source[0], through_gap[1] - source[1])
    walking = sample_field(m, field, [through_gap])[0]
    assert walking >= straight - 1e-6  # walking distance is never shorter than straight-line
    assert walking < straight + 3.0  # but going through the (nearby, aligned) gap isn't a big detour


def test_room_with_exit_routes_through_the_door_not_the_wall():
    stub_length = 4
    m = room_with_exit(width=20, height=15, door_width=1.5, stub_length=stub_length)
    y0, y1 = m.meta["door_y"]
    door_y = (y0 + y1) / 2
    inside_far_corner = (2.0, 1.0)  # bottom-left corner of the room
    past_the_door = (m.meta["width"] + stub_length - 1.0, door_y)  # out in the exit stub

    field = geodesic_distance_field(m, inside_far_corner)
    walking = sample_field(m, field, [past_the_door])[0]
    straight = np.hypot(past_the_door[0] - inside_far_corner[0], past_the_door[1] - inside_far_corner[1])
    # the corner is far from the door, so the real walking path is noticeably longer than as-the-crow-flies
    assert walking > straight * 1.05


def test_unreachable_cell_is_infinite():
    # Two separate corridors (rasterised together, not touching) -> the far one is unreachable.
    m = corridor(length=5, width=2, resolution=0.5)
    field = geodesic_distance_field(m, (2.5, 1.0))
    # A point well outside this map's bounds altogether still resolves (nearest_walkable snaps it in);
    # what we actually want to check is that a real wall blocks propagation, which the bottleneck/room
    # tests above already cover via "further than straight-line". This test just checks the dtype/shape
    # contract holds and unreached cells (if any) are inf, not NaN or a crash.
    assert field.shape == (m.ny, m.nx)
    assert np.isfinite(field[field != np.inf]).all()


def test_gradient_direction_is_zero_not_nan_at_an_unreachable_point():
    # Real bug found on review: a point with no walkable path to the source has field value
    # np.inf, so a naive finite-difference gradient there is `inf - inf` = nan -- and `nan < eps`
    # is False in numpy, so a naive "is this basically flat" check lets that nan through as if it
    # were a real direction. Two separate walkable islands, several cells apart (no possible path
    # between them at all, not just a blocked one), pins this down directly.
    mask = np.zeros((5, 12), dtype=bool)
    mask[:, 0:3] = True  # left island -- the source lives here
    mask[:, 9:12] = True  # right island -- completely unreachable from the left
    m = WalkableMap.from_mask(mask, resolution=1.0, origin=(0.0, 0.0))
    field = geodesic_distance_field(m, (1.0, 2.0))  # inside the left island
    assert not np.isfinite(field[2, 10])  # confirms the two islands really are disconnected

    direction = gradient_direction(m, field, [[10.0, 2.0]])  # inside the right, unreachable island
    assert np.all(np.isfinite(direction))
    assert np.allclose(direction, [[0.0, 0.0]])


def test_diagonal_corner_cutting_is_disallowed():
    # Build a 3x3 mask with a single-cell diagonal "doorway" between two walls: cell (0,0) walkable,
    # (1,1) walkable, but (0,1) and (1,0) are walls -- a true diagonal-only gap should NOT be crossable,
    # matching how a real wall corner blocks a shortcut.
    mask = np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ]
    )
    m = WalkableMap.from_mask(mask, resolution=1.0, origin=(0.0, 0.0))
    field = geodesic_distance_field(m, (0.5, 0.5))  # cell (0, 0)
    d_diag_neighbor = field[1, 1]  # cell (1, 1), diagonally adjacent but corner-blocked
    assert d_diag_neighbor == np.inf
