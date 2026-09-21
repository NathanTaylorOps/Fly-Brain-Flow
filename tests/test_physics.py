"""Collision/movement physics tests."""

import numpy as np

from flybrainflow.world.map import bottleneck, corridor, room_with_exit
from flybrainflow.world.physics import max_overlap, step


def test_free_agent_moves_exactly_where_it_wants():
    m = corridor(length=20, width=5)
    pos = [[10.0, 2.5]]
    vel = [[1.0, 0.0]]
    new_pos, actual_vel = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m)
    assert np.allclose(new_pos, [[11.0, 2.5]], atol=1e-6)
    assert np.allclose(actual_vel, [[1.0, 0.0]], atol=1e-6)


def test_max_speed_is_enforced():
    m = corridor(length=20, width=5)
    pos = [[10.0, 2.5]]
    vel = [[10.0, 0.0]]  # way over the cap
    new_pos, actual_vel = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m, max_speed_mps=1.3)
    assert np.allclose(new_pos, [[11.3, 2.5]], atol=1e-6)
    assert np.allclose(actual_vel, [[1.3, 0.0]], atol=1e-6)


def test_overlapping_agents_get_pushed_apart():
    m = corridor(length=20, width=5)
    pos = [[10.0, 2.5], [10.05, 2.5]]  # 5 cm apart, radius 0.25 each -- badly overlapping
    vel = [[0.0, 0.0], [0.0, 0.0]]
    new_pos, _ = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m)
    dist = np.linalg.norm(new_pos[0] - new_pos[1])
    assert dist >= 0.5 - 1e-6  # sum of radii
    assert abs(max_overlap(new_pos, 0.25)) < 1e-6


def test_coincident_agents_dont_crash_and_separate():
    m = corridor(length=20, width=5)
    pos = [[10.0, 2.5], [10.0, 2.5]]  # exactly on top of each other
    vel = [[0.0, 0.0], [0.0, 0.0]]
    new_pos, _ = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m, push_iterations=5)
    dist = np.linalg.norm(new_pos[0] - new_pos[1])
    assert dist > 0.0  # no longer coincident
    assert np.all(np.isfinite(new_pos))


def test_agent_cannot_be_driven_through_a_wall():
    m = corridor(length=20, width=5)
    pos = [[10.0, 4.85]]  # close to the top wall (width=5)
    vel = [[0.0, 5.0]]  # driving straight into it, hard, for a whole second -- a big overlap
    new_pos, _ = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m, push_iterations=10)
    assert m.is_walkable(new_pos)[0]
    assert m.distance_at(new_pos)[0] >= 0.25 - 1e-3  # circle actually clears the wall


def test_hard_fallback_catches_an_extreme_overshoot():
    # A huge desired velocity would land the agent far outside the map entirely; the hard
    # fallback (nearest_walkable) must still bring it back onto the walkable area.
    m = room_with_exit(width=20, height=15, door_width=1.5)
    pos = [[2.0, 2.0]]
    vel = [[10000.0, 10000.0]]
    new_pos, _ = step(pos, vel, radii=0.25, dt=1.0, walkable_map=m)
    assert m.is_walkable(new_pos)[0]


def test_many_agents_in_a_bottleneck_end_up_non_overlapping():
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0)
    rng = np.random.default_rng(0)
    # Cram 20 agents into a tight cluster right before the gap -- exactly the "clogged doorway"
    # situation this project cares about.
    gap_x = sum(m.meta["gap_x"]) / 2
    pos = np.c_[
        gap_x - 3 + rng.uniform(-1, 1, 20),
        m.meta["width"] / 2 + rng.uniform(-1, 1, 20),
    ]
    vel = np.tile([1.0, 0.0], (20, 1))  # everyone wants to push toward the gap
    # A tight cluster needs more relaxation passes to fully de-overlap than a typical single
    # step would use -- more iterations trades compute for accuracy, a real tuning knob, not a
    # magic number. 20 gets this cluster comfortably under the tolerance below.
    new_pos, _ = step(pos, vel, radii=0.25, dt=0.1, walkable_map=m, push_iterations=20)
    assert np.all(m.is_walkable(new_pos))
    assert max_overlap(new_pos, 0.25) < 0.01


def test_agent_and_wall_overlap_dont_permanently_fight_each_other():
    # Real bug, fixed: agent-agent push and wall push used to run as two separate passes, each to
    # its own full convergence -- so a wall push that shoved an agent back into another agent had
    # no way to get corrected, and more `push_iterations` genuinely never helped. Reproduced
    # exactly like this: two agents stacked 5 cm apart, both close enough to a wall that separating
    # them pushes one back into it. Before the fix this stayed at ~0.45 m overlap regardless of
    # push_iterations (3, 10, 20, 50 all identical); the interleaved agent/wall relax in `_relax`
    # must actually resolve it given enough rounds, the exact case M2's wall-adjacent bottleneck
    # congestion will hit constantly, not as a rare edge case.
    m = corridor(length=20, width=5)
    pos = [[5.0, 0.15], [5.0, 0.20]]  # both close to the y=0 wall, 5 cm apart
    vel = [[0.0, 0.0], [0.0, 0.0]]
    r = 0.3
    new_pos, _ = step(pos, vel, radii=r, dt=1.0, walkable_map=m, push_iterations=20)
    assert m.is_walkable(new_pos).all()
    assert max_overlap(new_pos, r) < 1e-6  # this used to be stuck at ~0.45 no matter the iterations


def test_actual_velocity_reflects_correction_not_just_intent():
    # An agent whose desired move is blocked by a wall should show a different actual velocity
    # than what it asked for -- that's the whole point of returning it separately.
    m = corridor(length=20, width=5)
    pos = [[10.0, 4.9]]
    vel = [[0.0, 3.0]]
    _, actual_vel = step(pos, vel, radii=0.3, dt=1.0, walkable_map=m)
    assert not np.allclose(actual_vel, vel, atol=0.1)
