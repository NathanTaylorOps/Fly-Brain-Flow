"""Direct unit tests for `TargetAssignment` -- extracted out of `Baseline`/`ToyBrain` for reuse,
but previously only exercised indirectly through those two brains' own tests. These pin down its
contract on its own: commit-once assignment, the `preferred` override, per-target field caching,
and `forget`'s cleanup."""

import numpy as np

from flybrainflow.brains.targeting import TargetAssignment
from flybrainflow.world.map import corridor


def test_each_id_is_assigned_to_its_nearest_target_by_straight_line_distance():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    assigned = ta.assign(ids=[0, 1], positions=np.array([[5.0, 2.5], [35.0, 2.5]]))
    assert list(assigned) == [0, 1]  # id 0 nearer the x=2 target, id 1 nearer the x=38 target


def test_assignment_is_one_time_and_does_not_change_if_the_agent_moves_closer_to_another_target():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    first = ta.assign(ids=[0], positions=np.array([[5.0, 2.5]]))
    assert first[0] == 0
    # Same id, now much closer to the *other* target -- must not re-route, since suppressing that
    # would suppress the queuing/choke-point behaviour this project exists to study.
    second = ta.assign(ids=[0], positions=np.array([[37.0, 2.5]]))
    assert second[0] == 0


def test_preferred_overrides_nearest_distance_for_a_fresh_id_only():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    # Spawned right next to target 0, but pinned to target 1 -- the case nearest-distance alone
    # can't handle (see the module's own docstring and SpawnConfig.source_targets).
    assigned = ta.assign(ids=[0], positions=np.array([[2.5, 2.5]]), preferred={0: 1})
    assert assigned[0] == 1
    # Already assigned -- a later `preferred` for the same id has no effect.
    again = ta.assign(ids=[0], positions=np.array([[2.5, 2.5]]), preferred={0: 0})
    assert again[0] == 1


def test_field_for_builds_once_and_caches_the_same_array_object():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[38.0, 2.5]])
    first = ta.field_for(0)
    second = ta.field_for(0)
    assert first is second  # cached, not recomputed


def test_forget_releases_ids_so_assigned_does_not_grow_forever():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[38.0, 2.5]])
    ta.assign(ids=[0, 1, 2], positions=np.array([[5.0, 2.5]] * 3))
    assert len(ta.assigned) == 3
    ta.forget([0, 1])
    assert set(ta.assigned) == {2}
    ta.forget([0])  # forgetting an id that's already gone is a no-op, not an error
    assert set(ta.assigned) == {2}


def test_construction_rejects_an_empty_target_list():
    m = corridor(length=40, width=5)
    try:
        TargetAssignment(m, target_positions=[])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_move_target_forces_field_for_to_recompute_not_serve_a_stale_field():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    before = ta.field_for(1)
    ta.move_target(1, [20.0, 2.5])  # move target 1 into the middle of the corridor
    after = ta.field_for(1)
    assert not np.array_equal(before, after)  # recomputed for the new position, not reused


def test_move_target_leaves_an_already_assigned_id_committed_to_the_same_index():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    first = ta.assign(ids=[0], positions=np.array([[5.0, 2.5]]))
    assert first[0] == 0
    # Relocate target 0 clear across the map -- the agent stays committed to *index* 0, not to
    # the position it used to occupy.
    ta.move_target(0, [39.0, 2.5])
    again = ta.assign(ids=[0], positions=np.array([[5.0, 2.5]]))
    assert again[0] == 0


def test_move_target_affects_distance_comparisons_for_ids_assigned_after_the_move():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    # At x=20, target 0 (still at x=2) is nearer than target 1 (at x=38).
    pre_move = ta.assign(ids=[0], positions=np.array([[20.0, 2.5]]))
    assert pre_move[0] == 0
    # Move target 0 far away and target 1 close in -- now target 1 is the nearer one at x=20.
    ta.move_target(0, [39.0, 2.5])
    ta.move_target(1, [21.0, 2.5])
    post_move = ta.assign(ids=[1], positions=np.array([[20.0, 2.5]]))
    assert post_move[0] == 1  # a fresh id, assigned only after the move, follows the new positions


def test_move_target_does_not_disturb_other_targets_cached_fields():
    m = corridor(length=40, width=5)
    ta = TargetAssignment(m, target_positions=[[2.0, 2.5], [38.0, 2.5]])
    other_before = ta.field_for(1)
    ta.move_target(0, [20.0, 2.5])
    other_after = ta.field_for(1)
    assert other_before is other_after  # untouched target's cached field is still the same object
