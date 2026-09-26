"""Baseline (social-force-flavoured) steering model tests."""

import numpy as np

from flybrainflow.agents import Population
from flybrainflow.brains import Baseline
from flybrainflow.scenario import Scenario
from flybrainflow.world import physics_step
from flybrainflow.world.map import bottleneck, corridor, room_with_exit


def test_single_agent_heads_toward_its_target_in_the_open():
    m = corridor(length=20, width=5)
    brain = Baseline(m, target_positions=[[18.0, 2.5]])
    vel = brain.desired_velocities(ids=[0], positions=[[2.0, 2.5]], radii=0.25, max_speed_mps=1.3)
    assert vel.shape == (1, 2)
    direction = vel[0] / np.linalg.norm(vel[0])
    assert direction[0] > 0.99  # straight down the open corridor, toward +x
    assert np.linalg.norm(vel[0]) > 1.0  # close to max speed with nothing nearby to slow it


def test_agent_actually_gets_around_a_wall_instead_of_stalling():
    # The whole reason to steer off the geodesic field and not a straight line: a target behind
    # a wall must not leave the agent stuck pressed against that wall forever.
    stub_length = 4.0
    m = room_with_exit(width=20, height=15, door_width=1.5, stub_length=stub_length)
    y0, y1 = m.meta["door_y"]
    door_y = (y0 + y1) / 2
    target = [m.meta["width"] + stub_length - 0.5, door_y]
    brain = Baseline(m, target_positions=[target])
    pos = np.array([[2.0, 2.0]])
    for _ in range(400):
        vel = brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3)
        pos, _ = physics_step(pos, vel, radii=0.25, dt=0.1, walkable_map=m, max_speed_mps=1.3)
    dist_to_target = np.linalg.norm(pos[0] - np.array(target))
    assert dist_to_target < 1.0, f"agent stalled {dist_to_target:.2f} m from the target"


def test_converging_agents_dont_pile_up():
    m = corridor(length=20, width=5)
    brain = Baseline(m, target_positions=[[18.0, 2.5]])
    rng = np.random.default_rng(0)
    pos = np.c_[np.full(10, 2.0) + rng.uniform(-0.1, 0.1, 10), 2.5 + rng.uniform(-0.3, 0.3, 10)]
    ids = list(range(10))
    for _ in range(50):
        vel = brain.desired_velocities(ids, pos, radii=0.25, max_speed_mps=1.3)
        pos, _ = physics_step(pos, vel, radii=0.25, dt=0.1, walkable_map=m, max_speed_mps=1.3)
    from flybrainflow.world import max_overlap

    assert max_overlap(pos, 0.25) < 0.01
    assert np.all(m.is_walkable(pos))


def test_target_assignment_is_sticky():
    m = corridor(length=20, width=5)
    brain = Baseline(m, target_positions=[[5.0, 2.5], [5.2, 2.5]])  # two close targets
    pos = np.array([[4.9, 2.5]])
    brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3)
    first_choice = brain._assigned[0]
    # Move the agent slightly closer to the *other* target -- a re-decided assignment would flip.
    pos = np.array([[5.15, 2.5]])
    brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3)
    assert brain._assigned[0] == first_choice


def test_preferred_targets_overrides_nearest_for_a_fresh_id_only():
    # Real bug, fixed: a source co-located with (or simply closer to) the "wrong" target always
    # won nearest-distance assignment, silently breaking any scenario built around a fly walking
    # to the FAR target (see scenarios/corridor_bidirectional.toml's own comment). `preferred_targets`
    # is the escape hatch -- confirm it actually wins for a first-seen id, and that (matching
    # nearest-distance's own stickiness) it has no effect once an id is already assigned.
    m = corridor(length=20, width=5)
    brain = Baseline(m, target_positions=[[1.0, 2.5], [18.0, 2.5]])  # index 0 near, index 1 far
    pos = np.array([[1.5, 2.5]])  # right next to target 0 -- nearest-distance would pick it

    brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3, preferred_targets={0: 1})
    assert brain._assigned[0] == 1  # overridden to the far target despite being right next to the near one

    # A second id with no entry in `preferred_targets` still gets ordinary nearest-distance.
    brain.desired_velocities(ids=[1], positions=pos, radii=0.25, max_speed_mps=1.3, preferred_targets={0: 1})
    assert brain._assigned[1] == 0

    # Once assigned, `preferred_targets` doesn't retroactively change anything -- same stickiness
    # as nearest-distance assignment (see test_target_assignment_is_sticky above).
    brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3, preferred_targets={0: 0})
    assert brain._assigned[0] == 1


def test_full_loop_a_spawned_fly_reaches_feeds_and_leaves():
    # Ties spawn/feed/leave (agents.py), steering (baseline.py) and movement (physics.py)
    # together end to end -- the same three-piece loop agents.py's own docstring describes.
    raw = {
        "scenario": {"name": "t", "boundary_mode": "open"},
        "map": {"source": "gen:corridor?length=10&width=5"},
        "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 1.0, "population_cap": 1},
        "targets": [{"position": [9.0, 2.5], "slots": 1, "feeding_time_s": 0.5}],
    }
    sc = Scenario.from_dict(raw)
    m = corridor(length=10, width=5)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    brain = Baseline(m, target_positions=[t.position for t in sc.targets])

    pop.spawn(dt=2.0)
    assert len(pop.agents) == 1
    left_ids: list[int] = []
    for _ in range(300):
        pop.spawn(dt=0.1)
        ids = pop.walking_ids()
        if ids:
            pos = pop.positions(ids)
            vel = brain.desired_velocities(ids, pos, radii=sc.agents.radius_m, max_speed_mps=sc.agents.max_speed_mps)
            new_pos, _ = physics_step(pos, vel, radii=sc.agents.radius_m, dt=0.1, walkable_map=m, max_speed_mps=sc.agents.max_speed_mps)
            pop.set_positions(ids, new_pos)
        pop.feed(dt=0.1, capture_radius=0.5)
        newly_left = pop.leave()
        brain.forget(newly_left)
        left_ids += newly_left
        if left_ids:
            break
    assert left_ids, "the fly never made it to the target and left"
    assert len(pop.agents) == 0
    assert brain._assigned == {}  # forget() actually released it, not just left it stale


def test_move_target_steers_an_already_assigned_agent_toward_the_new_position():
    m = corridor(length=40, width=5)
    brain = Baseline(m, target_positions=[[38.0, 2.5]])
    pos = np.array([[20.0, 2.5]])
    vel_before = brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3)
    assert vel_before[0, 0] > 0.99 * np.linalg.norm(vel_before[0])  # heading toward +x, as expected

    brain.move_target(0, [2.0, 2.5])  # relocate the same (already-assigned) target behind the agent
    vel_after = brain.desired_velocities(ids=[0], positions=pos, radii=0.25, max_speed_mps=1.3)
    assert brain._assigned[0] == 0  # still committed to target index 0
    assert vel_after[0, 0] < -0.99 * np.linalg.norm(vel_after[0])  # now heading toward -x instead


def test_forget_prevents_the_assigned_target_dict_from_growing_forever():
    # Real bug, fixed: `_assigned` kept every id ever spawned, forever, even once an agent had
    # long since fed and left. Over a realistic run length this grows without bound -- caught by
    # simulating a full open-boundary run where 300s produced 300+ flies with only a handful ever
    # alive at once, and the dict held onto all of them.
    raw = {
        "scenario": {"name": "t", "boundary_mode": "open"},
        "map": {"source": "gen:corridor?length=15&width=5"},
        "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 3.0, "population_cap": 10},
        "targets": [{"position": [13.0, 2.5], "slots": 5, "feeding_time_s": 0.3}],
    }
    sc = Scenario.from_dict(raw)
    m = corridor(length=15, width=5)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    brain = Baseline(m, target_positions=[t.position for t in sc.targets])

    for _ in range(1500):  # 150 simulated seconds
        pop.spawn(dt=0.1)
        ids = pop.walking_ids()
        if ids:
            pos = pop.positions(ids)
            vel = brain.desired_velocities(ids, pos, radii=0.25, max_speed_mps=1.3)
            new_pos, _ = physics_step(pos, vel, radii=0.25, dt=0.1, walkable_map=m, max_speed_mps=1.3)
            pop.set_positions(ids, new_pos)
        pop.feed(dt=0.1, capture_radius=0.5)
        brain.forget(pop.leave())

    assert pop._next_id > 50  # plenty of flies have cycled through by now
    # The dict should track only currently-alive agents, not everyone who ever spawned.
    assert len(brain._assigned) <= len(pop.agents)
