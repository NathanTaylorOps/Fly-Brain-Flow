"""Toy brain tests.

Mirrors `test_baseline.py`'s shape where the contract is the same (per-id steering vectors,
`forget()` releasing state), plus tests specific to the two real bugs this brain went through
during development -- see `flybrainflow/brains/toy.py`'s module docstring and docs/JOURNAL.md
for the honest account of both:

  1. Turn-rate instability: heading is an integrator (this tick's turn compounds onto every
     previous tick's), unlike `Baseline`'s stateless-per-tick steering. A few agents pressed
     together for a while, with no guard rail on the turn signal itself, spun at 400+ deg/s.
  2. The bilateral (cross-product) turn signal is mathematically zero both facing the target
     (0 degrees off) AND facing directly away from it (180 degrees off) -- a fly rotated past
     ~90 degrees off course by avoidance had nothing pulling it back and would cruise away
     forever. Fixed by steering the designed reflex on heading error (atan2, P-controller)
     instead of the bilateral signal.
"""

import numpy as np

from flybrainflow.agents import Population
from flybrainflow.brains import ToyBrain, placeholder_personality_table
from flybrainflow.brains.toy import _TURN_MAX
from flybrainflow.scenario import Scenario
from flybrainflow.world import OdorField, OdorSource, Wind, physics_step
from flybrainflow.world.map import corridor


class Target:
    def __init__(self, position, kind="sugar"):
        self.position = tuple(position)
        self.kind = kind


def _odor(m, target, strength=1.0, range_m=60.0, wind_speed=0.0):
    return OdorField(
        Wind(0.0, wind_speed),
        [OdorSource(kind=target.kind, position=target.position, strength=strength, range_m=range_m)],
        walkable_map=m,
    )


def test_single_fly_heads_toward_and_reaches_its_target_in_the_open():
    m = corridor(length=20, width=5)
    target = Target([18.0, 2.5])
    odor = _odor(m, target)
    brain = ToyBrain(m, [target], scenario_seed=0)
    pos = np.array([[2.0, 2.5]])
    for _ in range(800):
        vel = brain.desired_velocities([0], pos, radii=0.25, max_speed_mps=1.3, personalities=[0], odor_field=odor, dt=0.05)
        pos, _ = physics_step(pos, vel, radii=0.25, dt=0.05, walkable_map=m, max_speed_mps=1.3)
    dist = np.linalg.norm(pos[0] - np.array(target.position))
    assert dist < 1.0, f"fly stalled {dist:.2f} m from its target"


def test_several_flies_converge_without_getting_stuck_facing_away():
    # Regression test for bug #2 above: before the heading-error fix, several flies started
    # close together would have some of them rotated past ~90 degrees off course by mutual
    # avoidance and then cruise away from the target indefinitely, never correcting.
    m = corridor(length=20, width=10)
    target = Target([18.0, 5.0])
    odor = _odor(m, target)
    brain = ToyBrain(m, [target], scenario_seed=1)
    n = 6
    ids = list(range(n))
    rng = np.random.default_rng(0)
    pos = np.c_[np.full(n, 1.5) + rng.uniform(-0.2, 0.2, n), 5.0 + rng.uniform(-1.0, 1.0, n)]
    radii = np.full(n, 0.25)
    personalities = rng.integers(0, 20, n)
    for _ in range(1200):
        vel = brain.desired_velocities(ids, pos, radii, max_speed_mps=1.3, personalities=personalities, odor_field=odor, dt=0.05)
        pos, _ = physics_step(pos, vel, radii=radii, dt=0.05, walkable_map=m, max_speed_mps=1.3)
    dist = np.linalg.norm(pos - np.array(target.position), axis=1)
    assert np.all(dist < 1.5), f"some flies never converged: {dist}"


def test_turn_signal_never_exceeds_its_guard_rail():
    # Regression test for bug #1: crowd several flies together (heavy mutual avoidance drive)
    # and check the guard rail actually holds, rather than re-deriving the 400+ deg/s failure
    # from scratch each time by inspecting headings.
    m = corridor(length=10, width=5)
    target = Target([8.0, 2.5])
    odor = _odor(m, target)
    brain = ToyBrain(m, [target], scenario_seed=2)
    n = 10
    ids = list(range(n))
    rng = np.random.default_rng(1)
    pos = np.c_[np.full(n, 2.0) + rng.uniform(-0.05, 0.05, n), 2.5 + rng.uniform(-0.3, 0.3, n)]
    radii = np.full(n, 0.25)
    personalities = rng.integers(0, 20, n)
    dt = 0.05
    max_step_rad = _TURN_MAX * 1.0 * dt  # _TURN_GAIN is 1.0; see toy.py
    for _ in range(100):
        prev_heading = np.array([brain._heading.get(i, None) for i in ids], dtype=object)
        vel = brain.desired_velocities(ids, pos, radii, max_speed_mps=1.3, personalities=personalities, odor_field=odor, dt=dt)
        pos, _ = physics_step(pos, vel, radii=radii, dt=dt, walkable_map=m, max_speed_mps=1.3)
        new_heading = np.array([brain._heading[i] for i in ids])
        for i, before in zip(ids, prev_heading):
            if before is None:
                continue
            delta = (new_heading[ids.index(i)] - before + np.pi) % (2 * np.pi) - np.pi
            assert abs(delta) <= max_step_rad + 1e-9, f"fly {i} turned {np.degrees(delta):.1f} deg in one tick"


def test_forget_releases_all_three_per_agent_dicts():
    m = corridor(length=10, width=5)
    target = Target([8.0, 2.5])
    odor = _odor(m, target)
    brain = ToyBrain(m, [target], scenario_seed=0)
    pos = np.array([[2.0, 2.5], [2.5, 2.5]])
    brain.desired_velocities([0, 1], pos, radii=0.25, max_speed_mps=1.3, personalities=[0, 0], odor_field=odor, dt=0.05)
    assert 0 in brain._activity and 0 in brain._heading and 0 in brain._rng
    brain.forget([0])
    assert 0 not in brain._activity
    assert 0 not in brain._heading
    assert 0 not in brain._rng
    assert 0 not in brain._targets.assigned
    # id 1 untouched
    assert 1 in brain._activity and 1 in brain._heading and 1 in brain._rng


def test_personality_actually_changes_behaviour():
    # Not a claim about which personality should be faster in general -- just that the plumbing
    # responds to the dial at all, using the placeholder table's own declared range. A strong odor
    # source is deliberately avoided here: the documented DNp09 "very high drive = brake" quirk
    # (`_forward_speed_curve`) means a higher `baseline_speed` fly can overshoot the curve's peak
    # sooner once its food-drive term stacks on top, and actually net *slower* in that regime --
    # a real, intentional property of the design, not something to paper over in this test. A weak
    # odor source keeps drive away from that crossover so the comparison isolates `baseline_speed`.
    m = corridor(length=20, width=5)
    target = Target([18.0, 2.5])
    odor = _odor(m, target, strength=0.1)
    table = placeholder_personality_table()
    slow = int(np.argmin(table["baseline_speed"]))
    fast = int(np.argmax(table["baseline_speed"]))
    assert slow != fast

    def run(personality):
        brain = ToyBrain(m, [target], scenario_seed=0, personality_table=table)
        pos = np.array([[2.0, 2.5]])
        for _ in range(100):
            vel = brain.desired_velocities([0], pos, radii=0.25, max_speed_mps=1.3, personalities=[personality], odor_field=odor, dt=0.05)
            pos, _ = physics_step(pos, vel, radii=0.25, dt=0.05, walkable_map=m, max_speed_mps=1.3)
        return pos[0, 0]

    assert run(fast) > run(slow)


def test_reproducible_per_fly_behaviour_from_scenario_seed():
    m = corridor(length=20, width=5)
    target = Target([18.0, 2.5])
    odor = _odor(m, target)

    def run():
        brain = ToyBrain(m, [target], scenario_seed=42)
        pos = np.array([[2.0, 2.5]])
        trace = []
        for _ in range(50):
            vel = brain.desired_velocities([7], pos, radii=0.25, max_speed_mps=1.3, personalities=[3], odor_field=odor, dt=0.05)
            pos, _ = physics_step(pos, vel, radii=0.25, dt=0.05, walkable_map=m, max_speed_mps=1.3)
            trace.append(pos.copy())
        return trace

    a, b = run(), run()
    for pa, pb in zip(a, b):
        assert np.allclose(pa, pb)


def test_different_agent_ids_do_not_share_a_noise_stream():
    # Reproducibility is per-(scenario_seed, agent_id), not a single shared stream that would
    # make one fly's path depend on what order ids happen to be processed in.
    m = corridor(length=20, width=5)
    target = Target([18.0, 2.5])
    odor = _odor(m, target)
    brain_a = ToyBrain(m, [target], scenario_seed=5)
    brain_b = ToyBrain(m, [target], scenario_seed=5)
    pos_a = np.array([[2.0, 2.5]])
    pos_b = np.array([[2.0, 2.5]])
    for _ in range(20):
        vel_a = brain_a.desired_velocities([0], pos_a, radii=0.25, max_speed_mps=1.3, personalities=[0], odor_field=odor, dt=0.05)
        pos_a, _ = physics_step(pos_a, vel_a, radii=0.25, dt=0.05, walkable_map=m, max_speed_mps=1.3)
    for _ in range(20):
        vel_b = brain_b.desired_velocities([3], pos_b, radii=0.25, max_speed_mps=1.3, personalities=[0], odor_field=odor, dt=0.05)
        pos_b, _ = physics_step(pos_b, vel_b, radii=0.25, dt=0.05, walkable_map=m, max_speed_mps=1.3)
    # Different ids under the same scenario seed get different (decorrelated) noise draws, so
    # their paths should differ even though everything else about the setup is identical.
    assert not np.allclose(pos_a, pos_b)


def test_full_loop_a_spawned_fly_reaches_feeds_and_leaves():
    # Same end-to-end shape as Baseline's equivalent test: spawn/feed/leave (agents.py), steering
    # (toy.py) and movement (physics.py) tied together, with `forget()` actually wired up.
    raw = {
        "scenario": {"name": "t", "boundary_mode": "open"},
        "map": {"source": "gen:corridor?length=10&width=5"},
        "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 1.0, "population_cap": 1},
        "targets": [{"kind": "sugar", "position": [9.0, 2.5], "slots": 1, "feeding_time_s": 0.5}],
    }
    sc = Scenario.from_dict(raw)
    m = corridor(length=10, width=5)
    target = Target(sc.targets[0].position, sc.targets[0].kind)
    odor = _odor(m, target)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    brain = ToyBrain(m, [target], scenario_seed=sc.meta.seed)

    pop.spawn(dt=2.0)
    assert len(pop.agents) == 1
    left_ids: list[int] = []
    for _ in range(600):
        pop.spawn(dt=0.1)
        ids = pop.walking_ids()
        if ids:
            pos = pop.positions(ids)
            personalities = [pop.agents[i].personality for i in ids]
            vel = brain.desired_velocities(
                ids, pos, radii=sc.agents.radius_m, max_speed_mps=sc.agents.max_speed_mps,
                personalities=personalities, odor_field=odor, dt=0.1,
            )
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
    assert brain._activity == {} and brain._heading == {} and brain._rng == {}
