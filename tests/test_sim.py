"""Sim loop tests: the first place `agents.py`, `brains/` and `world/` actually run together
end to end, across a full tick (spawn -> steer -> move -> feed -> leave) and across more than one
cohort sharing a venue.
"""

import numpy as np

from flybrainflow.agents import Population
from flybrainflow.brains import Baseline
from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim
from flybrainflow.world import load_map_for_scenario, max_overlap, physics_step


def _scenario(**overrides):
    raw = {
        "scenario": {"name": "t", "boundary_mode": "open", "seed": overrides.pop("seed", 0)},
        "map": {"source": overrides.pop("map", "gen:corridor?length=15&width=5")},
        "spawn": {
            "sources": overrides.pop("sources", [[1.0, 2.5]]),
            "rate_per_s": overrides.pop("rate_per_s", 2.0),
            "population_cap": overrides.pop("population_cap", 8),
        },
        "targets": overrides.pop(
            "targets", [{"kind": "sugar", "position": [13.0, 2.5], "slots": 4, "feeding_time_s": 1.0}]
        ),
        "baseline": overrides.pop("baseline", {"enabled": True}),
    }
    return Scenario.from_dict(raw)


def test_a_single_tick_runs_without_error_and_moves_the_spawned_fly():
    # rate_per_s=2.0 needs half a second of accumulated dt before the fractional spawn
    # accumulator (see agents.py) actually produces anyone -- tick past that point first.
    sc = _scenario(population_cap=1)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    sim.tick(1.0)
    assert sim.t == 1.0
    ids = sim.cohorts["brain"].walking_ids()
    assert len(ids) == 1
    pos = sim.cohorts["brain"].positions(ids)[0]
    assert pos[0] > 1.0  # nudged toward the target at x=13, away from the x=1 source


def test_baseline_disabled_produces_only_the_brain_cohort():
    sc = _scenario(baseline={"enabled": False})
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    assert set(sim.cohorts) == {"brain"}
    assert set(sim.brains) == {"brain"}
    for _ in range(20):
        sim.tick(0.1)  # must not crash with no baseline cohort in the mix


def test_both_cohorts_run_reach_and_feed_at_a_shared_target():
    # The end-to-end claim this whole loop exists to prove: two cohorts, sharing one venue, one
    # target, one physics call -- both actually make it there and feed, not just one of them.
    sc = _scenario(population_cap=6, rate_per_s=3.0)
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(1))
    for _ in range(800):
        sim.tick(0.1)
    # Some agents from each cohort should have cycled all the way through (spawned, fed, left) by
    # now -- `_next_id` only advances past the population cap once flies have actually left.
    assert sim.cohorts["brain"]._next_id > sc.spawn.population_cap
    assert sim.cohorts["baseline"]._next_id > sc.spawn.population_cap


def test_cohorts_share_ids_and_target_slots_through_a_real_run():
    # Not just that `build_cohorts` wires the sharing up (see test_agents.py) -- that it actually
    # holds after hundreds of real ticks of spawning, steering, moving, feeding and leaving.
    sc = _scenario(
        population_cap=4,
        rate_per_s=3.0,
        targets=[{"kind": "sugar", "position": [13.0, 2.5], "slots": 2, "feeding_time_s": 1.0}],
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(2))
    for _ in range(500):
        sim.tick(0.1)
        brain_ids = set(sim.cohorts["brain"].agents)
        baseline_ids = set(sim.cohorts["baseline"].agents)
        assert brain_ids.isdisjoint(baseline_ids)
        # Both cohorts see the same shared slot list (same object) -- never each thinking it has
        # the target's full 2 slots to itself.
        assert sim.cohorts["brain"].slots_used is sim.cohorts["baseline"].slots_used
        assert sum(sim.cohorts["brain"].slots_used) <= 2


def test_no_agent_from_either_cohort_ever_leaves_the_walkable_map():
    sc = _scenario(population_cap=10, rate_per_s=4.0, map="gen:corridor?length=15&width=3")
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(3))
    for _ in range(300):
        sim.tick(0.1)
        for pop in sim.cohorts.values():
            ids = pop.walking_ids()
            if ids:
                assert sim.map.is_walkable(pop.positions(ids)).all()


def test_forgetting_a_left_agent_actually_reaches_its_brain():
    # `ToyBrain`/`Baseline` both leak per-agent state if `forget()` never gets called (see their
    # own docstrings/tests) -- this checks the sim loop is actually the one calling it, not just
    # that the two pieces work in isolation.
    sc = Scenario.from_dict(
        {
            "scenario": {"name": "t", "boundary_mode": "open", "seed": 0},
            "map": {"source": "gen:corridor?length=6&width=5"},
            "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 5.0, "population_cap": 1},
            "targets": [{"kind": "sugar", "position": [4.0, 2.5], "slots": 1, "feeding_time_s": 0.2}],
            "baseline": {"enabled": False},
        }
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    brain = sim.brains["brain"]
    for _ in range(300):
        sim.tick(0.05)
        if brain._heading == {} and sim.t > 1.0:
            break
    assert brain._heading == {} and brain._activity == {} and brain._rng == {}
    assert len(sim.cohorts["brain"].agents) == 0


def test_same_seed_reproduces_a_full_multi_cohort_run_exactly():
    # Real gap, fixed: docs/PLAN.md's own test list names "same seed gives the same run" as a
    # top-level invariant, but the only existing reproducibility test drove a single ToyBrain
    # instance directly (tests/test_toy.py) -- nothing checked it at the level that actually
    # matters in production: a full Sim, both cohorts, spawn timing and personality draws and
    # multi-agent steering all going through their own independent rng streams together. Two
    # completely separate Sim instances built from the same seed, run the same number of ticks,
    # must end up bit-for-bit identical -- not just "similar populations", but literally the same
    # ids at the same positions with the same status.
    sc = _scenario(population_cap=8, rate_per_s=3.0, seed=99)
    sim_a = Sim.from_scenario(sc, rng=np.random.default_rng(99))
    sim_b = Sim.from_scenario(sc, rng=np.random.default_rng(99))
    for _ in range(300):
        sim_a.tick(0.1)
        sim_b.tick(0.1)
    for tag in sim_a.cohorts:
        agents_a = sim_a.cohorts[tag].agents
        agents_b = sim_b.cohorts[tag].agents
        assert set(agents_a) == set(agents_b), f"{tag}: different ids survived"
        for i in agents_a:
            a, b = agents_a[i], agents_b[i]
            assert a.status == b.status and a.target_index == b.target_index
            assert np.allclose(a.position, b.position, atol=1e-12), f"{tag} agent {i}: positions diverged"


def test_source_targets_makes_two_streams_actually_cross_the_corridor():
    # Real bug, fixed: corridor_bidirectional.toml's two sources sat right next to a target each
    # (a "two-stream lane test"), so nearest-distance assignment always sent every fly straight
    # back to its own end -- nobody ever crossed. `source_targets` pins each source to the FAR
    # target instead. This is the sim-level proof it actually works: flies born at x=1 should be
    # assigned target 0 (at x=39, per this scenario's source_targets) and get moving in the +x
    # direction, and vice versa for flies born at x=39 -- and, over a real run, a growing number of
    # agents should be genuinely present in the middle of the corridor (which never happened
    # before this fix).
    sc = Scenario.from_dict(
        {
            "scenario": {"name": "t", "boundary_mode": "open", "seed": 5},
            "map": {"source": "gen:corridor?length=40&width=5"},
            "spawn": {
                "sources": [
                    {"position": [1.0, 2.5], "target": 0},
                    {"position": [39.0, 2.5], "target": 1},
                ],
                "rate_per_s": 3.0,
                "population_cap": 40,
            },
            "targets": [
                {"kind": "sugar", "position": [39.0, 2.5], "slots": 6, "feeding_time_s": 2.0},
                {"kind": "sugar", "position": [1.0, 2.5], "slots": 6, "feeding_time_s": 2.0},
            ],
            "baseline": {"enabled": False},
        }
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(5))
    mid_counts = []
    for tick in range(600):
        sim.tick(0.1)
        ids = sim.cohorts["brain"].walking_ids()
        if ids and tick % 100 == 99:
            xs = sim.cohorts["brain"].positions(ids)[:, 0]
            mid_counts.append(int(np.sum((xs > 10) & (xs < 30))))
    # Some agents should genuinely be mid-corridor -- impossible if everyone just walked straight
    # back to their own end's target, which is exactly what the pre-fix nearest-distance bug did.
    assert max(mid_counts) > 0, "no agent ever reached the middle of the corridor -- streams aren't crossing"


def test_arrivals_minus_departures_equals_current_population():
    # Real gap, fixed: docs/PLAN.md states this as one of five specific invariants the project
    # promises to hold ("arrivals minus departures equals current population"), but nothing
    # actually asserted it directly -- every other test only checked population counts
    # incidentally (e.g. that _next_id grows past the cap). Drive Population's own lifecycle calls
    # directly (spawn
    # returns arrival ids, leave returns departure ids -- both are the real, single source of
    # truth), and assert arrivals - departures == len(agents) after every tick, for real, over a
    # run with heavy concurrent spawn/feed/leave churn (a small population_cap and a fast
    # feeding_time_s so many flies fully cycle through).
    sc = Scenario.from_dict(
        {
            "scenario": {"name": "t", "boundary_mode": "open", "seed": 3},
            "map": {"source": "gen:corridor?length=10&width=5"},
            "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 4.0, "population_cap": 5},
            "targets": [{"position": [9.0, 2.5], "slots": 2, "feeding_time_s": 0.3}],
        }
    )
    m = load_map_for_scenario(sc)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(3))
    brain = Baseline(m, target_positions=[t.position for t in sc.targets])
    arrivals = 0
    departures = 0
    for _ in range(2000):
        arrivals += len(pop.spawn(0.05))
        ids = pop.walking_ids()
        if ids:
            pos = pop.positions(ids)
            vel = brain.desired_velocities(ids, pos, radii=sc.agents.radius_m, max_speed_mps=sc.agents.max_speed_mps)
            new_pos, _ = physics_step(pos, vel, radii=sc.agents.radius_m, dt=0.05, walkable_map=m, max_speed_mps=sc.agents.max_speed_mps)
            pop.set_positions(ids, new_pos)
        pop.feed(0.05)
        departed = pop.leave()
        brain.forget(departed)
        departures += len(departed)
        assert arrivals - departures == len(pop.agents)
    assert arrivals > 10 and departures > 10  # the run actually churned through real flies


def test_move_target_reroutes_a_walking_fly_and_moves_capture_with_it():
    # The actual M1 Step 0 claim, exercised at the level that matters: a fly already committed to
    # a target, mid-walk, re-routes when that target moves -- and capture only fires at the new
    # spot, not the frozen scenario config's original one. Long, narrow corridor so "closer to the
    # old end" vs "closer to the new end" is unambiguous from x-position alone.
    sc = _scenario(
        population_cap=1,
        rate_per_s=999.0,  # spawn immediately, no need to wait out the accumulator
        map="gen:corridor?length=30&width=5",
        sources=[[1.0, 2.5]],
        targets=[{"kind": "sugar", "position": [29.0, 2.5], "slots": 4, "feeding_time_s": 1.0}],
        baseline={"enabled": False},
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    sim.tick(0.1)  # spawn the one fly
    fly_id = sim.cohorts["brain"].walking_ids()[0]
    pop = sim.cohorts["brain"]
    for _ in range(20):  # a real, physics-moved head start -- not sitting right on the spawn point
        sim.tick(0.1)
    pos = pop.positions([fly_id])

    # Drive `_steer` directly at this fixed, already-advanced position (not a further multi-second
    # tick loop, which would race against the fly actually reaching and being captured by whichever
    # target it's currently heading for -- this isolates "does it re-route", not "does it eventually
    # arrive"). ToyBrain's turn rate is bounded (see toy.py's `_TURN_MAX`), so heading takes several
    # calls' worth of dt to settle, same as it would over several real ticks.
    for _ in range(30):
        vel = sim._steer("brain", [fly_id], pos, dt=0.1)
    assert vel[0, 0] > 0, "fly wasn't heading toward the original (far, +x) target"

    # Move the target well behind the fly's current position instead.
    sim.move_target(0, (0.2, 2.5))
    assert sim.target_moves == [(sim.t, 0, (0.2, 2.5))]

    for _ in range(30):
        vel = sim._steer("brain", [fly_id], pos, dt=0.1)
    assert vel[0, 0] < 0, "fly didn't re-route toward the moved target"

    # Capture: sitting exactly on the OLD position must not trigger a feed; sitting on the NEW one
    # must. Drive Population.feed() directly (bypassing steering) so this isolates capture, not
    # movement -- same technique test_agents.py already uses for feed()'s own live-position tests.
    pop.agents[fly_id].position = np.array([29.0, 2.5])  # the scenario's original, frozen position
    live_positions = [s.position for s in sim.odor_field.sources]
    started = pop.feed(0.1, assigned_targets=sim.brains["brain"].assigned_targets(), target_positions=live_positions)
    assert fly_id not in started, "captured at the target's old, stale position"

    pop.agents[fly_id].position = np.array([0.2, 2.5])  # where the target actually is now
    started = pop.feed(0.1, assigned_targets=sim.brains["brain"].assigned_targets(), target_positions=live_positions)
    assert fly_id in started, "not captured at the target's new, live position"


def test_target_moves_schedule_fires_from_scenario_toml_and_reaches_the_recording():
    # The declarative half of Step 0: a `[[target_moves]]` scenario schedule fires on its own,
    # with no test code calling `move_target` directly, and the fired move ends up in `sim.target_moves`
    # (what `viewer_export.export_for_viewer`'s own `target_moves` parameter consumes) with the
    # right time, index and position.
    sc = Scenario.from_dict(
        {
            "scenario": {"name": "t", "boundary_mode": "open", "seed": 0},
            "map": {"source": "gen:corridor?length=15&width=5"},
            "spawn": {"sources": [[1.0, 2.5]], "rate_per_s": 2.0, "population_cap": 1},
            "targets": [{"kind": "sugar", "position": [13.0, 2.5], "slots": 4, "feeding_time_s": 1.0}],
            "baseline": {"enabled": False},
            "target_moves": [{"at_s": 2.0, "target": 0, "to": [7.0, 2.5]}],
        }
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(0))
    for _ in range(15):  # 1.5s -- before the scheduled move
        sim.tick(0.1)
    assert sim.target_moves == []
    assert sim.odor_field.sources[0].position == (13.0, 2.5)

    for _ in range(10):  # crosses t=2.0
        sim.tick(0.1)
    assert len(sim.target_moves) == 1
    fired_t, fired_index, fired_xy = sim.target_moves[0]
    # `self.t` accumulates via repeated `+= dt`, so it lands at 2.0 only up to float error (e.g.
    # 2.0000000000000004, not exactly 2.0) -- same accumulation every other `sim.t` check in this
    # file already tolerates by construction (single-tick assertions never accumulate error);
    # `at_s` is compared with a small tolerance for the same reason.
    assert abs(fired_t - 2.0) < 1e-9
    assert (fired_index, fired_xy) == (0, (7.0, 2.5))
    assert sim.odor_field.sources[0].position == (7.0, 2.5)
    # Both brains -- not just the one that happens to be active -- must have picked it up, since
    # `move_target` loops every brain generically; check the field actually moved for the sole
    # ("brain") cohort's own TargetAssignment.
    assert tuple(sim.brains["brain"]._targets.target_positions[0]) == (7.0, 2.5)


def test_a_dense_bottleneck_keeps_agents_from_both_cohorts_reasonably_separated():
    # Cross-cohort avoidance is the whole point of running them through one shared physics call --
    # if a fly-brained and a baseline agent could walk through each other, the "same venue" claim
    # would be theatre. Overlap is expected to be nonzero at real density (physics.py's own
    # documented limitation), so this checks it stays bounded, not that it's exactly zero.
    sc = _scenario(
        population_cap=10,
        rate_per_s=3.0,
        map="gen:corridor?length=15&width=3",
        targets=[{"kind": "sugar", "position": [13.0, 1.5], "slots": 3, "feeding_time_s": 1.5}],
    )
    sim = Sim.from_scenario(sc, rng=np.random.default_rng(4))
    for _ in range(400):
        sim.tick(0.1)
        combined = [pop.positions(pop.walking_ids()) for pop in sim.cohorts.values() if pop.walking_ids()]
        if combined:
            pos = np.concatenate(combined, axis=0)
            # This scenario intentionally packs two full cohorts (20 agents combined) through a
            # 3-slot target in a 3 m-wide corridor -- denser than physics.py's own single-cohort
            # bottleneck test. Some overlap at that density is the documented, known-about
            # approximation (see physics.py's module docstring), not a bug; the bound here is
            # "stays bounded", not "stays near zero".
            assert max_overlap(pos, 0.25) < 0.3
