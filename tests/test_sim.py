"""Sim loop tests: the first place `agents.py`, `brains/` and `world/` actually run together
end to end, across a full tick (spawn -> steer -> move -> feed -> leave) and across more than one
cohort sharing a venue.
"""

import numpy as np

from flybrainflow.scenario import Scenario
from flybrainflow.sim import Sim
from flybrainflow.world import max_overlap


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


def test_source_targets_makes_two_streams_actually_cross_the_corridor():
    # Real bug, fixed: corridor_bidirectional.toml's two sources sat right next to a target each
    # (a "two-stream lane test"), so nearest-distance assignment always sent every fly straight
    # back to its own end -- nobody ever crossed. `source_targets` pins each source to the FAR
    # target instead. This is the sim-level proof it actually works: flies born at x=1 should be
    # assigned target 0 (at x=39, per this scenario's source_targets) and get moving in the +x
    # direction, and vice versa for flies born at x=39 -- and, over a real run, a growing number of
    # agents should be genuinely present in the middle of the corridor (which never happened
    # before this fix; see docs/JOURNAL.md).
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
