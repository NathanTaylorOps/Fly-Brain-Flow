"""Agent lifecycle (spawn / feed / leave) tests."""

import numpy as np

from flybrainflow.agents import IdAllocator, Population, TargetSlots, build_cohorts
from flybrainflow.scenario import Scenario


def _scenario(**overrides):
    raw = {
        "scenario": {"name": "t", "boundary_mode": overrides.pop("boundary_mode", "open")},
        "map": {"source": "gen:corridor"},
        "spawn": {
            "sources": overrides.pop("sources", [[1.0, 1.0]]),
            "rate_per_s": overrides.pop("rate_per_s", 1.0),
            "population_cap": overrides.pop("population_cap", 100),
        },
        "targets": overrides.pop(
            "targets", [{"position": [5.0, 1.0], "slots": 6, "feeding_time_s": 2.0}]
        ),
        "agents": overrides.pop("agents", {}),
        "baseline": overrides.pop("baseline", {}),
    }
    return Scenario.from_dict(raw)


def test_closed_boundary_scenarios_are_rejected():
    sc = _scenario(boundary_mode="closed", sources=[], targets=[])
    try:
        Population.from_scenario(sc)
    except ValueError as e:
        assert "closed" in str(e) or "open" in str(e)
    else:
        raise AssertionError("expected a ValueError for a closed-mode scenario")


def test_spawn_respects_rate_and_is_capped():
    sc = _scenario(rate_per_s=2.0, population_cap=3)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    # A big dt would want to spawn 20 agents at rate 2/s -- the cap must win.
    new_ids = pop.spawn(dt=10.0)
    assert len(new_ids) == 3
    assert len(pop.agents) == 3
    # Cap is full: another spawn call adds nobody more.
    assert pop.spawn(dt=10.0) == []
    assert len(pop.agents) == 3


def test_spawn_accumulates_fractional_dt():
    sc = _scenario(rate_per_s=1.0, population_cap=100)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    # Half a fly's worth of spawn budget shouldn't produce anyone yet.
    assert pop.spawn(dt=0.5) == []
    assert len(pop.agents) == 0
    # The other half tips it over into one new agent.
    assert len(pop.spawn(dt=0.5)) == 1
    assert len(pop.agents) == 1


def test_spawn_backlog_does_not_build_up_while_capped():
    # Real bug, fixed: the fractional accumulator used to keep growing every tick the cap was
    # blocking spawns, so a long stretch at capacity built an unbounded backlog that cashed in
    # as one instant burst the moment any slot freed -- refilling the cap on the very next tick
    # regardless of how small that tick's dt was. That breaks the whole point of a population
    # cap (arrivals should stay rate-limited even once a slot frees up).
    sc = _scenario(rate_per_s=2.0, population_cap=3)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    pop.spawn(dt=10.0)  # hits the cap immediately
    for _ in range(50):
        pop.spawn(dt=10.0)  # capped for a long stretch -- must not accumulate unbounded debt
    assert pop._spawn_accumulator == 0.0

    # Free one slot, then take a tiny tick: it should NOT refill instantly.
    freed_id = next(iter(pop.agents))
    pop.agents[freed_id].status = "feeding"
    pop.agents[freed_id].feed_remaining_s = 0.0
    pop.agents[freed_id].target_index = 0
    pop.slots_used[0] = 1
    pop.leave()
    assert len(pop.agents) == 2
    assert pop.spawn(dt=0.001) == []  # far too little budget for a new arrival yet
    assert len(pop.agents) == 2


def test_spawned_agents_start_at_a_declared_source():
    sc = _scenario(sources=[[1.0, 1.0], [9.0, 1.0]], rate_per_s=100.0, population_cap=20)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    pop.spawn(dt=1.0)
    positions = pop.positions()
    sources = np.array(sc.spawn.sources)
    for p in positions:
        assert any(np.allclose(p, s) for s in sources)


def test_personality_assignment_uniform20_and_fallback():
    sc = _scenario(rate_per_s=100.0, population_cap=50)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    pop.spawn(dt=1.0)
    personalities = [a.personality for a in pop.agents.values()]
    assert all(0 <= p < 20 for p in personalities)

    sc2 = _scenario(rate_per_s=100.0, population_cap=50, agents={"personality_mix": "something_else"})
    pop2 = Population.from_scenario(sc2, rng=np.random.default_rng(0))
    pop2.spawn(dt=1.0)
    assert all(a.personality == 0 for a in pop2.agents.values())


def test_set_positions_moves_walking_and_ignores_feeding():
    sc = _scenario(rate_per_s=100.0, population_cap=2)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    ids = pop.spawn(dt=1.0)
    assert len(ids) == 2
    pop.agents[ids[1]].status = "feeding"  # pretend it's already feeding
    stray_target = np.array([99.0, 99.0])
    pop.set_positions(ids, np.array([[2.0, 2.0], stray_target]))
    assert np.allclose(pop.agents[ids[0]].position, [2.0, 2.0])
    assert not np.allclose(pop.agents[ids[1]].position, stray_target)


def test_feed_occupies_a_slot_and_blocks_once_full():
    sc = _scenario(
        rate_per_s=100.0,
        population_cap=2,
        targets=[{"position": [5.0, 1.0], "slots": 1, "feeding_time_s": 2.0}],
    )
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    ids = pop.spawn(dt=1.0)
    # Put both agents right on top of the (single-slot) target.
    pop.set_positions(ids, np.array([[5.0, 1.0], [5.0, 1.0]]))
    started = pop.feed(dt=0.0, capture_radius=0.5)
    assert len(started) == 1
    fed_id, blocked_id = started[0], [i for i in ids if i not in started][0]
    assert pop.agents[fed_id].status == "feeding"
    assert pop.agents[blocked_id].status == "walking"
    assert pop.slots_used == [1]


def test_feed_completes_after_feeding_time_and_leave_frees_the_slot():
    sc = _scenario(
        rate_per_s=100.0,
        population_cap=1,
        targets=[{"position": [5.0, 1.0], "slots": 1, "feeding_time_s": 1.0}],
    )
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    ids = pop.spawn(dt=1.0)
    pop.set_positions(ids, np.array([[5.0, 1.0]]))
    pop.feed(dt=0.0, capture_radius=0.5)
    assert pop.agents[ids[0]].status == "feeding"

    # Not done yet after half the feeding time.
    pop.feed(dt=0.5)
    assert pop.leave() == []
    assert pop.agents[ids[0]].status == "feeding"

    # The other half finishes it.
    pop.feed(dt=0.5)
    left = pop.leave()
    assert left == ids
    assert ids[0] not in pop.agents
    assert pop.slots_used == [0]


def test_freed_slot_can_be_reused():
    sc = _scenario(
        rate_per_s=100.0,
        population_cap=2,
        targets=[{"position": [5.0, 1.0], "slots": 1, "feeding_time_s": 1.0}],
    )
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    a, b = pop.spawn(dt=1.0)
    pop.set_positions([a], np.array([[5.0, 1.0]]))
    pop.feed(dt=0.0, capture_radius=0.5)
    assert pop.agents[a].status == "feeding" and pop.agents[b].status == "walking"

    pop.feed(dt=1.0)  # finishes agent a
    pop.leave()
    assert pop.slots_used == [0]

    # Now b can take the freed slot.
    pop.set_positions([b], np.array([[5.0, 1.0]]))
    started = pop.feed(dt=0.0, capture_radius=0.5)
    assert started == [b]
    assert pop.slots_used == [1]


def test_a_lone_population_defaults_to_the_brain_tag_and_owns_its_own_state():
    sc = _scenario(rate_per_s=100.0, population_cap=1)
    pop = Population.from_scenario(sc, rng=np.random.default_rng(0))
    ids = pop.spawn(dt=1.0)
    assert pop.agents[ids[0]].brain == "brain"


def test_build_cohorts_makes_two_pools_when_baseline_is_enabled():
    sc = _scenario(rate_per_s=100.0, population_cap=5, baseline={"enabled": True})
    cohorts = build_cohorts(sc, rng=np.random.default_rng(0))
    assert set(cohorts) == {"brain", "baseline"}
    for tag, pop in cohorts.items():
        pop.spawn(dt=1.0)
        assert all(a.brain == tag for a in pop.agents.values())


def test_build_cohorts_makes_one_pool_when_baseline_is_disabled():
    sc = _scenario(rate_per_s=100.0, population_cap=5, baseline={"enabled": False})
    cohorts = build_cohorts(sc, rng=np.random.default_rng(0))
    assert set(cohorts) == {"brain"}


def test_cohorts_share_agent_ids_without_colliding():
    # Real gap, fixed: two independent `Population`s would each number their own agents from
    # zero, so "agent 3" would mean two different flies depending which cohort's dict you looked
    # it up in -- a problem the moment anything (a brain's per-agent state, a combined physics
    # call) needs to index both cohorts together.
    sc = _scenario(rate_per_s=100.0, population_cap=10, baseline={"enabled": True})
    cohorts = build_cohorts(sc, rng=np.random.default_rng(0))
    cohorts["brain"].spawn(dt=1.0)
    cohorts["baseline"].spawn(dt=1.0)
    brain_ids = set(cohorts["brain"].agents)
    baseline_ids = set(cohorts["baseline"].agents)
    assert brain_ids.isdisjoint(baseline_ids)
    assert len(brain_ids) == 10 and len(baseline_ids) == 10


def test_cohorts_share_target_slots_instead_of_each_getting_the_full_capacity():
    # Real gap, fixed: two independent `Population`s would each track a target's slots for
    # themselves, so a single-slot target could be double-booked -- one fly from each cohort,
    # both "fed", when physically only one fly fits.
    sc = _scenario(
        rate_per_s=100.0,
        population_cap=1,
        targets=[{"position": [5.0, 1.0], "slots": 1, "feeding_time_s": 2.0}],
        baseline={"enabled": True},
    )
    cohorts = build_cohorts(sc, rng=np.random.default_rng(0))
    brain_ids = cohorts["brain"].spawn(dt=1.0)
    baseline_ids = cohorts["baseline"].spawn(dt=1.0)
    cohorts["brain"].set_positions(brain_ids, np.array([[5.0, 1.0]]))
    cohorts["baseline"].set_positions(baseline_ids, np.array([[5.0, 1.0]]))

    started_brain = cohorts["brain"].feed(dt=0.0, capture_radius=0.5)
    started_baseline = cohorts["baseline"].feed(dt=0.0, capture_radius=0.5)
    # Whichever cohort fed first claimed the target's one slot; the other must have been turned
    # away, not double-booked into a second, nonexistent slot.
    assert len(started_brain) + len(started_baseline) == 1
    assert cohorts["brain"].slots_used == cohorts["baseline"].slots_used  # the same shared list


def test_target_slots_and_id_allocator_can_still_be_built_and_passed_by_hand():
    # `build_cohorts` is the normal path, but nothing stops wiring two `Population`s together
    # manually with the same effect -- useful to pin down the contract these two small helper
    # classes offer on their own, decoupled from `build_cohorts`'s own defaults.
    sc = _scenario(rate_per_s=100.0, population_cap=3)
    slots = TargetSlots(sc.targets)
    ids = IdAllocator(start=100)
    a = Population.from_scenario(sc, rng=np.random.default_rng(0), brain="a", slots=slots, ids=ids)
    b = Population.from_scenario(sc, rng=np.random.default_rng(1), brain="b", slots=slots, ids=ids)
    a.spawn(dt=1.0)
    b.spawn(dt=1.0)
    assert min(a.agents) >= 100
    assert set(a.agents).isdisjoint(b.agents)
    assert a.slots_used is b.slots_used
