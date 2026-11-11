import numpy as np
import pytest

from flybrainflow.arena import ArenaSim


def test_seeded_independent_agents_and_reaction_delays():
    a, b = ArenaSim(population=50, seed=4), ArenaSim(population=50, seed=4)
    assert np.std(a.preferred_speed) > .05
    assert np.std(a.reaction_delay) > .1
    a.control({'alarm': True}); b.control({'alarm': True})
    a.step(10); b.step(10)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert set(a.agent_status) == {'wandering', 'evacuating'}
    assert not np.shares_memory(a.positions, b.positions)


def test_default_crowd_evacuates_by_crossing_exits_and_does_not_respawn():
    sim = ArenaSim(population=200, seed=7)
    sim.control({'alarm': True})
    for _ in range(800):
        previous = sim.positions.copy()
        active_before = sim.active.copy()
        sim._advance(.05)
        # Check intervening segments, including the final step of evacuated disks.
        points = previous[active_before, None, :] + (sim.positions[active_before]-previous[active_before])[:, None, :] * np.linspace(0, 1, 8)[None, :, None]
        assert sim.world.map.is_walkable(points.reshape(-1, 2)).all()
        assert sim._max_overlap < .02
        if not sim.active.any():
            break
    assert not sim.active.any()
    assert sum(sim.evacuated_by_exit.values()) == 200
    sim.step(20)
    assert sim.state()['remaining'] == 0
    assert sim.state()['phase'] == 'complete'


def test_closed_exit_physically_blocks_and_all_closed_stays_inside():
    sim = ArenaSim(population=50)
    for e in sim.world.exits:
        sim.control({'exit': {'id': e['id'], 'open': False}})
    sim.positions[0] = [59.5, 24]
    sim.velocities[0] = [20., 0]
    sim.control({'alarm': True})
    sim.step(100, dt=.05)
    assert sim.active.all()
    assert sim.positions[0, 0] <= 60-sim.radius
    assert sim.world.map.is_walkable(sim.positions).all()


def test_closing_gate_with_agent_touching_it_returns_agent_inward():
    sim = ArenaSim(population=50)
    sim.positions[0] = [59.9, 24]
    sim.control({'exit': {'id': 'east', 'open': False}})
    assert sim.positions[0, 0] < 60-sim.radius
    assert sim._safe(sim.positions[:1])[0]
    sim.control({'alarm': True})
    sim.step(100)
    assert sim.positions[0, 0] < 59.5


def test_wall_sweep_and_overlapping_contacts_remain_walkable():
    sim = ArenaSim(population=50)
    sim.positions[:2] = [[25.2, 34], [24.9, 34]]
    sim.velocities[:2] = [[20, 0], [20, 0]]
    for _ in range(10):
        previous = sim.positions.copy()
        sim._advance(.1)
        points = previous[:, None, :] + (sim.positions-previous)[:, None, :] * np.linspace(0, 1, 12)[None, :, None]
        assert sim.world.map.is_walkable(points.reshape(-1, 2)).all()
    assert np.linalg.norm(sim.positions[0]-sim.positions[1]) > .47


def test_invalid_control_is_atomic_and_reset_is_reproducible():
    sim = ArenaSim(population=50)
    before = sim.state()
    with pytest.raises(ValueError):
        sim.control({'alarm': True, 'exit': {'id': 'missing', 'open': False}})
    assert sim.alarm_time is None
    assert sim.state()['config'] == before['config']
    with pytest.raises(ValueError):
        sim.reset({'population': 501})
    assert sim.state()['population'] == 50
    start = sim.positions.copy()
    sim.step(20)
    sim.reset()
    np.testing.assert_array_equal(sim.positions, start)


def test_hazard_controls_exposure_and_model_label():
    sim = ArenaSim(population=50)
    sim.control({'hazard_position': sim.positions[0].tolist(), 'hazard_strength': 2})
    sim.step(10)
    assert sim.exposure[0] > 0
    assert sim.environment()['hazard']['enabled']
    state = sim.state()
    assert not state['capabilities']['neural_simulation']
    assert state['metrics']['peak_local_density'] > 0
    assert state['model']['label'] == 'Behavioral crowd baseline'


def test_gathering_keeps_initial_population_for_100_seconds():
    sim = ArenaSim(population=100)
    # An agent already approaching an open exit must also remain inside.
    sim.positions[0] = [59.5, 24]
    sim.headings[0] = 0.
    sim.wander_phase[0] = 0.
    for _ in range(1000):
        sim._advance(.1)
    state = sim.state()
    assert state['remaining'] == 100
    assert state['evacuated'] == 0
    assert state['phase'] == 'calm'
    assert sim._safe(sim.positions).all()


def test_calm_agents_avoid_local_repellent_without_alarm():
    clean, cue = ArenaSim(population=50, seed=11), ArenaSim(population=50, seed=11)
    for sim in (clean, cue):
        sim.positions[0] = [33., 24.]
        sim.headings[0] = np.pi
        sim.wander_phase[0] = 0.
        sim.control({'wind_speed_mps': 0})
    cue.control({'hazard_position': [32., 24.], 'hazard_strength': 3})
    cue.world.advance(5)  # Build a measured local plume before the paired trial.
    for _ in range(10):
        clean._advance(.05)
        cue._advance(.05)
    assert cue.positions[0, 0] > clean.positions[0, 0] + .1
    assert cue.velocities[0, 0] > clean.velocities[0, 0] + .1
    assert cue.agent_status[0] == 'avoiding'
    assert cue.alarm_time is None and cue.active.all()
    assert not cue.state()['capabilities']['learning']
