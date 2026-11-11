import numpy as np
from flybrainflow.brains.forager import ForagerBrain
from flybrainflow.world.map import corridor
from flybrainflow.world.fields import OdorField, OdorSource, Wind
from flybrainflow.world.physics import step


def _walk(brain, odor, map_, start, target, seconds=60, dt=.05):
    position = np.array([start], float)
    trajectory = []
    for tick in range(round(seconds / dt)):
        if hasattr(odor, 'advance'):
            odor.advance(dt)
        vel = brain.desired_velocities([1], position, [.15], 1.3,
                    personalities=[0], odor_field=odor, dt=dt)
        position, _ = step(position, vel, [.15], dt, map_, max_speed_mps=1.3)
        assert map_.is_walkable(position).all()
        trajectory.append(position[0].copy())
        if np.linalg.norm(position[0] - target) < .45:
            return True, (tick+1)*dt, np.array(trajectory)
    return False, seconds, np.array(trajectory)


def test_local_odor_reaches_source_across_seeds():
    map_ = corridor(length=12, width=8, resolution=.2)
    for seed in (3, 7, 42):
        source = OdorSource('sugar', (10., 4.), range_m=15)
        odor = OdorField(Wind(180, .3), [source], walkable_map=map_)
        brain = ForagerBrain(map_, seed=seed)
        success, elapsed, _ = _walk(brain, odor, map_, (2, 4), source.position)
        assert success, (seed, elapsed)
        assert brain.behavioral_snapshot()['controller'] == 'behavioral'


def test_moved_odor_recovers_without_target_notification():
    map_ = corridor(length=12, width=8, resolution=.2)
    source = OdorSource('sugar', (10., 4.), range_m=15)
    odor = OdorField(Wind(180, .3), [source], walkable_map=map_)
    brain = ForagerBrain(map_, seed=7)
    success, _, path = _walk(brain, odor, map_, (2, 4), source.position)
    assert success
    source.move_to((2., 6.))
    # No move_target or coordinates sent to the controller.
    success, _, _ = _walk(brain, odor, map_, path[-1], source.position)
    assert success


def test_no_odor_explores_and_unsupported_neural_controls_absent():
    map_ = corridor(length=12, width=8, resolution=.2)
    odor = OdorField(Wind(), [], walkable_map=map_)
    brain = ForagerBrain(map_, seed=3)
    _, _, path = _walk(brain, odor, map_, (2, 4), (-100, -100), seconds=10)
    assert np.linalg.norm(path[-1] - path[0]) > 1
    assert brain.control_capabilities == {'odor': True, 'stimulus': False, 'silence': False}
    brain.forget([1])
    assert brain.behavioral_snapshot()['agents'] == {}


def test_dynamic_maze_reaches_food_across_seeds_through_actual_gates():
    from flybrainflow.world.live_maze import maze_map, DynamicOdorField, MAZE_FOOD, MAZE_SPAWN
    m = maze_map()
    for seed in (3, 7, 42):
        odor = DynamicOdorField(Wind(180, 1), [OdorSource('sugar', MAZE_FOOD)], m)
        odor.advance(30)  # Explicit environmental plume warmup, not agent pretraining.
        brain = ForagerBrain(m, seed=seed)
        success, elapsed, path = _walk(brain, odor, m, MAZE_SPAWN, MAZE_FOOD, seconds=90, dt=.1)
        assert success, (seed, elapsed, path[-1])
        # Every separating wall is crossed at its opening, not through an obstacle.
        for wall_x, low, high in ((4, 2, 4), (8, 8, 10), (12, 3, 5)):
            at_gate = path[np.abs(path[:, 0] - wall_x) < .15]
            assert len(at_gate) > 0
            assert np.all((at_gate[:, 1] >= low-.05) & (at_gate[:, 1] <= high+.05))


def test_dynamic_plume_move_and_wind_reversal_recovers():
    from flybrainflow.world.live_maze import maze_map, DynamicOdorField, MAZE_FOOD, MAZE_SPAWN
    m = maze_map()
    source = OdorSource('sugar', MAZE_FOOD)
    odor = DynamicOdorField(Wind(180, 1), [source], m)
    odor.advance(30)
    brain = ForagerBrain(m, seed=7)
    success, _, path = _walk(brain, odor, m, MAZE_SPAWN, MAZE_FOOD, seconds=90, dt=.1)
    assert success
    source.move_to(MAZE_SPAWN)
    odor.wind.set(direction_deg=0)
    # The old plume persists; the controller is not told a new target coordinate.
    success, elapsed, path = _walk(brain, odor, m, path[-1], MAZE_SPAWN, seconds=150, dt=.1)
    assert success, (elapsed, path[-1])


def test_actual_live_maze_factory_enters_feeding():
    from flybrainflow.live_demo import maze_sim
    for seed in (3, 7, 42):
        sim = maze_sim(seed)
        reached = False
        for _ in range(1200):
            sim.tick(.05)
            agents = sim.cohorts['brain'].agents.values()
            assert all(sim.map.is_walkable([a.position])[0] for a in agents)
            if any(a.status == 'feeding' for a in agents):
                reached = True
                break
        assert reached, seed
