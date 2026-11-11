import numpy as np
from scipy.spatial import cKDTree
from flybrainflow.world.arena_world import ArenaWorld


def test_every_exit_reachable_from_spawn_cells():
    world = ArenaWorld()
    points = world.spawn_points(500, seed=12)
    for exit_ in world.exits:
        direction, chosen = world.navigation(points, [exit_['id']])
        assert np.all(chosen == exit_['id'])
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1)


def test_spawn_separation_reproducibility_and_wall_clearance():
    world = ArenaWorld()
    p = world.spawn_points(500, seed=3)
    np.testing.assert_array_equal(p, world.spawn_points(500, seed=3))
    assert world.map.is_walkable(p).all()
    assert np.min(world.map.distance_at(p)) >= .65
    distance = cKDTree(p).query(p, k=2)[0][:, 1]
    assert distance.min() >= .48


def test_closed_exits_excluded_and_all_closed_stops_routing():
    w = ArenaWorld()
    w.set_exit('east', False)
    d, chosen = w.navigation([[56,24]], ['east','west'])
    assert chosen[0] == 'west'
    assert np.linalg.norm(d[0]) > 0
    d, chosen = w.navigation([[32,24]], [])
    assert not d.any() and chosen[0] == ''
    assert w.crossed_exit([[60.1,24]])[0] == ''


def test_exit_crossing_is_not_proximity_capture():
    w = ArenaWorld()
    result = w.crossed_exit([[59.9,24],[60.1,24],[61,30],[32,44.1],[32,3.9],[3.9,24]])
    assert result.tolist() == ['', 'east', '', 'north', 'south', 'west']


def test_single_agent_routes_around_obstacles_to_every_exit():
    w = ArenaWorld()
    # Points on opposite sides of stage/seating require an actual routed detour.
    for ident, start in [('north',[32.,30.]), ('east',[18.,24.]),
                         ('west',[46.,24.]), ('south',[32.,39.])]:
        p = np.array([start])
        for _ in range(2000):
            d, _ = w.navigation(p, [ident])
            p += d * .1
            assert w.map.is_walkable(p)[0]
            if w.crossed_exit(p, [ident])[0]:
                break
        else:
            raise AssertionError(f'No escape through {ident}')


def test_hazard_is_advected_and_snapshot_matches_sensor():
    w = ArenaWorld()
    w.set_hazard([32,24], enabled=True, direction_deg=0, speed_mps=1)
    w.advance(2)
    assert w.hazard.sample([[32,24]])[0] > 0
    assert w.hazard.sample([[34,24]])[0] > w.hazard.sample([[30,24]])[0]
    snapshot = w.environment_snapshot()
    np.testing.assert_allclose(w.hazard.sample(w.map.cell_centres()), snapshot['concentration'])
    assert np.array(snapshot['concentration'])[~w.map.mask.ravel()].sum() == 0


def test_hazard_penalty_can_redirect_agents_away_from_exposed_exit():
    w = ArenaWorld()
    p = [[56,24]]
    assert w.navigation(p)[1][0] == 'east'
    w.set_hazard([57,24], strength=20)
    w.advance(30)
    assert w.navigation(p, hazard_weight=50)[1][0] != 'east'
