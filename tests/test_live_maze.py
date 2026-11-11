import numpy as np
from scipy.ndimage import label
from flybrainflow.world.live_maze import maze_map, DynamicOdorField, MAZE_FOOD, MAZE_SPAWN
from flybrainflow.world.fields import OdorSource, Wind
from flybrainflow.world.map import WalkableMap


def test_maze_has_three_gates_reachable_target_and_airflow():
    m = maze_map()
    components, _ = label(m.mask)
    points = [MAZE_SPAWN, MAZE_FOOD, *m.meta['gates']]
    ix, iy = m.world_to_cell(points)
    assert np.all(m.mask[iy, ix])
    assert len(set(components[iy, ix])) == 1
    field = DynamicOdorField(Wind(0, 1), [OdorSource('sugar', MAZE_FOOD)], m)
    assert field.airflow.has_opening
    assert np.max(np.abs(field.local_wind(m.cell_centres()))) > .1
    field.wind.set(direction_deg=90)
    assert np.max(np.abs(field.local_wind(m.cell_centres()))) > .1


def test_transport_stays_positive_and_cannot_cross_sealed_wall():
    mask = np.ones((9, 13), dtype=bool)
    mask[:, 6] = False
    m = WalkableMap.from_mask(mask, .25)
    # All border cells are walkable, leaving no declared air inlet/outlet.
    m.air_open = np.zeros_like(mask)
    field = DynamicOdorField(Wind(0, 0), [OdorSource('sugar', (.75, 1))], m)
    field.advance(5)
    c = field._selected_grid()
    assert np.all(c >= 0) and np.isfinite(c).all()
    assert np.all(c[~mask] == 0)
    assert c[:, :6].sum() > 0
    assert c[:, 7:].sum() == 0


def test_wind_snapshot_and_sensor_agree_and_reverse():
    m = maze_map()
    f = DynamicOdorField(Wind(0, 2), [OdorSource('sugar', MAZE_FOOD)], m)
    cells = m.cell_centres()
    a = f.local_wind(cells)
    snap = f.environment_snapshot()
    np.testing.assert_allclose(a[:, 0], snap['velocity_x'])
    np.testing.assert_allclose(a[:, 1], snap['velocity_y'])
    f.wind.set(direction_deg=180)
    np.testing.assert_allclose(f.local_wind(cells), -a, atol=1e-12)


def test_source_move_emits_at_new_position_preserves_old_plume():
    m = maze_map()
    source = OdorSource('sugar', MAZE_FOOD)
    f = DynamicOdorField(Wind(180, 1), [source], m)
    f.advance(.5)
    old = f.sample([MAZE_FOOD])[0]
    assert old > 0
    source.move_to(MAZE_SPAWN)
    f.advance(.1)
    assert f.sample([MAZE_SPAWN])[0] > 0
    assert f.sample([MAZE_FOOD])[0] > 0
    f.reset()
    assert f.sample([MAZE_FOOD, MAZE_SPAWN]).sum() == 0


def test_mass_conservation_in_sealed_grid_without_decay_or_sources():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:-1, 1:-1] = True
    m = WalkableMap.from_mask(mask, .25)
    m.air_open = np.zeros_like(mask)
    source = OdorSource('sugar', (1, 1), strength=0)
    f = DynamicOdorField(Wind(0, 0), [source], m, decay=0)
    f._fields[0][4, 4] = 10
    f.advance(1)
    np.testing.assert_allclose(f._fields[0].sum(), 10, atol=1e-12)
