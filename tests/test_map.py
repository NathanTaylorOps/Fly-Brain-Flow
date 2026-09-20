"""Map loader tests. Plain functions, no fixtures, so they run under pytest or a bare runner."""

import json
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np

from flybrainflow.world import load_map
from flybrainflow.world.dxf import from_dxf
from flybrainflow.world.map import WalkableMap, bottleneck, corridor, ring, room_with_exit
from flybrainflow.world.roads import from_geojson_roads
from flybrainflow.world.svg import from_svg, parse_path, parse_transform


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_corridor_area_matches_geometry():
    m = corridor(length=40, width=5, resolution=0.1)
    assert abs(m.walkable_area_m2 - 200.0) < 2.0  # 1% tolerance for cell quantisation
    assert m.is_walkable([[20, 2.5]])[0]
    assert not m.is_walkable([[20, 7.0]])[0]
    assert not m.is_walkable([[-5, 2.5]])[0]  # outside the grid entirely


def test_corridor_has_a_wall_border():
    m = corridor(length=10, width=2, resolution=0.1, padding=0.5)
    xmin, ymin, xmax, ymax = m.bounds
    assert xmin < 0 and ymin < 0 and xmax > 10 and ymax > 2
    assert not m.mask[0, :].any() and not m.mask[-1, :].any()
    assert not m.mask[:, 0].any() and not m.mask[:, -1].any()


def test_bottleneck_gap_is_the_only_way_through():
    m = bottleneck(length=40, width=8, gap_width=1.2, gap_length=2.0, resolution=0.1)
    x = 20.0
    assert m.is_walkable([[x, 4.0]])[0]  # centre of the gap
    assert not m.is_walkable([[x, 1.0]])[0]  # inside the lower obstacle
    assert not m.is_walkable([[x, 7.0]])[0]  # inside the upper obstacle
    assert m.is_walkable([[10.0, 1.0]])[0]  # wide part is open
    column = m.mask[:, m.world_to_cell([[x, 0]])[0][0]]
    assert abs(column.sum() * m.resolution - 1.2) < 0.25


def test_ring_is_an_annulus():
    m = ring(radius=36.6, lane_width=3.5, resolution=0.25)
    assert m.is_walkable([[36.6, 0.0]])[0]
    assert not m.is_walkable([[0.0, 0.0]])[0]
    assert not m.is_walkable([[50.0, 0.0]])[0]
    expected = math.pi * ((36.6 + 1.75) ** 2 - (36.6 - 1.75) ** 2)
    assert abs(m.walkable_area_m2 - expected) / expected < 0.02
    assert abs(m.meta["circumference"] - 230.0) < 0.1


def test_room_with_exit_has_a_door():
    m = room_with_exit(width=20, height=15, door_width=1.5, stub_length=4, resolution=0.1)
    assert m.is_walkable([[10, 7.5]])[0]
    assert m.is_walkable([[22, 7.5]])[0]  # in the stub, through the door
    assert not m.is_walkable([[22, 2.0]])[0]  # outside the room, not in the stub


# ---------------------------------------------------------------------------
# Coordinates and fields
# ---------------------------------------------------------------------------


def test_world_cell_roundtrip():
    m = corridor(length=10, width=4, resolution=0.2)
    pts = np.array([[0.05, 0.05], [3.33, 1.11], [9.99, 3.99]])
    ix, iy = m.world_to_cell(pts)
    back = m.cell_to_world(ix, iy)
    assert np.all(np.abs(back - pts) <= m.resolution / 2 + 1e-9)


def test_distance_to_wall_peaks_at_half_width():
    m = corridor(length=20, width=4, resolution=0.1)
    d = m.distance_at([[10.0, 2.0], [10.0, 0.05]])
    assert abs(d[0] - 2.0) < 0.15
    assert d[1] < 0.15
    assert m.distance_at([[10.0, 9.0]])[0] == 0.0


def test_nearest_walkable_projects_inside():
    m = corridor(length=20, width=4, resolution=0.1)
    outside = np.array([[10.0, 6.0], [-3.0, 2.0], [10.0, 2.0]])
    p = m.nearest_walkable(outside)
    assert m.is_walkable(p).all()
    assert np.allclose(p[2], outside[2])  # already-legal point untouched
    assert abs(p[0][0] - 10.0) < 0.2 and p[0][1] < 4.0


def test_save_load_roundtrip():
    m = bottleneck(resolution=0.2)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "m.npz"
        m.save(f)
        m2 = WalkableMap.load(f)
    assert np.array_equal(m.mask, m2.mask)
    assert m2.resolution == m.resolution and m2.origin == m.origin
    assert m2.meta["gap_width"] == m.meta["gap_width"]
    assert len(m2.holes) == 2


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

SVG = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="20000mm" height="10000mm" viewBox="0 0 20000 10000">
  <g inkscape:label="walkable">
    <rect x="0" y="0" width="20000" height="10000"/>
  </g>
  <g id="walls" transform="translate(5000 2000)">
    <path d="M 0 0 h 2000 v 3000 h -2000 z"/>
    <circle cx="8000" cy="4000" r="1000"/>
  </g>
</svg>
"""


def test_svg_units_flip_and_obstacles():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "plan.svg"
        f.write_text(SVG)
        m = from_svg(f, resolution=0.1)
    assert abs(m.meta["units_to_m"] - 0.001) < 1e-12  # 20000 units over 20000 mm
    assert abs(m.width_m - 21.0) < 0.2 and abs(m.height_m - 11.0) < 0.2  # 20 x 10 m + padding
    # Rect obstacle: SVG (5000..7000, 2000..5000) mm, y flipped: world y = 10 - y_svg
    assert not m.is_walkable([[6.0, 6.5]])[0]
    assert m.is_walkable([[6.0, 2.0]])[0]
    # Circle obstacle at SVG (13000, 6000) r 1000 -> world (13, 4)
    assert not m.is_walkable([[13.0, 4.0]])[0]
    assert m.is_walkable([[13.0, 5.5]])[0]
    expected = 200.0 - 6.0 - math.pi * 1.0**2
    assert abs(m.walkable_area_m2 - expected) < 1.5


def test_svg_unlabelled_shapes_are_walkable_with_warning():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 5"><polygon points="0,0 10,0 10,5 0,5"/></svg>'
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "u.svg"
        f.write_text(svg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = from_svg(f, resolution=0.1, units_to_m=1.0)
    assert any("walkable" in str(x.message) for x in w)
    assert abs(m.walkable_area_m2 - 50.0) < 1.0


def test_svg_path_parser_handles_relative_and_arcs():
    rings = parse_path("M10 10 l 20 0 l 0 20 l -20 0 z", 8)
    assert len(rings) == 1 and abs(_area(rings[0]) - 400) < 1e-6
    # A full circle drawn as two arcs
    rings = parse_path("M 0 0 A 5 5 0 0 1 10 0 A 5 5 0 0 1 0 0 Z", 8)
    assert len(rings) == 1 and abs(_area(rings[0]) - math.pi * 25) / (math.pi * 25) < 0.01
    # Cubic curve closes back
    rings = parse_path("M0 0 C 0 10 10 10 10 0 Z", 16)
    assert len(rings) == 1 and _area(rings[0]) > 50


def test_svg_transform_composition():
    m = parse_transform("translate(10 5) scale(2)")
    from flybrainflow.world.geom import apply

    p = apply(m, [[1, 1]])[0]
    assert np.allclose(p, [12, 7])


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------


def _dxf(entities: str, insunits: int = 6) -> str:
    return (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n%d\n0\nENDSEC\n" % insunits
        + "0\nSECTION\n2\nENTITIES\n" + entities + "0\nENDSEC\n0\nEOF\n"
    )


def test_dxf_lwpolyline_lines_and_bulge():
    # Walkable outline as four separate LINEs (must be chained), an obstacle as a closed
    # LWPOLYLINE with one bulge (a semicircular end), units in inches.
    lines = ""
    box = [(0, 0), (400, 0), (400, 200), (0, 200)]
    for (x0, y0), (x1, y1) in zip(box, box[1:] + box[:1]):
        lines += f"0\nLINE\n8\nWALKABLE\n10\n{x0}\n20\n{y0}\n11\n{x1}\n21\n{y1}\n"
    poly = (
        "0\nLWPOLYLINE\n8\nWALLS\n90\n4\n70\n1\n"
        "10\n100\n20\n50\n"
        "10\n150\n20\n50\n42\n1.0\n"  # bulge=1 -> semicircle from (150,50) to (150,150)
        "10\n150\n20\n150\n"
        "10\n100\n20\n150\n"
    )
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "plan.dxf"
        f.write_text(_dxf(lines + poly, insunits=1))
        m = from_dxf(f, resolution=0.05)
    inch = 0.0254
    assert abs(m.meta["units_to_m"] - inch) < 1e-12
    assert abs(m.width_m - (400 * inch + 1.0)) < 0.1
    # Inside the rectangular part of the obstacle
    assert not m.is_walkable([[120 * inch, 100 * inch]])[0]
    # Inside the bulged semicircle (centre (150,100), r=50): point (180,100) is inside
    assert not m.is_walkable([[180 * inch, 100 * inch]])[0]
    # Outside the bulge but inside the room
    assert m.is_walkable([[230 * inch, 100 * inch]])[0]
    expected = (400 * 200 - (50 * 100 + math.pi * 50**2 / 2)) * inch**2
    assert abs(m.walkable_area_m2 - expected) / expected < 0.02


def test_dxf_unclosed_chain_is_ignored_with_warning():
    lines = "0\nLINE\n8\nWALKABLE\n10\n0\n20\n0\n11\n10\n21\n0\n0\nLINE\n8\nWALKABLE\n10\n10\n20\n0\n11\n10\n21\n5\n"
    circle = "0\nCIRCLE\n8\nWALKABLE\n10\n50\n20\n50\n40\n10\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "open.dxf"
        f.write_text(_dxf(lines + circle))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = from_dxf(f, resolution=0.2)
    assert any("never closed" in str(x.message) for x in w)
    assert abs(m.walkable_area_m2 - math.pi * 100) / (math.pi * 100) < 0.03


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def test_geojson_roads_widths_and_projection():
    lat0, lon0 = -27.47, 153.02
    dlat = 100 / 111_000  # ~100 m north-south
    dlon = 100 / (111_000 * math.cos(math.radians(lat0)))  # ~100 m east-west
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"highway": "primary"}, "geometry": {"type": "LineString", "coordinates": [[lon0 - dlon, lat0], [lon0 + dlon, lat0]]}},
            {"type": "Feature", "properties": {"highway": "footway"}, "geometry": {"type": "LineString", "coordinates": [[lon0, lat0 - dlat], [lon0, lat0 + dlat]]}},
        ],
    }
    m = from_geojson_roads(data, resolution=0.5)
    assert m.meta["projection"]["kind"] == "equirectangular"
    assert abs(m.width_m - (200 + 2 * (10 / 2 + 2))) < 3  # 200 m road + half-width + padding each side
    # Primary road is 10 m wide; footway 2 m. Sample cross-sections away from the junction.
    cx, cy = m.width_m / 2, m.height_m / 2
    col = m.mask[:, m.world_to_cell([[cx + 60, 0]])[0][0]]
    assert abs(col.sum() * m.resolution - 10.0) < 1.5
    row = m.mask[m.world_to_cell([[0, cy + 60]])[1][0], :]
    assert abs(row.sum() * m.resolution - 2.0) < 1.5
    assert m.is_walkable([[cx, cy]])[0]  # the junction


def test_geojson_metric_passthrough():
    data = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"highway": "service"}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 0]]}}]}
    m = from_geojson_roads(data, resolution=0.5)
    assert m.meta["projection"]["kind"] == "metric_passthrough"
    assert abs(m.walkable_area_m2 - 100 * 4) / 400 < 0.1


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_load_map_generator_string():
    m = load_map("gen:corridor?length=12&width=3", resolution=0.1)
    assert m.name == "corridor" and abs(m.walkable_area_m2 - 36) < 1
    m = load_map("gen:ring?radius=10&lane_width=2", resolution=0.25)
    assert m.name == "ring"


def test_load_map_rejects_unknown():
    for bad in ("gen:hexagon", "plan.xyz"):
        try:
            load_map(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have raised")


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
