"""Street maps -> WalkableMap.

Roads come in as centrelines (GeoJSON LineString / MultiLineString features
with an OpenStreetMap-style `highway` property) and go out as a drivable band
of the right width around each one, all unioned on the grid.

Two ways in:

    from_geojson_roads("roads.geojson", resolution=0.5)
        Needs nothing beyond the standard library. Export the file from
        Overpass Turbo, QGIS, or osmnx (`ox.save_graph_geopackage` / to_file).

    from_osm_bbox(north, south, east, west, resolution=0.5)
        Downloads live from OpenStreetMap. Needs the optional `osm` extra
        (pip install -e '.[osm]') and network access.

Coordinates in lon/lat are projected to local metres around the centre of the
data; already-metric coordinates are used as they are. The map is rebased so
its lower-left corner is (0, 0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import geom
from .map import WalkableMap

# Drivable width in metres by OSM highway class (both directions, kerb to kerb).
DEFAULT_WIDTHS: dict[str, float] = {
    "motorway": 14.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 7.0,
    "unclassified": 6.0,
    "residential": 6.0,
    "living_street": 5.0,
    "service": 4.0,
    "pedestrian": 6.0,
    "footway": 2.0,
    "path": 1.5,
    "cycleway": 2.5,
}
DEFAULT_WIDTH = 5.0


def from_geojson_roads(
    source: str | Path | dict[str, Any],
    resolution: float = 0.5,
    widths: dict[str, float] | None = None,
    default_width: float = DEFAULT_WIDTH,
    padding: float = 2.0,
    name: str = "roads",
) -> WalkableMap:
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    lines, classes = extract_lines(data)
    if not lines:
        raise ValueError("no LineString/MultiLineString features with coordinates found")
    lines, proj = to_local_metres(lines)
    widths = {**DEFAULT_WIDTHS, **(widths or {})}
    per_line_width = [float(widths.get(c, default_width)) for c in classes]
    m = rasterise_roads(lines, per_line_width, resolution=resolution, padding=padding, name=name)
    m.meta.update({"projection": proj, "n_roads": len(lines), "source": str(source) if not isinstance(source, dict) else "dict"})
    return m


def from_osm_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    resolution: float = 0.5,
    network_type: str = "drive",
    widths: dict[str, float] | None = None,
    default_width: float = DEFAULT_WIDTH,
    padding: float = 2.0,
) -> WalkableMap:
    try:
        import osmnx as ox
    except ImportError as e:  # pragma: no cover
        raise ImportError("from_osm_bbox needs the 'osm' extra: pip install -e '.[osm]'") from e
    g = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type, simplify=True)
    lines: list[np.ndarray] = []
    classes: list[str] = []
    for u, v, d in g.edges(data=True):
        if "geometry" in d:
            coords = np.array(d["geometry"].coords, float)
        else:
            coords = np.array([[g.nodes[u]["x"], g.nodes[u]["y"]], [g.nodes[v]["x"], g.nodes[v]["y"]]], float)
        hw = d.get("highway", "unclassified")
        classes.append(hw[0] if isinstance(hw, list) else str(hw))
        lines.append(coords)
    lines, proj = to_local_metres(lines)
    widths = {**DEFAULT_WIDTHS, **(widths or {})}
    per_line_width = [float(widths.get(c, default_width)) for c in classes]
    m = rasterise_roads(lines, per_line_width, resolution=resolution, padding=padding, name="osm")
    m.meta.update({"projection": proj, "bbox": (north, south, east, west), "network_type": network_type})
    return m


# ---------------------------------------------------------------------------


def extract_lines(data: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    feats = data.get("features", [data] if data.get("type") == "Feature" else [])
    lines, classes = [], []
    for f in feats:
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        hw = props.get("highway", props.get("class", "unclassified"))
        hw = hw[0] if isinstance(hw, list) else str(hw)
        if g.get("type") == "LineString":
            lines.append(np.array(g["coordinates"], float)[:, :2])
            classes.append(hw)
        elif g.get("type") == "MultiLineString":
            for part in g["coordinates"]:
                lines.append(np.array(part, float)[:, :2])
                classes.append(hw)
    return lines, classes


def to_local_metres(lines: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
    allpts = np.vstack(lines)
    if geom.looks_like_lonlat(allpts):
        lon0, lat0 = float(allpts[:, 0].mean()), float(allpts[:, 1].mean())
        lines = [geom.lonlat_to_local_m(l[:, 0], l[:, 1], lon0, lat0) for l in lines]
        return lines, {"kind": "equirectangular", "lon0": lon0, "lat0": lat0}
    return [l.copy() for l in lines], {"kind": "metric_passthrough"}


def rasterise_roads(
    lines: list[np.ndarray],
    widths: list[float],
    resolution: float,
    padding: float,
    name: str = "roads",
) -> WalkableMap:
    """Burn centrelines into a grid, then widen each by half its width with a distance transform."""
    allpts = np.vstack(lines)
    wmax = max(widths) / 2 + padding
    xmin, ymin = allpts.min(axis=0) - wmax
    xmax, ymax = allpts.max(axis=0) + wmax
    # Rebase to (0, 0)
    shift = np.array([-xmin, -ymin])
    lines = [l + shift for l in lines]
    nx = int(np.ceil((xmax - xmin) / resolution))
    ny = int(np.ceil((ymax - ymin) / resolution))
    mask = np.zeros((ny, nx), bool)

    for l, w in zip(lines, widths):
        for a, b in zip(l[:-1], l[1:]):
            _paint_segment(mask, a, b, w / 2, resolution)
    m = WalkableMap(mask=mask, resolution=resolution, origin=(0.0, 0.0), name=name)
    m.outer = []  # roads are defined by their centrelines, not rings
    m.meta = {"centrelines": [l.tolist() for l in lines], "widths": widths, "offset": (float(shift[0]), float(shift[1]))}
    return m


def _paint_segment(mask: np.ndarray, a: np.ndarray, b: np.ndarray, half_w: float, res: float) -> None:
    """Mark every cell whose centre lies within `half_w` of segment ab (exact distance, round caps).

    Only the cells in the segment's bounding box (grown by half_w) are looked at,
    so a city's worth of roads stays cheap.
    """
    ny, nx = mask.shape
    lo = np.minimum(a, b) - half_w
    hi = np.maximum(a, b) + half_w
    ix0, iy0 = max(0, int(np.floor(lo[0] / res))), max(0, int(np.floor(lo[1] / res)))
    ix1, iy1 = min(nx - 1, int(np.floor(hi[0] / res))), min(ny - 1, int(np.floor(hi[1] / res)))
    if ix1 < ix0 or iy1 < iy0:
        return
    cx = (np.arange(ix0, ix1 + 1) + 0.5) * res
    cy = (np.arange(iy0, iy1 + 1) + 0.5) * res
    X, Y = np.meshgrid(cx, cy)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        d2 = (X - a[0]) ** 2 + (Y - a[1]) ** 2
    else:
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0.0, 1.0)
        d2 = (X - (a[0] + t * ab[0])) ** 2 + (Y - (a[1] + t * ab[1])) ** 2
    mask[iy0 : iy1 + 1, ix0 : ix1 + 1] |= d2 <= half_w**2
