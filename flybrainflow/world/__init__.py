"""The world: maps now; odour, wind and physics as M0 continues.

    from flybrainflow.world import load_map
    m = load_map("gen:corridor?length=40&width=5", resolution=0.1)
    m = load_map("venues/lobby.svg", resolution=0.1, units_to_m=0.001)
    m = load_map("roads.geojson", resolution=0.5)
    m = load_map("osm:-27.46,-27.47,153.03,153.02", resolution=0.5)   # north,south,east,west
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .map import GENERATORS, WalkableMap

__all__ = ["WalkableMap", "load_map", "load_map_for_scenario"]


def load_map(source: str, resolution: float = 0.1, units_to_m: float | None = None, base_dir: str | Path | None = None, **options: Any) -> WalkableMap:
    """Turn a scenario `map.source` string into a WalkableMap."""
    src = source.strip()
    if src.startswith("gen:"):
        spec = urlsplit(src[4:])
        name = spec.path
        if name not in GENERATORS:
            raise ValueError(f"unknown generator '{name}'; choose from {sorted(GENERATORS)}")
        kwargs = {k: float(v) for k, v in parse_qsl(spec.query)}
        kwargs.update(options)
        return GENERATORS[name](resolution=resolution, **kwargs)
    if src.startswith("osm:"):
        from .roads import from_osm_bbox

        n, s, e, w = (float(v) for v in src[4:].split(","))
        return from_osm_bbox(n, s, e, w, resolution=resolution, **options)

    path = Path(src)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    ext = path.suffix.lower()
    if ext == ".svg":
        from .svg import from_svg

        return from_svg(path, resolution=resolution, units_to_m=units_to_m, **options)
    if ext == ".dxf":
        from .dxf import from_dxf

        return from_dxf(path, resolution=resolution, units_to_m=units_to_m, **options)
    if ext in (".geojson", ".json"):
        from .roads import from_geojson_roads

        return from_geojson_roads(path, resolution=resolution, **options)
    if ext == ".npz":
        return WalkableMap.load(path)
    raise ValueError(f"don't know how to load a map from '{source}'")


def load_map_for_scenario(scenario) -> WalkableMap:
    """Convenience: build the map a Scenario asks for, resolving paths relative to the scenario file."""
    cfg = scenario.map
    base = scenario.path.parent if scenario.path else None
    units = None if cfg.units_to_m == 1.0 else cfg.units_to_m
    return load_map(cfg.source, resolution=cfg.walkable_grid_m, units_to_m=units, base_dir=base, **cfg.options)
