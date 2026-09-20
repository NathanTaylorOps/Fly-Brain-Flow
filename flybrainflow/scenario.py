"""Scenario files: TOML in, validated typed config out.

The schema is the one in docs/PLAN.md ("Scenario settings"). Every run is
described by one of these files so it can be repeated exactly.

    from flybrainflow.scenario import Scenario
    sc = Scenario.from_toml("scenarios/corridor_bidirectional.toml")
    sc.spawn.rate_per_s, sc.targets[0].slots, sc.map.source ...

Validation is deliberately strict and the error messages say which key is
wrong. A scenario that loads is a scenario that can run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCALE_MODES = ("pedestrian", "car")
BOUNDARY_MODES = ("open", "closed")
TARGET_KINDS = ("sugar", "mate")


class ScenarioError(ValueError):
    """A scenario file is missing something or has a bad value."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioMeta:
    name: str
    seed: int = 0
    duration_s: float = 300.0
    scale_mode: str = "pedestrian"
    boundary_mode: str = "open"
    timescale_ratio: float = 1.0  # brain milliseconds per world millisecond


@dataclass(frozen=True)
class MapConfig:
    source: str  # "gen:corridor?length=40&width=5" | "venues/x.svg" | "roads.geojson" | "osm:N,S,E,W"
    walkable_grid_m: float = 0.1
    units_to_m: float = 1.0  # for SVG/DXF: multiply file units by this to get metres
    options: dict[str, Any] = field(default_factory=dict)  # loader-specific extras, passed through


@dataclass(frozen=True)
class WindConfig:
    direction_deg: float = 0.0  # direction the wind blows TOWARD, degrees, 0 = +x, 90 = +y
    speed_mps: float = 0.0


@dataclass(frozen=True)
class SpawnConfig:
    sources: tuple[tuple[float, float], ...]
    rate_per_s: float = 1.0
    population_cap: int = 100


@dataclass(frozen=True)
class TargetConfig:
    kind: str
    position: tuple[float, float]
    slots: int = 6
    feeding_time_s: float = 2.0
    odor_strength: float = 1.0
    odor_range_m: float = 30.0


@dataclass(frozen=True)
class AgentsConfig:
    personality_mix: str = "uniform20"
    sensing_radius_m: float = 3.0
    field_of_view_deg: float = 270.0
    radius_m: float = 0.25
    max_speed_mps: float = 1.3


@dataclass(frozen=True)
class BaselineConfig:
    enabled: bool = True


@dataclass(frozen=True)
class Scenario:
    meta: ScenarioMeta
    map: MapConfig
    wind: WindConfig
    spawn: SpawnConfig
    targets: tuple[TargetConfig, ...]
    agents: AgentsConfig
    baseline: BaselineConfig
    path: Path | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "Scenario":
        path = Path(path)
        with path.open("rb") as f:
            raw = tomllib.load(f)
        sc = cls.from_dict(raw)
        return sc.with_path(path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Scenario":
        meta = _parse_meta(_section(raw, "scenario"))
        map_cfg = _parse_map(_section(raw, "map"))
        wind = _parse_wind(raw.get("wind", {}))
        spawn = _parse_spawn(_section(raw, "spawn"), meta.boundary_mode)
        targets = _parse_targets(raw.get("targets", []), meta.boundary_mode)
        agents = _parse_agents(raw.get("agents", {}))
        baseline = BaselineConfig(enabled=bool(raw.get("baseline", {}).get("enabled", True)))
        return cls(meta, map_cfg, wind, spawn, targets, agents, baseline)

    def with_path(self, path: Path) -> "Scenario":
        return Scenario(
            self.meta, self.map, self.wind, self.spawn, self.targets, self.agents, self.baseline, path
        )

    # -- convenience --------------------------------------------------------

    @property
    def max_exit_rate_per_s(self) -> float:
        """targets x slots / feeding time — the capacity rule from the plan."""
        return sum(t.slots / t.feeding_time_s for t in self.targets)

    def resolve(self, relative: str) -> Path:
        """Resolve a path in the scenario file relative to the file's folder."""
        base = self.path.parent if self.path else Path.cwd()
        return (base / relative).resolve()


# ---------------------------------------------------------------------------
# Parsers — each one names the key it is complaining about.
# ---------------------------------------------------------------------------


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ScenarioError(f"[{key}] section is required")
    return raw[key]


def _need(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ScenarioError(f"[{where}] is missing '{key}'")
    return d[key]


def _positive(value: Any, where: str, key: str, strict: bool = True) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ScenarioError(f"[{where}] '{key}' must be a number, got {value!r}") from None
    if (strict and v <= 0) or (not strict and v < 0):
        raise ScenarioError(f"[{where}] '{key}' must be {'> 0' if strict else '>= 0'}, got {v}")
    return v


def _xy(value: Any, where: str, key: str) -> tuple[float, float]:
    try:
        x, y = value
        return float(x), float(y)
    except (TypeError, ValueError):
        raise ScenarioError(f"[{where}] '{key}' must be [x, y], got {value!r}") from None


def _one_of(value: Any, allowed: tuple[str, ...], where: str, key: str) -> str:
    if value not in allowed:
        raise ScenarioError(f"[{where}] '{key}' must be one of {allowed}, got {value!r}")
    return value


def _parse_meta(d: dict[str, Any]) -> ScenarioMeta:
    name = str(_need(d, "name", "scenario"))
    seed = int(d.get("seed", 0))
    return ScenarioMeta(
        name=name,
        seed=seed,
        duration_s=_positive(d.get("duration_s", 300.0), "scenario", "duration_s"),
        scale_mode=_one_of(d.get("scale_mode", "pedestrian"), SCALE_MODES, "scenario", "scale_mode"),
        boundary_mode=_one_of(d.get("boundary_mode", "open"), BOUNDARY_MODES, "scenario", "boundary_mode"),
        timescale_ratio=_positive(d.get("timescale_ratio", 1.0), "scenario", "timescale_ratio"),
    )


def _parse_map(d: dict[str, Any]) -> MapConfig:
    source = str(_need(d, "source", "map"))
    known = {"source", "walkable_grid_m", "units_to_m"}
    options = {k: v for k, v in d.items() if k not in known}
    return MapConfig(
        source=source,
        walkable_grid_m=_positive(d.get("walkable_grid_m", 0.1), "map", "walkable_grid_m"),
        units_to_m=_positive(d.get("units_to_m", 1.0), "map", "units_to_m"),
        options=options,
    )


def _parse_wind(d: dict[str, Any]) -> WindConfig:
    return WindConfig(
        direction_deg=float(d.get("direction_deg", 0.0)),
        speed_mps=_positive(d.get("speed_mps", 0.0), "wind", "speed_mps", strict=False),
    )


def _parse_spawn(d: dict[str, Any], boundary_mode: str) -> SpawnConfig:
    raw_sources = d.get("sources", [])
    if boundary_mode == "open" and not raw_sources:
        raise ScenarioError("[spawn] 'sources' must list at least one entry point in open mode")
    sources = tuple(_xy(s, "spawn", "sources") for s in raw_sources)
    cap = int(d.get("population_cap", 100))
    if cap <= 0:
        raise ScenarioError(f"[spawn] 'population_cap' must be > 0, got {cap}")
    return SpawnConfig(
        sources=sources,
        rate_per_s=_positive(d.get("rate_per_s", 1.0), "spawn", "rate_per_s", strict=False),
        population_cap=cap,
    )


def _parse_targets(items: list[dict[str, Any]], boundary_mode: str) -> tuple[TargetConfig, ...]:
    if boundary_mode == "open" and not items:
        raise ScenarioError("at least one [[targets]] entry is required in open mode")
    out = []
    for i, t in enumerate(items):
        where = f"targets[{i}]"
        slots = int(t.get("slots", 6))
        if slots <= 0:
            raise ScenarioError(f"[{where}] 'slots' must be > 0, got {slots}")
        out.append(
            TargetConfig(
                kind=_one_of(t.get("kind", "sugar"), TARGET_KINDS, where, "kind"),
                position=_xy(_need(t, "position", where), where, "position"),
                slots=slots,
                feeding_time_s=_positive(t.get("feeding_time_s", 2.0), where, "feeding_time_s"),
                odor_strength=_positive(t.get("odor_strength", 1.0), where, "odor_strength", strict=False),
                odor_range_m=_positive(t.get("odor_range_m", 30.0), where, "odor_range_m"),
            )
        )
    return tuple(out)


def _parse_agents(d: dict[str, Any]) -> AgentsConfig:
    fov = float(d.get("field_of_view_deg", 270.0))
    if not 0 < fov <= 360:
        raise ScenarioError(f"[agents] 'field_of_view_deg' must be in (0, 360], got {fov}")
    return AgentsConfig(
        personality_mix=str(d.get("personality_mix", "uniform20")),
        sensing_radius_m=_positive(d.get("sensing_radius_m", 3.0), "agents", "sensing_radius_m"),
        field_of_view_deg=fov,
        radius_m=_positive(d.get("radius_m", 0.25), "agents", "radius_m"),
        max_speed_mps=_positive(d.get("max_speed_mps", 1.3), "agents", "max_speed_mps"),
    )
