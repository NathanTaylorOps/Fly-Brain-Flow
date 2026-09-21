"""Odour and wind: what a fly can smell, and which way the air is moving.

The plume itself is not a fluid simulation, and doesn't try to be: each
source (a sugar or mate target) puts out a smell that decays with distance
and gets stretched downwind. Good enough to give flies something to climb;
not a claim about real odour physics.

Distance can route around walls (geodesic, via `geodesic.py`) instead of
cutting through them — pass a `walkable_map` and it does. Without one, a fly
on the wrong side of a barrier would smell the target straight through the
wall and steer itself into the wall instead of toward the gap: fine in an
open room, wrong anywhere with real barriers, which is the whole point of a
stadium or train-station venue.

The wind's downwind/upwind stretch can use a real, wall-bending airflow field
(potential flow, via `airflow.py`) instead of one global vector — pass an
`airflow` field and each point gets stretched by the wind that actually
reaches it, deflected and sped up around obstacles the way real air is.
Without one, wind is a single uniform vector everywhere, which is wrong near
any corner or barrier for the same reason straight-line odour distance is.

    wind = Wind(direction_deg=90, speed_mps=1.5)      # blowing toward +y
    src = OdorSource(kind="sugar", position=(10, 5), strength=1.0, range_m=15)
    field = OdorField(wind, [src], walkable_map=m, airflow=AirflowField.build(m))
    field.sample_by_kind(agent_positions)["sugar"]     -> concentration, per agent
    field.gradient(agent_positions, kind="sugar")       -> unit-ish direction toward more, per agent

Targets move (that's the whole point of "movable sugar"); an `OdorSource` is
a plain mutable object, so update `.position` in place each step and the next
`sample()` call picks it up — nothing to rebuild. The airflow field itself is
geometry-only (it doesn't depend on where targets are), so it's built once
per map and reused for every source and every wind-dial change.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np

from .airflow import AirflowField
from .geodesic import geodesic_distance_field
from .geodesic import gradient_direction as _geo_gradient_direction
from .geodesic import sample_field as _sample_geo_field

# How much the wind stretches (downwind) or compresses (upwind) the plume's
# effective range. Pure tuning constant, not measured from anything.
WIND_STRETCH = 0.5
MIN_SCALE = 0.15
MAX_SCALE = 4.0


@dataclass
class Wind:
    """Uniform wind, mutable so a scenario (or a person watching the demo) can change it mid-run."""

    direction_deg: float = 0.0  # direction the wind blows TOWARD; 0 = +x, 90 = +y
    speed_mps: float = 0.0

    def set(self, direction_deg: float | None = None, speed_mps: float | None = None) -> None:
        if direction_deg is not None:
            self.direction_deg = float(direction_deg)
        if speed_mps is not None:
            if speed_mps < 0:
                raise ValueError(f"speed_mps must be >= 0, got {speed_mps}")
            self.speed_mps = float(speed_mps)

    @property
    def vector(self) -> np.ndarray:
        """(2,) world-frame velocity: unit direction times speed."""
        r = np.deg2rad(self.direction_deg)
        return np.array([np.cos(r), np.sin(r)]) * self.speed_mps

    @property
    def unit(self) -> np.ndarray:
        r = np.deg2rad(self.direction_deg)
        return np.array([np.cos(r), np.sin(r)])


@dataclass
class OdorSource:
    """One plume source. `kind` picks which sensory channel it drives ("sugar" or "mate")."""

    kind: str
    position: tuple[float, float]
    strength: float = 1.0
    range_m: float = 30.0

    def move_to(self, xy: tuple[float, float]) -> None:
        self.position = (float(xy[0]), float(xy[1]))


@dataclass
class OdorField:
    wind: Wind
    sources: list[OdorSource] = dc_field(default_factory=list)
    walkable_map: Any | None = None  # WalkableMap; when set, distance routes around walls
    airflow: AirflowField | None = None  # when set, wind bends around walls instead of one vector
    connectivity: int = 8

    def __post_init__(self) -> None:
        self._geo_cache: dict[int, tuple[tuple[float, float], np.ndarray]] = {}

    # -- sampling -------------------------------------------------------

    def sample(self, xy, kind: str | None = None, source: OdorSource | None = None) -> np.ndarray:
        """Summed concentration at each point, over sources matching `kind` (or all sources) by
        default. Pass `source` instead (a specific `OdorSource` object, matched by identity) to
        restrict to exactly *that one* plume -- see `gradient`'s docstring for why a fly steering
        toward one committed target needs this instead of the kind-wide sum."""
        p = np.asarray(xy, float).reshape(-1, 2)
        total = np.zeros(len(p))
        for s in self.sources:
            if source is not None:
                if s is not source:
                    continue
            elif kind is not None and s.kind != kind:
                continue
            total += self._plume(p, s)
        return total

    def sample_by_kind(self, xy) -> dict[str, np.ndarray]:
        p = np.asarray(xy, float).reshape(-1, 2)
        out: dict[str, np.ndarray] = {}
        for s in self.sources:
            c = self._plume(p, s)
            out[s.kind] = out.get(s.kind, np.zeros(len(p))) + c
        return out

    def gradient(
        self, xy, kind: str | None = None, eps: float | None = None, source: OdorSource | None = None
    ) -> np.ndarray:
        """(n, 2) finite-difference gradient of concentration, roughly pointing toward the nearest/strongest source.

        Default `eps` is 0.05 m in open (straight-line) mode. In occluded mode the field is only
        as fine as the map's grid, so `eps` defaults to 1.5 grid cells — smaller than that and two
        sample points can land in the same cell and give a false zero gradient right next to a wall.

        Real bug, found and fixed: `kind` sums *every* source of that kind into one combined
        plume, which is exactly right for "how strong does sugar smell overall" but wrong for "which
        way should I walk" once two same-kind targets exist -- a fly that had committed (via
        `TargetAssignment`) to a *far* sugar target still felt the combined gradient, which a
        *near* sugar target (right where it spawned) completely dominated, since the near plume's
        gradient totally swamps the far one's at any point closer to the near source. The result
        was a fly that visibly spiralled around the near target's spawn point instead of walking
        toward the one it had actually committed to -- see `ToyBrain._sense`'s own comment on this
        for the full story, and `docs/JOURNAL.md`. Pass `source` (the exact `OdorSource` a fly has
        committed to) to isolate that one plume instead of the kind-wide sum; `kind` remains for
        the genuinely kind-wide question ("total ambient sugar smell here", not used by steering).
        """
        p = np.asarray(xy, float).reshape(-1, 2)
        if eps is None:
            eps = self.walkable_map.resolution * 1.5 if self.walkable_map is not None else 0.05
        gx = (self.sample(p + [eps, 0.0], kind, source) - self.sample(p - [eps, 0.0], kind, source)) / (2 * eps)
        gy = (self.sample(p + [0.0, eps], kind, source) - self.sample(p - [0.0, eps], kind, source)) / (2 * eps)
        return np.c_[gx, gy]

    def kinds(self) -> set[str]:
        return {s.kind for s in self.sources}

    def local_wind(self, xy) -> np.ndarray:
        """(n, 2) wind velocity at each point -- the real, bent local value when an airflow field
        was given, otherwise the single global wind vector broadcast to every point. Public so
        anything that needs "which way is the air actually moving here" (a brain's own wind
        sensing, say -- Johnston's organ in the real neuron table) can reuse the exact same
        computation `_plume` uses internally, rather than re-deriving it."""
        p = np.asarray(xy, float).reshape(-1, 2)
        if self.airflow is not None:
            return self.airflow.velocity_at(p, self.wind.direction_deg, self.wind.speed_mps)
        return np.tile(self.wind.vector, (len(p), 1))

    # -- internals --------------------------------------------------------

    def _plume(self, p: np.ndarray, s: OdorSource) -> np.ndarray:
        src = np.asarray(s.position, float)
        d = p - src  # straight-line delta
        straight_r = np.linalg.norm(d, axis=1)
        occluded = self.walkable_map is not None
        r = self._geodesic_r(p, s) if occluded else straight_r
        if self.wind.speed_mps > 0:
            if occluded:
                # The "is this point downwind of the source" question needs the direction along
                # the actual walking path back to the source, not a straight line that may cut
                # through the very wall the scent has to detour around. This is the direction a
                # local wind vector should be compared against near a bend -- using the
                # straight-line bearing here was the gap in the "local wind" fix from the airflow
                # rebuild: that made the wind vector itself bend correctly, but still judged
                # alignment against a bearing that doesn't.
                unit_d = self._geodesic_direction(p, s)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    unit_d = np.divide(d, straight_r[:, None], out=np.zeros_like(d), where=straight_r[:, None] > 0)
            # One formula either way: without an airflow field, `local_wind` below is just the
            # single global vector broadcast to every point, which makes this mathematically the
            # same as the old airflow-less special case (alignment against `self.wind.unit` at a
            # constant `self.wind.speed_mps`) -- so there's no separate branch to keep in sync.
            local_wind = self.local_wind(p)
            local_speed = np.linalg.norm(local_wind, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                local_unit = np.divide(
                    local_wind, local_speed[:, None], out=np.zeros_like(local_wind), where=local_speed[:, None] > 0
                )
            alignment = np.einsum("ij,ij->i", unit_d, local_unit)  # +1 fully downwind, -1 fully upwind
            scale = np.clip(1.0 + WIND_STRETCH * local_speed * alignment, MIN_SCALE, MAX_SCALE)
        else:
            scale = np.ones_like(straight_r)
        eff_range = max(s.range_m, 1e-6) * scale
        with np.errstate(over="ignore", invalid="ignore"):
            return s.strength * np.exp(-r / eff_range)

    def _geodesic_r(self, p: np.ndarray, s: OdorSource) -> np.ndarray:
        return _sample_geo_field(self.walkable_map, self._geo_field(s), p)

    def _geodesic_direction(self, p: np.ndarray, s: OdorSource) -> np.ndarray:
        """Unit vector at each point, pointing along the shortest walkable path away from `s` --
        the wall-aware equivalent of the straight-line `(p - src) / |p - src|` used in open mode."""
        return _geo_gradient_direction(self.walkable_map, self._geo_field(s), p)

    def _geo_field(self, s: OdorSource) -> np.ndarray:
        cached = self._geo_cache.get(id(s))
        if cached is None or cached[0] != s.position:
            fld = geodesic_distance_field(self.walkable_map, s.position, connectivity=self.connectivity)
            cached = (s.position, fld)
            self._geo_cache[id(s)] = cached
        return cached[1]


# ---------------------------------------------------------------------------


def from_scenario(scenario, walkable_map: Any | None = None, airflow: AirflowField | None = None) -> OdorField:
    """Build the wind + one source per [[targets]] entry from a loaded Scenario.

    Pass the scenario's own map (`load_map_for_scenario(scenario)`) to get wall-aware odour, and
    `AirflowField.build(that map)` to also get wall-bending wind. Without them, odour ignores
    walls and wind is one uniform vector — fine for the open generator venues, wrong for anything
    with real barriers.
    """
    wind = Wind(direction_deg=scenario.wind.direction_deg, speed_mps=scenario.wind.speed_mps)
    sources = [
        OdorSource(kind=t.kind, position=t.position, strength=t.odor_strength, range_m=t.odor_range_m)
        for t in scenario.targets
    ]
    return OdorField(wind, sources, walkable_map=walkable_map, airflow=airflow)
