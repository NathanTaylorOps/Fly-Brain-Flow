"""The walkable map: where agents are allowed to be.

Internally a map is a boolean grid (True = walkable) at a fixed resolution in
metres, plus the source rings it was rasterised from. Every loader — SVG, DXF,
road files, the built-in generators — produces the same thing, so the rest of
the simulation never cares where a map came from.

    from flybrainflow.world.map import corridor, WalkableMap
    m = corridor(length=40, width=5, resolution=0.1)
    m.is_walkable([[20, 2.5], [20, 7.0]])        -> [True, False]
    m.distance_to_wall                             -> metres to the nearest wall, per cell
    m.nearest_walkable([[20, 7.0]])                -> the closest legal point

Coordinates are metres, x to the right, y up. The grid origin is the world
position of the lower-left corner of cell (0, 0); world coordinates from the
source file are preserved, not rebased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage

from . import geom

Ring = np.ndarray


@dataclass
class WalkableMap:
    mask: np.ndarray  # (ny, nx) bool
    resolution: float  # metres per cell
    origin: tuple[float, float]  # world (x, y) of the lower-left corner of cell (0, 0)
    name: str = ""
    outer: list[Ring] = field(default_factory=list)  # source walkable rings, world metres
    holes: list[Ring] = field(default_factory=list)  # source obstacle rings, world metres
    meta: dict[str, Any] = field(default_factory=dict)
    # Cells outside `mask` that are still connected to the open sky for airflow purposes (a real
    # doorway or open end), independent of pedestrian walkability. None means "no explicit
    # openings declared" -- AirflowField then falls back to treating the raster's outer edge as
    # open, a coarser approximation that can be wrong when padding is only a cell or two deep.
    air_open: np.ndarray | None = field(default=None, repr=False, compare=False)
    _dist: np.ndarray | None = field(default=None, repr=False, compare=False)
    _nearest_idx: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -- shape --------------------------------------------------------------

    @property
    def ny(self) -> int:
        return int(self.mask.shape[0])

    @property
    def nx(self) -> int:
        return int(self.mask.shape[1])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x0, y0 = self.origin
        return x0, y0, x0 + self.nx * self.resolution, y0 + self.ny * self.resolution

    @property
    def width_m(self) -> float:
        return self.nx * self.resolution

    @property
    def height_m(self) -> float:
        return self.ny * self.resolution

    @property
    def walkable_area_m2(self) -> float:
        return float(self.mask.sum()) * self.resolution**2

    # -- coordinates --------------------------------------------------------

    def world_to_cell(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """World metres -> (ix, iy) integer cell indices. May be out of range; see in_bounds."""
        p = np.asarray(xy, float).reshape(-1, 2)
        ix = np.floor((p[:, 0] - self.origin[0]) / self.resolution).astype(int)
        iy = np.floor((p[:, 1] - self.origin[1]) / self.resolution).astype(int)
        return ix, iy

    def cell_to_world(self, ix, iy) -> np.ndarray:
        """Cell indices -> world metres at the cell centre."""
        ix = np.asarray(ix, float)
        iy = np.asarray(iy, float)
        x = self.origin[0] + (ix + 0.5) * self.resolution
        y = self.origin[1] + (iy + 0.5) * self.resolution
        return np.c_[x, y]

    def in_bounds(self, xy) -> np.ndarray:
        ix, iy = self.world_to_cell(xy)
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def is_walkable(self, xy) -> np.ndarray:
        """True where the point lies in a walkable cell. Outside the grid is never walkable."""
        ix, iy = self.world_to_cell(xy)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(len(ix), dtype=bool)
        out[ok] = self.mask[iy[ok], ix[ok]]
        return out

    def cell_centres(self) -> np.ndarray:
        """(ny*nx, 2) world coordinates of every cell centre, row-major."""
        iy, ix = np.mgrid[0 : self.ny, 0 : self.nx]
        return self.cell_to_world(ix.ravel(), iy.ravel())

    # -- distance fields ----------------------------------------------------

    @property
    def distance_to_wall(self) -> np.ndarray:
        """Metres from each walkable cell to the nearest non-walkable cell (0 where not walkable)."""
        if self._dist is None:
            self._dist = ndimage.distance_transform_edt(self.mask) * self.resolution
        return self._dist

    def distance_at(self, xy) -> np.ndarray:
        ix, iy = self.world_to_cell(xy)
        ok = self.in_bounds(xy)
        out = np.zeros(len(ix))
        out[ok] = self.distance_to_wall[iy[ok], ix[ok]]
        return out

    def nearest_walkable(self, xy) -> np.ndarray:
        """Project points onto the nearest walkable cell centre. Walkable points are returned unchanged."""
        p = np.asarray(xy, float).reshape(-1, 2).copy()
        if self._nearest_idx is None:
            # For every non-walkable cell, the index of the nearest walkable cell.
            _, idx = ndimage.distance_transform_edt(~self.mask, return_indices=True)
            self._nearest_idx = idx  # shape (2, ny, nx): [row, col]
        ix, iy = self.world_to_cell(p)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        walkable = self.mask[iy, ix]
        bad = ~walkable
        if bad.any():
            rows = self._nearest_idx[0][iy[bad], ix[bad]]
            cols = self._nearest_idx[1][iy[bad], ix[bad]]
            p[bad] = self.cell_to_world(cols, rows)
        return p

    # -- io -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import json

        np.savez_compressed(
            path,
            mask=self.mask,
            resolution=self.resolution,
            origin=np.array(self.origin),
            name=self.name,
            outer=np.array([o.tolist() for o in self.outer], dtype=object),
            holes=np.array([h.tolist() for h in self.holes], dtype=object),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WalkableMap":
        import json

        z = np.load(path, allow_pickle=True)
        return cls(
            mask=z["mask"].astype(bool),
            resolution=float(z["resolution"]),
            origin=(float(z["origin"][0]), float(z["origin"][1])),
            name=str(z["name"]),
            outer=[np.asarray(o, float) for o in z["outer"]],
            holes=[np.asarray(h, float) for h in z["holes"]],
            meta=json.loads(str(z["meta"])),
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_rings(
        cls,
        outer: list[Ring],
        holes: list[Ring] | None = None,
        resolution: float = 0.1,
        padding: float = 0.5,
        name: str = "",
        meta: dict[str, Any] | None = None,
    ) -> "WalkableMap":
        """Rasterise walkable rings minus obstacle rings onto a grid.

        `padding` (metres) adds a non-walkable border around the geometry so
        there is always a wall to bump into at the edge of the world.
        """
        outer = [geom.dedupe_closing_point(np.asarray(r, float)) for r in outer if len(r) >= 3]
        holes = [geom.dedupe_closing_point(np.asarray(r, float)) for r in (holes or []) if len(r) >= 3]
        if not outer:
            raise ValueError("a map needs at least one walkable ring with 3+ points")
        xmin, ymin, xmax, ymax = geom.bounds(outer)
        x0, y0 = xmin - padding, ymin - padding
        nx = int(np.ceil((xmax + padding - x0) / resolution))
        ny = int(np.ceil((ymax + padding - y0) / resolution))
        m = cls(mask=np.zeros((ny, nx), bool), resolution=resolution, origin=(x0, y0), name=name, meta=meta or {})
        centres = m.cell_centres()
        inside = np.zeros(len(centres), bool)
        for r in outer:
            inside |= MplPath(r).contains_points(centres)
        for h in holes:
            inside &= ~MplPath(h).contains_points(centres)
        m.mask = inside.reshape(ny, nx)
        m.outer, m.holes = outer, holes
        return m

    @classmethod
    def from_mask(cls, mask: np.ndarray, resolution: float, origin=(0.0, 0.0), name="", meta=None) -> "WalkableMap":
        return cls(mask=np.asarray(mask, bool), resolution=resolution, origin=tuple(origin), name=name, meta=meta or {})


# ---------------------------------------------------------------------------
# Generators — simple test venues, in metres, lower-left corner at (0, 0)
# ---------------------------------------------------------------------------


def _rect(x0, y0, x1, y1) -> Ring:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def corridor(
    length: float = 40.0, width: float = 5.0, resolution: float = 0.1, padding: float = 0.5, open_ends: bool = False
) -> WalkableMap:
    """A straight corridor. Spawn at either end for the two-stream lane test.

    `open_ends=True` removes the padding at the two short ends specifically (side walls stay),
    so the walkable area reaches the map's edge there. Pedestrians still can't leave (the sim
    boundary is still the sim boundary) but it means the corridor has a real doorway for wind to
    enter and leave through — without this, `AirflowField` correctly reports zero wind here,
    because a box walled on every side genuinely has nowhere for outside air to come from.
    """
    m = WalkableMap.from_rings(
        [_rect(0, 0, length, width)],
        resolution=resolution,
        padding=padding,
        name="corridor",
        meta={"length": length, "width": width, "open_ends": open_ends},
    )
    if open_ends:
        _open_ends_x(m, y_lo=0.0, y_hi=width)
    return m


def bottleneck(
    length: float = 40.0,
    width: float = 8.0,
    gap_width: float = 1.2,
    gap_length: float = 2.0,
    resolution: float = 0.1,
    padding: float = 0.5,
    open_ends: bool = False,
) -> WalkableMap:
    """A corridor that narrows to `gap_width` for `gap_length` metres in the middle. The doorway test.

    See `corridor()` for what `open_ends=True` does and why it's needed before `AirflowField`
    will show any real wind through the gap.
    """
    if gap_width >= width:
        raise ValueError("gap_width must be narrower than width")
    xa, xb = (length - gap_length) / 2, (length + gap_length) / 2
    wall = (width - gap_width) / 2
    holes = [_rect(xa, 0, xb, wall), _rect(xa, width - wall, xb, width)]
    m = WalkableMap.from_rings(
        [_rect(0, 0, length, width)],
        holes,
        resolution=resolution,
        padding=padding,
        name="bottleneck",
        meta={
            "length": length,
            "width": width,
            "gap_width": gap_width,
            "gap_length": gap_length,
            "gap_x": (xa, xb),
            "open_ends": open_ends,
        },
    )
    if open_ends:
        _open_ends_x(m, y_lo=0.0, y_hi=width)
    return m


def _open_ends_x(m: WalkableMap, y_lo: float, y_hi: float) -> None:
    """Declare the entire left/right padding depth, for y in [y_lo, y_hi), as open to the sky for
    airflow (`AirflowField`) — a real doorway. This does NOT change where pedestrians can walk
    (`m.mask` is untouched); it only tells the airflow solver where outside air can actually get
    in, since a wall this end simply isn't there to block it."""
    iy0, iy1 = m.world_to_cell([[0.0, y_lo], [0.0, y_hi]])[1]
    iy0, iy1 = int(iy0), int(min(iy1 + 1, m.ny))
    length = m.meta.get("length", 0.0)
    ix_left = int(m.world_to_cell([[0.0, y_lo]])[0][0])  # first walkable column, at the x=0 end
    # Query a touch past `length`, not exactly at it -- `length` can land exactly on a cell
    # boundary, which made the naive "+1" version miss the right-hand opening entirely.
    ix_right = int(m.world_to_cell([[length + m.resolution / 2, y_lo]])[0][0])
    if m.air_open is None:
        m.air_open = np.zeros_like(m.mask)
    m.air_open[iy0:iy1, 0:ix_left] = True
    m.air_open[iy0:iy1, ix_right : m.nx] = True


def ring(radius: float = 36.6, lane_width: float = 3.5, resolution: float = 0.25, padding: float = 0.5) -> WalkableMap:
    """An annular loop centred at (0, 0): centreline radius `radius`, drivable band `lane_width`.

    Default circumference 230 m matches the Sugiyama et al. (2008) ring-road experiment.
    """
    r_out = radius + lane_width / 2
    r_in = radius - lane_width / 2
    if r_in <= 0:
        raise ValueError("lane_width too wide for radius")
    ext = r_out + padding
    n = int(np.ceil(2 * ext / resolution))
    origin = (-ext, -ext)
    m = WalkableMap(mask=np.zeros((n, n), bool), resolution=resolution, origin=origin, name="ring")
    c = m.cell_centres()
    rr = np.hypot(c[:, 0], c[:, 1])
    m.mask = ((rr >= r_in) & (rr <= r_out)).reshape(n, n)
    ang = np.linspace(0, 2 * np.pi, 361)[:-1]
    m.outer = [np.c_[r_out * np.cos(ang), r_out * np.sin(ang)]]
    m.holes = [np.c_[r_in * np.cos(ang), r_in * np.sin(ang)]]
    m.meta = {"centre": (0.0, 0.0), "radius": radius, "lane_width": lane_width, "circumference": 2 * np.pi * radius}
    return m


def room_with_exit(
    width: float = 20.0,
    height: float = 15.0,
    door_width: float = 1.5,
    stub_length: float = 4.0,
    resolution: float = 0.1,
    padding: float = 0.5,
) -> WalkableMap:
    """A rectangular room with one doorway in the right-hand wall leading to a short exit corridor."""
    y0 = (height - door_width) / 2
    rings = [_rect(0, 0, width, height), _rect(width - resolution, y0, width + stub_length, y0 + door_width)]
    return WalkableMap.from_rings(
        rings,
        resolution=resolution,
        padding=padding,
        name="room_with_exit",
        meta={"width": width, "height": height, "door_width": door_width, "door_y": (y0, y0 + door_width)},
    )


GENERATORS = {"corridor": corridor, "bottleneck": bottleneck, "ring": ring, "room": room_with_exit}
