"""DXF floor plans (Chief Architect, AutoCAD, ...) -> WalkableMap.

A dependency-free reader for the ASCII DXF entities a floor plan uses:
LWPOLYLINE (with bulge arcs), POLYLINE/VERTEX, LINE, ARC and CIRCLE.

Walkable vs obstacle is decided by layer name, same words as the SVG reader:
a layer containing "walk", "floor" or "drivable" is walkable; "wall",
"obstacle", "block", "column" or "furniture" is an obstacle. If no layer is
labelled, every closed shape is walkable.

Open LINE/ARC segments are chained end-to-end into closed loops where they
meet (within `join_tol` drawing units), because CAD exports often draw an
outline as separate lines. Segments that never close are ignored with a
warning.

Units come from the header ($INSUNITS) unless `units_to_m` is given.
DXF y already points up, so nothing is flipped.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from . import geom
from .map import WalkableMap
from .svg import OBSTACLE_WORDS, WALKABLE_WORDS

_INSUNITS_TO_M = {0: None, 1: 0.0254, 2: 0.3048, 3: 1609.344, 4: 0.001, 5: 0.01, 6: 1.0, 7: 1000.0, 8: 2.54e-5, 9: 2.54e-8, 10: 0.9144}


def from_dxf(
    path: str | Path,
    resolution: float = 0.1,
    units_to_m: float | None = None,
    padding: float = 0.5,
    rebase: bool = True,
    join_tol: float = 1e-3,
    walkable_words=WALKABLE_WORDS,
    obstacle_words=OBSTACLE_WORDS,
) -> WalkableMap:
    pairs = read_pairs(path)
    header_units = header_insunits(pairs)
    scale_m = units_to_m if units_to_m is not None else (_INSUNITS_TO_M.get(header_units) or 1.0)
    if units_to_m is None and not _INSUNITS_TO_M.get(header_units):
        warnings.warn(f"{path}: no usable $INSUNITS in header; assuming drawing units are metres")

    closed, open_segs = entities(pairs)
    closed += chain_segments(open_segs, join_tol)

    walk, holes, unlabelled = [], [], []
    for ring, layer in closed:
        lab = layer.lower()
        if any(w in lab for w in obstacle_words):
            holes.append(ring)
        elif any(w in lab for w in walkable_words):
            walk.append(ring)
        else:
            unlabelled.append(ring)
    if not walk:
        if unlabelled:
            walk = unlabelled
            if not holes:
                warnings.warn(f"{path}: no walkable/obstacle layers; treating every closed shape as walkable")
        else:
            raise ValueError(f"{path}: no closed walkable shapes found")
    elif unlabelled:
        warnings.warn(f"{path}: {len(unlabelled)} shape(s) on unlabelled layers ignored")

    walk = [r * scale_m for r in walk]
    holes = [h * scale_m for h in holes]
    offset = (0.0, 0.0)
    if rebase:
        xmin, ymin, _, _ = geom.bounds(walk)
        offset = (-xmin, -ymin)
        walk = [r + offset for r in walk]
        holes = [h + offset for h in holes]
    return WalkableMap.from_rings(
        walk,
        holes,
        resolution=resolution,
        padding=padding,
        name=Path(path).stem,
        meta={"source": str(path), "units_to_m": scale_m, "offset": offset, "insunits": header_units},
    )


# ---------------------------------------------------------------------------
# Low-level parsing
# ---------------------------------------------------------------------------


def read_pairs(path: str | Path) -> list[tuple[int, str]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        code = lines[i].strip()
        if not code.lstrip("-").isdigit():
            continue
        pairs.append((int(code), lines[i + 1].strip()))
    return pairs


def header_insunits(pairs: list[tuple[int, str]]) -> int | None:
    for i, (code, val) in enumerate(pairs):
        if code == 9 and val == "$INSUNITS" and i + 1 < len(pairs) and pairs[i + 1][0] == 70:
            return int(pairs[i + 1][1])
    return None


def entities(pairs: list[tuple[int, str]]) -> tuple[list[tuple[np.ndarray, str]], list[tuple[np.ndarray, str]]]:
    """Return (closed rings, open polylines), each tagged with its layer."""
    closed: list[tuple[np.ndarray, str]] = []
    open_segs: list[tuple[np.ndarray, str]] = []

    # Split into entity blocks inside the ENTITIES section.
    in_entities = False
    blocks: list[list[tuple[int, str]]] = []
    for code, val in pairs:
        if code == 2 and val == "ENTITIES":
            in_entities = True
            continue
        if in_entities and code == 0 and val == "ENDSEC":
            in_entities = False
            continue
        if not in_entities:
            continue
        if code == 0:
            blocks.append([(code, val)])
        elif blocks:
            blocks[-1].append((code, val))

    i = 0
    while i < len(blocks):
        b = blocks[i]
        kind = b[0][1]
        layer = _first(b, 8, "0")
        if kind == "LWPOLYLINE":
            ring, is_closed = _lwpolyline(b)
            (closed if is_closed else open_segs).append((ring, layer))
        elif kind == "POLYLINE":
            flags = int(_first(b, 70, "0"))
            verts, bulges = [], []
            i += 1
            while i < len(blocks) and blocks[i][0][1] == "VERTEX":
                vb = blocks[i]
                verts.append((float(_first(vb, 10, "0")), float(_first(vb, 20, "0"))))
                bulges.append(float(_first(vb, 42, "0")))
                i += 1
            # blocks[i] should be SEQEND
            ring = _expand_bulges(np.array(verts, float), bulges, closed=bool(flags & 1))
            (closed if flags & 1 else open_segs).append((ring, layer))
        elif kind == "LINE":
            seg = np.array([[float(_first(b, 10)), float(_first(b, 20))], [float(_first(b, 11)), float(_first(b, 21))]])
            open_segs.append((seg, layer))
        elif kind == "ARC":
            cx, cy, r = float(_first(b, 10)), float(_first(b, 20)), float(_first(b, 40))
            a0, a1 = float(_first(b, 50, "0")), float(_first(b, 51, "360"))
            pts = np.vstack([[[cx + r * np.cos(np.radians(a0)), cy + r * np.sin(np.radians(a0))]], geom.flatten_arc_center(cx, cy, r, a0, a1, ccw=True)])
            open_segs.append((pts, layer))
        elif kind == "CIRCLE":
            cx, cy, r = float(_first(b, 10)), float(_first(b, 20)), float(_first(b, 40))
            a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            closed.append((np.c_[cx + r * np.cos(a), cy + r * np.sin(a)], layer))
        i += 1
    return closed, open_segs


def _first(block, code: int, default: str = "0") -> str:
    for c, v in block:
        if c == code:
            return v
    return default


def _lwpolyline(block) -> tuple[np.ndarray, bool]:
    flags = int(_first(block, 70, "0"))
    verts: list[list[float]] = []
    bulges: list[float] = []
    for code, val in block:
        if code == 10:
            verts.append([float(val), 0.0])
            bulges.append(0.0)
        elif code == 20 and verts:
            verts[-1][1] = float(val)
        elif code == 42 and verts:
            bulges[-1] = float(val)
    ring = _expand_bulges(np.array(verts, float), bulges, closed=bool(flags & 1))
    return ring, bool(flags & 1)


def _expand_bulges(verts: np.ndarray, bulges: list[float], closed: bool) -> np.ndarray:
    if len(verts) == 0:
        return verts
    if not any(abs(b) > 1e-12 for b in bulges):
        return verts
    out = [verts[0]]
    n = len(verts)
    last = n if closed else n - 1
    for k in range(last):
        p0, p1 = verts[k], verts[(k + 1) % n]
        seg = geom.flatten_bulge(p0, p1, bulges[k])
        out.extend(seg)
    arr = np.array(out)
    return geom.dedupe_closing_point(arr) if closed else arr


def chain_segments(segs: list[tuple[np.ndarray, str]], tol: float) -> list[tuple[np.ndarray, str]]:
    """Greedily join open polylines end-to-end into closed rings."""
    remaining = [(np.asarray(s, float), layer) for s, layer in segs if len(s) >= 2]
    rings: list[tuple[np.ndarray, str]] = []
    while remaining:
        cur, layer = remaining.pop(0)
        progressed = True
        while progressed:
            progressed = False
            if len(cur) >= 3 and np.linalg.norm(cur[0] - cur[-1]) <= tol:
                rings.append((geom.dedupe_closing_point(cur), layer))
                cur = None
                break
            for j, (s, _) in enumerate(remaining):
                if np.linalg.norm(cur[-1] - s[0]) <= tol:
                    cur = np.vstack([cur, s[1:]])
                elif np.linalg.norm(cur[-1] - s[-1]) <= tol:
                    cur = np.vstack([cur, s[::-1][1:]])
                elif np.linalg.norm(cur[0] - s[-1]) <= tol:
                    cur = np.vstack([s, cur[1:]])
                elif np.linalg.norm(cur[0] - s[0]) <= tol:
                    cur = np.vstack([s[::-1], cur[1:]])
                else:
                    continue
                remaining.pop(j)
                progressed = True
                break
        if cur is not None:
            warnings.warn(f"DXF: an open chain of {len(cur)} points on layer '{layer}' never closed; ignored")
    return rings
