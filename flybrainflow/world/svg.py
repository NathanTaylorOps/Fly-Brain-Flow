"""SVG floor plans -> WalkableMap.

A small, dependency-free reader for the shapes a floor plan actually uses:
rect, polygon, polyline, circle, ellipse and path (M L H V C S Q T A Z, absolute
and relative), with group transforms applied.

Which shapes are walkable and which are obstacles is decided by name: any
`id`, `class` or Inkscape layer label containing "walk" is walkable; containing
"wall", "obstacle" or "block" is an obstacle. If nothing is labelled, every
closed shape is treated as walkable, which is right for "here is the outline
of the room" and wrong for anything more complicated — label your layers.

SVG y points down; world y points up. The reader flips it and, by default,
rebases the drawing so the walkable area's lower-left corner sits at (0, 0).
Units: pass `units_to_m` (e.g. 0.001 for a drawing in millimetres). If the
root element has a physical width and a viewBox, the scale is detected.
"""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from . import geom
from .map import WalkableMap

SVG_NS = "{http://www.w3.org/2000/svg}"
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"

WALKABLE_WORDS = ("walk", "floor", "drivable")
OBSTACLE_WORDS = ("wall", "obstacle", "block", "column", "furniture")

_UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "pt": 0.0254 / 72, "pc": 0.0254 / 6, "px": 0.0254 / 96, "": None}


def from_svg(
    path: str | Path,
    resolution: float = 0.1,
    units_to_m: float | None = None,
    padding: float = 0.5,
    rebase: bool = True,
    walkable_words=WALKABLE_WORDS,
    obstacle_words=OBSTACLE_WORDS,
    curve_segments: int = 16,
) -> WalkableMap:
    root = ET.parse(str(path)).getroot()
    scale_m = units_to_m if units_to_m is not None else (detect_unit_scale(root) or 1.0)
    shapes = collect_shapes(root, curve_segments=curve_segments)
    if not shapes:
        raise ValueError(f"{path}: no closed shapes found")

    walk, holes, unlabelled = [], [], []
    for ring, label in shapes:
        lab = label.lower()
        if any(w in lab for w in obstacle_words):
            holes.append(ring)
        elif any(w in lab for w in walkable_words):
            walk.append(ring)
        else:
            unlabelled.append(ring)
    if not walk:
        if holes and unlabelled:
            walk = unlabelled  # labelled obstacles, unlabelled floor: reasonable
        elif unlabelled:
            walk = unlabelled
            warnings.warn(f"{path}: nothing labelled walkable/obstacle; treating every closed shape as walkable")
        else:
            raise ValueError(f"{path}: obstacles found but no walkable area")
    elif unlabelled:
        warnings.warn(f"{path}: {len(unlabelled)} unlabelled shape(s) ignored")

    flip = np.array([[scale_m, 0], [0, -scale_m]])
    walk = [r @ flip.T for r in walk]
    holes = [r @ flip.T for r in holes]
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
        meta={"source": str(path), "units_to_m": scale_m, "offset": offset, "y_flipped": True},
    )


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def detect_unit_scale(root: ET.Element) -> float | None:
    """Metres per user unit, if the root has a physical width and a viewBox."""
    w = root.get("width")
    vb = root.get("viewBox")
    if not w or not vb:
        return None
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([a-zA-Z]*)\s*", w)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    unit_m = _UNIT_TO_M.get(unit)
    if unit_m is None:
        return None
    vb_w = float(vb.replace(",", " ").split()[2])
    if vb_w == 0:
        return None
    return val * unit_m / vb_w


def collect_shapes(root: ET.Element, curve_segments: int = 16) -> list[tuple[np.ndarray, str]]:
    """Walk the tree; return (ring, label) for every closed shape, with transforms applied."""
    out: list[tuple[np.ndarray, str]] = []

    def visit(el: ET.Element, m: np.ndarray, label: str) -> None:
        tag = el.tag.replace(SVG_NS, "")
        own = " ".join(filter(None, (el.get("id", ""), el.get("class", ""), el.get(INKSCAPE_LABEL, ""))))
        label = (label + " " + own).strip()
        m = m @ parse_transform(el.get("transform", ""))
        if tag in ("defs", "clipPath", "mask", "symbol", "marker", "pattern"):
            return
        rings = shape_rings(el, tag, curve_segments)
        for r in rings:
            r = geom.apply(m, r)
            if len(r) >= 3 and geom.ring_area(r) > 0:
                out.append((r, label))
        for child in el:
            visit(child, m, label)

    visit(root, geom.identity(), "")
    return out


def shape_rings(el: ET.Element, tag: str, n: int) -> list[np.ndarray]:
    g = lambda k, d="0": float(el.get(k, d))  # noqa: E731
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
    if tag == "polygon":
        return [_points(el.get("points", ""))]
    if tag == "polyline":
        p = _points(el.get("points", ""))
        return [p] if len(p) >= 3 and np.allclose(p[0], p[-1]) else []
    if tag == "circle":
        cx, cy, r = g("cx"), g("cy"), g("r")
        return [_ellipse(cx, cy, r, r)]
    if tag == "ellipse":
        return [_ellipse(g("cx"), g("cy"), g("rx"), g("ry"))]
    if tag == "path":
        return [r for r in parse_path(el.get("d", ""), n) if len(r) >= 3]
    return []


def _points(s: str) -> np.ndarray:
    nums = [float(t) for t in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
    return np.array(nums, float).reshape(-1, 2) if len(nums) >= 2 else np.zeros((0, 2))


def _ellipse(cx, cy, rx, ry, n=72) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]


# ---------------------------------------------------------------------------
# transform="..."
# ---------------------------------------------------------------------------

_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(s: str) -> np.ndarray:
    m = geom.identity()
    if not s:
        return m
    for name, args in _TRANSFORM_RE.findall(s):
        a = [float(t) for t in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if name == "matrix" and len(a) == 6:
            m = m @ geom.matrix(*a)
        elif name == "translate":
            m = m @ geom.translate(a[0], a[1] if len(a) > 1 else 0.0)
        elif name == "scale":
            m = m @ geom.scale(a[0], a[1] if len(a) > 1 else None)
        elif name == "rotate":
            m = m @ geom.rotate(a[0], *(a[1:3] if len(a) == 3 else (0.0, 0.0)))
        else:
            warnings.warn(f"SVG transform '{name}' not supported; ignored")
    return m


# ---------------------------------------------------------------------------
# path d="..."
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def parse_path(d: str, n: int = 16) -> list[np.ndarray]:
    """Flatten a path into closed rings. Open subpaths are dropped (a floor plan outline is closed)."""
    tokens = _TOKEN_RE.findall(d)
    rings: list[np.ndarray] = []
    pts: list[np.ndarray] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    last_ctrl: np.ndarray | None = None
    last_cmd = ""
    i = 0

    def close():
        nonlocal pts
        if len(pts) >= 3:
            rings.append(np.array(pts))
        pts = []

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
        else:
            cmd = last_cmd if last_cmd not in ("M", "m") else ("L" if last_cmd == "M" else "l")
        upper = cmd.upper()
        rel = cmd.islower()
        argc = _ARGC[upper]
        args = [float(t) for t in tokens[i : i + argc]]
        if len(args) < argc:
            break
        i += argc
        o = cur if rel else np.zeros(2)

        if upper == "M":
            if pts:
                close()
            cur = o + np.array(args)
            start = cur.copy()
            pts = [cur.copy()]
        elif upper == "L":
            cur = o + np.array(args)
            pts.append(cur.copy())
        elif upper == "H":
            cur = np.array([o[0] + args[0], cur[1]]) if rel else np.array([args[0], cur[1]])
            pts.append(cur.copy())
        elif upper == "V":
            cur = np.array([cur[0], o[1] + args[0]]) if rel else np.array([cur[0], args[0]])
            pts.append(cur.copy())
        elif upper == "C":
            p1, p2, p3 = o + np.array(args[0:2]), o + np.array(args[2:4]), o + np.array(args[4:6])
            pts.extend(geom.flatten_cubic(cur, p1, p2, p3, n))
            last_ctrl, cur = p2, p3
        elif upper == "S":
            p1 = 2 * cur - last_ctrl if (last_cmd.upper() in ("C", "S") and last_ctrl is not None) else cur
            p2, p3 = o + np.array(args[0:2]), o + np.array(args[2:4])
            pts.extend(geom.flatten_cubic(cur, p1, p2, p3, n))
            last_ctrl, cur = p2, p3
        elif upper == "Q":
            p1, p2 = o + np.array(args[0:2]), o + np.array(args[2:4])
            pts.extend(geom.flatten_quadratic(cur, p1, p2, n))
            last_ctrl, cur = p1, p2
        elif upper == "T":
            p1 = 2 * cur - last_ctrl if (last_cmd.upper() in ("Q", "T") and last_ctrl is not None) else cur
            p2 = o + np.array(args[0:2])
            pts.extend(geom.flatten_quadratic(cur, p1, p2, n))
            last_ctrl, cur = p1, p2
        elif upper == "A":
            rx, ry, rot, large, sweep = args[0], args[1], args[2], bool(args[3]), bool(args[4])
            p1 = o + np.array(args[5:7])
            pts.extend(geom.flatten_svg_arc(cur, rx, ry, rot, large, sweep, p1))
            cur = p1
        elif upper == "Z":
            cur = start.copy()
            close()
        last_cmd = cmd
    # An unclosed trailing subpath that returns to its start is still a ring.
    if pts and len(pts) >= 3 and np.allclose(pts[0], pts[-1], atol=1e-9):
        close()
    return rings
