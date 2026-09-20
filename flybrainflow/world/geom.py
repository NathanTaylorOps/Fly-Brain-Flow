"""Small geometry helpers shared by the map loaders.

Everything works on plain numpy arrays of shape (N, 2). No geometry library —
the walkable area lives on a grid (see map.py), so polygons only need to be
good enough to rasterise.
"""

from __future__ import annotations

import math

import numpy as np

Ring = np.ndarray  # (N, 2) float, implicitly closed


# ---------------------------------------------------------------------------
# 2D affine transforms as 3x3 matrices
# ---------------------------------------------------------------------------


def identity() -> np.ndarray:
    return np.eye(3)


def translate(tx: float, ty: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2], m[1, 2] = tx, ty
    return m


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    m = np.eye(3)
    m[0, 0], m[1, 1] = sx, sx if sy is None else sy
    return m


def rotate(deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    if cx or cy:
        return translate(cx, cy) @ r @ translate(-cx, -cy)
    return r


def matrix(a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.c_[pts, np.ones(len(pts))]
    return (hom @ m.T)[:, :2]


# ---------------------------------------------------------------------------
# Curve flattening
# ---------------------------------------------------------------------------


def flatten_cubic(p0, p1, p2, p3, n: int = 16) -> np.ndarray:
    t = np.linspace(0, 1, n + 1)[1:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def flatten_quadratic(p0, p1, p2, n: int = 12) -> np.ndarray:
    t = np.linspace(0, 1, n + 1)[1:, None]
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def flatten_arc_center(cx, cy, r, start_deg, end_deg, ccw: bool = True, max_seg_deg: float = 5.0) -> np.ndarray:
    """Points along a circular arc from start to end angle (degrees). Excludes the start point."""
    sweep = (end_deg - start_deg) % 360.0
    if not ccw:
        sweep = sweep - 360.0 if sweep != 0 else 0.0
    if sweep == 0.0:
        sweep = 360.0 if ccw else -360.0
    n = max(2, int(math.ceil(abs(sweep) / max_seg_deg)))
    ang = np.radians(start_deg + sweep * np.linspace(0, 1, n + 1)[1:])
    return np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]


def flatten_svg_arc(p0, rx, ry, x_rot_deg, large_arc, sweep, p1, max_seg_deg: float = 5.0) -> np.ndarray:
    """SVG elliptical arc (endpoint parameterisation) -> points, excluding p0.

    Standard conversion from the SVG implementation notes (F.6.5).
    """
    x1, y1 = map(float, p0)
    x2, y2 = map(float, p1)
    if (x1, y1) == (x2, y2):
        return np.zeros((0, 2))
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return np.array([[x2, y2]])
    phi = math.radians(x_rot_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large_arc == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x1 + x2) / 2
    cy = sin_p * cxp + cos_p * cyp + (y1 + y2) / 2

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / length)))
        return -a if ux * vy - uy * vx < 0 else a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi
    n = max(2, int(math.ceil(abs(math.degrees(dtheta)) / max_seg_deg)))
    t = theta1 + dtheta * np.linspace(0, 1, n + 1)[1:]
    xs = cx + rx * np.cos(t) * cos_p - ry * np.sin(t) * sin_p
    ys = cy + rx * np.cos(t) * sin_p + ry * np.sin(t) * cos_p
    return np.c_[xs, ys]


def flatten_bulge(p0, p1, bulge: float, max_seg_deg: float = 5.0) -> np.ndarray:
    """DXF bulge segment between p0 and p1 -> points, excluding p0.

    bulge = tan(included_angle / 4); positive = counter-clockwise.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if abs(bulge) < 1e-12:
        return p1[None, :]
    theta = 4.0 * math.atan(bulge)  # included angle, signed
    chord = float(np.linalg.norm(p1 - p0))
    if chord < 1e-12:
        return np.zeros((0, 2))
    r = chord / (2.0 * math.sin(abs(theta) / 2.0))
    mid = (p0 + p1) / 2.0
    d = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
    # perpendicular to the chord; side depends on the sign of the bulge
    perp = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]]) / chord
    centre = mid + (-perp if theta > 0 else perp) * d * (1 if abs(theta) <= math.pi else -1)
    a0 = math.degrees(math.atan2(p0[1] - centre[1], p0[0] - centre[0]))
    a1 = math.degrees(math.atan2(p1[1] - centre[1], p1[0] - centre[0]))
    return flatten_arc_center(centre[0], centre[1], r, a0, a1, ccw=theta > 0, max_seg_deg=max_seg_deg)


# ---------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------


def ring_area(ring: Ring) -> float:
    """Unsigned polygon area (shoelace)."""
    r = np.asarray(ring, float)
    if len(r) < 3:
        return 0.0
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_closed(ring: Ring, tol: float = 1e-9) -> bool:
    r = np.asarray(ring, float)
    return len(r) >= 3 and float(np.linalg.norm(r[0] - r[-1])) <= tol


def dedupe_closing_point(ring: Ring) -> Ring:
    r = np.asarray(ring, float)
    if len(r) >= 2 and np.allclose(r[0], r[-1]):
        return r[:-1]
    return r


def bounds(rings: list[Ring]) -> tuple[float, float, float, float]:
    pts = np.vstack([np.asarray(r, float) for r in rings if len(r)])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


# ---------------------------------------------------------------------------
# Lon/lat -> local metres (good enough for a few km around a centre point)
# ---------------------------------------------------------------------------

EARTH_R = 6_371_008.8


def lonlat_to_local_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Equirectangular projection centred on (lon0, lat0). Returns (N, 2) metres."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.c_[x, y]


def looks_like_lonlat(pts: np.ndarray) -> bool:
    p = np.asarray(pts, float)
    return bool(np.all(np.abs(p[:, 0]) <= 180) and np.all(np.abs(p[:, 1]) <= 90) and np.ptp(p, axis=0).max() < 2.0)
