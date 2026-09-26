"""Turns a recorded run (`Recorder.to_arrays()` or `Recorder.load()`) plus the `WalkableMap` and
`Scenario` it was recorded from into one compact JSON file the browser-based 2D playback viewer
(`viewer/index.html`) can load directly -- no server, no build step.

    export_for_viewer(recording, walkable_map, scenario, "viewer/run.json")
    # then open viewer/index.html and load run.json

Deliberately NOT the eventual Three.js/live/WebSocket viewer the plan describes under "Two ideas
carry most of the design" -- that's showing all 166k real neurons firing, which only makes sense
once a real brain exists to click into, and is explicitly M3 (inspector) scope. What M0's own
milestone line actually calls for ("recorder and viewer run — all on the toy brain") is smaller:
proof a recorded run can be played back and looked at. This is that -- a flat top-down animation,
one dot per agent, colour-coded by cohort, target(s) marked, venue walls drawn from the map's own
polygon geometry (`WalkableMap.outer`/`.holes`) rather than re-deriving them from the raster mask.

Targets vs. target_moves: a target's *starting* position lives in the frozen `scenario.targets`
config for the whole run, but a target can now move mid-run. Rather than re-baking a moving
target's position into every single frame (bloating the payload and duplicating the same position
across many ticks where nothing changed), the payload keeps two separate things: `"targets"` is
"where things started" -- one entry per target, straight from `scenario.targets`, unchanged
whether or not it ever moved (this also doubles as the correct answer for a scenario where no
target ever moves, and as the fallback position before a target's first move). `"target_moves"`
is the sparse timeline of what changed and when -- one entry per move, in the order they happened,
each naming which target moved, to where, and at what simulated time. The viewer reconstructs a
moving target's position at any playback time by taking its starting position from `"targets"`
and applying every move up to that time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_for_viewer(recording: dict, walkable_map, scenario, path: str | Path, target_moves=None) -> dict:
    """Builds the JSON payload and writes it to `path`; also returns it (mainly so tests can check
    the shape without re-reading the file).

    `target_moves`, when given, is an iterable describing every target move that happened during
    the recorded run. Each item is either a `(t, index, (x, y))` tuple/list, or an object exposing
    `.t`, `.index` and `.position` attributes -- `index` is the target's position in
    `scenario.targets`. Order doesn't matter; entries are sorted by `t` before being written. See
    the module docstring for why this is kept separate from `"targets"`."""
    data = _build_payload(recording, walkable_map, scenario, target_moves=target_moves)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _build_payload(recording: dict, walkable_map, scenario, target_moves=None) -> dict:
    x0, y0, x1, y1 = walkable_map.bounds
    frames = _group_into_frames(recording)
    return {
        "bounds": [x0, y0, x1, y1],
        "walls": {
            "outer": [np.asarray(ring, float).tolist() for ring in walkable_map.outer],
            "holes": [np.asarray(ring, float).tolist() for ring in walkable_map.holes],
        },
        "targets": [{"kind": t.kind, "position": [float(t.position[0]), float(t.position[1])]} for t in scenario.targets],
        "target_moves": _build_target_moves(target_moves),
        "frames": frames,
    }


def _build_target_moves(target_moves) -> list[dict]:
    """Normalises the `target_moves` argument (see `export_for_viewer`'s docstring for the
    accepted item shapes) into the sparse, time-sorted list the JSON payload carries. Always
    returns a list -- `[]` when `target_moves` is `None` or empty -- so the viewer's JS can rely
    on the key always being present rather than checking for it."""
    if not target_moves:
        return []
    normalised = []
    for move in target_moves:
        if hasattr(move, "t") and hasattr(move, "index") and hasattr(move, "position"):
            t, index, position = move.t, move.index, move.position
        else:
            t, index, position = move
        x, y = position
        normalised.append({"t": float(t), "index": int(index), "position": [float(x), float(y)]})
    normalised.sort(key=lambda m: m["t"])
    return normalised


def _group_into_frames(recording: dict) -> list[dict]:
    """The recording is one row per (tick, agent) -- flat, columnar, good for analysis. The viewer
    wants the opposite shape: one entry per tick, listing every agent alive that tick. This is the
    only real reshaping this module does; everything else is a straight field-by-field copy."""
    n = len(recording["t"])
    if n == 0:
        return []
    order = np.argsort(recording["t"], kind="stable")
    ts, ids, brains = recording["t"][order], recording["id"][order], recording["brain"][order]
    xs, ys, statuses = recording["x"][order], recording["y"][order], recording["status"][order]

    frames: list[dict] = []
    current_t: float | None = None
    current_agents: list[dict] | None = None
    for idx in range(n):
        t = float(ts[idx])
        if current_t is None or t != current_t:
            if current_agents is not None:
                frames.append({"t": current_t, "agents": current_agents})
            current_t = t
            current_agents = []
        current_agents.append(
            {
                "id": int(ids[idx]),
                "brain": str(brains[idx]),
                "x": float(xs[idx]),
                "y": float(ys[idx]),
                "status": str(statuses[idx]),
            }
        )
    if current_agents is not None:
        frames.append({"t": current_t, "agents": current_agents})
    return frames
