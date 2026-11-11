"""Loopback HTTP service for the behavioral arena experiment.

World state is owned by one locked session. The frontend requests bounded steps;
there is no hidden autonomous progression or public multi-user service here.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlsplit


PRESETS = ("open", "blocked", "surge")
DEFAULT_CONFIG = {"population": 200, "seed": 7, "preset": "open",
                  "wind_direction_deg": 180.0, "wind_speed_mps": 1.0,
                  "hazard_strength": 0.0, "hazard_position": [32.0, 24.0]}


def number(value, name, low, high, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    if integer and value != int(value):
        raise ValueError(f"{name} must be a whole number")
    return int(value) if integer else float(value)


def validate_config(values):
    if not isinstance(values, dict):
        raise ValueError("Configuration must be an object")
    values = dict(values)
    if "scenario" in values:
        if "preset" in values:
            raise ValueError("Use either scenario or preset, not both")
        values["preset"] = values.pop("scenario")
    unknown = set(values) - DEFAULT_CONFIG.keys()
    if unknown:
        raise ValueError(f"Unknown configuration: {sorted(unknown)[0]}")
    specs = {"population": (50, 500, True), "seed": (0, 2**32-1, True),
             "wind_direction_deg": (0, 360, False), "wind_speed_mps": (0, 3, False),
             "hazard_strength": (0, 5, False)}
    for key, (low, high, integer) in specs.items():
        if key in values:
            values[key] = number(values[key], key, low, high, integer=integer)
    if "preset" in values and values["preset"] not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}")
    if "hazard_position" in values:
        xy = values["hazard_position"]
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError("hazard_position must be [x,y]")
        values["hazard_position"] = [number(v, "hazard_position", -1e5, 1e5) for v in xy]
    return values


def make_sim(config):
    from .arena import ArenaSim
    return ArenaSim(config)


class ArenaSession:
    def __init__(self, config=None, *, sim_factory=None):
        self.lock = threading.RLock()
        self.sim_factory = sim_factory or make_sim
        self.config = dict(DEFAULT_CONFIG)
        self.reset(config or {})

    def reset(self, values=None):
        updates = validate_config(values or {})
        with self.lock:
            candidate = {**self.config, **updates}
            sim = self.sim_factory(candidate)
            # A failed construction leaves the existing simulation/config intact.
            self.sim = sim
            self.config = candidate
            self.steps = 0
            self.wall_s = 0.0
            self.advanced_s = 0.0
            return self.state()

    def step(self, values=None):
        values = {} if values is None else values
        if not isinstance(values, dict) or set(values) - {"steps", "dt"}:
            raise ValueError("Step accepts only steps and dt")
        steps = number(values.get("steps", 1), "steps", 1, 10, integer=True)
        dt = number(values.get("dt", 0.05), "dt", 0.005, 0.05)
        with self.lock:
            started = time.perf_counter()
            self.sim.step(steps=steps, dt=dt)
            self.wall_s += time.perf_counter() - started
            self.advanced_s += steps * dt
            self.steps += steps
            return self.state()

    def control(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("Control must be a nonempty object")
        allowed = {"alarm", "exit", "preset", "wind_direction_deg", "wind_speed_mps",
                   "hazard_strength", "hazard_position"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown control: {sorted(unknown)[0]}")
        # Discrete actions have distinct failure conditions; send them separately.
        # The two wind dials are a single validated update.
        if len(values) > 1 and set(values) - {"wind_direction_deg", "wind_speed_mps"}:
            raise ValueError("Send one control action at a time; wind dials may be combined")
        values = dict(values)
        numeric = {k: v for k, v in values.items() if k in DEFAULT_CONFIG}
        values.update(validate_config(numeric))
        if "alarm" in values and not isinstance(values["alarm"], bool):
            raise ValueError("alarm must be true or false")
        if "exit" in values:
            exit_ = values["exit"]
            if (not isinstance(exit_, dict) or set(exit_) != {"id", "open"}
                    or not isinstance(exit_["id"], str) or not exit_["id"]
                    or not isinstance(exit_["open"], bool)):
                raise ValueError("exit must contain a string id and boolean open")
        if "hazard_position" in values:
            xy = values["hazard_position"]
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError("hazard_position must be [x,y]")
            values["hazard_position"] = [number(v, "hazard_position", -1e6, 1e6) for v in xy]
        with self.lock:
            if "preset" in values:
                return self.reset({"preset": values["preset"]})
            self.sim.control(values)
            for key in DEFAULT_CONFIG:
                if key in values:
                    self.config[key] = values[key]
            return self.state()

    def performance(self):
        return {"compute_rtf": self.advanced_s / self.wall_s if self.wall_s else None,
                "steps": self.steps, "wall_s": self.wall_s, "advanced_s": self.advanced_s}

    def state(self):
        with self.lock:
            state = dict(self.sim.state())
            state["performance"] = self.performance()
            return state

    def environment(self):
        with self.lock:
            return self.sim.environment()

    def export(self):
        with self.lock:
            result = dict(self.sim.export())
            result["performance"] = self.performance()
            return result


class ArenaHandler(BaseHTTPRequestHandler):
    server_version = "FlyBrainFlowArena/0.1"

    def _json(self, data, status=200, *, download=False):
        try:
            payload = json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            status = 500
            payload = b'{"error":"Simulation returned an invalid JSON value"}'
            download = False
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="arena-experiment.json"')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        actions = {"/api/state": self.server.session.state,
                   "/api/environment": self.server.session.environment,
                   "/api/export": self.server.session.export}
        if path in actions:
            return self._json(actions[path](), download=path == "/api/export")
        if path.startswith("/api/"):
            return self._json({"error": "Unknown endpoint"}, 404)
        # Resolve symlinks and encoded separators as well as '..'; never serve
        # directories or any file outside the intentionally narrow viewer root.
        name = "arena.html" if path == "/" else unquote(path).lstrip("/")
        candidate = (self.server.viewer_dir / name).resolve()
        if not candidate.is_relative_to(self.server.viewer_dir) or not candidate.is_file():
            return self._json({"error": "Not found"}, 404)
        try:
            payload = candidate.read_bytes()
        except OSError:
            return self._json({"error": "Not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self._json({"error": "Cross-origin requests are not allowed"}, 403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json({"error": "Use application/json"}, 415)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192:
                raise ValueError("JSON body must be 1–8192 bytes")
            values = json.loads(self.rfile.read(length))
            if not isinstance(values, dict):
                raise ValueError("JSON body must be an object")
            path = urlsplit(self.path).path
            handlers = {"/api/step": self.server.session.step,
                        "/api/control": self.server.session.control,
                        "/api/reset": self.server.session.reset}
            if path == "/api/export":
                if values:
                    raise ValueError("Export takes an empty object")
                return self._json(self.server.session.export(), download=True)
            if path not in handlers:
                return self._json({"error": "Unknown endpoint"}, 404)
            return self._json(handlers[path](values))
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        except (OSError, RuntimeError, ImportError) as exc:
            self.log_error("Arena request failed: %s", exc)
            return self._json({"error": str(exc)}, 500)


def create_arena_server(session, *, host="127.0.0.1", port=8766, viewer_dir=None):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Arena service currently supports loopback hosting only")
    server = ThreadingHTTPServer((host, port), ArenaHandler)
    server.daemon_threads = True
    server.session = session
    server.viewer_dir = Path(viewer_dir or Path(__file__).resolve().parents[1] / "viewer").resolve()
    return server
