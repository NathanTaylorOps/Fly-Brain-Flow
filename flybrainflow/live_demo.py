"""Small, local live-simulation service. No prerecorded motion or background stepping.

The browser requests bounded advances; all calls serialize under one session lock.
An injected ``sim_factory(seed)`` supplies a real controller without changing HTTP.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from .scenario import Scenario
from .sim import Sim


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _integer(value, name, low, high):
    value = _number(value, name, low, high)
    if int(value) != value:
        raise ValueError(f"{name} must be a whole number")
    return int(value)


def default_scenario(seed=42):
    return Scenario.from_dict({
        "scenario": {"name": "live_single_fly", "seed": seed},
        "map": {"source": "gen:corridor?length=12&width=8&open_ends=1", "walkable_grid_m": 0.2},
        "spawn": {"sources": [[2.0, 4.0]], "rate_per_s": 10, "population_cap": 1},
        "targets": [{"kind": "sugar", "position": [10.0, 4.0], "feeding_time_s": 5}],
        "baseline": {"enabled": False},
    })


def toy_factory(seed):
    """Explicit plumbing mode, never described as a measured connectome."""
    return Sim.from_scenario(default_scenario(seed), rng=np.random.default_rng(seed))


class LiveMazeSim(Sim):
    """Advance the odor transport before the controller samples this world tick."""
    def tick(self, dt):
        self.odor_field.advance(dt)
        super().tick(dt)


def maze_sim(seed, brain_factory=None, *, warmup_s=60.0):
    from .brains.forager import ForagerBrain
    from .world.live_maze import maze_map, DynamicOdorField
    from .world.fields import Wind, OdorSource
    from .world.airflow import AirflowField

    arena = maze_map()
    scenario = Scenario.from_dict({
        "scenario": {"name": "live_odor_maze", "seed": seed},
        "map": {"source": "gen:corridor?length=16&width=12&open_ends=1", "walkable_grid_m": arena.resolution},
        "wind": {"direction_deg": 180, "speed_mps": 1.3},
        "spawn": {"sources": [[1.5, 6.0]], "rate_per_s": 10, "population_cap": 1},
        "targets": [{"kind": "sugar", "position": [14.5, 7.0], "feeding_time_s": 6}],
        "agents": {"radius_m": 0.16, "max_speed_mps": 1.4},
        "baseline": {"enabled": False},
    })
    field = DynamicOdorField(Wind(180, 1.3),
                             [OdorSource("sugar", (14.5, 7.0), strength=1.0)],
                             walkable_map=arena, airflow=AirflowField.build(arena))
    if warmup_s:
        field.advance(warmup_s)
    brain = brain_factory(seed, arena) if brain_factory else ForagerBrain(arena, seed=seed)
    sim = LiveMazeSim(scenario, walkable_map=arena, odor_field=field,
                      brains={"brain": brain}, rng=np.random.default_rng(seed))
    sim.cohorts["brain"].spawn(0.1)
    return sim


def maze_session(*, seed=42, connectome_factory=None, connectome_dataset=None, warmup_s=60):
    controllers = {"behavior": {
        "factory": lambda seed: maze_sim(seed, warmup_s=warmup_s),
        "mode": "Behavioral foraging model",
        "dataset": {"name": "Local odor sensing and exploration", "real_connectome": False},
        "notice": "Behavior model · local odor sensing and exploration; not connectome-controlled. Odor transport is an approximate grid model, prewarmed before the fly starts.",
    }}
    if connectome_factory is not None:
        controllers["connectome"] = {
            "factory": lambda seed: maze_sim(seed, connectome_factory, warmup_s=warmup_s),
            "mode": "Real connectome · experimental decoder",
            "dataset": connectome_dataset or {"name": "MaleCNS", "real_connectome": True},
            "notice": "Real MaleCNS neural simulation with an engineered 2D movement decoder. Food-seeking is not validated. Connections below 3 synapses are omitted.",
        }
    return LiveSession(seed=seed, controllers=controllers, controller="behavior")


class LiveSession:
    def __init__(self, sim_factory=None, *, seed=42, mode="toy", dataset=None, notice=None,
                 controllers=None, controller=None):
        self.sim_factory = sim_factory or toy_factory
        self.mode = mode
        self.dataset = dataset or {"name": "Synthetic toy network", "real_connectome": False}
        self.notice = notice or "Plumbing demonstration: movement uses the synthetic ToyBrain, not measured fly wiring."
        self.controllers = controllers or {}
        self.controller = controller or ("behavior" if "behavior" in self.controllers else "toy")
        if self.controllers:
            self._select_controller(self.controller)
        self.lock = threading.RLock()
        self.reset(seed)

    def _select_controller(self, name):
        spec = self.controllers[name]
        self.sim_factory, self.mode = spec["factory"], spec["mode"]
        self.dataset, self.notice = spec["dataset"], spec["notice"]
        self.controller = name

    def reset(self, seed=None):
        with self.lock:
            seed = _integer(self.seed if seed is None else seed, "seed", 0, 2**32 - 1)
            # Build before replacing so a failed load leaves the previous session intact.
            sim = self.sim_factory(seed)
            self.sim = sim
            self.seed = seed
            self.steps = 0
            self.wall_s = 0.0
            self.controls = {"stimulus_hz": 0.0, "silenced": False, "stimulus_group": "loom", "odor_enabled": bool(getattr(sim.brains['brain'], 'odor_enabled', False))}
            self._previous_positions = {}
            self._headings = {}
            return self.state()

    def step(self, *, steps=1, dt=0.05):
        steps = _integer(steps, "steps", 1, 20)
        dt = float(_number(dt, "dt", 0.001, 0.1))
        with self.lock:
            started = time.perf_counter()
            for _ in range(steps):
                before = {a.id: a.position.copy() for p in self.sim.cohorts.values() for a in p.agents.values()}
                self.sim.tick(dt)
                for p in self.sim.cohorts.values():
                    for a in p.agents.values():
                        if a.id in before:
                            delta = a.position - before[a.id]
                            if np.linalg.norm(delta) > 1e-10:
                                self._headings[a.id] = float(np.arctan2(delta[1], delta[0]))
                live_ids = {a.id for p in self.sim.cohorts.values() for a in p.agents.values()}
                self._headings = {i: h for i, h in self._headings.items() if i in live_ids}
                self.steps += 1
            self.wall_s += time.perf_counter() - started
            return self.state()

    def control(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("control must be a nonempty object")
        unknown = set(values) - {"target", "stimulus_hz", "silenced", "stimulus_group", "odor_enabled", "controller", "wind_direction_deg", "wind_speed_mps"}
        if unknown:
            raise ValueError(f"Unknown control: {sorted(unknown)[0]}")
        with self.lock:
            if "controller" in values:
                name = values["controller"]
                if not isinstance(name, str) or name not in self.controllers:
                    raise ValueError("Requested controller is unavailable")
                # Switching is its own transaction; do not partially apply a compound request.
                if len(values) != 1:
                    raise ValueError("Switch controller separately from other controls")
                if name != self.controller:
                    old_field = self.sim.odor_field
                    old_spec = (self.sim_factory, self.mode, self.dataset, self.notice, self.controller)
                    self._select_controller(name)
                    try:
                        self.reset(self.seed)
                    except Exception:
                        self.sim_factory, self.mode, self.dataset, self.notice, self.controller = old_spec
                        raise
                    # The maze is shared by both controllers. Preserve the full transport
                    # field so a switch keeps existing odor history as well as wind/food.
                    self.sim.odor_field = old_field
                    for i, source in enumerate(old_field.sources):
                        self.sim.move_target(i, source.position)
                return self.state()
            brain = self.sim.brains["brain"]
            caps = self._capabilities()
            wind = {}
            if "wind_direction_deg" in values:
                wind["direction_deg"] = float(_number(values["wind_direction_deg"], "wind_direction_deg", 0, 360)) % 360
            if "wind_speed_mps" in values:
                wind["speed_mps"] = float(_number(values["wind_speed_mps"], "wind_speed_mps", 0, 3))
            target = None
            if "target" in values:
                target = values["target"]
                if not isinstance(target, dict) or set(target) != {"index", "position"}:
                    raise ValueError("target must contain index and position")
                index = _integer(target["index"], "target.index", 0, len(self.sim.odor_field.sources) - 1)
                xy = target["position"]
                if not isinstance(xy, list) or len(xy) != 2:
                    raise ValueError("target.position must be [x,y]")
                xy = [float(_number(v, "target.position", -1e6, 1e6)) for v in xy]
                if not self.sim.map.is_walkable([xy])[0]:
                    raise ValueError("Target must be inside the walkable arena")
                target = index, xy
            updates = {k: values[k] for k in ("stimulus_hz", "silenced", "stimulus_group", "odor_enabled") if k in values}
            if updates and not callable(getattr(brain, "set_controls", None)):
                raise ValueError("This controller does not support neural stimulation or silencing")
            if ("stimulus_hz" in updates or "stimulus_group" in updates) and not caps["stimulus"]:
                raise ValueError("Neural stimulation is unavailable in behavioral mode")
            if "silenced" in updates and not caps["silence"]:
                raise ValueError("Neural silencing is unavailable in behavioral mode")
            if "odor_enabled" in updates and not caps["odor"]:
                raise ValueError("Odor sensing control is unavailable")
            if "stimulus_hz" in updates:
                updates["stimulus_hz"] = float(_number(updates["stimulus_hz"], "stimulus_hz", 0, 1000))
            if "silenced" in updates and not isinstance(updates["silenced"], bool):
                raise ValueError("silenced must be true or false")
            if "odor_enabled" in updates and not isinstance(updates["odor_enabled"], bool):
                raise ValueError("odor_enabled must be true or false")
            if "stimulus_group" in updates and updates["stimulus_group"] not in ("sugar", "loom"):
                raise ValueError("stimulus_group must be sugar or loom")
            if updates:
                brain.set_controls(**updates)
                self.controls.update(updates)
            if target:
                self.sim.move_target(*target)
            if wind:
                self.sim.odor_field.wind.set(**wind)
            return self.state()

    def _capabilities(self):
        brain = self.sim.brains["brain"]
        supported = callable(getattr(brain, "set_controls", None))
        declared = getattr(brain, "control_capabilities", {})
        return {"stimulus": declared.get("stimulus", supported),
                "silence": declared.get("silence", supported),
                "odor": declared.get("odor", supported), "wind": True}

    def environment(self):
        with self.lock:
            snapshot = getattr(self.sim.odor_field, "environment_snapshot", None)
            return snapshot() if callable(snapshot) else {}

    def state(self):
        with self.lock:
            brain = self.sim.brains["brain"]
            telemetry = getattr(brain, "neural_snapshot", None)
            neural = telemetry() if callable(telemetry) else {}
            behavioral = getattr(brain, "behavioral_snapshot", None)
            headings = getattr(brain, "headings", getattr(brain, "_heading", self._headings))
            agents = [{
                "id": int(a.id), "position": a.position.tolist(), "cohort": tag,
                "status": a.status, "heading": float(headings.get(a.id, self._headings.get(a.id, 0.0))),
            } for tag, pop in self.sim.cohorts.items() for a in pop.agents.values()]
            wind = self.sim.odor_field.wind
            return {
                "time_s": self.sim.t, "seed": self.seed, "mode": self.mode,
                "controller": self.controller,
                "available_controllers": list(self.controllers) or [self.controller],
                "dataset": self.dataset() if callable(self.dataset) else self.dataset, "notice": self.notice,
                "rtf": self.sim.t / self.wall_s if self.wall_s else None,
                "controls": {**self.controls, "wind_direction_deg": wind.direction_deg,
                             "wind_speed_mps": wind.speed_mps},
                "arena": {"bounds": list(self.sim.map.bounds),
                          "outer": [r.tolist() for r in self.sim.map.outer],
                          "holes": [r.tolist() for r in self.sim.map.holes]},
                "targets": [{"index": i, "position": list(s.position), "kind": self.sim.scenario.targets[i].kind}
                            for i, s in enumerate(self.sim.odor_field.sources)],
                "agents": agents, "neural": neural,
                "behavior": behavioral() if callable(behavioral) else {},
                "capabilities": self._capabilities(),
                "metrics": {"steps": self.steps, "wall_s": self.wall_s, "simulated_s": self.sim.t},
            }


class LiveHandler(SimpleHTTPRequestHandler):
    server_version = "FlyBrainFlow/0.1"

    def _json(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/state":
            return self._json(self.server.session.state())
        if path == "/api/environment":
            return self._json(self.server.session.environment())
        if path == "/api/anatomy":
            brain = self.server.session.sim.brains['brain']
            return self._json(brain.anatomy_snapshot() if hasattr(brain, 'anatomy_snapshot') else {})
        if path.startswith("/api/"):
            return self._json({"error": "Unknown endpoint"}, 404)
        if path == "/":
            self.path = "/maze.html"
        super().do_GET()

    def do_POST(self):
        # Local service: foreign websites must not be able to mutate its simulation.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self._json({"error": "Cross-origin requests are not allowed"}, 403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json({"error": "Use application/json"}, 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("JSON body must be 1–8192 bytes")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            path = urlsplit(self.path).path
            if path == "/api/step":
                if set(data) - {"steps", "dt"}:
                    raise ValueError("step accepts only steps and dt")
                result = self.server.session.step(**data)
            elif path == "/api/reset":
                if set(data) - {"seed"}:
                    raise ValueError("reset accepts only seed")
                result = self.server.session.reset(**data)
            elif path == "/api/control":
                result = self.server.session.control(data)
            else:
                return self._json({"error": "Unknown endpoint"}, 404)
            return self._json(result)
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        except (OSError, RuntimeError, ImportError) as exc:
            # A missing optional connectome bundle/dependency must leave a useful
            # message in the browser instead of silently dropping its connection.
            self.log_error("Simulation request failed: %s", exc)
            return self._json({"error": str(exc)}, 500)


def create_server(session, *, host="127.0.0.1", port=8765, viewer_dir=None):
    viewer_dir = viewer_dir or Path(__file__).resolve().parents[1] / "viewer"
    server = ThreadingHTTPServer((host, port), partial(LiveHandler, directory=str(viewer_dir)))
    server.daemon_threads = True
    server.session = session
    return server
