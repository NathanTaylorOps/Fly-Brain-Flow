import json
import threading
import urllib.error
import urllib.request

import pytest

from flybrainflow.arena_server import ArenaSession, create_arena_server


class SmallSim:
    """Isolate HTTP/session guarantees from the physical model's own tests."""
    def __init__(self, config):
        self.config = dict(config)
        self.time = 0
        self.controls = []
        if config["seed"] == 999:
            raise ValueError("Fixture failed to initialize")

    def step(self, *, steps, dt):
        self.time += steps * dt

    def state(self):
        return {"time": self.time, "config": self.config, "agents": [], "metrics": {}}

    def environment(self):
        return {"bounds": [0, 0, 16, 12]}

    def control(self, values):
        self.controls.append(values)

    def export(self):
        return {"state": self.state(), "model": "test fixture"}


def test_bounded_steps_and_atomic_reset():
    session = ArenaSession(sim_factory=SmallSim)
    stepped = session.step({"steps": 2, "dt": .05})
    assert stepped["time"] == pytest.approx(.1)
    assert stepped["performance"]["steps"] == 2
    assert stepped["performance"]["compute_rtf"] > 0
    with pytest.raises(ValueError, match="initialize"):
        session.reset({"seed": 999})
    assert session.state()["time"] == pytest.approx(.1)
    assert session.config["seed"] == 7
    reset = session.reset({"population": 300, "scenario": "blocked"})
    assert reset["time"] == 0
    assert reset["config"]["preset"] == "blocked"
    assert reset["config"]["population"] == 300
    assert reset["performance"]["compute_rtf"] is None


@pytest.mark.parametrize("values", [{"steps": 11}, {"steps": 0}, {"steps": True},
                                   {"dt": float("nan")}, {"dt": .1}, {"unexpected": 2}])
def test_invalid_steps_never_advance(values):
    session = ArenaSession(sim_factory=SmallSim)
    with pytest.raises(ValueError):
        session.step(values)
    assert session.state()["time"] == 0


def test_controls_validate_before_mutation():
    session = ArenaSession(sim_factory=SmallSim)
    session.control({"wind_direction_deg": 90, "wind_speed_mps": 2})
    assert len(session.sim.controls) == 1
    for bad in ({"alarm": "true"}, {"wind_speed_mps": float("inf")},
                {"exit": {"id": "east", "open": 1}},
                {"hazard_position": [1, float("nan")]},
                {"alarm": True, "hazard_strength": 1}, {"population": 500}):
        with pytest.raises(ValueError):
            session.control(bad)
    assert len(session.sim.controls) == 1


def test_preset_switch_resets_compute_clock_and_preserves_environment_config():
    session = ArenaSession(sim_factory=SmallSim)
    session.control({"hazard_position": [20, 20]})
    session.control({"wind_speed_mps": 2})
    session.step({"steps": 2})
    state = session.control({"preset": "surge"})
    assert state["time"] == 0
    assert state["performance"]["steps"] == 0
    assert state["config"]["wind_speed_mps"] == 2
    assert state["config"]["hazard_position"] == [20, 20]


def test_only_loopback_hosting():
    with pytest.raises(ValueError, match="loopback"):
        create_arena_server(ArenaSession(sim_factory=SmallSim), host="0.0.0.0")


def test_http_json_export_and_path_containment(tmp_path):
    root = tmp_path / "viewer"
    root.mkdir()
    (root / "arena.html").write_text("<html>Arena</html>", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside-viewer-root", encoding="utf-8")
    session = ArenaSession(sim_factory=SmallSim)
    server = create_arena_server(session, port=0, viewer_dir=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/") as response:
            assert b"Arena" in response.read()
        with opener.open(base + "/api/export") as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert json.load(response)["model"] == "test fixture"
        req = urllib.request.Request(base + "/api/step", data=b'{"steps":2,"dt":0.05}',
                                     headers={"Content-Type": "application/json"})
        with opener.open(req) as response:
            assert json.load(response)["time"] == pytest.approx(.1)
        for path in ("/%2e%2e/outside.txt", "/%2e%2e%5coutside.txt", "/api/missing"):
            with pytest.raises(urllib.error.HTTPError) as failure:
                opener.open(base + path)
            assert failure.value.code == 404
        req = urllib.request.Request(base + "/api/reset", data=b'{}',
                                     headers={"Content-Type": "application/json", "Origin": "https://foreign.invalid"})
        with pytest.raises(urllib.error.HTTPError) as failure:
            opener.open(req)
        assert failure.value.code == 403
        assert session.state()["time"] == pytest.approx(.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_arena_emits_finite_json():
    session = ArenaSession({"population": 50, "seed": 7})
    session.control({"alarm": True})
    state = session.step({"steps": 2, "dt": .05})
    assert len(state["agents"]) <= 50
    assert state.get("time_s", state.get("time")) == pytest.approx(.1)
    for payload in (state, session.environment(), session.export()):
        json.dumps(payload, allow_nan=False)
