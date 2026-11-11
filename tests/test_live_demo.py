import json
import threading
import urllib.error
import urllib.request

import pytest

from flybrainflow.live_demo import LiveSession, create_server, maze_session


def test_live_step_and_reset_are_reproducible():
    session = LiveSession(seed=7)
    assert session.state()["dataset"]["real_connectome"] is False
    first = session.step(steps=10, dt=0.1)
    assert first["time_s"] == pytest.approx(1.0)
    assert len(first["agents"]) == 1
    assert first["rtf"] > 0
    session.reset(7)
    second = session.step(steps=10, dt=0.1)
    assert first["agents"] == second["agents"]


def test_control_rejects_unsupported_neural_mode_and_invalid_positions():
    session = LiveSession()
    with pytest.raises(ValueError, match="does not support"):
        session.control({"silenced": True})
    with pytest.raises(ValueError, match="walkable"):
        session.control({"target": {"index": 0, "position": [-100, 5]}})
    state = session.control({"target": {"index": 0, "position": [7, 3]}})
    assert state["targets"][0]["position"] == [7, 3]
    assert session.sim.target_moves == [(0, 0, (7, 3))]


@pytest.mark.parametrize("kwargs", [{"dt": float("nan")}, {"dt": 1}, {"steps": 21}, {"steps": 1.5}])
def test_steps_are_bounded(kwargs):
    session = LiveSession()
    with pytest.raises(ValueError):
        session.step(**kwargs)
    assert session.sim.t == 0


def test_http_advances_and_rejects_foreign_control():
    server = create_server(LiveSession(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        req = urllib.request.Request(url + "/api/step", data=b'{"steps":2,"dt":0.1}', headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            assert json.load(response)["time_s"] == pytest.approx(0.2)
        req = urllib.request.Request(url + "/api/reset", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://foreign.invalid"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        assert error.value.code == 403
        assert server.session.sim.t == pytest.approx(0.2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_maze_transport_controls_and_honest_behavior_mode():
    session = maze_session(seed=7, warmup_s=1)
    state = session.state()
    assert state["controller"] == "behavior"
    assert state["dataset"]["real_connectome"] is False
    assert not state["capabilities"]["stimulus"]
    assert not state["capabilities"]["silence"]
    assert state["neural"] == {}
    assert len(state["agents"]) == 1  # Visible immediately, before the first tick.
    env = session.environment()
    assert env["time_s"] == pytest.approx(1)
    assert state["time_s"] == 0
    assert sum(env["concentration"]) > 0
    updated = session.control({"wind_direction_deg": 270, "wind_speed_mps": 2.0})
    assert updated["controls"]["wind_speed_mps"] == 2
    session.step(steps=2, dt=0.1)
    assert session.environment()["time_s"] == pytest.approx(1.2)
    with pytest.raises(ValueError, match="behavioral"):
        session.control({"stimulus_hz": 100, "wind_speed_mps": 3})
    assert session.state()["controls"]["wind_speed_mps"] == 2  # Reject atomically.


def test_controller_switch_preserves_environment_and_resets_agent_clock():
    session = maze_session(seed=9, warmup_s=0)
    # A second test controller exercises mode-switching without a giant graph fixture.
    session.controllers["alternative"] = {**session.controllers["behavior"], "mode": "Test controller"}
    session.control({"wind_direction_deg": 90, "wind_speed_mps": 0.5})
    session.control({"target": {"index": 0, "position": [13, 3]}})
    session.step(steps=3, dt=0.1)
    field = session.sim.odor_field
    state = session.control({"controller": "alternative"})
    assert session.sim.odor_field is field
    assert state["time_s"] == 0
    assert state["controls"]["wind_direction_deg"] == 90
    assert state["targets"][0]["position"] == [13, 3]
    assert state["mode"] == "Test controller"
    assert state["metrics"]["steps"] == 0
