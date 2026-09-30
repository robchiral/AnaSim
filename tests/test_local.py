"""Native HTTP transport, session recovery, and recording lifecycle."""

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from anasim.local import DISCONNECT_SECONDS, LocalServer


@pytest.fixture
def server(tmp_path):
    with LocalServer(recordings_dir=tmp_path) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(server, kind, *, client="browser", **fields):
    body = json.dumps({"type": kind, "client": client, **fields}).encode()
    req = Request(server.url + "api", body, {"Content-Type": "application/json", "Origin": server.origin})
    with urlopen(req, timeout=5) as response:
        return json.load(response)


def test_native_session_recovers_waveforms_controls_and_recording(server, tmp_path):
    with urlopen(server.url) as page:
        assert b"app.js" in page.read()
    with urlopen(server.url + "runtime.json") as runtime:
        assert json.load(runtime) == {"backend": "native"}
    assert request(server, "init")[0]["type"] == "ready"
    request(server, "create", params={"mode": "awake"})
    request(server, "cmd", id=1, name="record", args={"active": True})
    request(server, "cmd", id=2, name="airway", args={"mode": "ETT"})
    request(server, "cmd", id=3, name="run", args={"running": True})
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snap = json.loads(request(server, "poll")[0]["snap"])
        if snap["time"] >= 0.15:
            break
        time.sleep(0.02)
    assert snap["time"] >= 0.15
    assert snap["waves"]["ecg"]
    assert snap["recording"]
    old_session = server.session

    restored = request(server, "init", client="reloaded")
    snap = json.loads(restored[2]["snap"])
    assert server.session is old_session
    assert not snap["running"] and not snap["recording"]
    assert snap["controls"]["airway"] == "ETT"
    assert len(snap["waves"]["ecg"]) >= 15
    assert not old_session.engine.recorder.is_recording
    assert len(list(tmp_path.glob("*.csv"))) == 1
    assert len(next(tmp_path.glob("*.csv")).read_text().splitlines()) >= 2
    with pytest.raises(HTTPError, match="409"):
        request(server, "cmd", id=4, name="run", args={"running": True})

    request(server, "cmd", client="reloaded", id=5, name="record", args={"active": True})
    # Local recordings are kept on disk only, not also sent for download.
    assert request(server, "cmd", client="reloaded", id=6, name="record", args={"active": False})[0]["value"] is None
    assert len(list(tmp_path.glob("*.csv"))) == 2
    request(server, "cmd", client="reloaded", id=7, name="record", args={"active": True})
    request(server, "detach", client="reloaded")
    assert not server.session.engine.recorder.is_recording


def test_disconnect_timeout_and_shutdown_finish_recordings(server, tmp_path):
    request(server, "init")
    request(server, "create", params={})
    request(server, "cmd", id=1, name="record", args={"active": True})
    with server.lock:
        server.last_seen -= DISCONNECT_SECONDS + 1
        server.last_tick -= 1
        server.service_actions()
        assert server.client is None
        assert not server.session.engine.recorder.is_recording
    request(server, "init")
    request(server, "cmd", id=2, name="record", args={"active": True})
    server.shutdown()
    server.server_close()
    assert not server.session.engine.recorder.is_recording
    assert len(list(tmp_path.glob("*.csv"))) == 2


def test_local_server_rejects_cross_origin_and_unscoped_requests(server):
    for req in (
        Request(server.origin + "/runtime.json"),
        Request(server.url + "api", b'{}', {"Content-Type": "application/json", "Origin": "https://example.com"}),
        Request(server.url + "api", b'{}', {"Content-Type": "text/plain"}),
        Request(server.url + "runtime.json", headers={"Host": "example.com"}),
        Request(server.url + "../pyproject.toml"),
    ):
        with pytest.raises(HTTPError):
            urlopen(req, timeout=3)
