"""Local HTTP server for the browser interface; the simulation runs in this process."""

import json
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from anasim.core.recorder import RecordingError
from anasim.web import WebSession, catalog

ASSETS = Path(__file__).with_name("web_assets")
# Fixed types: Windows registry mappings can serve .js as text/plain, which blocks modules.
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json"}
DISCONNECT_SECONDS = 5.0


class LocalServer(ThreadingHTTPServer):
    """One session, with serialized commands and a server-owned clock."""

    def __init__(self, port=0, recordings_dir="recordings"):
        self.lock = threading.RLock()
        self.session = None
        self.client = None
        self.error = None
        self.closed = False
        self.last_seen = self.last_tick = time.monotonic()
        self.recordings_dir = str(Path(recordings_dir).resolve())
        self.prefix = f"/{secrets.token_urlsafe(24)}/"
        super().__init__(("127.0.0.1", port), LocalHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.url = self.origin + self.prefix

    def service_actions(self):
        with self.lock:
            now = time.monotonic()
            dt = now - self.last_tick
            if dt < 0.05:
                return
            self.last_tick = now
            if self.session is None:
                return
            if self.client and now - self.last_seen > DISCONNECT_SECONDS:
                self.detach()
            if self.client and not self.error:
                try:
                    self.session.step(dt)
                except Exception as error:
                    self.error = str(error)
                    self.finish_recording()

    def finish_recording(self):
        if self.session is not None:
            try:
                self.session.close()
            except RecordingError as error:
                self.session.notice = str(error)
                print(f"Recording failed: {error}", file=sys.stderr)

    def detach(self):
        self.finish_recording()
        self.client = None

    def dispatch(self, message):
        """Return worker-protocol messages. Caller holds the session lock."""
        if self.closed:
            raise PermissionError("AnaSim has shut down")
        kind = message["type"]
        client = message["client"]
        if not isinstance(client, str) or not client:
            raise ValueError("A browser connection ID is required")
        if kind == "init":
            self.finish_recording()
            self.client = client
            self.last_seen = self.last_tick = time.monotonic()
            messages = [{"type": "ready", "catalog": json.loads(catalog())}]
            if self.session is not None:
                self.session.replay_waves()
                messages.append({"type": "created", "info": json.loads(self.session.info())})
                messages.append({"type": "tick", "snap": json.dumps(self.session.snapshot())})
                if self.error:
                    messages.append({"type": "error", "message": self.error})
            return messages
        if client != self.client:
            raise PermissionError("This connection has closed")
        self.last_seen = time.monotonic()
        if kind == "detach":
            self.detach()
            return []
        if kind == "create":
            try:
                session = WebSession(message["params"], self.recordings_dir, retain_recordings=True)
            except Exception as error:  # reported like the browser worker does
                return [{"type": "create_error", "message": str(error)}]
            self.finish_recording()
            self.session = session
            self.error = None
            self.last_tick = time.monotonic()
            return [
                {"type": "created", "info": json.loads(session.info())},
                {"type": "tick", "snap": json.dumps(session.snapshot())},
            ]
        if kind == "poll":
            if self.error:
                return [{"type": "error", "message": self.error}]
            return [] if self.session is None else [
                {"type": "tick", "snap": json.dumps(self.session.snapshot())}
            ]
        if kind == "cmd":
            result = {"type": "result", "id": message["id"]}
            try:
                if self.session is None or self.error:
                    raise ValueError("The simulation has stopped. Start a new session to continue.")
                result["value"] = json.loads(self.session.command(message["name"], json.dumps(message.get("args", {}))))
            except Exception as error:  # reported like the browser worker does
                result["error"] = str(error)
            return [result]
        raise ValueError(f"Unknown message: {kind!r}")

    def server_close(self):
        with self.lock:
            self.closed = True
            self.detach()
        super().server_close()


class LocalHandler(BaseHTTPRequestHandler):
    """Serve packaged assets and same-origin JSON commands only."""

    timeout = 2.0
    server: LocalServer

    def log_message(self, *_args):
        pass

    def reply(self, status, body, content_type="application/json"):
        if isinstance(body, (list, dict)):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def route(self):
        if self.headers.get("Host") != urlsplit(self.server.origin).netloc:
            return None
        path = urlsplit(self.path).path
        if not path.startswith(self.server.prefix):
            return None
        return path[len(self.server.prefix):]

    def do_GET(self):
        route = self.route()
        if route == "runtime.json":
            self.reply(200, {"backend": "native"})
        elif route is not None and (not route or ("/" not in route and route in {p.name for p in ASSETS.iterdir()})):
            path = ASSETS / (route or "index.html")
            self.reply(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if (self.route() != "api"
                or self.headers.get("Origin") not in (None, self.server.origin)
                or self.headers.get_content_type() != "application/json"):
            self.reply(403, {"error": "Request rejected"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size")
            message = json.loads(self.rfile.read(length))
            if not isinstance(message, dict):
                raise ValueError("Expected a JSON object")
            with self.server.lock:
                messages = self.server.dispatch(message)
        except PermissionError as error:
            self.reply(409, {"error": str(error)})
        except (ValueError, TypeError, KeyError) as error:
            self.reply(400, {"error": str(error)})
        else:
            self.reply(200, messages)


def run(*, port=0, open_browser=True, recordings_dir="recordings") -> int:
    with LocalServer(port, recordings_dir) as server:
        print(f"AnaSim: {server.url}", flush=True)
        print(f"Recordings: {server.recordings_dir}\nPress Ctrl+C to stop.", flush=True)
        if open_browser and not webbrowser.open(server.url):
            print("Open the URL above in a browser.", flush=True)
        try:
            server.serve_forever(poll_interval=0.02)
        except KeyboardInterrupt:
            pass
    return 0
