"""
vasimov/web/server/gateway.py
Unified Web Control & Telemetry Gateway for Virtual Asimov 1.

Features:
- Serves the technical workstation web interface (HTML, CSS, JS).
- Backward-compatible endpoints: /ground_truth, /status, /control.
- Modern REST API: /api/state, /api/control, /api/camera/*, /api/recordings.
- Live MJPEG camera video streaming at ~25 FPS with sequence and sim_time headers.
- Sim-only WebSocket streaming server broadcasting canonical TelemetryFrame.
- Real-time command dispatch with explicit lifecycle acknowledgement.
"""

from __future__ import annotations
import asyncio
import http.server
import json
import logging
import mimetypes
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

import websockets

from edge.core import EdgeCore
from edge.sim import SimBackend, compute_projected_gravity
from web.server.canonical_frame import CanonicalFrameBuilder
from web.server.camera_streamer import CameraStreamer
from web.server.control_handler import ControlHandler
from web.server.replay_manager import ReplayManager

log = logging.getLogger("vasimov.web.gateway")

_CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"


class WebGateway:
    """Central HTTP & WebSocket Server Gateway for Virtual Asimov 1."""

    def __init__(
        self,
        backend: SimBackend,
        core: EdgeCore,
        telemetry_engine=None,
        http_port: int = 8852,
        ws_port: int = 8854,
        host: str = "0.0.0.0",
    ):
        self.backend = backend
        self.core = core
        self.telemetry = telemetry_engine
        self.http_port = http_port
        self.ws_port = ws_port
        self.host = host

        self.running: bool = False
        self._shutdown_event = threading.Event()

        # Subsystems
        self.frame_builder = CanonicalFrameBuilder(backend, core)
        self.camera_streamer = CameraStreamer(backend)
        self.replay_manager = ReplayManager()
        self.control_handler = ControlHandler(backend, core, telemetry_engine, replay_manager=self.replay_manager)

        # State cache
        self.latest_frame: Optional[Dict[str, Any]] = None
        self.latest_json: str = "{}"
        self.broadcast_sequence: int = 0

        # Threads and servers
        self.http_server: Optional[http.server.HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

        self.ws_clients: Set[Any] = set()
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_server = None

        # Telemetry broadcast thread
        self._broadcast_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start HTTP server, WebSocket server, and Camera streamer."""
        if self.running:
            return
        self.running = True
        self._shutdown_event.clear()

        # 1. Start Camera Streamer worker
        self.camera_streamer.start()

        # 2. Start HTTP server
        self._start_http()

        # 3. Start WebSocket server
        self._start_websocket()

        # 4. Start periodic telemetry broadcast loop (~25-30 Hz)
        self._broadcast_thread = threading.Thread(
            target=self._telemetry_broadcast_loop,
            daemon=True,
            name="vasimov-telemetry-broadcast",
        )
        self._broadcast_thread.start()

        log.info(
            "[GATEWAY] Started Virtual Asimov Web Gateway:\n"
            "   Dashboard: http://%s:%d\n"
            "   WebSocket: ws://%s:%d\n"
            "   Cameras  : %s",
            self.host,
            self.http_port,
            self.host,
            self.ws_port,
            self.camera_streamer.all_cameras,
        )

    def stop(self) -> None:
        """Stop all gateway servers cleanly."""
        self.running = False
        self._shutdown_event.set()

        # Stop Camera Streamer
        self.camera_streamer.stop()

        # Stop HTTP server
        if self.http_server is not None:
            try:
                self.http_server.shutdown()
                self.http_server.server_close()
            except Exception:
                pass
            self.http_server = None

        # Stop WebSocket loop
        if self._ws_loop is not None and self._ws_loop.is_running():
            try:
                self._ws_loop.call_soon_threadsafe(self._ws_loop.stop)
            except Exception:
                pass

        log.info("[GATEWAY] Stopped.")

    # ── Telemetry Broadcast Loop ─────────────────────────────────────────────

    def _telemetry_broadcast_loop(self) -> None:
        """Periodic canonical telemetry broadcaster (~25-30 Hz)."""
        while self.running and not self._shutdown_event.is_set():
            t0 = time.perf_counter()

            if self.replay_manager.active:
                if self.replay_manager.playing:
                    fr = self.replay_manager.step(1)
                else:
                    fr = self.replay_manager.get_current_frame()
                frame = {
                    "replay": True,
                    "replay_data": fr,
                    **self.frame_builder.build_frame(
                        event_log=self.control_handler.recent_events,
                        camera_meta=self.camera_streamer.get_camera_metadata(),
                    ),
                }
            else:
                frame = self.frame_builder.build_frame(
                    event_log=self.control_handler.recent_events,
                    camera_meta=self.camera_streamer.get_camera_metadata(),
                )

            self.latest_frame = frame
            try:
                payload = json.dumps(frame)
                self.latest_json = payload
                self._broadcast_ws(payload)
            except Exception as e:
                log.debug("[GATEWAY] Serialization error: %s", e)

            # Sleep to match ~25-30 Hz
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.005, (1.0 / 25.0) - elapsed)
            time.sleep(sleep_t)

    # ── WebSocket Server ─────────────────────────────────────────────────────

    def _start_websocket(self) -> None:
        """Launch background WebSocket server thread."""
        self._ws_thread = threading.Thread(
            target=self._run_ws_event_loop,
            daemon=True,
            name="vasimov-ws-gateway",
        )
        self._ws_thread.start()

    def _run_ws_event_loop(self) -> None:
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)

        async def handler(websocket):
            self.ws_clients.add(websocket)
            # Immediately send latest state
            if self.latest_json and self.latest_json != "{}":
                try:
                    await websocket.send(self.latest_json)
                except Exception:
                    pass

            try:
                async for message in websocket:
                    # Handle client command over websocket
                    try:
                        data = json.loads(message)
                        action = data.get("action", "")
                        params = data.get("params", data.get("command_data", {}))
                        ack = self.control_handler.handle_command(action, params)
                        await websocket.send(json.dumps({"type": "ack", "ack": ack}))
                    except Exception as err:
                        await websocket.send(json.dumps({"type": "error", "message": str(err)}))
            except Exception:
                pass
            finally:
                self.ws_clients.discard(websocket)

        async def run_server():
            async with websockets.serve(handler, self.host, self.ws_port) as server:
                self._ws_server = server
                while self.running:
                    await asyncio.sleep(0.1)

        try:
            self._ws_loop.run_until_complete(run_server())
        except Exception as e:
            if self.running:
                log.warning("[GATEWAY] WebSocket server error: %s", e)
        finally:
            self._ws_loop.close()

    def _broadcast_ws(self, payload: str) -> None:
        """Push payload to all connected WebSocket clients."""
        if not self.ws_clients or self._ws_loop is None or not self._ws_loop.is_running():
            return

        async def _send_all():
            dead = []
            for ws in list(self.ws_clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.ws_clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._ws_loop)
        except Exception:
            pass

    # ── HTTP Server ──────────────────────────────────────────────────────────

    def _start_http(self) -> None:
        gateway = self

        class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Keep quiet

            def end_headers(self) -> None:
                # Add CORS headers for local development
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                super().end_headers()

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                # 1. Dashboard Root & Legacy Paths
                if path in ("/", "/index.html", "/dashboard", "/dashboard/index.html"):
                    html_file = _CLIENT_DIR / "index.html"
                    if not html_file.exists():
                        html_file = _DASHBOARD_DIR / "index.html"
                    if html_file.exists():
                        content = html_file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    self.send_error(404, "Index HTML not found")
                    return

                # 2. Static CSS / JS assets
                if path.startswith("/css/") or path.startswith("/js/"):
                    asset_path = _CLIENT_DIR / path.lstrip("/")
                    if asset_path.exists() and asset_path.is_file():
                        content_type, _ = mimetypes.guess_type(str(asset_path))
                        content = asset_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", content_type or "text/plain")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    self.send_error(404, f"Asset '{path}' not found")
                    return

                # 3. Backward-compatible /ground_truth endpoint
                if path == "/ground_truth":
                    frame = gateway.backend.get_ground_truth_frame()
                    body = json.dumps(frame).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # 4. Backward-compatible /status endpoint
                if path == "/status":
                    sim_z = float(gateway.backend.data.qpos[2])
                    quat = [float(gateway.backend.data.qpos[3]), float(gateway.backend.data.qpos[4]), float(gateway.backend.data.qpos[5]), float(gateway.backend.data.qpos[6])]
                    grav = compute_projected_gravity(np.array(quat))
                    data = {
                        "mode": gateway.backend.core.mode.name,
                        "gantry_active": gateway.backend.gantry_active,
                        "base_z": sim_z,
                        "projected_gravity_z": float(grav[2]),
                        "fault_latched": gateway.backend.core.fault_latched,
                        "error_flags": gateway.backend.core.error_flags,
                        "active_controller": gateway.backend.core.active_controller,
                        "battery_soc": gateway.backend.core.battery_soc,
                        "battery_voltage": gateway.backend.core.battery_voltage,
                        "battery_discharge_rate": gateway.backend.core.battery_discharge_rate,
                    }
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # 5. Modern /api/state snapshot
                if path == "/api/state":
                    body = (gateway.latest_json or "{}").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # 6. /api/camera/list
                if path == "/api/camera/list":
                    cams = gateway.camera_streamer.get_camera_metadata()
                    body = json.dumps(cams).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # 7. /api/camera/frame snapshot
                if path == "/api/camera/frame":
                    cam_name = query.get("camera", ["front_camera"])[0]
                    xray = query.get("xray", ["0"])[0] in ("1", "true")
                    frame = gateway.camera_streamer.get_latest_frame(cam_name, xray)
                    if frame and frame.get("jpeg"):
                        jpeg = frame["jpeg"]
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.send_header("X-Camera-Name", cam_name)
                        self.send_header("X-Sequence", str(frame.get("seq", 0)))
                        self.send_header("X-Sim-Time", f"{frame.get('sim_time', 0.0):.4f}")
                        self.send_header("X-Timestamp", str(frame.get("wall_time", 0.0)))
                        self.send_header("X-FPS", "25.0")
                        self.end_headers()
                        self.wfile.write(jpeg)
                        return
                    self.send_error(503, "Camera frame not available")
                    return

                # 8. /api/camera/stream MJPEG live stream
                if path == "/api/camera/stream":
                    cam_name = query.get("camera", ["front_camera"])[0]
                    xray = query.get("xray", ["0"])[0] in ("1", "true")

                    self.send_response(200)
                    boundary = "vasimov_frame_boundary"
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.end_headers()

                    last_seq = -1
                    try:
                        while gateway.running:
                            frame = gateway.camera_streamer.get_latest_frame(cam_name, xray)
                            if frame and frame.get("jpeg") and frame.get("seq") != last_seq:
                                last_seq = frame.get("seq", 0)
                                jpeg = frame["jpeg"]
                                sim_t = frame.get("sim_time", 0.0)

                                part_header = (
                                    f"--{boundary}\r\n"
                                    f"Content-Type: image/jpeg\r\n"
                                    f"Content-Length: {len(jpeg)}\r\n"
                                    f"X-Sequence: {last_seq}\r\n"
                                    f"X-Sim-Time: {sim_t:.4f}\r\n\r\n"
                                ).encode("ascii")

                                self.wfile.write(part_header)
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                            time.sleep(0.04)  # ~25 FPS
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return

                # 9. /api/recordings
                if path == "/api/recordings":
                    recs = gateway.replay_manager.list_recordings()
                    body = json.dumps(recs).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_error(404, f"Path '{path}' not found")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

                try:
                    req_data = json.loads(body)
                except Exception:
                    req_data = {}

                # 1. Backward-compatible /control endpoint
                if path == "/control":
                    cmd = req_data.get("command", "").lower()
                    if cmd == "gantry":
                        state = req_data.get("state", "off").lower()
                        gateway.backend.set_gantry(state == "on")
                        resp = {"status": "ok", "gantry": gateway.backend.gantry_active}
                    elif cmd == "push":
                        force = float(req_data.get("force", 40.0))
                        direction = str(req_data.get("dir", "x"))
                        duration = float(req_data.get("duration", 0.1))
                        gateway.backend.apply_push(force, direction, duration)
                        resp = {"status": "ok", "pushed": force}
                    elif cmd == "inject":
                        fault = str(req_data.get("fault", "")).lower()
                        if fault == "fall":
                            gateway.backend.core.inject_fall()
                        elif fault == "overtemp":
                            gateway.backend.core.inject_overtemp(float(req_data.get("temp", 85.0)))
                        elif fault == "can":
                            gateway.backend.core.inject_can_fault()
                        resp = {"status": "ok", "injected": fault}
                    elif cmd in ("set_battery", "battery"):
                        soc = req_data.get("soc", req_data.get("percentage"))
                        rate = req_data.get("rate", req_data.get("discharge_rate"))
                        voltage = req_data.get("voltage")
                        gateway.backend.core.set_battery(soc=soc, discharge_rate=rate, voltage=voltage)
                        resp = {
                            "status": "ok",
                            "battery_soc": gateway.backend.core.battery_soc,
                            "discharge_rate": gateway.backend.core.battery_discharge_rate,
                        }
                    elif cmd == "restart":
                        gateway.backend.core.virtual_restart()
                        gateway.backend.reset()
                        resp = {"status": "ok", "restarted": True}
                    else:
                        resp = {"status": "error", "message": f"Unknown command '{cmd}'"}

                    out = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return

                # 2. Modern Unified /api/control endpoint
                if path.startswith("/api/control"):
                    sub_action = path.replace("/api/control", "").lstrip("/").lower()
                    action = sub_action if sub_action else req_data.get("action", "")
                    params = req_data.get("params", req_data)

                    ack = gateway.control_handler.handle_command(action, params)
                    out = json.dumps(ack).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return

                # 3. Camera settings: /api/camera/orbit
                if path == "/api/camera/orbit":
                    azimuth = req_data.get("azimuth")
                    elevation = req_data.get("elevation")
                    distance = req_data.get("distance")
                    gateway.camera_streamer.set_orbit_camera(azimuth, elevation, distance)
                    out = json.dumps({"status": "ok"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return

                # 4. Replay controls: /api/replay
                if path == "/api/replay":
                    action = req_data.get("action", "load")
                    if action == "load":
                        name = req_data.get("name", "")
                        ok = gateway.replay_manager.load_recording(name)
                        resp = {"status": "ok" if ok else "error", "loaded": ok}
                    elif action == "play":
                        gateway.replay_manager.play()
                        resp = {"status": "ok", "playing": True}
                    elif action == "pause":
                        gateway.replay_manager.pause()
                        resp = {"status": "ok", "playing": False}
                    elif action == "seek":
                        idx = req_data.get("index", 0)
                        fr = gateway.replay_manager.seek(idx)
                        resp = {"status": "ok", "frame": fr}
                    elif action == "stop":
                        gateway.replay_manager.stop_replay()
                        resp = {"status": "ok", "stopped": True}
                    else:
                        resp = {"status": "error", "message": f"Unknown replay action '{action}'"}

                    out = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return

                self.send_error(404, f"POST endpoint '{path}' not found")

        try:
            self.http_server = http.server.HTTPServer((self.host, self.http_port), GatewayRequestHandler)
            self._http_thread = threading.Thread(
                target=self.http_server.serve_forever,
                daemon=True,
                name="vasimov-http-gateway",
            )
            self._http_thread.start()
        except Exception as e:
            log.error("[GATEWAY] Failed to bind HTTP port %d: %s", self.http_port, e)
            self.http_server = None
