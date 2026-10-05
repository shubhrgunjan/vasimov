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
import numpy as np

from edge.core import EdgeCore
from edge.sim import SimBackend, compute_projected_gravity
from web.server.canonical_frame import CanonicalFrameBuilder
from web.server.camera_streamer import CameraStreamer
from web.server.control_handler import ControlHandler
from web.server.replay_manager import ReplayManager
from web.server.webrtc_gateway import WebRTCGateway
from edge.core import EdgeMode
from web.server.binary_protocol import (
    pack_state_packet, unpack_command_packet,
    STATE_PACKET_SIZE, COMMAND_PACKET_SIZE, STATE_MAGIC, COMMAND_MAGIC,
    CommandType, StateFlags, RobotMode, PolicyId, RecoveryState
)

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
        self.control_handler.camera_streamer = self.camera_streamer
        self.webrtc_gateway = WebRTCGateway(backend, core, self.control_handler)

        # State cache
        self.latest_frame: Optional[Dict[str, Any]] = None
        self.latest_json: str = "{}"
        self.latest_binary_packet: Optional[bytes] = None
        self.broadcast_sequence: int = 0
        self.binary_sequence: int = 0
        self.binary_ws_clients: Set[Any] = set()

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

        # 1. Start Camera Streamer worker & hook WebRTC frame ingestion
        self.camera_streamer.add_frame_listener(self.webrtc_gateway.on_camera_frame)
        self.camera_streamer.start()

        # 1b. Start WebRTC & LiveKit Gateway
        self.webrtc_gateway.start()

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
            "   WebRTC/LK: ws://%s:7880\n"
            "   Cameras  : %s",
            self.host,
            self.http_port,
            self.host,
            self.ws_port,
            self.host,
            self.camera_streamer.all_cameras,
        )

    def stop(self) -> None:
        """Stop all gateway servers cleanly."""
        self.running = False
        self._shutdown_event.set()

        # Stop WebRTC Gateway
        self.webrtc_gateway.stop()

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

    # ── Binary State Protocol Methods ────────────────────────────────────────
    def build_binary_state_packet(self) -> bytes:
        """Pack the latest authoritative MuJoCo state into fixed 288-byte VAS1 packet."""
        self.binary_sequence += 1
        backend = self.backend
        core = self.core
        d = backend.data

        with backend._lock:
            sim_time = float(d.time)
            qpos = d.qpos
            qvel = d.qvel

            robot_mode = int(core.mode.value) if hasattr(core.mode, "value") else 0
            policy_id = 1 if core.mode == EdgeMode.MOVE else 0

            recovery_state = 0
            if core.fault_latched:
                recovery_state = 1
            elif core.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
                recovery_state = 2

            safety_state = int(core.error_flags) if hasattr(core, "error_flags") else 0

            flags = 0
            if core.fault_latched:
                flags |= StateFlags.FAULT_LATCHED
            if backend.gantry_active:
                flags |= StateFlags.GANTRY_ACTIVE

            raw_contacts = backend.get_contacts()
            if raw_contacts.get("left_foot", {}).get("contact", False):
                flags |= StateFlags.CONTACT_LEFT
            if raw_contacts.get("right_foot", {}).get("contact", False):
                flags |= StateFlags.CONTACT_RIGHT

            base_pos = [float(qpos[0]), float(qpos[1]), float(qpos[2])]
            # Convert MuJoCo [w, x, y, z] to standard [x, y, z, w]
            base_quat = [float(qpos[4]), float(qpos[5]), float(qpos[6]), float(qpos[3])]
            lin_vel = [float(qvel[0]), float(qvel[1]), float(qvel[2])]
            ang_vel = [float(qvel[3]), float(qvel[4]), float(qvel[5])]

            # All 25 joint positions and velocities in SIM_JOINTS order
            joint_positions = [float(qpos[adr]) for adr in backend.actuator_qposadr]
            joint_velocities = [float(qvel[adr]) for adr in backend.actuator_dofadr]

            cmd_vx = float(getattr(core, "current_vx", 0.0))
            cmd_vy = float(getattr(core, "current_vy", 0.0))
            cmd_wz = float(getattr(core, "current_vyaw", 0.0))

        return pack_state_packet(
            sequence=self.binary_sequence,
            timestamp=sim_time,
            robot_mode=robot_mode,
            policy_id=policy_id,
            recovery_state=recovery_state,
            safety_state=safety_state,
            flags=flags,
            base_position=base_pos,
            base_orientation=base_quat,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            command_linear_x=cmd_vx,
            command_linear_y=cmd_vy,
            command_yaw=cmd_wz,
        )

    def _broadcast_binary_ws(self, payload: bytes) -> None:
        """Push binary state packet to all registered binary WebSocket clients."""
        if not self.binary_ws_clients or self._ws_loop is None or not self._ws_loop.is_running():
            return

        async def _send_all():
            dead = []
            for ws in list(self.binary_ws_clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.binary_ws_clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._ws_loop)
        except Exception:
            pass

    # ── Telemetry Broadcast Loop ─────────────────────────────────────────────

    def _telemetry_broadcast_loop(self) -> None:
        """Periodic state broadcaster: binary state @ 50 Hz, JSON @ ~25 Hz."""
        tick = 0
        while self.running and not self._shutdown_event.is_set():
            t0 = time.perf_counter()
            tick += 1

            # 1. Binary state broadcast at 50 Hz (every tick)
            if self.binary_ws_clients:
                try:
                    bin_packet = self.build_binary_state_packet()
                    self.latest_binary_packet = bin_packet
                    self._broadcast_binary_ws(bin_packet)
                except Exception as e:
                    log.debug("[GATEWAY] Binary state packing error: %s", e)



            # Check for PC environment preset change at 5 Hz
            if tick % 10 == 0:
                try:
                    cur_preset = getattr(self.backend.env_manager, "current_preset", None)
                    if cur_preset and getattr(self, "_last_broadcast_preset", None) != cur_preset:
                        self._last_broadcast_preset = cur_preset
                        env_state = self.backend.get_environment_state()
                        self._broadcast_text_to_all(json.dumps({"type": "environment", "data": env_state}))
                except Exception:
                    pass

            # 2. JSON telemetry broadcast at ~25 Hz (every 2nd tick)
            if self.ws_clients and (tick % 2 == 0):
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

            # Cadence paced to 50 Hz (20 ms interval)
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.002, (1.0 / 50.0) - elapsed)
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
                    # A. Binary VAC1 Command packet handling (Asimov Controller 3D)
                    if isinstance(message, (bytes, bytearray)):
                        raw_bytes = bytes(message)
                        if len(raw_bytes) == COMMAND_PACKET_SIZE and raw_bytes[:4] == COMMAND_MAGIC:
                            cmd = unpack_command_packet(raw_bytes)
                            if websocket not in self.binary_ws_clients:
                                self.binary_ws_clients.add(websocket)
                                self.ws_clients.discard(websocket)

                            ctype = cmd["command_type"]
                            action = ""
                            params = {}
                            if ctype == CommandType.WALK:
                                action = "walk"
                                params = {"vx": cmd["linear_x"], "vy": cmd["linear_y"], "wz": cmd["yaw_rate"]}
                            elif ctype == CommandType.STAND:
                                action = "stand"
                            elif ctype == CommandType.CROUCH:
                                action = "crouch"
                            elif ctype == CommandType.RECOVERY:
                                action = "getup"
                            elif ctype == CommandType.RESET:
                                action = "reset"
                            elif ctype in (CommandType.ESTOP, CommandType.DAMP):
                                action = "damp"

                            if action:
                                self.control_handler.handle_command(action, params)
                            continue

                        # WebRTC Gateway BinaryProtocol fallback
                        from web.server.webrtc_gateway import BinaryProtocol
                        parsed = BinaryProtocol.unpack_control(raw_bytes)
                        if parsed:
                            ack = self.control_handler.handle_command(parsed["action"], parsed["params"])
                            await websocket.send(json.dumps({"type": "ack", "ack": ack}))
                            continue

                    # B. Text JSON Command / Subscription handling
                    try:
                        data = json.loads(message)
                        action = data.get("action", "")
                        params = data.get("params", data.get("command_data", {}))

                        # Client requested binary streaming
                        if (action == "subscribe" and params.get("format") == "binary") or data.get("format") == "binary":
                            self.binary_ws_clients.add(websocket)
                            self.ws_clients.discard(websocket)
                            await websocket.send(json.dumps({"type": "subscribed", "format": "binary"}))
                            try:
                                env_state = self.backend.get_environment_state()
                                await websocket.send(json.dumps({"type": "environment", "data": env_state}))
                            except Exception:
                                pass
                            continue

                        if action == "get_environment":
                            try:
                                env_state = self.backend.get_environment_state()
                                await websocket.send(json.dumps({"type": "environment", "data": env_state}))
                            except Exception as e:
                                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                            continue

                        if action == "environment":
                            ack = self.control_handler.handle_command(action, params)
                            try:
                                env_state = self.backend.get_environment_state()
                                env_msg = json.dumps({"type": "environment", "data": env_state, "ack": ack})
                                self._broadcast_text_to_all(env_msg)
                            except Exception:
                                await websocket.send(json.dumps({"type": "ack", "ack": ack}))
                            continue

                        ack = self.control_handler.handle_command(action, params)
                        await websocket.send(json.dumps({"type": "ack", "ack": ack}))
                    except Exception as err:
                        await websocket.send(json.dumps({"type": "error", "message": str(err)}))
            except Exception:
                pass
            finally:
                self.ws_clients.discard(websocket)
                self.binary_ws_clients.discard(websocket)

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

    def _broadcast_text_to_all(self, payload: str) -> None:
        """Push text payload to all connected clients (both JSON and binary subscribers)."""
        if self._ws_loop is None or not self._ws_loop.is_running():
            return
        all_clients = list(self.ws_clients) + list(self.binary_ws_clients)
        if not all_clients:
            return

        async def _send_all():
            for ws in all_clients:
                try:
                    await ws.send(payload)
                except Exception:
                    pass

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._ws_loop)
        except Exception:
            pass

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

                # 1b. Dedicated WebRTC Native Viewport HTML
                if path in ("/webrtc_player.html", "/webrtc_player", "/player"):
                    player_file = _CLIENT_DIR / "webrtc_player.html"
                    if player_file.exists():
                        content = player_file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    self.send_error(404, "WebRTC Player HTML not found")
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

                # 3b. Local session discovery endpoint
                if path in ("/session", "/api/session"):
                    req_host = self.headers.get("Host", "127.0.0.1").split(":")[0]
                    host = req_host if req_host not in ("0.0.0.0", "") else "127.0.0.1"
                    data = {
                        "running": True,
                        "session_id": "asimov-local-sim",
                        "transport": "websocket",
                        "ws_url": f"ws://{host}:{gateway.ws_port}",
                        "http_url": f"http://{host}:{gateway.http_port}",
                        "webrtc_url": f"ws://{host}:7880",
                        "room": "asimov-teleop",
                        "robot_id": "asimov-1",
                        "format": "binary",
                        "protocol": "VAS1",
                    }
                    body = json.dumps(data).encode("utf-8")
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

                # 5b. WebRTC Token & LiveKit room config
                if path == "/api/webrtc/token":
                    identity = query.get("identity", ["android-phone"])[0]
                    tok = gateway.webrtc_gateway.generate_token(identity=identity)
                    req_host = self.headers.get("Host", "127.0.0.1").split(":")[0]
                    lk_host = req_host if req_host not in ("0.0.0.0", "") else "127.0.0.1"
                    resp = {
                        "status": "ok",
                        "url": f"ws://{lk_host}:7880",
                        "token": tok,
                        "room": gateway.webrtc_gateway.room_name,
                        "active_camera": gateway.webrtc_gateway.active_camera,
                    }
                    body = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # 5c. WebRTC Live Diagnostics & Latency Stats
                if path == "/api/webrtc/stats":
                    stats = gateway.webrtc_gateway.get_stats()
                    body = json.dumps(stats).encode("utf-8")
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

                # 6b. Continuous 60 FPS multipart MJPEG stream (high-performance fallback over standard HTTP)
                if path in ("/api/camera/stream", "/api/camera/mjpeg"):
                    cam_name = query.get("camera", ["chase_camera"])[0]
                    xray = query.get("xray", ["0"])[0] in ("1", "true")
                    if hasattr(gateway, "camera_streamer") and gateway.camera_streamer is not None:
                        gateway.camera_streamer.primary_camera = cam_name
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    last_seq = -1
                    try:
                        while gateway.running:
                            frame = gateway.camera_streamer.get_latest_frame(cam_name, xray)
                            if frame and frame.get("jpeg") and frame.get("seq", 0) != last_seq:
                                last_seq = frame.get("seq", 0)
                                jpeg = frame["jpeg"]
                                header = (
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                                )
                                self.wfile.write(header)
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.016)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return

                # 7. /api/camera/frame snapshot
                if path == "/api/camera/frame":
                    cam_name = query.get("camera", ["chase_camera"])[0]
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
                    if "action" in req_data:
                        action = req_data.get("action", "")
                        params = req_data.get("params", {})
                        resp = gateway.control_handler.handle_command(action, params)
                        out = json.dumps(resp).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(out)))
                        self.end_headers()
                        self.wfile.write(out)
                        return

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

                # 2b. WebRTC Camera Track Selector
                if path == "/api/webrtc/camera":
                    cam_name = req_data.get("camera", "chase_camera")
                    gateway.webrtc_gateway.switch_camera(cam_name)
                    if hasattr(gateway, "camera_streamer") and gateway.camera_streamer is not None:
                        gateway.camera_streamer.primary_camera = cam_name
                    resp = {"status": "ok", "camera": gateway.webrtc_gateway.active_camera}
                    out = json.dumps(resp).encode("utf-8")
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
            self.http_server = http.server.ThreadingHTTPServer((self.host, self.http_port), GatewayRequestHandler)
            self._http_thread = threading.Thread(
                target=self.http_server.serve_forever,
                daemon=True,
                name="vasimov-http-gateway",
            )
            self._http_thread.start()
        except Exception as e:
            log.error("[GATEWAY] Failed to bind HTTP port %d: %s", self.http_port, e)
            self.http_server = None
