"""
vasimov/web/server/webrtc_gateway.py
Hardware-Accelerated WebRTC Streaming & Realtime Teleoperation Gateway.

Features:
- LiveKit WebRTC video streaming with H.264 hardware encoding where available.
- Realtime binary data channels:
    * Topic 'commands': Compact 32-byte binary control packets with newest-sequence priority.
    * Topic 'telemetry': Compact 85-byte binary telemetry frames at ~30 Hz.
- Decoupled media, control, and telemetry pipelines.
- Dynamic camera switching across all MuJoCo model cameras.
- Comprehensive latency and performance instrumentation (T0-T9, RTT, FPS).
- Automatic local livekit-server process lifecycle management.
"""

from __future__ import annotations
import asyncio
import logging
import math
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from livekit import api, rtc

log = logging.getLogger("vasimov.web.webrtc")

# ── Binary Protocol Specifications ──────────────────────────────────────────
# Control Packet Schema (Phone -> Robot):
# [Magic: 4s ('ACTL')] [Seq: I (uint32)] [TimestampUs: Q (uint64)]
# [Action: B (uint8)] [vx: f] [vy: f] [wz: f] [Buttons: H (uint16)] [CamIdx: B (uint8)]
# Total: 4 + 4 + 8 + 1 + 4 + 4 + 4 + 2 + 1 = 32 bytes
CTRL_STRUCT = struct.Struct("<4sIQBfffHB")
CTRL_MAGIC = b"ACTL"

ACTION_MAP = {
    0: "idle",
    1: "stand",
    2: "damp",
    3: "stop",
    4: "walk",
    5: "fall",
    6: "getup",
    7: "reset",
    8: "switch_cam",
}
ACTION_REVERSE_MAP = {v: k for k, v in ACTION_MAP.items()}

# Telemetry Packet Schema (Robot -> Phone):
# [Magic: 4s ('ATEL')] [Seq: I (uint32)] [SimTime: f] [WallTimeMs: d]
# [Mode: B] [Submode: B]
# [Pos: 3f (x,y,z)] [Quat: 4f (w,x,y,z)]
# [LinVel: 3f (vx,vy,vz)] [AngVel: 3f (wx,wy,wz)]
# [Contacts: B] [Faults: B] [BatterySoc: B] [PingEchoUs: Q (uint64)]
# Total: 4 + 4 + 4 + 8 + 1 + 1 + 12 + 16 + 12 + 12 + 1 + 1 + 1 + 8 = 85 bytes
TELEM_STRUCT = struct.Struct("<4sIfdBB3f4f3f3fBBBQ")
TELEM_MAGIC = b"ATEL"

MODE_CODES = {
    "UNKNOWN": 0,
    "DAMP": 1,
    "STAND": 2,
    "MOVE": 3,
    "FAULT": 4,
    "ESTOP": 5,
}

SUBMODE_CODES = {
    "NONE": 0,
    "WALK_POLICY": 1,
    "TRAJECTORY": 2,
    "MANUAL": 3,
}


class BinaryProtocol:
    """High-performance binary packing and unpacking for low-latency teleoperation."""

    @staticmethod
    def unpack_control(data: bytes) -> Optional[Dict[str, Any]]:
        if len(data) != CTRL_STRUCT.size:
            return None
        magic, seq, ts_us, action_code, vx, vy, wz, buttons, cam_idx = CTRL_STRUCT.unpack(data)
        if magic != CTRL_MAGIC:
            return None

        action = ACTION_MAP.get(action_code, "idle")
        return {
            "seq": seq,
            "timestamp_us": ts_us,
            "action": action,
            "action_code": action_code,
            "params": {"vx": vx, "vy": vy, "wz": wz, "camera_index": cam_idx},
            "buttons": buttons,
            "camera_index": cam_idx,
        }

    @staticmethod
    def pack_control(
        seq: int,
        action: str,
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        buttons: int = 0,
        cam_idx: int = 0,
        ts_us: Optional[int] = None,
    ) -> bytes:
        if ts_us is None:
            ts_us = int(time.time() * 1_000_000)
        action_code = ACTION_REVERSE_MAP.get(action.lower(), 0)
        return CTRL_STRUCT.pack(CTRL_MAGIC, seq, ts_us, action_code, vx, vy, wz, buttons, cam_idx)

    @staticmethod
    def pack_telemetry(
        seq: int,
        sim_time: float,
        wall_time_ms: float,
        mode_str: str,
        submode_str: str,
        pos: Sequence[float],
        quat: Sequence[float],
        lin_vel: Sequence[float],
        ang_vel: Sequence[float],
        left_foot: bool,
        right_foot: bool,
        fault_latched: bool,
        fall_detected: bool,
        battery_soc: float,
        ping_echo_us: int = 0,
    ) -> bytes:
        mode_code = MODE_CODES.get(mode_str.upper(), 0)
        submode_code = SUBMODE_CODES.get(submode_str.upper(), 0)
        contacts_mask = (1 if left_foot else 0) | (2 if right_foot else 0)
        fault_mask = (1 if fault_latched else 0) | (2 if fall_detected else 0)
        batt_byte = max(0, min(100, int(battery_soc)))

        p = (pos[0], pos[1], pos[2]) if len(pos) >= 3 else (0.0, 0.0, 0.0)
        q = (quat[0], quat[1], quat[2], quat[3]) if len(quat) >= 4 else (1.0, 0.0, 0.0, 0.0)
        v = (lin_vel[0], lin_vel[1], lin_vel[2]) if len(lin_vel) >= 3 else (0.0, 0.0, 0.0)
        w = (ang_vel[0], ang_vel[1], ang_vel[2]) if len(ang_vel) >= 3 else (0.0, 0.0, 0.0)

        return TELEM_STRUCT.pack(
            TELEM_MAGIC,
            seq,
            sim_time,
            wall_time_ms,
            mode_code,
            submode_code,
            p[0], p[1], p[2],
            q[0], q[1], q[2], q[3],
            v[0], v[1], v[2],
            w[0], w[1], w[2],
            contacts_mask,
            fault_mask,
            batt_byte,
            ping_echo_us,
        )


class LiveKitProcessManager:
    """Manages the local livekit-server daemon."""

    def __init__(self, port: int = 7880):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self._owns_process = False

    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self) -> bool:
        if self.is_port_in_use():
            log.info("[WEBRTC] livekit-server is already running on port %d", self.port)
            return True

        livekit_bin = shutil.which("livekit-server") or "/opt/homebrew/bin/livekit-server"
        if not os.path.exists(livekit_bin):
            log.warning("[WEBRTC] livekit-server executable not found at %s", livekit_bin)
            return False

        try:
            cmd = [livekit_bin, "--dev", "--bind", "0.0.0.0"]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._owns_process = True
            log.info("[WEBRTC] Launched managed livekit-server (PID %d)", self.process.pid)

            # Wait up to 3 seconds for port to become active
            for _ in range(30):
                time.sleep(0.1)
                if self.is_port_in_use():
                    log.info("[WEBRTC] livekit-server is ready and accepting WebRTC connections")
                    return True

            log.error("[WEBRTC] livekit-server failed to bind port %d within timeout", self.port)
            return False
        except Exception as e:
            log.error("[WEBRTC] Failed to start livekit-server: %s", e)
            return False

    def stop(self) -> None:
        if self._owns_process and self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self._owns_process = False
            log.info("[WEBRTC] Stopped managed livekit-server")


class WebRTCGateway:
    """
    Central Realtime WebRTC Media, Telemetry, and Control Gateway.
    Decoupled from MuJoCo physics loop (200 Hz).
    """

    def __init__(
        self,
        backend: Any,
        core: Any,
        control_handler: Any,
        api_key: str = "devkey",
        api_secret: str = "secret",
        livekit_url: str = "ws://127.0.0.1:7880",
        room_name: str = "vasimov-teleop",
        default_width: int = 960,
        default_height: int = 540,
        target_fps: int = 60,
    ):
        self.backend = backend
        self.core = core
        self.control_handler = control_handler
        self.api_key = api_key
        self.api_secret = api_secret
        self.livekit_url = livekit_url
        self.room_name = room_name
        self.width = default_width
        self.height = default_height
        self.target_fps = target_fps

        self.running = False
        self._shutdown_event = threading.Event()
        self._proc_mgr = LiveKitProcessManager()

        # WebRTC resources
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.room: Optional[rtc.Room] = None
        self.video_source: Optional[rtc.VideoSource] = None
        self.video_track: Optional[rtc.LocalVideoTrack] = None
        self.track_publication: Optional[rtc.LocalTrackPublication] = None

        # Camera selection & state
        self.active_camera: str = "chase_camera"
        self.active_xray: bool = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._rgb_lock = threading.Lock()

        # Realtime control sequencing (drops stale commands)
        self.latest_control_seq: int = 0
        self.latest_command_ts_us: int = 0
        self.commands_received: int = 0
        self.commands_applied: int = 0
        self.commands_stale_dropped: int = 0

        # Telemetry broadcast sequencing (~30 Hz)
        self.telemetry_seq: int = 0
        self._telemetry_thread: Optional[threading.Thread] = None

        # Performance instrumentation (T0-T9)
        self.frames_rendered: int = 0
        self.frames_published: int = 0
        self.last_frame_ts: float = 0.0
        self.actual_publish_fps: float = 0.0
        self.last_rtt_ms: float = 0.0
        self.recent_rtt_samples: List[float] = []

    def start(self) -> None:
        """Initialize livekit-server, launch publisher event loop and telemetry loop."""
        if self.running:
            return
        self.running = True
        self._shutdown_event.clear()

        # 1. Start livekit-server if not running
        self._proc_mgr.start()

        # 2. Launch background asyncio WebRTC thread
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="vasimov-webrtc-publisher",
        )
        self._thread.start()

        # 3. Launch ~30 Hz binary telemetry thread
        self._telemetry_thread = threading.Thread(
            target=self._run_telemetry_loop,
            daemon=True,
            name="vasimov-webrtc-telemetry",
        )
        self._telemetry_thread.start()
        log.info("[WEBRTC] Gateway started (target %dx%d @ %d FPS, H.264)", self.width, self.height, self.target_fps)

    def stop(self) -> None:
        """Clean shutdown of WebRTC room, threads, and managed process."""
        self.running = False
        self._shutdown_event.set()

        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._disconnect_room(), self._loop)
                future.result(timeout=1.0)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        self._proc_mgr.stop()
        log.info("[WEBRTC] Gateway stopped.")

    # ── Token Generation API for Clients ──────────────────────────────────────
    def generate_token(self, identity: str = "android-teleop", is_admin: bool = False) -> str:
        """Mint a signed LiveKit JWT token for mobile or web clients."""
        grants = api.VideoGrants(
            room_join=True,
            room=self.room_name,
            can_publish=is_admin,
            can_subscribe=True,
            can_publish_data=True,
        )
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_name(f"Operator ({identity})")
            .with_grants(grants)
            .to_jwt()
        )
        return token

    # ── Camera Feed Ingestion (from MuJoCo Offscreen Worker) ──────────────────
    def on_camera_frame(self, camera_name: str, rgb: np.ndarray, sim_time: float) -> None:
        """
        Receives uncompressed raw RGB numpy array from MuJoCo renderer.
        Zero PIL or JPEG overhead. Directly captures into WebRTC VideoSource.
        """
        if not self.running or camera_name != self.active_camera:
            return

        self.frames_rendered += 1
        t_capture = time.perf_counter()

        # Send frame into WebRTC video source
        if self.video_source is not None and self._loop is not None and self._loop.is_running():
            h, w = rgb.shape[:2]
            if w != self.width or h != self.height:
                self.width = w
                self.height = h

            try:
                vf = rtc.VideoFrame(
                    self.width,
                    self.height,
                    rtc.VideoBufferType.RGB24,
                    rgb.tobytes(),
                )
                self.video_source.capture_frame(vf)
                self.frames_published += 1

                # Calculate live publish FPS
                dt = t_capture - self.last_frame_ts
                self.last_frame_ts = t_capture
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.actual_publish_fps = 0.9 * self.actual_publish_fps + 0.1 * inst_fps
            except Exception as e:
                log.debug("[WEBRTC] Frame capture error: %s", e)

    def switch_camera(self, camera_name: str) -> None:
        """Switch the active streamed camera track dynamically without interrupting physics."""
        self.active_camera = camera_name
        log.info("[WEBRTC] Switched active teleoperation camera to '%s'", camera_name)

    # ── Realtime Telemetry Broadcast Loop (~30 Hz) ────────────────────────────
    def _run_telemetry_loop(self) -> None:
        """Periodically packs and broadcasts compact binary telemetry."""
        target_period = 1.0 / 30.0  # 30 Hz UI telemetry rate

        while self.running and not self._shutdown_event.is_set():
            t0 = time.perf_counter()
            self.telemetry_seq += 1

            try:
                sim_t = float(self.backend.data.time)
                wall_ms = time.time() * 1000.0
                mode_str = self.core.mode.name
                submode_str = getattr(self.core, "move_submode", None)
                submode_name = submode_str.name if submode_str else "NONE"

                pos = [float(self.backend.data.qpos[0]), float(self.backend.data.qpos[1]), float(self.backend.data.qpos[2])]
                quat = [float(self.backend.data.qpos[3]), float(self.backend.data.qpos[4]), float(self.backend.data.qpos[5]), float(self.backend.data.qpos[6])]
                vel = [float(self.backend.data.qvel[0]), float(self.backend.data.qvel[1]), float(self.backend.data.qvel[2])]
                ang_vel = [float(self.backend.data.qvel[3]), float(self.backend.data.qvel[4]), float(self.backend.data.qvel[5])]

                left_foot = bool(self.core.left_foot_contact) if hasattr(self.core, "left_foot_contact") else True
                right_foot = bool(self.core.right_foot_contact) if hasattr(self.core, "right_foot_contact") else True
                fault = bool(self.core.fault_latched)
                fall = bool(getattr(self.core, "fall_detected", False))
                soc = float(getattr(self.core, "battery_soc", 100.0))

                # Echo back latest client timestamp for round-trip latency calculation
                echo_us = self.latest_command_ts_us

                payload = BinaryProtocol.pack_telemetry(
                    seq=self.telemetry_seq,
                    sim_time=sim_t,
                    wall_time_ms=wall_ms,
                    mode_str=mode_str,
                    submode_str=submode_name,
                    pos=pos,
                    quat=quat,
                    lin_vel=vel,
                    ang_vel=ang_vel,
                    left_foot=left_foot,
                    right_foot=right_foot,
                    fault_latched=fault,
                    fall_detected=fall,
                    battery_soc=soc,
                    ping_echo_us=echo_us,
                )

                # Broadcast over lossy/unreliable data channel (zero queuing delay)
                if self.room and self.room.local_participant and self._loop:
                    asyncio.run_coroutine_threadsafe(
                        self.room.local_participant.publish_data(payload, reliable=False, topic="telemetry"),
                        self._loop,
                    )
            except Exception as e:
                log.debug("[WEBRTC] Telemetry pack/broadcast error: %s", e)

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.005, target_period - elapsed)
            time.sleep(sleep_t)

    # ── Incoming Command Handling ─────────────────────────────────────────────
    def handle_incoming_data(self, data: bytes, topic: str, participant: Any) -> None:
        """
        Dispatches incoming data from WebRTC DataChannels.
        Applies strict newest-sequence priority: stale continuous commands are dropped.
        """
        if topic != "commands":
            return

        self.commands_received += 1
        now_us = int(time.time() * 1_000_000)

        # 1. Binary control unpacking
        parsed = BinaryProtocol.unpack_control(data)
        if parsed is None:
            # Fallback: attempt JSON parsing
            try:
                import json
                jdata = json.loads(data.decode("utf-8"))
                parsed = {
                    "seq": jdata.get("seq", self.commands_received),
                    "timestamp_us": jdata.get("timestamp_us", now_us),
                    "action": jdata.get("action", "idle"),
                    "params": jdata.get("params", {}),
                }
            except Exception:
                return

        seq = parsed["seq"]
        ts_us = parsed["timestamp_us"]
        action = parsed["action"]
        params = parsed["params"]

        # Track RTT if client timestamp is valid
        if ts_us > 0 and now_us > ts_us:
            rtt_ms = (now_us - ts_us) / 1000.0
            if rtt_ms < 500.0:
                self.last_rtt_ms = rtt_ms
                self.recent_rtt_samples.append(rtt_ms)
                if len(self.recent_rtt_samples) > 50:
                    self.recent_rtt_samples.pop(0)

        self.latest_command_ts_us = ts_us

        # 2. Sequence check for continuous controls (walk / velocity)
        if action == "walk":
            if seq <= self.latest_control_seq and (self.latest_control_seq - seq) < 10000:
                self.commands_stale_dropped += 1
                return  # Drop stale velocity command

            self.latest_control_seq = seq

        # 3. Handle camera switch command
        if action == "switch_cam":
            cam_idx = params.get("camera_index", 0)
            if hasattr(self.backend, "camera_streamer"):
                cams = self.backend.camera_streamer.all_cameras
                if 0 <= cam_idx < len(cams):
                    self.switch_camera(cams[cam_idx])
            return

        # 4. Dispatch to EdgeCore / ControlHandler
        try:
            ack = self.control_handler.handle_command(action, params)
            self.commands_applied += 1

            # Send lifecycle acknowledgement back to client
            if self.room and self.room.local_participant and self._loop:
                import json
                ack_bytes = json.dumps({"type": "ack", "ack": ack}).encode("utf-8")
                asyncio.run_coroutine_threadsafe(
                    self.room.local_participant.publish_data(ack_bytes, reliable=True, topic="ack"),
                    self._loop,
                )
        except Exception as e:
            log.error("[WEBRTC] Error applying command '%s': %s", action, e)

    # ── Asyncio WebRTC Connection & Event Loop ────────────────────────────────
    def _run_event_loop(self) -> None:
        """Runs the background asyncio event loop for LiveKit WebRTC."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def main_task():
            await asyncio.sleep(0.5)

            pub_token = self.generate_token(identity="sim-robot", is_admin=True)
            self.room = rtc.Room()

            @self.room.on("data_received")
            def on_data_received(data_packet: rtc.DataPacket):
                self.handle_incoming_data(
                    data=data_packet.data,
                    topic=data_packet.topic,
                    participant=data_packet.participant,
                )

            try:
                await self.room.connect(self.livekit_url, pub_token)
                log.info("[WEBRTC] Connected to LiveKit room '%s' as publisher", self.room_name)
            except Exception as e:
                log.error("[WEBRTC] Failed to connect to LiveKit: %s", e)
                return

            self.video_source = rtc.VideoSource(self.width, self.height)
            self.video_track = rtc.LocalVideoTrack.create_video_track("camera_teleop", self.video_source)

            publish_options = rtc.TrackPublishOptions(
                video_codec=rtc.VideoCodec.H264,
                source=rtc.TrackSource.SOURCE_CAMERA,
            )

            try:
                self.track_publication = await self.room.local_participant.publish_track(
                    self.video_track,
                    publish_options,
                )
                log.info("[WEBRTC] Published H.264 video track (SID: %s)", self.track_publication.sid)
            except Exception as e:
                log.error("[WEBRTC] Failed to publish video track: %s", e)
                return

            while self.running and not self._shutdown_event.is_set():
                await asyncio.sleep(0.2)

        try:
            self._loop.run_until_complete(main_task())
        except Exception as e:
            if self.running:
                log.warning("[WEBRTC] Event loop terminated: %s", e)
        finally:
            self._loop.close()

    async def _disconnect_room(self) -> None:
        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception:
                pass
            self.room = None

    # ── Performance & Observability Statistics ────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """Provides real-time engineering diagnostics for debug screens."""
        avg_rtt = sum(self.recent_rtt_samples) / max(1, len(self.recent_rtt_samples))
        return {
            "status": "CONNECTED" if (self.room and self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED) else "OFFLINE",
            "codec": "H.264 (Hardware where available)",
            "resolution": f"{self.width}x{self.height}",
            "target_fps": self.target_fps,
            "actual_fps": round(self.actual_publish_fps, 1),
            "active_camera": self.active_camera,
            "frames_published": self.frames_published,
            "commands_received": self.commands_received,
            "commands_applied": self.commands_applied,
            "commands_stale_dropped": self.commands_stale_dropped,
            "rtt_ms": round(avg_rtt, 1),
            "sim_hz": 200.0,
            "telemetry_hz": 30.0,
        }
