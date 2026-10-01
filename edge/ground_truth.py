#!/usr/bin/env python3
"""
vasimov/edge/ground_truth.py
Sim-only Ground-Truth Streaming Server.

Carries high-rate (~50 Hz) simulation-only ground-truth state:
- Simulation time, real-time factor, state machine mode, gantry state, fault state
- Pelvis 6-DoF pose, tilt angle, base linear and angular velocity
- Per-joint telemetry for all 25 joints (firmware order + sim name):
  * Position, velocity, target setpoint, position error
  * Applied torque, effort limit, % effort limit
  * Joint range, % of range
  * Active Kp, Kd gains
- Ankle kinematics & dynamics:
  * Orthogonal pitch & roll next to pushrod motor A & B equivalents
- Foot contact flags and 3D contact forces from derived touch/force sensors
- Virtual battery state

NOTE: This stream has NO hardware equivalent. It is strictly a virtual
simulation observability and debugging feed for Virtual Asimov 1.
"""

from __future__ import annotations
import asyncio
import json
import logging
import math
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set
import numpy as np

import websockets

log = logging.getLogger("vasimov.ground_truth")

DEFAULT_WS_PORT: int = 8854
DEFAULT_UDP_PORT: int = 8855


class GroundTruthServer:
    """
    WebSocket and optional UDP broadcast server for sim-only ground truth.
    """

    def __init__(self, ws_port: int = DEFAULT_WS_PORT, udp_port: Optional[int] = None):
        self.ws_port = ws_port
        self.udp_port = udp_port
        self.clients: Set[Any] = set()
        self.latest_frame: Optional[Dict[str, Any]] = None
        self.latest_json: str = "{}"
        self.running: bool = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws_server = None
        self._udp_sock: Optional[socket.socket] = None

        if self.udp_port is not None:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self) -> None:
        """Start WebSocket streaming server in a background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="vasimov-ground-truth-ws",
        )
        self._thread.start()
        log.info("[GROUND TRUTH] WebSocket server starting on ws://127.0.0.1:%d", self.ws_port)

    def _run_event_loop(self) -> None:
        """Internal asyncio event loop thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(websocket):
            self.clients.add(websocket)
            # Immediately send latest state if available
            if self.latest_json and self.latest_json != "{}":
                try:
                    await websocket.send(self.latest_json)
                except Exception:
                    pass
            try:
                await websocket.wait_closed()
            finally:
                self.clients.discard(websocket)

        async def run_server():
            async with websockets.serve(handler, "127.0.0.1", self.ws_port) as server:
                self._ws_server = server
                log.info("[GROUND TRUTH] Listening on ws://127.0.0.1:%d", self.ws_port)
                while self.running:
                    await asyncio.sleep(0.1)

        try:
            self._loop.run_until_complete(run_server())
        except Exception as e:
            if self.running:
                log.warning("[GROUND TRUTH] Server error: %s", e)
        finally:
            self._loop.close()

    def broadcast(self, frame: Dict[str, Any]) -> None:
        """
        Broadcast a ground-truth frame (called from sim thread).
        """
        self.latest_frame = frame
        payload = json.dumps(frame)
        self.latest_json = payload

        # 1. Send via UDP if enabled
        if self._udp_sock is not None and self.udp_port is not None:
            try:
                self._udp_sock.sendto(payload.encode("utf-8"), ("127.0.0.1", self.udp_port))
            except Exception:
                pass

        # 2. Broadcast via WebSocket if clients connected
        if not self.clients or self._loop is None or not self._loop.is_running():
            return

        async def _send_all():
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
        except Exception:
            pass

    def stop(self) -> None:
        """Stop server and disconnect clients."""
        self.running = False
        if self._udp_sock is not None:
            self._udp_sock.close()
            self._udp_sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        log.info("[GROUND TRUTH] Server stopped")
