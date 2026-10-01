#!/usr/bin/env python3
"""
vasimov/edge/transports/udp_transport.py
UDP Transport Server for Virtual Asimov Edge.

Speaks the official asimov.io wire protocol:
- Commands IN: UDP <bind_host>:8850, deserializes asimov.io.RobotCommand.
- State OUT: UDP <state_host>:8851, serializes asimov.io.RobotState at 10 Hz.

PROVENANCE:
- Matches menlo.asimov.transport.udp.UdpTransport:
    * COMMAND_PORT = 8850 [VERIFIED: udp.py line 46]
    * STATE_PORT = 8851 [VERIFIED: udp.py line 48]
    * PROTOCOL_VERSION = 1 [VERIFIED: robots.py line 60]
"""

from __future__ import annotations
import logging
import socket
import threading
import time
from typing import Optional, Tuple
import numpy as np

from asimov_protocol.v1 import asimov_command_pb2, asimov_common_pb2, asimov_state_pb2
from edge.core import EdgeCore
from edge.sim import SimBackend, compute_projected_gravity

log = logging.getLogger("vasimov.edge.udp")

COMMAND_PORT: int = 8850
STATE_PORT: int = 8851
STATE_RATE_HZ: float = 10.0


class UdpEdgeTransport:
    """
    UDP socket server running the command listener and 10 Hz state publisher.
    """

    def __init__(
        self,
        core: EdgeCore,
        backend: SimBackend,
        bind_host: str = "0.0.0.0",
        command_port: int = COMMAND_PORT,
        state_host: str = "127.0.0.1",
        state_port: int = STATE_PORT,
        follow_sender: bool = False,
    ):
        self.core = core
        self.backend = backend
        self.bind_host = bind_host
        self.command_port = command_port
        self.state_host = state_host
        self.state_port = state_port
        self.follow_sender = follow_sender

        self._rx_sock: Optional[socket.socket] = None
        self._tx_sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start UDP sockets and worker threads."""
        self._stop.clear()

        # RX socket (listen for commands on COMMAND_PORT)
        rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx_sock.bind((self.bind_host, self.command_port))
        rx_sock.settimeout(0.2)
        self._rx_sock = rx_sock

        # TX socket (send state to STATE_PORT)
        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx_sock = tx_sock

        # Start threads
        self._rx_thread = threading.Thread(target=self._command_listener_loop, name="vasimov-udp-rx", daemon=True)
        self._tx_thread = threading.Thread(target=self._state_publisher_loop, name="vasimov-udp-tx", daemon=True)

        self._rx_thread.start()
        self._tx_thread.start()

        log.info(
            "[UDP TRANSPORT] Ready: Listening on %s:%d, Pushing telemetry to %s:%d @ %.1f Hz",
            self.bind_host, self.command_port, self.state_host, self.state_port, STATE_RATE_HZ
        )

    def stop(self) -> None:
        """Stop worker threads and close sockets."""
        self._stop.set()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._tx_thread and self._tx_thread.is_alive():
            self._tx_thread.join(timeout=1.0)

        if self._rx_sock:
            self._rx_sock.close()
            self._rx_sock = None
        if self._tx_sock:
            self._tx_sock.close()
            self._tx_sock = None

    def _command_listener_loop(self) -> None:
        """Listen for incoming bare asimov.io.RobotCommand datagrams."""
        sock = self._rx_sock
        while not self._stop.is_set():
            try:
                data, sender = sock.recvfrom(65535)
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue

            if self.follow_sender and sender[0] != self.state_host:
                log.info("[UDP] Updating state target to sender: %s", sender[0])
                self.state_host = sender[0]

            cmd = asimov_command_pb2.RobotCommand()
            try:
                cmd.ParseFromString(data)
            except Exception as e:
                log.warning("[UDP] Undecodable datagram (%d bytes): %s", len(data), e)
                continue

            self._dispatch_command(cmd, sender)

    def _dispatch_command(self, cmd: asimov_command_pb2.RobotCommand, sender: Tuple[str, int]) -> None:
        """Process one valid RobotCommand."""
        # 1. Trajectory Command (HasField "all_trajectory")
        if cmd.HasField("all_trajectory"):
            traj = cmd.all_trajectory
            kp = list(traj.kp) if traj.kp else None
            kd = list(traj.kd) if traj.kd else None
            positions = list(traj.positions)
            self.core.command_trajectory(positions, kp, kd, controller="sdk")
            return

        # 2. Velocity Command (HasField "policy")
        if cmd.HasField("policy"):
            p = cmd.policy
            self.core.command_velocity(p.vx, p.vy, p.vyaw, controller="sdk")
            return

        # 3. Mode Command (STAND / DAMP)
        if cmd.mode == asimov_common_pb2.CONTROL_MODE_STAND:
            current_sim_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
            self.core.command_stand(controller="sdk", current_sim_pos=current_sim_pos)
            return

        if cmd.mode == asimov_common_pb2.CONTROL_MODE_DAMP:
            self.core.command_damp(controller="sdk")
            return

        log.debug("[UDP] Unhandled RobotCommand mode=%d, control=%d", cmd.mode, cmd.command_control)

    def _state_publisher_loop(self) -> None:
        """Push RobotState datagrams on an exact 10 Hz monotonic grid (next += 0.1s)."""
        interval = 1.0 / STATE_RATE_HZ
        sock = self._tx_sock
        next_deadline = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            sleep_time = next_deadline - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            try:
                dest = (self.state_host, self.state_port)
                # Snapshot sim sensors
                data = self.backend.data
                sim_pos = [float(data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                sim_vel = [float(data.qvel[adr]) for adr in self.backend.actuator_dofadr]
                base_quat = [float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6])]
                base_gyro = [float(data.qvel[3]), float(data.qvel[4]), float(data.qvel[5])]
                projected_gravity = list(compute_projected_gravity(np.array(base_quat)))

                # Build RobotState protobuf
                state_msg = self.core.build_robot_state(
                    sim_joint_pos=sim_pos,
                    sim_joint_vel=sim_vel,
                    imu_quat=base_quat,
                    imu_gyro=base_gyro,
                    imu_gravity=projected_gravity,
                )

                payload = state_msg.SerializeToString()
                sock.sendto(payload, dest)
            except Exception as e:
                log.debug("[UDP TX] State send error: %s", e)

            # Advance next grid deadline
            next_deadline += interval
            # If we fell behind by more than 1 interval, catch up to prevent bursting
            if next_deadline < time.monotonic():
                next_deadline = time.monotonic() + interval
