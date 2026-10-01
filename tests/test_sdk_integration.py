#!/usr/bin/env python3
"""
vasimov/tests/test_sdk_integration.py
Full End-to-End Integration Tests using the OFFICIAL, UNMODIFIED menlo-sdk.

Tests:
1. Connect over UDP transport (host: 127.0.0.1, commands 8850, state 8851).
2. Handshake & Preflight checks: protocol v1, 25 joints, preflight OK.
3. Posture transitions: DAMP -> STAND (armed in STAND, joint_pos near default pose).
4. Trajectory streaming at 50 Hz moving L_Shoulder_Pitch and L_Elbow.
5. Trajectory watchdog: 2.5s silence triggers auto-DAMP.
6. Drop rules: DAMP drops trajectory / velocity.
7. NoPolicy stub: velocity command handling.
8. Safety faults: inject fall, verify latching in FAULT_DAMP, STAND refusal, virtual restart.
9. Ankle kinematics round trip and enum numbering verification.
"""

from __future__ import annotations
import http.client
import json
import logging
import sys
import threading
import time
import unittest
from pathlib import Path
import numpy as np

# Add vasimov to sys.path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

# Import official unmodified menlo SDK
from menlo.asimov import (
    ConnectionConfig,
    Mode,
    NotReadyError,
    Robot,
    RobotFaultedError,
    UdpConfig,
    robots,
)
from asimov_protocol.v1 import asimov_common_pb2, edge_cloud_pb2

from edge.core import EdgeCore, EdgeMode
from edge.sim import SimBackend, SimControlServer
from edge.transports.udp_transport import UdpEdgeTransport, COMMAND_PORT, STATE_PORT
from adapter import JointAdapter, SIM_JOINTS

log = logging.getLogger("vasimov.test.sdk_integration")


class TestSdkIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start Virtual Edge in background threads for the test session."""
        cls.core = EdgeCore()
        cls.backend = SimBackend(core=cls.core, auto_gantry=True)

        # Control server
        cls.control_server = SimControlServer(cls.backend, port=8852)
        cls.control_server.start()

        # UDP Transport
        cls.transport = UdpEdgeTransport(
            core=cls.core,
            backend=cls.backend,
            command_port=8850,
            state_host="127.0.0.1",
            state_port=8851,
        )
        cls.transport.start()

        # Background simulation step loop
        cls.running = True
        cls.sim_thread = threading.Thread(target=cls._run_sim_loop, daemon=True, name="test-sim-loop")
        cls.sim_thread.start()

        # Let simulation warm up and start pushing telemetry
        time.sleep(0.5)

    @classmethod
    def _run_sim_loop(cls):
        """Simulation loop running at physics rate."""
        sim_dt = cls.backend.dt
        while cls.running:
            t0 = time.perf_counter()
            cls.backend.step()
            elapsed = time.perf_counter() - t0
            sleep_s = sim_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    @classmethod
    def tearDownClass(cls):
        """Cleanly stop Virtual Edge."""
        cls.running = False
        cls.transport.stop()
        cls.control_server.stop()

    def _post_control(self, payload: dict) -> dict:
        """Helper to post to local HTTP control server."""
        conn = http.client.HTTPConnection("127.0.0.1", 8852, timeout=2.0)
        body = json.dumps(payload)
        conn.request("POST", "/control", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return json.loads(data.decode("utf-8"))

    def test_01_connect_and_preflight(self):
        """Verify unmodified Robot connects over UDP and passes preflight."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1", command_port=8850))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            self.assertTrue(robot.connected)
            info = robot.info
            self.assertEqual(info.dof, 25)
            self.assertEqual(info.protocol_version, 1)
            self.assertIn("drive", info.capabilities)
            self.assertIn("state", info.capabilities)

            state = robot.get_state()
            self.assertEqual(len(state.joints), 25)
            self.assertIsNotNone(state.gravity)
            self.assertIsNotNone(state.quat)

            # Preflight stand should be OK from DAMP
            check = robot.preflight("stand")
            self.assertTrue(check.ok, f"Preflight stand failed: {check}")

    def test_02_damp_and_stand_transitions(self):
        """Verify damp() and stand() work and robot arms in STAND."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            # 1. Damp
            robot.damp(wait=True, timeout=2.0)
            state = robot.get_state()
            self.assertIn(state.mode, (Mode.DAMP, Mode.FAULT_DAMP))

            # 2. Stand
            # stand() sends STAND and waits until robot is armed in STAND (upright >= 0.5s)
            robot.stand(wait=True, timeout=6.0)
            state = robot.get_state()
            self.assertEqual(state.mode, Mode.STAND)
            self.assertTrue(robot.armed)

            # Check that joint positions are near the official default pose
            default_fw = self.core.default_pose_fw
            for i, j in enumerate(state.joints):
                expected = default_fw[i]
                self.assertAlmostEqual(
                    j.pos, expected, delta=0.08,
                    msg=f"Joint {j.name} pos={j.pos:.3f} differs from default={expected:.3f}"
                )

    def test_03_trajectory_streaming_and_watchdog(self):
        """Stream trajectory at 50 Hz moving arm joints, then verify auto-DAMP on silence."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.stand(wait=True, timeout=6.0)

            # Start pose
            start_pose = list(robot.get_state().joint_pos)
            target_pose = list(start_pose)
            # Move L_Shoulder_Pitch (idx 12) to -0.5 and L_Elbow (idx 15) to 0.7
            target_pose[12] = -0.50
            target_pose[15] = 0.70

            # Stream at 50 Hz for 1.0 s (50 setpoints)
            steps = 50
            dt = 0.020  # 50 Hz
            for step in range(1, steps + 1):
                alpha = step / steps
                current_target = [
                    p0 + (p1 - p0) * alpha
                    for p0, p1 in zip(start_pose, target_pose, strict=True)
                ]
                robot.trajectory(current_target, timeout=0.0)
                time.sleep(dt)

            # Verify robot reached near target and is in MOVE
            time.sleep(0.1)
            state = robot.get_state()
            self.assertEqual(state.mode, Mode.MOVE)
            self.assertAlmostEqual(state.joints[12].pos, -0.50, delta=0.10)
            self.assertAlmostEqual(state.joints[15].pos, 0.70, delta=0.10)

            # Stop streaming for 2.5 s -> Edge trajectory watchdog should auto-DAMP
            log.info("Stopping trajectory stream; waiting 2.5s for auto-DAMP...")
            auto_damped = robot.wait_until(lambda s: s.mode is Mode.DAMP, timeout=3.5)
            self.assertEqual(auto_damped.mode, Mode.DAMP)

    def test_04_damp_drop_rules(self):
        """Verify that in DAMP, motion commands are refused by SDK or dropped."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.damp(wait=True, timeout=2.0)
            check = robot.preflight("trajectory")
            self.assertFalse(check.ok)
            self.assertTrue(check.has("wrong_mode"))

    def test_05_velocity_in_stand_hits_nopolicy(self):
        """Verify velocity commands in STAND hit NoPolicy stub and stay in STAND."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.stand(wait=True, timeout=6.0)

            # Calling balance() sends zero velocity to enter MOVE;
            # Virtual Edge NoPolicy stub intercepts velocity, logs loudly, and remains in STAND
            robot.balance(wait=False)
            time.sleep(0.2)
            state = robot.get_state()
            self.assertEqual(state.mode, Mode.STAND)

    def test_06_safety_fault_fall_and_restart(self):
        """Inject fall fault: verify FAULT_DAMP latches, STAND refused, restart clears."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.stand(wait=True, timeout=6.0)

            # Inject fall fault via local control interface
            res = self._post_control({"command": "inject", "fault": "fall"})
            self.assertEqual(res.get("status"), "ok")

            # Verify robot state reflects fault latch (wait up to 2s for 10Hz packet)
            deadline = time.monotonic() + 2.0
            state = None
            while time.monotonic() < deadline:
                s = robot.get_state()
                if s and s.faulted:
                    state = s
                    break
                time.sleep(0.05)
            self.assertIsNotNone(state, "Timed out waiting for faulted state in telemetry")
            self.assertTrue(state.faulted)
            self.assertEqual(state.mode, Mode.FAULT_DAMP)
            self.assertNotEqual(state.error_flags, 0)

            # Preflight stand should now report faulted
            check = robot.preflight("stand")
            self.assertFalse(check.ok)
            self.assertTrue(check.has("faulted"))

            # Calling stand() should raise RobotFaultedError
            with self.assertRaises(RobotFaultedError):
                robot.stand(timeout=1.0)

            # Virtual restart via control interface
            res_rst = self._post_control({"command": "restart"})
            self.assertEqual(res_rst.get("status"), "ok")

            # Wait up to 2s for clear sample after restart
            deadline = time.monotonic() + 2.0
            cleared = None
            while time.monotonic() < deadline:
                s = robot.get_state()
                if s and not s.faulted and s.mode is Mode.DAMP:
                    cleared = s
                    break
                time.sleep(0.05)
            self.assertIsNotNone(cleared, "Timed out waiting for fault clear after restart")
            self.assertFalse(cleared.faulted)
            self.assertEqual(cleared.error_flags, 0)

            # STAND now succeeds again!
            robot.stand(wait=True, timeout=6.0)
            self.assertEqual(robot.get_state().mode, Mode.STAND)

    def test_07_ankle_roundtrip_and_enum_trap(self):
        """Verify ankle mapping round trip and enum numbers."""
        # Ankle kinematics round trip
        for pitch, roll in [(-0.30, 0.0), (0.15, -0.05), (0.0, 0.08)]:
            sim_vec = [0.0] * 25
            sim_vec[4] = pitch
            sim_vec[5] = roll
            fw_vec = JointAdapter.sim_to_firmware_positions(sim_vec)
            sim_rec = JointAdapter.firmware_to_sim_positions(fw_vec)
            self.assertAlmostEqual(sim_rec[4], pitch, delta=1e-6)
            self.assertAlmostEqual(sim_rec[5], roll, delta=1e-6)

        # ENUM TRAP Verification
        # Common / State
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_DAMP, 0)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_STAND, 1)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_MOVE, 2)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_FAULT_DAMP, 5)

        # Edge Cloud Command Mode: MODE_STAND = 0, MODE_DAMP = 1
        self.assertEqual(edge_cloud_pb2.MODE_STAND, 0)
        self.assertEqual(edge_cloud_pb2.MODE_DAMP, 1)

        # Edge Cloud Firmware Mode: FW_MODE_DAMP = 0, FW_MODE_STAND = 1, FW_MODE_MOVE = 2
        self.assertEqual(edge_cloud_pb2.FW_MODE_DAMP, 0)
        self.assertEqual(edge_cloud_pb2.FW_MODE_STAND, 1)
        self.assertEqual(edge_cloud_pb2.FW_MODE_MOVE, 2)

    def test_08_sdk_ankle_trajectory_and_limits(self):
        """Via Robot.trajectory command small ankle pitch/roll changes and verify limits."""
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1", command_port=8850))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.stand(wait=True, timeout=6.0)

            # 1. Test SDK ankle limit checking
            current_targets = list(robot.get_state().joint_pos)
            # Motor values corresponding to an excessive pitch (> 0.35 + 0.02 = 0.37 rad)
            # A = 2.02 * 0.45 = 0.909, B = -0.909
            bad_targets = list(current_targets)
            bad_targets[4] = 0.95
            bad_targets[5] = -0.95
            with self.assertRaises(ValueError):
                robot.trajectory(bad_targets, timeout=0.0)

            # 2. Command valid small ankle pitch & roll change via SDK
            # Desired pitch = -0.20 rad, roll = +0.03 rad
            # A = 2.02*(-0.20) - 0.8*(0.03) = -0.404 - 0.024 = -0.428
            # B = -2.02*(-0.20) - 0.8*(0.03) = 0.404 - 0.024 = 0.380
            valid_targets = list(current_targets)
            valid_targets[4] = -0.428
            valid_targets[5] = 0.380

            # Stream at 50 Hz for 0.5 s (25 steps)
            for _ in range(25):
                robot.trajectory(valid_targets, timeout=0.0)
                time.sleep(0.02)

            time.sleep(0.1)
            state = robot.get_state()
            self.assertAlmostEqual(state.joints[4].pos, -0.428, delta=0.08)
            self.assertAlmostEqual(state.joints[5].pos, 0.380, delta=0.08)

            # Decoded ankle pitch and roll match target
            pitch = (state.joints[4].pos - state.joints[5].pos) / (2 * robots.ANKLE_K_PITCH)
            roll = -(state.joints[4].pos + state.joints[5].pos) / (2 * robots.ANKLE_K_ROLL)
            self.assertAlmostEqual(pitch, -0.20, delta=0.04)
            self.assertAlmostEqual(roll, 0.03, delta=0.03)


if __name__ == "__main__":
    unittest.main()
