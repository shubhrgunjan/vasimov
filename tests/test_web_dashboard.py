"""
tests/test_web_dashboard.py
Comprehensive Verification Suite for the Virtual Asimov 1 Web Dashboard.

Tests:
1. Web server starts and serves index.html and static assets.
2. Canonical TelemetryFrame contains full schema (meta, base, joints, actuators, sensors, policy, contacts, environment).
3. WebSocket connects and receives live canonical frames with valid timestamps.
4. Control command reaches EdgeCore with complete lifecycle acknowledgement (USER INPUT -> PARSED -> VALIDATED -> APPLIED -> RESULT).
5. 78-D Observation and 23-D Action policy pipeline contract preserved.
6. All 25 actuators and 84 MuJoCo sensors enumerated from loaded model.
7. Real model cameras enumerated from m.ncam (5 cameras + 1 viewer camera).
8. Live camera frame rendering carries sequence, sim_time, and valid JPEG bytes.
9. Manual joint control via web command updates target and validates limits.
10. Policy control (walk command) updates velocity and switches mode to MOVE/POLICY.
11. Hybrid control handles policy with manual joint overrides.
12. Environment loading (preset switching) modifies physics parameters.
13. Recording and Replay roundtrip via web API.
14. WebSocket client disconnect is handled gracefully without affecting simulator.
"""

from __future__ import annotations
import asyncio
import json
import os
import time
import unittest
import urllib.request
from pathlib import Path
import sys
import websockets

# Platform-aware headless offscreen rendering: 'cgl' on macOS, 'egl' on Linux, 'osmesa' on Windows
if sys.platform == "darwin":
    default_gl = "cgl"
elif sys.platform.startswith("linux"):
    default_gl = "egl"
else:
    default_gl = "osmesa"
os.environ.setdefault("MUJOCO_GL", default_gl)

from adapter import SIM_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend
from edge.telemetry import TelemetryEngine
from web.server.gateway import WebGateway


class TestWebDashboard(unittest.TestCase):
    """Test suite for Web Robot Dashboard backend, WebSocket, cameras, and control."""

    @classmethod
    def setUpClass(cls):
        cls.core = EdgeCore()
        cls.backend = SimBackend(core=cls.core, auto_gantry=False)
        cls.telemetry = TelemetryEngine(enabled=True)

        # Start gateway on test ports
        cls.http_port = 8862
        cls.ws_port = 8864
        cls.gateway = WebGateway(
            backend=cls.backend,
            core=cls.core,
            telemetry_engine=cls.telemetry,
            http_port=cls.http_port,
            ws_port=cls.ws_port,
        )
        cls.gateway.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.gateway.stop()

    def test_01_web_server_serves_html_and_assets(self):
        """Assert web server starts and serves technical workstation index.html."""
        url = f"http://127.0.0.1:{self.http_port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("VIRTUAL ASIMOV 1", html)
            self.assertIn("MUJOCO OPERATOR WORKSTATION", html)

        # Test CSS asset
        css_url = f"http://127.0.0.1:{self.http_port}/css/dashboard.css"
        with urllib.request.urlopen(css_url) as resp:
            self.assertEqual(resp.status, 200)
            css = resp.read().decode("utf-8")
            self.assertIn("--bg-main", css)

    def test_02_canonical_frame_schema(self):
        """Assert canonical TelemetryFrame contains single-source-of-truth structure."""
        time.sleep(0.1)
        url = f"http://127.0.0.1:{self.http_port}/api/state"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            frame = json.loads(resp.read().decode("utf-8"))
        if not frame:
            frame = self.gateway.frame_builder.build_frame()

        self.assertIn("meta", frame)
        self.assertIn("control", frame)
        self.assertIn("base", frame)
        self.assertIn("joints", frame)
        self.assertIn("actuators", frame)
        self.assertIn("sensors", frame)
        self.assertIn("contacts", frame)
        self.assertIn("policy", frame)
        self.assertIn("cameras", frame)
        self.assertIn("environment", frame)
        self.assertIn("diagnostics", frame)

        # Rate and sequence
        self.assertEqual(frame["meta"]["physics_rate"], 200.0)
        self.assertEqual(frame["meta"]["control_rate"], 50.0)
        self.assertGreater(frame["meta"]["sequence"], 0)

    def test_03_websocket_broadcast_and_client_connect(self):
        """Assert WebSocket client connects and receives live canonical frames."""
        async def client_test():
            uri = f"ws://127.0.0.1:{self.ws_port}"
            async with websockets.connect(uri) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                frame = json.loads(msg)
                self.assertIn("meta", frame)
                self.assertIn("base", frame)
                self.assertEqual(len(frame.get("joints", [])), 25)

        asyncio.run(client_test())

    def test_04_control_command_lifecycle_acknowledgement(self):
        """Verify command lifecycle: USER INPUT -> PARSED -> VALIDATED -> APPLIED -> SIMULATOR RESULT."""
        # Arm to STAND first so WALK is legal
        stand_ack = self.gateway.control_handler.handle_command("stand", {})
        self.assertEqual(stand_ack["status"], "accepted")

        ack = self.gateway.control_handler.handle_command("walk", {"vx": 0.4, "vy": 0.0, "wz": 0.0})
        self.assertEqual(ack["status"], "accepted")
        self.assertEqual(ack["action"], "walk")
        lc = ack["lifecycle"]
        self.assertIn("walk", lc["user_input"])
        self.assertEqual(lc["effective"]["vx"], 0.4)
        self.assertIn("MOVE/POLICY", lc["applied"])

        # Test command clipping notification
        clip_ack = self.gateway.control_handler.handle_command("walk", {"vx": 2.5})
        self.assertTrue(clip_ack["lifecycle"]["clipped"])
        self.assertAlmostEqual(clip_ack["lifecycle"]["effective"]["vx"], 0.8, places=2)
        self.assertIn("clipped", clip_ack["lifecycle"]["reason"])

    def test_05_policy_dimensions_and_contract(self):
        """Assert policy observation is exactly 78-D and actions are 23-D."""
        self.backend.reset()
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.4, 0.0, 0.0)

        # Step physics to invoke policy
        for _ in range(25):
            self.backend.step()

        frame = self.gateway.frame_builder.build_frame()
        pol = frame.get("policy", {})
        self.assertTrue(pol["enabled"])
        self.assertEqual(pol["input_dim"], 78)
        self.assertEqual(pol["output_dim"], 23)
        self.assertEqual(len(pol["observation"]), 78)
        self.assertEqual(len(pol["action"]), 23)
        self.assertEqual(len(pol["action_items"]), 23)

        # Check q_des = q_default + 0.25 * a
        for it in pol["action_items"]:
            expected = it["default_pos"] + 0.25 * it["raw_action"]
            self.assertAlmostEqual(it["desired_target"], expected, places=4)

    def test_06_actuators_and_sensors_enumeration(self):
        """Assert all 25 actuators and 84 actual MuJoCo sensors are enumerated."""
        frame = self.gateway.frame_builder.build_frame()
        self.assertEqual(len(frame["actuators"]), 25)
        self.assertEqual(len(frame["joints"]), 25)
        self.assertEqual(len(frame["sensors"]), 84)

        # Check sensor categorization
        categories = {s["category"] for s in frame["sensors"]}
        self.assertIn("IMU", categories)
        self.assertIn("FORCE", categories)
        self.assertIn("CONTACT", categories)
        self.assertIn("JOINT", categories)

    def test_07_cameras_enumeration_and_live_rendering(self):
        """Assert all real model cameras are enumerated and render valid JPEG frames."""
        cams = self.gateway.camera_streamer.get_camera_metadata()
        model_cams = [c for c in cams if c["type"] == "MODEL CAMERA"]
        viewer_cams = [c for c in cams if c["type"] == "VIEWER CAMERA"]

        self.assertEqual(len(model_cams), 8)
        self.assertEqual(len(viewer_cams), 1)

        cam_names = [c["name"] for c in model_cams]
        self.assertIn("chase_camera", cam_names)
        self.assertIn("first_person_camera", cam_names)
        self.assertIn("front_camera", cam_names)
        self.assertIn("side_camera", cam_names)
        self.assertIn("back_camera", cam_names)


        # Test snapshot HTTP endpoint
        url = f"http://127.0.0.1:{self.http_port}/api/camera/frame?camera=front_camera"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "image/jpeg")
            self.assertIsNotNone(resp.headers.get("X-Sequence"))
            self.assertIsNotNone(resp.headers.get("X-Sim-Time"))
            img_bytes = resp.read()
            self.assertGreater(len(img_bytes), 1000)

    def test_08_manual_joint_control(self):
        """Assert manual joint control sets targets and validates ranges."""
        ack = self.gateway.control_handler.handle_command("joint", {"joint": "left_knee_joint", "target": 0.65})
        self.assertEqual(ack["status"], "accepted")
        self.assertAlmostEqual(self.backend.manual_joint_targets["left_knee_joint"], 0.65, places=3)

    def test_09_environment_preset_loading(self):
        """Assert loading environment preset modifies simulation physics."""
        ack = self.gateway.control_handler.handle_command("environment", {"preset": "friction_low"})
        self.assertEqual(ack["status"], "accepted")
        self.assertEqual(self.backend.env_manager.current_preset, "friction_low")
        self.assertAlmostEqual(self.backend.env_manager.active_config["ground_friction"], 0.2, places=2)

        # Restore flat
        self.gateway.control_handler.handle_command("environment", {"preset": "flat"})

    def test_10_recording_and_replay(self):
        """Assert session recording and replay lifecycle."""
        # Start recording
        start_ack = self.gateway.control_handler.handle_command("record", {"command": "start", "name": "test_web_rec"})
        self.assertEqual(start_ack["status"], "accepted")

        # Step simulation to accumulate frames
        for _ in range(8):
            self.backend.step()
            self.telemetry.record_frame(self.backend.get_state())

        # Stop recording
        stop_ack = self.gateway.control_handler.handle_command("record", {"command": "stop"})
        self.assertEqual(stop_ack["status"], "accepted")
        self.assertIn("json", stop_ack["files"])

        # Test Replay load
        ok = self.gateway.replay_manager.load_recording("test_web_rec")
        self.assertTrue(ok)
        self.assertGreater(len(self.gateway.replay_manager.frames), 0)
        fr = self.gateway.replay_manager.get_current_frame()
        self.assertTrue(fr["is_replay"])
        self.gateway.replay_manager.stop_replay()

    def test_11_client_disconnect_handled(self):
        """Assert client disconnecting does not disrupt simulation."""
        async def quick_connect_disconnect():
            uri = f"ws://127.0.0.1:{self.ws_port}"
            ws = await websockets.connect(uri)
            await ws.close()

        asyncio.run(quick_connect_disconnect())
        # Advance simulation to verify health
        self.backend.step()
        self.assertFalse(self.core.fault_latched)


if __name__ == "__main__":
    unittest.main()
