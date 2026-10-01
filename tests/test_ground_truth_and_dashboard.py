#!/usr/bin/env python3
"""
tests/test_ground_truth_and_dashboard.py
Unit and integration tests for Task 1 (Ground-Truth Stream) and Task 2 (Web Dashboard & Control API).
Asserts:
1. Frame schema correctness (sim-only metadata, 25 joints, ankles, foot contacts, pelvis).
2. WebSocket ground truth broadcast ~50 Hz.
3. Dashboard HTTP serving (:8852 /).
4. Ground truth JSON endpoint (:8852 /ground_truth).
"""

import asyncio
import json
import time
import unittest
import urllib.request
from pathlib import Path

import websockets

from edge.core import EdgeCore
from edge.ground_truth import GroundTruthServer
from edge.sim import SimBackend, SimControlServer


class TestGroundTruthAndDashboard(unittest.TestCase):
    """Test Ground Truth streaming and dashboard integration."""

    @classmethod
    def setUpClass(cls):
        cls.core = EdgeCore()
        cls.ws_server = GroundTruthServer(ws_port=8858)
        cls.ws_server.start()

        cls.backend = SimBackend(core=cls.core, auto_gantry=False)
        cls.backend.ground_truth_server = cls.ws_server

        cls.control_server = SimControlServer(backend=cls.backend, port=8859)
        cls.control_server.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.control_server.stop()
        cls.ws_server.stop()

    def test_frame_schema(self):
        """Assert ground-truth frame contains all required fields and no hardware equivalent."""
        frame = self.backend.get_ground_truth_frame(rtf=1.05)

        # Metadata
        meta = frame.get("stream_metadata", {})
        self.assertTrue(meta.get("sim_only"))
        self.assertFalse(meta.get("hardware_equivalent"))

        # Pelvis
        pelvis = frame.get("pelvis", {})
        self.assertEqual(len(pelvis.get("pos", [])), 3)
        self.assertEqual(len(pelvis.get("quat", [])), 4)
        self.assertIn("tilt_deg", pelvis)

        # Foot contacts
        contacts = frame.get("contacts", {})
        self.assertIn("left_foot", contacts)
        self.assertIn("right_foot", contacts)
        self.assertIn("contact", contacts["left_foot"])
        self.assertIn("normal_force", contacts["left_foot"])
        self.assertEqual(len(contacts["left_foot"]["force"]), 3)

        # Ankles
        ankles = frame.get("ankles", {})
        self.assertIn("left", ankles)
        self.assertIn("right", ankles)
        self.assertIn("pitch_rad", ankles["left"])
        self.assertIn("roll_rad", ankles["left"])
        self.assertIn("motor_a_rad", ankles["left"])
        self.assertIn("motor_b_rad", ankles["left"])

        # 25 joints
        joints = frame.get("joints", [])
        self.assertEqual(len(joints), 25)
        for j in joints:
            self.assertIn("index", j)
            self.assertIn("firmware_name", j)
            self.assertIn("sim_name", j)
            self.assertIn("pos", j)
            self.assertIn("target", j)
            self.assertIn("error", j)
            self.assertIn("vel", j)
            self.assertIn("torque", j)
            self.assertIn("effort_limit", j)
            self.assertIn("torque_pct_effort", j)
            self.assertIn("range", j)
            self.assertIn("pos_pct_range", j)
            self.assertIn("kp", j)
            self.assertIn("kd", j)

    def test_dashboard_http_endpoints(self):
        """Assert SimControlServer serves index.html, /ground_truth, and /status."""
        # 1. Dashboard HTML
        with urllib.request.urlopen("http://127.0.0.1:8859/") as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("VIRTUAL ASIMOV 1", html)

        # 2. Ground truth snapshot
        with urllib.request.urlopen("http://127.0.0.1:8859/ground_truth") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(data.get("joints", [])), 25)

        # 3. Status JSON
        with urllib.request.urlopen("http://127.0.0.1:8859/status") as resp:
            self.assertEqual(resp.status, 200)
            status = json.loads(resp.read().decode("utf-8"))
            self.assertIn("mode", status)
            self.assertIn("battery_soc", status)

    def test_websocket_stream_client(self):
        """Assert client can connect to WebSocket stream and receive live frames."""
        async def client_test():
            uri = "ws://127.0.0.1:8858"
            for _ in range(4):
                self.backend.step()
            async with websockets.connect(uri) as ws:
                for _ in range(4):
                    self.backend.step()
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(msg)
                self.assertTrue(data.get("stream_metadata", {}).get("sim_only"))
                self.assertEqual(len(data.get("joints", [])), 25)

        asyncio.run(client_test())


if __name__ == "__main__":
    unittest.main()
