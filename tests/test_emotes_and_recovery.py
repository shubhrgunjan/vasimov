"""
tests/test_emotes_and_recovery.py
Unit and integration tests for Emotes, Safe Fall, and Getup Recovery.

Verifies:
1. Emote catalog and procedural trajectory generator (hello, bow, squat, cheer, dance, nod, shake).
2. SimBackend emote playback, active status, stepping, and cancellation.
3. Safe fall transition into compliant DAMP mode with gantry release.
4. Getup recovery: fault clearance, base height & orientation restoration, STAND arming.
5. Web ControlHandler dispatch of fall, getup, and emote commands.
6. SimConsole command execution and single-key shortcut dispatch (f, g, h, b, c, k, t).
"""

from __future__ import annotations
import unittest
import mujoco
from adapter import SIM_JOINTS
from edge.core import EdgeCore, EdgeMode
from edge.sim import SimBackend
from edge.gestures import EmoteController, EMOTE_CATALOG
from tools.sim_console import SimConsole
from web.server.control_handler import ControlHandler


class TestEmotesAndRecovery(unittest.TestCase):
    """Test suite for emotes, safe fall, and getup recovery."""

    def setUp(self):
        self.core = EdgeCore()
        self.backend = SimBackend(self.core)

    def test_01_emote_catalog_and_controller(self):
        """Verify emote catalog definitions and trajectory evaluation."""
        controller = EmoteController(self.core.default_pose_sim)
        self.assertFalse(controller.is_active)
        self.assertIsNone(controller.current_emote_name)

        # Start hello emote
        self.assertTrue(controller.start_emote("hello"))
        self.assertTrue(controller.is_active)
        self.assertEqual(controller.current_emote_name, "hello")
        self.assertEqual(controller.progress, 0.0)

        # Step controller
        targets = controller.step(0.1)
        self.assertIsNotNone(targets)
        self.assertIn("right_shoulder_pitch_joint", targets)
        self.assertIn("right_elbow_joint", targets)

        # Cancel emote
        controller.stop_emote()
        self.assertFalse(controller.is_active)

        # Unknown emote returns False
        self.assertFalse(controller.start_emote("unknown_super_emote"))

    def test_02_backend_emote_integration(self):
        """Verify SimBackend starts, steps, and stops emotes seamlessly."""
        self.backend.reset()
        self.assertFalse(self.backend.emote_controller.is_active)

        # Play hello
        ok = self.backend.play_emote("hello")
        self.assertTrue(ok)
        self.assertTrue(self.backend.emote_controller.is_active)

        # Step simulation physics
        for _ in range(10):
            self.backend.step()

        # Telemetry state includes emote info
        state = self.backend.get_state()
        self.assertIn("emote", state)
        self.assertTrue(state["emote"]["active"])
        self.assertEqual(state["emote"]["name"], "hello")
        self.assertGreater(state["emote"]["progress"], 0.0)

        # Stop emote
        self.backend.stop_emote()
        self.assertFalse(self.backend.emote_controller.is_active)

    def test_03_safe_fall_execution(self):
        """Verify safe fall transitions to DAMP and releases gantry."""
        self.backend.reset()
        self.core.command_stand("sdk")
        self.core.stand_settled = True
        self.backend.set_gantry(True)

        res = self.backend.safe_fall()
        self.assertEqual(res["status"], "safe_fall")
        self.assertEqual(self.core.mode, EdgeMode.DAMP)
        self.assertFalse(self.backend.gantry_active)
        self.assertEqual(self.core.current_vx, 0.0)

    def test_04_getup_recovery(self):
        """Verify getup recovery clears faults and restores upright standing stance."""
        self.backend.reset()
        # Inject fall fault to lock in FAULT_DAMP
        self.core.inject_fall()
        self.assertTrue(self.core.fault_latched)
        self.assertEqual(self.core.mode, EdgeMode.FAULT_DAMP)

        # Simulate base fallen to ground
        self.backend.data.qpos[2] = 0.20  # on floor
        self.backend.data.qpos[3:7] = [0.7071, 0.7071, 0.0, 0.0]  # tilted

        # Execute getup
        res = self.backend.getup()
        self.assertEqual(res["status"], "recovered")
        self.assertFalse(self.core.fault_latched)
        self.assertEqual(self.core.mode, EdgeMode.STAND)
        self.assertTrue(self.backend.gantry_active)
        self.assertAlmostEqual(self.backend.data.qpos[2], 0.60962, places=3)
        self.assertAlmostEqual(self.backend.data.qpos[3], 1.0, places=3)  # upright quat

    def test_05_control_handler_dispatch(self):
        """Verify web ControlHandler supports fall, getup, and emote commands."""
        handler = ControlHandler(self.backend, self.core)

        # Safe fall
        ack_fall = handler.handle_command("safefall", {})
        self.assertEqual(ack_fall["status"], "accepted")
        self.assertEqual(self.core.mode, EdgeMode.DAMP)

        # Getup
        ack_getup = handler.handle_command("getup", {})
        self.assertEqual(ack_getup["status"], "accepted")
        self.assertEqual(self.core.mode, EdgeMode.STAND)

        # Emote hello
        ack_emote = handler.handle_command("emote", {"name": "hello"})
        self.assertEqual(ack_emote["status"], "accepted")
        self.assertTrue(self.backend.emote_controller.is_active)

        # Direct emote command: bow
        ack_bow = handler.handle_command("bow", {})
        self.assertEqual(ack_bow["status"], "accepted")
        self.assertEqual(self.backend.emote_controller.current_emote_name, "bow")

    def test_06_sim_console_commands_and_shortcuts(self):
        """Verify SimConsole parses and executes new commands and shortcut keys."""
        console = SimConsole(use_viewer=False, realtime=False)
        try:
            # Stand
            console.execute_command("stand")
            self.assertEqual(console.core.mode, EdgeMode.STAND)

            # Emote commands
            console.execute_command("hello")
            self.assertEqual(console.backend.emote_controller.current_emote_name, "hello")

            console.execute_command("bow")
            self.assertEqual(console.backend.emote_controller.current_emote_name, "bow")

            console.execute_command("squat")
            self.assertEqual(console.backend.emote_controller.current_emote_name, "squat")

            console.execute_command("cheer")
            self.assertEqual(console.backend.emote_controller.current_emote_name, "cheer")

            console.execute_command("dance")
            self.assertEqual(console.backend.emote_controller.current_emote_name, "dance")

            # Shortcut keys
            console.execute_command("f")  # safe fall
            self.assertEqual(console.core.mode, EdgeMode.DAMP)

            console.execute_command("g")  # getup
            self.assertEqual(console.core.mode, EdgeMode.STAND)
            self.assertAlmostEqual(console.backend.data.qpos[2], 0.60962, places=3)

            console.execute_command("h")  # hello
            self.assertEqual(console.backend.emote_controller.current_emote_name, "hello")

            console.execute_command("stop")
            self.assertFalse(console.backend.emote_controller.is_active)
        finally:
            console.stop()

    def test_07_dynamic_physics_obstacles(self):
        """Verify dynamic obstacles have 6-DoF freejoints, mass, and fall under gravity."""
        self.backend.load_environment("obstacles")
        self.assertGreater(self.backend.model.nbody, 25)

        # Check for freejoint on obstacles
        freejoints = []
        for j in range(self.backend.model.njnt):
            if self.backend.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                b_id = self.backend.model.jnt_bodyid[j]
                if b_id != self.backend.pelvis_body_id and b_id != 0:
                    freejoints.append(j)

        self.assertGreaterEqual(len(freejoints), 1)

        # Step physics: dynamic bodies should fall under gravity and have valid coordinates
        for _ in range(200):
            self.backend.step()

        first_obs_jnt = freejoints[0]
        qadr = self.backend.model.jnt_qposadr[first_obs_jnt]
        obs_z = self.backend.data.qpos[qadr + 2]
        # Obstacle should rest on floor above z=0
        self.assertGreater(obs_z, 0.0)
        self.assertLess(obs_z, 1.0)

    def test_08_dynamic_obstacle_spawning(self):
        """Verify spawn_dynamic_obstacle repositions physics objects in front of robot."""
        self.backend.load_environment("obstacles")
        res = self.backend.spawn_dynamic_obstacle(distance=1.2, height_offset=0.20)
        self.assertEqual(res["status"], "spawned")
        self.assertIn("obstacle", res)

        pos = res["pos"]
        self.assertAlmostEqual(pos[0], 1.2, delta=0.5)
        self.assertGreater(pos[2], 0.10)

    def test_09_fall_latch_toggle_and_direct_policy(self):
        """Verify fall_latch_enabled=False prevents forced FAULT_DAMP during tilt."""
        self.backend.reset()
        self.core.command_stand("sdk")
        self.core.stand_settled = True
        self.core.fall_latch_enabled = False

        # Simulate severe tilt (gz > -0.50)
        sim_t = float(self.backend.data.time)
        severe_tilt_gravity = [0.8, 0.0, -0.2]  # tilt > 75 deg
        self.core.step_state(sim_t, severe_tilt_gravity)

        # Mode should NOT be forced to FAULT_DAMP
        self.assertNotEqual(self.core.mode, EdgeMode.FAULT_DAMP)
        self.assertTrue(self.core.fault_fall)  # alert is recorded
        self.assertFalse(self.core.fault_latched)

        # Re-enabling latch causes immediate latching on tilt
        self.core.fall_latch_enabled = True
        self.core.step_state(sim_t + 0.01, severe_tilt_gravity)
        self.assertEqual(self.core.mode, EdgeMode.FAULT_DAMP)
        self.assertTrue(self.core.fault_latched)


if __name__ == "__main__":
    unittest.main()

