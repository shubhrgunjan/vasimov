"""
tests/test_step6_interactive.py
Unit and integration tests for Step 6: Full Interactive Virtual Robot Prototype.

Tests:
1. Console command parsing and validation (walk, stop, mode, select, set, add, reset).
2. 78-D Observation and 23-D Action policy pipeline contract & scaling.
3. Manual joint target propagation to actuator target and motor model.
4. Live telemetry extraction directly from MuJoCo physics state.
5. Deterministic simulation reset with fixed seed.
6. Environment presets loading (flat, obstacle_basic, friction_low).
7. Flight data recording, CSV/JSON serialization, and replay roundtrip.
8. Full end-to-end integration test (launch -> stand -> walk -> telemetry -> stop).
"""

from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np

from adapter import SIM_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend
from edge.telemetry import TelemetryEngine
from tools.sim_console import SimConsole


class TestStep6InteractivePrototype(unittest.TestCase):
    """Test suite for Step 6 interactive features, control modes, and telemetry."""

    def setUp(self):
        self.core = EdgeCore()
        self.backend = SimBackend(self.core)
        self.telemetry = TelemetryEngine()

    def test_01_console_command_execution(self):
        """Verify console parsing and command dispatch without crashing."""
        console = SimConsole(use_viewer=False, realtime=False)
        try:
            # Stand
            console.execute_command("stand")
            self.assertEqual(console.core.mode, EdgeMode.STAND)

            # Walk
            console.execute_command("walk 0.4")
            self.assertEqual(console.core.mode, EdgeMode.MOVE)
            self.assertEqual(console.core.move_submode, MoveSubmode.POLICY)
            self.assertAlmostEqual(console.core.current_vx, 0.4, places=3)

            # Stop
            console.execute_command("stop")
            self.assertAlmostEqual(console.core.current_vx, 0.0, places=3)

            # Mode switches
            console.execute_command("mode manual")
            self.assertEqual(console.backend.control_mode, "manual")
            console.execute_command("mode hybrid")
            self.assertEqual(console.backend.control_mode, "hybrid")
            console.execute_command("mode policy")
            self.assertEqual(console.backend.control_mode, "policy")

            # Joint manipulation
            console.execute_command("select 4")
            self.assertEqual(console.backend.selected_joint, "left_knee_joint")
            console.execute_command("set 0.5")
            self.assertAlmostEqual(console.backend.manual_joint_targets["left_knee_joint"], 0.5, places=3)
            console.execute_command("add 0.05")
            self.assertAlmostEqual(console.backend.manual_joint_targets["left_knee_joint"], 0.55, places=3)
            console.execute_command("sub 0.10")
            self.assertAlmostEqual(console.backend.manual_joint_targets["left_knee_joint"], 0.45, places=3)

            # Reset
            console.execute_command("reset 42")
            self.assertEqual(console.backend.control_mode, "policy")
            self.assertAlmostEqual(console.backend.data.qpos[2], 0.60962, places=3)
        finally:
            console.stop()

    def test_02_policy_io_dimensions_and_contract(self):
        """Verify 78-D observation and 23-D action dimensions and scaling."""
        self.backend.reset()
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.4, 0.0, 0.0)

        # Run several simulation ticks to invoke policy
        for _ in range(20):
            self.backend.step()

        obs_state = self.backend.get_policy_observation()
        self.assertTrue(obs_state["available"])
        self.assertEqual(obs_state["dim"], 78)
        self.assertEqual(len(obs_state["raw"]), 78)
        self.assertEqual(len(obs_state["labels"]), 78)

        act_state = self.backend.get_policy_action()
        self.assertTrue(act_state["available"])
        self.assertEqual(act_state["dim"], 23)
        self.assertEqual(len(act_state["raw"]), 23)
        self.assertEqual(act_state["action_scale"], 0.25)

        # Verify action synthesis: q_des = default + 0.25 * raw_action
        for item in act_state["items"]:
            expected_q = item["default_pos"] + 0.25 * item["raw_action"]
            self.assertAlmostEqual(item["desired_target"], expected_q, places=5)

    def test_03_manual_and_hybrid_joint_control(self):
        """Verify manual targets propagate to motor targets and hybrid override works."""
        self.backend.reset()
        self.backend.set_control_mode("manual")

        # Set left knee target to 0.7 rad
        self.backend.select_joint("left_knee_joint")
        self.backend.set_manual_joint_target("left_knee_joint", 0.70)

        # Step physics
        for _ in range(10):
            self.backend.step()

        j_states = self.backend.get_joint_state()
        knee_state = next(j for j in j_states if j["name"] == "left_knee_joint")
        self.assertAlmostEqual(knee_state["target"], 0.70, places=3)

        # Now test hybrid mode: policy handles locomotion, but knee is overridden
        self.backend.set_control_mode("hybrid")
        self.backend.set_manual_joint_target("left_knee_joint", 0.65)
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.3, 0.0, 0.0)

        for _ in range(10):
            self.backend.step()

        j_states = self.backend.get_joint_state()
        knee_state = next(j for j in j_states if j["name"] == "left_knee_joint")
        self.assertTrue(knee_state["is_overridden"])
        self.assertAlmostEqual(knee_state["target"], 0.65, places=3)

    def test_04_telemetry_source_of_truth(self):
        """Verify telemetry values reflect genuine MuJoCo data."""
        self.backend.reset()
        for _ in range(10):
            self.backend.step()

        state = self.backend.get_state()
        # Verify base z matches mjData
        self.assertAlmostEqual(state["base"]["z"], float(self.backend.data.qpos[2]), places=5)
        # Verify step count
        self.assertEqual(state["step_count"], self.backend.step_count)
        # Verify contact count matches mjData.ncon
        self.assertEqual(state["contacts"]["contact_count"], int(self.backend.data.ncon))
        # Verify formatting does not fail
        status_txt = self.telemetry.format_status_block(state)
        self.assertIn("ASIMOV 1 STATUS", status_txt)
        joint_txt = self.telemetry.format_joint_table(state)
        self.assertIn("left_hip_pitch_joint", joint_txt)

    def test_05_deterministic_reset(self):
        """Verify deterministic reset with fixed seed produces identical trajectories."""
        # Run 1 with seed 123
        self.backend.reset(seed=123)
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.3, 0.0, 0.0)
        traj_1 = []
        for _ in range(30):
            self.backend.step()
            traj_1.append(float(self.backend.data.qpos[2]))

        # Run 2 with seed 123
        self.backend.reset(seed=123)
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.3, 0.0, 0.0)
        traj_2 = []
        for _ in range(30):
            self.backend.step()
            traj_2.append(float(self.backend.data.qpos[2]))

        self.assertEqual(len(traj_1), len(traj_2))
        for z1, z2 in zip(traj_1, traj_2):
            self.assertAlmostEqual(z1, z2, places=7)

    def test_06_environment_presets_loading(self):
        """Verify multiple environment presets can be loaded and modify physics."""
        # 1. Flat preset
        flat_state = self.backend.load_environment("flat")
        self.assertEqual(flat_state["preset"], "flat")
        self.assertAlmostEqual(flat_state["ground_friction"], 1.0, places=2)

        # 2. Obstacle basic preset
        obs_state = self.backend.load_environment("obstacle_basic")
        self.assertEqual(obs_state["preset"], "obstacle_basic")
        self.assertEqual(obs_state["obstacle_count"], 2)

        # 3. Low friction preset
        fric_state = self.backend.load_environment("friction_low")
        self.assertEqual(fric_state["preset"], "friction_low")
        self.assertAlmostEqual(fric_state["ground_friction"], 0.2, places=2)

        # Cleanup back to flat
        self.backend.load_environment("flat")

    def test_07_recording_and_replay_roundtrip(self):
        """Verify session recording, JSON/CSV artifact creation, and replay."""
        self.backend.reset()
        self.core.command_stand("sdk", current_time=0.0)
        self.core.stand_settled = True
        self.core.command_velocity(0.3, 0.0, 0.0)

        rec_name = "test_roundtrip_unit"
        self.telemetry.start_recording(rec_name, metadata={"test": True, "seed": 42})

        for _ in range(25):
            self.backend.step()
            self.telemetry.record_frame(self.backend.get_state())

        json_path, csv_path = self.telemetry.stop_recording()
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        loaded = self.telemetry.load_recording(rec_name)
        self.assertEqual(loaded["metadata"]["name"], rec_name)
        self.assertEqual(len(loaded["frames"]), 25)
        self.assertTrue(loaded["metadata"]["custom_metadata"]["test"])

    def test_08_full_integration_smoke_test(self):
        """End-to-end integration: launch -> stand -> walk -> telemetry -> pause -> step -> stop."""
        console = SimConsole(use_viewer=False, realtime=False)
        try:
            # 1. Reset
            console.execute_command("reset 100")
            self.assertAlmostEqual(console.backend.data.qpos[2], 0.60962, places=3)

            # 2. Stand
            console.execute_command("stand")
            self.assertEqual(console.core.mode, EdgeMode.STAND)

            # Advance 50 steps
            for _ in range(50):
                console.backend.step()

            # 3. Walk 0.4
            console.execute_command("walk 0.4")
            self.assertEqual(console.core.mode, EdgeMode.MOVE)
            self.assertEqual(console.core.move_submode, MoveSubmode.POLICY)

            # Advance 100 steps
            for _ in range(100):
                console.backend.step()

            # 4. Telemetry check
            state = console.backend.get_state()
            self.assertTrue(state["policy_obs"]["available"])
            self.assertTrue(state["policy_act"]["available"])
            self.assertGreater(state["base"]["x"], -0.1)

            # 5. Pause & Step
            console.execute_command("pause")
            self.assertTrue(console.backend.paused)
            t_before = console.backend.data.time
            console.execute_command("step 2")
            # Stepping 2 control ticks = 8 physics steps
            for _ in range(8):
                console.backend.step()
            self.assertGreater(console.backend.data.time, t_before)

            # 6. Stop
            console.execute_command("resume")
            console.execute_command("stop")
            self.assertAlmostEqual(console.core.current_vx, 0.0, places=3)
        finally:
            console.stop()


if __name__ == "__main__":
    unittest.main()
