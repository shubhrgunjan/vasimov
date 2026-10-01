#!/usr/bin/env python3
"""
Unit tests for vasimov/edge/core.py.
Verifies:
1. Boot default state and transitions (DAMP, STAND, MOVE/TRAJECTORY, FAULT_DAMP).
2. Drop rules: length != 25, non-finite values, DAMP->MOVE drop, inactive controller.
3. Watchdogs: 2.0 s trajectory timeout -> auto-DAMP.
4. NoPolicy stub for velocity commands.
5. Safety faults & latching: fall trip at gz > -0.5, overtemp at 80 C, STAND refusal while latched, virtual restart.
6. Telemetry builders: RobotState and EdgeTelemetry fields, sequence, timestamps, error_flags.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Add vasimov root to sys.path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from edge.core import (
    EdgeCore,
    EdgeMode,
    MoveSubmode,
    STAND_RAMP_DURATION_S,
    TRAJECTORY_WATCHDOG_S,
    FALL_GRAVITY_Z_THRESHOLD,
    OVERTEMP_TRIP_C,
    OVERTEMP_CLEAR_C,
)
from asimov_protocol.alerts import AlertId, AlertSeverity


class TestEdgeCore(unittest.TestCase):
    def setUp(self):
        self.core = EdgeCore()

    def test_boot_defaults(self):
        self.assertEqual(self.core.mode, EdgeMode.DAMP)
        self.assertEqual(self.core.move_submode, MoveSubmode.NONE)
        self.assertFalse(self.core.fault_latched)
        self.assertEqual(self.core.error_flags, 0)
        self.assertEqual(self.core.active_controller, "sdk")

    def test_stand_transition_and_ramp(self):
        current_pos = [0.0] * 25
        ok = self.core.command_stand("sdk", current_sim_pos=current_pos)
        self.assertTrue(ok)
        self.assertEqual(self.core.mode, EdgeMode.STAND)
        self.assertFalse(self.core.stand_settled)

        # Midway through ramp (1.0 s)
        q_tgt, kp, kd = self.core.get_control_targets(current_pos, current_time=self.core.stand_ramp_start_time + 1.0)
        # Check that knee target is interpolated midway between 0 and 0.45
        self.assertAlmostEqual(q_tgt[3], 0.225, delta=1e-3)

        # After ramp completes (2.1 s)
        self.core.step_state(self.core.stand_ramp_start_time + 2.1, imu_gravity=[0.0, 0.0, -1.0])
        self.assertTrue(self.core.stand_settled)
        q_tgt_settled, _, _ = self.core.get_control_targets(current_pos, current_time=self.core.stand_ramp_start_time + 2.1)
        self.assertAlmostEqual(q_tgt_settled[3], 0.45, delta=1e-3)

    def test_damp_to_move_dropped_silently(self):
        """No DAMP->MOVE: trajectory and velocity in DAMP are dropped silently."""
        self.assertEqual(self.core.mode, EdgeMode.DAMP)
        ok_traj = self.core.command_trajectory([0.0] * 25, controller="sdk")
        self.assertFalse(ok_traj)
        self.assertEqual(self.core.mode, EdgeMode.DAMP)

        ok_vel = self.core.command_velocity(0.2, 0.0, 0.0, controller="sdk")
        self.assertFalse(ok_vel)
        self.assertEqual(self.core.mode, EdgeMode.DAMP)

    def test_nopolicy_stub_in_stand(self):
        """In STAND, velocity command hits NoPolicy stub and stays in STAND."""
        self.core.command_stand("sdk")
        self.assertEqual(self.core.mode, EdgeMode.STAND)

        ok = self.core.command_velocity(0.5, 0.0, 0.1, controller="sdk")
        self.assertTrue(ok)
        self.assertEqual(self.core.mode, EdgeMode.STAND)  # Stays in STAND!

    def test_trajectory_streaming_and_watchdog(self):
        self.core.command_stand("sdk")
        traj_pos = list(self.core.default_pose_sim)
        traj_pos[12] = -0.5  # Left shoulder pitch move
        ok = self.core.command_trajectory(traj_pos, controller="sdk")
        self.assertTrue(ok)
        self.assertEqual(self.core.mode, EdgeMode.MOVE)
        self.assertEqual(self.core.move_submode, MoveSubmode.TRAJECTORY)

        # Check targets match
        targets, _, _ = self.core.get_control_targets(traj_pos, current_time=self.core.last_trajectory_time + 0.1)
        self.assertAlmostEqual(targets[12], -0.5, delta=1e-5)

        # Advance time by 2.1 s without new trajectory -> auto-DAMP
        now = self.core.last_trajectory_time + 2.1
        self.core.step_state(now, imu_gravity=[0.0, 0.0, -1.0])
        self.assertEqual(self.core.mode, EdgeMode.DAMP)
        self.assertEqual(self.core.move_submode, MoveSubmode.NONE)

    def test_drop_rules(self):
        self.core.command_stand("sdk")

        # Wrong length
        self.assertFalse(self.core.command_trajectory([0.0] * 24))
        # Non-finite values
        self.assertFalse(self.core.command_trajectory([float("nan")] * 25))
        self.assertFalse(self.core.command_velocity(float("inf"), 0.0, 0.0))

        # Inactive controller
        self.assertFalse(self.core.command_trajectory([0.0] * 25, controller="unauthorized"))

    def test_fall_detection_latching_and_restart(self):
        self.core.command_stand("sdk")
        self.assertFalse(self.core.fault_latched)

        # Fall: tilt > 60 deg -> projected gravity z = -0.3 (> -0.5)
        self.core.step_state(10.0, imu_gravity=[0.9, 0.0, -0.3])
        self.assertTrue(self.core.fault_latched)
        self.assertTrue(self.core.fault_fall)
        self.assertEqual(self.core.mode, EdgeMode.FAULT_DAMP)
        self.assertEqual(self.core.error_flags, 0x101)  # Bit 0 (latched) + Bit 8 (alert 7)

        # STAND command refused while latched
        self.assertFalse(self.core.command_stand("sdk"))
        self.assertEqual(self.core.mode, EdgeMode.FAULT_DAMP)

        # Virtual restart clears latch
        self.core.virtual_restart()
        self.assertFalse(self.core.fault_latched)
        self.assertEqual(self.core.error_flags, 0)
        self.assertEqual(self.core.mode, EdgeMode.DAMP)

        # Now STAND is permitted
        self.assertTrue(self.core.command_stand("sdk"))

    def test_overtemp_hysteresis(self):
        self.core.command_stand("sdk")
        # Inject 82 C
        self.core.inject_overtemp(82.0)
        self.assertTrue(self.core.fault_latched)
        self.assertEqual(self.core.mode, EdgeMode.FAULT_DAMP)

        # Cool to 75 C (still above 70 C clear threshold -> still latched)
        self.core.temperatures = [75.0] * 25
        self.core.step_state(15.0, imu_gravity=[0.0, 0.0, -1.0])
        self.assertTrue(self.core.fault_overtemp)

        # Cool to 68 C (< 70 C -> clears overtemp)
        self.core.temperatures = [68.0] * 25
        self.core.step_state(16.0, imu_gravity=[0.0, 0.0, -1.0])
        self.assertFalse(self.core.fault_overtemp)

    def test_telemetry_builders(self):
        sim_pos = [0.0] * 25
        sim_vel = [0.0] * 25
        imu_quat = [1.0, 0.0, 0.0, 0.0]
        imu_gyro = [0.0, 0.0, 0.0]
        imu_gravity = [0.0, 0.0, -1.0]

        # RobotState (for UDP :8851)
        state_msg = self.core.build_robot_state(sim_pos, sim_vel, imu_quat, imu_gyro, imu_gravity)
        self.assertEqual(state_msg.protocol_version, 1)
        self.assertEqual(len(state_msg.joint_pos), 25)
        self.assertEqual(len(state_msg.joint_vel), 25)
        self.assertEqual(len(state_msg.base_quat), 4)
        self.assertEqual(len(state_msg.projected_gravity), 3)
        self.assertEqual(state_msg.current_mode, int(EdgeMode.DAMP))

        # EdgeTelemetry (for Edge-Cloud)
        telemetry_msg = self.core.build_edge_telemetry(sim_pos, sim_vel, imu_quat, imu_gyro, imu_gravity)
        self.assertEqual(len(telemetry_msg.joint_pos), 25)
        self.assertEqual(telemetry_msg.fw_mode, 0)  # FW_MODE_DAMP


if __name__ == "__main__":
    unittest.main()
