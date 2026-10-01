#!/usr/bin/env python3
"""
Unit tests for vasimov/adapter.py using standard library unittest.
Verifies:
1. Round-trip exactness for positions, velocities, and torques.
2. Principle of virtual work conservation on ankle coupling.
3. Official standing pose conversion and check_ankle_limits validation.
4. Error checking on dimension mismatches.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import yaml

# Add vasimov root to path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from adapter import (
    JointAdapter,
    FIRMWARE_JOINTS,
    SIM_JOINTS,
    ANKLE_K_PITCH,
    ANKLE_K_ROLL,
    ANKLE_PITCH_LIMIT_RAD,
    ANKLE_ROLL_LIMIT_RAD,
    ankle_pr_to_ab,
    ankle_ab_to_pr,
    ankle_vel_pr_to_ab,
    ankle_vel_ab_to_pr,
    ankle_torque_pr_to_ab,
    ankle_torque_ab_to_pr,
)
from menlo.asimov import robots


class TestJointAdapter(unittest.TestCase):
    def test_joint_counts_and_names(self):
        self.assertEqual(len(FIRMWARE_JOINTS), 25)
        self.assertEqual(len(SIM_JOINTS), 25)
        self.assertEqual(FIRMWARE_JOINTS, robots.ASIMOV_1_BIPED_JOINTS)

    def test_ankle_kinematics_roundtrip(self):
        np.random.seed(42)
        for _ in range(50):
            p_in = float(np.random.uniform(-0.35, 0.35))
            r_in = float(np.random.uniform(-0.10, 0.10))

            a, b = ankle_pr_to_ab(p_in, r_in)
            p_out, r_out = ankle_ab_to_pr(a, b)

            self.assertAlmostEqual(p_in, p_out, delta=1e-12)
            self.assertAlmostEqual(r_in, r_out, delta=1e-12)

    def test_ankle_velocity_roundtrip(self):
        np.random.seed(43)
        for _ in range(50):
            vp_in = float(np.random.uniform(-5.0, 5.0))
            vr_in = float(np.random.uniform(-5.0, 5.0))

            va, vb = ankle_vel_pr_to_ab(vp_in, vr_in)
            vp_out, vr_out = ankle_vel_ab_to_pr(va, vb)

            self.assertAlmostEqual(vp_in, vp_out, delta=1e-12)
            self.assertAlmostEqual(vr_in, vr_out, delta=1e-12)

    def test_ankle_torque_roundtrip(self):
        np.random.seed(44)
        for _ in range(50):
            tp_in = float(np.random.uniform(-30.0, 30.0))
            tr_in = float(np.random.uniform(-15.0, 15.0))

            ta, tb = ankle_torque_pr_to_ab(tp_in, tr_in)
            tp_out, tr_out = ankle_torque_ab_to_pr(ta, tb)

            self.assertAlmostEqual(tp_in, tp_out, delta=1e-12)
            self.assertAlmostEqual(tr_in, tr_out, delta=1e-12)

    def test_virtual_work_conservation(self):
        """Verify that tau_a * v_a + tau_b * v_b == tau_p * v_p + tau_r * v_r."""
        np.random.seed(45)
        for _ in range(50):
            vp = float(np.random.uniform(-2.0, 2.0))
            vr = float(np.random.uniform(-2.0, 2.0))
            tp = float(np.random.uniform(-25.0, 25.0))
            tr = float(np.random.uniform(-10.0, 10.0))

            va, vb = ankle_vel_pr_to_ab(vp, vr)
            ta, tb = ankle_torque_pr_to_ab(tp, tr)

            power_joint = tp * vp + tr * vr
            power_motor = ta * va + tb * vb
            self.assertAlmostEqual(power_joint, power_motor, delta=1e-10)

    def test_full_vector_roundtrip(self):
        np.random.seed(12345)
        for _ in range(20):
            sim_pos = np.random.uniform(-1.0, 1.0, size=25)
            # Keep ankles in realistic ranges
            sim_pos[4] = np.random.uniform(-0.35, 0.35)
            sim_pos[5] = np.random.uniform(-0.10, 0.10)
            sim_pos[10] = np.random.uniform(-0.35, 0.35)
            sim_pos[11] = np.random.uniform(-0.10, 0.10)

            fw_pos = JointAdapter.sim_to_firmware_positions(sim_pos)
            sim_rec = JointAdapter.firmware_to_sim_positions(fw_pos)
            np.testing.assert_allclose(sim_pos, sim_rec, atol=1e-12)

            # Test velocity roundtrip
            sim_vel = np.random.uniform(-2.0, 2.0, size=25)
            fw_vel = JointAdapter.sim_to_firmware_velocities(sim_vel)
            sim_vel_rec = JointAdapter.firmware_to_sim_velocities(fw_vel)
            np.testing.assert_allclose(sim_vel, sim_vel_rec, atol=1e-12)

            # Test torque roundtrip
            sim_tau = np.random.uniform(-10.0, 10.0, size=25)
            fw_tau = JointAdapter.sim_to_firmware_torques(sim_tau)
            sim_tau_rec = JointAdapter.firmware_to_sim_torques(fw_tau)
            np.testing.assert_allclose(sim_tau, sim_tau_rec, atol=1e-12)

    def test_default_standing_pose_mapping(self):
        gains_yaml = _VASIMOV_DIR / "config" / "gains.yaml"
        with open(gains_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pose = cfg["default_pose"]["joints"]
        sim_pos = [float(pose[name]) for name in SIM_JOINTS]

        # Convert to firmware representation
        fw_pos = JointAdapter.sim_to_firmware_positions(sim_pos)

        # Check official limits check function from official menlo.asimov.robots
        robots.check_ankle_limits(fw_pos)

        # Ankle wire positions must equal original pitch/roll
        wire_pos = robots.trajectory_wire_positions(fw_pos)
        self.assertAlmostEqual(wire_pos[4], pose["left_ankle_pitch_joint"], delta=1e-6)
        self.assertAlmostEqual(wire_pos[5], pose["left_ankle_roll_joint"], delta=1e-6)
        self.assertAlmostEqual(wire_pos[10], pose["right_ankle_pitch_joint"], delta=1e-6)
        self.assertAlmostEqual(wire_pos[11], pose["right_ankle_roll_joint"], delta=1e-6)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            JointAdapter.sim_to_firmware_positions([0.0] * 24)
        with self.assertRaises(ValueError):
            JointAdapter.firmware_to_sim_positions([0.0] * 26)


if __name__ == "__main__":
    unittest.main()
