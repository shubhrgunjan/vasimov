#!/usr/bin/env python3
"""
vasimov/tests/test_motor_model.py
Unit tests for each layer of the layered motor model (L0 - L4).
"""

import math
from pathlib import Path
import sys
import unittest
import numpy as np

# Add vasimov root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motor_model import MotorModel, MotorState


class TestMotorModel(unittest.TestCase):
    def test_l0_pd_and_clipping(self):
        """L0: Verify standard PD law and clipping to effort limit."""
        mm = MotorModel()
        mm.set_layer_l1(False)
        mm.set_layer_l3(False)

        cfg = mm.motors[0]
        q = np.zeros(mm.num_motors)
        qdot = np.zeros(mm.num_motors)
        q_des = np.zeros(mm.num_motors)
        qdot_des = np.zeros(mm.num_motors)

        # Case 1: Small error within torque limit
        q_des[0] = 0.05  # 0.05 rad * 250 = 12.5 Nm
        torques, states = mm.step(q, qdot, q_des, qdot_des, dt=0.005)
        expected_tau = cfg.kp * 0.05
        self.assertAlmostEqual(torques[0], expected_tau, places=4)
        self.assertAlmostEqual(states[cfg.joint_name].clamped_torque, expected_tau, places=4)

        # Case 2: Large error exceeding effort limit (45 Nm)
        q_des[0] = 1.0  # 1.0 rad * 250 = 250 Nm -> clipped to 45.0 Nm
        torques, states = mm.step(q, qdot, q_des, qdot_des, dt=0.005)
        self.assertAlmostEqual(torques[0], cfg.effort_limit, places=4)
        self.assertEqual(states[cfg.joint_name].raw_pd_torque, cfg.kp * 1.0)
        self.assertEqual(states[cfg.joint_name].clamped_torque, cfg.effort_limit)

        # Case 3: Negative error exceeding effort limit
        q_des[0] = -1.0
        torques, states = mm.step(q, qdot, q_des, qdot_des, dt=0.005)
        self.assertAlmostEqual(torques[0], -cfg.effort_limit, places=4)

    def test_l1_velocity_dependent_torque_limit(self):
        """L1: Verify speed-dependent torque derating envelope tau_max(w)."""
        mm = MotorModel()
        mm.set_layer_l1(True)
        mm.set_layer_l3(False)

        cfg = mm.motors[0]
        effort = cfg.effort_limit
        v_limit = cfg.velocity_limit

        # Zero velocity: 100% effort available
        limit_0 = mm.compute_torque_limit(0, 0.0)
        self.assertAlmostEqual(limit_0, effort, places=4)

        # Half velocity limit: 50% effort available
        limit_half = mm.compute_torque_limit(0, 0.5 * v_limit)
        self.assertAlmostEqual(limit_half, 0.5 * effort, places=4)

        # At velocity limit: 0% effort available
        limit_full = mm.compute_torque_limit(0, v_limit)
        self.assertAlmostEqual(limit_full, 0.0, places=4)

        # Exceeding velocity limit: clamped at 0.0 (no negative limits)
        limit_over = mm.compute_torque_limit(0, 1.5 * v_limit)
        self.assertEqual(limit_over, 0.0)

        # Negative velocity has same symmetric limit
        limit_neg_half = mm.compute_torque_limit(0, -0.5 * v_limit)
        self.assertAlmostEqual(limit_neg_half, 0.5 * effort, places=4)

    def test_l2_rotor_armature_and_damping(self):
        """L2: Verify armature and damping parameters are populated for all motors."""
        mm = MotorModel()
        self.assertTrue(mm.layer_opts.l2_enabled)
        self.assertEqual(mm.layer_opts.l2_mode, "mujoco_xml")

        for m in mm.motors:
            self.assertGreater(m.armature, 0.0, f"Joint {m.joint_name} has invalid armature {m.armature}")
            self.assertGreater(m.damping, 0.0, f"Joint {m.joint_name} has invalid damping {m.damping}")

    def test_l3_command_lag_and_delay(self):
        """L3: Verify first-order command lag filter and discrete step delay."""
        mm = MotorModel()
        q = np.zeros(mm.num_motors)
        qdot = np.zeros(mm.num_motors)
        q_des = np.zeros(mm.num_motors)
        qdot_des = np.zeros(mm.num_motors)

        # Test 1: First-order low pass filter
        mm.set_layer_l1(False)
        lag_tau = 0.02
        dt = 0.005
        mm.set_layer_l3(True, mode="lag", lag_tau=lag_tau)

        q_des[0] = 0.1  # Step target: raw torque = 25.0 Nm
        alpha = dt / (dt + lag_tau)  # 0.005 / 0.025 = 0.2

        # Step 1
        t1, _ = mm.step(q, qdot, q_des, qdot_des, dt=dt)
        expected_t1 = alpha * 25.0
        self.assertAlmostEqual(t1[0], expected_t1, places=3)

        # Step 2
        t2, _ = mm.step(q, qdot, q_des, qdot_des, dt=dt)
        expected_t2 = alpha * 25.0 + (1.0 - alpha) * expected_t1
        self.assertAlmostEqual(t2[0], expected_t2, places=3)

        # Test 2: Discrete step delay
        mm.set_layer_l3(True, mode="delay")
        mm.layer_opts.l3_delay_steps = 2
        mm.reset()

        # Initial output should be 0.0 for first 2 steps
        t_d1, _ = mm.step(q, qdot, q_des, qdot_des, dt=dt)
        self.assertEqual(t_d1[0], 0.0)
        t_d2, _ = mm.step(q, qdot, q_des, qdot_des, dt=dt)
        self.assertEqual(t_d2[0], 0.0)
        # Step 3 should receive step 1 torque
        t_d3, _ = mm.step(q, qdot, q_des, qdot_des, dt=dt)
        self.assertAlmostEqual(t_d3[0], 25.0, places=3)

    def test_l4_thermal_and_current_not_modeled(self):
        """L4: Verify current and temperature return 'NOT MODELED' without fabricated values."""
        mm = MotorModel()
        q = np.zeros(mm.num_motors)
        qdot = np.zeros(mm.num_motors)
        q_des = np.zeros(mm.num_motors)
        qdot_des = np.zeros(mm.num_motors)

        _, states = mm.step(q, qdot, q_des, qdot_des, dt=0.005)
        for name, state in states.items():
            self.assertEqual(state.current, "NOT MODELED", f"Joint {name} returned fake current!")
            self.assertEqual(state.temperature, "NOT MODELED", f"Joint {name} returned fake temperature!")


if __name__ == "__main__":
    unittest.main()
