#!/usr/bin/env python3
"""
vasimov/tests/test_ankle_map.py
Unit tests for DifferentialAnkleMap module:
- Round trip conversions (ab -> pr -> ab, pr -> ab -> pr)
- Velocity and torque mapping consistency & virtual work preservation
- Joint limit clipping
- Warning emission
"""

import math
from pathlib import Path
import sys
import unittest
import warnings
import numpy as np

# Add vasimov root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ankle_map import DifferentialAnkleMap, AnkleLimits


class TestAnkleMap(unittest.TestCase):
    def setUp(self):
        self.limits = AnkleLimits(
            pitch_min=-0.35,
            pitch_max=0.35,
            roll_min=-0.10,
            roll_max=0.10,
            pitch_effort_limit=40.0,
            roll_effort_limit=17.0,
            actuator_effort_limit=36.0,
        )
        self.ankle_map = DifferentialAnkleMap(pitch_ratio=1.0, roll_ratio=1.0, limits=self.limits)

    def test_roundtrip_pr_to_ab_to_pr(self):
        """Test that pr -> ab -> pr recovers original pitch and roll."""
        test_points = [
            (0.0, 0.0),
            (0.25, 0.05),
            (-0.30, -0.08),
            (0.15, -0.05),
            (-0.20, 0.07),
        ]
        for p, r in test_points:
            qa, qb = self.ankle_map.pr_to_ab(p, r)
            p_rec, r_rec = self.ankle_map.ab_to_pr(qa, qb)
            self.assertAlmostEqual(p, p_rec, places=9)
            self.assertAlmostEqual(r, r_rec, places=9)

    def test_roundtrip_ab_to_pr_to_ab(self):
        """Test that ab -> pr -> ab recovers original actuator angles."""
        test_points = [
            (0.0, 0.0),
            (0.10, 0.10),
            (0.20, -0.15),
            (-0.25, 0.05),
            (0.35, 0.25),
        ]
        for qa, qb in test_points:
            p, r = self.ankle_map.ab_to_pr(qa, qb)
            qa_rec, qb_rec = self.ankle_map.pr_to_ab(p, r)
            self.assertAlmostEqual(qa, qa_rec, places=9)
            self.assertAlmostEqual(qb, qb_rec, places=9)

    def test_velocity_and_virtual_work_consistency(self):
        """
        Verify velocity mapping and that virtual power is strictly preserved:
        P = tau_ab^T * dq_ab == tau_pr^T * dq_pr
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            dqa, dqb = rng.uniform(-5.0, 5.0, size=2)
            tau_p, tau_r = rng.uniform(-20.0, 20.0, size=2)

            # Map velocity ab -> pr
            dqp, dqr = self.ankle_map.velocity_ab_to_pr(dqa, dqb)
            # Map back
            dqa_rec, dqb_rec = self.ankle_map.velocity_pr_to_ab(dqp, dqr)
            self.assertAlmostEqual(dqa, dqa_rec, places=8)
            self.assertAlmostEqual(dqb, dqb_rec, places=8)

            # Map torque pr -> ab
            tau_a, tau_b = self.ankle_map.torque_pr_to_ab(tau_p, tau_r)
            # Map back
            tau_p_rec, tau_r_rec = self.ankle_map.torque_ab_to_pr(tau_a, tau_b)
            self.assertAlmostEqual(tau_p, tau_p_rec, places=8)
            self.assertAlmostEqual(tau_r, tau_r_rec, places=8)

            # Virtual power test
            power_ab = tau_a * dqa + tau_b * dqb
            power_pr = tau_p * dqp + tau_r * dqr
            self.assertAlmostEqual(power_ab, power_pr, places=8)

    def test_joint_limits_clipping(self):
        """Test clipping for joint angles and actuator torques."""
        # Pitch exceeding max (0.35)
        p_clip, r_clip = self.ankle_map.clip_pr(0.50, 0.05)
        self.assertEqual(p_clip, 0.35)
        self.assertEqual(r_clip, 0.05)

        # Roll exceeding min (-0.10)
        p_clip, r_clip = self.ankle_map.clip_pr(-0.20, -0.25)
        self.assertEqual(p_clip, -0.20)
        self.assertEqual(r_clip, -0.10)

        # Torques exceeding actuator peak limit (36.0 Nm)
        ta_clip, tb_clip = self.ankle_map.clip_ab_torques(50.0, -45.0)
        self.assertEqual(ta_clip, 36.0)
        self.assertEqual(tb_clip, -36.0)

    def test_non_unity_ratios(self):
        """Verify behavior with non-unity crank transmission ratios."""
        custom_map = DifferentialAnkleMap(pitch_ratio=0.8, roll_ratio=1.2)
        qa, qb = 0.20, -0.10
        p, r = custom_map.ab_to_pr(qa, qb)
        # p = 0.8 * (0.2 - 0.1) / 2 = 0.8 * 0.05 = 0.04
        # r = 1.2 * (0.2 - -0.1) / 2 = 1.2 * 0.15 = 0.18
        self.assertAlmostEqual(p, 0.04, places=8)
        self.assertAlmostEqual(r, 0.18, places=8)

        # Roundtrip
        qa_rec, qb_rec = custom_map.pr_to_ab(p, r)
        self.assertAlmostEqual(qa, qa_rec, places=8)
        self.assertAlmostEqual(qb, qb_rec, places=8)


if __name__ == "__main__":
    unittest.main()
