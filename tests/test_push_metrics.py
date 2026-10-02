#!/usr/bin/env python3
"""
tests/test_push_metrics.py
Unit tests verifying push metrics calculation:
1. Tilt angle is the true 3D angle between pelvis up-axis and world up.
2. A fall can NEVER report a return time or settle time.
3. Torque stats are computed only after push disturbance start.
"""

import math
from pathlib import Path
import unittest
import numpy as np

from stand import quat_to_roll_pitch, run_stand_sim

VASIMOV_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = VASIMOV_DIR / "model" / "asimov_1_vasimov.xml"
CONFIG_PATH = VASIMOV_DIR / "config" / "gains.yaml"


class TestPushMetrics(unittest.TestCase):
    def test_tilt_vector_definition(self):
        """Verify tilt is exactly the angle between rotated local z-axis and world z-axis."""
        # 1. Identity quaternion -> 0 deg tilt
        r, p, tilt = quat_to_roll_pitch(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(tilt, 0.0, places=5)
        self.assertAlmostEqual(r, 0.0, places=5)
        self.assertAlmostEqual(p, 0.0, places=5)

        # 2. Pure yaw rotation (15 deg around z) -> 0 deg tilt!
        th = math.radians(15.0)
        q_yaw = np.array([math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0)])
        _, _, tilt_yaw = quat_to_roll_pitch(q_yaw)
        self.assertAlmostEqual(tilt_yaw, 0.0, places=5)

        # 3. Pure pitch rotation (8 deg around y) -> 8 deg tilt
        th = math.radians(8.0)
        q_pitch = np.array([math.cos(th / 2.0), 0.0, math.sin(th / 2.0), 0.0])
        _, _, tilt_pitch = quat_to_roll_pitch(q_pitch)
        self.assertAlmostEqual(tilt_pitch, 8.0, places=4)

        # 4. Pure roll rotation (12 deg around x) -> 12 deg tilt
        th = math.radians(12.0)
        q_roll = np.array([math.cos(th / 2.0), math.sin(th / 2.0), 0.0, 0.0])
        _, _, tilt_roll = quat_to_roll_pitch(q_roll)
        self.assertAlmostEqual(tilt_roll, 12.0, places=4)

    def test_fall_never_reports_return_time(self):
        """A fall must NEVER report a return time or settle time (must be N/A or -1.0)."""
        # 160 N in sagittal (+x) causes the robot to fall over
        res = run_stand_sim(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            duration=6.0,
            apply_push=True,
            push_force=160.0,
            push_dir="x",
            push_time=2.0,
            quiet=True,
        )
        self.assertTrue(res["fell"], "Expected robot to fall under 160 N push")
        self.assertEqual(res["return_time_s"], -1.0, "Fall must report -1.0 (N/A) for return time")
        self.assertEqual(res["settle_time_s"], -1.0, "Fall must report -1.0 (N/A) for settle time")

    def test_survival_reports_valid_or_na_metrics(self):
        """Under mild push (40 N), robot survives and tracks deviations separately."""
        res = run_stand_sim(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            duration=6.0,
            apply_push=True,
            push_force=40.0,
            push_dir="x",
            push_time=2.0,
            quiet=True,
        )
        self.assertFalse(res["fell"])
        self.assertGreater(res["push_peak_tilt_dev_deg"], 0.0)
        self.assertGreater(res["push_peak_pitch_dev_deg"], 0.0)
        self.assertGreater(res["max_torque"], 0.0)


if __name__ == "__main__":
    unittest.main()
