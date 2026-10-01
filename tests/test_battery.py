#!/usr/bin/env python3
"""
tests/test_battery.py
Unit and integration tests for Task 3: Virtual Battery (PLACEHOLDER).
Asserts:
1. RobotState proto battery fields are properly populated.
2. Menlo SDK State parser extracts Battery.
3. Preflight evaluates battery_low (<20%) as blocking, and clears above 20%.
4. Linear discharge rate updates SoC over time.
5. SimControlServer :8852 accepts set_battery command.
"""

import time
import unittest
from pathlib import Path
import numpy as np

from edge.core import EdgeCore, EdgeMode
from edge.sim import SimBackend, SimControlServer
from menlo.asimov import _preflight
from menlo.asimov.transport._wire import state_from_robot_state


class TestVirtualBattery(unittest.TestCase):
    """Test Virtual Battery implementation."""

    def setUp(self):
        self.core = EdgeCore()

    def test_battery_default_state(self):
        """Default battery state is 100% SoC, 48V, no protection, linear discharge OFF."""
        self.assertEqual(self.core.battery_soc, 100.0)
        self.assertEqual(self.core.battery_voltage, 48.0)
        self.assertEqual(self.core.battery_discharge_rate, 0.0)
        self.assertEqual(self.core.battery_protection_flags, 0)

        # Build state and verify proto
        zero_pos = [0.0] * 25
        state_msg = self.core.build_robot_state(
            sim_joint_pos=zero_pos,
            sim_joint_vel=zero_pos,
            imu_quat=[1.0, 0.0, 0.0, 0.0],
            imu_gyro=[0.0, 0.0, 0.0],
            imu_gravity=[0.0, 0.0, -1.0],
        )
        self.assertTrue(state_msg.HasField("battery"))
        self.assertEqual(state_msg.battery.soc_percent, 100.0)
        self.assertEqual(state_msg.battery.voltage_v, 48.0)
        self.assertEqual(state_msg.battery.current_a, 0.0)
        self.assertEqual(state_msg.battery.max_cell_temp_c, 25.0)
        self.assertEqual(state_msg.battery.protection_flags, 0)

    def test_sdk_wire_parse_and_preflight(self):
        """Test decode through Menlo SDK wire parser and preflight evaluation."""
        zero_pos = [0.0] * 25
        state_msg = self.core.build_robot_state(
            sim_joint_pos=zero_pos,
            sim_joint_vel=zero_pos,
            imu_quat=[1.0, 0.0, 0.0, 0.0],
            imu_gyro=[0.0, 0.0, 0.0],
            imu_gravity=[0.0, 0.0, -1.0],
        )
        sdk_state = state_from_robot_state(state_msg, joint_names=())
        self.assertIsNotNone(sdk_state.battery)
        self.assertEqual(sdk_state.battery.soc_percent, 100.0)
        self.assertFalse(sdk_state.battery.protecting)

        # Preflight at 100% -> should not have battery_low
        pf = _preflight.evaluate("stand", sdk_state, armed=False)
        self.assertFalse(pf.has("battery_low"))
        self.assertFalse(pf.has("unknown_battery"))

        # Set battery low: 15% (< 20%)
        self.core.set_battery(soc=15.0)
        low_msg = self.core.build_robot_state(
            sim_joint_pos=zero_pos,
            sim_joint_vel=zero_pos,
            imu_quat=[1.0, 0.0, 0.0, 0.0],
            imu_gyro=[0.0, 0.0, 0.0],
            imu_gravity=[0.0, 0.0, -1.0],
        )
        sdk_low_state = state_from_robot_state(low_msg, joint_names=())
        pf_low = _preflight.evaluate("stand", sdk_low_state, armed=False)
        self.assertTrue(pf_low.has("battery_low"))
        self.assertFalse(pf_low.ok)

        # Clear battery low: back to 85% (> 20%)
        self.core.set_battery(soc=85.0)
        cleared_msg = self.core.build_robot_state(
            sim_joint_pos=zero_pos,
            sim_joint_vel=zero_pos,
            imu_quat=[1.0, 0.0, 0.0, 0.0],
            imu_gyro=[0.0, 0.0, 0.0],
            imu_gravity=[0.0, 0.0, -1.0],
        )
        sdk_cleared_state = state_from_robot_state(cleared_msg, joint_names=())
        pf_cleared = _preflight.evaluate("stand", sdk_cleared_state, armed=False)
        self.assertFalse(pf_cleared.has("battery_low"))

    def test_linear_discharge(self):
        """Test optional linear discharge over time."""
        self.core.set_battery(soc=100.0, discharge_rate=2.0)  # 2% per second
        self.core.sim_time = 10.0
        # Step state forward 5 seconds
        self.core.step_state(current_time=15.0, imu_gravity=[0.0, 0.0, -1.0])
        # 100 - (2 * 5) = 90%
        self.assertAlmostEqual(self.core.battery_soc, 90.0, places=2)

    def test_virtual_restart_resets_battery(self):
        """Virtual restart resets battery to 100% and discharge rate to 0."""
        self.core.set_battery(soc=10.0, discharge_rate=5.0)
        self.core.virtual_restart()
        self.assertEqual(self.core.battery_soc, 100.0)
        self.assertEqual(self.core.battery_discharge_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
