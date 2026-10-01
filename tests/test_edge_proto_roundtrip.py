#!/usr/bin/env python3
"""Serialize/parse round-trip tests for EdgeTelemetry, EdgeEvent, and overtemp self-clearing."""

import unittest
from asimov_protocol.v1 import asimov_common_pb2, edge_cloud_pb2
from edge.core import EdgeCore, EdgeMode


class TestEdgeProtoRoundtripAndFaults(unittest.TestCase):
    """Test proto serialization round-trips and fault self-clearing behavior."""

    def test_edge_telemetry_roundtrip(self):
        """Build EdgeTelemetry, serialize to bytes, parse back, and verify fields."""
        core = EdgeCore()
        msg = core.build_edge_telemetry(
            sim_joint_pos=[0.1] * 25,
            sim_joint_vel=[0.05] * 25,
            imu_quat=[1.0, 0.0, 0.0, 0.0],
            imu_gyro=[0.01, -0.02, 0.03],
            imu_gravity=[0.0, 0.0, -1.0],
        )
        payload = msg.SerializeToString()
        self.assertGreater(len(payload), 0)

        parsed = edge_cloud_pb2.EdgeTelemetry()
        parsed.ParseFromString(payload)

        self.assertEqual(parsed.fw_mode, edge_cloud_pb2.FW_MODE_DAMP)
        self.assertEqual(len(parsed.joint_pos), 25)
        self.assertAlmostEqual(parsed.joint_pos[0], 0.1, places=3)
        self.assertEqual(len(parsed.imu_quat), 4)
        self.assertEqual(len(parsed.imu_gravity), 3)

    def test_edge_event_and_diagnostics_roundtrip(self):
        """Build diagnostics and controller EdgeEvents, serialize, parse back."""
        core = EdgeCore()
        diag = core.build_diagnostics()
        diag_payload = diag.SerializeToString()

        parsed_diag = edge_cloud_pb2.EdgeDiagnostics()
        parsed_diag.ParseFromString(diag_payload)
        self.assertTrue(parsed_diag.provisioned)
        self.assertTrue(parsed_diag.cloud_connected)
        self.assertEqual(parsed_diag.controller, "sdk")

        # Controller event
        core.set_active_controller("gamepad", reason="manual override")
        self.assertEqual(core.active_controller, "gamepad")
        self.assertGreater(len(core.pending_events), 0)

        evt = core.pending_events[-1]
        evt_payload = evt.SerializeToString()
        parsed_evt = edge_cloud_pb2.EdgeEvent()
        parsed_evt.ParseFromString(evt_payload)
        self.assertEqual(parsed_evt.controller.current, "gamepad")

    def test_overtemp_self_clearing_vs_fall_latched(self):
        """Verify overtemp self-clears below 70 C while fall stays latched until restart."""
        core = EdgeCore()
        # 1. Initially clean DAMP -> stand succeeds
        self.assertTrue(core.command_stand("sdk", current_sim_pos=[0.0]*25))
        self.assertEqual(core.mode, EdgeMode.STAND)

        # 2. Inject overtemp trip at 85 C -> trips FAULT_DAMP
        core.inject_overtemp(85.0)
        self.assertEqual(core.mode, EdgeMode.FAULT_DAMP)
        self.assertNotEqual(core.error_flags, 0)

        # STAND is refused while in overtemp FAULT_DAMP
        self.assertFalse(core.command_stand("sdk", current_sim_pos=[0.0]*25))

        # 3. Cool down below 70 C (e.g. 65 C) -> self-clears back to DAMP!
        core.step_state(current_time=1.0, imu_gravity=[0.0, 0.0, -1.0])
        core.clear_overtemp(65.0)
        self.assertEqual(core.mode, EdgeMode.DAMP)
        self.assertEqual(core.error_flags, 0)
        self.assertFalse(core.fault_latched)

        # STAND succeeds again without restart!
        self.assertTrue(core.command_stand("sdk", current_sim_pos=[0.0]*25))
        self.assertEqual(core.mode, EdgeMode.STAND)

        # 4. Fall detection trips -> latches FAULT_DAMP
        core.inject_fall()
        self.assertEqual(core.mode, EdgeMode.FAULT_DAMP)
        self.assertTrue(core.fault_latched)
        self.assertFalse(core.command_stand("sdk", current_sim_pos=[0.0]*25))

        # Cooling or stepping upright does NOT clear fall latch!
        core.step_state(current_time=2.0, imu_gravity=[0.0, 0.0, -1.0])
        self.assertTrue(core.fault_latched)
        self.assertEqual(core.mode, EdgeMode.FAULT_DAMP)
        self.assertFalse(core.command_stand("sdk", current_sim_pos=[0.0]*25))

        # Only virtual restart clears fall latch!
        core.virtual_restart()
        self.assertFalse(core.fault_latched)
        self.assertEqual(core.mode, EdgeMode.DAMP)
        self.assertTrue(core.command_stand("sdk", current_sim_pos=[0.0]*25))


if __name__ == "__main__":
    unittest.main()
