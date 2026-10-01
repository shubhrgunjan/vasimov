#!/usr/bin/env python3
"""Unit tests asserting official protobuf wire enums against single source of truth."""

import unittest
from asimov_protocol.v1 import asimov_common_pb2, edge_cloud_pb2
from edge.core import EdgeMode


class TestWireProtocolEnums(unittest.TestCase):
    """Assert all wire protocol enums match the single source of truth."""

    def test_asimov_common_control_modes(self):
        """asimov.io.ControlMode wire numbers."""
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_DAMP, 0)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_STAND, 1)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_MOVE, 2)
        self.assertEqual(asimov_common_pb2.CONTROL_MODE_FAULT_DAMP, 5)

    def test_edge_mode_enum_parity(self):
        """vasimov.edge.core.EdgeMode matches asimov_common_pb2.ControlMode."""
        self.assertEqual(int(EdgeMode.DAMP), 0)
        self.assertEqual(int(EdgeMode.STAND), 1)
        self.assertEqual(int(EdgeMode.MOVE), 2)
        self.assertEqual(int(EdgeMode.FAULT_DAMP), 5)

    def test_edge_cloud_command_modes(self):
        """menlo.edge.Mode (CloudCommand.mode). Note the inversion trap!"""
        # In CloudCommand: STAND=0, DAMP=1
        self.assertEqual(edge_cloud_pb2.MODE_STAND, 0)
        self.assertEqual(edge_cloud_pb2.MODE_DAMP, 1)

    def test_edge_cloud_telemetry_firmware_modes(self):
        """menlo.edge.FirmwareMode (EdgeTelemetry.fw_mode)."""
        self.assertEqual(edge_cloud_pb2.FW_MODE_DAMP, 0)
        self.assertEqual(edge_cloud_pb2.FW_MODE_STAND, 1)
        self.assertEqual(edge_cloud_pb2.FW_MODE_MOVE, 2)


if __name__ == "__main__":
    unittest.main()
