#!/usr/bin/env python3
"""
tests/test_policy_registry.py
Unit tests for the pluggable policy architecture:
- PolicyRegistry discovery and manifest loading
- PolicyManifest schema validation and missing model handling
- Runtime abstraction (ONNXPolicyRuntime)
- Adapters (OfficialLocomotionAdapter 78-D and GetupSafefallAdapter 375-D)
- SafetyLayer limits, NaN protection, and deterministic reset
- PolicyManager arbitration and switching
"""

import math
from pathlib import Path
import unittest
import numpy as np

VASIMOV_DIR = Path(__file__).resolve().parent.parent

from edge.policy_contract import PolicyManifest
from edge.policy_registry import PolicyRegistry
from edge.policy_runtime import ONNXPolicyRuntime
from edge.policy_adapter import OfficialLocomotionAdapter, GetupSafefallAdapter
from edge.policy_base import BasePolicy
from edge.policy_manager import PolicyManager, ArbitrationMode
from edge.robot_command import UnifiedRobotCommand
from edge.robot_state import RobotState
from edge.safety import SafetyLayer


class TestPolicyPluggableArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = PolicyRegistry(root_dir=VASIMOV_DIR)
        cls.registry.discover_policies()

    def test_01_registry_discovery(self):
        """Verify discovery finds both official locomotion and custom recovery policies."""
        policies = self.registry.list_policies()
        names = [p["name"] for p in policies]
        self.assertIn("official_locomotion", names)
        self.assertIn("getup_safefall", names)
        self.assertIn("getup", names)
        self.assertIn("safefall", names)

    def test_02_manifest_validation(self):
        """Verify manifests parse correctly and validate required specifications."""
        loco_manifest = self.registry.get_manifest("official_locomotion")
        self.assertEqual(loco_manifest.observation.dim, 78)
        self.assertEqual(loco_manifest.action.dim, 23)
        self.assertEqual(loco_manifest.control.frequency_hz, 50.0)
        self.assertEqual(len(loco_manifest.action.joint_order), 23)
        self.assertEqual(len(loco_manifest.validate()), 0)

        getup_manifest = self.registry.get_manifest("getup_safefall")
        self.assertEqual(getup_manifest.observation.dim, 375)
        self.assertEqual(getup_manifest.action.dim, 23)
        self.assertEqual(getup_manifest.control.frequency_hz, 50.0)
        self.assertEqual(len(getup_manifest.action.joint_order), 23)
        self.assertEqual(len(getup_manifest.validate()), 0)

    def test_03_deep_validation_checks(self):
        """Verify deep validation confirms model weights, tensor shapes, and datatypes."""
        is_valid, checks = self.registry.validate_policy("official_locomotion")
        self.assertTrue(is_valid, f"Validation failed: {checks}")
        self.assertTrue(any("78" in c for c in checks))

        is_valid_rec, checks_rec = self.registry.validate_policy("getup_safefall")
        self.assertTrue(is_valid_rec, f"Validation failed: {checks_rec}")
        self.assertTrue(any("375" in c for c in checks_rec))

    def test_04_validation_failure_on_invalid_policy(self):
        """Verify validation correctly fails and identifies problems for invalid policies."""
        is_valid, checks = self.registry.validate_policy("nonexistent_policy")
        self.assertFalse(is_valid)
        self.assertTrue(any("✗" in c for c in checks))

    def test_05_runtime_and_adapter_instantiation(self):
        """Instantiate official locomotion and custom getup_safefall policies."""
        loco_policy = self.registry.instantiate("official_locomotion")
        self.assertIsInstance(loco_policy, BasePolicy)
        self.assertEqual(loco_policy.name, "official_locomotion")
        self.assertIsInstance(loco_policy.adapter, OfficialLocomotionAdapter)

        getup_policy = self.registry.instantiate("getup_safefall")
        self.assertIsInstance(getup_policy, BasePolicy)
        self.assertEqual(getup_policy.name, "getup_safefall")
        self.assertIsInstance(getup_policy.adapter, GetupSafefallAdapter)

    def test_06_observation_construction_78d(self):
        """Test building 78-D observation from a clean synthetic RobotState."""
        loco_policy = self.registry.instantiate("official_locomotion")
        state = RobotState(
            sim_time=0.0,
            base_ang_vel_body=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            command_vel=(0.4, 0.0, 0.0),
            joint_positions=dict(loco_policy.default_pos),
            joint_velocities={jn: 0.0 for jn in loco_policy.policy_joints},
        )

        obs = loco_policy.adapter.build_observation(state)
        self.assertEqual(obs.shape, (1, 78))
        self.assertEqual(obs.dtype, np.float32)
        # Check command slice [0, 6:9]
        np.testing.assert_allclose(obs[0, 6:9], [0.4, 0.0, 0.0], atol=1e-5)
        # Check projected gravity [0, 3:6]
        np.testing.assert_allclose(obs[0, 3:6], [0.0, 0.0, -1.0], atol=1e-5)

    def test_07_observation_construction_375d_stacked_history(self):
        """Test building 375-D stacked history observation for in-house recovery policy."""
        rec_policy = self.registry.instantiate("getup_safefall")
        state = RobotState(
            sim_time=0.0,
            base_ang_vel_body=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            joint_positions=dict(rec_policy.default_pos),
            joint_velocities={jn: 0.0 for jn in rec_policy.policy_joints},
        )

        obs = rec_policy.adapter.build_observation(state)
        self.assertEqual(obs.shape, (1, 375))
        self.assertEqual(obs.dtype, np.float32)
        # Verify finite values
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_08_safety_layer_nan_inf_protection(self):
        """Verify SafetyLayer intercepts NaN/Inf policy outputs and forces safe DAMP compliance."""
        safety = SafetyLayer()
        state = RobotState(
            sim_time=0.0,
            joint_positions={"left_hip_pitch_joint": 0.2},
        )
        bad_cmd = UnifiedRobotCommand(
            joint_targets={"left_hip_pitch_joint": float("nan")},
            kp={"left_hip_pitch_joint": 40.0},
            kd={"left_hip_pitch_joint": 2.0},
            policy_name="faulty_policy",
        )
        safe_cmd = safety.filter_command(bad_cmd, state)
        self.assertEqual(safe_cmd.mode, "DAMP")
        self.assertEqual(safe_cmd.kp["left_hip_pitch_joint"], 0.0)
        self.assertEqual(safe_cmd.kd["left_hip_pitch_joint"], 2.0)
        self.assertEqual(safe_cmd.joint_targets["left_hip_pitch_joint"], 0.2)

    def test_09_safety_layer_joint_limits_clamping(self):
        """Verify SafetyLayer strictly clamps joint targets exceeding mechanical boundaries."""
        safety = SafetyLayer(
            joint_ranges={"left_knee_joint": (-0.1, 1.8)},
            max_target_delta_per_step=10.0,
        )
        state = RobotState(sim_time=0.0, joint_positions={"left_knee_joint": 0.5})
        cmd = UnifiedRobotCommand(
            joint_targets={"left_knee_joint": 3.5},  # Way above 1.8 max
            kp={"left_knee_joint": 40.0},
            kd={"left_knee_joint": 2.0},
            policy_name="test_pol",
        )
        safe_cmd = safety.filter_command(cmd, state)
        self.assertAlmostEqual(safe_cmd.joint_targets["left_knee_joint"], 1.8)

    def test_10_policy_manager_switching_and_step(self):
        """Verify PolicyManager can dynamically load, switch, and step policies deterministically."""
        manager = PolicyManager(registry=self.registry, default_policy_name="official_locomotion")
        self.assertEqual(manager.active_policy_name, "official_locomotion")

        # Step locomotion policy
        state = RobotState(
            sim_time=0.02,
            base_ang_vel_body=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            command_vel=(0.2, 0.0, 0.0),
            joint_positions=dict(manager.active_policy.default_pos),
            joint_velocities={jn: 0.0 for jn in manager.active_policy.policy_joints},
        )
        cmd = manager.step(state)
        self.assertIsNotNone(cmd)
        self.assertEqual(len(cmd.joint_targets), 23)

        # Switch to getup_safefall
        manager.set_active_policy("getup_safefall")
        self.assertEqual(manager.active_policy_name, "getup_safefall")
        state.sim_time = 0.04
        cmd_rec = manager.step(state)
        self.assertIsNotNone(cmd_rec)
        self.assertEqual(len(cmd_rec.joint_targets), 23)


if __name__ == "__main__":
    unittest.main()
