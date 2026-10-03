"""
tests/test_openhorizon_recovery_integration.py
Comprehensive regression and integration tests verifying OpenHorizon Safe-Fall
and Get-Up policy integration in VASIMOV.

Verifies:
1. Fall detection does NOT kill OpenHorizon recovery policy (legacy FAULT_DAMP suppressed).
2. Safe-Fall state loads and executes safefall.onnx (not getup.onnx).
3. Get-Up state loads and executes getup.onnx.
4. Gyro angular velocity is not double-rotated (local pelvis body frame preserved).
5. 375-D temporal history lifecycle matches OpenHorizon reference contract.
6. Recovery knee effort is not starved/clamped to 25 Nm (matches XML 45 Nm limit).
7. Learned Get-Up NEVER teleports the robot, overwrites qpos, or attaches gantry.
8. Default runtime profile = Official Menlo locomotion + OpenHorizon recovery suite.
9. Official-only mode remains available without auto-recovery.
10. End-to-end autonomous composition state machine transitions correctly:
    LOCOMOTION -> (tilt > 30°) -> SAFEFALL -> (settled) -> GETUP -> (upright) -> LOCOMOTION.
"""

from __future__ import annotations
import math
from pathlib import Path
import unittest
import numpy as np

from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend
from edge.policy_registry import PolicyRegistry
from edge.policy_manager import PolicyManager, CompositionState, ArbitrationMode
from edge.robot_state import RobotState
from edge.policy_adapter import GetupSafefallAdapter


class TestOpenHorizonRecoveryIntegration(unittest.TestCase):
    """Deep integration and regression test suite for OpenHorizon recovery integration."""

    def setUp(self):
        self.core = EdgeCore()
        self.registry = PolicyRegistry()

    def test_01_fall_does_not_kill_openhorizon_recovery(self):
        """Root Cause 1: Fall latch must NOT kill policy control when recovery is enabled."""
        backend = SimBackend(
            self.core,
            locomotion_policy="official_locomotion",
            recovery_policy="openhorizon_recovery",
            auto_compose=True,
        )
        self.assertTrue(backend.policy_manager.auto_compose)
        self.assertTrue(self.core.recovery_mode_active)

        # Start in MOVE / POLICY
        self.core.mode = EdgeMode.MOVE
        self.core.move_submode = MoveSubmode.POLICY

        # Induce severe fall tilt (gz > -0.50, tilt > 60 deg)
        fallen_gravity = [0.0, 0.866, -0.10]  # tilt ~ 84 deg
        self.core.step_state(current_time=0.10, imu_gravity=fallen_gravity)

        # Verify: fault_fall is recorded, but FAULT_DAMP is NOT latched
        self.assertTrue(self.core.fault_fall)
        self.assertFalse(self.core.fault_latched)
        self.assertEqual(self.core.mode, EdgeMode.MOVE)
        self.assertEqual(self.core.move_submode, MoveSubmode.POLICY)

        # Step backend simulation: policy execution must NOT be silenced
        backend.step()
        self.assertIsNotNone(backend.policy_manager.cached_command)
        self.assertGreater(len(self.core.current_policy_targets), 0)

    def test_02_safefall_loads_and_executes_safefall_model(self):
        """Root Cause 2: Safe-Fall state MUST execute safefall.onnx (not getup.onnx)."""
        manager = PolicyManager(registry=self.registry, default_policy_name="official_locomotion")
        manager.configure_multi_policy(
            locomotion="official_locomotion",
            recovery="openhorizon_recovery",
            auto_compose=True,
        )

        self.assertIsNotNone(manager.safefall_policy)
        # safefall.onnx model path check
        model_name = Path(manager.safefall_policy.manifest.runtime.model_path).name
        self.assertIn("safefall", model_name.lower())
        self.assertEqual(manager.safefall_policy.manifest.name, "safefall")

        # Step safefall policy
        state = RobotState.zeros()
        state.base_ang_vel_body = np.array([0.5, 0.1, -0.2], dtype=np.float64)
        cmd = manager.safefall_policy.step(state)
        self.assertIsNotNone(cmd)
        self.assertEqual(len(cmd.joint_targets), 23)

    def test_03_getup_loads_and_executes_getup_model(self):
        """Root Cause 2: Get-Up state MUST execute getup.onnx."""
        manager = PolicyManager(registry=self.registry, default_policy_name="official_locomotion")
        manager.configure_multi_policy(
            locomotion="official_locomotion",
            recovery="openhorizon_recovery",
            auto_compose=True,
        )

        self.assertIsNotNone(manager.getup_policy)
        model_name = Path(manager.getup_policy.manifest.runtime.model_path).name
        self.assertIn("getup", model_name.lower())
        self.assertEqual(manager.getup_policy.manifest.name, "getup")

        # Step getup policy
        state = RobotState.zeros()
        state.pelvis_pos = np.array([0.0, 0.0, 0.16], dtype=np.float64)
        cmd = manager.getup_policy.step(state)
        self.assertIsNotNone(cmd)
        self.assertEqual(len(cmd.joint_targets), 23)

    def test_04_gyro_not_double_rotated(self):
        """Root Cause 4: IMU gyro angular velocity must NOT be double-rotated by R^T."""
        backend = SimBackend(self.core, locomotion_policy="official_locomotion")

        # Supine fallen posture (pitch -90 deg)
        # R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        R_supine = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        backend.data.xmat[backend.pelvis_body_id] = R_supine.flatten()

        # In MuJoCo, qvel[3:6] for freejoint is ALREADY body-frame angular velocity
        w_body_known = np.array([0.50, 0.20, -0.30], dtype=np.float64)
        backend.data.qvel[3:6] = w_body_known

        # Build robot state
        state = backend._build_robot_state()

        # Assert: base_ang_vel_body is directly w_body_known (NOT double rotated)
        np.testing.assert_allclose(state.base_ang_vel_body, w_body_known, rtol=1e-5, atol=1e-5)

        # Assert: base_ang_vel_world is R @ w_body_known
        w_world_expected = R_supine @ w_body_known
        np.testing.assert_allclose(state.base_ang_vel_world, w_world_expected, rtol=1e-5, atol=1e-5)

        # Assert: GetupSafefallAdapter extracts w_body_known directly
        manifest = self.registry.get_manifest("openhorizon_safefall")
        adapter = GetupSafefallAdapter(manifest)
        term_vec = adapter._get_term_vector("base_ang_vel", state)
        np.testing.assert_allclose(term_vec, w_body_known.astype(np.float32), rtol=1e-5, atol=1e-5)

    def test_05_recovery_history_contract(self):
        """Root Cause 7 & 10: 375-D term-major observation history stacking contract."""
        manifest = self.registry.get_manifest("openhorizon_safefall")
        adapter = GetupSafefallAdapter(manifest)
        adapter.reset()

        state = RobotState.zeros()
        obs = adapter.build_observation(state)

        # Assert observation shape and dimensions
        self.assertEqual(obs.shape, (1, 375))
        self.assertEqual(obs.dtype, np.float32)

        # Cold-start fill: all 5 history slots must be populated with initial step
        self.assertIn("base_ang_vel", adapter.buf)
        self.assertEqual(len(adapter.buf["base_ang_vel"]), 5)
        self.assertEqual(len(adapter.buf["projected_gravity"]), 5)
        self.assertEqual(len(adapter.buf["joint_pos"]), 5)
        self.assertEqual(len(adapter.buf["joint_vel"]), 5)
        self.assertEqual(len(adapter.buf["actions"]), 5)

        # Step 2: history must roll FIFO (pop oldest, append newest)
        state.step_count = 1
        state.base_ang_vel_body = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        obs2 = adapter.build_observation(state)
        np.testing.assert_allclose(adapter.buf["base_ang_vel"][-1], np.array([0.25, 0.50, 0.75], dtype=np.float32))
        np.testing.assert_allclose(adapter.buf["base_ang_vel"][0], np.zeros(3, dtype=np.float32))

    def test_06_recovery_knee_effort_limits_not_clamped_to_25nm(self):
        """Root Cause 5: Recovery knee effort limit must match XML 45.0 Nm (not 25.0 Nm)."""
        getup_manifest = self.registry.get_manifest("getup")
        safefall_manifest = self.registry.get_manifest("safefall")
        suite_manifest = self.registry.get_manifest("getup_safefall")

        # Verify manifest pd_gains
        for m in (getup_manifest, safefall_manifest, suite_manifest):
            knee_cfg = m.pd_gains.get("knee", {})
            self.assertEqual(float(knee_cfg.get("effort_limit", 0.0)), 45.0, f"Knee effort in {m.name} must be 45.0 Nm")
            hip_pitch_cfg = m.pd_gains.get("hip_pitch", {})
            self.assertEqual(float(hip_pitch_cfg.get("effort_limit", 0.0)), 45.0)
            hip_roll_cfg = m.pd_gains.get("hip_roll", {})
            self.assertEqual(float(hip_roll_cfg.get("effort_limit", 0.0)), 45.0)

        # Verify adapter joint_gains
        adapter = GetupSafefallAdapter(getup_manifest)
        self.assertEqual(adapter.joint_gains["left_knee_joint"][2], 45.0)
        self.assertEqual(adapter.joint_gains["right_knee_joint"][2], 45.0)

    def test_07_learned_getup_never_teleports(self):
        """Root Cause 6: Learned Get-Up must NEVER teleport the robot or attach gantry."""
        backend = SimBackend(
            self.core,
            locomotion_policy="official_locomotion",
            recovery_policy="openhorizon_recovery",
            auto_compose=True,
        )

        # Place robot in fallen supine posture on the ground
        backend.reset()
        backend.set_gantry(False)
        backend.data.qpos[0] = 1.25
        backend.data.qpos[1] = -0.50
        backend.data.qpos[2] = 0.165  # fallen on floor
        backend.data.qpos[3:7] = [0.7071, 0.7071, 0.0, 0.0]  # pitched

        # Execute learned getup
        res = backend.getup()

        # Assert: coordinates are NOT teleported
        self.assertAlmostEqual(float(backend.data.qpos[0]), 1.25, places=3)
        self.assertAlmostEqual(float(backend.data.qpos[1]), -0.50, places=3)
        self.assertAlmostEqual(float(backend.data.qpos[2]), 0.165, places=3)  # UNCHANGED! Not 0.60962!
        self.assertAlmostEqual(float(backend.data.qpos[3]), 0.7071, places=3)  # Orientation UNCHANGED!

        # Assert: virtual gantry is NOT engaged
        self.assertFalse(backend.gantry_active)
        self.assertFalse(res["gantry"])

        # Assert: OpenHorizon getup.onnx is the active policy in MOVE/POLICY mode
        self.assertEqual(self.core.mode, EdgeMode.MOVE)
        self.assertEqual(self.core.move_submode, MoveSubmode.POLICY)
        self.assertIn("getup", backend.policy_manager.active_policy_name.lower())
        self.assertEqual(backend.policy_manager.comp_state, CompositionState.GETUP)

    def test_08_default_profile_configuration(self):
        """Section 12: Default profile must configure Menlo locomotion + OpenHorizon recovery."""
        prof = self.registry.get_profile("default")
        self.assertEqual(prof["locomotion"], "official_locomotion")
        self.assertEqual(prof["safefall"], "openhorizon_safefall")
        self.assertEqual(prof["getup"], "openhorizon_getup")
        self.assertTrue(prof["auto_compose"])

    def test_09_official_only_profile(self):
        """Section 13: Official-only mode must remain available with legacy fault behavior."""
        prof = self.registry.get_profile("official_locomotion")
        self.assertEqual(prof["locomotion"], "official_locomotion")
        self.assertIsNone(prof.get("safefall"))
        self.assertFalse(prof["auto_compose"])

        # Verify legacy fall fault trip in Menlo-only mode
        core_menlo = EdgeCore()
        core_menlo.fall_latch_enabled = True
        core_menlo.recovery_mode_active = False
        core_menlo.step_state(current_time=0.10, imu_gravity=[0.0, 0.866, -0.10])
        self.assertTrue(core_menlo.fault_latched)
        self.assertEqual(core_menlo.mode, EdgeMode.FAULT_DAMP)

    def test_10_recovery_state_transitions(self):
        """Section 15: State machine arbitration transitions LOCOMOTION -> SAFEFALL -> GETUP -> LOCOMOTION."""
        manager = PolicyManager(registry=self.registry)
        manager.configure_multi_policy(
            locomotion="official_locomotion",
            recovery="openhorizon_recovery",
            auto_compose=True,
        )

        # 1. Starts in LOCOMOTION
        self.assertEqual(manager.comp_state, CompositionState.LOCOMOTION)
        state = RobotState.zeros()
        state.pelvis_pos[2] = 0.639
        state.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        pol = manager._arbitrate_composition(state)
        self.assertEqual(pol.name, "official_locomotion")

        # 2. Disturbance push -> SAFEFALL (tilt > 30 deg)
        state.projected_gravity = np.array([0.60, 0.0, -0.80], dtype=np.float64)  # tilt ~ 36.8 deg
        pol = manager._arbitrate_composition(state)
        self.assertEqual(manager.comp_state, CompositionState.SAFEFALL)
        self.assertIn("safefall", pol.name.lower())

        # 3. Ground settle -> GETUP (z < 0.35m, low velocities for 0.25s)
        state.pelvis_pos[2] = 0.16
        state.base_lin_vel_world[:] = 0.05
        state.base_ang_vel_body[:] = 0.10
        state.dt = 0.02
        for _ in range(15):  # 0.30s
            pol = manager._arbitrate_composition(state)
        self.assertEqual(manager.comp_state, CompositionState.GETUP)
        self.assertIn("getup", pol.name.lower())

        # 4. Standing recovery -> LOCOMOTION (upright gz < -0.85, z > 0.55m for 0.8s)
        state.pelvis_pos[2] = 0.62
        state.projected_gravity = np.array([0.0, 0.0, -0.98], dtype=np.float64)
        for _ in range(45):  # 0.90s
            pol = manager._arbitrate_composition(state)
        self.assertEqual(manager.comp_state, CompositionState.LOCOMOTION)
        self.assertEqual(pol.name, "official_locomotion")


if __name__ == "__main__":
    unittest.main()
