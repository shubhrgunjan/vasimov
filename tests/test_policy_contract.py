#!/usr/bin/env python3
"""
tests/test_policy_contract.py
Unit tests verifying the verified policy contract:
1. Observation length and ordering with synthetic upright default state (78-D).
2. ONNX I/O shapes and types: obs [1, 78] -> actions [1, 23].
3. Finite actions and reasonable bounds produced at default standing pose.
"""

from pathlib import Path
import unittest
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import yaml
from tools.policy_standalone import StandalonePolicyRunner, OBS_DIM, ACTION_DIM

VASIMOV_DIR = Path(__file__).resolve().parent.parent
POLICY_PATH = VASIMOV_DIR / "assets" / "policy" / "policy.onnx"
ENV_YAML_PATH = VASIMOV_DIR / "assets" / "policy" / "env.yaml"


class TestPolicyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ort is None:
            raise unittest.SkipTest("onnxruntime is not installed")
        cls.runner = StandalonePolicyRunner(
            policy_path=POLICY_PATH,
            env_yaml_path=ENV_YAML_PATH,
        )

    def test_onnx_io_shapes_and_types(self):
        """Assert ONNX model takes [1, 78] float32 obs and returns [1, 23] float32 actions."""
        session = self.runner.session
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "obs")
        self.assertEqual(inputs[0].shape, [1, 78])
        self.assertEqual(inputs[0].type, "tensor(float)")

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].name, "actions")
        self.assertEqual(outputs[0].shape, [1, 23])
        self.assertEqual(outputs[0].type, "tensor(float)")

    def test_observation_length_and_order_with_synthetic_upright_state(self):
        """Build observation vector with synthetic upright default state and assert 78-D contract."""
        self.runner.reset()

        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        prev_act = np.zeros(ACTION_DIM, dtype=np.float32)

        obs = self.runner.build_observation(command=cmd, prev_action=prev_act)

        # 1. Total length
        self.assertEqual(len(obs), OBS_DIM)
        self.assertEqual(obs.shape, (78,))
        self.assertEqual(obs.dtype, np.float32)

        # 2. Slice terms validation:
        # base_ang_vel: [0:3] should be 0.0 at rest
        np.testing.assert_allclose(obs[0:3], [0.0, 0.0, 0.0], atol=1e-5)

        # projected_gravity: [3:6] for upright robot in world [0, 0, -1] is [0, 0, -1]
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-3)

        # command: [6:9]
        np.testing.assert_allclose(obs[6:9], [0.0, 0.0, 0.0], atol=1e-5)

        # joint_pos_slot01 (9), slot23 (8), slot45 (6): relative to default should be 0.0
        np.testing.assert_allclose(obs[9:18], np.zeros(9), atol=1e-5)
        np.testing.assert_allclose(obs[18:26], np.zeros(8), atol=1e-5)
        np.testing.assert_allclose(obs[26:32], np.zeros(6), atol=1e-5)

        # joint_vel_slot01 (9), slot23 (8), slot45 (6): at rest should be 0.0
        np.testing.assert_allclose(obs[32:41], np.zeros(9), atol=1e-5)
        np.testing.assert_allclose(obs[41:49], np.zeros(8), atol=1e-5)
        np.testing.assert_allclose(obs[49:55], np.zeros(6), atol=1e-5)

        # actions: [55:78]
        np.testing.assert_allclose(obs[55:78], np.zeros(23), atol=1e-5)

    def test_finite_actions_at_default_pose(self):
        """Inference on synthetic upright default state must yield finite actions within [-1, 1]."""
        self.runner.reset()
        obs = self.runner.build_observation(
            command=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            prev_action=np.zeros(23, dtype=np.float32),
        )
        actions = self.runner.session.run(["actions"], {"obs": obs.reshape(1, 78)})[0][0]

        self.assertEqual(len(actions), 23)
        self.assertTrue(np.all(np.isfinite(actions)), "Actions must be all finite")
        self.assertTrue(np.all(actions >= -2.0) and np.all(actions <= 2.0), "Actions must stay within normal range")

    def test_shared_observation_and_action_equivalence(self):
        """Feeding identical recorded states to both standalone and edge must yield identical obs and actions."""
        from edge.policy import PolicyController

        edge_controller = PolicyController(policy_onnx_path=POLICY_PATH, env_yaml_path=ENV_YAML_PATH)
        self.runner.reset()

        # Generate a non-trivial synthetic state:
        # Rotated pelvis (roll 5 deg, pitch -8 deg)
        roll = np.radians(5.0)
        pitch = np.radians(-8.0)
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_test = Ry @ Rx
        omega_world = np.array([0.15, -0.22, 0.35], dtype=np.float64)
        cmd = (0.3, -0.1, 0.4)

        # Set runner state
        self.runner.data.xmat[self.runner.pelvis_id] = R_test.flatten()
        self.runner.data.qvel[3:6] = omega_world

        # Non-zero joint offsets and velocities
        j_pos_dict = {}
        j_vel_dict = {}
        for i, jn in enumerate(self.runner.policy_joints):
            offset = 0.05 * np.sin(i + 1)
            vel = 0.5 * np.cos(i + 1)
            self.runner.data.qpos[self.runner.jnt_qposadr[jn]] = self.runner.default_pos[jn] + offset
            self.runner.data.qvel[self.runner.jnt_dofadr[jn]] = vel
            j_pos_dict[jn] = float(self.runner.data.qpos[self.runner.jnt_qposadr[jn]])
            j_vel_dict[jn] = float(self.runner.data.qvel[self.runner.jnt_dofadr[jn]])

        prev_act = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
        edge_controller.prev_action = np.copy(prev_act)

        # 1. Standalone observation & actions
        obs_standalone = self.runner.build_observation(
            command=np.array(cmd, dtype=np.float32),
            prev_action=prev_act,
        )
        raw_act_standalone = self.runner.session.run(["actions"], {"obs": obs_standalone.reshape(1, OBS_DIM)})[0][0]
        from edge.policy_builder import compute_desired_joint_targets
        targets_standalone = compute_desired_joint_targets(raw_act_standalone, self.runner.policy_joints, self.runner.default_pos)

        # 2. Edge policy observation & actions
        targets_edge = edge_controller.step(
            sim_time=0.020,
            command_vel=cmd,
            pelvis_xmat=R_test,
            pelvis_ang_vel_world=omega_world,
            joint_positions=j_pos_dict,
            joint_velocities=j_vel_dict,
        )
        obs_edge = edge_controller.current_observation
        raw_act_edge = edge_controller.current_actions

        # Assert identical 78-D observations
        np.testing.assert_array_equal(obs_standalone, obs_edge, err_msg="Observations between standalone and edge must be bit-exact identical")

        # Assert identical 23-D actions
        np.testing.assert_array_equal(raw_act_standalone, raw_act_edge, err_msg="Actions between standalone and edge must be bit-exact identical")

        # Assert identical targets
        for jn in self.runner.policy_joints:
            self.assertAlmostEqual(targets_standalone[jn], targets_edge[jn], places=6)


if __name__ == "__main__":
    unittest.main()
