"""
tests/test_custom_environment.py
Unit and integration tests for VASIMOV custom environment loading and follow/FPV cameras.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco

from edge.environment import EnvironmentManager, PRESETS
from tools.sim_console import SimConsole


class TestCustomEnvironment(unittest.TestCase):
    """Test custom environment discovery, YAML/XML parsing, and model compilation."""

    def setUp(self):
        self.em = EnvironmentManager()

    def tearDown(self):
        self.em.cleanup()

    def test_list_presets_includes_builtin_and_custom(self):
        presets = self.em.list_presets()
        # Builtins
        self.assertIn("flat", presets)
        self.assertIn("obstacles", presets)
        self.assertIn("playground", presets)
        # Custom files from environments/
        self.assertIn("corridor", presets)
        self.assertIn("arena", presets)
        self.assertIn("obstacle_course", presets)

    def test_load_corridor_yaml(self):
        model, path = self.em.create_model_for_preset("corridor")
        self.assertIsNotNone(model)
        self.assertEqual(self.em.current_preset, "corridor")
        self.assertAlmostEqual(self.em.active_config["ground_friction"], 1.2)
        # Check that corridor obstacle bodies/geoms exist in compiled model
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "corridor_crate")
        self.assertNotEqual(body_id, -1, "corridor_crate body should be present in compiled MjModel")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_railing")
        self.assertNotEqual(geom_id, -1, "left_railing geom should be present in compiled MjModel")

    def test_load_arena_xml(self):
        model, path = self.em.create_model_for_preset("arena")
        self.assertIsNotNone(model)
        self.assertEqual(self.em.current_preset, "arena")
        # Check boundary wall and dynamic crate exist
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arena_wall_north")
        self.assertNotEqual(wall_id, -1, "arena_wall_north geom should be present in compiled MjModel")
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arena_box_red")
        self.assertNotEqual(box_id, -1, "arena_box_red body should be present in compiled MjModel")

    def test_load_obstacle_course_yaml(self):
        model, path = self.em.create_model_for_preset("obstacle_course")
        self.assertIsNotNone(model)
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "course_ball_1")
        self.assertNotEqual(ball_id, -1)

    def test_load_pair_lab_xml(self):
        model, path = self.em.create_model_for_preset("pair_lab")
        self.assertIsNotNone(model)
        self.assertEqual(self.em.current_preset, "pair_lab")
        # Verify walls, tables, and chairs exist in compiled model
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_north")
        self.assertNotEqual(wall_id, -1, "wall_north should be present")
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_big")
        self.assertNotEqual(table_id, -1, "table_big should be present")
        chair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chair_1")
        self.assertNotEqual(chair_id, -1, "chair_1 body should be present")

    def test_load_arbitrary_temp_file(self):
        # Create a temporary custom YAML environment file
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(
                """name: temp_custom_env
description: "Temporary custom test course"
ground_friction: 1.45
mass_scale: 1.05
obstacles:
  - name: "temp_box"
    type: "box"
    pos: [1.0, 0.5, 0.2]
    size: [0.1, 0.1, 0.1]
    mass: 2.0
    dynamic: true
"""
            )
            temp_path = Path(f.name)

        try:
            model, path = self.em.create_model_for_preset(str(temp_path))
            self.assertIsNotNone(model)
            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "temp_box")
            self.assertNotEqual(box_id, -1)
            self.assertAlmostEqual(self.em.active_config["ground_friction"], 1.45)
        finally:
            if temp_path.exists():
                temp_path.unlink()
            self.em.cleanup()

    def test_unknown_environment_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.em.create_model_for_preset("non_existent_env_xyz_123")


class TestFollowAndFPVCameras(unittest.TestCase):
    """Test follow and FPV camera kinematics and switching."""

    @classmethod
    def setUpClass(cls):
        cls.base_path = Path(__file__).resolve().parent.parent / "model" / "asimov_1_vasimov.xml"
        cls.model = mujoco.MjModel.from_xml_path(str(cls.base_path))
        cls.data = mujoco.MjData(cls.model)

    def test_camera_existence(self):
        cam_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]
        self.assertIn("chase_camera", cam_names)
        self.assertIn("first_person_behind", cam_names)
        self.assertIn("first_person_camera", cam_names)

    def test_chase_camera_moves_with_pelvis(self):
        """Verify chase camera tracks pelvis translation rigidly in world space."""
        mujoco.mj_forward(self.model, self.data)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "chase_camera")
        init_xpos = np.copy(self.data.cam_xpos[cam_id])

        # Translate robot by +2.5m in X and +1.0m in Y
        dx, dy = 2.5, 1.0
        self.data.qpos[0] += dx
        self.data.qpos[1] += dy
        mujoco.mj_forward(self.model, self.data)

        moved_xpos = self.data.cam_xpos[cam_id]
        np.testing.assert_allclose(moved_xpos[0], init_xpos[0] + dx, atol=1e-3)
        np.testing.assert_allclose(moved_xpos[1], init_xpos[1] + dy, atol=1e-3)

    def test_chase_camera_rotates_with_pelvis_yaw(self):
        """Verify chase camera rotates with pelvis heading."""
        # Reset position to origin
        self.data.qpos[:3] = [0.0, 0.0, 0.61]
        # Rotate pelvis 90 degrees CCW (+Z yaw)
        self.data.qpos[3] = np.cos(np.pi / 4)
        self.data.qpos[4:6] = 0.0
        self.data.qpos[6] = np.sin(np.pi / 4)
        mujoco.mj_forward(self.model, self.data)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "chase_camera")
        cam_pos = self.data.cam_xpos[cam_id]
        # Facing +Y, camera 1.5m behind robot should be at -Y (~ -1.5) and X ~ 0
        self.assertAlmostEqual(cam_pos[0], 0.0, places=2)
        self.assertAlmostEqual(cam_pos[1], -1.5, places=2)

    def test_fpv_head_camera_orientation(self):
        """Verify first person camera points along head forward direction."""
        self.data.qpos[:7] = [0, 0, 0.61, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "first_person_camera")
        cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)
        view_dir = -cam_mat[:, 2]
        # Facing straight forward along +X
        self.assertAlmostEqual(view_dir[0], 1.0, places=2)
        self.assertAlmostEqual(view_dir[1], 0.0, places=2)

    def test_sim_console_camera_switching_and_cycling(self):
        """Test SimConsole camera alias resolution and cycling without GUI."""
        console = SimConsole(use_viewer=False, env_preset="flat")
        try:
            # Set camera by aliases
            ok, msg = console.set_camera("chase")
            self.assertTrue(ok)
            self.assertEqual(console.active_camera, "chase_camera")

            ok, msg = console.set_camera("fpv")
            self.assertTrue(ok)
            self.assertEqual(console.active_camera, "first_person_camera")

            ok, msg = console.set_camera("free")
            self.assertTrue(ok)
            self.assertEqual(console.active_camera, "free")

            # Test camera cycling
            c1 = console.cycle_camera()
            self.assertEqual(c1, "chase_camera")
            c2 = console.cycle_camera()
            self.assertEqual(c2, "first_person_behind")
            c3 = console.cycle_camera()
            self.assertEqual(c3, "first_person_camera")
            c4 = console.cycle_camera()
            self.assertEqual(c4, "free")
            c5 = console.cycle_camera()
            self.assertEqual(c5, "chase_camera")
        finally:
            console.stop()


if __name__ == "__main__":
    unittest.main()
