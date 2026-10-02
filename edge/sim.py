#!/usr/bin/env python3
"""
vasimov/edge/sim.py
MuJoCo Simulation Backend with Virtual Gantry and Motor Model Integration.

Features:
- Steps MuJoCo at the physics rate (200 Hz, dt=0.005 s).
- Pacing: realtime wall-clock pacing (--realtime) or max-speed (--fast) for tests.
- Uses Step-2 MotorModel (vasimov/motor_model.py) with pd_update="per_physics_step".
- Gains per mode:
    * DAMP: kp=0, kd=2.0 (ASSUMED damping).
    * STAND & TRAJECTORY: default gains from vasimov/config/gains.yaml.
- Virtual Gantry:
    * Weld equality constraint on pelvis at settled height (0.60962 m).
    * Active at boot; auto-released 1.0 s after STAND ramp finishes (3.0 s after boot).
    * Re-engageable / releasable via control interface or Python API.
    * In DAMP with gantry OFF: robot collapses realistically under gravity.
- JSON Control Interface:
    * HTTP / JSON server on localhost:8852 (gantry on|off, push, inject, restart, status).
"""

from __future__ import annotations
import http.server
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml
import mujoco

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

import ankle_map
from adapter import JointAdapter, SIM_JOINTS, FIRMWARE_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from motor_model import MotorModel, MotorState

log = logging.getLogger("vasimov.edge.sim")

# Settled base height from Task 0.5 [VERIFIED]
SETTLED_BASE_Z: float = 0.60962
GANTRY_AUTO_RELEASE_DELAY_S: float = 1.0  # 1s settle after 2s STAND ramp


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """
    Project world gravity vector [0, 0, -1] into body frame using base quat [w, x, y, z].
    """
    w, x, y, z = base_quat
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    a = v * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c


def quat_to_euler_deg(quat: Sequence[float]) -> Tuple[float, float, float]:
    """Convert [w, x, y, z] to roll, pitch, yaw in degrees."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class SimBackend:
    """
    Manages the MuJoCo simulation environment, motor dynamics, and virtual gantry.
    """

    def __init__(
        self,
        core: EdgeCore,
        model_path: Optional[Union[str, Path]] = None,
        gains_path: Optional[Union[str, Path]] = None,
        pd_update: str = "per_physics_step",
        auto_gantry: bool = True,
        enable_l1: bool = False,
    ):
        self.core = core
        if model_path is None:
            model_path = _VASIMOV_DIR / "model" / "asimov_1_vasimov.xml"
        if gains_path is None:
            gains_path = _VASIMOV_DIR / "config" / "gains.yaml"

        self.model_path = Path(model_path)
        self.gains_path = Path(gains_path)
        self.pd_update = pd_update
        self.auto_gantry = auto_gantry
        self.ground_truth_server = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # 0.005 s (200 Hz)

        # Load joints config for telemetry
        with open(_VASIMOV_DIR / "config" / "joints.yaml", "r", encoding="utf-8") as f:
            self.joint_configs = yaml.safe_load(f)["joints"]

        # Foot contact sensor addresses (from derived model)
        def get_sensor(name: str) -> Tuple[int, int]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])
            return -1, 0

        self.sensor_left_touch = get_sensor("left_foot_touch")
        self.sensor_right_touch = get_sensor("right_foot_touch")
        self.sensor_left_force = get_sensor("left_foot_force")
        self.sensor_right_force = get_sensor("right_foot_force")

        # Ground truth streaming & RTF tracking
        self.step_count: int = 0
        self.last_rtf_wall_time: float = time.perf_counter()
        self.last_rtf_sim_time: float = 0.0
        self.current_rtf: float = 1.0
        self.last_ground_truth_frame: Optional[Dict[str, Any]] = None

        # Initialize MotorModel
        self.motor_model = MotorModel(
            motors_yaml_path=_VASIMOV_DIR / "config" / "motors.yaml",
            joints_yaml_path=_VASIMOV_DIR / "config" / "joints.yaml",
            gains_yaml_path=self.gains_path,
        )
        self.motor_model.set_layer_l1(enable_l1)

        # Find pelvis body id and virtual gantry equality constraint id
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        if self.pelvis_body_id == -1:
            self.pelvis_body_id = 1

        self.gantry_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "virtual_gantry")

        # Map actuators and joint names to qpos/qvel indices
        self.actuator_qposadr = []
        self.actuator_dofadr = []
        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self.actuator_qposadr.append(self.model.jnt_qposadr[jnt_id])
            self.actuator_dofadr.append(self.model.jnt_dofadr[jnt_id])

        self.jnt_name_to_qposadr = {}
        self.jnt_name_to_dofadr = {}
        for i in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.jnt_name_to_qposadr[jname] = self.model.jnt_qposadr[i]
            self.jnt_name_to_dofadr[jname] = self.model.jnt_dofadr[i]

        # Policy controller (Menlo/asimov1-locomotion-0818)
        try:
            from edge.policy import PolicyController
            self.policy_controller: Optional[PolicyController] = PolicyController()
        except Exception as e:
            log.warning("[SIM] PolicyController could not be initialized: %s", e)
            self.policy_controller = None

        # Push disturbance tracking
        self.push_remaining_steps: int = 0
        self.push_force_vec = np.zeros(6, dtype=np.float64)

        # Gantry state
        self.gantry_active: bool = True
        self.gantry_released_by_auto: bool = False
        self._lock = threading.RLock()

        # Step 6: Mode, pause, manual joint tracking, and environment
        self.control_mode: str = "policy"  # "policy", "manual", "hybrid"
        self.paused: bool = False
        self.single_step_requested: int = 0
        self.selected_joint: str = "left_hip_pitch_joint"
        self.manual_joint_targets: Dict[str, float] = {
            jn: float(self.core.default_pose_sim[i]) for i, jn in enumerate(SIM_JOINTS)
        }
        self.manual_joint_overrides: Dict[str, float] = {}
        self.last_applied_targets: List[float] = list(self.core.default_pose_sim)
        self.last_applied_torques: List[float] = [0.0] * self.model.nu
        self.last_applied_kp: List[float] = list(self.core.default_kp)
        self.last_applied_kd: List[float] = list(self.core.default_kd)

        from edge.environment import EnvironmentManager
        self.env_manager = EnvironmentManager(self.model_path)

        # Reset robot to settled pose with gantry active
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulation state and re-engage gantry."""
        with self._lock:
            if seed is not None:
                np.random.seed(seed)
            mujoco.mj_resetData(self.model, self.data)

            # Set floating base at settled height
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 0.0
            self.data.qpos[2] = SETTLED_BASE_Z
            self.data.qpos[3] = 1.0  # quat w
            self.data.qpos[4:7] = 0.0
            self.data.qvel[:] = 0.0

            # Initialize joint angles to default standing pose
            for i in range(self.model.nu):
                qpos_adr = self.actuator_qposadr[i]
                self.data.qpos[qpos_adr] = self.core.default_pose_sim[i]

            self.data.ctrl[:] = 0.0
            self.data.xfrc_applied[:] = 0.0

            # Re-engage gantry
            self.gantry_active = True
            if self.gantry_eq_id != -1:
                self.data.eq_active[self.gantry_eq_id] = 1
                self.model.eq_data[self.gantry_eq_id, 3:6] = [0.0, 0.0, -SETTLED_BASE_Z]
            self.gantry_released_by_auto = False

            if hasattr(self, "policy_controller") and self.policy_controller is not None:
                self.policy_controller.reset()

            # Reset interactive manual/hybrid controls & pause state
            self.manual_joint_overrides.clear()
            self.manual_joint_targets = {
                jn: float(self.core.default_pose_sim[i]) for i, jn in enumerate(SIM_JOINTS)
            }
            self.last_applied_targets = list(self.core.default_pose_sim)
            self.last_applied_torques = [0.0] * self.model.nu
            self.control_mode = "policy"
            self.paused = False
            self.single_step_requested = 0

            # Reset EdgeCore command and safety state
            self.core.virtual_restart()
            self.core.mode = EdgeMode.DAMP
            self.core.move_submode = MoveSubmode.NONE
            self.core.current_vx = 0.0
            self.core.current_vy = 0.0
            self.core.current_vyaw = 0.0
            self.core.stand_settled = False
            self.core.stand_ramp_start_time = 0.0

            mujoco.mj_forward(self.model, self.data)

    def set_gantry(self, active: bool) -> None:
        """Toggle virtual gantry weld equality constraint."""
        with self._lock:
            self.gantry_active = bool(active)
            if self.gantry_eq_id != -1:
                self.data.eq_active[self.gantry_eq_id] = 1 if self.gantry_active else 0
                if self.gantry_active:
                    # Re-anchor weld to current pelvis position so engaging gantry does not snap back to origin
                    self.model.eq_data[self.gantry_eq_id, 3:6] = -self.data.xpos[self.pelvis_body_id]
                    self.data.qvel[0:6] = 0.0
            if self.gantry_active:
                self.gantry_released_by_auto = True
            log.info("[SIM] Virtual Gantry %s", "ENGAGED" if self.gantry_active else "RELEASED (free motion)")

    def apply_push(self, force_n: float, direction: str = "x", duration_s: float = 0.1) -> None:
        """Apply a disturbance force to the pelvis."""
        with self._lock:
            steps = max(1, int(round(duration_s / self.dt)))
            self.push_remaining_steps = steps
            self.push_force_vec[:] = 0.0
            idx = 0 if direction.lower() == "x" else (1 if direction.lower() == "y" else 2)
            self.push_force_vec[idx] = float(force_n)
            log.warning("[SIM] Applying push: %.1f N in '%s' for %.2fs (%d steps)", force_n, direction, duration_s, steps)

    def step(self, realtime: bool = False) -> None:
        """Step physics by one integration step (200 Hz, 0.005 s)."""
        should_broadcast = False
        frame = None
        with self._lock:
            # Handle pause & single-stepping
            if self.paused and self.single_step_requested <= 0:
                return
            if self.single_step_requested > 0:
                self.single_step_requested -= 1

            sim_time = self.data.time

            # Apply push disturbance if active
            if self.push_remaining_steps > 0:
                self.data.xfrc_applied[self.pelvis_body_id, :] = self.push_force_vec
                self.push_remaining_steps -= 1
            else:
                self.data.xfrc_applied[self.pelvis_body_id, :] = 0.0

            # Reset auto-release flag while ramping into STAND
            if self.core.mode == EdgeMode.STAND and not self.core.stand_settled:
                self.gantry_released_by_auto = False

            # Auto-release gantry: 1.0s after STAND ramp finishes or on MOVE/POLICY
            if self.auto_gantry and self.gantry_active:
                if self.core.mode == EdgeMode.STAND and self.core.stand_settled and not self.gantry_released_by_auto:
                    ramp_finished_time = self.core.stand_ramp_start_time + 2.0
                    if sim_time >= ramp_finished_time + GANTRY_AUTO_RELEASE_DELAY_S:
                        log.info("[SIM] STAND ramp + 1.0s settle elapsed. Auto-releasing virtual gantry.")
                        self.gantry_active = False
                        if self.gantry_eq_id != -1:
                            self.data.eq_active[self.gantry_eq_id] = 0
                        self.gantry_released_by_auto = True
                elif self.core.mode == EdgeMode.MOVE and self.core.move_submode == MoveSubmode.POLICY and not self.gantry_released_by_auto:
                    log.info("[SIM] In MOVE/POLICY. Auto-releasing virtual gantry for locomotion.")
                    self.gantry_active = False
                    if self.gantry_eq_id != -1:
                        self.data.eq_active[self.gantry_eq_id] = 0
                    self.gantry_released_by_auto = True

            # Read current sensor state
            current_pos = [float(self.data.qpos[adr]) for adr in self.actuator_qposadr]
            current_vel = [float(self.data.qvel[adr]) for adr in self.actuator_dofadr]
            base_quat = np.array([self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]], dtype=np.float64)
            base_gyro = [float(self.data.qvel[3]), float(self.data.qvel[4]), float(self.data.qvel[5])]
            projected_gravity = compute_projected_gravity(base_quat)

            # Step EdgeCore timers, watchdogs & safety checks
            self.core.step_state(sim_time, projected_gravity)

            # Step policy controller if in MOVE/POLICY
            if (
                self.core.mode == EdgeMode.MOVE
                and self.core.move_submode == MoveSubmode.POLICY
                and self.policy_controller is not None
            ):
                j_pos = {
                    jn: float(self.data.qpos[self.jnt_name_to_qposadr[jn]])
                    for jn in self.policy_controller.policy_joints
                    if jn in self.jnt_name_to_qposadr
                }
                j_vel = {
                    jn: float(self.data.qvel[self.jnt_name_to_dofadr[jn]])
                    for jn in self.policy_controller.policy_joints
                    if jn in self.jnt_name_to_dofadr
                }
                pelvis_xmat = self.data.xmat[self.pelvis_body_id]
                pelvis_ang_vel = np.array(self.data.qvel[3:6], dtype=np.float64)
                cmd_vel = (self.core.current_vx, self.core.current_vy, self.core.current_vyaw)

                targets_dict = self.policy_controller.step(
                    sim_time=sim_time,
                    command_vel=cmd_vel,
                    pelvis_xmat=pelvis_xmat,
                    pelvis_ang_vel_world=pelvis_ang_vel,
                    joint_positions=j_pos,
                    joint_velocities=j_vel,
                )

                tgt_list = []
                kp_list = []
                kd_list = []
                for i, jn in enumerate(SIM_JOINTS):
                    if jn in targets_dict:
                        tgt_list.append(targets_dict[jn])
                        kp, kd, _ = self.policy_controller.joint_gains[jn]
                        kp_list.append(kp)
                        kd_list.append(kd)
                    else:
                        tgt_list.append(self.core.default_pose_sim[i])
                        kp_list.append(40.0)
                        kd_list.append(2.0)

                self.core.current_policy_targets = tuple(tgt_list)
                self.core.current_policy_kp = tuple(kp_list)
                self.core.current_policy_kd = tuple(kd_list)

            # Get target positions and gains depending on control mode
            if self.control_mode == "manual":
                targets = tuple(
                    self.manual_joint_targets.get(SIM_JOINTS[i], self.core.default_pose_sim[i])
                    for i in range(len(SIM_JOINTS))
                )
                effective_kp = list(self.core.default_kp)
                effective_kd = list(self.core.default_kd)
            elif self.control_mode == "hybrid":
                base_targets, kp_vals, kd_vals = self.core.get_control_targets(current_pos, sim_time)
                hybrid_targets = list(base_targets)
                for jn, override_val in self.manual_joint_overrides.items():
                    if jn in SIM_JOINTS:
                        idx = SIM_JOINTS.index(jn)
                        hybrid_targets[idx] = float(override_val)
                targets = tuple(hybrid_targets)
                if self.core.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
                    effective_kp = [0.0] * 25
                    effective_kd = [2.0] * 25
                else:
                    effective_kp = kp_vals
                    effective_kd = kd_vals
            else:
                targets, kp_vals, kd_vals = self.core.get_control_targets(current_pos, sim_time)
                if self.core.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
                    effective_kp = [0.0] * 25
                    effective_kd = [2.0] * 25
                else:
                    effective_kp = kp_vals
                    effective_kd = kd_vals

            self.last_applied_targets = list(targets)
            self.last_applied_kp = list(effective_kp)
            self.last_applied_kd = list(effective_kd)

            # Compute torques via MotorModel
            q_arr = np.array(current_pos, dtype=np.float64)
            qdot_arr = np.array(current_vel, dtype=np.float64)
            q_des_arr = np.array(targets, dtype=np.float64)
            qdot_des_arr = np.zeros(self.model.nu, dtype=np.float64)

            applied_torques, _ = self.motor_model.step(
                q=q_arr,
                qdot=qdot_arr,
                q_des=q_des_arr,
                qdot_des=qdot_des_arr,
                dt=self.dt,
                kp_override=effective_kp,
                kd_override=effective_kd,
            )
            self.data.ctrl[:] = applied_torques
            self.last_applied_torques = list(applied_torques)

            # Step physics
            mujoco.mj_step(self.model, self.data)

            # Step counter and RTF tracking
            self.step_count += 1
            if self.step_count % 50 == 0:
                now_wall = time.perf_counter()
                sim_elapsed = self.data.time - self.last_rtf_sim_time
                wall_elapsed = now_wall - self.last_rtf_wall_time
                if wall_elapsed > 0:
                    self.current_rtf = max(0.01, sim_elapsed / wall_elapsed)
                self.last_rtf_wall_time = now_wall
                self.last_rtf_sim_time = self.data.time

            # Ground-truth stream at ~50 Hz (every 4 steps of 200 Hz = 50 Hz)
            if self.step_count % 4 == 0 and self.ground_truth_server is not None:
                should_broadcast = True
                frame = self.get_ground_truth_frame()

        if should_broadcast and frame is not None:
            self.ground_truth_server.broadcast(frame)

    def get_ground_truth_frame(self, rtf: Optional[float] = None) -> Dict[str, Any]:
        """Construct the complete ~50 Hz sim-only ground-truth payload."""
        sim_time = float(self.data.time)
        effective_rtf = float(rtf) if rtf is not None else self.current_rtf

        # Pelvis 6-DoF pose, velocity & tilt
        pelvis_pos = [float(self.data.qpos[i]) for i in range(3)]
        pelvis_quat = [float(self.data.qpos[i]) for i in range(3, 7)]
        pelvis_lin_vel = [float(self.data.qvel[i]) for i in range(3)]
        pelvis_ang_vel = [float(self.data.qvel[i]) for i in range(3, 6)]
        R = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        body_lin_vel = [float(x) for x in (R.T @ np.array(pelvis_lin_vel))]
        proj_grav = compute_projected_gravity(np.array(pelvis_quat))
        norm_g = math.sqrt(sum(g * g for g in proj_grav)) or 1.0
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -proj_grav[2] / norm_g))))

        # Foot contacts
        def read_sensor(adr_dim: Tuple[int, int]) -> List[float]:
            adr, dim = adr_dim
            if adr != -1:
                return [float(x) for x in self.data.sensordata[adr:adr + dim]]
            return [0.0] * dim

        left_touch = read_sensor(self.sensor_left_touch)
        right_touch = read_sensor(self.sensor_right_touch)
        left_force = read_sensor(self.sensor_left_force)
        right_force = read_sensor(self.sensor_right_force)

        f_l_z = abs(left_force[2]) if len(left_force) >= 3 else 0.0
        f_r_z = abs(right_force[2]) if len(right_force) >= 3 else 0.0

        # Ankles: Joint Pitch/Roll vs Pushrod Motors A/B
        q_pitch_l = float(self.data.qpos[self.actuator_qposadr[4]])
        q_roll_l = float(self.data.qpos[self.actuator_qposadr[5]])
        tau_pitch_l = float(self.data.ctrl[4])
        tau_roll_l = float(self.data.ctrl[5])
        m_a_l, m_b_l = ankle_map.pr_to_ab(q_pitch_l, q_roll_l)
        tau_a_l, tau_b_l = ankle_map.torque_pr_to_ab(tau_pitch_l, tau_roll_l)

        q_pitch_r = float(self.data.qpos[self.actuator_qposadr[10]])
        q_roll_r = float(self.data.qpos[self.actuator_qposadr[11]])
        tau_pitch_r = float(self.data.ctrl[10])
        tau_roll_r = float(self.data.ctrl[11])
        m_a_r, m_b_r = ankle_map.pr_to_ab(q_pitch_r, q_roll_r)
        tau_a_r, tau_b_r = ankle_map.torque_pr_to_ab(tau_pitch_r, tau_roll_r)

        # Targets and active gains
        current_sim_pos = [float(self.data.qpos[adr]) for adr in self.actuator_qposadr]
        targets, kp_vals, kd_vals = self.core.get_control_targets(current_sim_pos, sim_time)

        # 25 Joints in firmware order (0..24)
        joints_payload = []
        for j_cfg in self.joint_configs:
            idx = j_cfg["index"]
            fw_name = j_cfg["firmware_name"]
            sim_name = j_cfg["sim_joint"]
            pos_min, pos_max = j_cfg["sim_range"]
            effort_limit = float(j_cfg["effort_limit"])

            sim_idx = SIM_JOINTS.index(sim_name)
            pos = float(self.data.qpos[self.actuator_qposadr[sim_idx]])
            vel = float(self.data.qvel[self.actuator_dofadr[sim_idx]])
            tgt = float(targets[sim_idx])
            err = tgt - pos
            tau = float(self.data.ctrl[sim_idx])
            pct_eff = min(100.0, (abs(tau) / effort_limit) * 100.0) if effort_limit > 0 else 0.0
            rng = pos_max - pos_min
            pct_rng = min(100.0, max(0.0, ((pos - pos_min) / rng) * 100.0)) if rng > 0 else 50.0

            joints_payload.append({
                "index": idx,
                "firmware_name": fw_name,
                "sim_name": sim_name,
                "pos": pos,
                "vel": vel,
                "target": tgt,
                "error": err,
                "torque": tau,
                "effort_limit": effort_limit,
                "torque_pct_effort": pct_eff,
                "range": [pos_min, pos_max],
                "pos_pct_range": pct_rng,
                "kp": float(kp_vals[sim_idx]),
                "kd": float(kd_vals[sim_idx]),
            })

        frame = {
            "stream_metadata": {
                "sim_only": True,
                "hardware_equivalent": False,
                "description": "Virtual Asimov sim-only ground-truth stream. Has no hardware equivalent.",
            },
            "sim_time": sim_time,
            "real_time_factor": effective_rtf,
            "mode": self.core.mode.name,
            "gantry_active": self.gantry_active,
            "fault_state": {
                "latched": self.core.fault_latched,
                "fall": self.core.fault_fall,
                "overtemp": self.core.fault_overtemp,
                "can": self.core.fault_can,
                "error_flags": self.core.error_flags,
            },
            "battery": {
                "soc_percent": self.core.battery_soc,
                "voltage_v": self.core.battery_voltage,
                "discharge_rate_pct_s": self.core.battery_discharge_rate,
            },
            "pelvis": {
                "pos": pelvis_pos,
                "quat": pelvis_quat,
                "tilt_deg": tilt_deg,
                "lin_vel": pelvis_lin_vel,
                "ang_vel": pelvis_ang_vel,
                "body_lin_vel": body_lin_vel,
            },
            "contacts": {
                "left_foot": {
                    "contact": bool((left_touch[0] > 0.5 if left_touch else False) or f_l_z > 5.0),
                    "touch": left_touch[0] if left_touch else 0.0,
                    "force": left_force,
                    "normal_force": f_l_z,
                },
                "right_foot": {
                    "contact": bool((right_touch[0] > 0.5 if right_touch else False) or f_r_z > 5.0),
                    "touch": right_touch[0] if right_touch else 0.0,
                    "force": right_force,
                    "normal_force": f_r_z,
                },
            },
            "ankles": {
                "left": {
                    "pitch_rad": q_pitch_l,
                    "roll_rad": q_roll_l,
                    "motor_a_rad": m_a_l,
                    "motor_b_rad": m_b_l,
                    "motor_a_torque_nm": tau_a_l,
                    "motor_b_torque_nm": tau_b_l,
                },
                "right": {
                    "pitch_rad": q_pitch_r,
                    "roll_rad": q_roll_r,
                    "motor_a_rad": m_a_r,
                    "motor_b_rad": m_b_r,
                    "motor_a_torque_nm": tau_a_r,
                    "motor_b_torque_nm": tau_b_r,
                },
            },
            "joints": joints_payload,
            "policy": {
                "active": (self.core.mode == EdgeMode.MOVE and self.core.move_submode == MoveSubmode.POLICY),
                "command_vel": [self.core.current_vx, self.core.current_vy, self.core.current_vyaw],
                "observation": (
                    [float(x) for x in self.policy_controller.current_observation]
                    if (self.policy_controller is not None and self.policy_controller.current_observation is not None)
                    else []
                ),
                "actions": (
                    [float(x) for x in self.policy_controller.current_actions]
                    if (self.policy_controller is not None and self.policy_controller.current_actions is not None)
                    else []
                ),
            },
        }
        self.last_ground_truth_frame = frame
        return frame

    def step_once(self, control_steps: int = 1) -> None:
        """Request stepping by N control steps (N * 4 physics steps)."""
        with self._lock:
            self.single_step_requested += control_steps * 4

    def step_control_tick(self) -> None:
        """Advance physics synchronously by exactly 1 control tick (4 physics steps = 0.020s)."""
        with self._lock:
            old_paused = self.paused
            self.paused = False
            for _ in range(4):
                self.step()
            self.paused = old_paused

    def set_paused(self, paused: bool) -> None:
        """Pause or resume simulation stepping."""
        with self._lock:
            self.paused = bool(paused)
            log.info("[SIM] Simulation %s", "PAUSED" if self.paused else "RESUMED")

    def set_control_mode(self, mode: str) -> None:
        """Switch control mode between policy, manual, and hybrid."""
        valid_modes = ("policy", "manual", "hybrid")
        m = mode.lower().strip()
        if m not in valid_modes:
            raise ValueError(f"Invalid control mode '{mode}'. Choose from: {valid_modes}")
        with self._lock:
            self.control_mode = m
            if m == "manual":
                for i, jn in enumerate(SIM_JOINTS):
                    qpos_adr = self.actuator_qposadr[i]
                    self.manual_joint_targets[jn] = float(self.data.qpos[qpos_adr])
            log.info("[SIM] Control mode set to: %s", self.control_mode.upper())

    def select_joint(self, joint_name_or_index: Union[str, int]) -> str:
        """Select a joint for manual manipulation by name or 1-based/0-based index."""
        with self._lock:
            if isinstance(joint_name_or_index, int):
                idx = joint_name_or_index
                if 1 <= idx <= len(SIM_JOINTS):
                    idx = idx - 1
                if 0 <= idx < len(SIM_JOINTS):
                    self.selected_joint = SIM_JOINTS[idx]
                else:
                    raise IndexError(f"Joint index {joint_name_or_index} out of range [1, {len(SIM_JOINTS)}]")
            else:
                jname = str(joint_name_or_index).strip()
                if jname in SIM_JOINTS:
                    self.selected_joint = jname
                else:
                    matches = [jn for jn in SIM_JOINTS if jname.lower() in jn.lower()]
                    if matches:
                        self.selected_joint = matches[0]
                    else:
                        raise KeyError(f"Unknown joint '{jname}'. Available: {SIM_JOINTS}")
            return self.selected_joint

    def set_manual_joint_target(self, joint_name: str, target: float) -> None:
        """Set manual target for a joint (clamped to joint range)."""
        with self._lock:
            if joint_name not in SIM_JOINTS:
                raise KeyError(f"Unknown joint '{joint_name}'")
            idx = SIM_JOINTS.index(joint_name)
            jnt_id = self.model.actuator_trnid[idx, 0]
            j_min = float(self.model.jnt_range[jnt_id, 0])
            j_max = float(self.model.jnt_range[jnt_id, 1])
            clamped = float(np.clip(target, j_min, j_max))
            self.manual_joint_targets[joint_name] = clamped
            self.manual_joint_overrides[joint_name] = clamped

    def add_manual_joint_delta(self, joint_name: str, delta: float) -> float:
        """Increment manual joint target by delta (clamped to joint range)."""
        with self._lock:
            current = self.manual_joint_targets.get(joint_name, 0.0)
            new_target = current + delta
            self.set_manual_joint_target(joint_name, new_target)
            return self.manual_joint_targets[joint_name]

    def clear_manual_overrides(self) -> None:
        """Clear all manual overrides in hybrid mode."""
        with self._lock:
            self.manual_joint_overrides.clear()

    def load_environment(self, preset_name: str) -> Dict[str, Any]:
        """Load an environment preset and re-initialize model/data."""
        with self._lock:
            new_model, gen_path = self.env_manager.create_model_for_preset(preset_name)
            self.model = new_model
            self.data = mujoco.MjData(self.model)

            self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
            if self.pelvis_body_id == -1:
                self.pelvis_body_id = 1
            self.gantry_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "virtual_gantry")

            def get_sensor(name: str) -> Tuple[int, int]:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid != -1:
                    return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])
                return -1, 0

            self.sensor_left_touch = get_sensor("left_foot_touch")
            self.sensor_right_touch = get_sensor("right_foot_touch")
            self.sensor_left_force = get_sensor("left_foot_force")
            self.sensor_right_force = get_sensor("right_foot_force")

            self.actuator_qposadr = []
            self.actuator_dofadr = []
            for i in range(self.model.nu):
                jnt_id = self.model.actuator_trnid[i, 0]
                self.actuator_qposadr.append(self.model.jnt_qposadr[jnt_id])
                self.actuator_dofadr.append(self.model.jnt_dofadr[jnt_id])

            self.jnt_name_to_qposadr = {}
            self.jnt_name_to_dofadr = {}
            for i in range(self.model.njnt):
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                self.jnt_name_to_qposadr[jname] = self.model.jnt_qposadr[i]
                self.jnt_name_to_dofadr[jname] = self.model.jnt_dofadr[i]

            self.reset()
            return self.get_environment_state()

    def get_base_state(self) -> Dict[str, Any]:
        """Return floating base position, quaternion, euler angles, and velocities."""
        with self._lock:
            pos = [float(self.data.qpos[i]) for i in range(3)]
            quat = [float(self.data.qpos[i]) for i in range(3, 7)]
            r_deg, p_deg, y_deg = quat_to_euler_deg(quat)
            lin_vel_world = [float(self.data.qvel[i]) for i in range(3)]
            ang_vel_world = [float(self.data.qvel[i]) for i in range(3, 6)]
            R = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
            lin_vel_body = [float(x) for x in (R.T @ np.array(lin_vel_world))]
            ang_vel_body = [float(x) for x in (R.T @ np.array(ang_vel_world))]
            return {
                "x": pos[0], "y": pos[1], "z": pos[2],
                "pos": pos,
                "quat": quat,
                "roll_deg": r_deg, "pitch_deg": p_deg, "yaw_deg": y_deg,
                "rpy_deg": [r_deg, p_deg, y_deg],
                "lin_vel_world": lin_vel_world,
                "ang_vel_world": ang_vel_world,
                "lin_vel_body": lin_vel_body,
                "ang_vel_body": ang_vel_body,
                "body_vx": lin_vel_body[0], "body_vy": lin_vel_body[1], "body_vz": lin_vel_body[2],
                "body_wx": ang_vel_body[0], "body_wy": ang_vel_body[1], "body_wz": ang_vel_body[2],
            }

    def get_imu(self) -> Dict[str, Any]:
        """Return simulated IMU readings (angular velocity, projected gravity, orientation)."""
        with self._lock:
            quat = [float(self.data.qpos[i]) for i in range(3, 7)]
            ang_vel_world = [float(self.data.qvel[i]) for i in range(3, 6)]
            R = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
            ang_vel_body = [float(x) for x in (R.T @ np.array(ang_vel_world))]
            proj_grav = compute_projected_gravity(np.array(quat))
            r_deg, p_deg, y_deg = quat_to_euler_deg(quat)
            qacc_base = [float(self.data.qacc[i]) for i in range(3)]
            lin_acc_body = [float(x) for x in (R.T @ np.array(qacc_base))]
            return {
                "simulated": True,
                "description": "Simulated base IMU. Has no direct physical sensor counterpart.",
                "angular_velocity_body": ang_vel_body,
                "projected_gravity": [float(x) for x in proj_grav],
                "orientation_quat": quat,
                "orientation_rpy_deg": [r_deg, p_deg, y_deg],
                "linear_acceleration_body": lin_acc_body,
            }

    def get_joint_state(self) -> List[Dict[str, Any]]:
        """Return complete state for all joints."""
        with self._lock:
            joints_list = []
            for i, jn in enumerate(SIM_JOINTS):
                qpos_adr = self.actuator_qposadr[i]
                dof_adr = self.actuator_dofadr[i]
                jnt_id = self.model.actuator_trnid[i, 0]
                q_val = float(self.data.qpos[qpos_adr])
                qdot_val = float(self.data.qvel[dof_adr])
                tgt = float(self.last_applied_targets[i]) if i < len(self.last_applied_targets) else q_val
                tau = float(self.last_applied_torques[i]) if i < len(self.last_applied_torques) else 0.0
                j_min = float(self.model.jnt_range[jnt_id, 0])
                j_max = float(self.model.jnt_range[jnt_id, 1])
                tau_limit = float(self.model.actuator_forcerange[i, 1])

                status = "OK"
                if q_val < j_min - 0.01 or q_val > j_max + 0.01:
                    status = "VIOLATION_POS"
                elif abs(tau) >= tau_limit - 0.5:
                    status = "SATURATED"

                is_overridden = (self.control_mode == "hybrid" and jn in self.manual_joint_overrides)
                is_selected = (jn == self.selected_joint)

                joints_list.append({
                    "index": i + 1,
                    "name": jn,
                    "pos": q_val,
                    "vel": qdot_val,
                    "target": tgt,
                    "error": tgt - q_val,
                    "torque": tau,
                    "torque_limit": tau_limit,
                    "limit_min": j_min,
                    "limit_max": j_max,
                    "limit_status": status,
                    "kp": float(self.last_applied_kp[i]) if i < len(self.last_applied_kp) else 0.0,
                    "kd": float(self.last_applied_kd[i]) if i < len(self.last_applied_kd) else 0.0,
                    "is_overridden": is_overridden,
                    "is_selected": is_selected,
                })
            return joints_list

    def get_contacts(self) -> Dict[str, Any]:
        """Return foot contacts, forces, and collisions from MuJoCo physics."""
        with self._lock:
            def read_sensor(adr_dim: Tuple[int, int]) -> List[float]:
                adr, dim = adr_dim
                if adr != -1:
                    return [float(x) for x in self.data.sensordata[adr:adr + dim]]
                return [0.0] * dim

            left_touch = read_sensor(self.sensor_left_touch)
            right_touch = read_sensor(self.sensor_right_touch)
            left_force = read_sensor(self.sensor_left_force)
            right_force = read_sensor(self.sensor_right_force)

            f_l_z = abs(left_force[2]) if len(left_force) >= 3 else 0.0
            f_r_z = abs(right_force[2]) if len(right_force) >= 3 else 0.0

            l_contact = bool((left_touch[0] > 0.5 if left_touch else False) or f_l_z > 5.0)
            r_contact = bool((right_touch[0] > 0.5 if right_touch else False) or f_r_z > 5.0)

            other_contacts = []
            for c_idx in range(min(self.data.ncon, 50)):
                c = self.data.contact[c_idx]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
                is_foot_floor = ("floor" in (g1, g2)) and ("foot" in g1 or "foot" in g2 or "ankle" in g1 or "ankle" in g2)
                if not is_foot_floor and g1 != g2:
                    other_contacts.append(f"{g1} <-> {g2}")

            return {
                "sim_time": float(self.data.time),
                "left_foot": {
                    "contact": l_contact,
                    "touch": left_touch[0] if left_touch else 0.0,
                    "force_z": f_l_z,
                    "force_vec": left_force,
                },
                "right_foot": {
                    "contact": r_contact,
                    "touch": right_touch[0] if right_touch else 0.0,
                    "force_z": f_r_z,
                    "force_vec": right_force,
                },
                "contact_count": int(self.data.ncon),
                "other_contacts": other_contacts[:10],
            }

    def get_policy_observation(self) -> Dict[str, Any]:
        """Return 78-D observation breakdown and labeled entries."""
        with self._lock:
            if self.policy_controller is None or self.policy_controller.current_observation is None:
                return {"dim": 0, "available": False, "raw": [], "labels": []}

            obs = self.policy_controller.current_observation
            labels = []
            labels.extend(["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"])
            labels.extend(["proj_gravity_x", "proj_gravity_y", "proj_gravity_z"])
            labels.extend(["cmd_vx", "cmd_vy", "cmd_vyaw"])
            labels.extend([f"pos_{jn}" for jn in self.policy_controller.slot01_names])
            labels.extend([f"pos_{jn}" for jn in self.policy_controller.slot23_names])
            labels.extend([f"pos_{jn}" for jn in self.policy_controller.slot45_names])
            labels.extend([f"vel_{jn}" for jn in self.policy_controller.slot01_names])
            labels.extend([f"vel_{jn}" for jn in self.policy_controller.slot23_names])
            labels.extend([f"vel_{jn}" for jn in self.policy_controller.slot45_names])
            labels.extend([f"prev_act_{jn}" for jn in self.policy_controller.policy_joints])

            return {
                "dim": len(obs),
                "available": True,
                "raw": [float(x) for x in obs],
                "labels": labels,
                "base_ang_vel": [float(x) for x in obs[0:3]],
                "projected_gravity": [float(x) for x in obs[3:6]],
                "command": [float(x) for x in obs[6:9]],
                "joint_pos": [float(x) for x in obs[9:32]],
                "joint_vel": [float(x) for x in obs[32:55]],
                "prev_actions": [float(x) for x in obs[55:78]],
            }

    def get_policy_action(self) -> Dict[str, Any]:
        """Return 23-D action breakdown, scaling, and desired joint targets."""
        with self._lock:
            if self.policy_controller is None or self.policy_controller.current_actions is None:
                return {"dim": 0, "available": False, "raw": [], "items": []}

            raw_act = self.policy_controller.current_actions
            policy_joints = self.policy_controller.policy_joints
            default_pos = self.policy_controller.default_pos
            action_scale = 0.25

            items = []
            for i, jn in enumerate(policy_joints):
                r = float(raw_act[i])
                scaled = r * action_scale
                d = float(default_pos.get(jn, 0.0))
                q_des = d + scaled
                kp, kd, eff = self.policy_controller.joint_gains.get(jn, (40.0, 2.0, 50.0))
                items.append({
                    "joint": jn,
                    "raw_action": r,
                    "action_scale": action_scale,
                    "scaled_delta": scaled,
                    "default_pos": d,
                    "desired_target": q_des,
                    "kp": kp,
                    "kd": kd,
                    "effort_limit": eff,
                })

            return {
                "dim": len(raw_act),
                "available": True,
                "action_scale": action_scale,
                "raw": [float(x) for x in raw_act],
                "items": items,
            }

    def get_actuator_state(self) -> List[Dict[str, Any]]:
        """Return actuator command targets, applied torques, and limits."""
        with self._lock:
            actuators = []
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
                actuators.append({
                    "index": i,
                    "name": name,
                    "ctrl": float(self.data.ctrl[i]),
                    "target": float(self.last_applied_targets[i]) if i < len(self.last_applied_targets) else 0.0,
                    "torque": float(self.last_applied_torques[i]) if i < len(self.last_applied_torques) else 0.0,
                    "force_limit": float(self.model.actuator_forcerange[i, 1]),
                })
            return actuators

    def get_environment_state(self) -> Dict[str, Any]:
        """Return environment configuration, ground friction, and obstacles."""
        with self._lock:
            cfg = self.env_manager.active_config
            return {
                "preset": self.env_manager.current_preset,
                "description": cfg.get("description", ""),
                "ground_friction": float(cfg.get("ground_friction", 1.0)),
                "mass_scale": float(cfg.get("mass_scale", 1.0)),
                "gravity": [float(x) for x in cfg.get("gravity", [0.0, 0.0, -9.81])],
                "obstacles": cfg.get("obstacles", []),
                "obstacle_count": len(cfg.get("obstacles", [])),
            }

    def get_state(self) -> Dict[str, Any]:
        """Return consolidated simulation state."""
        with self._lock:
            return {
                "sim_time": float(self.data.time),
                "wall_time": time.time(),
                "dt": float(self.dt),
                "step_count": int(self.step_count),
                "rtf": float(self.current_rtf),
                "paused": bool(self.paused),
                "control_mode": self.control_mode,
                "gantry_active": bool(self.gantry_active),
                "edge_mode": self.core.mode.name,
                "move_submode": self.core.move_submode.name,
                "command": {
                    "vx": float(self.core.current_vx),
                    "vy": float(self.core.current_vy),
                    "vyaw": float(self.core.current_vyaw),
                },
                "base": self.get_base_state(),
                "imu": self.get_imu(),
                "contacts": self.get_contacts(),
                "joints": self.get_joint_state(),
                "policy_obs": self.get_policy_observation(),
                "policy_act": self.get_policy_action(),
                "environment": self.get_environment_state(),
            }


# ── JSON Control Server (localhost:8852) ─────────────────────────────────────
class SimControlServer:
    """Lightweight HTTP server on localhost:8852 for sim control and dashboard."""

    def __init__(self, backend: SimBackend, port: int = 8852):
        self.backend = backend
        self.port = port
        self.server: Optional[http.server.HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        backend = self.backend

        class RequestHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Keep quiet

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html", "/dashboard"):
                    dash_path = _VASIMOV_DIR / "dashboard" / "index.html"
                    if dash_path.exists():
                        payload = dash_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    else:
                        self.send_error(404, "Dashboard HTML file not found")
                        return

                elif self.path == "/ground_truth":
                    frame = backend.get_ground_truth_frame()
                    payload = json.dumps(frame).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                elif self.path == "/status":
                    sim_z = float(backend.data.qpos[2])
                    quat = [float(backend.data.qpos[3]), float(backend.data.qpos[4]), float(backend.data.qpos[5]), float(backend.data.qpos[6])]
                    grav = compute_projected_gravity(np.array(quat))
                    data = {
                        "mode": backend.core.mode.name,
                        "gantry_active": backend.gantry_active,
                        "base_z": sim_z,
                        "projected_gravity_z": float(grav[2]),
                        "fault_latched": backend.core.fault_latched,
                        "error_flags": backend.core.error_flags,
                        "active_controller": backend.core.active_controller,
                        "battery_soc": backend.core.battery_soc,
                        "battery_voltage": backend.core.battery_voltage,
                        "battery_discharge_rate": backend.core.battery_discharge_rate,
                    }
                    payload = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if self.path == "/control":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length).decode("utf-8")
                    try:
                        req = json.loads(body)
                    except Exception:
                        self.send_error(400, "Invalid JSON")
                        return

                    cmd = req.get("command", "").lower()
                    if cmd == "gantry":
                        state = req.get("state", "off").lower()
                        backend.set_gantry(state == "on")
                        resp = {"status": "ok", "gantry": backend.gantry_active}
                    elif cmd == "push":
                        force = float(req.get("force", 40.0))
                        direction = str(req.get("dir", "x"))
                        duration = float(req.get("duration", 0.1))
                        backend.apply_push(force, direction, duration)
                        resp = {"status": "ok", "pushed": force}
                    elif cmd == "inject":
                        fault = str(req.get("fault", "")).lower()
                        if fault == "fall":
                            backend.core.inject_fall()
                        elif fault == "overtemp":
                            backend.core.inject_overtemp(float(req.get("temp", 85.0)))
                        elif fault == "can":
                            backend.core.inject_can_fault()
                        resp = {"status": "ok", "injected": fault}
                    elif cmd in ("set_battery", "battery"):
                        soc = req.get("soc", req.get("percentage"))
                        rate = req.get("rate", req.get("discharge_rate"))
                        voltage = req.get("voltage")
                        backend.core.set_battery(soc=soc, discharge_rate=rate, voltage=voltage)
                        resp = {"status": "ok", "battery_soc": backend.core.battery_soc, "discharge_rate": backend.core.battery_discharge_rate}
                    elif cmd == "restart":
                        backend.core.virtual_restart()
                        backend.reset()
                        resp = {"status": "ok", "restarted": True}
                    else:
                        resp = {"status": "error", "message": f"Unknown command '{cmd}'"}

                    payload = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)

        try:
            self.server = http.server.HTTPServer(("127.0.0.1", self.port), RequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="vasimov-sim-control")
            self.thread.start()
            log.info("[SIM CONTROL] Listening on http://127.0.0.1:%d", self.port)
        except Exception as e:
            log.warning("[SIM CONTROL] Could not bind port %d (%s); control server disabled", self.port, e)

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
