"""
vasimov/edge/policy_adapter.py
Policy Observation and Action Adapters.

Translates canonical RobotState into policy-specific observation tensors,
and translates raw policy output tensors into UnifiedRobotCommands.
"""

from __future__ import annotations
import abc
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from edge.policy_contract import PolicyManifest
from edge.robot_command import UnifiedRobotCommand
from edge.robot_state import RobotState

log = logging.getLogger("vasimov.edge.policy_adapter")


class BasePolicyAdapter(abc.ABC):
    """Abstract base adapter between RobotState and a Policy."""

    def __init__(self, manifest: PolicyManifest):
        self.manifest = manifest

    @abc.abstractmethod
    def build_observation(self, state: RobotState) -> np.ndarray:
        """Construct model input tensor from RobotState."""
        pass

    @abc.abstractmethod
    def process_action(self, raw_action: np.ndarray, state: RobotState) -> UnifiedRobotCommand:
        """Convert raw policy output tensor into UnifiedRobotCommand."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal history, filters, or recurrent state."""
        pass


class OfficialLocomotionAdapter(BasePolicyAdapter):
    """
    Adapter for official Menlo Asimov 1 locomotion policy (Menlo/asimov1-locomotion-0818).
    Constructs 78-D observation and maps 23-D action vector to position targets.
    """

    def __init__(self, manifest: PolicyManifest):
        super().__init__(manifest)
        self.policy_joints = list(manifest.action.joint_order)
        self.default_pos = dict(manifest.default_joint_pos)

        # Slot grouping for the 78-D vector (from official env.yaml)
        self.slot01_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
        ]
        self.slot23_names = [
            "left_hip_yaw_joint",
            "left_knee_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ]
        self.slot45_names = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "right_wrist_yaw_joint",
            "left_wrist_yaw_joint",
        ]

        # Extract PD gains from manifest or defaults
        self.joint_gains: Dict[str, Tuple[float, float, float]] = {}
        for jn in self.policy_joints:
            kp = 150.0 if "hip" in jn or "knee" in jn else (110.0 if "ankle" in jn else 40.0)
            kd = 5.0 if "hip" in jn or "knee" in jn or "ankle" in jn else 2.0
            eff = 40.0
            if manifest.pd_gains:
                for grp, gc in manifest.pd_gains.items():
                    joints_list = gc.get("joints", [])
                    if any(jn == j or (j.startswith(".*_") and jn.endswith(j[3:])) for j in joints_list):
                        kp = float(gc.get("kp", kp))
                        kd = float(gc.get("kd", kd))
                        eff = float(gc.get("effort_limit", eff))
                        break
            self.joint_gains[jn] = (kp, kd, eff)

        self.prev_action = np.zeros(23, dtype=np.float32)

    def reset(self) -> None:
        self.prev_action = np.zeros(23, dtype=np.float32)

    def build_observation(self, state: RobotState) -> np.ndarray:
        from edge.policy_builder import build_policy_observation
        obs = build_policy_observation(
            pelvis_xmat=state.pelvis_xmat,
            pelvis_ang_vel_world=state.base_ang_vel_world,
            command=state.command_vel,
            joint_pos_map=state.joint_positions,
            joint_vel_map=state.joint_velocities,
            prev_action=self.prev_action,
            default_pos_map=self.default_pos,
            slot01_names=self.slot01_names,
            slot23_names=self.slot23_names,
            slot45_names=self.slot45_names,
        )
        return obs.reshape(1, 78)

    def process_action(self, raw_action: np.ndarray, state: RobotState) -> UnifiedRobotCommand:
        from edge.policy_builder import compute_desired_joint_targets
        raw_act = np.asarray(raw_action, dtype=np.float32).flatten()
        self.prev_action = np.copy(raw_act)
        targets = compute_desired_joint_targets(
            raw_actions=raw_act,
            policy_joints=self.policy_joints,
            default_pos_map=self.default_pos,
        )

        kp_dict = {jn: self.joint_gains[jn][0] for jn in self.policy_joints}
        kd_dict = {jn: self.joint_gains[jn][1] for jn in self.policy_joints}
        effort_dict = {jn: self.joint_gains[jn][2] for jn in self.policy_joints}

        return UnifiedRobotCommand(
            joint_targets=targets,
            kp=kp_dict,
            kd=kd_dict,
            effort_limits=effort_dict,
            policy_name=self.manifest.name,
            timestamp=state.sim_time,
            mode="PD",
            raw_actions=raw_act,
        )


class GetupSafefallAdapter(BasePolicyAdapter):
    """
    Adapter for in-house Asimov 1 Get-up and Safe-fall policies (OpenHorizon Labs).
    Builds 375-D term-major observation history tensor (5 steps * 75-D terms).
    """

    def __init__(self, manifest: PolicyManifest):
        super().__init__(manifest)
        self.action_joints = list(manifest.action.joint_order)
        self.obs_joints = list(manifest.observation.obs_joint_order)
        self.default_pos = dict(manifest.default_joint_pos)
        self.history_len = manifest.observation.history_length  # 5

        # Terms with their scales
        self.terms = [(t.name, float(t.scale)) for t in manifest.observation.terms]
        if not self.terms:
            self.terms = [
                ("base_ang_vel", 0.25),
                ("projected_gravity", 1.0),
                ("joint_pos", 1.0),
                ("joint_vel", 0.1),
                ("actions", 1.0),
            ]

        # Extract PD gains from manifest
        self.joint_gains: Dict[str, Tuple[float, float, float]] = {}
        for jn in self.action_joints:
            kp = 150.0 if "hip" in jn or "knee" in jn else (110.0 if "ankle" in jn else 40.0)
            kd = 5.0 if "hip" in jn or "knee" in jn or "ankle" in jn else 2.0
            eff = 40.0
            if manifest.pd_gains:
                for grp, gc in manifest.pd_gains.items():
                    joints_list = gc.get("joints", [])
                    if any(jn == j or (j.startswith(".*_") and jn.endswith(j[3:])) for j in joints_list):
                        kp = float(gc.get("kp", kp))
                        kd = float(gc.get("kd", kd))
                        eff = float(gc.get("effort_limit", eff))
                        break
            self.joint_gains[jn] = (kp, kd, eff)

        self.last_action = np.zeros(len(self.action_joints), dtype=np.float32)
        self.buf: Dict[str, List[np.ndarray]] = {}

        # Fall trigger
        self.fall_triggered: bool = False
        self.fall_trigger_cfg = manifest.fall_trigger.get("deploy_defaults", {}) if manifest.fall_trigger else {}
        self.tilt_trigger_deg = float(self.fall_trigger_cfg.get("tilt_deg", 30.0))
        self.ang_vel_trigger = float(self.fall_trigger_cfg.get("ang_vel", 2.0))

    def reset(self) -> None:
        self.last_action = np.zeros(len(self.action_joints), dtype=np.float32)
        self.buf = {}
        self.fall_triggered = False

    def _get_term_vector(self, name: str, state: RobotState) -> np.ndarray:
        if name == "base_ang_vel":
            # IMU gyro, base frame
            R = state.pelvis_xmat
            return (R.T @ state.base_ang_vel_world).astype(np.float32)

        elif name == "projected_gravity":
            # unit gravity in pelvis frame
            return state.projected_gravity.astype(np.float32)

        elif name == "joint_pos":
            # (q - q_default) in obs_joint_order
            return np.array([
                float(state.joint_positions.get(jn, 0.0) - self.default_pos.get(jn, 0.0))
                for jn in self.obs_joints
            ], dtype=np.float32)

        elif name == "joint_vel":
            # qd in obs_joint_order
            return np.array([
                float(state.joint_velocities.get(jn, 0.0))
                for jn in self.obs_joints
            ], dtype=np.float32)

        elif name in ("actions", "prev_action"):
            return self.last_action.copy()

        else:
            log.warning("Unknown observation term: %s", name)
            return np.zeros(3, dtype=np.float32)

    def build_observation(self, state: RobotState) -> np.ndarray:
        # Check fall trigger if defined for safe-fall
        if self.manifest.fall_trigger is not None and not self.fall_triggered:
            tilt = state.tilt_deg
            w_sq = float(np.dot(state.base_ang_vel_world, state.base_ang_vel_world))
            if tilt > self.tilt_trigger_deg or w_sq > (self.ang_vel_trigger ** 2):
                self.fall_triggered = True
                log.info(
                    "[SAFEFALL TRIGGER] Tilt=%.1f deg (thresh=%.1f), |w|=%.2f rad/s -> Safe-fall policy active!",
                    tilt, self.tilt_trigger_deg, math.sqrt(w_sq)
                )

        parts = []
        for name, scale in self.terms:
            v = (self._get_term_vector(name, state) * scale).astype(np.float32)
            # Cold-start fill with initial step if empty
            if name not in self.buf:
                self.buf[name] = [v.copy() for _ in range(self.history_len)]
            else:
                self.buf[name].pop(0)
                self.buf[name].append(v.copy())
            parts.append(np.concatenate(self.buf[name]))

        obs = np.concatenate(parts).astype(np.float32)
        return obs.reshape(1, self.manifest.observation.dim)

    def process_action(self, raw_action: np.ndarray, state: RobotState) -> UnifiedRobotCommand:
        raw_act = np.asarray(raw_action, dtype=np.float32).flatten()

        # If fall trigger configured and not yet triggered, hold default pose (action 0)
        is_active = self.fall_triggered or (self.manifest.fall_trigger is None)
        applied_act = raw_act if is_active else np.zeros_like(raw_act)
        self.last_action = np.copy(applied_act)

        targets = {}
        kp_dict = {}
        kd_dict = {}
        effort_dict = {}

        for i, jn in enumerate(self.action_joints):
            a_val = float(applied_act[i]) if i < len(applied_act) else 0.0
            q_def = self.default_pos.get(jn, 0.0)
            targets[jn] = q_def + self.manifest.action.scale * a_val
            kp, kd, eff = self.joint_gains[jn]
            kp_dict[jn] = kp
            kd_dict[jn] = kd
            effort_dict[jn] = eff

        return UnifiedRobotCommand(
            joint_targets=targets,
            kp=kp_dict,
            kd=kd_dict,
            effort_limits=effort_dict,
            policy_name=self.manifest.name,
            timestamp=state.sim_time,
            mode="PD",
            raw_actions=applied_act,
        )
