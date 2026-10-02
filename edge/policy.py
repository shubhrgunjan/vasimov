#!/usr/bin/env python3
"""
vasimov/edge/policy.py
Locomotion Policy Controller for Virtual Asimov 1 Edge.

Directly runs the official Hugging Face ONNX policy (Menlo/asimov1-locomotion-0818)
inside the Virtual Edge at 50 Hz on simulation time:
- Step 5 contract verified against assets/policy/policy.onnx and assets/policy/env.yaml.
- Inputs: 78-D observation vector built from MuJoCo floating base and joint state.
- Outputs: 23 joint position targets: q_des = q_default + 0.25 * a.
- Gains: Training stiffness Kp and damping Kd extracted from env.yaml actuators.
- Velocity clipping: Clamps commanded vx, vy, vyaw to contract limits with warning logs.
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml

from edge.policy_builder import (
    build_policy_observation,
    compute_desired_joint_targets,
    OBS_DIM,
    ACTION_DIM,
    ACTION_SCALE,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

log = logging.getLogger("vasimov.edge.policy")

# Contract velocity limits from env.yaml (commands.twist.ranges) [VERIFIED]
VX_MIN, VX_MAX = -0.6, 0.8
VY_MIN, VY_MAX = -0.5, 0.5
WZ_MIN, WZ_MAX = -0.8, 0.8

ACTION_SCALE: float = 0.25
OBS_DIM: int = 78
ACTION_DIM: int = 23


class PolicyController:
    """Policy controller running official Asimov 1 locomotion policy inside Virtual Edge."""

    def __init__(
        self,
        policy_onnx_path: Optional[Path] = None,
        env_yaml_path: Optional[Path] = None,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required for PolicyController")

        base_dir = Path(__file__).resolve().parent.parent
        self.policy_path = policy_onnx_path or (base_dir / "assets" / "policy" / "policy.onnx")
        self.env_yaml_path = env_yaml_path or (base_dir / "assets" / "policy" / "env.yaml")

        if not self.policy_path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {self.policy_path}")
        if not self.env_yaml_path.exists():
            raise FileNotFoundError(f"env.yaml not found: {self.env_yaml_path}")

        # 1. Initialize ONNX runtime session on CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.policy_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # 2. Parse env.yaml
        with open(self.env_yaml_path, "r", encoding="utf-8") as f:
            self.env_cfg = yaml.unsafe_load(f)

        self.policy_joints: List[str] = list(self.env_cfg["actions"]["joint_pos"]["joint_names"])
        assert len(self.policy_joints) == ACTION_DIM

        obs_cfg = self.env_cfg["observations"]["policy"]
        self.slot01_names: List[str] = list(obs_cfg["joint_pos_slot01"]["params"]["asset_cfg"]["joint_names"])
        self.slot23_names: List[str] = list(obs_cfg["joint_pos_slot23"]["params"]["asset_cfg"]["joint_names"])
        self.slot45_names: List[str] = list(obs_cfg["joint_pos_slot45"]["params"]["asset_cfg"]["joint_names"])

        # Default joint positions
        init_pos_cfg = self.env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
        self.default_pos: Dict[str, float] = {}
        for jn in self.policy_joints:
            for k, v in init_pos_cfg.items():
                if re.fullmatch(k, jn):
                    self.default_pos[jn] = float(v)
                    break
            if jn not in self.default_pos:
                self.default_pos[jn] = 0.0

        # Actuator gains and effort limits from env.yaml
        actuators_cfg = self.env_cfg["scene"]["robot"]["actuators"]
        self.joint_gains: Dict[str, Tuple[float, float, float]] = {}  # kp, kd, effort
        for act_group, act_c in actuators_cfg.items():
            exprs = act_c["joint_names_expr"]
            kp = float(act_c["stiffness"])
            kd = float(act_c["damping"])
            eff = float(act_c["effort_limit"])
            for jn in self.policy_joints:
                for expr in exprs:
                    if re.fullmatch(expr, jn):
                        self.joint_gains[jn] = (kp, kd, eff)

        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_step_sim_time: float = -1.0
        self.current_observation: Optional[np.ndarray] = None
        self.current_actions: Optional[np.ndarray] = None
        self.cached_targets: Dict[str, float] = dict(self.default_pos)

    def reset(self) -> None:
        """Reset internal recurrent / history action state."""
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_step_sim_time = -1.0
        self.current_observation = None
        self.current_actions = None
        self.cached_targets = dict(self.default_pos)

    @staticmethod
    def clip_velocity_command(vx: float, vy: float, vyaw: float) -> Tuple[float, float, float]:
        """Clip velocity command to contract limits and log if modified."""
        cvx = float(np.clip(vx, VX_MIN, VX_MAX))
        cvy = float(np.clip(vy, VY_MIN, VY_MAX))
        cvyaw = float(np.clip(vyaw, WZ_MIN, WZ_MAX))

        if not (math.isclose(cvx, vx, abs_tol=1e-4) and math.isclose(cvy, vy, abs_tol=1e-4) and math.isclose(cvyaw, vyaw, abs_tol=1e-4)):
            log.warning(
                "[POLICY] Commanded velocity (vx=%.2f, vy=%.2f, vyaw=%.2f) clipped to contract range: (vx=%.2f, vy=%.2f, vyaw=%.2f)",
                vx, vy, vyaw, cvx, cvy, cvyaw
            )
        return cvx, cvy, cvyaw

    def step(
        self,
        sim_time: float,
        command_vel: Tuple[float, float, float],
        pelvis_xmat: np.ndarray,
        pelvis_ang_vel_world: np.ndarray,
        joint_positions: Dict[str, float],
        joint_velocities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Evaluate policy at 50 Hz (every 0.020 s of sim_time).
        Returns desired joint positions for all 23 policy joints.
        """
        # If less than 19.9 ms since last tick, return cached targets (50 Hz decimation hold)
        if self.last_step_sim_time >= 0.0 and (sim_time - self.last_step_sim_time) < 0.0199:
            return self.cached_targets

        self.last_step_sim_time = sim_time

        # 1-5. Build 78-D observation using unified builder
        obs = build_policy_observation(
            pelvis_xmat=pelvis_xmat,
            pelvis_ang_vel_world=pelvis_ang_vel_world,
            command=command_vel,
            joint_pos_map=joint_positions,
            joint_vel_map=joint_velocities,
            prev_action=self.prev_action,
            default_pos_map=self.default_pos,
            slot01_names=self.slot01_names,
            slot23_names=self.slot23_names,
            slot45_names=self.slot45_names,
        )
        self.current_observation = obs

        # 6. Run ONNX inference
        raw_act = self.session.run(["actions"], {"obs": obs.reshape(1, OBS_DIM)})[0][0]
        self.prev_action = np.copy(raw_act)
        self.current_actions = raw_act

        # 7. Compute desired joint positions using unified builder
        targets = compute_desired_joint_targets(
            raw_actions=raw_act,
            policy_joints=self.policy_joints,
            default_pos_map=self.default_pos,
        )
        self.cached_targets = targets
        return targets
