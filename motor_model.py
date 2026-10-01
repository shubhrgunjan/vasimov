#!/usr/bin/env python3
"""
vasimov/motor_model.py
Layered Motor Model for Virtual Asimov 1.

Layers:
- L0 (always on): Standard PD control law with torque clipping to effort_limit.
- L1 (velocity-dependent torque limit): tau_max(w) = effort * max(0, 1 - |w|/vel_limit).
     ASSUMED shape. Default follows official training sim (OFF in Isaac / MuJoCo).
- L2 (armature & friction): Handled directly in MuJoCo model XML via joint armature/damping.
- L3 (command delay / first-order lag): Default OFF. Parameters ASSUMED (tau=10ms, delay=1 step).
- L4 (current & thermal dynamics): "NOT MODELED" (no winding/thermal datasheets in upstream repos/docs).

Per-motor telemetry output:
- position (rad)
- velocity (rad/s)
- applied_torque (N*m)
- current: "NOT MODELED"
- temperature: "NOT MODELED"
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml


@dataclass
class MotorState:
    """Per-motor runtime telemetry and status."""
    joint_name: str
    firmware_name: str
    position: float
    velocity: float
    raw_pd_torque: float
    clamped_torque: float
    applied_torque: float
    effort_limit: float
    velocity_limit: float
    current: Union[str, float] = "NOT MODELED"
    temperature: Union[str, float] = "NOT MODELED"


@dataclass
class MotorConfig:
    """Per-motor parameter configuration."""
    joint_name: str
    firmware_name: str
    actuator_id: int
    kp: float
    kd: float
    effort_limit: float
    velocity_limit: float
    armature: float
    damping: float


@dataclass
class LayerOptions:
    """Layer toggle states and parameter settings."""
    l0_enabled: bool = True
    l1_enabled: bool = False  # Matches official training sim baseline
    l2_enabled: bool = True   # Evaluated in MuJoCo physics
    l2_mode: str = "mujoco_xml"
    l3_enabled: bool = False  # Default OFF
    l3_mode: str = "lag"      # "lag" or "delay"
    l3_lag_tau: float = 0.01  # 10 ms first-order lag (ASSUMED)
    l3_delay_steps: int = 1   # 1 step delay (ASSUMED)
    l4_enabled: bool = False
    l4_status: str = "NOT MODELED"


class MotorModel:
    """
    Simulates layered actuator dynamics for the 25 hinge joints of Asimov 1.
    """

    def __init__(
        self,
        motors_yaml_path: Optional[Union[str, Path]] = None,
        joints_yaml_path: Optional[Union[str, Path]] = None,
        gains_yaml_path: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent
        if motors_yaml_path is None:
            motors_yaml_path = base_dir / "config" / "motors.yaml"
        if joints_yaml_path is None:
            joints_yaml_path = base_dir / "config" / "joints.yaml"
        if gains_yaml_path is None:
            gains_yaml_path = base_dir / "config" / "gains.yaml"

        self.motors_yaml_path = Path(motors_yaml_path)
        self.joints_yaml_path = Path(joints_yaml_path)
        self.gains_yaml_path = Path(gains_yaml_path)

        self.layer_opts = LayerOptions()
        self.motors: List[MotorConfig] = []
        self.joint_name_to_idx: Dict[str, int] = {}
        self.firmware_name_to_idx: Dict[str, int] = {}

        # L3 state storage
        self._l3_prev_torques: np.ndarray = np.zeros(0, dtype=np.float64)
        self._l3_delay_queues: List[deque] = []

        self._load_config()
        self.reset()

    def _load_config(self) -> None:
        """Load and parse YAML configurations."""
        # 1. Load motors.yaml for layer switches
        if self.motors_yaml_path.exists():
            with open(self.motors_yaml_path, "r", encoding="utf-8") as f:
                m_cfg = yaml.safe_load(f) or {}
            layers = m_cfg.get("layers", {})
            l0 = layers.get("l0", {})
            l1 = layers.get("l1", {})
            l2 = layers.get("l2", {})
            l3 = layers.get("l3", {})
            l4 = layers.get("l4", {})

            self.layer_opts.l0_enabled = bool(l0.get("enabled", True))
            self.layer_opts.l1_enabled = bool(l1.get("enabled", False))
            self.layer_opts.l2_enabled = bool(l2.get("enabled", True))
            self.layer_opts.l3_enabled = bool(l3.get("enabled", False))
            self.layer_opts.l3_mode = str(l3.get("mode", "lag"))
            self.layer_opts.l3_lag_tau = float(l3.get("lag_tau", 0.01))
            self.layer_opts.l3_delay_steps = int(l3.get("delay_steps", 1))
            self.layer_opts.l4_enabled = bool(l4.get("enabled", False))
            self.layer_opts.l4_status = str(l4.get("status", "NOT MODELED"))

        # 2. Load gains.yaml for default kp, kd
        gains_map = {}
        if self.gains_yaml_path.exists():
            with open(self.gains_yaml_path, "r", encoding="utf-8") as f:
                g_cfg = yaml.safe_load(f) or {}
            gains_map = g_cfg.get("gains", {})

        # 3. Load joints.yaml for limits and metadata
        if not self.joints_yaml_path.exists():
            raise FileNotFoundError(f"Missing required joints.yaml at {self.joints_yaml_path}")

        with open(self.joints_yaml_path, "r", encoding="utf-8") as f:
            j_cfg = yaml.safe_load(f) or {}

        joints_list = j_cfg.get("joints", [])
        self.motors = []
        self.joint_name_to_idx = {}
        self.firmware_name_to_idx = {}

        for row in joints_list:
            idx = int(row["index"])
            fw_name = row["firmware_name"]
            sim_name = row["sim_joint"]
            effort = float(row.get("effort_limit", 40.0))
            vel_limit = float(row.get("velocity_limit", 10.0))
            armature = float(row.get("armature", 0.01))
            damping = float(row.get("damping", 1.0))

            # Gains from gains.yaml or sensible defaults from official docs
            g_entry = gains_map.get(sim_name, {})
            kp = float(g_entry.get("kp", 100.0))
            kd = float(g_entry.get("kd", 4.0))

            cfg = MotorConfig(
                joint_name=sim_name,
                firmware_name=fw_name,
                actuator_id=idx,
                kp=kp,
                kd=kd,
                effort_limit=effort,
                velocity_limit=vel_limit,
                armature=armature,
                damping=damping,
            )
            self.motors.append(cfg)
            self.joint_name_to_idx[sim_name] = idx
            self.firmware_name_to_idx[fw_name] = idx

        self.num_motors = len(self.motors)
        self._l3_prev_torques = np.zeros(self.num_motors, dtype=np.float64)
        self._l3_delay_queues = [
            deque([0.0] * max(1, self.layer_opts.l3_delay_steps), maxlen=max(1, self.layer_opts.l3_delay_steps))
            for _ in range(self.num_motors)
        ]

    def reset(self) -> None:
        """Reset internal filter states and delay buffers."""
        if hasattr(self, "num_motors") and self.num_motors > 0:
            self._l3_prev_torques = np.zeros(self.num_motors, dtype=np.float64)
            self._l3_delay_queues = [
                deque([0.0] * max(1, self.layer_opts.l3_delay_steps), maxlen=max(1, self.layer_opts.l3_delay_steps))
                for _ in range(self.num_motors)
            ]

    def set_layer_l1(self, enabled: bool) -> None:
        """Toggle Layer 1 (Velocity-dependent torque clipping)."""
        self.layer_opts.l1_enabled = enabled

    def set_layer_l3(
        self,
        enabled: bool,
        mode: Optional[str] = None,
        lag_tau: Optional[float] = None,
        delay_steps: Optional[int] = None,
    ) -> None:
        """Toggle Layer 3 (Command lag / delay)."""
        self.layer_opts.l3_enabled = enabled
        if mode is not None:
            self.layer_opts.l3_mode = mode
        if lag_tau is not None:
            self.layer_opts.l3_lag_tau = lag_tau
        if delay_steps is not None:
            self.layer_opts.l3_delay_steps = delay_steps
        self.reset()

    def compute_torque_limit(self, motor_idx: int, velocity: float) -> float:
        """
        Compute maximum allowed torque for motor_idx under current velocity.
        - L0 only: tau_max = effort_limit
        - L1 active: tau_max = effort_limit * max(0.0, 1.0 - |velocity| / velocity_limit)
        """
        cfg = self.motors[motor_idx]
        base_effort = cfg.effort_limit

        if not self.layer_opts.l1_enabled:
            return base_effort

        # L1 velocity-dependent derating envelope
        if cfg.velocity_limit <= 1e-6:
            derating = 1.0
        else:
            speed_ratio = abs(velocity) / cfg.velocity_limit
            derating = max(0.0, 1.0 - speed_ratio)

        return base_effort * derating

    def step(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        q_des: np.ndarray,
        qdot_des: np.ndarray,
        dt: float = 0.005,
        kp_override: Optional[Sequence[float]] = None,
        kd_override: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, MotorState]]:
        """
        Compute actuator torques for all 25 joints.

        Args:
            q: Current joint positions (rad), length 25
            qdot: Current joint velocities (rad/s), length 25
            q_des: Target joint positions (rad), length 25
            qdot_des: Target joint velocities (rad/s), length 25
            dt: Simulation timestep (s)
            kp_override: Optional per-joint proportional gains (length 25)
            kd_override: Optional per-joint derivative gains (length 25)

        Returns:
            applied_torques: Array of torques (N*m) to apply to MuJoCo actuators, shape (25,)
            states: Dictionary of MotorState objects keyed by joint name
        """
        applied_torques = np.zeros(self.num_motors, dtype=np.float64)
        states: Dict[str, MotorState] = {}

        for i, cfg in enumerate(self.motors):
            pos = float(q[i])
            vel = float(qdot[i])
            target_pos = float(q_des[i])
            target_vel = float(qdot_des[i])

            kp = float(kp_override[i]) if kp_override is not None else cfg.kp
            kd = float(kd_override[i]) if kd_override is not None else cfg.kd

            # --- Layer 0: PD control law ---
            # tau_pd = kp * (q_des - q) + kd * (qdot_des - qdot)
            err_pos = target_pos - pos
            err_vel = target_vel - vel
            raw_pd = kp * err_pos + kd * err_vel

            # --- Layer 1: Velocity-dependent torque saturation ---
            tau_max = self.compute_torque_limit(i, vel)
            clamped = float(np.clip(raw_pd, -tau_max, tau_max))

            # --- Layer 3: Command delay or first-order lag ---
            if self.layer_opts.l3_enabled:
                if self.layer_opts.l3_mode == "lag":
                    tau_const = max(1e-5, self.layer_opts.l3_lag_tau)
                    alpha = dt / (dt + tau_const)
                    out_torque = alpha * clamped + (1.0 - alpha) * self._l3_prev_torques[i]
                    self._l3_prev_torques[i] = out_torque
                elif self.layer_opts.l3_mode == "delay":
                    out_torque = self._l3_delay_queues[i].popleft()
                    self._l3_delay_queues[i].append(clamped)
                else:
                    out_torque = clamped
            else:
                out_torque = clamped

            applied_torques[i] = out_torque

            # --- Layer 4: Current & Thermal Dynamics ---
            # If motor datasheet is found, model current and temp rise.
            # Upstream status: NOT MODELED
            current_val = "NOT MODELED"
            temp_val = "NOT MODELED"

            states[cfg.joint_name] = MotorState(
                joint_name=cfg.joint_name,
                firmware_name=cfg.firmware_name,
                position=pos,
                velocity=vel,
                raw_pd_torque=raw_pd,
                clamped_torque=clamped,
                applied_torque=out_torque,
                effort_limit=cfg.effort_limit,
                velocity_limit=cfg.velocity_limit,
                current=current_val,
                temperature=temp_val,
            )

        return applied_torques, states


if __name__ == "__main__":
    mm = MotorModel()
    print(f"MotorModel initialized successfully with {mm.num_motors} motors.")
    for i, m in enumerate(mm.motors):
        print(f"[{i:02d}] {m.firmware_name:<16} -> {m.joint_name:<26} effort={m.effort_limit:4.1f} vel={m.velocity_limit:5.2f} kp={m.kp:4.0f} kd={m.kd:3.1f}")
