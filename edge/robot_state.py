"""
vasimov/edge/robot_state.py
Canonical, policy-independent representation of robot physical state.

Decouples physical simulation, sensor estimation, and kinematics from
any specific policy observation tensor layout.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class RobotState:
    """
    Canonical snapshot of robot physical state generated at each simulation / control step.
    
    All quantities use SI units (m, rad, s, N, N*m).
    All frames follow the standard robotics conventions:
    - world: fixed simulation world frame (+Z up)
    - body: pelvis body frame (+X forward, +Y left, +Z up)
    """

    # Timing
    sim_time: float = 0.0
    dt: float = 0.005
    step_count: int = 0

    # Base kinematics
    pelvis_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    pelvis_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    pelvis_xmat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    # Base velocities
    base_lin_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_ang_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_lin_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    base_ang_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Projected gravity in pelvis body frame [gx, gy, gz]. Upright: [0, 0, -1]
    projected_gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float64))

    # Joint states (name -> value)
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)

    # Actuator states & torques
    actuator_ctrl: Dict[str, float] = field(default_factory=dict)
    applied_torques: Dict[str, float] = field(default_factory=dict)

    # Foot contacts & reaction forces
    contacts: Dict[str, Any] = field(default_factory=dict)

    # High-level operator commands
    command_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, vyaw)
    mode: str = "MOVE"
    gantry_active: bool = False

    # Previous policy actions (policy_name -> array)
    previous_actions: Dict[str, np.ndarray] = field(default_factory=dict)

    # Extensible metadata (environment friction, camera frames, alerts)
    custom: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def zeros(cls) -> "RobotState":
        """Construct a default zeroed RobotState."""
        return cls()

    @property
    def tilt_deg(self) -> float:
        """Angle between body +Z and gravity vertical in degrees."""
        gz = float(np.clip(-self.projected_gravity[2], -1.0, 1.0))
        return float(np.degrees(np.arccos(gz)))

    @property
    def is_fallen(self) -> bool:
        """True if robot tilt exceeds standard fall threshold (> 45 deg) or base z < 0.35m."""
        return self.tilt_deg > 45.0 or (self.pelvis_pos[2] < 0.35 and not self.gantry_active)
