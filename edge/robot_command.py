"""
vasimov/edge/robot_command.py
Unified Robot Command representation produced by policy adapters and consumed by actuators.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class UnifiedRobotCommand:
    """
    Standardized command contract between policy adapters and the robot safety/actuator layer.
    
    Prevents policies from writing arbitrary raw tensors directly to MuJoCo data structures.
    """

    # Target joint angles (rad)
    joint_targets: Dict[str, float] = field(default_factory=dict)

    # Joint impedance / servo gains (Kp, Kd)
    kp: Dict[str, float] = field(default_factory=dict)
    kd: Dict[str, float] = field(default_factory=dict)

    # Joint effort / torque limits (N*m)
    effort_limits: Dict[str, float] = field(default_factory=dict)

    # Optional feedforward torque (N*m)
    feedforward_torques: Dict[str, float] = field(default_factory=dict)

    # Source metadata
    policy_name: str = "unknown"
    timestamp: float = 0.0
    mode: str = "PD"  # "PD", "POSITION", "TORQUE", "DAMP"

    # Raw model output for flight recorder / telemetry
    raw_actions: Optional[np.ndarray] = None
