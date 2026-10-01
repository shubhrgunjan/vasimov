#!/usr/bin/env python3
"""
vasimov/ankle_map.py
Swappable differential ankle mapping module for Asimov 1.

Hardware Context:
Asimov 1 physical ankles (Left: Actuators 4, 5; Right: Actuators 10, 11) utilize two parallel
linear/crank actuators (Ankle A and Ankle B) driving the foot assembly differentially.
Neither upstream repo (asimov-mjlab, isaac_asimov) nor official docs publish the physical
crank/pushrod kinematics equations; both official simulators model and control orthogonal
ankle_pitch and ankle_roll joints directly.

This module provides a clean, swappable linear differential approximation:
- ab_to_pr: Actuator A/B angles -> Pitch / Roll joint angles
- pr_to_ab: Pitch / Roll joint angles -> Actuator A/B angles
- Velocity mappings via Jacobian J and J^{-1}
- Torque mappings via virtual work (J^T and J^{-T})
- Joint limit checking and clipping

A startup warning is emitted indicating this is an ASSUMED approximation.
"""

from __future__ import annotations
import sys
import warnings
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# Emit startup warning when module is imported
_STARTUP_WARNING_EMITTED = False

def _emit_ankle_warning():
    global _STARTUP_WARNING_EMITTED
    if not _STARTUP_WARNING_EMITTED:
        msg = (
            "[VASIMOV ANKLE WARNING] Using approximate linear differential ankle kinematic/torque mapping "
            "(1:1 ratio, ASSUMED). Official hardware pushrod linkage geometry is unpublished. "
            "Upstream RL training sim (isaac_asimov, asimov-mjlab) controls orthogonal pitch/roll directly."
        )
        print(msg, file=sys.stderr)
        warnings.warn(msg, UserWarning, stacklevel=3)
        _STARTUP_WARNING_EMITTED = True


@dataclass
class AnkleLimits:
    """Joint and actuator limits for Asimov 1 ankle (from asimov_1.urdf)."""
    # Pitch limits (rad)
    pitch_min: float = -0.35
    pitch_max: float = 0.35
    # Roll limits (rad)
    roll_min: float = -0.10
    roll_max: float = 0.10
    # Maximum torques (N*m)
    pitch_effort_limit: float = 40.0
    roll_effort_limit: float = 17.0
    actuator_effort_limit: float = 36.0  # Synapticon EC-A4310-P2-36 peak torque (18 Nm rated)
    # Velocity limits (rad/s)
    pitch_vel_limit: float = 9.32
    roll_vel_limit: float = 9.32
    actuator_vel_limit: float = 9.32


class DifferentialAnkleMap:
    """
    Linear differential ankle transmission mapping between parallel actuators (A, B)
    and orthogonal joint coordinates (pitch, roll).
    """

    def __init__(
        self,
        pitch_ratio: float = 1.0,
        roll_ratio: float = 1.0,
        limits: AnkleLimits = AnkleLimits(),
    ):
        """
        Args:
            pitch_ratio: Transmission ratio from symmetric actuator displacement to pitch (ASSUMED 1.0).
            roll_ratio: Transmission ratio from asymmetric actuator displacement to roll (ASSUMED 1.0).
            limits: AnkleLimits data containing physical boundaries.
        """
        _emit_ankle_warning()
        self.pitch_ratio = float(pitch_ratio)
        self.roll_ratio = float(roll_ratio)
        self.limits = limits

        # Forward Jacobian J: [qdot_p, qdot_r]^T = J * [qdot_a, qdot_b]^T
        # q_p = c_p * (q_a + q_b) / 2
        # q_r = c_r * (q_a - q_b) / 2
        self.J = np.array([
            [self.pitch_ratio / 2.0, self.pitch_ratio / 2.0],
            [self.roll_ratio / 2.0, -self.roll_ratio / 2.0],
        ], dtype=np.float64)

        # Inverse Jacobian J_inv: [qdot_a, qdot_b]^T = J_inv * [qdot_p, qdot_r]^T
        # q_a = q_p / c_p + q_r / c_r
        # q_b = q_p / c_p - q_r / c_r
        self.J_inv = np.array([
            [1.0 / self.pitch_ratio, 1.0 / self.roll_ratio],
            [1.0 / self.pitch_ratio, -1.0 / self.roll_ratio],
        ], dtype=np.float64)

    def ab_to_pr(self, q_a: float, q_b: float) -> Tuple[float, float]:
        """
        Convert actuator angles (A, B) to ankle joint angles (pitch, roll).

        Args:
            q_a: Actuator A angle (rad)
            q_b: Actuator B angle (rad)

        Returns:
            pitch: Ankle pitch angle (rad)
            roll: Ankle roll angle (rad)
        """
        pitch = self.pitch_ratio * (q_a + q_b) / 2.0
        roll = self.roll_ratio * (q_a - q_b) / 2.0
        return float(pitch), float(roll)

    def pr_to_ab(self, pitch: float, roll: float) -> Tuple[float, float]:
        """
        Convert ankle joint angles (pitch, roll) to actuator angles (A, B).

        Args:
            pitch: Ankle pitch angle (rad)
            roll: Ankle roll angle (rad)

        Returns:
            q_a: Actuator A angle (rad)
            q_b: Actuator B angle (rad)
        """
        q_a = (pitch / self.pitch_ratio) + (roll / self.roll_ratio)
        q_b = (pitch / self.pitch_ratio) - (roll / self.roll_ratio)
        return float(q_a), float(q_b)

    def velocity_ab_to_pr(self, dq_a: float, dq_b: float) -> Tuple[float, float]:
        """
        Map actuator velocities [dq_a, dq_b] to joint velocities [dq_pitch, dq_roll].
        """
        dq_ab = np.array([dq_a, dq_b], dtype=np.float64)
        dq_pr = self.J @ dq_ab
        return float(dq_pr[0]), float(dq_pr[1])

    def velocity_pr_to_ab(self, dq_pitch: float, dq_roll: float) -> Tuple[float, float]:
        """
        Map joint velocities [dq_pitch, dq_roll] to actuator velocities [dq_a, dq_b].
        """
        dq_pr = np.array([dq_pitch, dq_roll], dtype=np.float64)
        dq_ab = self.J_inv @ dq_pr
        return float(dq_ab[0]), float(dq_ab[1])

    def torque_pr_to_ab(self, tau_pitch: float, tau_roll: float) -> Tuple[float, float]:
        """
        Map desired joint torques [tau_pitch, tau_roll] to actuator torques [tau_a, tau_b].
        By Principle of Virtual Work:
            tau_ab = J^T * tau_pr
        """
        tau_pr = np.array([tau_pitch, tau_roll], dtype=np.float64)
        tau_ab = self.J.T @ tau_pr
        return float(tau_ab[0]), float(tau_ab[1])

    def torque_ab_to_pr(self, tau_a: float, tau_b: float) -> Tuple[float, float]:
        """
        Map actuator torques [tau_a, tau_b] to resultant joint torques [tau_pitch, tau_roll].
        By Principle of Virtual Work:
            tau_pr = J^{-T} * tau_ab = (J^{-1})^T * tau_ab
        """
        tau_ab = np.array([tau_a, tau_b], dtype=np.float64)
        tau_pr = self.J_inv.T @ tau_ab
        return float(tau_pr[0]), float(tau_pr[1])

    def clip_pr(self, pitch: float, roll: float) -> Tuple[float, float]:
        """Clip pitch and roll within URDF physical limits."""
        c_pitch = float(np.clip(pitch, self.limits.pitch_min, self.limits.pitch_max))
        c_roll = float(np.clip(roll, self.limits.roll_min, self.limits.roll_max))
        return c_pitch, c_roll

    def clip_ab_torques(self, tau_a: float, tau_b: float) -> Tuple[float, float]:
        """Clip actuator torques to motor effort limits."""
        limit = self.limits.actuator_effort_limit
        c_tau_a = float(np.clip(tau_a, -limit, limit))
        c_tau_b = float(np.clip(tau_b, -limit, limit))
        return c_tau_a, c_tau_b


# Module-level convenience functions
_default_map = DifferentialAnkleMap()

def ab_to_pr(q_a: float, q_b: float) -> Tuple[float, float]:
    return _default_map.ab_to_pr(q_a, q_b)

def pr_to_ab(pitch: float, roll: float) -> Tuple[float, float]:
    return _default_map.pr_to_ab(pitch, roll)

def velocity_ab_to_pr(dq_a: float, dq_b: float) -> Tuple[float, float]:
    return _default_map.velocity_ab_to_pr(dq_a, dq_b)

def velocity_pr_to_ab(dq_p: float, dq_r: float) -> Tuple[float, float]:
    return _default_map.velocity_pr_to_ab(dq_p, dq_r)

def torque_pr_to_ab(tau_p: float, tau_r: float) -> Tuple[float, float]:
    return _default_map.torque_pr_to_ab(tau_p, tau_r)

def torque_ab_to_pr(tau_a: float, tau_b: float) -> Tuple[float, float]:
    return _default_map.torque_ab_to_pr(tau_a, tau_b)


if __name__ == "__main__":
    print("Testing ankle module standalone:")
    p_in, r_in = 0.20, -0.05
    qa, qb = pr_to_ab(p_in, r_in)
    p_out, r_out = ab_to_pr(qa, qb)
    print(f"Roundtrip: input=({p_in}, {r_in}) -> AB=({qa:.4f}, {qb:.4f}) -> output=({p_out:.4f}, {r_out:.4f})")
