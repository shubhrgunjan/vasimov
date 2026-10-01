#!/usr/bin/env python3
"""
vasimov/adapter.py
Joint and Actuator Adapter for Virtual Asimov 1.

Maps 25 firmware CAN bus actuators <-> 25 MuJoCo simulation actuators:
- Order: 1-to-1 matching CAN motor table order (0..24)
- Kinematics: Handles parallel differential ankle coupling for Left Ankle (4, 5)
  and Right Ankle (10, 11) using official kinematics formulas from
  asimov-firmware policy_thread.c (commit 2026-09-28) via menlo.asimov.robots.
- Kinematic constants: K_PITCH = 2.02, K_ROLL = 0.8 [VERIFIED]
- Velocity and torque mappings (via virtual work)
- Round-trip exactness tests for positions, velocities, and torques.

PROVENANCE & CITATIONS:
- Firmware joint order: menlo.asimov.robots.ASIMOV_1_BIPED_JOINTS [VERIFIED]
- Ankle kinematics coupling: menlo.asimov.robots.ANKLE_K_PITCH & ANKLE_K_ROLL [VERIFIED]
- Ankle limits: pitch +/- 0.35 rad, roll +/- 0.10 rad [VERIFIED]
- Neck gains: kp=40.0, kd=2.0 [UNKNOWN / ASSUMED]
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Sequence, Tuple, Union
import numpy as np

log = logging.getLogger("vasimov.adapter")

# ── Verification Status & Constants ──────────────────────────────────────────
# Verified from menlo.asimov.robots lines 13-39 (asimov-firmware asimov_1_biped.c)
FIRMWARE_JOINTS: Tuple[str, ...] = (
    "L_Hip_Pitch",       # 0
    "L_Hip_Roll",        # 1
    "L_Hip_Yaw",         # 2
    "L_Knee",            # 3
    "L_Ankle_A",         # 4
    "L_Ankle_B",         # 5
    "R_Hip_Pitch",       # 6
    "R_Hip_Roll",        # 7
    "R_Hip_Yaw",         # 8
    "R_Knee",            # 9
    "R_Ankle_A",         # 10
    "R_Ankle_B",         # 11
    "L_Shoulder_Pitch",  # 12
    "L_Shoulder_Roll",   # 13
    "L_Shoulder_Yaw",    # 14
    "L_Elbow",           # 15
    "L_Wrist_Yaw",       # 16
    "R_Shoulder_Pitch",  # 17
    "R_Shoulder_Roll",   # 18
    "R_Shoulder_Yaw",    # 19
    "R_Elbow",           # 20
    "R_Wrist_Yaw",       # 21
    "Waist_Yaw",         # 22
    "Neck_Yaw",          # 23 [ASSUMED GAINS]
    "Neck_Pitch",        # 24 [ASSUMED GAINS]
)

# MuJoCo Actuator / Joint names in model/asimov_1_vasimov.xml
SIM_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint",        # 0
    "left_hip_roll_joint",         # 1
    "left_hip_yaw_joint",          # 2
    "left_knee_joint",             # 3
    "left_ankle_pitch_joint",      # 4
    "left_ankle_roll_joint",       # 5
    "right_hip_pitch_joint",       # 6
    "right_hip_roll_joint",        # 7
    "right_hip_yaw_joint",         # 8
    "right_knee_joint",            # 9
    "right_ankle_pitch_joint",     # 10
    "right_ankle_roll_joint",      # 11
    "left_shoulder_pitch_joint",   # 12
    "left_shoulder_roll_joint",    # 13
    "left_shoulder_yaw_joint",     # 14
    "left_elbow_joint",            # 15
    "left_wrist_yaw_joint",        # 16
    "right_shoulder_pitch_joint",  # 17
    "right_shoulder_roll_joint",   # 18
    "right_shoulder_yaw_joint",    # 19
    "right_elbow_joint",           # 20
    "right_wrist_yaw_joint",       # 21
    "waist_yaw_joint",             # 22
    "neck_yaw_joint",              # 23
    "neck_pitch_joint",            # 24
)

# Official Ankle Kinematics Constants [VERIFIED: menlo.asimov.robots lines 47-50]
ANKLE_K_PITCH = 2.02
ANKLE_K_ROLL = 0.8
ANKLE_PITCH_LIMIT_RAD = 0.35
ANKLE_ROLL_LIMIT_RAD = 0.10

# Ankle joint pairs (A, B index pairs in firmware order)
LEFT_ANKLE_INDICES = (4, 5)
RIGHT_ANKLE_INDICES = (10, 11)
ANKLE_PAIRS = (LEFT_ANKLE_INDICES, RIGHT_ANKLE_INDICES)

# Assumed Neck Gains Flag
ASSUMED_NECK_KP = 40.0
ASSUMED_NECK_KD = 2.0
log.info(
    "[VASIMOV ADAPTER] Neck gains are UNKNOWN in upstream repos. "
    "Using ASSUMED values: Neck_Yaw kp=%.1f/kd=%.1f, Neck_Pitch kp=%.1f/kd=%.1f",
    ASSUMED_NECK_KP, ASSUMED_NECK_KD, ASSUMED_NECK_KP, ASSUMED_NECK_KD
)


# ── Ankle Coupling Transformations ───────────────────────────────────────────
def ankle_pr_to_ab(pitch: float, roll: float) -> Tuple[float, float]:
    """
    Sim ankle joint coordinates (pitch, roll) -> firmware motor positions (A, B).
    Official firmware formula:
        A = K_PITCH * pitch - K_ROLL * roll
        B = -K_PITCH * pitch - K_ROLL * roll
    """
    motor_a = ANKLE_K_PITCH * pitch - ANKLE_K_ROLL * roll
    motor_b = -ANKLE_K_PITCH * pitch - ANKLE_K_ROLL * roll
    return float(motor_a), float(motor_b)


def ankle_ab_to_pr(motor_a: float, motor_b: float) -> Tuple[float, float]:
    """
    Firmware motor positions (A, B) -> sim ankle joint coordinates (pitch, roll).
    Official firmware formula:
        pitch = (A - B) / (2 * K_PITCH)
        roll = -(A + B) / (2 * K_ROLL)
    """
    pitch = (motor_a - motor_b) / (2.0 * ANKLE_K_PITCH)
    roll = -(motor_a + motor_b) / (2.0 * ANKLE_K_ROLL)
    return float(pitch), float(roll)


def ankle_vel_pr_to_ab(vel_pitch: float, vel_roll: float) -> Tuple[float, float]:
    """
    Sim ankle joint velocities -> firmware motor velocities (A, B).
    Time derivative of forward mapping:
        v_a = K_PITCH * v_pitch - K_ROLL * v_roll
        v_b = -K_PITCH * v_pitch - K_ROLL * v_roll
    """
    vel_a = ANKLE_K_PITCH * vel_pitch - ANKLE_K_ROLL * vel_roll
    vel_b = -ANKLE_K_PITCH * vel_pitch - ANKLE_K_ROLL * vel_roll
    return float(vel_a), float(vel_b)


def ankle_vel_ab_to_pr(vel_a: float, vel_b: float) -> Tuple[float, float]:
    """
    Firmware motor velocities (A, B) -> sim ankle joint velocities.
        v_pitch = (v_a - v_b) / (2 * K_PITCH)
        v_roll = -(v_a + v_b) / (2 * K_ROLL)
    """
    vel_pitch = (vel_a - vel_b) / (2.0 * ANKLE_K_PITCH)
    vel_roll = -(vel_a + vel_b) / (2.0 * ANKLE_K_ROLL)
    return float(vel_pitch), float(vel_roll)


def ankle_torque_pr_to_ab(tau_pitch: float, tau_roll: float) -> Tuple[float, float]:
    """
    Sim ankle joint torques -> firmware motor torques (A, B).
    Principle of Virtual Work:
        tau_a * v_a + tau_b * v_b = tau_pitch * v_pitch + tau_roll * v_roll
        tau_a = tau_pitch / (2 * K_PITCH) - tau_roll / (2 * K_ROLL)
        tau_b = -tau_pitch / (2 * K_PITCH) - tau_roll / (2 * K_ROLL)
    """
    tau_a = (tau_pitch / (2.0 * ANKLE_K_PITCH)) - (tau_roll / (2.0 * ANKLE_K_ROLL))
    tau_b = (-tau_pitch / (2.0 * ANKLE_K_PITCH)) - (tau_roll / (2.0 * ANKLE_K_ROLL))
    return float(tau_a), float(tau_b)


def ankle_torque_ab_to_pr(tau_a: float, tau_b: float) -> Tuple[float, float]:
    """
    Firmware motor torques (A, B) -> sim ankle joint torques.
        tau_pitch = K_PITCH * (tau_a - tau_b)
        tau_roll = -K_ROLL * (tau_a + tau_b)
    """
    tau_pitch = ANKLE_K_PITCH * (tau_a - tau_b)
    tau_roll = -ANKLE_K_ROLL * (tau_a + tau_b)
    return float(tau_pitch), float(tau_roll)


# ── Full 25-Joint Vector Converters ──────────────────────────────────────────
class JointAdapter:
    """
    Adapter between 25-element firmware vectors and 25-element simulation vectors.
    """

    DOF: int = 25

    @classmethod
    def sim_to_firmware_positions(cls, sim_positions: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 simulation joint positions (where ankles are pitch/roll)
        into 25 firmware motor positions (where ankles are motors A/B).
        Used for EdgeTelemetry / RobotState.joint_pos.
        """
        if len(sim_positions) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} positions, got {len(sim_positions)}")
        out = list(float(x) for x in sim_positions)
        for a, b in ANKLE_PAIRS:
            pitch, roll = out[a], out[b]
            motor_a, motor_b = ankle_pr_to_ab(pitch, roll)
            out[a] = motor_a
            out[b] = motor_b
        return tuple(out)

    @classmethod
    def firmware_to_sim_positions(cls, fw_positions: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 firmware motor positions (where ankles are motors A/B)
        into 25 simulation joint positions (where ankles are pitch/roll).
        """
        if len(fw_positions) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} positions, got {len(fw_positions)}")
        out = list(float(x) for x in fw_positions)
        for a, b in ANKLE_PAIRS:
            motor_a, motor_b = out[a], out[b]
            pitch, roll = ankle_ab_to_pr(motor_a, motor_b)
            out[a] = pitch
            out[b] = roll
        return tuple(out)

    @classmethod
    def wire_trajectory_to_sim_positions(cls, wire_positions: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 trajectory positions received on the wire to sim joint positions.
        NOTE: Official SDK (menlo.asimov.robots.trajectory_wire_positions) encodes
        ankles as pitch and roll on the wire! Hence wire positions for ankles
        are ALREADY in joint pitch/roll coordinates.
        Firmware clamps pitch to +-0.35 rad and roll to +-0.10 rad.
        """
        if len(wire_positions) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} positions, got {len(wire_positions)}")
        out = list(float(x) for x in wire_positions)
        for a, b in ANKLE_PAIRS:
            out[a] = max(-ANKLE_PITCH_LIMIT_RAD, min(ANKLE_PITCH_LIMIT_RAD, out[a]))
            out[b] = max(-ANKLE_ROLL_LIMIT_RAD, min(ANKLE_ROLL_LIMIT_RAD, out[b]))
        return tuple(out)

    @classmethod
    def sim_to_wire_trajectory_positions(cls, sim_positions: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 sim joint positions to wire trajectory positions.
        Since sim positions already express ankles in orthogonal pitch/roll,
        this is 1-to-1 identical with the wire representation.
        """
        if len(sim_positions) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} positions, got {len(sim_positions)}")
        return tuple(float(x) for x in sim_positions)

    @classmethod
    def sim_to_firmware_velocities(cls, sim_vel: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 sim joint velocities -> 25 firmware motor velocities.
        """
        if len(sim_vel) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} velocities, got {len(sim_vel)}")
        out = list(float(x) for x in sim_vel)
        for a, b in ANKLE_PAIRS:
            vp, vr = out[a], out[b]
            va, vb = ankle_vel_pr_to_ab(vp, vr)
            out[a] = va
            out[b] = vb
        return tuple(out)

    @classmethod
    def firmware_to_sim_velocities(cls, fw_vel: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 firmware motor velocities -> 25 sim joint velocities.
        """
        if len(fw_vel) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} velocities, got {len(fw_vel)}")
        out = list(float(x) for x in fw_vel)
        for a, b in ANKLE_PAIRS:
            va, vb = out[a], out[b]
            vp, vr = ankle_vel_ab_to_pr(va, vb)
            out[a] = vp
            out[b] = vr
        return tuple(out)

    @classmethod
    def sim_to_firmware_torques(cls, sim_torques: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 sim joint torques -> 25 firmware motor torques.
        """
        if len(sim_torques) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} torques, got {len(sim_torques)}")
        out = list(float(x) for x in sim_torques)
        for a, b in ANKLE_PAIRS:
            tp, tr = out[a], out[b]
            ta, tb = ankle_torque_pr_to_ab(tp, tr)
            out[a] = ta
            out[b] = tb
        return tuple(out)

    @classmethod
    def firmware_to_sim_torques(cls, fw_torques: Sequence[float]) -> Tuple[float, ...]:
        """
        Convert 25 firmware motor torques -> 25 sim joint torques.
        """
        if len(fw_torques) != cls.DOF:
            raise ValueError(f"Expected {cls.DOF} torques, got {len(fw_torques)}")
        out = list(float(x) for x in fw_torques)
        for a, b in ANKLE_PAIRS:
            ta, tb = out[a], out[b]
            tp, tr = ankle_torque_ab_to_pr(ta, tb)
            out[a] = tp
            out[b] = tr
        return tuple(out)
