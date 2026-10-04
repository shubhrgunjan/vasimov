"""
vasimov/web/server/binary_protocol.py
High-Performance Versioned Binary State & Control Protocol (VAS1 & VAC1).

Designed for 50 Hz real-time state streaming to Android/web clients:
- Fixed 288-byte state packet (VAS1)
- Fixed 32-byte control command packet (VAC1)
- Zero-copy / struct packing with strict byte alignment.
"""

from __future__ import annotations
import enum
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

# ── Magic & Constants ────────────────────────────────────────────────────────
STATE_MAGIC = b"VAS1"
COMMAND_MAGIC = b"VAC1"
PROTOCOL_VERSION = 1

STATE_PACKET_SIZE = 288
COMMAND_PACKET_SIZE = 32

# Struct format strings (Little-Endian "<")
# VAS1: 4s magic, 6x uint8, uint16 reserved, uint32 seq, double time, 3f pos, 4f quat, 3f lin_vel, 3f ang_vel, 25f qpos, 25f qvel, 3f cmd
STATE_FMT = "<4sBBBBBBHId3f4f3f3f25f25f3f"
# VAC1: 4s magic, uint8 ver, uint8 cmd_type, uint16 flags, uint32 seq, double time, 3f (vx, vy, wz)
COMMAND_FMT = "<4sBBHId3f"

assert struct.calcsize(STATE_FMT) == STATE_PACKET_SIZE
assert struct.calcsize(COMMAND_FMT) == COMMAND_PACKET_SIZE


class RobotMode(enum.IntEnum):
    DAMP = 0
    STAND = 1
    MOVE = 2
    FAULT_DAMP = 5


class PolicyId(enum.IntEnum):
    NONE = 0
    OFFICIAL_LOCOMOTION = 1
    SAFEFALL = 2
    GETUP = 3
    MANUAL_OVERRIDE = 4


class RecoveryState(enum.IntEnum):
    NORMAL = 0
    FALL_DETECTED = 1
    DAMPING_ACTIVE = 2
    GETUP_IN_PROGRESS = 3
    STAND_ARMED = 4


class CommandType(enum.IntEnum):
    WALK = 1
    STAND = 2
    CROUCH = 3
    RECOVERY = 4
    RESET = 5
    ESTOP = 6
    DAMP = 7


class SafetyFlags(enum.IntFlag):
    FALL_DETECTED = 0x01
    JOINT_LIMIT_REACHED = 0x02
    MOTOR_OVERTEMP = 0x04
    CAN_BUS_ERROR = 0x08
    WATCHDOG_TIMEOUT = 0x10


class StateFlags(enum.IntFlag):
    FAULT_LATCHED = 0x01
    GANTRY_ACTIVE = 0x02
    CONTACT_LEFT = 0x04
    CONTACT_RIGHT = 0x08


def pack_state_packet(
    sequence: int,
    timestamp: float,
    robot_mode: int,
    policy_id: int,
    recovery_state: int,
    safety_state: int,
    flags: int,
    base_position: Sequence[float],
    base_orientation: Sequence[float],
    linear_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    command_linear_x: float = 0.0,
    command_linear_y: float = 0.0,
    command_yaw: float = 0.0,
) -> bytes:
    """Pack authoritative robot state into a fixed 288-byte VAS1 packet."""
    if len(joint_positions) != 25 or len(joint_velocities) != 25:
        raise ValueError(f"Expected 25 joint positions/velocities, got {len(joint_positions)} and {len(joint_velocities)}")

    return struct.pack(
        STATE_FMT,
        STATE_MAGIC,
        PROTOCOL_VERSION,
        int(flags) & 0xFF,
        int(robot_mode) & 0xFF,
        int(policy_id) & 0xFF,
        int(recovery_state) & 0xFF,
        int(safety_state) & 0xFF,
        0,  # reserved uint16
        int(sequence) & 0xFFFFFFFF,
        float(timestamp),
        float(base_position[0]), float(base_position[1]), float(base_position[2]),
        float(base_orientation[0]), float(base_orientation[1]), float(base_orientation[2]), float(base_orientation[3]),
        float(linear_velocity[0]), float(linear_velocity[1]), float(linear_velocity[2]),
        float(angular_velocity[0]), float(angular_velocity[1]), float(angular_velocity[2]),
        *(float(x) for x in joint_positions),
        *(float(x) for x in joint_velocities),
        float(command_linear_x),
        float(command_linear_y),
        float(command_yaw),
    )


def unpack_state_packet(data: bytes) -> Dict[str, Any]:
    """Unpack a 288-byte VAS1 packet into a validated dictionary."""
    if len(data) != STATE_PACKET_SIZE:
        raise ValueError(f"Invalid packet size: expected {STATE_PACKET_SIZE} bytes, got {len(data)}")

    unpacked = struct.unpack(STATE_FMT, data)

    magic = unpacked[0]
    if magic != STATE_MAGIC:
        raise ValueError(f"Invalid state magic: expected {STATE_MAGIC}, got {magic}")

    version = unpacked[1]
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: expected {PROTOCOL_VERSION}, got {version}")

    flags = unpacked[2]
    robot_mode = unpacked[3]
    policy_id = unpacked[4]
    recovery_state = unpacked[5]
    safety_state = unpacked[6]
    sequence = unpacked[8]
    timestamp = unpacked[9]

    base_pos = list(unpacked[10:13])
    base_quat = list(unpacked[13:17])
    lin_vel = list(unpacked[17:20])
    ang_vel = list(unpacked[20:23])

    joint_positions = list(unpacked[23:48])
    joint_velocities = list(unpacked[48:73])

    cmd_x = unpacked[73]
    cmd_y = unpacked[74]
    cmd_yaw = unpacked[75]

    return {
        "magic": "VAS1",
        "protocol_version": version,
        "flags": flags,
        "robot_mode": robot_mode,
        "policy_id": policy_id,
        "recovery_state": recovery_state,
        "safety_state": safety_state,
        "sequence_number": sequence,
        "timestamp": timestamp,
        "base_position": base_pos,
        "base_orientation": base_quat,
        "linear_velocity": lin_vel,
        "angular_velocity": ang_vel,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "command_echo": {
            "linear_x": cmd_x,
            "linear_y": cmd_y,
            "yaw": cmd_yaw,
        },
    }


def pack_command_packet(
    sequence: int,
    timestamp: float,
    command_type: int,
    linear_x: float = 0.0,
    linear_y: float = 0.0,
    yaw_rate: float = 0.0,
    flags: int = 0,
) -> bytes:
    """Pack an Android teleoperation command into a fixed 32-byte VAC1 packet."""
    return struct.pack(
        COMMAND_FMT,
        COMMAND_MAGIC,
        PROTOCOL_VERSION,
        int(command_type) & 0xFF,
        int(flags) & 0xFFFF,
        int(sequence) & 0xFFFFFFFF,
        float(timestamp),
        float(linear_x),
        float(linear_y),
        float(yaw_rate),
    )


def unpack_command_packet(data: bytes) -> Dict[str, Any]:
    """Unpack a 32-byte VAC1 packet."""
    if len(data) != COMMAND_PACKET_SIZE:
        raise ValueError(f"Invalid command packet size: expected {COMMAND_PACKET_SIZE} bytes, got {len(data)}")

    unpacked = struct.unpack(COMMAND_FMT, data)

    magic = unpacked[0]
    if magic != COMMAND_MAGIC:
        raise ValueError(f"Invalid command magic: expected {COMMAND_MAGIC}, got {magic}")

    version = unpacked[1]
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Protocol version mismatch: expected {PROTOCOL_VERSION}, got {version}")

    cmd_type = unpacked[2]
    flags = unpacked[3]
    sequence = unpacked[4]
    timestamp = unpacked[5]
    vx = unpacked[6]
    vy = unpacked[7]
    wz = unpacked[8]

    return {
        "magic": "VAC1",
        "protocol_version": version,
        "command_type": cmd_type,
        "flags": flags,
        "sequence_number": sequence,
        "timestamp": timestamp,
        "linear_x": vx,
        "linear_y": vy,
        "yaw_rate": wz,
    }
