import sys
from pathlib import Path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

import pytest
from web.server.binary_protocol import (
    pack_state_packet, unpack_state_packet,
    pack_command_packet, unpack_command_packet,
    STATE_PACKET_SIZE, COMMAND_PACKET_SIZE, PROTOCOL_VERSION,
    RobotMode, PolicyId, RecoveryState, CommandType, StateFlags
)

def test_state_pack_unpack():
    joint_pos = [0.1 * i for i in range(25)]
    joint_vel = [0.02 * i for i in range(25)]
    data = pack_state_packet(
        sequence=10,
        timestamp=5.25,
        robot_mode=RobotMode.MOVE,
        policy_id=PolicyId.OFFICIAL_LOCOMOTION,
        recovery_state=RecoveryState.NORMAL,
        safety_state=0,
        flags=StateFlags.CONTACT_LEFT | StateFlags.CONTACT_RIGHT,
        base_position=[0.1, -0.2, 0.61],
        base_orientation=[0.0, 0.0, 0.1, 0.995],
        linear_velocity=[0.4, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, -0.2],
        joint_positions=joint_pos,
        joint_velocities=joint_vel,
        command_linear_x=0.4,
        command_linear_y=0.0,
        command_yaw=-0.2,
    )
    assert len(data) == STATE_PACKET_SIZE
    unpacked = unpack_state_packet(data)
    assert unpacked["sequence_number"] == 10
    assert abs(unpacked["timestamp"] - 5.25) < 1e-6
    assert unpacked["robot_mode"] == RobotMode.MOVE
    assert len(unpacked["joint_positions"]) == 25
    assert abs(unpacked["joint_positions"][5] - 0.5) < 1e-5

def test_command_pack_unpack():
    data = pack_command_packet(
        sequence=55,
        timestamp=100.5,
        command_type=CommandType.WALK,
        linear_x=0.5,
        linear_y=-0.1,
        yaw_rate=0.3,
    )
    assert len(data) == COMMAND_PACKET_SIZE
    cmd = unpack_command_packet(data)
    assert cmd["sequence_number"] == 55
    assert cmd["command_type"] == CommandType.WALK
    assert abs(cmd["linear_x"] - 0.5) < 1e-5

def test_malformed_packets():
    with pytest.raises(ValueError, match="Invalid packet size"):
        unpack_state_packet(b"too_short")

    with pytest.raises(ValueError, match="Invalid command packet size"):
        unpack_command_packet(b"short")

def test_joint_length_validation():
    with pytest.raises(ValueError, match="Expected 25"):
        pack_state_packet(
            sequence=1, timestamp=0.0, robot_mode=0, policy_id=0,
            recovery_state=0, safety_state=0, flags=0,
            base_position=[0,0,0], base_orientation=[0,0,0,1],
            linear_velocity=[0,0,0], angular_velocity=[0,0,0],
            joint_positions=[0.0]*20,
            joint_velocities=[0.0]*25,
        )
