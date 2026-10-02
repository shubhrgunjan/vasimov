"""
vasimov/edge/policy_builder.py
Unified observation and action builder for Asimov 1 locomotion policy contract.

Single Source of Truth shared by:
- edge/policy.py (Virtual Edge runtime)
- tools/policy_standalone.py (Standalone verification harness)
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

OBS_DIM: int = 78
ACTION_DIM: int = 23
ACTION_SCALE: float = 0.25
ANG_VEL_SCALE: float = 0.25
JOINT_VEL_SCALE: float = 0.10


def build_policy_observation(
    pelvis_xmat: np.ndarray,
    pelvis_ang_vel_world: np.ndarray,
    command: Sequence[float],
    joint_pos_map: Dict[str, float],
    joint_vel_map: Dict[str, float],
    prev_action: np.ndarray,
    default_pos_map: Dict[str, float],
    slot01_names: Sequence[str],
    slot23_names: Sequence[str],
    slot45_names: Sequence[str],
) -> np.ndarray:
    """
    Construct the verified 78-D observation vector for Menlo/asimov1-locomotion-0818.

    Observation layout:
    1. base_ang_vel: 3-D, pelvis body-frame angular velocity * 0.25
    2. projected_gravity: 3-D, body-frame gravity vector [0, 0, -1] * R
    3. velocity_commands: 3-D, [vx, vy, vyaw]
    4. joint_pos_slot01: 9-D, (q - q_default) * 1.0
    5. joint_pos_slot23: 8-D, (q - q_default) * 1.0
    6. joint_pos_slot45: 6-D, (q - q_default) * 1.0
    7. joint_vel_slot01: 9-D, qdot * 0.1
    8. joint_vel_slot23: 8-D, qdot * 0.1
    9. joint_vel_slot45: 6-D, qdot * 0.1
    10. actions: 23-D, previous action vector
    Total: 3 + 3 + 3 + 9 + 8 + 6 + 9 + 8 + 6 + 23 = 78
    """
    R = np.asarray(pelvis_xmat, dtype=np.float64).reshape(3, 3)

    # 1. Base angular velocity in local pelvis frame (scaled by 0.25)
    omega_world = np.asarray(pelvis_ang_vel_world, dtype=np.float64).reshape(3)
    omega_body = R.T @ omega_world

    # 2. Projected gravity in local pelvis frame
    grav_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    grav_body = R.T @ grav_world

    # 3. Commanded velocity
    cmd = np.asarray(command, dtype=np.float32).reshape(3)

    # 4-6. Joint positions relative to default
    pos_01 = [float(joint_pos_map.get(jn, 0.0) - default_pos_map.get(jn, 0.0)) for jn in slot01_names]
    pos_23 = [float(joint_pos_map.get(jn, 0.0) - default_pos_map.get(jn, 0.0)) for jn in slot23_names]
    pos_45 = [float(joint_pos_map.get(jn, 0.0) - default_pos_map.get(jn, 0.0)) for jn in slot45_names]

    # 7-9. Joint velocities scaled by 0.1
    vel_01 = [float(joint_vel_map.get(jn, 0.0) * JOINT_VEL_SCALE) for jn in slot01_names]
    vel_23 = [float(joint_vel_map.get(jn, 0.0) * JOINT_VEL_SCALE) for jn in slot23_names]
    vel_45 = [float(joint_vel_map.get(jn, 0.0) * JOINT_VEL_SCALE) for jn in slot45_names]

    # 10. Previous action
    act = np.asarray(prev_action, dtype=np.float32).reshape(ACTION_DIM)

    obs = np.concatenate([
        ANG_VEL_SCALE * omega_body,
        grav_body,
        cmd,
        pos_01,
        pos_23,
        pos_45,
        vel_01,
        vel_23,
        vel_45,
        act,
    ]).astype(np.float32)

    assert obs.shape == (OBS_DIM,), f"Observation vector dimension mismatch: {obs.shape} != ({OBS_DIM},)"
    return obs


def compute_desired_joint_targets(
    raw_actions: np.ndarray,
    policy_joints: Sequence[str],
    default_pos_map: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute q_des = q_default + 0.25 * a for all policy joints.
    """
    raw_act = np.asarray(raw_actions, dtype=np.float32).flatten()
    assert len(raw_act) == len(policy_joints) == ACTION_DIM

    targets: Dict[str, float] = {}
    for i, jn in enumerate(policy_joints):
        targets[jn] = float(default_pos_map.get(jn, 0.0) + ACTION_SCALE * float(raw_act[i]))
    return targets
