#!/usr/bin/env python3
"""
vasimov/edge/gestures.py
Procedural Emote and Gesture Engine for Virtual Asimov 1.

Provides smooth, kinematic trajectory interpolation for humanoid robot emotes:
- hello / wave    : Raises right arm and waves hand side-to-side (3 cycles)
- bow             : Polite humanoid bow (torso + neck pitch down and return)
- squat / crouch  : Controlled knee & hip flexion with arms balancing forward
- cheer           : Double-arm overhead victory raise and celebratory hold
- dance           : Rhythmic waist yaw oscillation with alternating arm sways
- nod             : Head pitch nods up and down (affirmation gesture)
- shake           : Head yaw turns side to side (negation gesture)

All trajectories use smooth Hermite / sinusoidal easing to prevent torque spikes.
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple
from adapter import SIM_JOINTS

# Smooth ease in-out curve: s-curve from 0 to 1 for x in [0, 1]
def smooth_step(x: float) -> float:
    t = max(0.0, min(1.0, x))
    return t * t * (3.0 - 2.0 * t)


class EmoteDefinition:
    """Specification of an emote animation."""

    def __init__(
        self,
        name: str,
        description: str,
        duration: float,
        trajectory_fn: Callable[[float, Dict[str, float]], Dict[str, float]],
    ):
        self.name = name
        self.description = description
        self.duration = duration
        self.trajectory_fn = trajectory_fn

    def get_targets(self, elapsed: float, defaults: Dict[str, float]) -> Dict[str, float]:
        """Compute joint targets for time elapsed in [0, duration]."""
        return self.trajectory_fn(elapsed, defaults)


# ── Trajectory Generators ────────────────────────────────────────────────────

def _hello_wave_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Right arm raises, waves hand side-to-side, then lowers."""
    targets = dict(defs)
    # Phase 1: 0.0 - 0.7s: Raise arm
    # Phase 2: 0.7 - 2.3s: Wave wrist/shoulder
    # Phase 3: 2.3 - 3.0s: Lower arm
    t_tot = 3.0
    if t < 0.7:
        u = smooth_step(t / 0.7)
    elif t > 2.3:
        u = smooth_step((t_tot - t) / 0.7)
    else:
        u = 1.0

    # Arm raise angles:
    # right_shoulder_pitch_joint: default ~0.25 -> 1.30 rad
    # right_shoulder_roll_joint:  default ~0.05 -> 0.45 rad
    # right_elbow_joint:          default ~ -0.40 -> -1.45 rad
    targets["right_shoulder_pitch_joint"] = defs.get("right_shoulder_pitch_joint", 0.25) + u * (1.30 - defs.get("right_shoulder_pitch_joint", 0.25))
    targets["right_shoulder_roll_joint"] = defs.get("right_shoulder_roll_joint", 0.05) + u * (0.45 - defs.get("right_shoulder_roll_joint", 0.05))
    targets["right_elbow_joint"] = defs.get("right_elbow_joint", -0.40) + u * (-1.45 - defs.get("right_elbow_joint", -0.40))

    # Waving oscillation in phase 2
    if 0.7 <= t <= 2.3:
        wave_phase = (t - 0.7) * (2.0 * math.pi * 2.0)  # 2.0 Hz
        targets["right_wrist_yaw_joint"] = defs.get("right_wrist_yaw_joint", 0.0) + 0.50 * math.sin(wave_phase)
        targets["right_shoulder_yaw_joint"] = defs.get("right_shoulder_yaw_joint", 0.0) + 0.15 * math.sin(wave_phase)
    else:
        targets["right_wrist_yaw_joint"] = defs.get("right_wrist_yaw_joint", 0.0)

    # Slight head tilt towards wave
    targets["neck_yaw_joint"] = defs.get("neck_yaw_joint", 0.0) + u * 0.15
    return targets


def _bow_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Respectful humanoid bow."""
    # 0.0 -> 1.0s: Lean forward
    # 1.0 -> 2.0s: Hold bow
    # 2.0 -> 3.0s: Return upright
    t_tot = 3.0
    if t < 1.0:
        u = smooth_step(t / 1.0)
    elif t > 2.0:
        u = smooth_step((t_tot - t) / 1.0)
    else:
        u = 1.0

    targets = dict(defs)
    # Hip pitch forward: left is negative forward, right is positive forward
    targets["left_hip_pitch_joint"] = defs.get("left_hip_pitch_joint", -0.15) - u * 0.25
    targets["right_hip_pitch_joint"] = defs.get("right_hip_pitch_joint", 0.15) + u * 0.25
    # Ankle pitch compensation
    targets["left_ankle_pitch_joint"] = defs.get("left_ankle_pitch_joint", -0.30) + u * 0.15
    targets["right_ankle_pitch_joint"] = defs.get("right_ankle_pitch_joint", 0.30) - u * 0.15
    # Neck pitch down
    targets["neck_pitch_joint"] = defs.get("neck_pitch_joint", 0.0) + u * 0.30
    # Arms slightly pinned back
    targets["left_shoulder_pitch_joint"] = defs.get("left_shoulder_pitch_joint", -0.25) + u * 0.15
    targets["right_shoulder_pitch_joint"] = defs.get("right_shoulder_pitch_joint", 0.25) - u * 0.15
    return targets


def _squat_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Controlled humanoid squat / crouch and stand up."""
    # 0.0 -> 1.2s: Descend into squat
    # 1.2 -> 2.2s: Hold squat
    # 2.2 -> 3.5s: Ascend back to stand
    t_tot = 3.5
    if t < 1.2:
        u = smooth_step(t / 1.2)
    elif t > 2.2:
        u = smooth_step((t_tot - t) / 1.3)
    else:
        u = 1.0

    targets = dict(defs)
    # Flex knees: left is positive, right is negative
    targets["left_knee_joint"] = defs.get("left_knee_joint", 0.45) + u * 0.65
    targets["right_knee_joint"] = defs.get("right_knee_joint", -0.45) - u * 0.65
    # Pitch hips forward to balance center of mass
    targets["left_hip_pitch_joint"] = defs.get("left_hip_pitch_joint", -0.15) - u * 0.45
    targets["right_hip_pitch_joint"] = defs.get("right_hip_pitch_joint", 0.15) + u * 0.45
    # Ankles dorsiflex
    targets["left_ankle_pitch_joint"] = defs.get("left_ankle_pitch_joint", -0.30) - u * 0.05
    targets["right_ankle_pitch_joint"] = defs.get("right_ankle_pitch_joint", 0.30) + u * 0.05
    # Arms forward for balance: left shoulder pitch forward (more negative), right shoulder pitch forward (more positive)
    targets["left_shoulder_pitch_joint"] = defs.get("left_shoulder_pitch_joint", -0.25) - u * 0.60
    targets["right_shoulder_pitch_joint"] = defs.get("right_shoulder_pitch_joint", 0.25) + u * 0.60
    targets["left_elbow_joint"] = defs.get("left_elbow_joint", 0.40) + u * 0.20
    targets["right_elbow_joint"] = defs.get("right_elbow_joint", -0.40) - u * 0.20
    return targets


def _cheer_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Both arms raised overhead in victory cheer."""
    # 0.0 -> 0.8s: Raise arms
    # 0.8 -> 2.2s: Hold cheer / pulse
    # 2.2 -> 3.0s: Lower arms
    t_tot = 3.0
    if t < 0.8:
        u = smooth_step(t / 0.8)
    elif t > 2.2:
        u = smooth_step((t_tot - t) / 0.8)
    else:
        u = 1.0

    targets = dict(defs)
    # Left arm up: pitch -1.5, roll -0.6, elbow 0.8
    targets["left_shoulder_pitch_joint"] = defs.get("left_shoulder_pitch_joint", -0.25) + u * (-1.50 - defs.get("left_shoulder_pitch_joint", -0.25))
    targets["left_shoulder_roll_joint"] = defs.get("left_shoulder_roll_joint", -0.05) + u * (-0.60 - defs.get("left_shoulder_roll_joint", -0.05))
    targets["left_elbow_joint"] = defs.get("left_elbow_joint", 0.40) + u * (0.80 - defs.get("left_elbow_joint", 0.40))

    # Right arm up: pitch 1.5, roll 0.6, elbow -0.8
    targets["right_shoulder_pitch_joint"] = defs.get("right_shoulder_pitch_joint", 0.25) + u * (1.50 - defs.get("right_shoulder_pitch_joint", 0.25))
    targets["right_shoulder_roll_joint"] = defs.get("right_shoulder_roll_joint", 0.05) + u * (0.60 - defs.get("right_shoulder_roll_joint", 0.05))
    targets["right_elbow_joint"] = defs.get("right_elbow_joint", -0.40) + u * (-0.80 - defs.get("right_elbow_joint", -0.40))

    # Slight joyful pump in hold phase
    if 0.8 <= t <= 2.2:
        pump = 0.15 * math.sin((t - 0.8) * 2.0 * math.pi * 1.5)
        targets["left_elbow_joint"] += pump
        targets["right_elbow_joint"] -= pump
        targets["neck_pitch_joint"] = defs.get("neck_pitch_joint", 0.0) - 0.15  # look up
    return targets


def _dance_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Rhythmic waist and arm dance sway."""
    t_tot = 4.0
    u = smooth_step(t / 0.5) if t < 0.5 else (smooth_step((t_tot - t) / 0.5) if t > 3.5 else 1.0)

    targets = dict(defs)
    freq = 1.0  # 1 Hz sway
    phase = 2.0 * math.pi * freq * t
    sway = math.sin(phase) * u

    targets["waist_yaw_joint"] = defs.get("waist_yaw_joint", 0.0) + 0.35 * sway
    targets["neck_yaw_joint"] = defs.get("neck_yaw_joint", 0.0) - 0.20 * sway
    # Arms alternate
    targets["left_shoulder_pitch_joint"] = defs.get("left_shoulder_pitch_joint", -0.25) - 0.30 * sway
    targets["right_shoulder_pitch_joint"] = defs.get("right_shoulder_pitch_joint", 0.25) - 0.30 * sway
    targets["left_elbow_joint"] = defs.get("left_elbow_joint", 0.40) + 0.30 * sway
    targets["right_elbow_joint"] = defs.get("right_elbow_joint", -0.40) + 0.30 * sway
    return targets


def _nod_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Affirmative head nod."""
    t_tot = 2.0
    targets = dict(defs)
    if t <= t_tot:
        # 2 cycles of nod
        nod = 0.25 * math.sin(2.0 * math.pi * 1.0 * t)
        targets["neck_pitch_joint"] = defs.get("neck_pitch_joint", 0.0) + nod
    return targets


def _shake_trajectory(t: float, defs: Dict[str, float]) -> Dict[str, float]:
    """Negation head shake."""
    t_tot = 2.0
    targets = dict(defs)
    if t <= t_tot:
        # 2 cycles of head shake
        shake = 0.35 * math.sin(2.0 * math.pi * 1.0 * t)
        targets["neck_yaw_joint"] = defs.get("neck_yaw_joint", 0.0) + shake
    return targets


# ── Emote Catalog ────────────────────────────────────────────────────────────

EMOTE_CATALOG: Dict[str, EmoteDefinition] = {
    "hello": EmoteDefinition("hello", "Wave right hand in greeting", 3.0, _hello_wave_trajectory),
    "wave": EmoteDefinition("wave", "Wave right hand in greeting (alias of hello)", 3.0, _hello_wave_trajectory),
    "bow": EmoteDefinition("bow", "Polite humanoid bow with tilted head", 3.0, _bow_trajectory),
    "squat": EmoteDefinition("squat", "Deep crouch / squat and smooth return", 3.5, _squat_trajectory),
    "crouch": EmoteDefinition("crouch", "Deep crouch / squat (alias of squat)", 3.5, _squat_trajectory),
    "cheer": EmoteDefinition("cheer", "Triumphant double-arm overhead victory cheer", 3.0, _cheer_trajectory),
    "dance": EmoteDefinition("dance", "Rhythmic waist sway and arm groove", 4.0, _dance_trajectory),
    "nod": EmoteDefinition("nod", "Affirmative head nod", 2.0, _nod_trajectory),
    "shake": EmoteDefinition("shake", "Head shake side to side", 2.0, _shake_trajectory),
}


class EmoteController:
    """Manages active emote animation playback and blending."""

    def __init__(self, default_pose_sim: Sequence[float]):
        self.default_dict = {
            SIM_JOINTS[i]: float(default_pose_sim[i]) for i in range(len(SIM_JOINTS))
        }
        self.active_emote: Optional[EmoteDefinition] = None
        self.elapsed: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.active_emote is not None

    @property
    def current_emote_name(self) -> Optional[str]:
        return self.active_emote.name if self.active_emote else None

    @property
    def progress(self) -> float:
        if not self.active_emote or self.active_emote.duration <= 0.0:
            return 0.0
        return min(1.0, self.elapsed / self.active_emote.duration)

    def start_emote(self, name: str) -> bool:
        """Start playing an emote by name."""
        key = name.lower().strip()
        if key not in EMOTE_CATALOG:
            return False
        self.active_emote = EMOTE_CATALOG[key]
        self.elapsed = 0.0
        return True

    def stop_emote(self) -> None:
        """Immediately cancel active emote."""
        self.active_emote = None
        self.elapsed = 0.0

    def step(self, dt: float) -> Optional[Dict[str, float]]:
        """Advance emote time and return overridden joint targets."""
        if self.active_emote is None:
            return None

        self.elapsed += dt
        targets = self.active_emote.get_targets(self.elapsed, self.default_dict)

        if self.elapsed >= self.active_emote.duration:
            self.stop_emote()

        return targets

    @staticmethod
    def list_emotes() -> List[Dict[str, Any]]:
        """List all available emotes with descriptions and durations."""
        out = []
        seen = set()
        for k, v in EMOTE_CATALOG.items():
            if v.name not in seen:
                seen.add(v.name)
                out.append({
                    "name": v.name,
                    "description": v.description,
                    "duration": v.duration,
                })
        return out
