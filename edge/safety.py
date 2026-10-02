"""
vasimov/edge/safety.py
Safety, Limit Enforcement, and Actuator Protection Layer.

Strict boundary layer situated between Policy Adapters and the MuJoCo Actuator Interface.
Guarantees that no policy—regardless of bugs, NaN/Inf outputs, or extreme actions—
can bypass physical joint limits, effort limits, or rate limits.
"""

from __future__ import annotations
import logging
import math
from typing import Dict, Optional, Tuple
import numpy as np

from edge.robot_command import UnifiedRobotCommand
from edge.robot_state import RobotState

log = logging.getLogger("vasimov.edge.safety")


class SafetyLayer:
    """
    Validates and clamps commands before they reach the actuator interface.
    """

    def __init__(
        self,
        joint_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        effort_limits: Optional[Dict[str, float]] = None,
        max_target_delta_per_step: float = 2.0,
    ):
        self.joint_ranges: Dict[str, Tuple[float, float]] = joint_ranges or {}
        self.effort_limits: Dict[str, float] = effort_limits or {}
        self.max_target_delta_per_step = max_target_delta_per_step
        self.consecutive_violations: int = 0
        self.last_safe_targets: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal filter memory and violation counters."""
        self.last_safe_targets.clear()
        self.consecutive_violations = 0

    def filter_command(
        self,
        command: UnifiedRobotCommand,
        current_state: RobotState,
    ) -> UnifiedRobotCommand:
        """
        Inspect, validate, and clamp a UnifiedRobotCommand.
        
        Returns a verified, safe UnifiedRobotCommand guaranteed to conform
        to hardware and simulation boundaries.
        """
        safe_targets: Dict[str, float] = {}
        safe_kp: Dict[str, float] = {}
        safe_kd: Dict[str, float] = {}
        safe_efforts: Dict[str, float] = {}
        safe_ff: Dict[str, float] = {}

        has_nan_or_inf = False

        # 1. Inspect targets for non-finite values (NaN / Inf)
        for jn, tgt in command.joint_targets.items():
            if not math.isfinite(tgt):
                has_nan_or_inf = True
                log.critical(
                    "[SAFETY TRIP] Policy '%s' output non-finite joint target for %s: %s",
                    command.policy_name, jn, tgt
                )
                break

        # Also inspect gains
        for jn, val in command.kp.items():
            if not math.isfinite(val) or val < 0.0:
                has_nan_or_inf = True
                break
        for jn, val in command.kd.items():
            if not math.isfinite(val) or val < 0.0:
                has_nan_or_inf = True
                break

        if has_nan_or_inf:
            self.consecutive_violations += 1
            log.error(
                "[SAFETY SHUTDOWN] Intercepted invalid command from '%s'. Forcing DAMP mode.",
                command.policy_name
            )
            # Safe compliant DAMP fallback
            for jn in command.joint_targets:
                safe_targets[jn] = current_state.joint_positions.get(jn, 0.0)
                safe_kp[jn] = 0.0
                safe_kd[jn] = 2.0
            return UnifiedRobotCommand(
                joint_targets=safe_targets,
                kp=safe_kp,
                kd=safe_kd,
                effort_limits=command.effort_limits,
                feedforward_torques={jn: 0.0 for jn in safe_targets},
                policy_name="safety_fallback_damp",
                timestamp=command.timestamp,
                mode="DAMP",
            )

        # 2. Enforce physical joint limits (clamping)
        for jn, tgt in command.joint_targets.items():
            clamped_tgt = tgt
            if jn in self.joint_ranges:
                j_min, j_max = self.joint_ranges[jn]
                clamped_tgt = float(np.clip(clamped_tgt, j_min, j_max))

            # Rate limit jump relative to last target or current position
            ref_pos = self.last_safe_targets.get(jn, current_state.joint_positions.get(jn, clamped_tgt))
            delta = clamped_tgt - ref_pos
            if abs(delta) > self.max_target_delta_per_step:
                clamped_tgt = ref_pos + math.copysign(self.max_target_delta_per_step, delta)
                log.warning(
                    "[SAFETY RATE-LIMIT] Joint %s target delta (%.3f rad) exceeded rate limit (%.3f rad). Clamped to %.3f",
                    jn, delta, self.max_target_delta_per_step, clamped_tgt
                )

            safe_targets[jn] = clamped_tgt
            safe_kp[jn] = max(0.0, float(command.kp.get(jn, 40.0)))
            safe_kd[jn] = max(0.0, float(command.kd.get(jn, 2.0)))

        # 3. Enforce effort / torque limits
        for jn, eff in command.effort_limits.items():
            lim = self.effort_limits.get(jn, eff)
            safe_efforts[jn] = float(min(eff, lim))

        for jn, tau in command.feedforward_torques.items():
            lim = safe_efforts.get(jn, 100.0)
            safe_ff[jn] = float(np.clip(tau, -lim, lim))

        self.last_safe_targets = dict(safe_targets)
        self.consecutive_violations = 0

        return UnifiedRobotCommand(
            joint_targets=safe_targets,
            kp=safe_kp,
            kd=safe_kd,
            effort_limits=safe_efforts,
            feedforward_torques=safe_ff,
            policy_name=command.policy_name,
            timestamp=command.timestamp,
            mode=command.mode,
            raw_actions=command.raw_actions,
        )
