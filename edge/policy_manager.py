"""
vasimov/edge/policy_manager.py
Policy Manager and Deterministic Multi-Policy Arbiter for VASIMOV.

Enforces deterministic single-policy actuation ownership, schedules control
decimation, and manages safe behavioral transitions (locomotion, recovery, balance).
"""

from __future__ import annotations
import enum
import logging
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from edge.policy_base import BasePolicy
from edge.policy_contract import PolicyManifest
from edge.policy_registry import PolicyRegistry
from edge.robot_command import UnifiedRobotCommand
from edge.robot_state import RobotState
from edge.safety import SafetyLayer

log = logging.getLogger("vasimov.edge.policy_manager")


class CompositionState(enum.Enum):
    """High-level behavior arbitration state in multi-policy composition."""
    LOCOMOTION = "locomotion"
    SAFEFALL = "safefall"
    GETUP = "getup"
    MANUAL = "manual"


class ArbitrationMode(enum.Enum):
    """Policy arbitration mode."""
    SINGLE = "single"
    AUTO_COMPOSE = "auto_compose"


class PolicyManager:
    """
    Coordinates, schedules, and arbitrates policies running on Virtual Asimov 1.
    """

    def __init__(
        self,
        registry: Optional[PolicyRegistry] = None,
        default_policy_name: str = "official_locomotion",
        safety: Optional[SafetyLayer] = None,
    ):
        self.registry = registry or PolicyRegistry()
        self.safety = safety or SafetyLayer()

        self._policies: Dict[str, BasePolicy] = {}
        self.active_policy: Optional[BasePolicy] = None
        self.active_policy_name: str = "none"

        # Multi-policy slots
        self.locomotion_policy: Optional[BasePolicy] = None
        self.safefall_policy: Optional[BasePolicy] = None
        self.getup_policy: Optional[BasePolicy] = None
        self.recovery_policy: Optional[BasePolicy] = None
        self.balance_policy: Optional[BasePolicy] = None

        # Composition & Arbitration
        self.auto_compose: bool = False
        self.comp_state: CompositionState = CompositionState.LOCOMOTION
        self._settle_timer_s: float = 0.0
        self._stood_timer_s: float = 0.0

        # Timing & Decimation
        self.last_step_sim_time: float = -1.0
        self.cached_command: Optional[UnifiedRobotCommand] = None

        # Load default policy if available
        try:
            self.load_policy(default_policy_name)
        except Exception as e:
            log.warning("[POLICY MANAGER] Could not load default policy '%s': %s", default_policy_name, e)

    @property
    def mode(self) -> ArbitrationMode:
        """Arbitration mode compatibility property."""
        return ArbitrationMode.AUTO_COMPOSE if self.auto_compose else ArbitrationMode.SINGLE

    @mode.setter
    def mode(self, val: Any) -> None:
        """Allow setting mode via ArbitrationMode or boolean."""
        if isinstance(val, ArbitrationMode):
            self.auto_compose = (val == ArbitrationMode.AUTO_COMPOSE)
        elif isinstance(val, str):
            self.auto_compose = (val.lower() in ("auto_compose", "compose", "multi"))
        else:
            self.auto_compose = bool(val)

    def load_policy(self, name_or_path: str, as_active: bool = True) -> BasePolicy:
        """Load and instantiate a policy into the manager."""
        policy = self.registry.instantiate(name_or_path, safety=self.safety)
        self._policies[policy.name] = policy

        pname = policy.name.lower()
        if policy.category == "locomotion":
            self.locomotion_policy = policy
        elif pname in ("getup_safefall", "openhorizon_recovery"):
            self.recovery_policy = policy
        elif pname in ("safefall", "openhorizon_safefall") or (hasattr(policy.manifest.runtime, "model_path") and "safefall" in str(policy.manifest.runtime.model_path) and "getup" not in pname):
            self.safefall_policy = policy
            if self.recovery_policy is None:
                self.recovery_policy = policy
        elif pname in ("getup", "openhorizon_getup"):
            self.getup_policy = policy
            if self.recovery_policy is None:
                self.recovery_policy = policy
        elif policy.category == "recovery":
            self.recovery_policy = policy
        elif policy.category == "balance":
            self.balance_policy = policy

        if as_active or self.active_policy is None:
            self.set_active_policy(policy.name)

        log.info("[POLICY MANAGER] Loaded policy '%s' (category=%s, freq=%.1f Hz)", policy.name, policy.category, policy.manifest.control.frequency_hz)
        return policy

    def get_active_policy(self) -> Optional[BasePolicy]:
        """Return currently active policy instance."""
        return self.active_policy

    def register_policy(self, policy: BasePolicy, category: Optional[str] = None) -> None:
        """Register an instantiated policy instance."""
        self._policies[policy.name] = policy
        cat = category or policy.category
        pname = policy.name.lower()
        if cat == "locomotion":
            self.locomotion_policy = policy
        elif pname in ("getup_safefall", "openhorizon_recovery"):
            self.recovery_policy = policy
        elif pname in ("safefall", "openhorizon_safefall"):
            self.safefall_policy = policy
            if self.recovery_policy is None:
                self.recovery_policy = policy
        elif pname in ("getup", "openhorizon_getup"):
            self.getup_policy = policy
            if self.recovery_policy is None:
                self.recovery_policy = policy
        elif cat == "recovery":
            self.recovery_policy = policy
        elif cat == "balance":
            self.balance_policy = policy

    def set_active_policy(self, name: str) -> None:
        """Switch active policy for deterministic single-ownership control."""
        if name not in self._policies:
            # Try to load from registry
            self.load_policy(name, as_active=False)

        self.active_policy = self._policies[name]
        self.active_policy_name = name
        self.active_policy.reset()
        self.cached_command = None
        log.info("[POLICY MANAGER] Active policy switched to '%s'", name)

    def configure_multi_policy(
        self,
        locomotion: Optional[str] = None,
        recovery: Optional[str] = None,
        safefall: Optional[str] = None,
        getup: Optional[str] = None,
        balance: Optional[str] = None,
        auto_compose: bool = False,
    ) -> None:
        """Configure policies for multi-policy composition and arbitration."""
        if locomotion:
            self.locomotion_policy = self.load_policy(locomotion, as_active=False)

        # Handle recovery suite or individual recovery policies
        if recovery:
            if recovery in ("openhorizon_recovery", "getup_safefall", "openhorizon"):
                self.safefall_policy = self.load_policy("openhorizon_safefall", as_active=False)
                self.getup_policy = self.load_policy("openhorizon_getup", as_active=False)
                self.recovery_policy = self.load_policy("openhorizon_recovery", as_active=False)
            elif "safefall" in recovery:
                self.safefall_policy = self.load_policy(recovery, as_active=False)
            elif "getup" in recovery:
                self.getup_policy = self.load_policy(recovery, as_active=False)
            else:
                self.recovery_policy = self.load_policy(recovery, as_active=False)

        if safefall:
            self.safefall_policy = self.load_policy(safefall, as_active=False)
        if getup:
            self.getup_policy = self.load_policy(getup, as_active=False)
        if balance:
            self.balance_policy = self.load_policy(balance, as_active=False)

        self.auto_compose = auto_compose
        if auto_compose and self.locomotion_policy:
            self.active_policy = self.locomotion_policy
            self.active_policy_name = self.locomotion_policy.name
            self.comp_state = CompositionState.LOCOMOTION

    def reset(self) -> None:
        """Reset all instantiated policies and arbitration timers."""
        for p in self._policies.values():
            p.reset()
        self.last_step_sim_time = -1.0
        self.cached_command = None
        self.comp_state = CompositionState.LOCOMOTION
        self._settle_timer_s = 0.0
        self._stood_timer_s = 0.0

    def _arbitrate_composition(self, state: RobotState) -> BasePolicy:
        """
        Deterministic state machine arbitrating policy ownership:
        NORMAL (Official locomotion) -> (tilt > 30° or |w| > 2.0 rad/s)
            -> SAFE_FALL (OpenHorizon Safe-Fall, safefall.onnx) -> (settled on ground)
            -> GET_UP (OpenHorizon Get-Up, getup.onnx) -> (stable standing upright)
            -> NORMAL (Official locomotion)
        """
        safefall_pol = self.safefall_policy or self.recovery_policy
        getup_pol = self.getup_policy or self.recovery_policy

        if not self.auto_compose or not (safefall_pol or getup_pol) or not self.locomotion_policy:
            return self.active_policy or self.locomotion_policy

        w_sq = float(np.dot(state.base_ang_vel_body, state.base_ang_vel_body))
        tilt = state.tilt_deg

        if self.comp_state == CompositionState.LOCOMOTION:
            # Fall trigger: tilt > 30 deg or high angular rate > 2.0 rad/s
            if (tilt > 30.0 or w_sq > 4.0) and not state.gantry_active:
                log.info(
                    "[ARBITER] Fall detected (tilt=%.1f deg, |w|=%.2f rad/s). Handover: LOCOMOTION -> SAFEFALL",
                    tilt, math.sqrt(w_sq)
                )
                self.comp_state = CompositionState.SAFEFALL
                if safefall_pol:
                    safefall_pol.reset()
                    if hasattr(safefall_pol, "set_subpolicy"):
                        safefall_pol.set_subpolicy("safefall")
                    if hasattr(safefall_pol.adapter, "fall_triggered"):
                        safefall_pol.adapter.fall_triggered = True
                self._settle_timer_s = 0.0
                self.active_policy = safefall_pol
                self.active_policy_name = safefall_pol.name if safefall_pol else "safefall"
                return safefall_pol or self.locomotion_policy
            return self.locomotion_policy

        elif self.comp_state == CompositionState.SAFEFALL:
            # Settled check: robot is low and nearly still on ground
            lin_speed = float(np.linalg.norm(state.base_lin_vel_world))
            ang_speed = float(np.linalg.norm(state.base_ang_vel_body))
            is_grounded = (state.pelvis_pos[2] < 0.35) or (tilt > 50.0)
            if is_grounded and lin_speed < 0.25 and ang_speed < 0.60:
                self._settle_timer_s += state.dt
            else:
                self._settle_timer_s = 0.0

            if self._settle_timer_s >= 0.25:
                log.info(
                    "[ARBITER] Robot settled on ground (tilt=%.1f deg, z=%.2fm). Handover: SAFEFALL -> GETUP",
                    tilt, state.pelvis_pos[2]
                )
                self.comp_state = CompositionState.GETUP
                if getup_pol:
                    getup_pol.reset()
                    if hasattr(getup_pol, "set_subpolicy"):
                        getup_pol.set_subpolicy("getup")
                self._stood_timer_s = 0.0
                self.active_policy = getup_pol
                self.active_policy_name = getup_pol.name if getup_pol else "getup"
                return getup_pol or safefall_pol or self.locomotion_policy

            self.active_policy = safefall_pol
            self.active_policy_name = safefall_pol.name if safefall_pol else "safefall"
            return safefall_pol or self.locomotion_policy

        elif self.comp_state == CompositionState.GETUP:
            # Stood check: upright (gz < -0.85) and pelvis height > 0.55m
            is_upright = (state.projected_gravity[2] < -0.85) and (state.pelvis_pos[2] > 0.55)
            if is_upright:
                self._stood_timer_s += state.dt
            else:
                self._stood_timer_s = 0.0

            if self._stood_timer_s >= 0.80:
                log.info(
                    "[ARBITER] Robot recovered standing balance (z=%.2fm, gz=%.2f). Handover: GETUP -> LOCOMOTION",
                    state.pelvis_pos[2], state.projected_gravity[2]
                )
                self.comp_state = CompositionState.LOCOMOTION
                self.locomotion_policy.reset()
                self.active_policy = self.locomotion_policy
                self.active_policy_name = self.locomotion_policy.name
                return self.locomotion_policy

            self.active_policy = getup_pol
            self.active_policy_name = getup_pol.name if getup_pol else "getup"
            return getup_pol or self.locomotion_policy

        return self.active_policy or self.locomotion_policy

    def step(self, state: RobotState) -> Optional[UnifiedRobotCommand]:
        """
        Execute one control cycle through the arbitrated policy.
        Respects policy control decimation (e.g. 50 Hz rate on 200 Hz physics).
        """
        # Select active policy (via single ownership or auto arbitration)
        pol = self._arbitrate_composition(state) if self.auto_compose else self.active_policy
        if pol is None:
            return None

        # Check policy frequency decimation
        target_period = 1.0 / pol.manifest.control.frequency_hz
        if self.last_step_sim_time >= 0.0 and (state.sim_time - self.last_step_sim_time) < (target_period - 0.0001):
            return self.cached_command

        self.last_step_sim_time = state.sim_time
        cmd = pol.step(state)
        self.cached_command = cmd
        return cmd

    def get_telemetry(self) -> Dict[str, Any]:
        """Telemetry summary for web dashboard and diagnostics."""
        pol = self.active_policy
        t = pol.get_telemetry() if pol else {}
        t["manager_active_policy"] = self.active_policy_name
        t["auto_compose"] = self.auto_compose
        t["composition_state"] = self.comp_state.value
        t["loaded_policies"] = list(self._policies.keys())
        return t
