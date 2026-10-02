"""
vasimov/edge/policy_base.py
Standardized Policy Instance and Execution Interface.
Combines Manifest, Runtime, Adapter, and Safety enforcement into a cohesive controller.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional
import numpy as np

from edge.policy_contract import PolicyManifest
from edge.policy_runtime import BasePolicyRuntime, ONNXPolicyRuntime
from edge.policy_adapter import BasePolicyAdapter, OfficialLocomotionAdapter, GetupSafefallAdapter
from edge.robot_command import UnifiedRobotCommand
from edge.robot_state import RobotState
from edge.safety import SafetyLayer

log = logging.getLogger("vasimov.edge.policy_base")


class BasePolicy:
    """Standardized runnable policy interface for VASIMOV."""

    def __init__(
        self,
        manifest: PolicyManifest,
        runtime: Optional[BasePolicyRuntime] = None,
        adapter: Optional[BasePolicyAdapter] = None,
        safety: Optional[SafetyLayer] = None,
    ):
        self.manifest = manifest
        self.name = manifest.name
        self.category = manifest.category

        # Runtime
        if runtime is not None:
            self.runtime = runtime
        else:
            mpath = manifest.resolve_model_path()
            self.runtime = ONNXPolicyRuntime(model_path=mpath)

        # Adapter
        if adapter is not None:
            self.adapter = adapter
        elif "getup" in manifest.name or "safefall" in manifest.name or manifest.observation.dim == 375:
            self.adapter = GetupSafefallAdapter(manifest)
        else:
            self.adapter = OfficialLocomotionAdapter(manifest)

        # Safety
        self.safety = safety or SafetyLayer()

        # Telemetry & Performance
        self.last_inference_time_ms: float = 0.0
        self.last_step_sim_time: float = -1.0
        self.current_observation: Optional[np.ndarray] = None
        self.current_actions: Optional[np.ndarray] = None
        self.step_counter: int = 0

    @property
    def policy_joints(self) -> List[str]:
        return list(self.manifest.action.joint_order)

    @property
    def default_pos(self) -> Dict[str, float]:
        return dict(self.manifest.default_joint_pos)

    @property
    def joint_gains(self) -> Dict[str, Tuple[float, float, float]]:
        return getattr(self.adapter, "joint_gains", {})

    @property
    def slot01_names(self) -> List[str]:
        return getattr(self.adapter, "slot01_names", [])

    @property
    def slot23_names(self) -> List[str]:
        return getattr(self.adapter, "slot23_names", [])

    @property
    def slot45_names(self) -> List[str]:
        return getattr(self.adapter, "slot45_names", [])

    def reset(self) -> None:
        """Reset adapter history, runtime state, and telemetry counters."""
        self.adapter.reset()
        self.safety.reset()
        self.last_step_sim_time = -1.0
        self.current_observation = None
        self.current_actions = None
        self.step_counter = 0

    def step(self, state: RobotState) -> UnifiedRobotCommand:
        """
        Execute one policy control step:
        1. Construct observation tensor from RobotState
        2. Execute model inference
        3. Convert raw actions into UnifiedRobotCommand
        4. Validate and clamp through SafetyLayer
        """
        t0 = time.perf_counter()

        # 1. Observation
        obs = self.adapter.build_observation(state)
        self.current_observation = obs

        # 2. Inference
        outputs = self.runtime.run({"obs": obs})
        raw_act = outputs.get("actions")
        if raw_act is None:
            raw_act = next(iter(outputs.values()))
        self.current_actions = raw_act

        # 3. Action adaptation
        cmd = self.adapter.process_action(raw_act, state)

        # 4. Safety filtering
        safe_cmd = self.safety.filter_command(cmd, state)

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_step_sim_time = state.sim_time
        self.step_counter += 1

        return safe_cmd

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time diagnostic metrics for flight recorder / web telemetry."""
        obs_raw = [float(x) for x in self.current_observation.flatten()] if self.current_observation is not None else []
        act_raw = [float(x) for x in self.current_actions.flatten()] if self.current_actions is not None else []

        return {
            "name": self.name,
            "category": self.category,
            "frequency_hz": self.manifest.control.frequency_hz,
            "inference_time_ms": round(self.last_inference_time_ms, 2),
            "step_count": self.step_counter,
            "obs_dim": self.manifest.observation.dim,
            "act_dim": self.manifest.action.dim,
            "obs_preview": obs_raw[:10],
            "raw_actions": act_raw,
            "obs_norm": float(np.linalg.norm(obs_raw)) if obs_raw else 0.0,
            "act_norm": float(np.linalg.norm(act_raw)) if act_raw else 0.0,
        }
