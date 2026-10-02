"""
vasimov/edge/policy_contract.py
Formal Policy Contract and Machine-Readable Manifest Specification for VASIMOV.

Defines the exact schema, observation/action specs, timing, and validation rules
required for any Asimov 1 policy to plug into the simulator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml


@dataclass
class RuntimeSpec:
    """Runtime engine requirements for policy execution."""
    type: str = "onnx"                          # "onnx", "torch", "custom"
    model_path: str = "policy.onnx"             # Relative to policy directory
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlSpec:
    """Timing and control frequency contracts."""
    frequency_hz: float = 50.0                  # Policy execution rate
    physics_dt: float = 0.005                   # Expected physics timestep
    decimation: int = 4                         # Physics steps per policy step

    def __post_init__(self):
        if self.physics_dt > 0.0 and self.frequency_hz > 0.0:
            expected_decim = round(1.0 / (self.frequency_hz * self.physics_dt))
            if expected_decim > 0:
                self.decimation = expected_decim


@dataclass
class TermSpec:
    """Specification of an individual observation term."""
    name: str = ""
    dim: int = 0
    scale: float = 1.0
    meaning: str = ""


@dataclass
class ObservationSpec:
    """Specification of the input observation tensor."""
    dim: int = 78
    shape: List[int] = field(default_factory=lambda: [1, 78])
    dtype: str = "float32"
    history_length: int = 1
    layout: str = "flat"                        # "flat", "term-major"
    terms: List[TermSpec] = field(default_factory=list)
    obs_joint_order: List[str] = field(default_factory=list)


@dataclass
class ActionSpec:
    """Specification of the output action tensor and joint mapping."""
    dim: int = 23
    shape: List[int] = field(default_factory=lambda: [1, 23])
    dtype: str = "float32"
    type: str = "joint_position"                # "joint_position", "torque", "delta"
    joint_order: List[str] = field(default_factory=list)
    scale: float = 0.25
    offset: str = "default_joint_pos"
    clip: Optional[Tuple[float, float]] = None


@dataclass
class PolicyManifest:
    """
    Complete, validated machine-readable policy manifest.
    Describes everything needed to load, validate, and execute an Asimov 1 policy.
    """
    name: str
    version: str = "1.0"
    category: str = "locomotion"                # "locomotion", "recovery", "balance", "composite"
    description: str = ""

    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    action: ActionSpec = field(default_factory=ActionSpec)

    default_joint_pos: Dict[str, float] = field(default_factory=dict)
    pd_gains: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    motor_model: str = "standard_pd"
    fall_trigger: Optional[Dict[str, Any]] = None
    requirements: Dict[str, List[str]] = field(default_factory=lambda: {
        "sensors": ["imu_ang_vel", "projected_gravity"],
        "state": ["joint_positions", "joint_velocities"],
    })
    adapter_class: Optional[str] = None
    base_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> PolicyManifest:
        """Parse dictionary into PolicyManifest dataclass."""
        runtime_d = data.get("runtime", {})
        runtime = RuntimeSpec(
            type=runtime_d.get("type", "onnx"),
            model_path=runtime_d.get("model_path", runtime_d.get("path", "policy.onnx")),
            options=runtime_d.get("options", {}),
        )

        ctrl_d = data.get("control", {})
        control = ControlSpec(
            frequency_hz=float(ctrl_d.get("frequency_hz", data.get("rate_hz", 50.0))),
            physics_dt=float(ctrl_d.get("physics_dt", data.get("physics_dt", 0.005))),
            decimation=int(ctrl_d.get("decimation", 4)),
        )

        obs_d = data.get("observation", {})
        raw_terms = obs_d.get("terms", [])
        terms = [
            TermSpec(
                name=t.get("name", ""),
                dim=int(t.get("dim", 0)),
                scale=float(t.get("scale", 1.0)),
                meaning=t.get("meaning", ""),
            )
            for t in raw_terms
        ]
        obs_shape = obs_d.get("shape", [1, int(obs_d.get("dim", 78))])
        observation = ObservationSpec(
            dim=int(obs_d.get("dim", 78)),
            shape=list(obs_shape),
            dtype=obs_d.get("dtype", "float32"),
            history_length=int(obs_d.get("history_length", 1)),
            layout=obs_d.get("layout", "flat"),
            terms=terms,
            obs_joint_order=list(obs_d.get("obs_joint_order", [])),
        )

        act_d = data.get("action", {})
        act_shape = act_d.get("shape", [1, int(act_d.get("dim", 23))])
        action = ActionSpec(
            dim=int(act_d.get("dim", 23)),
            shape=list(act_shape),
            dtype=act_d.get("dtype", "float32"),
            type=act_d.get("type", "joint_position"),
            joint_order=list(act_d.get("joint_order", [])),
            scale=float(act_d.get("scale", 0.25)),
            offset=str(act_d.get("offset", "default_joint_pos")),
            clip=tuple(act_d["clip"]) if act_d.get("clip") else None,
        )

        return cls(
            name=data["name"],
            version=str(data.get("version", "1.0")),
            category=data.get("category", "locomotion"),
            description=data.get("description", ""),
            runtime=runtime,
            control=control,
            observation=observation,
            action=action,
            default_joint_pos=dict(data.get("default_joint_pos", {})),
            pd_gains=dict(data.get("pd_gains", {})),
            motor_model=data.get("motor_model", "standard_pd"),
            fall_trigger=data.get("fall_trigger"),
            requirements=dict(data.get("requirements", {})),
            adapter_class=data.get("adapter_class"),
            base_dir=base_dir,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> PolicyManifest:
        """Load manifest from yaml or json file."""
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d, base_dir=p.parent)

    def resolve_model_path(self) -> Path:
        """Resolve absolute path to model artifact."""
        rel = Path(self.runtime.model_path)
        if rel.is_absolute():
            return rel
        if self.base_dir is not None:
            return (self.base_dir / rel).resolve()
        return rel.resolve()

    def validate(self) -> List[str]:
        """
        Validate manifest contract consistency. Returns list of error strings (empty if valid).
        """
        errors = []
        if not self.name:
            errors.append("Manifest 'name' is required.")

        # Check runtime
        if self.runtime.type not in ("onnx", "torch", "custom"):
            errors.append(f"Unsupported runtime type: '{self.runtime.type}'. Must be onnx, torch, or custom.")

        # Check model file exists
        mpath = self.resolve_model_path()
        if not mpath.exists():
            errors.append(f"Model artifact not found at: {mpath}")

        # Check action dimensions
        if self.action.dim != 23:
            errors.append(f"Expected action dim 23 for Asimov 1 humanoid, got {self.action.dim}")
        if self.action.joint_order and len(self.action.joint_order) != self.action.dim:
            errors.append(f"Action joint order length ({len(self.action.joint_order)}) != action dim ({self.action.dim})")

        # Check control frequency
        if self.control.frequency_hz <= 0:
            errors.append(f"Invalid control frequency: {self.control.frequency_hz} Hz")

        # Check observation dim matches terms if terms are declared
        if self.observation.terms:
            total_term_dim = sum(t.dim for t in self.observation.terms)
            expected_obs_dim = total_term_dim * self.observation.history_length
            if self.observation.dim != expected_obs_dim:
                errors.append(
                    f"Observation dim mismatch: manifest specifies {self.observation.dim}, "
                    f"but terms sum to {total_term_dim} * history {self.observation.history_length} = {expected_obs_dim}"
                )

        return errors
