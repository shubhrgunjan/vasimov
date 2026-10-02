"""
vasimov/edge/policy_registry.py
Policy Discovery, Registry, and Formal Validation System.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

from edge.policy_contract import PolicyManifest
from edge.policy_base import BasePolicy
from edge.policy_runtime import ONNXPolicyRuntime
from edge.safety import SafetyLayer

log = logging.getLogger("vasimov.edge.policy_registry")


class PolicyRegistry:
    """
    Central registry and discovery engine for Asimov 1 policies.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        self.policies_dir = self.root_dir / "policies"
        self._manifests: Dict[str, PolicyManifest] = {}
        self.discover_policies()

    def discover_policies(self, extra_paths: Optional[Sequence[Path]] = None) -> None:
        """Scan policies/ directory and any additional paths for policy manifests."""
        search_dirs = [self.policies_dir]
        if extra_paths:
            search_dirs.extend([Path(p) for p in extra_paths])

        for sdir in search_dirs:
            if not sdir.exists():
                continue

            # Look for manifest.yaml or *_policy_spec.json
            for mpath in sdir.glob("**/manifest.yaml"):
                try:
                    manifest = PolicyManifest.from_yaml(mpath)
                    self._manifests[manifest.name] = manifest
                    log.debug("[REGISTRY] Discovered policy '%s' at %s", manifest.name, mpath)
                except Exception as e:
                    log.warning("[REGISTRY] Failed to load manifest at %s: %s", mpath, e)

            # Also check for JSON specs
            for jpath in sdir.glob("**/*_policy_spec.json"):
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    p_name = jpath.stem.replace("_policy_spec", "")
                    if p_name not in self._manifests:
                        data["name"] = p_name
                        manifest = PolicyManifest.from_dict(data, base_dir=jpath.parent)
                        self._manifests[p_name] = manifest
                        log.debug("[REGISTRY] Discovered spec policy '%s' at %s", p_name, jpath)
                except Exception as e:
                    log.debug("[REGISTRY] Could not parse json spec at %s: %s", jpath, e)

    discover = discover_policies

    def register_manifest(self, manifest: PolicyManifest) -> None:
        """Register a policy manifest programmatically."""
        self._manifests[manifest.name] = manifest

    def get_manifest(self, name_or_path: str) -> PolicyManifest:
        """Retrieve manifest by registered name or direct directory/file path."""
        if name_or_path in self._manifests:
            return self._manifests[name_or_path]

        # Check if direct file/dir path provided
        p = Path(name_or_path).resolve()
        if p.is_file() and p.name in ("manifest.yaml", "manifest.yml"):
            return PolicyManifest.from_yaml(p)
        if p.is_dir() and (p / "manifest.yaml").exists():
            return PolicyManifest.from_yaml(p / "manifest.yaml")
        if p.is_dir() and (p / "policy_spec.json").exists():
            with open(p / "policy_spec.json", "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            d.setdefault("name", p.name)
            return PolicyManifest.from_dict(d, base_dir=p)

        raise KeyError(
            f"Policy '{name_or_path}' not found in registry. Available policies: {list(self._manifests.keys())}"
        )

    def list_policies(self) -> List[Dict[str, Any]]:
        """Return structured summary list of all available policies."""
        summaries = []
        for name, m in sorted(self._manifests.items()):
            summaries.append({
                "name": name,
                "version": m.version,
                "category": m.category,
                "description": m.description,
                "runtime": m.runtime.type,
                "input_dim": m.observation.dim,
                "output_dim": m.action.dim,
                "frequency_hz": m.control.frequency_hz,
                "path": str(m.base_dir) if m.base_dir else "",
            })
        return summaries

    def format_policy_catalog(self) -> str:
        """Format an ANSI color terminal catalog of available policies."""
        lines = [
            "\n╔══════════════════════════════════════════════════════════════════════╗",
            "║                 VASIMOV AVAILABLE ROBOT POLICIES                     ║",
            "╚══════════════════════════════════════════════════════════════════════╝",
        ]
        for name, m in sorted(self._manifests.items()):
            lines.append(f"\n  ● \033[1;36m{name}\033[0m (v{m.version})")
            lines.append(f"    Category   : {m.category.upper()}")
            lines.append(f"    Runtime    : {m.runtime.type.upper()} ({m.runtime.model_path})")
            lines.append(f"    Input Dim  : {m.observation.dim} float32 (history={m.observation.history_length})")
            lines.append(f"    Output Dim : {m.action.dim} float32 ({m.action.type})")
            lines.append(f"    Frequency  : {m.control.frequency_hz:.1f} Hz (decimation={m.control.decimation})")
            if m.description:
                lines.append(f"    Summary    : {m.description}")
        lines.append("\n" + "─" * 72)
        return "\n".join(lines)

    def validate_policy(self, name_or_path: str, model_context: Any = None) -> Tuple[bool, List[str]]:
        """
        Run deep contract validation on a policy.
        Checks manifest structure, model file, runtime initialization, and tensor shapes.
        """
        checks: List[str] = []
        try:
            manifest = self.get_manifest(name_or_path)
            checks.append(f"✓ Manifest loaded: '{manifest.name}' (v{manifest.version})")
        except Exception as e:
            return False, [f"✗ Manifest load failed: {e}"]

        # 1. Structural validation
        errors = manifest.validate()
        if errors:
            return False, checks + [f"✗ {err}" for err in errors]
        checks.append("✓ Manifest schema & configuration valid")

        # 2. Model file exists
        mpath = manifest.resolve_model_path()
        if not mpath.exists():
            return False, checks + [f"✗ Model artifact missing: {mpath}"]
        checks.append(f"✓ Model artifact present: {mpath.name} ({mpath.stat().st_size / 1024:.1f} KB)")

        # 3. Runtime load & tensor shape validation
        if manifest.runtime.type == "onnx":
            try:
                runtime = ONNXPolicyRuntime(model_path=mpath)
                checks.append(f"✓ ONNX runtime initialized: inputs={runtime.input_names}, outputs={runtime.output_names}")

                # Validate input shape
                in_shapes = runtime.input_shapes
                for in_name, shape in in_shapes.items():
                    last_dim = shape[-1]
                    if isinstance(last_dim, int) and last_dim != manifest.observation.dim:
                        return False, checks + [
                            f"✗ Input shape mismatch for '{in_name}': model has {last_dim}, manifest expects {manifest.observation.dim}"
                        ]
                checks.append(f"✓ Input tensor shapes match observation dim ({manifest.observation.dim})")

                # Validate output shape
                out_shapes = runtime.output_shapes
                for out_name, shape in out_shapes.items():
                    last_dim = shape[-1]
                    if isinstance(last_dim, int) and last_dim != manifest.action.dim:
                        return False, checks + [
                            f"✗ Output shape mismatch for '{out_name}': model has {last_dim}, manifest expects {manifest.action.dim}"
                        ]
                checks.append(f"✓ Output tensor shapes match action dim ({manifest.action.dim})")
                runtime.close()

            except Exception as e:
                return False, checks + [f"✗ Model inference engine failure: {e}"]

        # 4. Joint mapping validation against model if model_context provided
        if model_context is not None:
            missing_act = []
            for jn in manifest.action.joint_order:
                try:
                    _ = model_context.joint(jn)
                except Exception:
                    missing_act.append(jn)
            if missing_act:
                return False, checks + [f"✗ Joints declared in action spec missing in MuJoCo model: {missing_act}"]
            checks.append(f"✓ All {len(manifest.action.joint_order)} action joints verified in robot kinematics")

        return True, checks

    def instantiate(
        self,
        name_or_path: str,
        safety: Optional[SafetyLayer] = None,
    ) -> BasePolicy:
        """Instantiate a runnable BasePolicy from registered name or directory."""
        manifest = self.get_manifest(name_or_path)
        return BasePolicy(manifest=manifest, safety=safety)
