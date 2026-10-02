"""
vasimov/edge/environment.py
Environment and scenario management system for Virtual Asimov 1.

Supports:
- Presets: flat, friction_low, friction_high, heavy_robot, light_robot, obstacle_basic, terrain_basic
- Custom object insertion: box, step, wall, table, cylinder
- Parameter configuration: ground friction, robot mass scaling, gravity
- Deterministic environment reset
"""

from __future__ import annotations
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import mujoco

log = logging.getLogger("vasimov.edge.environment")

_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _BASE_DIR / "model" / "asimov_1_vasimov.xml"

PRESETS = {
    "flat": {
        "description": "Standard flat horizontal plane with nominal friction (mu=1.0)",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "friction_low": {
        "description": "Slippery low-friction ice-like ground (mu=0.2)",
        "ground_friction": 0.2,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "friction_high": {
        "description": "High-grip high-friction rubberized ground (mu=1.8)",
        "ground_friction": 1.8,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "heavy_robot": {
        "description": "Payload-loaded heavy robot (+15% robot link mass scale)",
        "ground_friction": 1.0,
        "mass_scale": 1.15,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "light_robot": {
        "description": "Lightweight robot (-15% robot link mass scale)",
        "ground_friction": 1.0,
        "mass_scale": 0.85,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "obstacle_basic": {
        "description": "Flat ground with low step obstacles and blocks in walking corridor",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "step_block_1",
                "type": "box",
                "pos": [1.5, 0.0, 0.025],
                "size": [0.3, 0.5, 0.025],
                "rgba": [0.8, 0.3, 0.2, 1.0],
            },
            {
                "name": "step_block_2",
                "type": "box",
                "pos": [3.0, 0.1, 0.035],
                "size": [0.3, 0.5, 0.035],
                "rgba": [0.2, 0.6, 0.8, 1.0],
            },
        ],
    },
    "terrain_basic": {
        "description": "Stepped multi-level terrain platforms along walking path",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "terrain_platform_1",
                "type": "box",
                "pos": [1.2, -0.2, 0.02],
                "size": [0.4, 0.4, 0.02],
                "rgba": [0.7, 0.7, 0.3, 1.0],
            },
            {
                "name": "terrain_platform_2",
                "type": "box",
                "pos": [2.2, 0.2, 0.03],
                "size": [0.4, 0.4, 0.03],
                "rgba": [0.4, 0.7, 0.4, 1.0],
            },
            {
                "name": "terrain_cylinder_marker",
                "type": "cylinder",
                "pos": [3.5, 0.0, 0.15],
                "size": [0.08, 0.15, 0.0],
                "rgba": [0.9, 0.5, 0.1, 1.0],
            },
        ],
    },
}


class EnvironmentManager:
    """Manages scene presets, procedural obstacle compilation, and physics environment parameters."""

    def __init__(self, base_xml_path: Optional[Path] = None):
        self.base_xml_path = base_xml_path or DEFAULT_MODEL_PATH
        if not self.base_xml_path.exists():
            raise FileNotFoundError(f"Base model XML not found: {self.base_xml_path}")

        self.current_preset: str = "flat"
        self.active_config: Dict[str, Any] = copy.deepcopy(PRESETS["flat"])
        self.generated_model_path: Optional[Path] = None

    def list_presets(self) -> Dict[str, str]:
        """Return dict of available preset names to descriptions."""
        return {name: cfg["description"] for name, cfg in PRESETS.items()}

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Return copy of preset configuration."""
        if preset_name not in PRESETS:
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")
        return copy.deepcopy(PRESETS[preset_name])

    def create_model_for_preset(self, preset_name: str) -> Tuple[mujoco.MjModel, Path]:
        """
        Build an MjModel instance configured for the specified preset.
        Generates temporary XML in model/ directory if obstacles are present.
        """
        if preset_name not in PRESETS:
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")

        cfg = copy.deepcopy(PRESETS[preset_name])
        self.current_preset = preset_name
        self.active_config = cfg

        obstacles = cfg.get("obstacles", [])
        if not obstacles:
            # Clean model from base XML
            model = mujoco.MjModel.from_xml_path(str(self.base_xml_path))
            self.generated_model_path = self.base_xml_path
        else:
            # Inject obstacles into worldbody XML
            base_xml_text = self.base_xml_path.read_text(encoding="utf-8")
            obstacle_geoms = []
            for obs in obstacles:
                name = obs["name"]
                geom_type = obs.get("type", "box")
                px, py, pz = obs.get("pos", [0, 0, 0])
                sx, sy, sz = obs.get("size", [0.1, 0.1, 0.1])
                r, g, b, a = obs.get("rgba", [0.8, 0.4, 0.2, 1.0])
                if geom_type == "cylinder":
                    # For cylinder, size is [radius, half_length]
                    size_str = f"{sx} {sy}"
                else:
                    size_str = f"{sx} {sy} {sz}"

                geom_xml = (
                    f'<geom name="{name}" type="{geom_type}" size="{size_str}" '
                    f'pos="{px} {py} {pz}" rgba="{r} {g} {b} {a}" contype="1" conaffinity="1"/>'
                )
                obstacle_geoms.append(geom_xml)

            injected_xml = "\n".join(obstacle_geoms)
            modified_xml = base_xml_text.replace("</worldbody>", f"{injected_xml}\n</worldbody>")

            gen_path = self.base_xml_path.parent / f"_gen_env_{preset_name}.xml"
            gen_path.write_text(modified_xml, encoding="utf-8")
            self.generated_model_path = gen_path
            model = mujoco.MjModel.from_xml_path(str(gen_path))

        # Apply runtime environment parameters to model
        self.apply_runtime_parameters(model, cfg)
        log.info("[ENV] Loaded preset '%s' (friction=%.2f, mass_scale=%.2f, obstacles=%d)",
                 preset_name, cfg["ground_friction"], cfg["mass_scale"], len(obstacles))
        return model, self.generated_model_path

    def apply_runtime_parameters(self, model: mujoco.MjModel, cfg: Dict[str, Any]) -> None:
        """Apply ground friction, robot link mass scaling, and gravity to compiled MjModel."""
        # 1. Ground friction
        friction_val = float(cfg.get("ground_friction", 1.0))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id != -1:
            model.geom_friction[floor_id, 0] = friction_val

        # Also set friction on foot collision geoms
        for i in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and ("foot" in gname or "ankle" in gname) and "visual" not in gname:
                model.geom_friction[i, 0] = friction_val

        # 2. Mass scaling on robot bodies
        mass_scale = float(cfg.get("mass_scale", 1.0))
        if not np.isclose(mass_scale, 1.0, atol=1e-4):
            for i in range(model.nbody):
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if bname and bname != "world":
                    model.body_mass[i] *= mass_scale
                    model.body_inertia[i, :] *= mass_scale

        # 3. Gravity vector
        grav = cfg.get("gravity", [0.0, 0.0, -9.81])
        model.opt.gravity[:] = grav

    def cleanup(self) -> None:
        """Remove any generated XML files."""
        if self.generated_model_path and self.generated_model_path != self.base_xml_path:
            if self.generated_model_path.exists():
                try:
                    self.generated_model_path.unlink()
                except Exception as e:
                    log.warning("[ENV] Failed to unlink %s: %s", self.generated_model_path, e)
