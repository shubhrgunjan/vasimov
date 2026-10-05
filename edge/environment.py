"""
vasimov/edge/environment.py
Environment and scenario management system for Virtual Asimov 1.

Supports:
- Presets: flat, obstacles, playground, friction_low, friction_high, heavy_robot, light_robot, obstacle_basic, terrain_basic
- Real dynamic physics objects: boxes, balls, cylinders, trip barriers with 6-DoF freejoints, mass, inertia, and contact friction
- Parameter configuration: ground friction, robot mass scaling, gravity
- Runtime dynamic obstacle spawning and repositioning
- Deterministic environment reset
"""

from __future__ import annotations
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import mujoco

import re
import xml.etree.ElementTree as ET
import yaml

log = logging.getLogger("vasimov.edge.environment")

_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _BASE_DIR / "model" / "asimov_1_vasimov.xml"
DEFAULT_ENVIRONMENTS_DIR = _BASE_DIR / "environments"

PRESETS: Dict[str, Dict[str, Any]] = {
    "flat": {
        "description": "Standard flat horizontal plane with nominal friction (mu=1.0)",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [],
    },
    "obstacles": {
        "description": "Real physics dynamic obstacles (movable boxes, rolling ball, cylinder, trip bar)",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "dyn_box_1",
                "type": "box",
                "pos": [1.1, 0.0, 0.15],
                "size": [0.12, 0.12, 0.12],
                "mass": 3.0,
                "dynamic": True,
                "rgba": [0.88, 0.38, 0.15, 1.0],
            },
            {
                "name": "dyn_ball_1",
                "type": "sphere",
                "pos": [0.75, 0.14, 0.15],
                "size": [0.10],
                "mass": 1.5,
                "dynamic": True,
                "rgba": [0.15, 0.65, 0.95, 1.0],
            },
            {
                "name": "dyn_cylinder_1",
                "type": "cylinder",
                "pos": [1.5, -0.12, 0.18],
                "size": [0.08, 0.16],
                "mass": 2.5,
                "dynamic": True,
                "rgba": [0.95, 0.75, 0.15, 1.0],
            },
            {
                "name": "dyn_trip_bar",
                "type": "box",
                "pos": [0.55, 0.0, 0.04],
                "size": [0.14, 0.35, 0.04],
                "mass": 4.5,
                "dynamic": True,
                "rgba": [0.85, 0.22, 0.22, 1.0],
            },
        ],
    },
    "playground": {
        "description": "Rich dynamic physics playground arena with multi-shape interactive objects",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "play_crate_1",
                "type": "box",
                "pos": [1.3, 0.0, 0.16],
                "size": [0.15, 0.15, 0.15],
                "mass": 5.0,
                "dynamic": True,
                "rgba": [0.85, 0.40, 0.15, 1.0],
            },
            {
                "name": "play_ball_1",
                "type": "sphere",
                "pos": [0.7, -0.15, 0.12],
                "size": [0.10],
                "mass": 1.2,
                "dynamic": True,
                "rgba": [0.2, 0.7, 0.9, 1.0],
            },
            {
                "name": "play_ball_2",
                "type": "sphere",
                "pos": [1.8, 0.22, 0.15],
                "size": [0.12],
                "mass": 1.8,
                "dynamic": True,
                "rgba": [0.3, 0.85, 0.4, 1.0],
            },
            {
                "name": "play_pin_1",
                "type": "cylinder",
                "pos": [1.0, 0.18, 0.18],
                "size": [0.07, 0.18],
                "mass": 2.0,
                "dynamic": True,
                "rgba": [0.95, 0.6, 0.2, 1.0],
            },
            {
                "name": "play_pin_2",
                "type": "cylinder",
                "pos": [2.2, -0.18, 0.18],
                "size": [0.07, 0.18],
                "mass": 2.0,
                "dynamic": True,
                "rgba": [0.9, 0.3, 0.7, 1.0],
            },
            {
                "name": "play_stumble_block",
                "type": "box",
                "pos": [0.5, 0.0, 0.035],
                "size": [0.12, 0.30, 0.035],
                "mass": 4.0,
                "dynamic": True,
                "rgba": [0.75, 0.15, 0.15, 1.0],
            },
            {
                "name": "play_crate_2",
                "type": "box",
                "pos": [2.6, 0.10, 0.14],
                "size": [0.12, 0.12, 0.12],
                "mass": 3.0,
                "dynamic": True,
                "rgba": [0.4, 0.5, 0.9, 1.0],
            },
        ],
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
        "description": "Walking corridor with real dynamic stumbling blocks and collision cubes",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "step_block_1",
                "type": "box",
                "pos": [1.2, 0.0, 0.05],
                "size": [0.15, 0.35, 0.04],
                "mass": 4.0,
                "dynamic": True,
                "rgba": [0.8, 0.3, 0.2, 1.0],
            },
            {
                "name": "step_block_2",
                "type": "box",
                "pos": [2.5, 0.1, 0.12],
                "size": [0.12, 0.12, 0.12],
                "mass": 3.0,
                "dynamic": True,
                "rgba": [0.2, 0.6, 0.8, 1.0],
            },
        ],
    },
    "terrain_basic": {
        "description": "Multi-level terrain platforms and dynamic cylinders along walking path",
        "ground_friction": 1.0,
        "mass_scale": 1.0,
        "gravity": [0.0, 0.0, -9.81],
        "obstacles": [
            {
                "name": "terrain_platform_1",
                "type": "box",
                "pos": [1.2, -0.2, 0.02],
                "size": [0.4, 0.4, 0.02],
                "dynamic": False,
                "rgba": [0.7, 0.7, 0.3, 1.0],
            },
            {
                "name": "terrain_platform_2",
                "type": "box",
                "pos": [2.2, 0.2, 0.03],
                "size": [0.4, 0.4, 0.03],
                "dynamic": False,
                "rgba": [0.4, 0.7, 0.4, 1.0],
            },
            {
                "name": "terrain_cylinder_marker",
                "type": "cylinder",
                "pos": [3.2, 0.0, 0.18],
                "size": [0.08, 0.18],
                "mass": 2.5,
                "dynamic": True,
                "rgba": [0.9, 0.5, 0.1, 1.0],
            },
        ],
    },
}

# Alias for ease of access
PRESETS["physics_obstacles"] = PRESETS["obstacles"]



def _extract_custom_xml_parts(xml_str: str) -> Tuple[str, str]:
    """Extract <asset> children and <worldbody> children from a custom XML snippet."""
    cleaned = xml_str.strip()
    if not cleaned:
        return "", ""
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        root = ET.fromstring(f"<root>{cleaned}</root>")

    assets_parts: List[str] = []
    worldbody_parts: List[str] = []

    valid_worldbody_tags = {"body", "geom", "site", "camera", "light", "joint", "freejoint", "composite", "flex"}

    if root.tag == "mujoco":
        for child in root:
            if child.tag == "asset":
                assets_parts.extend(ET.tostring(a, encoding="unicode") for a in child)
            elif child.tag == "worldbody":
                worldbody_parts.extend(ET.tostring(w, encoding="unicode") for w in child)
            elif child.tag in valid_worldbody_tags:
                worldbody_parts.append(ET.tostring(child, encoding="unicode"))
            # Note: Top-level elements like compiler, visual, statistic, option, default
            # are top-level simulation settings and must NOT be injected into <worldbody>.
    elif root.tag == "worldbody":
        worldbody_parts.extend(ET.tostring(w, encoding="unicode") for w in root)
    elif root.tag == "root":
        for child in root:
            if child.tag == "asset":
                assets_parts.extend(ET.tostring(a, encoding="unicode") for a in child)
            elif child.tag == "worldbody":
                worldbody_parts.extend(ET.tostring(w, encoding="unicode") for w in child)
            elif child.tag in valid_worldbody_tags:
                worldbody_parts.append(ET.tostring(child, encoding="unicode"))
    elif root.tag in valid_worldbody_tags:
        worldbody_parts.append(ET.tostring(root, encoding="unicode"))

    return "\n".join(assets_parts), "\n".join(worldbody_parts)


def _combine_custom_xml_into_base(
    base_xml_text: str,
    custom_xml_content: Optional[str] = None,
    extra_worldbody_elements: Optional[List[str]] = None,
) -> str:
    """Inject custom assets and worldbody elements into base XML text."""
    assets_str, worldbody_str = _extract_custom_xml_parts(custom_xml_content) if custom_xml_content else ("", "")

    all_worldbody = []
    if extra_worldbody_elements:
        all_worldbody.extend(extra_worldbody_elements)
    if worldbody_str:
        all_worldbody.append(worldbody_str)

    modified_xml = base_xml_text
    if assets_str and "</asset>" in modified_xml:
        modified_xml = modified_xml.replace("</asset>", f"{assets_str}\n  </asset>")

    if all_worldbody:
        combined_wb = "\n".join(all_worldbody)
        # If custom environment supplies a floor (e.g. name="floor"), remove the base default floor
        # to prevent MuJoCo schema error: "Error: repeated name 'floor' in geom"
        if 'name="floor"' in combined_wb or "name='floor'" in combined_wb:
            modified_xml = re.sub(r'<geom\s+name=["\']floor["\'][^>]*/>', '<!-- overridden base floor -->', modified_xml)

        modified_xml = modified_xml.replace("</worldbody>", f"{combined_wb}\n  </worldbody>")

    return modified_xml


class EnvironmentManager:
    """Manages scene presets, custom YAML/XML environment files, procedural obstacles, and physics parameters."""

    def __init__(self, base_xml_path: Optional[Path] = None, environments_dir: Optional[Path] = None):
        self.base_xml_path = Path(base_xml_path or DEFAULT_MODEL_PATH).resolve()
        if not self.base_xml_path.exists():
            raise FileNotFoundError(f"Base model XML not found: {self.base_xml_path}")

        self.environments_dir = Path(environments_dir or DEFAULT_ENVIRONMENTS_DIR).resolve()
        self.current_preset: str = "flat"
        self.active_config: Dict[str, Any] = copy.deepcopy(PRESETS["flat"])
        self.generated_model_path: Optional[Path] = None

    def list_custom_environments(self) -> Dict[str, Path]:
        """Scan environments_dir for *.yaml, *.yml, and *.xml files."""
        result: Dict[str, Path] = {}
        if self.environments_dir.exists() and self.environments_dir.is_dir():
            for p in sorted(self.environments_dir.glob("*")):
                if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".xml"):
                    result[p.stem] = p.resolve()
        return result

    def list_presets(self) -> Dict[str, str]:
        """Return dict of available preset names and discovered custom files to descriptions."""
        result = {name: cfg["description"] for name, cfg in PRESETS.items() if name != "physics_obstacles"}
        custom_files = self.list_custom_environments()
        for stem, fpath in custom_files.items():
            if stem not in result:
                desc = f"Custom environment file ({fpath.name})"
                if fpath.suffix.lower() in (".yaml", ".yml"):
                    try:
                        data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and data.get("description"):
                            desc = str(data["description"])
                    except Exception:
                        pass
                result[stem] = desc
        return result

    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Return copy of preset configuration or custom environment config."""
        if preset_name in PRESETS:
            return copy.deepcopy(PRESETS[preset_name])
        custom_files = self.list_custom_environments()
        if preset_name in custom_files:
            cfg, _ = self.parse_custom_file(custom_files[preset_name])
            return cfg
        raise KeyError(f"Unknown preset or environment '{preset_name}'. Available: {list(self.list_presets().keys())}")

    def find_environment_file(self, name_or_path: Union[str, Path]) -> Optional[Path]:
        """Locate custom environment file from path, relative path, or stem in environments_dir."""
        p = Path(name_or_path)
        if p.exists() and p.is_file():
            return p.resolve()

        cwd_p = Path.cwd() / p
        if cwd_p.exists() and cwd_p.is_file():
            return cwd_p.resolve()

        base_p = _BASE_DIR / p
        if base_p.exists() and base_p.is_file():
            return base_p.resolve()

        env_p = self.environments_dir / p
        if env_p.exists() and env_p.is_file():
            return env_p.resolve()

        for ext in (".yaml", ".yml", ".xml"):
            cand = self.environments_dir / f"{name_or_path}{ext}"
            if cand.exists() and cand.is_file():
                return cand.resolve()

        return None

    def parse_custom_file(self, file_path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse a custom environment file (.yaml, .yml, or .xml) into config dict and XML snippet."""
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Custom environment file not found: {file_path}")

        ext = file_path.suffix.lower()
        file_text = file_path.read_text(encoding="utf-8")

        if ext in (".yaml", ".yml"):
            data = yaml.safe_load(file_text) or {}
            cfg = {
                "name": data.get("name", file_path.stem),
                "description": data.get("description", f"Custom YAML environment ({file_path.name})"),
                "ground_friction": float(data.get("ground_friction", 1.0)),
                "mass_scale": float(data.get("mass_scale", 1.0)),
                "gravity": list(data.get("gravity", [0.0, 0.0, -9.81])),
                "obstacles": data.get("obstacles", []),
            }
            raw_xml = data.get("xml_content", "")
            return cfg, raw_xml
        elif ext == ".xml":
            xml_dir = file_path.parent
            def resolve_asset_match(match):
                attr = match.group(1)
                rel_val = match.group(2)
                cand = (xml_dir / rel_val).resolve()
                if cand.exists():
                    return f'{attr}="{cand}"'
                return match.group(0)

            def resolve_asset_match(m):
                rel_path = m.group(1)
                cand = (xml_dir / rel_path).resolve()
                if cand.exists():
                    return f'file="{cand}"'
                return m.group(0)
            resolved_text = re.sub(r'file=["\']([^"\']+)["\']', resolve_asset_match, file_text)

            # Extract any box/sphere/cylinder bodies as obstacles for client telemetry
            obstacles = []
            try:
                root = ET.fromstring(file_text)
                for body in root.iter('body'):
                    bname = body.get('name', '')
                    pos_str = body.get('pos', '0 0 0')
                    pos = [float(x) for x in pos_str.split()]
                    geom = body.find('geom')
                    if geom is not None:
                        gtype = geom.get('type', 'box')
                        size_str = geom.get('size', '0.1 0.1 0.1')
                        size = [float(x) for x in size_str.split()]
                        obstacles.append({
                            'name': bname,
                            'type': gtype,
                            'pos': pos,
                            'size': size,
                            'dynamic': body.find('freejoint') is not None,
                        })
            except Exception:
                pass

            cfg = {
                "name": file_path.stem,
                "preset": file_path.stem,
                "description": f"PAIR Lab MUJOCO Environment ({file_path.name})",
                "ground_friction": 1.0,
                "mass_scale": 1.0,
                "gravity": [0.0, 0.0, -9.81],
                "obstacles": obstacles,
            }
            return cfg, resolved_text
        else:
            raise ValueError(f"Unsupported environment file extension '{ext}'. Must be .yaml, .yml, or .xml")

    def _build_obstacle_xmls(self, obstacles: List[Dict[str, Any]]) -> List[str]:
        """Convert obstacle dictionaries into MuJoCo XML body/geom strings."""
        obstacle_xmls: List[str] = []
        for obs in obstacles:
            name = obs["name"]
            geom_type = obs.get("type", "box")
            px, py, pz = obs.get("pos", [0, 0, 0])
            raw_size = obs.get("size", [0.1, 0.1, 0.1])
            r, g, b, a = obs.get("rgba", [0.8, 0.4, 0.2, 1.0])
            mass = float(obs.get("mass", 3.0))
            friction = float(obs.get("friction", 1.0))
            dynamic = bool(obs.get("dynamic", True))

            if geom_type == "cylinder":
                size_str = f"{raw_size[0]} {raw_size[1]}"
            elif geom_type == "sphere":
                size_str = f"{raw_size[0]}"
            else:  # box
                if len(raw_size) >= 3:
                    size_str = f"{raw_size[0]} {raw_size[1]} {raw_size[2]}"
                else:
                    size_str = f"{raw_size[0]} {raw_size[0]} {raw_size[0]}"

            if dynamic:
                body_xml = (
                    f'    <body name="{name}" pos="{px} {py} {pz}">\n'
                    f'      <freejoint name="{name}_joint" />\n'
                    f'      <geom name="{name}_geom" type="{geom_type}" size="{size_str}" '
                    f'mass="{mass}" friction="{friction} 0.005 0.0001" '
                    f'rgba="{r} {g} {b} {a}" contype="1" conaffinity="1" />\n'
                    f'    </body>'
                )
                obstacle_xmls.append(body_xml)
            else:
                geom_xml = (
                    f'    <geom name="{name}" type="{geom_type}" size="{size_str}" '
                    f'pos="{px} {py} {pz}" rgba="{r} {g} {b} {a}" contype="1" conaffinity="1" />'
                )
                obstacle_xmls.append(geom_xml)
        return obstacle_xmls

    def create_model_from_custom_file(self, file_path: Union[str, Path]) -> Tuple[mujoco.MjModel, Path]:
        """
        Build an MjModel instance from a custom environment file (.yaml, .yml, or .xml).
        Injects the environment definition into the Asimov 1 base XML.
        """
        fpath = Path(file_path).resolve()
        cfg, raw_xml = self.parse_custom_file(fpath)

        obstacles = cfg.get("obstacles", [])
        obstacle_xmls = self._build_obstacle_xmls(obstacles) if fpath.suffix.lower() != ".xml" else []

        base_xml_text = self.base_xml_path.read_text(encoding="utf-8")
        modified_xml = _combine_custom_xml_into_base(
            base_xml_text=base_xml_text,
            custom_xml_content=raw_xml,
            extra_worldbody_elements=obstacle_xmls,
        )

        clean_stem = re.sub(r"[^a-zA-Z0-9_]", "_", fpath.stem)
        gen_path = self.base_xml_path.parent / f"_gen_env_custom_{clean_stem}.xml"

        # Cleanup prior generated model file if different
        if self.generated_model_path and self.generated_model_path != self.base_xml_path and self.generated_model_path != gen_path:
            try:
                if self.generated_model_path.exists():
                    self.generated_model_path.unlink()
            except Exception:
                pass

        gen_path.write_text(modified_xml, encoding="utf-8")
        self.generated_model_path = gen_path
        self.current_preset = fpath.stem
        self.active_config = cfg

        model = mujoco.MjModel.from_xml_path(str(gen_path))
        self.apply_runtime_parameters(model, cfg)

        log.info("[ENV] Loaded custom environment from '%s' (friction=%.2f, mass_scale=%.2f, obstacles=%d)",
                 fpath.name, cfg["ground_friction"], cfg["mass_scale"], len(obstacles))
        return model, self.generated_model_path

    def create_model_for_preset(self, preset_or_path: str) -> Tuple[mujoco.MjModel, Path]:
        """
        Build an MjModel instance configured for the specified preset or custom file.
        Accepts:
        - Preset name: 'flat', 'obstacles', 'playground', etc.
        - Custom environment name from environments/: 'corridor', 'arena', etc.
        - Direct path to custom file: 'environments/arena.xml', '/path/to/my_env.yaml'
        """
        if preset_or_path in PRESETS:
            cfg = copy.deepcopy(PRESETS[preset_or_path])
            self.current_preset = preset_or_path
            self.active_config = cfg

            obstacles = cfg.get("obstacles", [])
            if not obstacles:
                # Clean model from base XML
                model = mujoco.MjModel.from_xml_path(str(self.base_xml_path))
                self.generated_model_path = self.base_xml_path
            else:
                obstacle_xmls = self._build_obstacle_xmls(obstacles)
                base_xml_text = self.base_xml_path.read_text(encoding="utf-8")
                modified_xml = _combine_custom_xml_into_base(
                    base_xml_text=base_xml_text,
                    extra_worldbody_elements=obstacle_xmls,
                )
                gen_path = self.base_xml_path.parent / f"_gen_env_{preset_or_path}.xml"
                gen_path.write_text(modified_xml, encoding="utf-8")
                self.generated_model_path = gen_path
                model = mujoco.MjModel.from_xml_path(str(gen_path))

            self.apply_runtime_parameters(model, cfg)
            log.info("[ENV] Loaded preset '%s' (friction=%.2f, mass_scale=%.2f, obstacles=%d dynamic=%s)",
                     preset_or_path, cfg["ground_friction"], cfg["mass_scale"], len(obstacles),
                     any(obs.get("dynamic", True) for obs in obstacles))
            return model, self.generated_model_path

        # Check for custom file
        custom_file = self.find_environment_file(preset_or_path)
        if custom_file is not None:
            return self.create_model_from_custom_file(custom_file)

        available = list(self.list_presets().keys())
        raise KeyError(
            f"Unknown environment preset or file '{preset_or_path}'. "
            f"Available presets and custom environments: {available}"
        )

    def apply_runtime_parameters(self, model: mujoco.MjModel, cfg: Dict[str, Any]) -> None:
        """Apply ground friction, robot link mass scaling, and gravity to compiled MjModel."""
        friction_val = float(cfg.get("ground_friction", 1.0))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id != -1:
            model.geom_friction[floor_id, 0] = friction_val

        for i in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and ("foot" in gname or "ankle" in gname) and "visual" not in gname:
                model.geom_friction[i, 0] = friction_val

        mass_scale = float(cfg.get("mass_scale", 1.0))
        if not np.isclose(mass_scale, 1.0, atol=1e-4):
            for i in range(model.nbody):
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if bname and bname != "world":
                    model.body_mass[i] *= mass_scale
                    model.body_inertia[i, :] *= mass_scale

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

