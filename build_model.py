#!/usr/bin/env python3
"""
vasimov/build_model.py
Derived MuJoCo Model Generator for Asimov 1.

Generates `vasimov/model/asimov_1_vasimov.xml` from `upstream/asimov-1/sim-model/xmls/asimov_1.xml`:
1. Adds `neck_yaw_joint` (axis z) and `neck_pitch_joint` (axis y) to enable full 25-joint articulation.
2. Configures official armature, damping, and frictionloss extracted from Isaac Lab / Asimov-mjlab.
3. Adds 25 <motor> actuators with effort limits matching `asimov_1.urdf`.
4. Adds sensors for joint position (jointpos), joint velocity (jointvel), and actuator force (actuatorfrc).
5. Sets simulation timestep dt = 0.005s (200 Hz).
6. Verifies that the model loads, njnt = 26 (1 free + 25 hinge), nu = 25, and mass = 32.2249 kg.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


# Official parameters mined from Isaac Lab (cyclotron/assets/robots/asimov_1.py) and asimov_1.urdf
JOINT_PARAMS = {
    # Left Leg (CAN IDs 1-6)
    "left_hip_pitch_joint": {
        "effort": 45.0, "vel_limit": 12.57, "armature": 0.0698, "damping": 5.0, "friction": 0.70,
    },
    "left_hip_roll_joint": {
        "effort": 45.0, "vel_limit": 3.98, "armature": 0.1400, "damping": 5.0, "friction": 0.20,
    },
    "left_hip_yaw_joint": {
        "effort": 28.0, "vel_limit": 5.45, "armature": 0.0687, "damping": 5.0, "friction": 0.70,
    },
    "left_knee_joint": {
        "effort": 45.0, "vel_limit": 12.25, "armature": 0.0330, "damping": 5.0, "friction": 0.70,
    },
    "left_ankle_pitch_joint": {
        "effort": 40.0, "vel_limit": 9.32, "armature": 0.0484, "damping": 5.0, "friction": 0.40,
    },
    "left_ankle_roll_joint": {
        "effort": 17.0, "vel_limit": 9.32, "armature": 0.0484, "damping": 5.0, "friction": 0.40,
    },

    # Right Leg (CAN IDs 7-12)
    "right_hip_pitch_joint": {
        "effort": 45.0, "vel_limit": 12.57, "armature": 0.0698, "damping": 5.0, "friction": 0.70,
    },
    "right_hip_roll_joint": {
        "effort": 45.0, "vel_limit": 3.98, "armature": 0.1400, "damping": 5.0, "friction": 0.20,
    },
    "right_hip_yaw_joint": {
        "effort": 28.0, "vel_limit": 5.45, "armature": 0.0687, "damping": 5.0, "friction": 0.70,
    },
    "right_knee_joint": {
        "effort": 45.0, "vel_limit": 12.25, "armature": 0.0330, "damping": 5.0, "friction": 0.70,
    },
    "right_ankle_pitch_joint": {
        "effort": 40.0, "vel_limit": 9.32, "armature": 0.0484, "damping": 5.0, "friction": 0.40,
    },
    "right_ankle_roll_joint": {
        "effort": 17.0, "vel_limit": 9.32, "armature": 0.0484, "damping": 5.0, "friction": 0.40,
    },

    # Torso (CAN ID 23)
    "waist_yaw_joint": {
        "effort": 40.0, "vel_limit": 12.57, "armature": 0.0698, "damping": 5.0, "friction": 0.70,
    },

    # Neck (CAN IDs 24-25, ASSUMED ranges based on humanoid design; effort 12.0 matching small actuators)
    "neck_yaw_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
        "axis": "0 0 1", "range": "-0.7854 0.7854",
    },
    "neck_pitch_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
        "axis": "0 1 0", "range": "-0.5236 0.5236",
    },

    # Right Arm (CAN IDs 18-22)
    "right_shoulder_pitch_joint": {
        "effort": 30.0, "vel_limit": 3.98, "armature": 0.1400, "damping": 5.0, "friction": 0.20,
    },
    "right_shoulder_roll_joint": {
        "effort": 25.0, "vel_limit": 12.25, "armature": 0.0330, "damping": 5.0, "friction": 0.70,
    },
    "right_shoulder_yaw_joint": {
        "effort": 20.0, "vel_limit": 5.45, "armature": 0.0687, "damping": 5.0, "friction": 0.70,
    },
    "right_elbow_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
    },
    "right_wrist_yaw_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
    },

    # Left Arm (CAN IDs 13-17)
    "left_shoulder_pitch_joint": {
        "effort": 30.0, "vel_limit": 3.98, "armature": 0.1400, "damping": 5.0, "friction": 0.20,
    },
    "left_shoulder_roll_joint": {
        "effort": 25.0, "vel_limit": 12.25, "armature": 0.0330, "damping": 5.0, "friction": 0.70,
    },
    "left_shoulder_yaw_joint": {
        "effort": 20.0, "vel_limit": 5.45, "armature": 0.0687, "damping": 5.0, "friction": 0.70,
    },
    "left_elbow_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
    },
    "left_wrist_yaw_joint": {
        "effort": 12.0, "vel_limit": 9.32, "armature": 0.0242, "damping": 2.0, "friction": 0.40,
    },
}

# The 25 hinge joints in official firmware CAN order (for actuator indexing)
FIRMWARE_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "waist_yaw_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
]


def build_derived_model(upstream_xml: Path, output_xml: Path) -> Path:
    if not upstream_xml.exists():
        raise FileNotFoundError(f"Upstream XML not found: {upstream_xml}")

    tree = ET.parse(upstream_xml)
    root = tree.getroot()

    # 1. Update meshdir in compiler tag so it finds STL meshes relative to output_xml
    compiler = root.find("compiler")
    if compiler is not None:
        # compute relative path from output_xml to upstream meshes directory
        meshes_dir = upstream_xml.parent.parent / "assets" / "meshes"
        rel_meshdir = Path(os.path.relpath(meshes_dir, output_xml.parent)).as_posix()
        compiler.set("meshdir", rel_meshdir)

    # 2. Update physics option: timestep = 0.005 s
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.005")

    # 3. Add neck joints
    neck_yaw_link = root.find(".//body[@name='neck_yaw_link']")
    if neck_yaw_link is not None:
        # Check if joint already exists
        if neck_yaw_link.find("joint[@name='neck_yaw_joint']") is None:
            yj = ET.Element("joint", {
                "name": "neck_yaw_joint",
                "type": "hinge",
                "ref": "0.0",
                "class": "motor",
                "axis": JOINT_PARAMS["neck_yaw_joint"]["axis"],
                "range": JOINT_PARAMS["neck_yaw_joint"]["range"],
                "armature": str(JOINT_PARAMS["neck_yaw_joint"]["armature"]),
                "damping": str(JOINT_PARAMS["neck_yaw_joint"]["damping"]),
                "frictionloss": str(JOINT_PARAMS["neck_yaw_joint"]["friction"]),
            })
            neck_yaw_link.insert(0, yj)

    neck_pitch_link = root.find(".//body[@name='neck_pitch_link']")
    if neck_pitch_link is not None:
        if neck_pitch_link.find("joint[@name='neck_pitch_joint']") is None:
            pj = ET.Element("joint", {
                "name": "neck_pitch_joint",
                "type": "hinge",
                "ref": "0.0",
                "class": "motor",
                "axis": JOINT_PARAMS["neck_pitch_joint"]["axis"],
                "range": JOINT_PARAMS["neck_pitch_joint"]["range"],
                "armature": str(JOINT_PARAMS["neck_pitch_joint"]["armature"]),
                "damping": str(JOINT_PARAMS["neck_pitch_joint"]["damping"]),
                "frictionloss": str(JOINT_PARAMS["neck_pitch_joint"]["friction"]),
            })
            neck_pitch_link.insert(0, pj)

    # 4. Update armature, damping, and frictionloss on all existing hinge joints
    for joint_elem in root.findall(".//joint"):
        jname = joint_elem.get("name")
        if jname in JOINT_PARAMS:
            p = JOINT_PARAMS[jname]
            joint_elem.set("armature", str(p["armature"]))
            joint_elem.set("damping", str(p["damping"]))
            joint_elem.set("frictionloss", str(p["friction"]))

    # 5. Create <actuator> element with 25 <motor> actuators
    existing_act = root.find("actuator")
    if existing_act is not None:
        root.remove(existing_act)

    actuator_section = ET.SubElement(root, "actuator")
    for jname in FIRMWARE_JOINT_ORDER:
        p = JOINT_PARAMS[jname]
        effort = p["effort"]
        motor_name = f"{jname}_actuator"
        ET.SubElement(actuator_section, "motor", {
            "name": motor_name,
            "joint": jname,
            "gear": "1",
            "ctrllimited": "true",
            "ctrlrange": f"{-effort} {effort}",
            "forcelimited": "true",
            "forcerange": f"{-effort} {effort}",
        })

    # 6. Update <sensor> section: preserve IMU sensors, add jointpos, jointvel, and actuatorfrc
    sensor_section = root.find("sensor")
    if sensor_section is None:
        sensor_section = ET.SubElement(root, "sensor")

    # Add sensors for each of the 25 hinge joints
    for jname in FIRMWARE_JOINT_ORDER:
        ET.SubElement(sensor_section, "jointpos", {
            "name": f"{jname}_pos",
            "joint": jname,
        })
        ET.SubElement(sensor_section, "jointvel", {
            "name": f"{jname}_vel",
            "joint": jname,
        })
        ET.SubElement(sensor_section, "actuatorfrc", {
            "name": f"{jname}_force",
            "actuator": f"{jname}_actuator",
        })

    # 7. Add foot contact / touch sensors (ground-truth stream only)
    ET.SubElement(sensor_section, "touch", {"name": "left_foot_touch", "site": "left_foot"})
    ET.SubElement(sensor_section, "touch", {"name": "right_foot_touch", "site": "right_foot"})
    ET.SubElement(sensor_section, "force", {"name": "left_foot_force", "site": "left_foot"})
    ET.SubElement(sensor_section, "force", {"name": "right_foot_force", "site": "right_foot"})

    # 8. Add virtual gantry equality constraint
    equality_section = root.find("equality")
    if equality_section is None:
        equality_section = ET.SubElement(root, "equality")
    if equality_section.find("weld[@name='virtual_gantry']") is None:
        ET.SubElement(equality_section, "weld", {"name": "virtual_gantry", "body1": "pelvis_link", "active": "true"})

    # 9. Write derived XML
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(output_xml), encoding="utf-8", xml_declaration=True)
    return output_xml


def verify_model(xml_path: Path):
    print("=" * 80)
    print("VERIFYING DERIVED ASIMOV 1 MODEL")
    print("=" * 80)
    print(f"Path: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    total_mass = float(sum(model.body_mass))
    expected_mass = 32.224913

    print(f"  nq (generalized coords) : {model.nq} (expected 32: 7 root + 25 hinge)")
    print(f"  nv (velocity DOFs)      : {model.nv} (expected 31: 6 root + 25 hinge)")
    print(f"  njnt (joints)           : {model.njnt} (expected 26: 1 free + 25 hinge)")
    print(f"  nu (actuators)          : {model.nu} (expected 25)")
    print(f"  nsensor                 : {model.nsensor} (expected 84: 5 IMU + 25*3 + 4 foot)")
    print(f"  opt.timestep            : {model.opt.timestep:.6f} s ({(1.0 / model.opt.timestep):.1f} Hz)")
    print(f"  Total Mass              : {total_mass:.6f} kg (expected {expected_mass:.6f} kg)")

    # Assertions
    assert model.nu == 25, f"Expected 25 actuators, got {model.nu}"
    assert model.njnt == 26, f"Expected 26 joints, got {model.njnt}"
    assert model.nsensor == 84, f"Expected 84 sensors, got {model.nsensor}"
    assert abs(total_mass - expected_mass) < 1e-4, f"Mass discrepancy! {total_mass} vs {expected_mass}"

    # Verify all 25 actuators correspond to FIRMWARE_JOINT_ORDER
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        target_jnt_id = model.actuator_trnid[i, 0]
        target_jnt = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_jnt_id)
        expected_jnt = FIRMWARE_JOINT_ORDER[i]
        assert target_jnt == expected_jnt, f"Actuator {i} mismatch: {target_jnt} vs {expected_jnt}"
        effort = JOINT_PARAMS[expected_jnt]["effort"]
        assert abs(model.actuator_ctrlrange[i, 1] - effort) < 1e-3, f"Effort mismatch on {act_name}"

    print("-" * 80)
    print("VERIFICATION RESULT: ALL CHECKS PASSED!")
    print("=" * 80)


def main():
    import os
    parser = argparse.ArgumentParser(description="Generate and verify derived Asimov 1 model.")
    default_upstream = Path(__file__).resolve().parent / "upstream" / "asimov-1" / "sim-model" / "xmls" / "asimov_1.xml"
    default_output = Path(__file__).resolve().parent / "model" / "asimov_1_vasimov.xml"

    parser.add_argument("--upstream", type=Path, default=default_upstream, help="Path to upstream asimov_1.xml")
    parser.add_argument("--output", type=Path, default=default_output, help="Path for derived model")
    args = parser.parse_args()

    out = build_derived_model(args.upstream, args.output)
    print(f"Generated derived model at: {out}")
    verify_model(out)


if __name__ == "__main__":
    main()
