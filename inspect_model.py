#!/usr/bin/env python3
"""
vasimov/inspect_model.py
MuJoCo Model Inspection Tool for Asimov 1.

Loads the full-robot MuJoCo model (asimov_1.xml) and extracts all model metadata,
joints, actuators, sensors, inertial properties, and floating-base kinematics.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import mujoco


def get_joint_type_name(jtype: int) -> str:
    types = {
        0: "free",
        1: "ball",
        2: "slide",
        3: "hinge",
    }
    return types.get(int(jtype), f"unknown({jtype})")


def get_sensor_type_name(stype: int) -> str:
    try:
        return mujoco.mjtSensor(stype).name
    except Exception:
        return f"sensor({stype})"


def inspect_model(model_path: Path) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lines = []
    lines.append("=" * 80)
    lines.append("ASIMOV 1 MUJOCO MODEL INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Source Model Path : {model_path.resolve()}")
    lines.append(f"MuJoCo Version    : {mujoco.__version__}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. MODEL SUMMARY DIMENSIONS")
    lines.append("-" * 80)
    lines.append(f"  nq (generalized coords) : {model.nq}")
    lines.append(f"  nv (degrees of freedom) : {model.nv}")
    lines.append(f"  nu (actuator count)     : {model.nu}")
    lines.append(f"  nbody (bodies count)    : {model.nbody}")
    lines.append(f"  njnt (joints count)     : {model.njnt}")
    lines.append(f"  ngeom (geoms count)     : {model.ngeom}")
    lines.append(f"  nsensor (sensors count) : {model.nsensor}")
    lines.append(f"  opt.timestep            : {model.opt.timestep:.6f} s ({(1.0 / model.opt.timestep):.1f} Hz)")
    lines.append(f"  opt.integrator          : {model.opt.integrator}")
    lines.append(f"  opt.solver              : {model.opt.solver}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. FLOATING BASE (FREE JOINT) DETAILS")
    lines.append("-" * 80)
    free_joints = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if free_joints:
        for fj_idx in free_joints:
            fj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, fj_idx)
            fj_body_id = model.jnt_bodyid[fj_idx]
            fj_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, fj_body_id)
            qpos_adr = model.jnt_qposadr[fj_idx]
            dof_adr = model.jnt_dofadr[fj_idx]
            init_pos = model.body_pos[fj_body_id]
            init_quat = model.body_quat[fj_body_id]
            lines.append(f"  Found floating base joint at index {fj_idx}:")
            lines.append(f"    Joint Name      : {fj_name}")
            lines.append(f"    Attached Body   : id={fj_body_id} ({fj_body_name})")
            lines.append(f"    qpos range      : [{qpos_adr} : {qpos_adr + 7}] (3 position + 4 quaternion coords)")
            lines.append(f"    dof range       : [{dof_adr} : {dof_adr + 6}] (3 translation + 3 rotation DOFs)")
            lines.append(f"    Initial Pos [m] : x={init_pos[0]:.6f}, y={init_pos[1]:.6f}, z={init_pos[2]:.6f}")
            lines.append(f"    Initial Quat    : w={init_quat[0]:.6f}, x={init_quat[1]:.6f}, y={init_quat[2]:.6f}, z={init_quat[3]:.6f}")
    else:
        lines.append("  No free joint detected (model is fixed-base).")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. INERTIAL & ROOT BODY PROPERTIES")
    lines.append("-" * 80)
    total_mass = float(sum(model.body_mass))
    root_id = 1 if model.nbody > 1 else 0
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_id)
    root_mass = float(model.body_mass[root_id])
    root_pos = model.body_pos[root_id]
    root_quat = model.body_quat[root_id]

    # Calculate center of mass in initial zero pose
    com = np.zeros(3)
    for b in range(1, model.nbody):
        com += model.body_mass[b] * data.xipos[b]
    com /= total_mass

    lines.append(f"  Total Robot Mass : {total_mass:.6f} kg")
    lines.append(f"  Root Body ID     : {root_id} ({root_name})")
    lines.append(f"  Root Body Mass   : {root_mass:.6f} kg")
    lines.append(f"  Root Body Pos    : [{root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}] m")
    lines.append(f"  Root Body Quat   : [{root_quat[0]:.6f}, {root_quat[1]:.6f}, {root_quat[2]:.6f}, {root_quat[3]:.6f}]")
    lines.append(f"  Robot CoM (init) : [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}] m")
    lines.append("")

    lines.append("-" * 80)
    lines.append("4. JOINTS TABLE (njnt = 24)")
    lines.append("-" * 80)
    lines.append(
        f"{'Idx':>3} | {'Joint Name':<28} | {'Type':<6} | {'qpos':<5} | {'dof':<4} | "
        f"{'Range [rad]':<22} | {'Damping':<8} | {'Armature':<10} | {'Friction':<8}"
    )
    lines.append("-" * 105)

    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        jtype = get_joint_type_name(model.jnt_type[i])
        qpos_adr = model.jnt_qposadr[i]
        dof_adr = model.jnt_dofadr[i]

        if model.jnt_limited[i]:
            jrange = f"[{model.jnt_range[i][0]:+.4f}, {model.jnt_range[i][1]:+.4f}]"
        else:
            jrange = "unlimited"

        damping = model.dof_damping[dof_adr]
        armature = model.dof_armature[dof_adr]
        friction = model.dof_frictionloss[dof_adr]

        lines.append(
            f"{i:3d} | {name:<28} | {jtype:<6} | {qpos_adr:5d} | {dof_adr:4d} | "
            f"{jrange:<22} | {damping:8.4f} | {armature:10.6f} | {friction:8.4f}"
        )
    lines.append("")

    lines.append("-" * 80)
    lines.append("5. ACTUATORS TABLE (nu = 0)")
    lines.append("-" * 80)
    if model.nu == 0:
        lines.append("  Notice: model.nu = 0. No actuators are defined in this XML file.")
        lines.append("  Explanation from sim-model/README.md:")
        lines.append("    'No actuators are defined in the XML — the training sim configures them in Python.'")
        lines.append("    'Torque motors for the standalone viewer (training adds position actuators")
        lines.append("     programmatically via BuiltinPositionActuatorCfg; the viewer applies its own")
        lines.append("     PD in JS and writes torque to ctrl).'")
        lines.append("  In simulation, joint-space torques are applied directly to data.qfrc_applied.")
    else:
        lines.append(
            f"{'Idx':>3} | {'Actuator Name':<25} | {'Type':<10} | {'Joint':<25} | "
            f"{'ctrlrange':<20} | {'forcerange':<20} | {'kp':<6} | {'kv':<6}"
        )
        lines.append("-" * 125)
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            trntype = model.actuator_trntype[i]
            target_jnt_id = model.actuator_trnid[i, 0]
            target_jnt = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_jnt_id) if target_jnt_id >= 0 else "N/A"
            ctrl_lim = bool(model.actuator_ctrllimited[i])
            ctrl_range = f"[{model.actuator_ctrlrange[i][0]:.2f}, {model.actuator_ctrlrange[i][1]:.2f}]" if ctrl_lim else "unlimited"
            frc_lim = bool(model.actuator_forcelimited[i])
            frc_range = f"[{model.actuator_forcerange[i][0]:.2f}, {model.actuator_forcerange[i][1]:.2f}]" if frc_lim else "unlimited"
            kp = model.actuator_gainprm[i, 0]
            kv = model.actuator_biasprm[i, 2] if model.actuator_biastype[i] != 0 else 0.0
            lines.append(
                f"{i:3d} | {act_name:<25} | {trntype:<10} | {target_jnt:<25} | "
                f"{ctrl_range:<20} | {frc_range:<20} | {kp:6.2f} | {kv:6.2f}"
            )
    lines.append("")

    lines.append("-" * 80)
    lines.append("6. SENSORS TABLE (nsensor = 5)")
    lines.append("-" * 80)
    lines.append(f"{'Idx':>3} | {'Sensor Name':<20} | {'Sensor Type':<22} | {'Dim':>3} | {'Attached Site/Body':<20}")
    lines.append("-" * 75)
    for i in range(model.nsensor):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
        stype = get_sensor_type_name(model.sensor_type[i])
        dim = model.sensor_dim[i]
        obj_id = model.sensor_objid[i]
        obj_type = model.sensor_objtype[i]
        if obj_type == mujoco.mjtObj.mjOBJ_SITE:
            obj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, obj_id) or f"site_{obj_id}"
        elif obj_type == mujoco.mjtObj.mjOBJ_BODY:
            obj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, obj_id) or f"body_{obj_id}"
        else:
            obj_name = f"obj_{obj_id}"
        lines.append(f"{i:3d} | {sname:<20} | {stype:<22} | {dim:3d} | {obj_name:<20}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("7. BODIES INERTIAL BREAKDOWN (nbody = 27)")
    lines.append("-" * 80)
    lines.append(f"{'ID':>3} | {'Body Name':<28} | {'Mass [kg]':>10} | {'Pos [m] (rel to parent)':<35}")
    lines.append("-" * 82)
    for b in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
        bmass = model.body_mass[b]
        bpos = model.body_pos[b]
        pos_str = f"[{bpos[0]:+.4f}, {bpos[1]:+.4f}, {bpos[2]:+.4f}]"
        lines.append(f"{b:3d} | {bname:<28} | {bmass:10.6f} | {pos_str:<35}")
    lines.append("-" * 80)
    lines.append(f"Total Model Mass (sum of bodies): {total_mass:.6f} kg")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect MuJoCo model for Asimov 1 humanoid.")
    default_model = Path(__file__).resolve().parent / "upstream" / "asimov-1" / "sim-model" / "xmls" / "asimov_1.xml"
    default_output = Path(__file__).resolve().parent / "reports" / "model_inspection.txt"

    parser.add_argument("--model", type=Path, default=default_model, help="Path to MuJoCo MJCF XML model")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to write inspection report")
    args = parser.parse_args()

    report_text = inspect_model(args.model)
    print(report_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Saved report to {args.output}]")


if __name__ == "__main__":
    main()
