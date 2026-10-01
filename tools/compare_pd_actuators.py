#!/usr/bin/env python3
"""Compare external explicit-torque PD vs MuJoCo-native <position kp kv> actuators."""

import csv
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import yaml

VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VASIMOV_DIR))
from motor_model import MotorModel
from stand import quat_to_roll_pitch

VASIMOV_DIR = Path(__file__).resolve().parent.parent
BASE_XML = VASIMOV_DIR / "model" / "asimov_1_vasimov.xml"
THROWAWAY_XML = VASIMOV_DIR / "model" / "throwaway_native_pos.xml"
CONFIG_PATH = VASIMOV_DIR / "config" / "gains.yaml"
OUTPUT_CSV = VASIMOV_DIR / "reports" / "raw" / "pd_parity_comparison.csv"


def build_throwaway_model():
    """Convert <motor> actuators to <position kp kv> actuators in a throwaway XML."""
    tree = ET.parse(BASE_XML)
    root = tree.getroot()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gains = cfg.get("gains", {})

    actuators_elem = root.find("actuator")
    assert actuators_elem is not None, "No <actuator> found"

    motors = list(actuators_elem.findall("motor"))
    for m in motors:
        actuators_elem.remove(m)
        name = m.get("name")
        joint = m.get("joint")
        forcerange = m.get("forcerange", "-40 40")
        
        # Get kp, kd from config
        j_gains = gains.get(joint, {})
        kp = j_gains.get("kp", 250.0)
        kd = j_gains.get("kd", 5.0)

        # Create position actuator
        pos_elem = ET.SubElement(actuators_elem, "position")
        pos_elem.set("name", name)
        pos_elem.set("joint", joint)
        pos_elem.set("kp", str(kp))
        pos_elem.set("kv", str(kd))
        pos_elem.set("forcelimited", "true")
        pos_elem.set("forcerange", forcerange)

    tree.write(THROWAWAY_XML, encoding="utf-8", xml_declaration=True)
    print(f"Created throwaway model with native <position> actuators: {THROWAWAY_XML}")


def run_test(model_path: Path, is_native: bool, apply_push: bool = False, push_force: float = 0.0, push_dir: str = "x"):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Disable gantry equality if present
    if model.neq > 0:
        data.eq_active[:] = 0

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pose_cfg = cfg.get("default_pose", {}).get("joints", {})
    root_z = float(cfg.get("default_pose", {}).get("root_z", 0.639))

    data.qpos[0:3] = [0.0, 0.0, root_z]
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.qvel[:] = 0.0

    targets = np.zeros(25, dtype=np.float64)
    for i in range(25):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        adr = model.jnt_qposadr[jnt_id]
        val = float(pose_cfg.get(jnt_name, 0.0))
        targets[i] = val
        data.qpos[adr] = val

    mujoco.mj_forward(model, data)

    motor_model = MotorModel(gains_yaml_path=CONFIG_PATH)
    motor_model.set_layer_l1(False)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    sim_dt = model.opt.timestep
    duration = 10.0
    total_steps = int(np.ceil(duration / sim_dt))
    push_steps = int(round(0.1 / sim_dt))
    push_start_step = int(round(3.0 / sim_dt))

    heights = []
    pitches = []
    max_torque = 0.0
    fell = False

    for step in range(total_steps):
        t = step * sim_dt
        if apply_push and push_start_step <= step < push_start_step + push_steps:
            f_vec = [push_force, 0.0, 0.0, 0.0, 0.0, 0.0] if push_dir == "x" else [0.0, push_force, 0.0, 0.0, 0.0, 0.0]
            data.xfrc_applied[pelvis_id, :] = f_vec
        else:
            data.xfrc_applied[pelvis_id, :] = 0.0

        if is_native:
            # Native position actuators: ctrl is desired position
            data.ctrl[:] = targets
        else:
            # External torque PD: compute torque explicitly
            current_q = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(25)])
            current_qdot = np.array([data.qvel[model.jnt_dofadr[model.actuator_trnid[i, 0]]] for i in range(25)])
            torques, _ = motor_model.step(q=current_q, qdot=current_qdot, q_des=targets, qdot_des=np.zeros(25), dt=sim_dt)
            data.ctrl[:] = torques

        mujoco.mj_step(model, data)

        tau_norm = float(np.max(np.abs(data.actuator_force if is_native else data.ctrl)))
        if tau_norm > max_torque:
            max_torque = tau_norm

        z = float(data.qpos[2])
        r, p, tilt = quat_to_roll_pitch(data.qpos[3:7])
        heights.append(z)
        pitches.append(p)
        if z < 0.35 or tilt > 60.0:
            fell = True

    settled_z = float(np.mean(heights[-int(3.0 / sim_dt):]))
    drift_rate = float(abs(heights[-1] - heights[-int(3.0 / sim_dt)]) / 3.0 * 1000.0)  # mm/s
    pitch_osc = float(np.max(np.abs(pitches[-int(3.0 / sim_dt):] - np.mean(pitches[-int(3.0 / sim_dt):]))))

    return {
        "fell": fell,
        "settled_z": settled_z,
        "drop_mm": (root_z - settled_z) * 1000.0,
        "drift_mm_s": drift_rate,
        "pitch_osc_deg": pitch_osc,
        "max_torque_nm": max_torque,
    }


def main():
    build_throwaway_model()

    scenarios = [
        ("Stand Settle (No Push)", False, 0.0, "x"),
        ("120 N Sagittal Push (+x)", True, 120.0, "x"),
        ("160 N Lateral Push (+y)", True, 160.0, "y"),
    ]

    print("\n" + "=" * 90)
    print("TASK 0.2: EXTERNAL EXPLICIT-TORQUE PD VS MUJOCO NATIVE <position kp kv>")
    print("=" * 90)
    print(f"{'Scenario':<30} {'Model':<12} {'Fell':<8} {'Settled Z':<12} {'Drop (mm)':<12} {'Drift (mm/s)':<14} {'Max Tau (N.m)'}")
    print("-" * 90)

    results = []

    for name, push, force, p_dir in scenarios:
        # 1. External torque
        res_ext = run_test(BASE_XML, is_native=False, apply_push=push, push_force=force, push_dir=p_dir)
        # 2. Native position
        res_nat = run_test(THROWAWAY_XML, is_native=True, apply_push=push, push_force=force, push_dir=p_dir)

        for label, r in [("External PD", res_ext), ("Native Pos", res_nat)]:
            print(f"{name:<30} {label:<12} {str(r['fell']):<8} {r['settled_z']:<12.5f} {r['drop_mm']:<12.2f} {r['drift_mm_s']:<14.4f} {r['max_torque_nm']:<10.1f}")
            results.append({
                "scenario": name,
                "actuator_type": label,
                "fell": r["fell"],
                "settled_z_m": f"{r['settled_z']:.5f}",
                "drop_mm": f"{r['drop_mm']:.2f}",
                "drift_mm_s": f"{r['drift_mm_s']:.4f}",
                "pitch_osc_deg": f"{r['pitch_osc_deg']:.4f}",
                "max_torque_nm": f"{r['max_torque_nm']:.2f}",
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "actuator_type", "fell", "settled_z_m", "drop_mm", "drift_mm_s", "pitch_osc_deg", "max_torque_nm"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved raw comparison data to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
