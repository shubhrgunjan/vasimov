#!/usr/bin/env python3
"""
vasimov/stand.py
Joint-Space PD Standing Controller and Push Disturbance Evaluation for Virtual Asimov 1.

Features (Updated for Step 3 Task 0):
- Uses derived model: vasimov/model/asimov_1_vasimov.xml (25 actuated hinge joints).
- Actuator torque commands driven via MotorModel (vasimov/motor_model.py).
- Official default standing stance (isaac_asimov/asimov_1.py ASIMOV_1_STANDING_INIT_STATE).
- Configurable PD update rate:
    * 'per_physics_step' (200 Hz): matches official training sims (asimov-mjlab BuiltinPositionActuator & isaac_asimov ImplicitActuatorCfg)
    * 'per_control_tick' (50 Hz): holds torque constant across 4 physics sub-steps.
- Settle-based PASS/FAIL criteria:
    * Height drift < 1 mm/s over last 3 seconds of simulation.
    * Pitch within 0.1 deg of its mean over last 3 seconds.
    * Initial drop (spawn height 0.639m to settled height) reported separately.
- Push disturbance metrics:
    * Peak horizontal and vertical deviation from pre-push state.
    * Time to return within 0.2 deg / 2 mm of pre-push value.
    * Time to settle within 0.1 deg / 1 mm of final mean.
    * Report-only push sweep: 40, 80, 120, 160 N for 0.1 s in sagittal and lateral directions.
- L1 toggle (--l1): compares baseline torque clip vs velocity-dependent derating envelope.
- Interactive viewer mode (--viewer) launched with mjpython.
- Exports detailed trajectory log to CSV.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
import mujoco
import mujoco.viewer

# Add vasimov root to path
_VASIMOV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_VASIMOV_DIR))
from motor_model import MotorModel, MotorState


def quat_to_roll_pitch(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert MuJoCo quaternion [w, x, y, z] to roll, pitch, and total tilt in degrees.
    """
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp) * 180.0 / np.pi

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0)) * 180.0 / np.pi

    # Exact tilt: angle between pelvis up-axis (local [0, 0, 1]) and world up [0, 0, 1]
    # In world coordinates, pelvis local z-axis dot [0, 0, 1] is R[2, 2] = (w^2 - x^2 - y^2 + z^2) / norm^2
    norm_sq = w * w + x * x + y * y + z * z
    pelvis_up_z = (w * w - x * x - y * y + z * z) / norm_sq if norm_sq > 1e-9 else 1.0
    tilt = float(np.arccos(np.clip(pelvis_up_z, -1.0, 1.0)) * 180.0 / np.pi)

    return float(roll), float(pitch), float(tilt)


def run_stand_sim(
    model_path: Path,
    config_path: Path,
    output_csv: Optional[Path] = None,
    duration: float = 10.0,
    log_interval: float = 0.5,
    decimation: int = 4,
    pd_update: str = "per_physics_step",
    enable_l1: bool = False,
    apply_push: bool = False,
    push_force: float = 40.0,
    push_time: float = 3.0,
    push_duration: float = 0.1,
    push_dir: str = "x",
    use_viewer: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Execute standing simulation and disturbance test.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        gains_cfg = yaml.safe_load(f)

    # Initialize MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Initialize MotorModel
    motor_model = MotorModel(
        motors_yaml_path=_VASIMOV_DIR / "config" / "motors.yaml",
        joints_yaml_path=_VASIMOV_DIR / "config" / "joints.yaml",
        gains_yaml_path=config_path,
    )
    motor_model.set_layer_l1(enable_l1)

    # Setup initial floating base pose
    root_z = float(gains_cfg.get("default_pose", {}).get("root_z", 0.639))
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = root_z
    data.qpos[3] = 1.0  # quat w
    data.qpos[4:7] = 0.0  # quat x, y, z
    data.qvel[:] = 0.0

    # Build actuator target arrays matching the 25 model actuators
    pose_cfg = gains_cfg.get("default_pose", {}).get("joints", {})
    num_act = model.nu
    assert num_act == 25, f"Expected 25 actuators, found {num_act}"

    q_targets = np.zeros(num_act, dtype=np.float64)
    qdot_targets = np.zeros(num_act, dtype=np.float64)
    actuator_joint_qposadr = []
    actuator_joint_dofadr = []

    for i in range(num_act):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        qpos_adr = model.jnt_qposadr[jnt_id]
        dof_adr = model.jnt_dofadr[jnt_id]
        actuator_joint_qposadr.append(qpos_adr)
        actuator_joint_dofadr.append(dof_adr)

        target_val = float(pose_cfg.get(jnt_name, 0.0))
        q_targets[i] = target_val
        data.qpos[qpos_adr] = target_val

    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if model.neq > 0:
        data.eq_active[:] = 0
    mujoco.mj_forward(model, data)

    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if pelvis_body_id == -1:
        pelvis_body_id = 1

    sim_dt = model.opt.timestep
    total_steps = int(np.ceil(duration / sim_dt))
    control_step_interval = max(1, int(decimation))

    # Metric tracking
    initial_spawn_z = float(data.qpos[2])
    max_tilt_deg = 0.0
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    max_torque_applied = 0.0
    fell = False
    l1_bound = False

    # Push tracking
    pre_push_pos = np.zeros(3)
    pre_push_roll = 0.0
    pre_push_pitch = 0.0
    pre_push_tilt = 0.0
    push_peak_disp = 0.0
    push_peak_tilt = 0.0
    push_peak_tilt_dev = 0.0
    push_peak_roll_dev = 0.0
    push_peak_pitch_dev = 0.0
    push_max_torque = 0.0
    return_time: Optional[float] = None
    settle_time: Optional[float] = None
    pre_push_captured = False

    history_t: List[float] = []
    history_z: List[float] = []
    history_pitch: List[float] = []
    history_tilt: List[float] = []

    csv_rows: List[Dict[str, Any]] = []
    last_log_t = -1.0

    if not quiet:
        print("=" * 80)
        print(f"RUNNING ASIMOV 1 STAND TEST ({'WITH' if enable_l1 else 'WITHOUT'} L1 MOTOR DERATING)")
        print(f"Model: {model_path.name} | Actuators: {num_act} | Physics dt: {sim_dt}s ({1.0/sim_dt:.0f} Hz)")
        print(f"PD Update Mode: {pd_update} | Decimation: {control_step_interval} ({1.0/(sim_dt*control_step_interval):.0f} Hz policy tick)")
        print(f"Spawn Height: {initial_spawn_z:.4f} m (Bent-knee Stance)")
        if apply_push:
            print(f"Disturbance: {push_force:.1f} N in +{push_dir} at t={push_time:.2f}s for {push_duration:.2f}s")
        print("-" * 80)
        print(f"{'Time (s)':<10} {'Pelvis Z (m)':<14} {'Roll (deg)':<12} {'Pitch (deg)':<12} {'Tilt (deg)':<12} {'Max Torque':<12} {'Push'}")
        print("-" * 80)

    cached_torques = np.zeros(num_act, dtype=np.float64)

    def step_simulation(step_idx: int) -> None:
        nonlocal max_tilt_deg, max_roll_deg, max_pitch_deg, max_torque_applied, fell, l1_bound
        nonlocal pre_push_pos, pre_push_roll, pre_push_pitch, pre_push_tilt, push_peak_disp, push_peak_tilt
        nonlocal push_peak_tilt_dev, push_peak_roll_dev, push_peak_pitch_dev, push_max_torque
        nonlocal return_time, settle_time, pre_push_captured, last_log_t, cached_torques

        current_t = data.time

        # 1. Update PD torque
        update_pd_now = False
        if pd_update == "per_physics_step":
            update_pd_now = True
        elif pd_update == "per_control_tick":
            update_pd_now = (step_idx % control_step_interval == 0)

        if update_pd_now:
            q_current = np.array([data.qpos[adr] for adr in actuator_joint_qposadr], dtype=np.float64)
            qdot_current = np.array([data.qvel[adr] for adr in actuator_joint_dofadr], dtype=np.float64)

            torques, states = motor_model.step(
                q=q_current,
                qdot=qdot_current,
                q_des=q_targets,
                qdot_des=qdot_targets,
                dt=sim_dt if pd_update == "per_physics_step" else (sim_dt * control_step_interval),
            )
            cached_torques = torques
            data.ctrl[:] = cached_torques

            # Check if L1 torque limit bound (derating was active and clipped command)
            if enable_l1:
                for s in states.values():
                    lim = motor_model.compute_torque_limit(motor_model.joint_name_to_idx[s.joint_name], s.velocity)
                    if lim < s.effort_limit - 0.1 and abs(s.raw_pd_torque) > lim:
                        l1_bound = True
        else:
            # Hold cached torques
            data.ctrl[:] = cached_torques

        # 2. Handle push disturbance pulse
        is_pushing = False
        data.xfrc_applied[:] = 0.0
        if apply_push:
            if current_t < push_time:
                pre_push_pos = np.copy(data.qpos[:3])
                pre_push_roll, pre_push_pitch, pre_push_tilt = quat_to_roll_pitch(data.qpos[3:7])
                pre_push_captured = True
            elif push_time <= current_t < (push_time + push_duration):
                is_pushing = True
                f_idx = 0 if push_dir == "x" else 1
                data.xfrc_applied[pelvis_body_id, f_idx] = push_force

        # 3. Advance physics
        mujoco.mj_step(model, data)

        # 4. Measure state
        pelvis_pos = data.qpos[:3]
        current_z = float(pelvis_pos[2])
        roll_d, pitch_d, tilt_d = quat_to_roll_pitch(data.qpos[3:7])
        peak_tq = float(np.max(np.abs(cached_torques)))

        history_t.append(current_t)
        history_z.append(current_z)
        history_pitch.append(pitch_d)
        history_tilt.append(tilt_d)

        max_tilt_deg = max(max_tilt_deg, tilt_d)
        max_roll_deg = max(max_roll_deg, abs(roll_d))
        max_pitch_deg = max(max_pitch_deg, abs(pitch_d))
        max_torque_applied = max(max_torque_applied, peak_tq)

        # Fall threshold (pelvis height collapsed below 0.40m or tilt > 30 deg)
        if current_z < 0.40 or tilt_d > 30.0:
            fell = True

        # 5. Push response metrics (evaluated after push start)
        if apply_push and current_t >= push_time and pre_push_captured:
            disp = float(np.linalg.norm(pelvis_pos - pre_push_pos))
            push_peak_disp = max(push_peak_disp, disp)
            push_peak_tilt = max(push_peak_tilt, tilt_d)

            roll_dev = abs(roll_d - pre_push_roll)
            pitch_dev = abs(pitch_d - pre_push_pitch)
            tilt_dev = abs(tilt_d - pre_push_tilt)
            height_dev = abs(current_z - pre_push_pos[2])

            push_peak_roll_dev = max(push_peak_roll_dev, roll_dev)
            push_peak_pitch_dev = max(push_peak_pitch_dev, pitch_dev)
            push_peak_tilt_dev = max(push_peak_tilt_dev, tilt_dev)

            # Torque stats only after push start
            push_max_torque = max(push_max_torque, peak_tq)

            # Return time: defined on tilt deviation (<0.2 deg) and height (<2 mm), N/A for falls
            if current_t > (push_time + push_duration) and not fell:
                if height_dev <= 0.002 and tilt_dev <= 0.2:
                    if return_time is None:
                        return_time = current_t - (push_time + push_duration)

        # 6. Logging
        if not quiet and (current_t - last_log_t >= log_interval or current_t == 0.0):
            last_log_t = current_t
            push_str = "PUSHING" if is_pushing else ("FELL" if fell else "")
            print(f"{current_t:<10.2f} {current_z:<14.4f} {roll_d:<12.2f} {pitch_d:<12.2f} {tilt_d:<12.2f} {peak_tq:<12.1f} {push_str}")

        if output_csv is not None:
            csv_rows.append({
                "time": round(current_t, 4),
                "pelvis_x": round(float(pelvis_pos[0]), 5),
                "pelvis_y": round(float(pelvis_pos[1]), 5),
                "pelvis_z": round(float(pelvis_pos[2]), 5),
                "roll_deg": round(roll_d, 3),
                "pitch_deg": round(pitch_d, 3),
                "tilt_deg": round(tilt_d, 3),
                "max_torque_nm": round(peak_tq, 2),
                "is_pushing": int(is_pushing),
            })

    # Execute simulation
    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for step in range(total_steps):
                step_start = time.time()
                step_simulation(step)
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < sim_dt:
                    time.sleep(sim_dt - elapsed)
                if not viewer.is_running():
                    break
    else:
        for step in range(total_steps):
            step_simulation(step)

    # Save CSV
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
        if not quiet:
            print(f"\n[OK] Trajectory log written to: {output_csv}")

    # Settle-based metrics over the last 3.0 seconds (t >= 7.0s)
    last3_indices = [i for i, t in enumerate(history_t) if t >= 7.0]
    if len(last3_indices) > 0:
        last3_z = [history_z[i] for i in last3_indices]
        last3_p = [history_pitch[i] for i in last3_indices]
        
        settled_height = float(np.mean(last3_z))
        drop_mm = (initial_spawn_z - settled_height) * 1000.0
        drift_rate_mm_s = abs(last3_z[-1] - last3_z[0]) / (history_t[last3_indices[-1]] - history_t[last3_indices[0]]) * 1000.0
        mean_pitch = float(np.mean(last3_p))
        max_pitch_dev_deg = float(np.max(np.abs(np.array(last3_p) - mean_pitch)))
    else:
        settled_height = float(data.qpos[2])
        drop_mm = 0.0
        drift_rate_mm_s = 999.0
        mean_pitch = 0.0
        max_pitch_dev_deg = 999.0

    # PASS criteria (settle-based per Task 0.2):
    # 1. Height drift < 1 mm/s over last 3 s
    # 2. Pitch within 0.1 deg of its mean over last 3 s
    # 3. Robot did not fall
    drift_pass = drift_rate_mm_s < 1.0
    pitch_pass = max_pitch_dev_deg < 0.1
    overall_pass = drift_pass and pitch_pass and (not fell)

    # Compute settle time after push (Task 0.1: defined on tilt deviation + height, N/A for falls)
    if apply_push and not fell and pre_push_captured:
        push_end_t = push_time + push_duration
        for i, t in enumerate(history_t):
            if t > push_end_t:
                all_settled = True
                for j in range(i, len(history_t)):
                    h_dev = abs(history_z[j] - pre_push_pos[2])
                    t_dev = abs(history_tilt[j] - pre_push_tilt)
                    if h_dev > 0.001 or t_dev > 0.1:
                        all_settled = False
                        break
                if all_settled:
                    settle_time = t - push_end_t
                    break

    # If the robot fell, return_time and settle_time MUST be None (N/A)
    if fell:
        return_time = None
        settle_time = None

    if not quiet:
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY RESULTS (SETTLE-BASED)")
        print("=" * 80)
        print(f"Spawn Height:           {initial_spawn_z:.4f} m")
        print(f"Settled Height:         {settled_height:.5f} m (Drop: {drop_mm:.2f} mm)")
        print(f"Height Drift (last 3s): {drift_rate_mm_s:.4f} mm/s (PASS criterion: < 1.0 mm/s) -> {'PASS' if drift_pass else 'FAIL'}")
        print(f"Pitch Dev from Mean:    {max_pitch_dev_deg:.4f}° (PASS criterion: < 0.10°) -> {'PASS' if pitch_pass else 'FAIL'}")
        print(f"Maximum Tilt:           {max_tilt_deg:.2f}° (Max Roll: {max_roll_deg:.2f}°, Max Pitch: {max_pitch_deg:.2f}°)")
        print(f"Robot Fell:             {'YES (FELL)' if fell else 'NO'}")
        print(f"Peak Motor Torque:      {max_torque_applied:.1f} N*m")
        if apply_push:
            print("-" * 80)
            print("PUSH DISTURBANCE METRICS (Report-Only)")
            print(f"Force Applied:          {push_force:.1f} N for {push_duration:.2f} s in +{push_dir}")
            print(f"Peak Displacement:      {push_peak_disp * 100.0:.2f} cm")
            print(f"Peak Tilt:              {push_peak_tilt:.2f}° (Dev: {push_peak_tilt_dev:.2f}°)")
            print(f"Peak Roll Deviation:    {push_peak_roll_dev:.2f}°")
            print(f"Peak Pitch Deviation:   {push_peak_pitch_dev:.2f}°")
            print(f"Post-Push Peak Torque:  {push_max_torque:.1f} N*m")
            print(f"Time to Return (<2mm/<0.2° tilt): {f'{return_time:.3f} s' if return_time is not None else 'N/A'}")
            print(f"Time to Settle (<1mm/<0.1° tilt): {f'{settle_time:.3f} s' if settle_time is not None else 'N/A'}")
            print(f"L1 Torque Saturation Bound:  {l1_bound}")
        print("=" * 80)
        print(f"OVERALL VERDICT:        {'PASS' if overall_pass else 'FAIL'}")
        print("=" * 80 + "\n")

    return {
        "pass": overall_pass,
        "drift_pass": drift_pass,
        "pitch_pass": pitch_pass,
        "fell": fell,
        "spawn_height_m": initial_spawn_z,
        "settled_height_m": settled_height,
        "drop_mm": drop_mm,
        "drift_rate_mm_s": drift_rate_mm_s,
        "max_pitch_dev_deg": max_pitch_dev_deg,
        "max_tilt_deg": max_tilt_deg,
        "max_torque": push_max_torque if apply_push else max_torque_applied,
        "push_peak_disp_cm": push_peak_disp * 100.0,
        "push_peak_tilt_deg": push_peak_tilt,
        "push_peak_tilt_dev_deg": push_peak_tilt_dev,
        "push_peak_roll_dev_deg": push_peak_roll_dev,
        "push_peak_pitch_dev_deg": push_peak_pitch_dev,
        "return_time_s": return_time if return_time is not None else -1.0,
        "settle_time_s": settle_time if settle_time is not None else -1.0,
        "l1_bound": l1_bound,
    }


def run_push_sweep(model_path: Path, config_path: Path, csv_output: Optional[Path] = None) -> None:
    """Run full push sweep across 40, 80, 120, 160 N in sagittal and lateral."""
    forces = [40.0, 80.0, 120.0, 160.0]
    directions = ["x", "y"]

    if csv_output is None:
        csv_output = Path(__file__).resolve().parent / "reports" / "raw" / "push_sweep.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 115)
    print("TASK 0.1 PUSH SWEEP BENCHMARK (40, 80, 120, 160 N for 0.1 s; tilt = angle pelvis up to world up)")
    print(f"Output CSV: {csv_output}")
    print("=" * 115)
    print(f"{'Dir':<5} {'Force':<8} {'Status':<10} {'Disp(cm)':<10} {'Tilt(deg)':<10} {'TiltDev':<10} {'RollDev':<10} {'PitchDev':<10} {'Return':<10} {'Settle':<10} {'MaxTau':<8} {'L1'}")
    print("-" * 115)

    first_fall: Dict[str, Optional[float]] = {"x": None, "y": None}
    csv_rows = []

    for p_dir in directions:
        for f in forces:
            traj_csv0 = csv_output.parent / f"push_run_{p_dir}_{int(f)}N.csv"
            traj_csv1 = csv_output.parent / f"push_run_{p_dir}_{int(f)}N_l1.csv"

            # Run without L1
            res0 = run_stand_sim(
                model_path=model_path,
                config_path=config_path,
                output_csv=traj_csv0,
                duration=10.0,
                apply_push=True,
                push_force=f,
                push_dir=p_dir,
                enable_l1=False,
                quiet=True,
            )
            # Run with L1
            res1 = run_stand_sim(
                model_path=model_path,
                config_path=config_path,
                output_csv=traj_csv1,
                duration=10.0,
                apply_push=True,
                push_force=f,
                push_dir=p_dir,
                enable_l1=True,
                quiet=True,
            )

            status = "FELL" if res0["fell"] else "SURVIVED"
            ret_str = f"{res0['return_time_s']:.3f}" if res0["return_time_s"] >= 0 else "N/A"
            set_str = f"{res0['settle_time_s']:.3f}" if res0["settle_time_s"] >= 0 else "N/A"
            l1_str = str(res1["l1_bound"])

            if res0["fell"] and first_fall[p_dir] is None:
                first_fall[p_dir] = f

            print(f"{p_dir:<5} {f:<8.0f} {status:<10} {res0['push_peak_disp_cm']:<10.2f} {res0['push_peak_tilt_deg']:<10.2f} {res0['push_peak_tilt_dev_deg']:<10.2f} {res0['push_peak_roll_dev_deg']:<10.2f} {res0['push_peak_pitch_dev_deg']:<10.2f} {ret_str:<10} {set_str:<10} {res0['max_torque']:<8.1f} {l1_str}")

            csv_rows.append({
                "direction": p_dir,
                "force_n": f,
                "status": status,
                "fell": res0["fell"],
                "peak_disp_cm": f"{res0['push_peak_disp_cm']:.2f}",
                "peak_tilt_deg": f"{res0['push_peak_tilt_deg']:.2f}",
                "peak_tilt_dev_deg": f"{res0['push_peak_tilt_dev_deg']:.2f}",
                "peak_roll_dev_deg": f"{res0['push_peak_roll_dev_deg']:.2f}",
                "peak_pitch_dev_deg": f"{res0['push_peak_pitch_dev_deg']:.2f}",
                "return_time_s": ret_str,
                "settle_time_s": set_str,
                "max_torque_nm": f"{res0['max_torque']:.2f}",
                "l1_bound": res1["l1_bound"],
                "knee_effort_limit_nm": 45.0,
                "knee_velocity_limit_rad_s": 12.25,
                "trajectory_file": traj_csv0.name,
            })

    print("-" * 115)
    print("FIRST FALL SUMMARY:")
    print(f"  * Sagittal (+x): First fall at {first_fall['x']} N")
    print(f"  * Lateral (+y):  First fall at {first_fall['y']} N")
    print("=" * 115)

    import csv
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "direction", "force_n", "status", "fell", "peak_disp_cm",
            "peak_tilt_deg", "peak_tilt_dev_deg", "peak_roll_dev_deg", "peak_pitch_dev_deg",
            "return_time_s", "settle_time_s", "max_torque_nm",
            "l1_bound", "knee_effort_limit_nm", "knee_velocity_limit_rad_s", "trajectory_file"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Successfully saved push sweep results to: {csv_output}")


def main():
    parser = argparse.ArgumentParser(description="Asimov 1 MuJoCo Standing Controller")
    parser.add_argument("--model", type=str, default="vasimov/model/asimov_1_vasimov.xml", help="Path to derived MJCF model")
    parser.add_argument("--config", type=str, default="vasimov/config/gains.yaml", help="Path to gains and pose config")
    parser.add_argument("--csv", type=str, default="vasimov/reports/stand_run.csv", help="Path to export CSV log")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--log-interval", type=float, default=0.5, help="Console log interval (s)")
    parser.add_argument("--decimation", type=int, default=4, help="Control decimation steps (default: 4 for 50Hz)")
    parser.add_argument("--pd-update", type=str, default="per_physics_step", choices=["per_physics_step", "per_control_tick"],
                        help="PD update frequency: per_physics_step (200Hz, official sims) or per_control_tick (50Hz)")
    parser.add_argument("--l1", action="store_true", help="Enable Layer 1 velocity-dependent torque limit")
    parser.add_argument("--push", action="store_true", help="Apply horizontal push disturbance to pelvis")
    parser.add_argument("--push-force", type=float, default=40.0, help="Push force in Newtons (default 40.0 N)")
    parser.add_argument("--push-time", type=float, default=3.0, help="Time to initiate push in seconds")
    parser.add_argument("--push-duration", type=float, default=0.1, help="Duration of push force in seconds")
    parser.add_argument("--push-dir", type=str, default="x", choices=["x", "y"], help="Push force direction")
    parser.add_argument("--sweep", action="store_true", help="Execute complete 40-160N push disturbance sweep in x and y")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer (use mjpython)")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.exists():
        candidate = script_dir / "model" / "asimov_1_vasimov.xml"
        if candidate.exists():
            model_path = candidate
        else:
            candidate_ws = script_dir / args.model
            if candidate_ws.exists():
                model_path = candidate_ws

    config_path = Path(args.config)
    if not config_path.exists():
        candidate = script_dir / "config" / "gains.yaml"
        if candidate.exists():
            config_path = candidate

    csv_path = Path(args.csv)
    if not csv_path.is_absolute() and not csv_path.parent.exists():
        candidate = script_dir / "reports" / csv_path.name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        csv_path = candidate if args.csv else None

    if args.sweep:
        run_push_sweep(model_path, config_path)
        sys.exit(0)

    results = run_stand_sim(
        model_path=model_path,
        config_path=config_path,
        output_csv=csv_path,
        duration=args.duration,
        log_interval=args.log_interval,
        decimation=args.decimation,
        pd_update=args.pd_update,
        enable_l1=args.l1,
        apply_push=args.push,
        push_force=args.push_force,
        push_time=args.push_time,
        push_duration=args.push_duration,
        push_dir=args.push_dir,
        use_viewer=args.viewer,
    )

    sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
