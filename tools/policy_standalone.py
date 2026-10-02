#!/usr/bin/env python3
"""
tools/policy_standalone.py
Standalone Locomotion Policy Harness for Virtual Asimov 1 (Task 2).

Directly runs the official Hugging Face ONNX policy (Menlo/asimov1-locomotion-0818)
inside MuJoCo without requiring Edge or Menlo SDK servers:
- Constructs 78-D observations exactly per the verified contract.
- Runs ONNX inference on CPU at 50 Hz (decimation 4 @ 200 Hz physics).
- Computes target joint positions: q_des = q_default + action_scale * action (scale=0.25).
- Evaluates PD torque control per physics step (dt=0.005 s) with training gains and effort clipping.
- Benchmarks experiments E1 to E7 with full metrics tracking.
- Exports per-run trajectory CSVs to reports/raw/policy_*.csv.
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
import mujoco

try:
    import onnxruntime as ort
except ImportError:
    ort = None

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from stand import quat_to_roll_pitch
from edge.policy_builder import (
    build_policy_observation,
    compute_desired_joint_targets,
    OBS_DIM,
    ACTION_DIM,
    ACTION_SCALE,
)

POLICY_RATE_HZ: float = 50.0
PHYSICS_DT: float = 0.005
DECIMATION: int = 4
FALL_PELVIS_Z_THRESHOLD: float = 0.40
FALL_TILT_DEG_THRESHOLD: float = 30.0


class StandalonePolicyRunner:
    """Standalone harness for running official Asimov 1 ONNX locomotion policy."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
        env_yaml_path: Optional[Path] = None,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required to run StandalonePolicyRunner")

        self.model_path = model_path or (_VASIMOV_DIR / "model" / "asimov_1_vasimov.xml")
        self.policy_path = policy_path or (_VASIMOV_DIR / "assets" / "policy" / "policy.onnx")
        self.env_yaml_path = env_yaml_path or (_VASIMOV_DIR / "assets" / "policy" / "env.yaml")

        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")
        if not self.policy_path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {self.policy_path}")
        if not self.env_yaml_path.exists():
            raise FileNotFoundError(f"env.yaml not found: {self.env_yaml_path}")

        # 1. Initialize ONNX runtime session on CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.policy_path), sess_options=opts, providers=["CPUExecutionProvider"])

        # 2. Parse env.yaml
        with open(self.env_yaml_path, "r", encoding="utf-8") as f:
            self.env_cfg = yaml.unsafe_load(f)

        self.policy_joints: List[str] = list(self.env_cfg["actions"]["joint_pos"]["joint_names"])
        assert len(self.policy_joints) == ACTION_DIM, f"Expected {ACTION_DIM} joints, got {len(self.policy_joints)}"

        # Observation slot groupings
        obs_cfg = self.env_cfg["observations"]["policy"]
        self.slot01_names: List[str] = list(obs_cfg["joint_pos_slot01"]["params"]["asset_cfg"]["joint_names"])
        self.slot23_names: List[str] = list(obs_cfg["joint_pos_slot23"]["params"]["asset_cfg"]["joint_names"])
        self.slot45_names: List[str] = list(obs_cfg["joint_pos_slot45"]["params"]["asset_cfg"]["joint_names"])

        # Default joint positions
        init_pos_cfg = self.env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
        self.default_pos: Dict[str, float] = {}
        for jn in self.policy_joints:
            for k, v in init_pos_cfg.items():
                if re.fullmatch(k, jn):
                    self.default_pos[jn] = float(v)
                    break
            if jn not in self.default_pos:
                self.default_pos[jn] = 0.0

        # Actuator gains and effort limits from env.yaml
        actuators_cfg = self.env_cfg["scene"]["robot"]["actuators"]
        self.joint_gains: Dict[str, Tuple[float, float, float]] = {}  # kp, kd, effort
        for act_group, act_c in actuators_cfg.items():
            exprs = act_c["joint_names_expr"]
            kp = float(act_c["stiffness"])
            kd = float(act_c["damping"])
            eff = float(act_c["effort_limit"])
            for jn in self.policy_joints:
                for expr in exprs:
                    if re.fullmatch(expr, jn):
                        self.joint_gains[jn] = (kp, kd, eff)

        # 3. Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Pelvis ID & Joint addresses
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.jnt_qposadr = {}
        self.jnt_dofadr = {}
        self.jnt_range = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.jnt_qposadr[name] = self.model.jnt_qposadr[i]
            self.jnt_dofadr[name] = self.model.jnt_dofadr[i]
            self.jnt_range[name] = tuple(self.model.jnt_range[i])

        # Actuator IDs
        self.act_jnt_names = []
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            self.act_jnt_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))

        # Sensor addresses
        self.sensor_left_touch = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_touch"),
            1,
        )
        self.sensor_right_touch = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_touch"),
            1,
        )
        self.sensor_left_force = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_force"),
            3,
        )
        self.sensor_right_force = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_force"),
            3,
        )

    def reset(self, root_z: float = 0.639) -> None:
        """Reset robot to upright bent-knee default stance with gantry released."""
        mujoco.mj_resetData(self.model, self.data)

        # Release virtual gantry weld if present
        if self.model.neq > 0:
            self.data.eq_active[:] = 0

        # Pelvis root pose
        self.data.qpos[0:3] = [0.0, 0.0, float(root_z)]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quat [w, x, y, z]
        self.data.qvel[:] = 0.0

        # Initialize joint angles
        for jn, q_val in self.default_pos.items():
            if jn in self.jnt_qposadr:
                self.data.qpos[self.jnt_qposadr[jn]] = q_val

        # Neck joints held at 0
        for neck_jn in ("neck_yaw_joint", "neck_pitch_joint"):
            if neck_jn in self.jnt_qposadr:
                self.data.qpos[self.jnt_qposadr[neck_jn]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def build_observation(
        self,
        command: np.ndarray,
        prev_action: np.ndarray,
    ) -> np.ndarray:
        """
        Build 78-D observation vector using unified edge.policy_builder.
        """
        pelvis_xmat = self.data.xmat[self.pelvis_id]
        pelvis_ang_vel_world = self.data.qvel[3:6]
        joint_pos_map = {jn: float(self.data.qpos[self.jnt_qposadr[jn]]) for jn in self.policy_joints}
        joint_vel_map = {jn: float(self.data.qvel[self.jnt_dofadr[jn]]) for jn in self.policy_joints}

        return build_policy_observation(
            pelvis_xmat=pelvis_xmat,
            pelvis_ang_vel_world=pelvis_ang_vel_world,
            command=command,
            joint_pos_map=joint_pos_map,
            joint_vel_map=joint_vel_map,
            prev_action=prev_action,
            default_pos_map=self.default_pos,
            slot01_names=self.slot01_names,
            slot23_names=self.slot23_names,
            slot45_names=self.slot45_names,
        )

    def run_experiment(
        self,
        duration_s: float,
        command_fn: Any,
        push_schedule: Optional[List[Dict[str, Any]]] = None,
        handoff_s: float = 0.0,
        output_csv: Optional[Path] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an experiment run with detailed metrics logging.
        - command_fn(t): returns np.ndarray([vx, vy, wz])
        - push_schedule: list of { "time": float, "duration": float, "force": float, "dir": "x"|"y" }
        - handoff_s: duration of initial standing PD posture before activating policy
        """
        self.reset()

        total_policy_steps = int(math.ceil(duration_s * POLICY_RATE_HZ))
        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        q_des = {jn: self.default_pos[jn] for jn in self.policy_joints}

        history_rows: List[Dict[str, Any]] = []

        # Tracking metrics
        fell = False
        survival_time = duration_s
        vel_errors_vx: List[float] = []
        vel_errors_vy: List[float] = []
        vel_errors_wz: List[float] = []
        pelvis_heights: List[float] = []
        pelvis_tilts: List[float] = []
        torques_pct_effort: List[float] = []
        joint_limit_violations = 0
        left_contact_steps = 0
        right_contact_steps = 0
        action_diffs: List[float] = []

        for p_step in range(total_policy_steps):
            t_policy = p_step * (1.0 / POLICY_RATE_HZ)

            # Get commanded velocity
            cmd = np.array(command_fn(t_policy), dtype=np.float32)
            assert cmd.shape == (3,), f"Command must be 3-D, got {cmd.shape}"

            # Check fall condition before stepping
            z_curr = float(self.data.qpos[2])
            roll_d, pitch_d, tilt_d = quat_to_roll_pitch(self.data.qpos[3:7])
            if (z_curr < FALL_PELVIS_Z_THRESHOLD or tilt_d > FALL_TILT_DEG_THRESHOLD) and not fell:
                fell = True
                survival_time = t_policy
                if not quiet:
                    print(f"  [FALL] Robot fell at t={t_policy:.2f}s (Z={z_curr:.3f}m, Tilt={tilt_d:.1f}°)")

            # Check if in initial standing handoff mode
            if t_policy < handoff_s:
                # Posture hold
                for jn in self.policy_joints:
                    q_des[jn] = self.default_pos[jn]
            else:
                # 1. Build observation
                obs = self.build_observation(command=cmd, prev_action=prev_action)

                # 2. Run ONNX inference
                raw_act = self.session.run(["actions"], {"obs": obs.reshape(1, OBS_DIM)})[0][0]

                # Action smoothness metric
                if p_step > 0:
                    action_diffs.append(float(np.mean(np.abs(raw_act - prev_action))))
                prev_action = np.copy(raw_act)

                # 3. Compute target joint angles using unified builder
                q_des = compute_desired_joint_targets(
                    raw_actions=raw_act,
                    policy_joints=self.policy_joints,
                    default_pos_map=self.default_pos,
                )

            # Advance physics across decimation steps (4 steps @ 200 Hz = 50 Hz)
            for sub_step in range(DECIMATION):
                t_sub = t_policy + sub_step * PHYSICS_DT

                # Apply push disturbance if scheduled
                self.data.xfrc_applied[self.pelvis_id, :] = 0.0
                if push_schedule:
                    for push in push_schedule:
                        p_t = push["time"]
                        p_dur = push.get("duration", 0.1)
                        if p_t <= t_sub < (p_t + p_dur):
                            f_idx = 0 if push.get("dir", "x") == "x" else 1
                            self.data.xfrc_applied[self.pelvis_id, f_idx] = float(push["force"])

                # PD Control Law per physics step
                step_max_pct = 0.0
                for a_idx, jn in enumerate(self.act_jnt_names):
                    jid = self.model.actuator_trnid[a_idx, 0]
                    if jn in q_des:
                        target = q_des[jn]
                        kp, kd, eff = self.joint_gains[jn]
                    else:
                        target = 0.0
                        kp, kd, eff = 40.0, 2.0, 12.0

                    q_curr = self.data.qpos[self.jnt_qposadr[jn]]
                    qdot_curr = self.data.qvel[self.jnt_dofadr[jn]]

                    tau = kp * (target - q_curr) - kd * qdot_curr
                    clipped_tau = np.clip(tau, -eff, eff)
                    self.data.ctrl[a_idx] = clipped_tau

                    pct = (abs(clipped_tau) / eff) * 100.0
                    if pct > step_max_pct:
                        step_max_pct = pct

                    # Check joint limits
                    r_min, r_max = self.jnt_range[jn]
                    if q_curr < (r_min - 0.01) or q_curr > (r_max + 0.01):
                        joint_limit_violations += 1

                mujoco.mj_step(self.model, self.data)

            # Record step metrics
            torques_pct_effort.append(step_max_pct)
            z_curr = float(self.data.qpos[2])
            roll_d, pitch_d, tilt_d = quat_to_roll_pitch(self.data.qpos[3:7])
            pelvis_heights.append(z_curr)
            pelvis_tilts.append(tilt_d)

            # Body local velocity
            R = self.data.xmat[self.pelvis_id].reshape(3, 3)
            v_body = R.T @ self.data.qvel[0:3]
            w_body = R.T @ self.data.qvel[3:6]

            # Tracking error (only evaluate after initial transient t >= 1.0s and before fall)
            if t_policy >= 1.0 and not fell:
                vel_errors_vx.append(abs(float(v_body[0]) - float(cmd[0])))
                vel_errors_vy.append(abs(float(v_body[1]) - float(cmd[1])))
                vel_errors_wz.append(abs(float(w_body[2]) - float(cmd[2])))

            # Foot contact sensors
            f_l_z = abs(self.data.sensordata[self.sensor_left_force[0] + 2]) if self.sensor_left_force[0] != -1 else 0.0
            f_r_z = abs(self.data.sensordata[self.sensor_right_force[0] + 2]) if self.sensor_right_force[0] != -1 else 0.0
            left_contact = f_l_z > 5.0
            right_contact = f_r_z > 5.0
            if left_contact:
                left_contact_steps += 1
            if right_contact:
                right_contact_steps += 1

            # Log row
            history_rows.append({
                "time": round(t_policy, 4),
                "cmd_vx": round(float(cmd[0]), 3),
                "cmd_vy": round(float(cmd[1]), 3),
                "cmd_wz": round(float(cmd[2]), 3),
                "pelvis_x": round(float(self.data.qpos[0]), 4),
                "pelvis_y": round(float(self.data.qpos[1]), 4),
                "pelvis_z": round(z_curr, 4),
                "pelvis_vx": round(float(v_body[0]), 4),
                "pelvis_vy": round(float(v_body[1]), 4),
                "pelvis_wz": round(float(w_body[2]), 4),
                "pelvis_roll_deg": round(roll_d, 2),
                "pelvis_pitch_deg": round(pitch_d, 2),
                "pelvis_tilt_deg": round(tilt_d, 2),
                "max_torque_pct": round(step_max_pct, 1),
                "left_contact": int(left_contact),
                "right_contact": int(right_contact),
                "fell": int(fell),
            })

            if fell:
                break

        # Save trajectory CSV
        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                if history_rows:
                    writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(history_rows)

        # Aggregate summary metrics
        total_eval_steps = len(history_rows) or 1
        mean_vx_err = float(np.mean(vel_errors_vx)) if vel_errors_vx else 0.0
        mean_vy_err = float(np.mean(vel_errors_vy)) if vel_errors_vy else 0.0
        mean_wz_err = float(np.mean(vel_errors_wz)) if vel_errors_wz else 0.0
        mean_height = float(np.mean(pelvis_heights)) if pelvis_heights else 0.0
        max_tilt = float(np.max(pelvis_tilts)) if pelvis_tilts else 0.0
        peak_tq_pct = float(np.max(torques_pct_effort)) if torques_pct_effort else 0.0
        left_duty = float(left_contact_steps / total_eval_steps)
        right_duty = float(right_contact_steps / total_eval_steps)
        mean_action_diff = float(np.mean(action_diffs)) if action_diffs else 0.0

        return {
            "fell": fell,
            "survival_time_s": survival_time,
            "duration_s": duration_s,
            "mean_vx_error_m_s": mean_vx_err,
            "mean_vy_error_m_s": mean_vy_err,
            "mean_wz_error_rad_s": mean_wz_err,
            "pelvis_height_mean_m": mean_height,
            "pelvis_tilt_max_deg": max_tilt,
            "peak_torque_pct_effort": peak_tq_pct,
            "joint_limit_violations": joint_limit_violations,
            "left_contact_duty": left_duty,
            "right_contact_duty": right_duty,
            "action_smoothness": mean_action_diff,
            "csv_path": str(output_csv) if output_csv else "",
        }


def run_experiment_suite(raw_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Execute complete E1 through E7 experiment matrix."""
    if raw_dir is None:
        raw_dir = _VASIMOV_DIR / "reports" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    runner = StandalonePolicyRunner()
    results: Dict[str, Dict[str, Any]] = {}

    print("=" * 110)
    print("RUNNING STANDALONE POLICY EXPERIMENT MATRIX (E1 - E7)")
    print("Model: asimov_1_vasimov.xml | Policy: Menlo/asimov1-locomotion-0818 | 50 Hz Policy Tick")
    print("=" * 110)
    print(f"{'Exp':<12} {'Verdict':<8} {'Surv (s)':<10} {'vx Err':<10} {'H (m)':<8} {'Max Tilt':<10} {'Peak Tau%':<10} {'Contact(L/R)':<14} {'One-Line Verdict'}")
    print("-" * 110)

    # ── E1: Zero Command 60s ─────────────────────────────────────────────────
    res_e1 = runner.run_experiment(
        duration_s=60.0,
        command_fn=lambda t: [0.0, 0.0, 0.0],
        output_csv=raw_dir / "policy_e1_zero_cmd_60s.csv",
    )
    v1 = "PASS" if (not res_e1["fell"] and res_e1["survival_time_s"] >= 60.0) else "FAIL"
    res_e1["verdict"] = v1
    res_e1["summary"] = f"Stable standing for 60.0s (mean height {res_e1['pelvis_height_mean_m']:.3f}m, tilt < {res_e1['pelvis_tilt_max_deg']:.1f}°)"
    results["E1"] = res_e1
    c1 = f"{res_e1['left_contact_duty']:.2f}/{res_e1['right_contact_duty']:.2f}"
    print(f"{'E1 (0 cmd)':<12} {v1:<8} {res_e1['survival_time_s']:<10.1f} {res_e1['mean_vx_error_m_s']:<10.3f} {res_e1['pelvis_height_mean_m']:<8.3f} {res_e1['pelvis_tilt_max_deg']:<10.1f} {res_e1['peak_torque_pct_effort']:<10.1f} {c1:<14} {res_e1['summary']}")

    # ── E2: Forward Velocity 0.2 / 0.4 / 0.6 m/s for 30s ─────────────────────
    for vx_target in (0.2, 0.4, 0.6):
        exp_key = f"E2_vx_{vx_target}"
        res_e2 = runner.run_experiment(
            duration_s=30.0,
            command_fn=lambda t, vx=vx_target: [vx, 0.0, 0.0],
            output_csv=raw_dir / f"policy_e2_vx_{vx_target}.csv",
        )
        # Pass bar: 30s without falling, mean vx error < 0.15 m/s
        v2 = "PASS" if (not res_e2["fell"] and res_e2["survival_time_s"] >= 30.0 and res_e2["mean_vx_error_m_s"] < 0.15) else "FAIL"
        res_e2["verdict"] = v2
        res_e2["summary"] = f"Forward walking vx={vx_target} m/s for 30s (mean vx error: {res_e2['mean_vx_error_m_s']:.3f} m/s)"
        results[exp_key] = res_e2
        c2 = f"{res_e2['left_contact_duty']:.2f}/{res_e2['right_contact_duty']:.2f}"
        print(f"{exp_key:<12} {v2:<8} {res_e2['survival_time_s']:<10.1f} {res_e2['mean_vx_error_m_s']:<10.3f} {res_e2['pelvis_height_mean_m']:<8.3f} {res_e2['pelvis_tilt_max_deg']:<10.1f} {res_e2['peak_torque_pct_effort']:<10.1f} {c2:<14} {res_e2['summary']}")

    # ── E3: Lateral Velocity +-0.2 m/s for 30s ───────────────────────────────
    for vy_target in (0.2, -0.2):
        exp_key = f"E3_vy_{'+' if vy_target > 0 else ''}{vy_target}"
        res_e3 = runner.run_experiment(
            duration_s=30.0,
            command_fn=lambda t, vy=vy_target: [0.0, vy, 0.0],
            output_csv=raw_dir / f"policy_e3_vy_{vy_target}.csv",
        )
        v3 = "PASS" if (not res_e3["fell"] and res_e3["survival_time_s"] >= 30.0) else "FAIL"
        res_e3["verdict"] = v3
        res_e3["summary"] = f"Lateral stepping vy={vy_target:+.1f} m/s for 30s (mean vy error: {res_e3['mean_vy_error_m_s']:.3f} m/s)"
        results[exp_key] = res_e3
        c3 = f"{res_e3['left_contact_duty']:.2f}/{res_e3['right_contact_duty']:.2f}"
        print(f"{exp_key:<12} {v3:<8} {res_e3['survival_time_s']:<10.1f} {res_e3['mean_vy_error_m_s']:<10.3f} {res_e3['pelvis_height_mean_m']:<8.3f} {res_e3['pelvis_tilt_max_deg']:<10.1f} {res_e3['peak_torque_pct_effort']:<10.1f} {c3:<14} {res_e3['summary']}")

    # ── E4: Yaw Turn +-0.5 rad/s for 30s ─────────────────────────────────────
    for wz_target in (0.5, -0.5):
        exp_key = f"E4_yaw_{'+' if wz_target > 0 else ''}{wz_target}"
        res_e4 = runner.run_experiment(
            duration_s=30.0,
            command_fn=lambda t, wz=wz_target: [0.0, 0.0, wz],
            output_csv=raw_dir / f"policy_e4_yaw_{wz_target}.csv",
        )
        v4 = "PASS" if (not res_e4["fell"] and res_e4["survival_time_s"] >= 30.0) else "FAIL"
        res_e4["verdict"] = v4
        res_e4["summary"] = f"In-place turning yaw={wz_target:+.1f} rad/s for 30s (mean wz error: {res_e4['mean_wz_error_rad_s']:.3f} rad/s)"
        results[exp_key] = res_e4
        c4 = f"{res_e4['left_contact_duty']:.2f}/{res_e4['right_contact_duty']:.2f}"
        print(f"{exp_key:<12} {v4:<8} {res_e4['survival_time_s']:<10.1f} {res_e4['mean_wz_error_rad_s']:<10.3f} {res_e4['pelvis_height_mean_m']:<8.3f} {res_e4['pelvis_tilt_max_deg']:<10.1f} {res_e4['peak_torque_pct_effort']:<10.1f} {c4:<14} {res_e4['summary']}")

    # ── E5: Combined Commands at Contract Limits ──────────────────────────────
    res_e5 = runner.run_experiment(
        duration_s=30.0,
        command_fn=lambda t: [0.5, 0.2, 0.4],
        output_csv=raw_dir / "policy_e5_combined_limits.csv",
    )
    v5 = "PASS" if (not res_e5["fell"] and res_e5["survival_time_s"] >= 30.0) else "FAIL"
    res_e5["verdict"] = v5
    res_e5["summary"] = f"Combined curve locomotion vx=0.5, vy=0.2, wz=0.4 for 30s (survived={res_e5['survival_time_s']:.1f}s)"
    results["E5"] = res_e5
    c5 = f"{res_e5['left_contact_duty']:.2f}/{res_e5['right_contact_duty']:.2f}"
    print(f"{'E5 (comb)':<12} {v5:<8} {res_e5['survival_time_s']:<10.1f} {res_e5['mean_vx_error_m_s']:<10.3f} {res_e5['pelvis_height_mean_m']:<8.3f} {res_e5['pelvis_tilt_max_deg']:<10.1f} {res_e5['peak_torque_pct_effort']:<10.1f} {c5:<14} {res_e5['summary']}")

    # ── E6: Push Disturbances 40/80/120 N while Standing & Walking ───────────
    for mode_name, vx_cmd in (("stand", 0.0), ("walk", 0.2)):
        for p_force in (40.0, 80.0, 120.0):
            exp_key = f"E6_{mode_name}_{int(p_force)}N"
            res_e6 = runner.run_experiment(
                duration_s=15.0,
                command_fn=lambda t, vx=vx_cmd: [vx, 0.0, 0.0],
                push_schedule=[{"time": 4.0, "duration": 0.1, "force": p_force, "dir": "x"}],
                output_csv=raw_dir / f"policy_e6_{mode_name}_{int(p_force)}N.csv",
            )
            v6 = "PASS" if not res_e6["fell"] else "FAIL"
            res_e6["verdict"] = v6
            res_e6["summary"] = f"Push disturbance {p_force:.0f} N while {mode_name}ing (status={'SURVIVED' if not res_e6['fell'] else 'FELL'})"
            results[exp_key] = res_e6
            c6 = f"{res_e6['left_contact_duty']:.2f}/{res_e6['right_contact_duty']:.2f}"
            print(f"{exp_key:<12} {v6:<8} {res_e6['survival_time_s']:<10.1f} {res_e6['mean_vx_error_m_s']:<10.3f} {res_e6['pelvis_height_mean_m']:<8.3f} {res_e6['pelvis_tilt_max_deg']:<10.1f} {res_e6['peak_torque_pct_effort']:<10.1f} {c6:<14} {res_e6['summary']}")

    # ── E7: Stand-to-Policy Handoff ───────────────────────────────────────────
    res_e7 = runner.run_experiment(
        duration_s=20.0,
        command_fn=lambda t: [0.3, 0.0, 0.0] if t >= 3.0 else [0.0, 0.0, 0.0],
        handoff_s=3.0,  # 3.0s in standing posture PD before switching to policy
        output_csv=raw_dir / "policy_e7_stand_handoff.csv",
    )
    v7 = "PASS" if not res_e7["fell"] else "FAIL"
    res_e7["verdict"] = v7
    res_e7["summary"] = f"Smooth handoff from STAND PD to Policy at t=3.0s (survived 20.0s, tilt < {res_e7['pelvis_tilt_max_deg']:.1f}°)"
    results["E7"] = res_e7
    c7 = f"{res_e7['left_contact_duty']:.2f}/{res_e7['right_contact_duty']:.2f}"
    print(f"{'E7 (handoff)':<12} {v7:<8} {res_e7['survival_time_s']:<10.1f} {res_e7['mean_vx_error_m_s']:<10.3f} {res_e7['pelvis_height_mean_m']:<8.3f} {res_e7['pelvis_tilt_max_deg']:<10.1f} {res_e7['peak_torque_pct_effort']:<10.1f} {c7:<14} {res_e7['summary']}")

    print("=" * 110)
    print("ALL STANDALONE POLICY EXPERIMENTS COMPLETE.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Virtual Asimov 1 Standalone Locomotion Policy Harness")
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite (E1 to E7)")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (s)")
    parser.add_argument("--csv", type=str, default="", help="Optional output trajectory CSV")
    args = parser.parse_args()

    if args.suite:
        run_experiment_suite()
    else:
        runner = StandalonePolicyRunner()
        csv_p = Path(args.csv) if args.csv else None
        res = runner.run_experiment(
            duration_s=args.duration,
            command_fn=lambda t: [args.vx, args.vy, args.wz],
            output_csv=csv_p,
        )
        print("\nEXPERIMENT RESULT:")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
