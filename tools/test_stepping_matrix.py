"""
tools/test_stepping_matrix.py
Task 3 Investigation: Why doesn't vx=0.2 start walking after balance?

Evaluates matrix:
- command vx in {0.1, 0.2, 0.3, 0.4}
- pre-balance in {0, 2, 5, 20} seconds
- observation noise in {off, seed_1, seed_2, seed_3} using exact env.yaml parameters
- impulse in {none, 5N_0.05s}
Total 128 runs.

Criteria for 'stepping started':
- >= 2 foot-contact alternations OR > 0.3 m displacement within 5.0 s of command start.

Also tests:
- Fine-grained minimal command threshold to break stationary standing attractor.
- Command ramp efficacy (e.g. ramp over 0.5s / 1.0s).
"""

from __future__ import annotations
import csv
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import mujoco

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from edge.policy_builder import (
    build_policy_observation,
    compute_desired_joint_targets,
    OBS_DIM,
    ACTION_DIM,
    ACTION_SCALE,
)
from tools.policy_standalone import StandalonePolicyRunner, POLICY_RATE_HZ, PHYSICS_DT, DECIMATION

# Exact env.yaml noise parameters
NOISE_CFG = {
    "ang_vel": 0.01,         # uniform [-0.01, 0.01] rad/s
    "gravity": 0.02,         # uniform [-0.02, 0.02]
    "joint_pos": 0.01,       # uniform [-0.01, 0.01] rad
    "joint_vel": 0.50,       # uniform [-0.50, 0.50] rad/s
}


def apply_observation_noise(
    obs: np.ndarray,
    rng: Optional[np.random.Generator],
    slot01_names: List[str],
    slot23_names: List[str],
    slot45_names: List[str],
) -> np.ndarray:
    """Apply uniform noise matching env.yaml parameters to raw 78-D observation vector."""
    if rng is None:
        return obs

    noisy = np.copy(obs)
    # 1. base_ang_vel: obs[0:3] = 0.25 * (omega + uniform(-0.01, 0.01))
    noisy[0:3] += 0.25 * rng.uniform(-NOISE_CFG["ang_vel"], NOISE_CFG["ang_vel"], size=3)

    # 2. projected_gravity: obs[3:6] = grav + uniform(-0.02, 0.02)
    noisy[3:6] += rng.uniform(-NOISE_CFG["gravity"], NOISE_CFG["gravity"], size=3)

    # 3. command: obs[6:9] -> NO NOISE (noise: null in env.yaml)

    # 4-6. joint_pos: obs[9:32] (23 joints) -> uniform(-0.01, 0.01)
    n_pos = 9 + 8 + 6
    noisy[9:9+n_pos] += rng.uniform(-NOISE_CFG["joint_pos"], NOISE_CFG["joint_pos"], size=n_pos)

    # 7-9. joint_vel: obs[32:55] (23 joints) -> 0.1 * uniform(-0.5, 0.5)
    noisy[32:32+n_pos] += 0.10 * rng.uniform(-NOISE_CFG["joint_vel"], NOISE_CFG["joint_vel"], size=n_pos)

    # 10. actions: obs[55:78] -> NO NOISE
    return noisy.astype(np.float32)


def run_single_matrix_trial(
    runner: StandalonePolicyRunner,
    cmd_vx: float,
    pre_balance_s: float,
    noise_mode: str,  # 'off', 'seed_1', 'seed_2', 'seed_3'
    impulse_mode: str, # 'none', '5N_0.05s'
    eval_duration_s: float = 5.0,
) -> Dict[str, Any]:
    """Execute one trial of the matrix and evaluate if stepping started."""
    runner.reset()
    m = runner.model
    d = runner.data
    mujoco.mj_forward(m, d)

    rng = None
    if noise_mode != "off":
        seed = int(noise_mode.split("_")[1])
        rng = np.random.default_rng(seed)

    prev_act = np.zeros(ACTION_DIM, dtype=np.float32)

    # 1. Pre-balance phase (cmd = 0)
    pre_steps = int(round(pre_balance_s * POLICY_RATE_HZ))
    for _ in range(pre_steps):
        obs = runner.build_observation(command=np.array([0.0, 0.0, 0.0], dtype=np.float32), prev_action=prev_act)
        obs = apply_observation_noise(obs, rng, runner.slot01_names, runner.slot23_names, runner.slot45_names)
        raw_act = runner.session.run(["actions"], {"obs": obs.reshape(1, OBS_DIM)})[0][0]
        prev_act = np.copy(raw_act)
        q_des = compute_desired_joint_targets(raw_act, runner.policy_joints, runner.default_pos)

        for _ in range(DECIMATION):
            for a_idx, jn in enumerate(runner.act_jnt_names):
                target = q_des.get(jn, 0.0)
                kp, kd, eff = runner.joint_gains.get(jn, (40.0, 2.0, 12.0))
                q = d.qpos[runner.jnt_qposadr[jn]]
                qdot = d.qvel[runner.jnt_dofadr[jn]]
                d.ctrl[a_idx] = np.clip(kp * (target - q) - kd * qdot, -eff, eff)
            mujoco.mj_step(m, d)

    # Record baseline state before walk command
    x_start = float(d.qpos[0])
    z_start = float(d.qpos[2])

    # 2. Walk command evaluation phase (eval_duration_s = 5.0 s)
    eval_steps = int(round(eval_duration_s * POLICY_RATE_HZ))
    impulse_steps_remaining = int(round(0.05 / PHYSICS_DT)) if impulse_mode == "5N_0.05s" else 0

    left_contacts: List[bool] = []
    right_contacts: List[bool] = []
    traj_x: List[float] = []

    last_stance: Optional[str] = None # 'L', 'R', 'BOTH', 'NONE'
    alternations = 0

    for step in range(eval_steps):
        cmd = np.array([cmd_vx, 0.0, 0.0], dtype=np.float32)
        obs = runner.build_observation(command=cmd, prev_action=prev_act)
        obs = apply_observation_noise(obs, rng, runner.slot01_names, runner.slot23_names, runner.slot45_names)
        raw_act = runner.session.run(["actions"], {"obs": obs.reshape(1, OBS_DIM)})[0][0]
        prev_act = np.copy(raw_act)
        q_des = compute_desired_joint_targets(raw_act, runner.policy_joints, runner.default_pos)

        for _ in range(DECIMATION):
            # Apply impulse if scheduled
            if impulse_steps_remaining > 0:
                d.xfrc_applied[runner.pelvis_id, 0] = 5.0 # 5 N forward
                impulse_steps_remaining -= 1
            else:
                d.xfrc_applied[runner.pelvis_id, :] = 0.0

            for a_idx, jn in enumerate(runner.act_jnt_names):
                target = q_des.get(jn, 0.0)
                kp, kd, eff = runner.joint_gains.get(jn, (40.0, 2.0, 12.0))
                q = d.qpos[runner.jnt_qposadr[jn]]
                qdot = d.qvel[runner.jnt_dofadr[jn]]
                d.ctrl[a_idx] = np.clip(kp * (target - q) - kd * qdot, -eff, eff)
            mujoco.mj_step(m, d)

        # Foot force sensing
        f_l_z = abs(d.sensordata[runner.sensor_left_force[0] + 2]) if runner.sensor_left_force[0] != -1 else 0.0
        f_r_z = abs(d.sensordata[runner.sensor_right_force[0] + 2]) if runner.sensor_right_force[0] != -1 else 0.0
        c_l = f_l_z > 5.0
        c_r = f_r_z > 5.0
        left_contacts.append(c_l)
        right_contacts.append(c_r)
        traj_x.append(float(d.qpos[0]))

        # Stance alternation logic
        if c_l and not c_r:
            curr_stance = "L"
        elif c_r and not c_l:
            curr_stance = "R"
        elif c_l and c_r:
            curr_stance = "BOTH"
        else:
            curr_stance = "FLIGHT"

        if curr_stance in ("L", "R"):
            if last_stance is not None and curr_stance != last_stance:
                alternations += 1
            last_stance = curr_stance

    displacement_x = float(d.qpos[0] - x_start)
    stepping_started = (alternations >= 2) or (displacement_x > 0.30)

    return {
        "cmd_vx": cmd_vx,
        "pre_balance_s": pre_balance_s,
        "noise_mode": noise_mode,
        "impulse_mode": impulse_mode,
        "displacement_x": round(displacement_x, 4),
        "alternations": alternations,
        "stepping_started": stepping_started,
        "final_z": round(float(d.qpos[2]), 4),
    }


def main():
    print("=" * 70)
    print("RUNNING TASK 3: STEPPING INITIATION MATRIX (128 TRIALS)")
    print("=" * 70)

    runner = StandalonePolicyRunner()

    commands = [0.1, 0.2, 0.3, 0.4]
    pre_balances = [0.0, 2.0, 5.0, 20.0]
    noise_modes = ["off", "seed_1", "seed_2", "seed_3"]
    impulse_modes = ["none", "5N_0.05s"]

    out_csv = _VASIMOV_DIR / "reports" / "raw" / "task3_stepping_matrix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total = len(commands) * len(pre_balances) * len(noise_modes) * len(impulse_modes)
    idx = 0

    for cmd in commands:
        for pre in pre_balances:
            for n_mode in noise_modes:
                for imp in impulse_modes:
                    idx += 1
                    res = run_single_matrix_trial(
                        runner=runner,
                        cmd_vx=cmd,
                        pre_balance_s=pre,
                        noise_mode=n_mode,
                        impulse_mode=imp,
                    )
                    results.append(res)
                    status = "STEPPED" if res["stepping_started"] else "STILL"
                    if idx % 16 == 0 or res["stepping_started"]:
                        print(f"[{idx:3d}/{total}] vx={cmd:.1f} | pre={pre:4.1f}s | noise={n_mode:<6} | imp={imp:<9} -> dx={res['displacement_x']:+.3f}m | alts={res['alternations']:2d} | {status}")

    # Write raw matrix CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved raw matrix results to {out_csv}")

    # ------------------------------------------------------------------------
    # FINE-GRAINED MINIMAL COMMAND THRESHOLD TEST
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TESTING MINIMAL COMMAND THRESHOLD FROM STATIONARY BALANCE (pre_balance=5.0s)")
    print("=" * 70)
    fine_csv = _VASIMOV_DIR / "reports" / "raw" / "task3_minimal_command_threshold.csv"
    fine_results = []
    for test_vx in np.linspace(0.15, 0.40, 26): # 0.15, 0.16, ... 0.40
        test_vx = round(float(test_vx), 3)
        res = run_single_matrix_trial(
            runner=runner,
            cmd_vx=test_vx,
            pre_balance_s=5.0,
            noise_mode="off",
            impulse_mode="none",
        )
        fine_results.append(res)
        status = "STEPPED" if res["stepping_started"] else "STILL"
        print(f"vx={test_vx:.3f} m/s -> dx={res['displacement_x']:+.3f}m | alts={res['alternations']:2d} | {status}")

    with open(fine_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fine_results[0].keys()))
        writer.writeheader()
        writer.writerows(fine_results)
    print(f"Saved threshold sweep to {fine_csv}")


if __name__ == "__main__":
    main()
