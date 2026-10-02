# Virtual Asimov 1 — Control & Data Pipeline Architecture

This document specifies the exact end-to-end control and data pipeline for the Virtual Asimov 1 prototype in MuJoCo, detailing file paths, data transformations, coordinate systems, and frequency domains.

---

## 1. End-to-End Pipeline Diagram

```text
               ┌───────────────────────────────┐
               │    USER INPUT / TERMINAL      │
               │   CLI: walk 0.4 / W/S/A/D     │
               └───────────────┬───────────────┘
                               │
                               ▼ [tools/sim_console.py: execute_command()]
               ┌───────────────────────────────┐
               │       COMMAND VALIDATION      │
               │  range clipping: vx=[-0.6,0.8]│
               │  vy=[-0.5,0.5], wz=[-0.8,0.8] │
               └───────────────┬───────────────┘
                               │
                               ▼ [edge/core.py: command_velocity()]
               ┌───────────────────────────────┐
               │      VIRTUAL EDGE CORE        │
               │ state machine: MOVE/POLICY    │
               │ watchdog, arbitration, ramp   │
               └───────────────┬───────────────┘
                               │
                               ▼ [edge/sim.py: step() @ 200 Hz]
    ┌──────────────────────────┴──────────────────────────┐
    │ 50 Hz Decimation (Every 4 physics steps = 0.020s)   │
    │                                                     │
    │  [edge/policy_builder.py: build_policy_observation]│
    │  Construct 78-D Observation Vector:                 │
    │   - Base ang vel (body frame) * 0.25                │
    │   - Projected gravity vector (body frame)           │
    │   - Commanded velocity [vx, vy, vyaw]               │
    │   - 23 Joint positions (q - q_default)              │
    │   - 23 Joint velocities * 0.10                      │
    │   - 23 Previous action history                      │
    │                          │                          │
    │                          ▼ [edge/policy.py]         │
    │  ONNX INFERENCE (asimov1-locomotion-0818 CPU)       │
    │   Input : [1, 78] float32                           │
    │   Output: [1, 23] float32 (raw actions in [-1, 1])  │
    │                          │                          │
    │                          ▼ [edge/policy_builder.py] │
    │  TARGET SYNTHESIS                                   │
    │   q_des = q_default + 0.25 * raw_action             │
    │   (for all 23 policy joints)                        │
    └──────────────────────────┬──────────────────────────┘
                               │
                               ▼ [edge/sim.py: manual/hybrid overrides]
               ┌───────────────────────────────┐
               │     CONTROL ARBITRATION       │
               │ Policy / Manual / Hybrid      │
               └───────────────┬───────────────┘
                               │
                               ▼ [motor_model.py: step() @ 200 Hz]
               ┌───────────────────────────────┐
               │      MOTOR DYNAMICS (L0-L4)   │
               │ PD: tau = Kp*(q_des - q) - Kd*w│
               │ tau_max(w) derating envelope  │
               │ effort clipping: [-tau_max, +]│
               └───────────────┬───────────────┘
                               │
                               ▼ [mujoco.mj_step(model, data)]
               ┌───────────────────────────────┐
               │        MUJOCO PHYSICS         │
               │ 200 Hz integration (dt=0.005s)│
               │ multibody dynamics + contacts │
               └───────────────┬───────────────┘
                               │
                               ▼ [edge/sim.py: get_state()]
               ┌───────────────────────────────┐
               │     TELEMETRY EXTRACTION      │
               │ Pelvis pose, IMU, 25 joints,  │
               │ Foot touch, vertical Fz       │
               └───────────────┬───────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
   [tools/sim_console.py] [reports/raw/*.csv] [reports/raw/*.json]
    Terminal Dashboard      CSV Flight Data    JSON Flight Replay
```

---

## 2. File Index & Responsibilities

| File Path | Role | Key Functions / Classes |
| :--- | :--- | :--- |
| [`tools/sim_console.py`](file:///home/shubhr/Shubhr/Projects/vasimov/tools/sim_console.py) | Interactive CLI REPL & Simulation Controller | `SimConsole`, `_physics_loop`, `execute_command`, `log_command_pipeline` |
| [`run_sim.sh`](file:///home/shubhr/Shubhr/Projects/vasimov/run_sim.sh) | Portable Project Launcher | Bash wrapper executing `python -m tools.sim_console` |
| [`edge/core.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/core.py) | Edge State Machine & Safety Arbiter | `EdgeCore`, `command_stand`, `command_velocity`, `step_state` |
| [`edge/sim.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/sim.py) | MuJoCo Physics Backend & State Manager | `SimBackend`, `step`, `reset`, `get_state`, `load_environment` |
| [`edge/environment.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/environment.py) | Environment Presets & Procedural Obstacles | `EnvironmentManager`, `PRESETS`, `create_model_for_preset` |
| [`edge/policy.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/policy.py) | ONNX Runtime Policy Runner | `PolicyController`, `step`, `clip_velocity_command` |
| [`edge/policy_builder.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/policy_builder.py) | Unified 78-D Obs & 23-D Action Builder | `build_policy_observation`, `compute_desired_joint_targets` |
| [`edge/telemetry.py`](file:///home/shubhr/Shubhr/Projects/vasimov/edge/telemetry.py) | Live Telemetry & Recording Engine | `TelemetryEngine`, `format_status_block`, `format_joint_table`, `record_frame` |
| [`motor_model.py`](file:///home/shubhr/Shubhr/Projects/vasimov/motor_model.py) | Actuator Dynamics & Speed-Torque Envelope | `MotorModel`, `step` (L0 PD, L1 envelope) |
| [`ankle_map.py`](file:///home/shubhr/Shubhr/Projects/vasimov/ankle_map.py) | Differential Pushrod Ankle Mapping | `DifferentialAnkleMap`, `pr_to_ab`, `torque_pr_to_ab` |
| [`adapter.py`](file:///home/shubhr/Shubhr/Projects/vasimov/adapter.py) | Joint Order & Coordinate Conversion | `JointAdapter`, `SIM_JOINTS`, `FIRMWARE_JOINTS` |

---

## 3. Data Flow Contracts

### 3.1 Velocity Command Bounds
Extracted from `assets/policy/env.yaml` (`commands.twist.ranges`):
- $v_x \in [-0.6, +0.8]\,\text{m/s}$
- $v_y \in [-0.5, +0.5]\,\text{m/s}$
- $\omega_z \in [-0.8, +0.8]\,\text{rad/s}$

### 3.2 78-D Policy Observation Vector
```text
Index Range  | Description                           | Scale Factor
[00 : 03]    | Base Angular Velocity (Pelvis Body)   | * 0.25
[03 : 06]    | Projected Gravity Vector (Pelvis Body)| * 1.0 (R^T * [0, 0, -1])
[06 : 09]    | Velocity Command [vx, vy, vyaw]       | * 1.0
[09 : 18]    | Joint Position Slot 01 (9 joints)     | (q - q_default) * 1.0
[18 : 26]    | Joint Position Slot 23 (8 joints)     | (q - q_default) * 1.0
[26 : 32]    | Joint Position Slot 45 (6 joints)     | (q - q_default) * 1.0
[32 : 41]    | Joint Velocity Slot 01 (9 joints)     | qdot * 0.10
[41 : 49]    | Joint Velocity Slot 23 (8 joints)     | qdot * 0.10
[49 : 55]    | Joint Velocity Slot 45 (6 joints)     | qdot * 0.10
[55 : 78]    | Previous Policy Action Vector (23)    | * 1.0
Total: 78 floats (float32)
```

### 3.3 23-D Policy Action Synthesis
```text
q_des[j] = q_default[j] + 0.25 * action_raw[j]
```
Evaluated for the 23 policy joints in verified Hugging Face joint order.
Actuator torques are computed by `motor_model.py` using per-joint stiffness ($K_p$) and damping ($K_d$) from `config/gains.yaml`.
