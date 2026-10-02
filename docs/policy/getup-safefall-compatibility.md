# Asimov 1 Policy Compatibility & Feasibility Report: Official Locomotion vs In-House Get-Up / Safe-Fall

## Executive Summary

This report documents the architectural, tensor, timing, and sensory contracts of the official locomotion policy (`Menlo/asimov1-locomotion-0818`) and the in-house recovery policies (`asimov1-getup-safefall`: `getup.onnx` and `safefall.onnx`).

All properties documented below have been verified against active model artifacts (`assets/policy/policy.onnx`, `asimov1-getup-safefall/policies/getup.onnx`, `safefall.onnx`) and specification schemas (`env.yaml`, `getup_policy_spec.json`, `safefall_policy_spec.json`, `run_policy_mujoco.py`).

---

## Direct Specification Comparison

| Property | Official Locomotion (`official_locomotion`) | In-House Safe-Fall & Get-Up (`getup_safefall`) |
| :--- | :--- | :--- |
| **Model Artifact** | `policy.onnx` (831.8 KB) | `getup.onnx` (1.44 MB), `safefall.onnx` (1.44 MB) |
| **Runtime Engine** | ONNX Runtime (`CPUExecutionProvider`) | ONNX Runtime (`CPUExecutionProvider`) |
| **Input Tensor Name** | `obs` | `obs` |
| **Input Shape** | `[1, 78]` (`float32`) | `[1, 375]` (`float32`) |
| **Output Tensor Name**| `actions` | `actions` |
| **Output Shape** | `[1, 23]` (`float32`) | `[1, 23]` (`float32`) |
| **Control Frequency** | 50.0 Hz (period $\Delta t = 0.020\,\text{s}$) | 50.0 Hz (period $\Delta t = 0.020\,\text{s}$) |
| **Physics Decimation**| 4 steps @ 200 Hz MuJoCo timestep ($0.005\,\text{s}$) | 4 steps @ 200 Hz MuJoCo timestep ($0.005\,\text{s}$) |
| **Statefulness** | Stateless graph with 1-step previous action buffer (23-D) | Stateful 5-step observation history buffer ($5 \times 75 = 375\text{-D}$) |
| **History Layout** | N/A (single frame with explicit 23-D prev_action slot) | Term-major: each term stacked oldest $\rightarrow$ newest flattened |
| **Observation Terms** | 1. `base_ang_vel` (3-D, $\times 0.25$)<br>2. `projected_gravity` (3-D, local)<br>3. `velocity_commands` (3-D: $v_x, v_y, \omega_z$)<br>4. `joint_pos_rel` (23-D in 3 slots: 9+8+6)<br>5. `joint_vel` (23-D, $\times 0.10$ in 3 slots)<br>6. `prev_action` (23-D) | 1. `base_ang_vel` (3-D, $\times 0.25$)<br>2. `projected_gravity` (3-D, unit vector)<br>3. `joint_pos` (23-D, $q - q_{\text{default}}$ in `obs_joint_order`)<br>4. `joint_vel` (23-D, $\times 0.10$ in `obs_joint_order`)<br>5. `actions` (23-D, previous policy action) |
| **Joint Coverage** | 23 actuated joints (excluding neck pitch/yaw) | 23 actuated joints (identical 23 joints) |
| **Joint Order (Obs)** | `slot01` (9) + `slot23` (8) + `slot45` (6) | `obs_joint_order` (23-element permutation) |
| **Joint Order (Act)** | `policy_joints` (23 joints, alphabetical by group) | `action["joint_order"]` (23 joints, identical ordering) |
| **Action Semantics** | Position delta: $q_{\text{des}} = q_{\text{default}} + 0.25 \times a$ | Position delta: $q_{\text{des}} = q_{\text{default}} + 0.25 \times a$ |
| **Default Joint Pos** | Extracted from `env.yaml` `robot.init_state` | Exact matching default pose vector in `spec["default_joint_pos"]` |
| **PD Gains** | Grouped by joint type ($K_p \in [40, 150]$, $K_d \in [2, 5]$) | Matching PD gains and rated effort limits |
| **Normalization** | None in graph; raw scaled values fed directly | Baked into ONNX graph; raw scaled values fed directly |
| **Required Contacts** | None (pure proprioceptive) | None (pure proprioceptive) |
| **Required IMU** | Pelvis gyro (`imu_ang_vel`) & projected gravity | Pelvis gyro (`imu_ang_vel`) & projected gravity |
| **Required Cameras** | None required for policy execution | None required for policy execution |
| **Trigger / FSM** | Continuous locomotion tracking commanded velocity | Safe-fall triggers on tilt $> 30^\circ$ or $\|\omega\| > 2.0\,\text{rad/s}$; get-up triggers once settled |

---

## Key Compatibility Findings

### 1. Unified 23-Joint Action Compatibility
Both policies control the exact same 23 humanoid degrees of freedom with the exact same action scaling ($0.25$) and default pose offsets. The neck pitch and yaw joints remain unactuated by both policies.

### 2. Observation Dimensions: 78 vs 375
- Official Locomotion accepts **78 float32 inputs** representing a single time frame with an explicit velocity command $(v_x, v_y, \omega_z)$ and the previous step's action vector.
- In-House Recovery accepts **375 float32 inputs** representing 5 consecutive timesteps ($5 \times 75 = 375$). The 75-D per-step observation consists of:
  $$\text{base\_ang\_vel}\,(3) + \text{projected\_gravity}\,(3) + \text{joint\_pos}\,(23) + \text{joint\_vel}\,(23) + \text{prev\_actions}\,(23) = 75$$
  Neither recovery policy takes velocity commands.

### 3. Term-Major History Stacking
The 375-D input is ordered **term-major**:
- 15 floats: 5 timesteps of `base_ang_vel` (oldest $\rightarrow$ newest)
- 15 floats: 5 timesteps of `projected_gravity`
- 115 floats: 5 timesteps of `joint_pos`
- 115 floats: 5 timesteps of `joint_vel`
- 115 floats: 5 timesteps of `actions`

Upon policy reset or handover, all 5 history slots are initialized with the initial observation to avoid cold-start discontinuity.

### 4. Pure Proprioception (Zero Contact/Vision Dependencies)
Neither policy requires force plate contacts, tactile sensor arrays, or cameras. Both operate strictly on standard simulated IMU (gyro + accelerometer / projected gravity) and joint encoders (positions + velocities).

---

## Conclusion & Architecture Recommendations

The in-house get-up and safe-fall policies are **100% functionally compatible** with the VASIMOV simulation platform. 

The modular architecture must provide:
1. A **Generic Policy Adapter** that can construct both 78-D (single frame + commands) and 375-D (5-step history buffer) observation tensors from a shared `RobotState`.
2. A **Manifest-driven Registry** that defines observation dimensions, action mappings, and PD gains declaratively.
3. A **Policy Manager / State Machine** capable of running either policy independently (`--policy official_locomotion` or `--policy getup_safefall`), or arbitrating between them cleanly without simultaneous actuator writes.
