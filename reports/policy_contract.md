# Locomotion Policy Contract & Primary Source Verification

**Virtual Asimov 1 in MuJoCo — Step 5 Policy Contract**  
**Date:** 2026-10-02  
**Status:** FULLY VERIFIED against primary sources downloaded from Hugging Face  

---

## 1. Primary Source Artifacts & Checkpoint Hashes

The official pretrained locomotion policy checkpoint was downloaded directly from Hugging Face repository [`Menlo/asimov1-locomotion-0818`](https://huggingface.co/Menlo/asimov1-locomotion-0818). The repository commit is pinned and verified:

* **Hugging Face Model ID:** `Menlo/asimov1-locomotion-0818`
* **Pinned Revision Hash (Git Commit SHA):** `18114a730668a983423e6361f0522eb62c76cb74`
* **License:** BSD-3-Clause

### File Integrity & Checksums

| File Path in Repo | File Size (bytes) | SHA-256 Checksum |
|---|:---:|---|
| [`assets/policy/policy.onnx`](file:///home/shubhr/Shubhr/Projects/vasimov/assets/policy/policy.onnx) | 831,809 | `47f70691ecd5afda2933b2e6430d2375004c943552284172a2d7b28b84188a0c` |
| [`assets/policy/env.yaml`](file:///home/shubhr/Shubhr/Projects/vasimov/assets/policy/env.yaml) | 61,971 | `bfccfef91349e19c3a422b4aa5ca8ed23f5fe593bd79086cd239f7e4bcfd6bf8` |
| [`assets/policy/agent.yaml`](file:///home/shubhr/Shubhr/Projects/vasimov/assets/policy/agent.yaml) | 2,199 | `1fd3435ad0c8420923cb1ca79c8e8a0a916e160709f89096c58610cdd46f545a` |
| [`assets/policy/README.md`](file:///home/shubhr/Shubhr/Projects/vasimov/assets/policy/README.md) | 1,852 | `0f809015c10f79043a3155a6f0dcabaa3e8b460ce101653fb3b6f86259c7358c` |

---

## 2. ONNX Model Inspection & Graph Architecture

Inspection performed via `onnx` (v1.23.1) and `onnxruntime` (v1.30.0):

* **IR Version:** 6
* **Producer:** `pytorch 2.7.0`
* **Input Tensor:**
  * **Name:** `obs`
  * **Shape:** `[1, 78]`
  * **Data Type:** `float32` (elem_type: 1)
* **Output Tensor:**
  * **Name:** `actions`
  * **Shape:** `[1, 23]`
  * **Data Type:** `float32` (elem_type: 1)
* **Graph Structure (7 Nodes Total):**
  1. `Gemm`: in=`['obs', 'actor.0.weight', 'actor.0.bias']` -> shape `[1, 512]`
  2. `Elu`: activation
  3. `Gemm`: in=`[512, 'actor.2.weight', 'actor.2.bias']` -> shape `[1, 256]`
  4. `Elu`: activation
  5. `Gemm`: in=`[256, 'actor.4.weight', 'actor.4.bias']` -> shape `[1, 128]`
  6. `Elu`: activation
  7. `Gemm`: in=`[128, 'actor.6.weight', 'actor.6.bias']` -> shape `[1, 23]` (`actions`)
* **In-Graph Normalization:** **NONE**. Raw observation floats are fed directly into the first Gemm layer. All observation scalings ($0.25$ for gyro, $0.10$ for joint velocities, etc.) must be applied prior to graph execution exactly per `env.yaml`.
* **Metadata Properties:** None stored in ONNX header.

---

## 3. Resolution of the Isaac Asimov Audit Conflict

* **The Conflict:** An earlier project audit note hypothesized a 12-DoF leg-only action space operating at 25 Hz (common in minimal bipedal locomotion baselines).
* **The Resolution (VERIFIED):** Inspection of `policy.onnx`, `env.yaml`, and `agent.yaml` proves conclusively:
  1. **Action Dimension is 23-D** (12 leg joints + 1 waist joint + 10 arm joints). Only the 2 neck joints (`neck_yaw_joint`, `neck_pitch_joint`) are unactuated by the policy.
  2. **Simulation Timestep:** $\Delta t_{\text{sim}} = 0.005\text{ s}$ (**200 Hz** physics).
  3. **Decimation:** $4$ physics steps per policy step $\implies$ **50 Hz policy control rate** ($\Delta t_{\text{ctrl}} = 0.020\text{ s}$).
  4. **AMP (Adversarial Motion Prior) Status:** Verified in `agent.yaml` lines 46–94 that AMP discriminator components (`amp_data`, `amp_reward_coef`, `amp_discr_hidden_dims: [256, 256]`, `motion_files`) are **strictly training-only** objective terms used during PPO policy optimization and do not run during inference.

---

## 4. Policy Observation Specification (78-D)

Observations are constructed at 50 Hz and concatenated in the exact sequential order defined in `env.yaml` under `observations.policy`:

| Term Name | Dim | Slice | Scale | Clip | Noise (Train Only) | Description & Joint Ordering |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `base_ang_vel` | 3 | `[0:3]` | `0.25` | None | Uniform $\pm 0.01$ | Pelvis angular velocity $[\omega_x, \omega_y, \omega_z]$ in body frame (IMU gyro) |
| `projected_gravity` | 3 | `[3:6]` | `1.00` | None | Uniform $\pm 0.02$ | Gravity vector $[0, 0, -1]$ projected into pelvis frame |
| `command` | 3 | `[6:9]` | `1.00` | None | None | Commanded velocity $[v_x, v_y, \omega_z]$ |
| `joint_pos_slot01` | 9 | `[9:18]` | `1.00` | None | Uniform $\pm 0.01$ | $q - q_{\text{default}}$ for: `left_hip_pitch_joint`, `left_hip_roll_joint`, `right_hip_pitch_joint`, `right_hip_roll_joint`, `waist_yaw_joint`, `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `left_shoulder_pitch_joint`, `left_shoulder_roll_joint` |
| `joint_pos_slot23` | 8 | `[18:26]` | `1.00` | None | Uniform $\pm 0.01$ | $q - q_{\text{default}}$ for: `left_hip_yaw_joint`, `left_knee_joint`, `right_hip_yaw_joint`, `right_knee_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint` |
| `joint_pos_slot45` | 6 | `[26:32]` | `1.00` | None | Uniform $\pm 0.01$ | $q - q_{\text{default}}$ for: `left_ankle_pitch_joint`, `left_ankle_roll_joint`, `right_ankle_pitch_joint`, `right_ankle_roll_joint`, `right_wrist_yaw_joint`, `left_wrist_yaw_joint` |
| `joint_vel_slot01` | 9 | `[32:41]` | `0.10` | None | Uniform $\pm 0.50$ | Joint angular velocities $\dot{q}$ for Slot 0 & 1 joints (same order as above) |
| `joint_vel_slot23` | 8 | `[41:49]` | `0.10` | None | Uniform $\pm 0.50$ | Joint angular velocities $\dot{q}$ for Slot 2 & 3 joints (same order as above) |
| `joint_vel_slot45` | 6 | `[49:55]` | `0.10` | None | Uniform $\pm 0.50$ | Joint angular velocities $\dot{q}$ for Slot 4 & 5 joints (same order as above) |
| `actions` | 23 | `[55:78]` | `1.00` | None | None | Previous 23-D raw action output from previous control step (init zeros) |
| **Total** | **78** | `[0:78]` | | | | |

---

## 5. Complete Policy Mapping Table (23 Actions)

Policy action targets are computed as:
$$q_{\text{des}} = q_{\text{default}} + \text{action\_scale} \times a, \quad \text{action\_scale} = 0.25$$

| Policy Idx | Joint Name | FW CAN Idx | Firmware Name | Sim Actuator | Default Pos (rad) | Training Kp | Training Kd | Effort Limit (N·m) |
|:---:|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 0 | `left_hip_pitch_joint` | 0 | `L_Hip_Pitch` | `left_hip_pitch_joint` | -0.15 | 150.0 | 5.0 | 45.0 |
| 1 | `left_hip_roll_joint` | 1 | `L_Hip_Roll` | `left_hip_roll_joint` | +0.00 | 150.0 | 5.0 | 45.0 |
| 2 | `left_hip_yaw_joint` | 2 | `L_Hip_Yaw` | `left_hip_yaw_joint` | +0.00 | 150.0 | 5.0 | 28.0 |
| 3 | `left_knee_joint` | 3 | `L_Knee` | `left_knee_joint` | +0.45 | 150.0 | 5.0 | 45.0 |
| 4 | `left_ankle_pitch_joint` | 4, 5 | `L_Ankle_A, B` | `left_ankle_pitch_joint` | -0.30 | 110.0 | 5.0 | 40.0 |
| 5 | `left_ankle_roll_joint` | 4, 5 | `L_Ankle_A, B` | `left_ankle_roll_joint` | +0.00 | 110.0 | 5.0 | 17.0 |
| 6 | `right_hip_pitch_joint` | 6 | `R_Hip_Pitch` | `right_hip_pitch_joint` | +0.15 | 150.0 | 5.0 | 45.0 |
| 7 | `right_hip_roll_joint` | 7 | `R_Hip_Roll` | `right_hip_roll_joint` | +0.00 | 150.0 | 5.0 | 45.0 |
| 8 | `right_hip_yaw_joint` | 8 | `R_Hip_Yaw` | `right_hip_yaw_joint` | +0.00 | 150.0 | 5.0 | 28.0 |
| 9 | `right_knee_joint` | 9 | `R_Knee` | `right_knee_joint` | -0.45 | 150.0 | 5.0 | 45.0 |
| 10 | `right_ankle_pitch_joint` | 10, 11 | `R_Ankle_A, B` | `right_ankle_pitch_joint` | +0.30 | 110.0 | 5.0 | 40.0 |
| 11 | `right_ankle_roll_joint` | 10, 11 | `R_Ankle_A, B` | `right_ankle_roll_joint` | +0.00 | 110.0 | 5.0 | 17.0 |
| 12 | `waist_yaw_joint` | 22 | `Waist_Yaw` | `waist_yaw_joint` | +0.00 | 65.0 | 5.0 | 40.0 |
| 13 | `right_shoulder_pitch_joint` | 17 | `R_Shoulder_Pitch` | `right_shoulder_pitch_joint` | +0.25 | 57.0 | 5.0 | 30.0 |
| 14 | `right_shoulder_roll_joint` | 18 | `R_Shoulder_Roll` | `right_shoulder_roll_joint` | +0.05 | 86.0 | 5.0 | 25.0 |
| 15 | `right_shoulder_yaw_joint` | 19 | `R_Shoulder_Yaw` | `right_shoulder_yaw_joint` | +0.00 | 96.0 | 5.0 | 20.0 |
| 16 | `right_elbow_joint` | 20 | `R_Elbow` | `right_elbow_joint` | -0.40 | 40.0 | 2.0 | 12.0 |
| 17 | `right_wrist_yaw_joint` | 21 | `R_Wrist_Yaw` | `right_wrist_yaw_joint` | +0.00 | 40.0 | 2.0 | 12.0 |
| 18 | `left_shoulder_pitch_joint` | 12 | `L_Shoulder_Pitch` | `left_shoulder_pitch_joint` | -0.25 | 57.0 | 5.0 | 30.0 |
| 19 | `left_shoulder_roll_joint` | 13 | `L_Shoulder_Roll` | `left_shoulder_roll_joint` | -0.05 | 86.0 | 5.0 | 25.0 |
| 20 | `left_shoulder_yaw_joint` | 14 | `L_Shoulder_Yaw` | `left_shoulder_yaw_joint` | +0.00 | 96.0 | 5.0 | 20.0 |
| 21 | `left_elbow_joint` | 15 | `L_Elbow` | `left_elbow_joint` | +0.40 | 40.0 | 2.0 | 12.0 |
| 22 | `left_wrist_yaw_joint` | 16 | `L_Wrist_Yaw` | `left_wrist_yaw_joint` | +0.00 | 40.0 | 2.0 | 12.0 |
| — | `neck_yaw_joint` (Not in policy) | 23 | `Neck_Yaw` | `neck_yaw_joint` | +0.00 | 40.0 | 2.0 | 12.0 |
| — | `neck_pitch_joint` (Not in policy)| 24 | `Neck_Pitch` | `neck_pitch_joint` | +0.00 | 40.0 | 2.0 | 12.0 |

---

## 6. Command Ranges & Physics Environment Parameters

* **Command Velocity Ranges (`commands.twist.ranges`):**
  * $v_x$ (Sagittal linear velocity): `[-0.6, 0.8]` m/s
  * $v_y$ (Lateral linear velocity): `[-0.5, 0.5]` m/s
  * $\omega_z$ (Yaw angular velocity): `[-0.8, 0.8]` rad/s
* **Physics Material Settings (`sim.physics_material`):**
  * `static_friction`: 1.0
  * `dynamic_friction`: 1.0
  * `restitution`: 0.0
  * `friction_combine_mode`: `multiply`
  * `restitution_combine_mode`: `multiply`
* **Actuator Model:** `DelayedPDActuator` in Isaac Lab, which implements torque PD law clipping at `effort_limit`. Reflected rotor inertias (`armature`) and viscous joint friction (`frictionloss`) are defined on the joints.
