# Locomotion Policy Contract & Readiness Audit

**Virtual Asimov 1 in MuJoCo — Step 4 Audit Report**  
*Date: 2026-10-01*  
*Status: Verified against official codebases & Hugging Face model repository*

---

## 1. Executive Summary & Checkpoint Verdict

| Item | Status | Details |
|---|---|---|
| **Pretrained Checkpoint Available?** | **YES** | Hosted on Hugging Face: [`Menlo/asimov1-locomotion-0818`](https://huggingface.co/Menlo/asimov1-locomotion-0818) |
| **Export Formats** | **ONNX** (`policy.onnx`, 831,809 bytes) + YAML configurations (`env.yaml`, `agent.yaml`) |
| **Policy Control Rate** | **50 Hz** ($dt = 0.020\text{ s}$, decimation = 4 from 200 Hz physics) |
| **Observation Dimension** | **78 floats** |
| **Action Dimension** | **23 floats** (Joint position setpoint offsets; Neck joints 23 & 24 omitted) |
| **Action Scale** | **0.25** |
| **Training Engines** | Isaac Lab (PhysX 5 TGS) & Asimov-mjlab (MuJoCo 3.x) |

---

## 2. Policy Observation Specification (78-D)

The policy expects 78 scalar inputs at 50 Hz, evaluated in the following exact order:

| Obs Term | Dim | Scale | Lag / Noise | Source & Description |
|---|:---:|:---:|---|---|
| `base_ang_vel` | 3 | 0.25 | Lag $[0, 1]$, Noise $\pm 0.01$ | Pelvis angular velocity $[\omega_x, \omega_y, \omega_z]$ in body frame (IMU gyro) |
| `projected_gravity` | 3 | 1.00 | Lag $[0, 2]$, Noise $\pm 0.02$ | World gravity $[0, 0, -1]$ projected into pelvis body frame $[g_x, g_y, g_z]$ |
| `velocity_command` | 3 | 1.00 | None | Commanded $[v_x, v_y, \omega_z]$ |
| `joint_pos_slot01` | 9 | 1.00 | Noise $\pm 0.01$ | Position relative to default pose for Slot 0 & 1 joints |
| `joint_pos_slot23` | 8 | 1.00 | Noise $\pm 0.01$ | Position relative to default pose for Slot 2 & 3 joints |
| `joint_pos_slot45` | 6 | 1.00 | Noise $\pm 0.01$ | Position relative to default pose for Slot 4 & 5 joints |
| `joint_vel_slot01` | 9 | 0.10 | Noise $\pm 0.50$ | Angular velocity for Slot 0 & 1 joints |
| `joint_vel_slot23` | 8 | 0.10 | Noise $\pm 0.50$ | Angular velocity for Slot 2 & 3 joints |
| `joint_vel_slot45` | 6 | 0.10 | Noise $\pm 0.50$ | Angular velocity for Slot 4 & 5 joints |
| `previous_actions` | 23 | 1.00 | None | Raw network output from preceding control step ($t - 1$) |
| **Total** | **78** | | | |

*Provenance:* Verified in `upstream/isaac_asimov/source/cyclotron/cyclotron/tasks/locomotion/velocity_env_cfg.py` lines 181–236 and `upstream/isaac_asimov/README.md` lines 148–150.

### CAN Slot Groupings
- **SLOT 0 & 1 (9 joints):** `left_hip_pitch_joint`, `left_hip_roll_joint`, `right_hip_pitch_joint`, `right_hip_roll_joint`, `waist_yaw_joint`, `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`
- **SLOT 2 & 3 (8 joints):** `left_hip_yaw_joint`, `left_knee_joint`, `right_hip_yaw_joint`, `right_knee_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint`
- **SLOT 4 & 5 (6 joints):** `left_ankle_pitch_joint`, `left_ankle_roll_joint`, `right_ankle_pitch_joint`, `right_ankle_roll_joint`, `right_wrist_yaw_joint`, `left_wrist_yaw_joint`

---

## 3. Action Specification & Target Computation (23-D)

The policy outputs 23 normalized actions $a \in [-1, 1]^{23}$.  
Target positions sent to the PD controllers are computed via:
$$q_{\text{des}} = q_{\text{default}} + \text{action\_scale} \times a$$
where $\text{action\_scale} = 0.25$.

### Joint Ordering & Mapping to Virtual Asimov Adapter

| Policy Index | Sim Joint Name | Firmware CAN Name | FW CAN Idx | Default Pos (rad) | Training Kp | Training Kd |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 0 | `left_hip_pitch_joint` | `L_Hip_Pitch` | 0 | 0.0 | 150.0 | 5.0 |
| 1 | `left_hip_roll_joint` | `L_Hip_Roll` | 1 | 0.0 | 150.0 | 5.0 |
| 2 | `left_hip_yaw_joint` | `L_Hip_Yaw` | 2 | 0.0 | 150.0 | 5.0 |
| 3 | `left_knee_joint` | `L_Knee` | 3 | 0.0 | 150.0 | 5.0 |
| 4 | `left_ankle_pitch_joint` | `L_Ankle_A` / `B` | 4, 5 | 0.0 | 110.0 | 5.0 |
| 5 | `left_ankle_roll_joint` | `L_Ankle_A` / `B` | 4, 5 | 0.0 | 110.0 | 5.0 |
| 6 | `right_hip_pitch_joint` | `R_Hip_Pitch` | 6 | 0.0 | 150.0 | 5.0 |
| 7 | `right_hip_roll_joint` | `R_Hip_Roll` | 7 | 0.0 | 150.0 | 5.0 |
| 8 | `right_hip_yaw_joint` | `R_Hip_Yaw` | 8 | 0.0 | 150.0 | 5.0 |
| 9 | `right_knee_joint` | `R_Knee` | 9 | 0.0 | 150.0 | 5.0 |
| 10 | `right_ankle_pitch_joint` | `R_Ankle_A` / `B` | 10, 11 | 0.0 | 110.0 | 5.0 |
| 11 | `right_ankle_roll_joint` | `R_Ankle_A` / `B` | 10, 11 | 0.0 | 110.0 | 5.0 |
| 12 | `waist_yaw_joint` | `Waist_Yaw` | 22 | 0.0 | 65.0 | 5.0 |
| 13 | `right_shoulder_pitch_joint` | `R_Shoulder_Pitch`| 17 | 0.0 | 57.0 | 5.0 |
| 14 | `right_shoulder_roll_joint` | `R_Shoulder_Roll` | 18 | 0.0 | 86.0 | 5.0 |
| 15 | `right_shoulder_yaw_joint` | `R_Shoulder_Yaw` | 19 | 0.0 | 96.0 | 5.0 |
| 16 | `right_elbow_joint` | `R_Elbow` | 20 | 0.0 | 40.0 | 2.0 |
| 17 | `right_wrist_yaw_joint` | `R_Wrist_Yaw` | 21 | 0.0 | 40.0 | 2.0 |
| 18 | `left_shoulder_pitch_joint` | `L_Shoulder_Pitch` | 12 | 0.0 | 57.0 | 5.0 |
| 19 | `left_shoulder_roll_joint` | `L_Shoulder_Roll` | 13 | 0.0 | 86.0 | 5.0 |
| 20 | `left_shoulder_yaw_joint` | `L_Shoulder_Yaw` | 14 | 0.0 | 96.0 | 5.0 |
| 21 | `left_elbow_joint` | `L_Elbow` | 15 | 0.0 | 40.0 | 2.0 |
| 22 | `left_wrist_yaw_joint` | `L_Wrist_Yaw` | 16 | 0.0 | 40.0 | 2.0 |
| — | `neck_yaw_joint` (Not in policy) | `Neck_Yaw` | 23 | 0.0 | 40.0 | 2.0 |
| — | `neck_pitch_joint` (Not in policy)| `Neck_Pitch` | 24 | 0.0 | 40.0 | 2.0 |

*Provenance:* Verified in `upstream/isaac_asimov/source/cyclotron/cyclotron/assets/robots/asimov_1.py` lines 26–160.

---

## 4. Command Velocity Ranges

- **Sagittal velocity ($v_x$):** $[-0.6, 0.8]\text{ m/s}$ (Isaac Lab) / $[-0.8, 0.8]\text{ m/s}$ (asimov-mjlab)
- **Lateral velocity ($v_y$):** $[-0.5, 0.5]\text{ m/s}$ (Isaac Lab) / $[-0.6, 0.6]\text{ m/s}$ (asimov-mjlab)
- **Yaw rate ($\omega_z$):** $[-0.8, 0.8]\text{ rad/s}$ (Isaac Lab) / $[-0.6, 0.6]\text{ rad/s}$ (asimov-mjlab)

---

## 5. Training Requirements & Performance (If Training from Scratch)

- **Compute Hardware:** Single NVIDIA RTX 4090 (24GB VRAM) or RTX A6000 (48GB VRAM).
- **Environment Scale:** 4096 parallel environments in Isaac Lab.
- **Training Time:**
  - Quick sanity test (128 envs, 100 iterations): ~10 minutes on RTX 4090.
  - Full convergence (5,000 iterations): ~2.5 to 4.0 hours on RTX 4090 / A6000.
- **Command:**
  ```bash
  ./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless
  ```

---

## 6. Open Questions for Step 5 (Locomotion Policy Runner)

1. **Policy Runner Runtime Engine:**
   Will ONNX Runtime (`onnxruntime` CPU/GPU package) be used to execute `policy.onnx` locally inside `vasimov/edge`, or will PyTorch JIT / RSL-RL runner be preferred?
2. **Observation History Buffer:**
   Does the current ONNX model in `Menlo/asimov1-locomotion-0818` contain internal recurrence / history, or does the host policy runner need to maintain an observation queue of previous actions? (README specifies 78 input floats including previous action).
3. **Upper Body & Neck Handling:**
   The policy drives 23 joints (legs, waist, arms), leaving neck yaw (joint 23) and neck pitch (joint 24) un-actuated by RL. Should the Edge hold the neck fixed at 0.0 rad with low compliance gains ($K_p=40, K_d=2$), or expose a separate SDK neck setpoint API?
4. **Arm Trajectory vs Locomotion Blending:**
   When the SDK commands arm trajectories (`Robot.trajectory()`), should the locomotion policy yield control of the arm joints to the SDK while retaining leg balance, or does the locomotion policy strictly commandeer all 23 joints during `MOVE`?
