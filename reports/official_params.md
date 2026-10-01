# Official Parameters & Parameter Mining Report: Asimov 1

**Document Status:** VERIFIED against official sources  
**Sources:**
- `github.com/menloresearch/isaac_asimov`:
  - `source/cyclotron/cyclotron/assets/robots/asimov_1.py` [VERIFIED]
  - `source/cyclotron/cyclotron/tasks/locomotion/velocity_env_cfg.py` [VERIFIED]
- `github.com/menloresearch/asimov-mjlab`:
  - `src/mjlab/asset_zoo/robots/asimov/asimov_constants.py` [VERIFIED]
- `github.com/menloresearch/asimov-1`:
  - `sim-model/xmls/asimov_1.xml` [VERIFIED]
  - `sim-model/urdf/asimov_1.urdf` [VERIFIED]
- Menlo Docs:
  - `docs.menlo.ai/asimov/1/program/api/robot-control` [VERIFIED]
- Installed SDK:
  - `asimov_protocol/v1/asimov_state_pb2.pyi`, `asimov_command_pb2.pyi` [VERIFIED]

---

## 1. Actuator Groups & Dynamic Parameters

In NVIDIA Isaac Lab (`isaac_asimov/source/cyclotron/cyclotron/assets/robots/asimov_1.py`, lines 53–164), actuators are categorized into 11 distinct joint groups modeled with delayed PD control (`DelayedPDActuatorCfg`):

| Actuator Group | Target Joints | Stiffness $K_p$ [N·m/rad] | Damping $K_d$ [N·m·s/rad] | Effort Limit [N·m] | Armature $I_r$ [kg·m²] | Friction $f_c$ [N·m] | Delay [steps] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hip_pitch`** | `*_hip_pitch_joint` | 150.0 | 5.0 | 45.0 | 0.0698 | 0.70 | 0–5 |
| **`hip_roll`** | `*_hip_roll_joint` | 150.0 | 5.0 | 45.0 | 0.1400 | 0.20 | 0–5 |
| **`hip_yaw`** | `*_hip_yaw_joint` | 150.0 | 5.0 | 28.0 | 0.0687 | 0.70 | 0–5 |
| **`knee`** | `*_knee_joint` | 150.0 | 5.0 | 45.0 | 0.0330 | 0.70 | 0–5 |
| **`ankle_pitch`**| `*_ankle_pitch_joint` | 110.0 | 5.0 | 40.0 | 0.0484 | 0.40 | 0–5 |
| **`ankle_roll`** | `*_ankle_roll_joint` | 110.0 | 5.0 | 17.0 | 0.0484 | 0.40 | 0–5 |
| **`waist`** | `waist_yaw_joint` | 65.0 | 5.0 | 40.0 | 0.0698 | 0.70 | 0–5 |
| **`shoulder_pitch`**| `*_shoulder_pitch_joint`| 57.0 | 5.0 | 30.0 | 0.1400 | 0.20 | 0–5 |
| **`shoulder_roll`** | `*_shoulder_roll_joint` | 86.0 | 5.0 | 25.0 | 0.0330 | 0.70 | 0–5 |
| **`shoulder_yaw`** | `*_shoulder_yaw_joint` | 96.0 | 5.0 | 20.0 | 0.0687 | 0.70 | 0–5 |
| **`elbow_wrist`**| `*_elbow_joint`, `*_wrist_yaw_joint`| 40.0 | 2.0 | 12.0 | 0.0242 | 0.40 | 0–5 |

### Synapticon Motor Specifications (`asimov-mjlab/asimov_constants.py`, lines 6–12):
- **Hip pitch:** `EC-A6416-P2-25` (Peak: 120 N·m, Rated: 55 N·m)
- **Hip roll:** `EC-A5013-H17-100` (Peak: 90 N·m, Rated: 45 N·m)
- **Hip yaw:** `EC-A3814-H14-107` (Peak: 60 N·m, Rated: 30 N·m)
- **Knee:** `EC-A4315-P2-36` (Peak: 75 N·m, Rated: 50 N·m)
- **Ankle pitch / roll:** `EC-A4310-P2-36` (Peak: 36 N·m, Rated: 18 N·m)

---

## 2. Velocity Limits & Parameter Conflicts with URDF

In `asimov_1.urdf` (`sim-model/urdf/asimov_1.urdf`), hardware joint limits explicitly constrain maximum angular velocity. This creates notable unevenness across joints due to actuator gearbox reduction ratios:

| Joint Name | URDF Effort [N·m] | URDF Velocity Limit [rad/s] | Approx. RPM | Isaac Lab Effort [N·m] | Conflict / Discrepancy Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `left_hip_pitch_joint` | 45.0 | **12.57** | 120.0 | 45.0 | Effort matches. Velocity limit unconstrained in Isaac Lab. |
| `left_hip_roll_joint` | 45.0 | **3.98** | 38.0 | 45.0 | **Low velocity limit** (100:1 harmonic gear ratio). |
| `left_hip_yaw_joint` | 28.0 | **5.45** | 52.0 | 28.0 | **Low velocity limit** (107:1 gear ratio). |
| `left_knee_joint` | 45.0 | **12.25** | 117.0 | 45.0 | Effort matches. Velocity limit unconstrained in Isaac Lab. |
| `left_ankle_pitch_joint` | 40.0 | **9.32** | 89.0 | 40.0 | Effort matches. |
| `left_ankle_roll_joint` | 17.0 | **9.32** | 89.0 | 17.0 | Effort matches. |
| `right_hip_pitch_joint` | 45.0 | **12.57** | 120.0 | 45.0 | Effort matches. |
| `right_hip_roll_joint` | 45.0 | **3.98** | 38.0 | 45.0 | **Low velocity limit** (100:1 harmonic gear ratio). |
| `right_hip_yaw_joint` | 28.0 | **5.45** | 52.0 | 28.0 | **Low velocity limit** (107:1 gear ratio). |
| `right_knee_joint` | 45.0 | **12.25** | 117.0 | 45.0 | Effort matches. |
| `right_ankle_pitch_joint`| 40.0 | **9.32** | 89.0 | 40.0 | Effort matches. |
| `right_ankle_roll_joint` | 17.0 | **9.32** | 89.0 | 17.0 | Effort matches. |
| `waist_yaw_joint` | 40.0 | **12.57** | 120.0 | 40.0 | Effort matches. |
| `neck_yaw_joint` | 12.0* | 9.32* | 89.0 | — | *Derived model estimate (URDF marked fixed). |
| `neck_pitch_joint` | 12.0* | 9.32* | 89.0 | — | *Derived model estimate (URDF marked fixed). |
| `right_shoulder_pitch_joint`| 30.0 | **3.98** | 38.0 | 30.0 | **Low velocity limit** (100:1 reduction). |
| `right_shoulder_roll_joint` | 25.0 | **12.25** | 117.0 | 25.0 | Effort matches. |
| `right_shoulder_yaw_joint` | 20.0 | **5.45** | 52.0 | 20.0 | **Low velocity limit** (107:1 reduction). |
| `right_elbow_joint` | 12.0 | **9.32** | 89.0 | 12.0 | Effort matches. |
| `right_wrist_yaw_joint` | 12.0 | **9.32** | 89.0 | 12.0 | Effort matches. |
| `left_shoulder_pitch_joint` | 30.0 | **3.98** | 38.0 | 30.0 | **Low velocity limit** (100:1 reduction). |
| `left_shoulder_roll_joint` | 25.0 | **12.25** | 117.0 | 25.0 | Effort matches. |
| `left_shoulder_yaw_joint` | 20.0 | **5.45** | 52.0 | 20.0 | **Low velocity limit** (107:1 reduction). |
| `left_elbow_joint` | 12.0 | **9.32** | 89.0 | 12.0 | Effort matches. |
| `left_wrist_yaw_joint` | 12.0 | **9.32** | 89.0 | 12.0 | Effort matches. |

### Armature / Reflected Inertia Conflict:
- In `asimov_1.xml`: `left_hip_pitch_joint armature = 0.095625`
- In `isaac_asimov`: `hip_pitch armature = 0.0698`
- In `asimov-mjlab`: `ARMATURE_HIP_PITCH = 0.0652`
The values are close (~0.065–0.095 kg·m²), but `asimov_1.xml` uses the higher value, whereas `isaac_asimov` and `asimov-mjlab` compute rotor inertia directly from the Synapticon motor spec sheets.

---

## 3. Official Default Standing Pose (Init State)

Published in `isaac_asimov/source/cyclotron/cyclotron/assets/robots/asimov_1.py`, lines 167–190 (`ASIMOV_1_STANDING_INIT_STATE`):

- **Pelvis Position:** `pos = (0.0, 0.0, 0.639)` m
- **Pelvis Orientation:** `quat = (1.0, 0.0, 0.0, 0.0)`
- **Joint Positions (Bent Knees Stance):**
  ```python
  {
      "left_hip_pitch_joint": -0.15,
      "right_hip_pitch_joint": 0.15,
      "left_hip_roll_joint": 0.0,
      "right_hip_roll_joint": 0.0,
      "left_hip_yaw_joint": 0.0,
      "right_hip_yaw_joint": 0.0,
      "left_knee_joint": 0.45,
      "right_knee_joint": -0.45,
      "left_ankle_pitch_joint": -0.30,
      "right_ankle_pitch_joint": 0.30,
      "left_ankle_roll_joint": 0.0,
      "right_ankle_roll_joint": 0.0,
      "waist_yaw_joint": 0.0,
      "neck_yaw_joint": 0.0,
      "neck_pitch_joint": 0.0,
      "left_shoulder_pitch_joint": -0.25,
      "right_shoulder_pitch_joint": 0.25,
      "left_shoulder_roll_joint": -0.05,
      "right_shoulder_roll_joint": 0.05,
      "left_shoulder_yaw_joint": 0.0,
      "right_shoulder_yaw_joint": 0.0,
      "left_elbow_joint": 0.40,
      "right_elbow_joint": -0.40,
      "left_wrist_yaw_joint": 0.0,
      "right_wrist_yaw_joint": 0.0,
  }
  ```
- **Action Scale:** `ASIMOV_1_ACTION_SCALE = 0.25` [VERIFIED: `asimov_1.py` line 23]. Commanded policy actions are scaled by 0.25 rad when added to default joint positions.

---

## 4. Simulation Timestep & Control Decimation

Published in `isaac_asimov/source/cyclotron/cyclotron/tasks/locomotion/velocity_env_cfg.py`, lines 537–540:

- **Physics Simulation Timestep (`sim.dt`):** `0.005 s` (**200 Hz**)
- **Control Decimation:** `4`
- **Policy Control Rate:** $\frac{200\text{ Hz}}{4} =$ **50 Hz** (`control_dt = 0.020 s`)
- **Render Interval:** `4` (synchronized with policy step)
- This 50 Hz control rate exactly matches Menlo's official firmware and wire-format documentation ([`docs.menlo.ai/asimov/1/program/api/robot-control`](https://docs.menlo.ai/asimov/1/program/api/robot-control)).

---

## 5. Policy Joint Order vs Firmware Joint Order

A critical finding from mining the sources is that the **Reinforcement Learning policy order** (`ASIMOV_1_JOINT_NAMES` in `isaac_asimov`, lines 26–50) **differs** from the **25-joint CAN firmware order**:

| Index | Isaac Lab Policy Order (`ASIMOV_1_JOINT_NAMES`) | Firmware Order (CAN Bus Order) | Difference / Inversion |
| :---: | :--- | :--- | :--- |
| **0** | `left_hip_pitch_joint` | `L_Hip_Pitch` | Same |
| **1** | `left_hip_roll_joint` | `L_Hip_Roll` | Same |
| **2** | `left_hip_yaw_joint` | `L_Hip_Yaw` | Same |
| **3** | `left_knee_joint` | `L_Knee` | Same |
| **4** | `left_ankle_pitch_joint` | `L_Ankle_A` | Ideal joint vs A/B motor |
| **5** | `left_ankle_roll_joint` | `L_Ankle_B` | Ideal joint vs A/B motor |
| **6** | `right_hip_pitch_joint` | `R_Hip_Pitch` | Same |
| **7** | `right_hip_roll_joint` | `R_Hip_Roll` | Same |
| **8** | `right_hip_yaw_joint` | `R_Hip_Yaw` | Same |
| **9** | `right_knee_joint` | `R_Knee` | Same |
| **10** | `right_ankle_pitch_joint` | `R_Ankle_A` | Ideal joint vs A/B motor |
| **11** | `right_ankle_roll_joint` | `R_Ankle_B` | Ideal joint vs A/B motor |
| **12** | `waist_yaw_joint` | `L_Shoulder_Pitch` | **INVERSION**: Policy places Waist at 12; Firmware places Left Arm at 12–16 |
| **13** | `right_shoulder_pitch_joint` | `L_Shoulder_Roll` | **INVERSION**: Policy places Right Arm at 13–17 |
| **14** | `right_shoulder_roll_joint` | `L_Shoulder_Yaw` | |
| **15** | `right_shoulder_yaw_joint` | `L_Elbow` | |
| **16** | `right_elbow_joint` | `L_Wrist_Yaw` | |
| **17** | `right_wrist_yaw_joint` | `R_Shoulder_Pitch` | **INVERSION**: Firmware places Right Arm at 17–21 |
| **18** | `left_shoulder_pitch_joint` | `R_Shoulder_Roll` | **INVERSION**: Policy places Left Arm at 18–22 |
| **19** | `left_shoulder_roll_joint` | `R_Shoulder_Yaw` | |
| **20** | `left_shoulder_yaw_joint` | `R_Elbow` | |
| **21** | `left_elbow_joint` | `R_Wrist_Yaw` | |
| **22** | `left_wrist_yaw_joint` | `Waist_Yaw` | Firmware places Waist at 22 |
| **23** | *(Not in policy)* | `Neck_Yaw` | Firmware only (CAN 24) |
| **24** | *(Not in policy)* | `Neck_Pitch` | Firmware only (CAN 25) |

---

## 6. Ankle Handling & A/B Kinematic Mapping

- **Status in Training Repositories:** Both `isaac_asimov` and `asimov-mjlab` **bypass the A/B linkage entirely**, training the policy exclusively on ideal orthogonal pitch and roll joints (`*_ankle_pitch_joint`, `*_ankle_roll_joint`).
- **Status in Firmware / SDK:** The physical robot firmware exposes raw actuator channels `Ankle_A` and `Ankle_B`.
- **Closed-Form Mapping Formula:** **UNPUBLISHED / NOT IN PUBLIC CODE**. Per project specifications, a swappable linear differential approximation module with measurable CAD crank parameters is constructed in Task 5.
