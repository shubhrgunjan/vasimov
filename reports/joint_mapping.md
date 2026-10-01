# Joint Mapping Report: Asimov 1 Firmware vs MuJoCo Simulation Model

**Document Status:** VERIFIED against official sources  
**Sources:**
- Menlo Docs: `https://docs.menlo.ai/asimov/1/program/api/robot-control#joint-order` [VERIFIED]
- Asimov 1 Official Repository: `github.com/menloresearch/asimov-1` (`sim-model/README.md`, `sim-model/xmls/asimov_1.xml`, `sim-model/urdf/asimov_1.urdf`) [VERIFIED]
- Menlo Locomotion / Build Guides: Parallel RSU Ankle Documentation [VERIFIED]

---

## 1. Overview and Key Questions Answered

### Does the MuJoCo model have 25 actuated joints?
**NO.** [VERIFIED: `sim-model/xmls/asimov_1.xml` line 93-266, `sim-model/README.md`]
- The official MJCF model (`sim-model/xmls/asimov_1.xml`) defines **23 hinge joints** and **1 free joint (`floating_base`)**.
- The model contains **0 predefined actuators (`model.nu = 0`)**. The official repository states:
  > *"No actuators are defined in the XML — the training sim configures them in Python. Torque motors for the standalone viewer (training adds position actuators programmatically via BuiltinPositionActuatorCfg; the viewer applies its own PD in JS and writes torque to ctrl)."*
- In contrast, the real robot firmware specifies **25 actuator channels** (CAN bus IDs 1 through 25) driving 25 actuators.

### Which firmware joints have no counterpart in the model?
- **`Neck_Yaw` (Firmware Index 23, CAN ID 24)**: [VERIFIED: MISSING in MJCF, fixed in URDF]
  In the MJCF model, `neck_yaw_link` is rigidly mounted to `waist_yaw_link` without a `<joint>` tag. In the URDF description (`asimov_1.urdf`), it is marked as `type="fixed"` (`neck_yaw_link_fixed`).
- **`Neck_Pitch` (Firmware Index 24, CAN ID 25)**: [VERIFIED: MISSING in MJCF, fixed in URDF]
  In the MJCF model, `neck_pitch_link` is rigidly mounted to `neck_yaw_link` without a `<joint>` tag. In the URDF description (`asimov_1.urdf`), it is marked as `type="fixed"` (`neck_pitch_link_fixed`).

### Which model joints have no counterpart in the firmware actuator array?
- **`floating_base` (MuJoCo Joint Index 0)**: [VERIFIED: EXTRA in MuJoCo]
  A 6-DOF unactuated spatial free joint representing the floating base of the robot root (`pelvis_link`), with 7 position coordinates (3 translation + 4 quaternion) and 6 velocity DOFs.

---

## 2. Joint Mapping Table

| Firmware Idx | Firmware Name | Group | CAN ID | MuJoCo Joint Name | MuJoCo Actuator Name | Range Match? | Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | `L_Hip_Pitch` | Left Leg | 1 | `left_hip_pitch_joint` | None (`nu=0`) | **MATCH** (`[-1.57, +1.57]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 1, 0]`. |
| **1** | `L_Hip_Roll` | Left Leg | 2 | `left_hip_roll_joint` | None (`nu=0`) | **MATCH** (`[-0.7854, +0.7854]` rad) | **MATCH** | Direct correspondence. Axis: `[1, 0, 0]`. |
| **2** | `L_Hip_Yaw` | Left Leg | 3 | `left_hip_yaw_joint` | None (`nu=0`) | **MATCH** (`[-0.7854, +0.7854]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 0, -1]`. |
| **3** | `L_Knee` | Left Leg | 4 | `left_knee_joint` | None (`nu=0`) | **MATCH** (`[0.0, +1.50]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 1, 0]`. |
| **4** | `L_Ankle_A` | Left Leg | 5 | `left_ankle_pitch_joint` | None (`nu=0`) | **DIFFERENT** | **DIFFERENT** | Physical robot has actuator motor A in parallel RSU linkage. MuJoCo has serial pitch hinge `[-0.35, +0.35]` rad. |
| **5** | `L_Ankle_B` | Left Leg | 6 | `left_ankle_roll_joint` | None (`nu=0`) | **DIFFERENT** | **DIFFERENT** | Physical robot has actuator motor B in parallel RSU linkage. MuJoCo has serial roll hinge `[-0.10, +0.10]` rad. |
| **6** | `R_Hip_Pitch` | Right Leg | 7 | `right_hip_pitch_joint` | None (`nu=0`) | **MATCH** (`[-1.57, +1.57]` rad) | **MATCH** | Direct correspondence. Axis: `[0, -1, 0]`. |
| **7** | `R_Hip_Roll` | Right Leg | 8 | `right_hip_roll_joint` | None (`nu=0`) | **MATCH** (`[-0.7854, +0.7854]` rad) | **MATCH** | Direct correspondence. Axis: `[1, 0, 0]`. |
| **8** | `R_Hip_Yaw` | Right Leg | 9 | `right_hip_yaw_joint` | None (`nu=0`) | **MATCH** (`[-0.7854, +0.7854]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 0, -1]`. |
| **9** | `R_Knee` | Right Leg | 10 | `right_knee_joint` | None (`nu=0`) | **MATCH** (`[-1.50, 0.0]` rad) | **MATCH** | Axis inverted: `[0, -1, 0]`; range is negative in MuJoCo. |
| **10** | `R_Ankle_A` | Right Leg | 11 | `right_ankle_pitch_joint` | None (`nu=0`) | **DIFFERENT** | **DIFFERENT** | Physical robot has actuator motor A in parallel RSU linkage. MuJoCo has serial pitch hinge `[-0.35, +0.35]` rad. |
| **11** | `R_Ankle_B` | Right Leg | 12 | `right_ankle_roll_joint` | None (`nu=0`) | **DIFFERENT** | **DIFFERENT** | Physical robot has actuator motor B in parallel RSU linkage. MuJoCo has serial roll hinge `[-0.10, +0.10]` rad. |
| **12** | `L_Shoulder_Pitch` | Left Arm | 13 | `left_shoulder_pitch_joint` | None (`nu=0`) | **MATCH** (`[-3.1416, +0.8727]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 1, 0]`. |
| **13** | `L_Shoulder_Roll` | Left Arm | 14 | `left_shoulder_roll_joint` | None (`nu=0`) | **MATCH** (`[-1.5708, 0.0]` rad) | **MATCH** | Direct correspondence. Axis: `[-1, 0, 0]`. |
| **14** | `L_Shoulder_Yaw` | Left Arm | 15 | `left_shoulder_yaw_joint` | None (`nu=0`) | **MATCH** (`[-1.5708, +1.5708]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 0, -1]`. |
| **15** | `L_Elbow` | Left Arm | 16 | `left_elbow_joint` | None (`nu=0`) | **MATCH** (`[0.0, +2.0944]` rad) | **MATCH** | Direct correspondence. `ref=+0.7854` in XML. |
| **16** | `L_Wrist_Yaw` | Left Arm | 17 | `left_wrist_yaw_joint` | None (`nu=0`) | **MATCH** (`[-3.1416, +3.1416]` rad) | **MATCH** | Skewed revolute axis: `[0.766, 0, -0.643]`. |
| **17** | `R_Shoulder_Pitch` | Right Arm | 18 | `right_shoulder_pitch_joint` | None (`nu=0`) | **MATCH** (`[-0.8727, +3.1416]` rad) | **MATCH** | Direct correspondence. Axis: `[0, -1, 0]`. |
| **18** | `R_Shoulder_Roll` | Right Arm | 19 | `right_shoulder_roll_joint` | None (`nu=0`) | **MATCH** (`[0.0, +1.5708]` rad) | **MATCH** | Direct correspondence. Axis: `[-1, 0, 0]`. |
| **19** | `R_Shoulder_Yaw` | Right Arm | 20 | `right_shoulder_yaw_joint` | None (`nu=0`) | **MATCH** (`[-1.5708, +1.5708]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 0, -1]`. |
| **20** | `R_Elbow` | Right Arm | 21 | `right_elbow_joint` | None (`nu=0`) | **MATCH** (`[-2.0944, 0.0]` rad) | **MATCH** | Direct correspondence. `ref=-0.7854` in XML. |
| **21** | `R_Wrist_Yaw` | Right Arm | 22 | `right_wrist_yaw_joint` | None (`nu=0`) | **MATCH** (`[-3.1416, +3.1416]` rad) | **MATCH** | Skewed revolute axis: `[0.766, 0, -0.643]`. |
| **22** | `Waist_Yaw` | Torso | 23 | `waist_yaw_joint` | None (`nu=0`) | **MATCH** (`[-1.5708, +1.5708]` rad) | **MATCH** | Direct correspondence. Axis: `[0, 0, 1]`. |
| **23** | `Neck_Yaw` | Head | 24 | *None* | *None* | **N/A** | **MISSING** | Fixed body in URDF and MJCF. No joint in simulation model. |
| **24** | `Neck_Pitch` | Head | 25 | *None* | *None* | **N/A** | **MISSING** | Fixed body in URDF and MJCF. No joint in simulation model. |
| — | *(Floating Base)* | Root | — | `floating_base` | *None* | Unlimited | **EXTRA** | 6-DOF freejoint on `pelvis_link`. |

---

## 3. Detailed Investigation of Ankle Actuation: A/B vs Pitch/Roll

### Physical Robot Ankle Architecture:
[VERIFIED: Official Menlo Build Documentation & Assembly Guide `step-9`, `step-10`, `step-13`]
- On the physical robot, the ankle is actuated by two distinct revolute brushless motors mounted side-by-side in the calf:
  - Left Leg: `L_Ankle_A` (Actuator ID 5) and `L_Ankle_B` (Actuator ID 6).
  - Right Leg: `R_Ankle_A` (Actuator ID 11) and `R_Ankle_B` (Actuator ID 12).
- These motors connect to the foot through a **parallel RSU (Revolute-Spherical-Universal) linkage**:
  - Each motor drives a linkage crank (`step-10`).
  - Steel push-pull rods with spherical rod ends (`step-13`) transmit motion from the cranks to the left and right sides of the foot plate.
  - When motors A and B push or pull in unison, the foot pitches up or down.
  - When motors A and B move in opposite directions, the foot rolls left or right.

### Simulation Model Architecture:
[VERIFIED: `sim-model/xmls/asimov_1.xml` lines 121–136, 164–179]
- In MuJoCo, the complex 4-bar parallel linkage is **NOT modeled**.
- Instead, each ankle is modeled as **two orthogonal serial hinge joints**:
  - `*_ankle_pitch_joint`: axis `[0, ±1, 0]`, range `[-0.35, +0.35]` rad (approx. ±20.0°).
  - `*_ankle_roll_joint`: axis `[-1, 0, 0]`, range `[-0.10, +0.10]` rad (approx. ±5.7°).

### Official Documentation on the A/B-to-Pitch/Roll Relationship:
[VERIFIED: Quoted from official Menlo training & sim2real guides]
> *"The Asimov 1 robot utilizes a parallel RSU (Revolute-Spherical-Universal) ankle mechanism rather than a traditional serial single-DOF design. Two revolute actuators work through parallel linkages to produce ankle pitch and roll.*
> 
> *In simulation: Ankle pitch and roll are modeled as directly actuated ideal joints.*
> 
> *Physical robot: Because the two actuators move endpoints on a straight bar, a kinematic mapping is required to convert actuator A/B positions into the pitch and roll coordinates needed for Sim2Real deployment."*

### What is Unclear / Missing in the Public Upstream Code:
- The exact mathematical transfer function (whether linear Jacobian approximation $\theta_{\text{pitch}} = c_1(A + B)$, $\theta_{\text{roll}} = c_2(A - B)$ or full nonlinear closed-form 4-bar linkage geometry with specific linkage lengths) is **NOT published** in `asimov-1` repo or `robot-control` API documentation.
- The firmware sends raw actuator telemetry for CAN bus IDs 5, 6, 11, and 12, whereas locomotion policy training environments operate entirely in orthogonal pitch/roll coordinates.
- *Per project requirements, NO conversion is implemented in this step. This mapping will be formally addressed in STEP 2.*
