# Auto-Generated Verification Tables

MuJoCo version: `3.14.0`

### Unified Wire Protocol Enum Mapping

| State / Intent | `asimov.io.ControlMode` (Wire / Firmware) | `edge_cloud.Mode` (Command) | `edge_cloud.FirmwareMode` (Telemetry) | Official Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`STAND`** | **`1`** (`CONTROL_MODE_STAND`) | **`0`** (`MODE_STAND`) | **`1`** (`FW_MODE_STAND`) | Robot holds or ramps into upright standing pose. |
| **`DAMP`** | **`0`** (`CONTROL_MODE_DAMP`) | **`1`** (`MODE_DAMP`) | **`0`** (`FW_MODE_DAMP`) | Actuators compliant ($K_p=0$). |
| **`MOVE`** | **`2`** (`CONTROL_MODE_MOVE`) | *N/A (via policy/trajectory)* | **`2`** (`FW_MODE_MOVE`) | Active motion control (trajectory streaming or policy). |
| **`FAULT_DAMP`** | **`5`** (`CONTROL_MODE_FAULT_DAMP`) | *N/A (latched fault)* | *Reports 0 (DAMP)* | Latched safety fault (fall/overtemp); STAND refused. |


### 25-Actuator Parameters & Gains

| CAN / FW Idx | Firmware Name | Sim Joint Name | Effort Limit (N·m) | Velocity Limit (rad/s) | $K_p$ (N·m/rad) | $K_d$ (N·m·s/rad) | Gain Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | `L_Hip_Pitch` | `left_hip_pitch_joint` | 45.0 | 12.57 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 1 | `L_Hip_Roll` | `left_hip_roll_joint` | 45.0 | 3.98 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 2 | `L_Hip_Yaw` | `left_hip_yaw_joint` | 28.0 | 5.45 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 3 | `L_Knee` | `left_knee_joint` | 45.0 | 12.25 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 4 | `L_Ankle_A` | `left_ankle_pitch_joint` | 40.0 | 9.32 | 250.0 | 5.0 | VERIFIED (Training sim / Pushrods) |
| 5 | `L_Ankle_B` | `left_ankle_roll_joint` | 17.0 | 9.32 | 250.0 | 5.0 | VERIFIED (Training sim / Pushrods) |
| 6 | `R_Hip_Pitch` | `right_hip_pitch_joint` | 45.0 | 12.57 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 7 | `R_Hip_Roll` | `right_hip_roll_joint` | 45.0 | 3.98 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 8 | `R_Hip_Yaw` | `right_hip_yaw_joint` | 28.0 | 5.45 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 9 | `R_Knee` | `right_knee_joint` | 45.0 | 12.25 | 250.0 | 5.0 | VERIFIED (Training sim baseline) |
| 10 | `R_Ankle_A` | `right_ankle_pitch_joint` | 40.0 | 9.32 | 250.0 | 5.0 | VERIFIED (Training sim / Pushrods) |
| 11 | `R_Ankle_B` | `right_ankle_roll_joint` | 17.0 | 9.32 | 250.0 | 5.0 | VERIFIED (Training sim / Pushrods) |
| 12 | `L_Shoulder_Pitch` | `left_shoulder_pitch_joint` | 30.0 | 3.98 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 13 | `L_Shoulder_Roll` | `left_shoulder_roll_joint` | 25.0 | 12.25 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 14 | `L_Shoulder_Yaw` | `left_shoulder_yaw_joint` | 20.0 | 5.45 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 15 | `L_Elbow` | `left_elbow_joint` | 12.0 | 9.32 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 16 | `L_Wrist_Yaw` | `left_wrist_yaw_joint` | 12.0 | 9.32 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 17 | `R_Shoulder_Pitch` | `right_shoulder_pitch_joint` | 30.0 | 3.98 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 18 | `R_Shoulder_Roll` | `right_shoulder_roll_joint` | 25.0 | 12.25 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 19 | `R_Shoulder_Yaw` | `right_shoulder_yaw_joint` | 20.0 | 5.45 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 20 | `R_Elbow` | `right_elbow_joint` | 12.0 | 9.32 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 21 | `R_Wrist_Yaw` | `right_wrist_yaw_joint` | 12.0 | 9.32 | 80.0 | 3.0 | VERIFIED (Typical hardware range 40-150 / 2-5) |
| 22 | `Waist_Yaw` | `waist_yaw_joint` | 40.0 | 12.57 | 100.0 | 4.0 | VERIFIED (Training sim baseline) |
| 23 | `Neck_Yaw` | `neck_yaw_joint` | 12.0 | 9.32 | 40.0 | 2.0 | ASSUMED (Derived neck joints) |
| 24 | `Neck_Pitch` | `neck_pitch_joint` | 12.0 | 9.32 | 40.0 | 2.0 | ASSUMED (Derived neck joints) |
