# Virtual Asimov Edge: Fidelity Specification

This document explicitly defines the fidelity of every command, telemetry field, system event, and physical simulation subsystem in Virtual Asimov Edge (`vasimov`).

Per project requirements, every item is strictly labeled into one of four categories:
- **`REAL-EQUIVALENT`**: Behaves identically to official hardware/firmware specification and wire protocol.
- **`APPROXIMATE`**: Accurately reproduces functional behavior and physical effects using analytical or engineering approximations.
- **`PLACEHOLDER`**: Implements the official protocol field with valid, non-crashing nominal constants; underlying physical subsystem is not yet modeled.
- **`NOT IMPLEMENTED`**: Explicitly omitted or unsupported in this version (never silently faked).

---

## 1. Commands (`asimov.io.RobotCommand` / `menlo.edge.CloudCommand`)

| Command Channel / Field | Classification | Description & Provenance |
| :--- | :--- | :--- |
| `mode = CONTROL_MODE_STAND` | **`REAL-EQUIVALENT`** | Initiates smooth 2.0s linear joint trajectory ramp from current pose to default standing pose (`config/joints.yaml`). Verified against SDK `Robot.stand()`. |
| `mode = CONTROL_MODE_DAMP` | **`REAL-EQUIVALENT`** | Sets joint stiffness $K_p = 0$, $K_d = 2.0$. If virtual gantry is released, robot physically collapses under gravity. Verified against SDK `Robot.damp()`. |
| `all_trajectory` (`JointSegment`) | **`REAL-EQUIVALENT`** | 25 positions in firmware order. Validates finite values, enforces 25-element length, supports optional per-packet $K_p/K_d$ overrides. Monitored by 2.0s trajectory watchdog (auto-DAMP on silence). |
| `policy` (Velocity $v_x, v_y, v_\text{yaw}$) | **`PLACEHOLDER`** | **NoPolicy Stub**: Neural locomotion policy is out of scope for Step 3. Edge logs loudly (`[VASIMOV NO-POLICY STUB]`), remains in `STAND`, and holds position without pretending to walk. Zero-velocity hold watchdog active after 2.0s silence. |
| `eol` (EOL Calibration Commands) | **`NOT IMPLEMENTED`** | Factory end-of-line motor encoder zeroing and CAN ID assignment are not simulated. |

---

## 2. Telemetry (`asimov.io.RobotState` on UDP `:8851`)

| Protobuf Field | Classification | Description & Provenance |
| :--- | :--- | :--- |
| `protocol_version` | **`REAL-EQUIVALENT`** | Value `1`. Verified against `asimov_protocol/v1/asimov_state.proto`. |
| `sequence` | **`REAL-EQUIVALENT`** | Monotonic 32-bit unsigned counter incremented on every 10 Hz state packet. |
| `timestamp_us` | **`REAL-EQUIVALENT`** | Firmware uptime clock in microseconds since boot/restart. Resets on virtual firmware restart, correctly triggering SDK `STATE_CLOCK_RESET_S` detection. |
| `current_mode` | **`REAL-EQUIVALENT`** | ControlMode enum: `DAMP=0`, `STAND=1`, `MOVE=2`, `FAULT_DAMP=5`. Verified against `asimov_common.proto`. |
| `joint_pos` | **`REAL-EQUIVALENT`** | 25 joint positions in official firmware order (CAN 0..24), converted from MuJoCo generalized coordinates via `JointAdapter` with coupled ankle pitch/roll $\leftrightarrow$ pushrod motors A/B. |
| `joint_vel` | **`REAL-EQUIVALENT`** | 25 joint velocities in firmware order via `JointAdapter.sim_to_firmware_velocities()`. |
| `joint_current` | **`PLACEHOLDER`** | Constant `0.0 A`. Actuator torque constants ($K_t$, N·m/A) are not published in upstream repositories. |
| `joint_temp` | **`PLACEHOLDER`** | Constant `25.0 °C` nominal room temperature (or dynamically set via fault injection). Actuator thermal dissipation dynamics are not modeled. |
| `base_quat` | **`REAL-EQUIVALENT`** | Pelvis orientation quaternion $[w, x, y, z]$ directly from MuJoCo freejoint `qpos[3:7]`. |
| `base_ang_vel` | **`REAL-EQUIVALENT`** | Pelvis angular velocity $[\omega_x, \omega_y, \omega_z]$ in body frame from MuJoCo `qvel[3:6]`. |
| `projected_gravity` | **`REAL-EQUIVALENT`** | World gravity vector $[0, 0, -1]$ rotated into base body frame via quaternion rotation matrix ($g_b = R(q)^T g_w$). Matches `isaac_asimov` observation term ($g_z \approx -1.0$ when upright). |
| `error_flags` | **`REAL-EQUIVALENT`** | Official bitfield (`bit 0 = FAULT_DAMP_LATCHED`, `bit 1+n = AlertId n`). Matches `asimov_protocol/error_flags.py`. |
| `active_alerts` | **`REAL-EQUIVALENT`** | Repeated `ActiveAlert` messages using real enum IDs from `asimov_protocol.alerts` (`FALL_DETECTED=7`, `MOTOR_OVERTEMP=2`, `CAN_BUS_OFF=1`). |
| `battery` | **`PLACEHOLDER`** | Virtual battery pack summary: nominal 48.0 V, SoC% (default 100.0%, configurable, optional linear discharge, settable via :8852 API). Tested against SDK preflight: trips `battery_low` below 20% and clears above 20%. |

---

## 3. Edge-Cloud Telemetry (`menlo.edge.EdgeTelemetry`)

> [!NOTE]
> UDP mode transmits `asimov.io.RobotState` directly. `EdgeTelemetry` builders are implemented and tested, but unexercised on the UDP transport.

| Protobuf Field | Classification | Description & Provenance |
| :--- | :--- | :--- |
| `timestamp_us` | **`IMPLEMENTED-UNEXERCISED`** | Wall-clock UNIX timestamp in microseconds. |
| `fw_timestamp_us` | **`IMPLEMENTED-UNEXERCISED`** | Firmware uptime in microseconds since boot/restart. |
| `sequence` | **`IMPLEMENTED-UNEXERCISED`** | Monotonic sequence number. |
| `fw_mode` | **`IMPLEMENTED-UNEXERCISED`** | Telemetry enum: `FW_MODE_DAMP=0`, `FW_MODE_STAND=1`, `FW_MODE_MOVE=2`. Verified against `edge_cloud.proto`. |
| `joint_pos`, `joint_vel` | **`IMPLEMENTED-UNEXERCISED`** | 25 joints in firmware order via `JointAdapter`. |
| `joint_current` | **`PLACEHOLDER`** | Vector of 25 zeros. |
| `joint_temp` | **`PLACEHOLDER`** | Vector of 25 values at `25.0 °C` (or injected temperature). |
| `imu_quat`, `imu_gyro`, `imu_gravity` | **`IMPLEMENTED-UNEXERCISED`** | MuJoCo pelvis sensor readings. |
| `error_flags` | **`IMPLEMENTED-UNEXERCISED`** | Safety fault bitfield. |
| `active_alerts`, `alert_count` | **`IMPLEMENTED-UNEXERCISED`** | Firmware alert catalog instances. |
| `fw_age_ms` | **`IMPLEMENTED-UNEXERCISED`** | `0 ms` (simulated edge is co-located with simulated firmware). |
| `last_video_timestamp_us` | **`NOT IMPLEMENTED`** | Constant `0` (video streaming out of scope for Step 3). |
| `last_audio_timestamp_us` | **`NOT IMPLEMENTED`** | Constant `0` (audio streaming out of scope for Step 3). |

---

## 4. Edge Events & System Services (`menlo.edge.EdgeEvent`)

| Event Type | Classification | Description & Provenance |
| :--- | :--- | :--- |
| `diagnostics` | **`IMPLEMENTED-UNEXERCISED`** | Emitted every ~1.0s. Reports `controller`, `cloud_connected=True`, `provisioned=True`, and synthetic CAN bus RX/TX counters. |
| `controller` | **`IMPLEMENTED-UNEXERCISED`** | Emitted when controller ownership switches in the arbiter (`set_active_controller`). |
| `error` | **`IMPLEMENTED-UNEXERCISED`** | Emitted for fatal faults or invalid protocol states using official error codes. |

---

## 5. Physical Simulation & Dynamics

| Subsystem | Classification | Description & Provenance |
| :--- | :--- | :--- |
| Physics Engine | **`APPROXIMATE`** | MuJoCo 3.14.0 at 200 Hz ($dt = 0.005\text{ s}$), `implicitfast` integrator, matching official training sim environment (`isaac_asimov` and `asimov-mjlab`), not physical hardware. |
| Motor Model L0 (PD Dynamics) | **`APPROXIMATE`** | $\tau = K_p (q_\text{des} - q) + K_d (\dot{q}_\text{des} - \dot{q})$, clamped to actuator peak torque limits from `asimov_1.urdf`. Training sim equivalent, not physical hardware windings. |
| Motor Model L1 (Torque Derating) | **`APPROXIMATE`** | Back-EMF speed-dependent torque envelope $\tau_\text{max}(\dot{q}) = \tau_\text{peak} \max(0, 1 - |\dot{q}|/\dot{q}_\text{no\_load})$. |
| Ankle Kinematic Coupling | **`REAL-EQUIVALENT`** | Pushrod kinematic geometry: $A = 2.02 \cdot \theta_\text{pitch} - 0.8 \cdot \theta_\text{roll}$, $B = -2.02 \cdot \theta_\text{pitch} - 0.8 \cdot \theta_\text{roll}$. Virtual work torque conservation: $\tau_\text{pitch} = 2.02(\tau_A - \tau_B)$, $\tau_\text{roll} = -0.8(\tau_A + \tau_B)$. Verified from firmware `policy_thread.c`. |
| Virtual Gantry | **`APPROXIMATE`** | MuJoCo `<equality><weld>` constraint fixing the pelvis link in translation and rotation at settled standing height ($z = 0.60962\text{ m}$). Active at boot; auto-released 1.0s after STAND ramp finishes; controllable via localhost HTTP API `:8852`. |
| Fall Detection | **`APPROXIMATE/ASSUMED`** | Trips when projected gravity $g_z > -0.50$ (tilt $> 60^\circ$). Documented in SDK comments as the firmware trip threshold, but classified ASSUMED without firmware C source. Latches `FAULT_DAMP` mode and `error_flags = 0x101`; clears only via virtual restart. |
| Thermal Safety | **`REAL-EQUIVALENT`** | Over-temperature trip at $\ge 80.0\text{ }^\circ\text{C}$ sets `FAULT_DAMP` (refusing `STAND`). Self-clears with hysteresis at $< 70.0\text{ }^\circ\text{C}$ returning to `DAMP` without requiring firmware restart. |

---

## 6. Generated Master Specification Tables

*Generated automatically via `tools/gen_docs_tables.py` directly from `config/joints.yaml` and wire protobuf definitions.*

### Unified Wire Protocol Enum Mapping

| State / Intent | `asimov.io.ControlMode` (Wire / Firmware) | `edge_cloud.Mode` (Command) | `edge_cloud.FirmwareMode` (Telemetry) | Official Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`STAND`** | **`1`** (`CONTROL_MODE_STAND`) | **`0`** (`MODE_STAND`) | **`1`** (`FW_MODE_STAND`) | Robot holds or ramps into upright standing pose. |
| **`DAMP`** | **`0`** (`CONTROL_MODE_DAMP`) | **`1`** (`MODE_DAMP`) | **`0`** (`FW_MODE_DAMP`) | Actuators compliant ($K_p=0$). |
| **`MOVE`** | **`2`** (`CONTROL_MODE_MOVE`) | *N/A (via policy/trajectory)* | **`2`** (`FW_MODE_MOVE`) | Active motion control (trajectory streaming or policy). |
| **`FAULT_DAMP`** | **`5`** (`CONTROL_MODE_FAULT_DAMP`) | *N/A (latched fault)* | *Reports 0 (DAMP)* | Latched safety fault (fall/overtemp); STAND refused. |

### 25-Joint Actuator Specifications & PD Gains

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

