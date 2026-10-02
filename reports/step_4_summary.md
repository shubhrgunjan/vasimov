# Virtual Asimov 1 (vAsimov) — Step 4 Output & Completion Record

**Project:** Virtual Asimov 1 in MuJoCo (`vasimov`)  
**Commit Reference:** `f50e01e` (`feat(vasimov): Virtual Asimov 1 in MuJoCo (Steps 1-4 Complete)`)  
**Overall Progress:** Step 4 of 6 Complete  
**Automated Tests:** 49/49 unit and integration tests passing (`tests/test_*.py`)  

---

## 1. Executive Summary: What Was Built & Verified in Step 4

Step 4 ("Observability, Fixes & Control") focused on establishing full simulation observability, browser-based control, deterministic replay verification, virtual battery modeling, and locomotion policy readiness, connecting directly to the official, unmodified `menlo-sdk` and `menlo` CLI.

### Summary of Completed Sub-Tasks

| Task | Component | Key Implementation & Outcome |
| :--- | :--- | :--- |
| **Task 1** | **Ground-Truth Telemetry Streamer** (`edge/ground_truth.py`) | Sim-only 50 Hz WebSocket server on port 8854 broadcasting ground-truth state (world pose, twist, touch contact forces for both soles, tracking errors, RTF). |
| **Task 2** | **Zero-Dependency Web Dashboard** (`dashboard/index.html`) | Embedded web dashboard on port 8852 with zero CDN dependencies (runs 100% offline). Includes 60 FPS Canvas telemetry charts, live FSM state badges, gantry control, and push force injection. |
| **Task 3** | **Virtual Battery Subsystem** (`edge/core.py`, `tests/test_battery.py`) | Implemented official `RobotState.battery` protobuf (48V nominal, SoC%, linear discharge). Verified against Menlo SDK preflight rules (trips `battery_low` < 20%, clears >= 20%). |
| **Task 4** | **Bit-Exact Record & Replay** (`tools/record_replay.py`) | Step-synchronized command ingestion guaranteeing exact 0.0 tolerance (`max_abs_diff_qpos = 0.0`, `max_abs_diff_qvel = 0.0`) across repeat runs. |
| **Task 5** | **Locomotion Policy Contract Audit** (`reports/policy_contract.md`) | Verified official pretrained ONNX checkpoint `Menlo/asimov1-locomotion-0818` (831 KB, 78-D observation, 23-D action, 50 Hz control rate). |
| **Task 6** | **Official Menlo SDK & CLI Parity** (`reports/cli_run.log`, `reports/edge_contract.md`) | Verified unmodified `menlo status`, `menlo stand`, and `menlo damp` commands over UDP ports 8850/8851. |

---

## 2. In-Depth Subsystem Breakdown

### Task 1: Ground-Truth Telemetry Server (`ws://127.0.0.1:8854`)
- Implemented an asynchronous WebSocket server decoupled from the official 10 Hz wire telemetry.
- Broadcasts JSON state frames at 50 Hz directly from MuJoCo physics:
  - **Base pose:** World coordinates [x, y, z] and orientation quaternion [qw, qx, qy, qz].
  - **Base twist:** Linear velocities [vx, vy, vz] and angular velocities [wx, wy, wz].
  - **Foot Contact Forces:** Left and right sole normal forces (Fz), tangential forces, and contact booleans from MuJoCo touch sensors.
  - **Actuator Diagnostics:** Commanded torques, applied torques, tracking errors (q_des - q_act), and motor derating saturation flags.
  - **Performance Metrics:** Step computation time, Real-Time Factor (RTF), and telemetry jitter.

### Task 2: Zero-Dependency Observability Web Dashboard (`http://127.0.0.1:8852/`)
- Hosted via a standard Python `http.server` embedded inside `edge/sim.py`.
- **Zero External Dependencies:** Built with native HTML5 Canvas drawing routines running at 60 FPS without external JavaScript libraries or CDNs.
- **Interactive Controls:**
  - **Virtual Gantry Toggle:** Real-time lock/release of the MuJoCo pelvis weld constraint.
  - **Perturbation Generator:** Apply sagittal (+-Fx) or lateral (+-Fy) force impulses (duration 0.05s to 0.5s, 0 to 200 N).
  - **Battery State Manipulator:** Real-time SoC percentage adjustment to trigger and test SDK preflight responses.
  - **Safety Fault Injection:** Direct triggering of `FALL_DETECTED` (tilt > 60 deg) and `MOTOR_OVERTEMP` (>= 80 deg C).

### Task 3: Virtual Battery Subsystem & Preflight Integration
- **Classification:** `PLACEHOLDER` with verified protocol compliance (see `FIDELITY.md`).
- Encoded in `asimov.io.RobotState.battery` (`asimov_state.proto`):
  - `soc_percent`: 100.0% nominal, configurable.
  - `voltage_v`: 48.0 V.
  - `current_a`: 0.0 A nominal.
  - `max_cell_temp_c`: 25.0 deg C.
  - `protection_flags`: 0.
- **SDK Preflight Verification:**
  - When SoC < 20.0%, `menlo status` and `Robot._preflight()` flag a blocking `battery_low` error.
  - When SoC >= 20.0%, the warning clears and allows arming and standing.

### Task 4: Deterministic Record & Replay System
- Synchronized recording of simulation inputs and trajectory state saved to `reports/raw/run_recording.npz`.
- Replay verified using `tools/record_replay.py`:
  - **Total Steps:** 1,200 steps (6.0 s of physics).
  - **Replay Time:** 0.506 s (RTF = 11.86x).
  - **Max Absolute Difference (qpos):** 0.000000.
  - **Max Absolute Difference (qvel):** 0.000000.
  - **Max Absolute Difference (ctrl):** 0.000000.
  - **Verdict:** Bit-identical reproducibility across independent executions.

### Task 5: Locomotion Policy Contract & Checkpoint Audit
- **Repository:** Hugging Face [`Menlo/asimov1-locomotion-0818`](https://huggingface.co/Menlo/asimov1-locomotion-0818)
- **Model Checkpoint:** `policy.onnx` (831,809 bytes)
- **Control Rate:** 50 Hz (dt = 0.020 s, decimation = 4 from 200 Hz physics)
- **Observation Dimension (78-D):**
  1. `base_ang_vel` (3 floats): Pelvis angular velocity [wx, wy, wz] in body frame (scale 0.25).
  2. `projected_gravity` (3 floats): World gravity [0, 0, -1] rotated into pelvis frame (scale 1.0).
  3. `velocity_command` (3 floats): Commanded [vx, vy, wz] (scale 1.0).
  4. `joint_pos_slot01` (9 floats): Positions relative to default pose for Slot 0 & 1 joints.
  5. `joint_pos_slot23` (8 floats): Positions relative to default pose for Slot 2 & 3 joints.
  6. `joint_pos_slot45` (6 floats): Positions relative to default pose for Slot 4 & 5 joints.
  7. `joint_vel_slot01` (9 floats): Angular velocities for Slot 0 & 1 joints (scale 0.10).
  8. `joint_vel_slot23` (8 floats): Angular velocities for Slot 2 & 3 joints (scale 0.10).
  9. `joint_vel_slot45` (6 floats): Angular velocities for Slot 4 & 5 joints (scale 0.10).
  10. `previous_actions` (23 floats): Action vector from previous control step (t - 1).
- **Action Dimension (23-D):**
  - Normalized setpoints a in [-1, 1]^23.
  - Desired joint position: `q_des = q_default + 0.25 * a`.
  - Drives legs (12 DOFs), waist (1 DOF), and arms (10 DOFs). Neck yaw and neck pitch (joints 23 & 24) are not part of the policy and are held by position stiffness.

### Task 6: Official Menlo SDK & CLI Parity
- **Ports:** Commands on UDP port 8850; state telemetry on UDP port 8851.
- **Transcript from `reports/cli_run.log`:**
  - `menlo status --json`: Reports `robot_mode = DAMP`, `armed = false`, `verdict = NOT READY`.
  - `menlo stand -y`: Transitions to `STAND`, ramps joints, verifies 0.5s upright posture (gz < -0.87), becomes `armed = true`, returns exit code 0.
  - `menlo damp -y`: Releases joint torque, robot collapses under gravity, triggers fall trip (gz > -0.5), latches `FAULT_DAMP` (`error_flags = 0x101`), returns exit code 0.

---

## 3. Telemetry Timing Benchmarks (60-Second Runs)

From `reports/raw/timing_metrics.csv`:

| Mode | Packets | Duration | Mean Rate | Jitter | Min Interval | Max Interval | SDK Observed | RTF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fast** | 601 | 60.05 s | 10.00 Hz | 2.95 ms | 67.73 ms | 128.91 ms | 10.00 Hz | **34.50x** |
| **realtime** | 601 | 60.09 s | 10.00 Hz | 2.37 ms | 89.25 ms | 108.53 ms | 10.00 Hz | **1.00x** |
| **realtime_viewer** | 600 | 60.02 s | 10.00 Hz | 2.24 ms | 95.07 ms | 104.97 ms | 10.00 Hz | **1.00x** |

---

## 4. Push Disturbance Sweep (L0 PD vs L1 Derating)

From `reports/raw/push_sweep.csv`:

| Direction | Force | Status | Peak Disp | Peak Tilt | Return Time | Settle Time | Max Torque | L1 Bound Hit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sagittal (+x)** | 40 N | SURVIVED | 1.15 cm | 3.90 deg | 0.825 s | 0.910 s | 16.00 N·m | No |
| **Sagittal (+x)** | 80 N | SURVIVED | 3.30 cm | 6.38 deg | 1.010 s | 2.895 s | 16.00 N·m | No |
| **Sagittal (+x)** | 120 N | FELL | 93.95 cm | 200.12 deg | N/A | N/A | 30.00 N·m | Yes |
| **Sagittal (+x)** | 160 N | FELL | 93.83 cm | 197.63 deg | N/A | N/A | 30.00 N·m | Yes |
| **Lateral (+y)** | 40 N | SURVIVED | 1.26 cm | 2.04 deg | 0.005 s | 0.105 s | 16.00 N·m | No |
| **Lateral (+y)** | 80 N | SURVIVED | 3.04 cm | 2.34 deg | 0.005 s | 0.445 s | 16.00 N·m | No |
| **Lateral (+y)** | 120 N | SURVIVED | 6.06 cm | 2.89 deg | 0.005 s | 1.600 s | 16.91 N·m | No |
| **Lateral (+y)** | 160 N | FELL | 95.85 cm | 195.64 deg | 0.005 s | N/A | 39.97 N·m | Yes |

---

## 5. Next Steps: Roadmap to Step 5 & Step 6

With Step 4 complete, the project moves to:
- **Step 5: Locomotion Policy Runner**
  - Integrate ONNX Runtime to execute `policy.onnx` locally inside the edge control loop.
  - Assemble the 78-D observation vector from MuJoCo sensors at 50 Hz.
  - Map policy 23-D action setpoint offsets to the 25-DoF joint adapter.
  - Execute `menlo balance` and `menlo walk --vx 0.2` without falling.
- **Step 6: Final Integration & Cloud/LiveKit Stubs**
  - Extended LiveKit/WebRTC streaming stubs and packaging.
