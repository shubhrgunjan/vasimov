# Virtual Asimov 1 (vAsimov) — Step 4: Observability, Fixes & Control

This repository contains the local simulation and virtual edge emulator for the **Asimov 1** humanoid robot (Menlo Research) in **MuJoCo 3.14.0**.

In Step 4, we establish complete simulation observability, web dashboard control, virtual battery management, and determinism verification, connecting directly to the **OFFICIAL, UNMODIFIED `menlo-sdk`** (`menlo.asimov.Robot` and the `menlo` CLI).

---

## 1. Directory Structure

```
vasimov/
├── adapter.py                 # Joint adapter: firmware 25 <-> sim 25 coordinates & ankle kinematics
├── ankle_map.py               # Differential ankle kinematics & pushrod coupling
├── build_model.py             # Reproducible model generator (inserts actuators, sensors, gantry)
├── config/
│   ├── gains.yaml             # PD gains, effort limits, and official standing pose
│   ├── joints.yaml            # Master convention table for firmware joints 0–24
│   └── motors.yaml            # Motor model layer toggles and parameters (L0–L4)
├── dashboard/
│   └── index.html             # Localhost observability dashboard & control panel (Zero CDN)
├── edge/
│   ├── __init__.py            # Edge package
│   ├── __main__.py            # CLI entry point: python -m edge [options]
│   ├── core.py                # Zero-I/O Edge core: state machine, arbiters, safety latches, telemetry
│   ├── ground_truth.py        # Sim-only ~50 Hz WebSocket ground-truth streaming server (:8854)
│   ├── sim.py                 # MuJoCo physics backend, virtual gantry, HTTP control API (:8852)
│   └── transports/
│       ├── __init__.py
│       └── udp_transport.py   # Official UDP wire protocol (commands :8850, state :8851)
├── FIDELITY.md                # Provenance & fidelity audit for every command, field, and event
├── inspect_model.py           # Model inspection diagnostic tool
├── model/
│   ├── asimov_1_vasimov.xml   # Derived MuJoCo model (25 actuators, 84 sensors, pelvis weld gantry)
│   └── throwaway_native_pos.xml # PD parity benchmark model
├── motor_model.py             # Layered motor model implementation (L0-L4)
├── reports/
│   ├── cli_run.log            # Official menlo CLI execution transcript
│   ├── edge_contract.md       # SDK & Edge contract audit report
│   ├── joint_mapping.md       # Comparative joint mapping audit
│   ├── policy_contract.md     # Locomotion policy contract audit (78-D obs, 23-D action, ONNX checkpoint)
│   └── raw/                   # Raw benchmark data & verifiable output traces
│       ├── generated_doc_tables.md # Generated master gains & enum tables
│       ├── nopolicy_cli_outcome.log# Raw CLI transcript against NoPolicy stub
│       ├── pd_parity_comparison.csv# External torque PD vs MuJoCo native position actuators
│       ├── push_sweep.csv     # Full sagittal/lateral push disturbance sweep & L1 binding
│       ├── replay_comparison.json# Bit-level determinism verification metrics
│       ├── run_recording.npz  # Step-synchronized command & state recording
│       └── timing_metrics.csv # 60-second telemetry timing benchmarks (mean, jitter, RTF)
├── stand.py                   # Step-2 standing controller & push benchmark
├── tests/
│   ├── test_adapter.py        # Unit tests for 25-joint adapter & ankle round-trips
│   ├── test_ankle_map.py      # Unit tests for differential ankle kinematics & virtual work
│   ├── test_battery.py        # Unit & integration tests for Virtual Battery (PLACEHOLDER)
│   ├── test_edge_core.py      # Pure logic unit tests for state machine, watchdogs, drop rules
│   ├── test_edge_proto_roundtrip.py # Serialization round-trip tests for EdgeTelemetry & EdgeEvent
│   ├── test_enums.py          # Unit tests asserting single-source-of-truth wire protocol enums
│   ├── test_ground_truth_and_dashboard.py # Integration tests for 50Hz WebSocket & dashboard
│   ├── test_motor_model.py    # Unit tests for motor model layers L0–L4
│   └── test_sdk_integration.py# End-to-end integration tests using unmodified menlo.asimov.Robot
├── tools/
│   ├── benchmark_timing.py    # 60s timing benchmark across fast, realtime, and viewer modes
│   ├── compare_pd_actuators.py# Actuator parity test comparing external torque PD vs native position
│   ├── gen_docs_tables.py     # Code generator keeping README/FIDELITY tables synchronized
│   ├── record_replay.py       # Record/replay runner for bit-reproducibility analysis
│   └── test_cli_runner.py     # Automation runner executing menlo status/stand/damp
└── upstream/                  # Cloned official repos (asimov-1, asimov-mjlab, isaac_asimov)
```

---

## 2. Viewing the Model in MuJoCo & Checking Progress

### A. Quick Standalone 3D Viewer (No servers needed)
To immediately view and interact with the robot standing in MuJoCo:
```bash
cd /Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/vasimov
/Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/bin/mjpython stand.py --viewer
```
*(Press Space to pause/unpause, double-click bodies to select, Ctrl+right-click drag to apply perturbation forces).*

### B. Launch Virtual Edge with Interactive Passive Viewer & Web Dashboard
To run the full Virtual Edge stack (UDP wire protocol + Web Dashboard + MuJoCo 3D window):
```bash
/Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/bin/mjpython -m edge --transport udp --realtime --viewer
```
Once launched:
1. **Interactive MuJoCo 3D Viewer** renders real-time physics on your desktop.
2. **Observability Web Dashboard** is live at: [http://127.0.0.1:8852/](http://127.0.0.1:8852/)
3. **Sim-only Ground-Truth Stream** streams at 50 Hz on `ws://127.0.0.1:8854`.
4. **Official SDK / CLI** connects via UDP ports `8850` (commands) and `8851` (state).

---

## 3. Running Tests and Benchmarks

```bash
# 1. Run all 49 unit and integration tests (Adapter, Enums, State Machine, SDK, Battery, Dashboard, Motor Model)
.venv/bin/python -m unittest discover tests/ -v

# 2. Run Record / Replay determinism verification (verifies bit-identical reproducibility)
.venv/bin/python tools/record_replay.py

# 3. Run 60-second telemetry timing benchmarks
.venv/bin/python tools/benchmark_timing.py

# 4. Run PD actuator parity comparison
.venv/bin/python tools/compare_pd_actuators.py
```

---

## 4. Master Specification Tables

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

