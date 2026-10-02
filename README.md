# Virtual Asimov 1 (vAsimov) — Full Interactive Prototype in MuJoCo

A fully usable, interactive simulation prototype of the **Asimov 1** humanoid robot (Menlo Research) in **MuJoCo 3.14.0**, running the official Hugging Face ONNX locomotion policy (`Menlo/asimov1-locomotion-0818`) with real-time telemetry, manual joint control, hybrid overrides, procedural environments, and session recording/replay.

---

## 1. What is this?

Virtual Asimov 1 is an open local robotics development and simulation platform that allows you to launch the Asimov 1 humanoid on your PC, personally control it, observe genuine physics-derived telemetry, test locomotion policies, manipulate individual joints, and build custom simulation scenarios without cloud dependencies.

### Core Capabilities:
- **Interactive Console & 3D Viewer**: Live interactive terminal REPL with optional native MuJoCo passive 3D window.
- **Three Control Modes**:
  1. **Policy Mode**: Official locomotion policy inference ($78\,\text{obs} \to 23\,\text{act} \to \text{PD actuators}$).
  2. **Manual Mode**: Direct joint selection, target setting, and incremental adjustment.
  3. **Hybrid Mode**: Policy handles walking while user manually overrides specific joints.
- **Deep Telemetry Engine**: Live base state, simulated IMU, all 25 joints (angles, velocities, torques, limits), foot contacts & forces, and complete 78-to-23 policy I/O pipeline.
- **Environment System**: Presets (`flat`, `friction_low`, `friction_high`, `heavy_robot`, `light_robot`, `obstacle_basic`, `terrain_basic`) with procedural obstacle insertion (`box`, `step`, `wall`, `table`, `cylinder`).
- **Flight Data Recording & Replay**: Session recording with JSON and CSV exports in `reports/raw/`.

---

## 2. Quickstart & Installation (Linux, macOS, Windows)

Virtual Asimov 1 is fully OS-independent and runs natively on **Linux**, **macOS** (Apple Silicon & Intel), and **Windows** (x86_64 & ARM64).

### A. Universal Setup (All Platforms)

If you have Python 3.10+ installed, you can run the universal setup script from any terminal:

```bash
python setup.py
```
*(On systems with multiple Python versions, use `python3 setup.py`)*.

---

### B. Platform-Specific Setup & Launch

#### 🐧 Linux (Ubuntu/Debian, Fedora, Arch)

1. **Environment Setup**:
   ```bash
   bash setup.sh
   # or: python3 setup.py
   ```
2. **Launch Interactive Console (Default CLI)**:
   ```bash
   ./run_sim.sh
   # On headless servers without X11: ./run_sim.sh --no-viewer
   ```
3. **Launch Live Web Robot Dashboard**:
   ```bash
   ./run_dashboard.sh
   # or: ./run_sim.sh --web
   # Browser UI: http://127.0.0.1:8852 (WebSocket: ws://127.0.0.1:8854)
   ```

#### 🍏 macOS (Apple Silicon M1/M2/M3/M4 & Intel)

1. **Environment Setup**:
   ```bash
   bash setup.sh
   # or: python3 setup.py
   ```
2. **Launch Interactive Console**:
   ```bash
   ./run_sim.sh
   # or: .venv/bin/python -m tools.sim_console
   ```
3. **Launch Live Web Robot Dashboard**:
   ```bash
   ./run_dashboard.sh
   # Browser UI: http://127.0.0.1:8852
   ```
*(Note for macOS Native 3D Viewer: If running interactive passive 3D viewer directly from terminal, the setup script automatically generates `.venv/bin/mjpython` for macOS Cocoa main-thread compatibility)*.

#### 🪟 Windows (Command Prompt, PowerShell, WSL2)

1. **Environment Setup**:
   - **Command Prompt (`cmd.exe`)**:
     ```cmd
     setup.bat
     ```
   - **PowerShell**:
     ```powershell
     .\setup.ps1
     ```
   - **Universal Python**:
     ```cmd
     python setup.py
     ```
2. **Launch Interactive Console**:
   ```cmd
   run_sim.bat
   ```
   *(or `.\.venv\Scripts\python -m tools.sim_console`)*.
3. **Launch Live Web Robot Dashboard**:
   ```cmd
   run_dashboard.bat
   ```
   *(or `.\.venv\Scripts\python -m tools.sim_console --web`)*.  
   Open your browser at:
   ```text
   http://127.0.0.1:8852
   ```

---

## 3. Web Dashboard Features & Workspaces

The web dashboard is an engineering workstation interface connected to the single MuJoCo physics source of truth:

- **LIVE Workspace**:
  - Live robot viewport rendering from real MuJoCo camera feeds.
  - **SOLID | X-RAY** toggle displaying internal joint/actuator linkages and contact forces.
  - Base Pose ($X, Y, Z$, Roll, Pitch, Yaw), Velocity ($V_x, V_y, \omega_z$), and active foot contact forces ($F_z$).
  - Primary controls: `STAND`, `WALK`, `STOP`, `RESET`, `PAUSE`, `STEP`.
  - Velocity sliders, interactive D-Pad, and keyboard shortcuts (`W`, `S`, `A`, `D`, `Q`, `E`, `Space`, `P`, `R`).
  - Event-driven dataflow pipeline: `INPUT → OBSERVATION → POLICY → ACTION → TARGET → ACTUATOR → MUJOCO → TELEMETRY`.
- **CAMERAS Workspace**:
  - Real-time grid of all real MuJoCo model cameras (`front_camera`, `side_camera`, `back_camera`, `direct_side_camera`, `direct_behind_camera`) plus interactive `viewer_camera`.
  - Timestamp, sequence number, and simulation time synchronization overlay.
- **MOTORS Workspace (25 Actuators)**:
  - Complete inventory of all 25 actuated robot joints.
  - Filter by group: `All`, `Legs`, `Arms`, `Waist`, `Active`, `Near limit`, `High torque`.
  - Real-time position vs. target mini-trace, torque, velocity, and effort percentage.
  - Manual target adjustment controls: `[-0.05]`, `[-0.01]`, `[+0.01]`, `[+0.05]`.
- **SENSORS Workspace (84 MuJoCo Sensors)**:
  - Runtime enumeration of all 84 sensors from the loaded model.
  - Categorized into `IMU`, `FORCE`, `CONTACT`, `JOINT`, `OTHER` with live vector inspection.
- **POLICY Workspace (78 in → 23 out)**:
  - Complete 78-dimensional observation vector grouped by contract: base gyro, projected gravity, velocity command, joint positions, joint velocities, previous actions.
  - Complete 23-dimensional policy action outputs with $q_{des} = q_{default} + 0.25 \cdot a$ scaling.
- **ENVIRONMENT Workspace**:
  - Switch between presets (`flat`, `friction_low`, `friction_high`, `heavy_robot`, `light_robot`, `obstacle_basic`, `terrain_basic`).
  - Disturbance injection: pelvis push ($F_x, F_y, F_z$).
- **RECORD & REPLAY Workspace**:
  - Flight data session recording with live timer (`REC mm:ss`).
  - Saved session browser and offline replay player.
- **RAW Workspace**:
  - Syntax-colored JSON inspection of the canonical `TelemetryFrame`.
  - Copy and download raw telemetry snapshots.

---

## 4. How to Control the Robot (Console & Shortcuts)

When the console starts, you are presented with the interactive prompt:

```text
asimov>
```

### Locomotion (Policy Mode)
```text
stand                   # Arm and ramp robot to settled standing pose
walk 0.4                # Walk forward at 0.4 m/s (ranges: vx [-0.6, 0.8], vy [-0.5, 0.5], wz [-0.8, 0.8])
walk 0.3 0.1 0.2        # Combined forward, lateral, and rotational velocity
stop                    # Zero commanded velocity (robot balances in place)
```

### Keyboard Shortcuts
You can also steer directly using single-character commands:
- `w` / `s`: Forward / backward velocity ($0.05\,\text{m/s}$ steps)
- `a` / `d`: Lateral velocity ($0.05\,\text{m/s}$ steps)
- `q` / `e`: Yaw angular velocity ($0.10\,\text{rad/s}$ steps)
- `space`: Stop / safe hold
- `p`: Pause / resume simulation
- `r`: Deterministic reset

### Manual Joint Control Mode
```text
mode manual             # Switch to direct joint control
joints                  # Print table of all 25 joints
select 4                # Select joint (index 4 = left_knee_joint)
set 0.60                # Set target in radians (clamped to joint range)
add 0.05                # Increment target by +0.05 rad
sub 0.05                # Decrement target by -0.05 rad
zero                    # Reset to default standing angle
```

### Hybrid Control Mode
```text
mode hybrid             # Policy handles locomotion; manual targets override selected joints
select left_knee_joint
set 0.55
walk 0.3                # Robot walks using ONNX policy while left knee holds 0.55 rad!
```

---

## 5. How to Inspect Telemetry

The simulator exposes genuine physics quantities from MuJoCo (no fake or fabricated data):

```text
state                   # Full status dashboard (pose, velocity, attitude, contacts, gait, limits)
obs                     # 78-D policy observation breakdown (angular vel, gravity, cmd, joint pos/vel)
action                  # 23-D policy action breakdown and q_des = q_default + 0.25*action synthesis
io                      # Complete 78 Obs -> 23 Act pipeline visualization
contacts                # Left & right foot contact booleans, touch sensors, and vertical forces (Fz)
sensors                 # Simulated IMU (body-frame gyro, projected gravity, orientation quat, linear accel)
joints                  # Comprehensive 25-joint status table with position limits and torque saturation
torque                  # Actuator control signals, applied torques, and percentage effort limits
telemetry <minimal|normal|verbose|raw> # Configure verbosity
telemetry rate <Hz>     # Set periodic background output rate
```

---

## 6. How to Load Environments

Easily switch simulation scenarios at runtime:

```text
env list                # List available presets
env load obstacle_basic # Load flat corridor with step blocks and obstacles
env load friction_low   # Load ice-like slippery ground (mu=0.2)
env load heavy_robot    # Load payload-scaled robot mass (+15%)
reset                   # Deterministically reset robot pose and simulation
```

---

## 7. How to Record and Replay

Capture flight sessions for analysis, machine learning, or regression testing:

```text
record start my_walk    # Start flight data recording
walk 0.4
record stop             # Stops recording and writes:
                        # - reports/raw/my_walk.json (complete telemetry frames)
                        # - reports/raw/my_walk.csv (timeseries tabular data)

replay my_walk          # Inspect flight recording metadata and verify duration
```

---

## 8. Control & Data Pipeline Architecture

```text
User command (Web Dashboard / Keyboard / CLI)
     ↓
Web Control Gateway (Lifecycle: Input → Parsed → Validated → Accepted/Rejected → Applied)
     ↓
Command validation & clipping (vx: [-0.6, 0.8], vy: [-0.5, 0.5], wz: [-0.8, 0.8])
     ↓
EdgeCore state machine (DAMP → STAND → MOVE/POLICY)
     ↓
50 Hz Observation builder (78-D: base gyro, gravity, cmd, 23 pos, 23 vel, 23 act history)
     ↓
ONNX inference (Menlo/asimov1-locomotion-0818, CPUExecutionProvider)
     ↓
23 raw actions
     ↓
Target synthesis: q_des = q_default + 0.25 * action
     ↓
Arbitration: Policy / Manual / Hybrid overrides
     ↓
MotorModel PD actuators (200 Hz, tau = Kp*(q_des - q) - Kd*w with speed-torque envelope)
     ↓
MuJoCo physics step (200 Hz, dt=0.005s)
     ↓
Ground truth telemetry extraction (base, IMU, joints, foot contacts & forces)
     ↓
Canonical TelemetryFrame (JSON) & Offscreen Camera Streamer (~25 FPS JPEG)
     ↓
Web Dashboard (WebSocket: ws://127.0.0.1:8854, HTTP: http://127.0.0.1:8852)
```

Detailed file mappings and transformations are documented in [docs/pipeline.md](file:///home/shubhr/Shubhr/Projects/vasimov/docs/pipeline.md).

---

## 9. Presentation Script

A complete 16-step reproducible presentation walkthrough is available in [docs/demo.md](file:///home/shubhr/Shubhr/Projects/vasimov/docs/demo.md).

To run an automated end-to-end acceptance demo:
```bash
.venv/bin/python examples/run_full_acceptance_demo.py
```
*(Verified transcript saved in `reports/raw/step6_acceptance_demo.txt`)*.

---

## 10. Running Tests

Run the complete 77-test verification suite with fault handler:

```bash
.venv/bin/python -X faulthandler -m unittest discover -s tests -v
```

All 77 tests pass cleanly covering:
- Unit & integration tests for web dashboard gateway, REST/WebSocket APIs, offscreen camera rendering, and command lifecycle.
- Unit & integration tests for console parsing, policy I/O, manual/hybrid joint modes, telemetry, deterministic reset, environment presets, and recording/replay roundtrips.
- Existing Step 5 tests: motor model layers, differential ankle mapping, wire protocol enums, zero-I/O EdgeCore state machine, and unmodified `menlo-sdk` integration.

---

## 11. Repository Structure

```text
vasimov/
├── run_sim.sh                 # One-command executable console launcher
├── run_dashboard.sh           # One-command executable live web dashboard launcher
├── web/                       # Web Dashboard Layer (local-first operator workstation)
│   ├── server/
│   │   ├── gateway.py         # HTTP REST/snapshot & WebSocket telemetry server
│   │   ├── canonical_frame.py # Single-source-of-truth canonical TelemetryFrame builder
│   │   ├── camera_streamer.py # Offscreen MuJoCo camera renderer & JPEG encoder
│   │   ├── control_handler.py # Command lifecycle validator & dispatcher
│   │   └── replay_manager.py  # Session recording scanner & replay player
│   └── client/                # Single-page technical operator interface
│       ├── index.html         # Engineering workstation UI
│       ├── css/dashboard.css  # Technical instrumentation styling (dark neutral)
│       └── js/                # Modular workspaces (live, cameras, motors, sensors, policy, env, raw)
├── tools/
│   ├── sim_console.py         # Full interactive CLI console & simulation runner
│   ├── verify_dashboard_e2e.py# End-to-end 31-step dashboard acceptance runner
│   ├── benchmark_timing.py    # Real-time factor (RTF) timing benchmarks
│   ├── compare_pd_actuators.py# Actuator parity tests
│   └── test_stepping_matrix.py# Stepping gait verification
├── edge/
│   ├── core.py                # Zero-I/O Edge state machine, safety monitors, watchdogs
│   ├── sim.py                 # MuJoCo physics backend, gantry, query APIs
│   ├── policy.py              # ONNX Runtime policy runner & velocity clipping
│   ├── policy_builder.py      # Unified 78-D observation and 23-D action builder
│   ├── environment.py         # Environment presets & procedural obstacle generation
│   ├── telemetry.py           # Telemetry formatting engine & flight data recorder
│   ├── ground_truth.py        # Sim-only 50 Hz WebSocket stream server (:8854)
│   └── transports/            # UDP wire protocol transport (:8850, :8851)
├── docs/
│   ├── demo.md                # Presentation demonstration guide
│   └── pipeline.md            # Detailed control & data pipeline specification
├── examples/
│   ├── demo_session.txt       # Reproducible demo command list
│   └── run_full_acceptance_demo.py # 24-step acceptance verification runner
├── model/
│   └── asimov_1_vasimov.xml   # Master MuJoCo robot model (25 actuators, sensors, gantry)
├── assets/
│   └── policy/                # Pinned ONNX locomotion checkpoint and env configs
├── config/
│   ├── gains.yaml             # PD gains, effort limits, and official standing pose
│   ├── joints.yaml            # Master convention table for firmware joints 0–24
│   └── motors.yaml            # Motor model layer toggles and parameters (L0–L4)
├── tests/                     # 77 comprehensive unit and integration tests
├── FIDELITY.md                # Provenance & fidelity audit for every command and field
└── reports/raw/               # Raw benchmarks, flight data recordings, and test logs
```

---

## 12. Extension Hooks for Future Work

The system's modular architecture enables drop-in extensions without rewriting the simulator:
- **Vision-Language-Action (VLM)**: Connect external planners to `console.core.command_velocity()` or trajectory endpoints.
- **Custom Reinforcement Learning Policies**: Swap the ONNX model in `edge/policy.py` or inherit from `PolicyController`.
- **Custom Simulated Sensors**: Add new sensor extractions in `edge/sim.py:get_state()` and format in `edge/telemetry.py`.
- **Procedural Terrains & Scenarios**: Define new scenes in `edge/environment.py:PRESETS`.
- **Hardware Integration**: The UDP wire protocol (`edge/transports/udp_transport.py`) matches the official Menlo firmware wire protocol.
