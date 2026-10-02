# Virtual Asimov 1 (vAsimov) — Step 5 Output & Completion Record

**Project:** Virtual Asimov 1 in MuJoCo (`vasimov`)  
**Overall Progress:** Step 5 of 6 Complete (Locomotion Policy Runner)  
**Automated Tests:** 57/57 unit and integration tests passing (`tests/test_*.py`)  
**Primary Checkpoint:** Hugging Face [`Menlo/asimov1-locomotion-0818`](https://huggingface.co/Menlo/asimov1-locomotion-0818) (`policy.onnx`, SHA-256: `47f70691ecd5afda2933b2e6430d2375004c943552284172a2d7b28b84188a0c`)

---

## 1. Executive Summary: What Was Built & Verified in Step 5

Step 5 ("Locomotion Policy Runner") establishes the local execution of the official Asimov 1 neural network locomotion policy in MuJoCo, integrated directly into the `EdgeCore` state machine and compatible with the official, unmodified `menlo-sdk` and `menlo` CLI over UDP wire protocols.

### Key Deliverables & Outcomes

| Task | Component | Key Implementation & Outcome |
| :--- | :--- | :--- |
| **Task 1** | **Policy Contract & Primary Sources** (`reports/policy_contract.md`, `assets/policy/`) | Verified official ONNX model architecture: IR v6, 7-node MLP (`[1, 78]` float32 input $\to$ `[1, 23]` float32 output), 50 Hz control rate ($\Delta t = 0.020\text{ s}$ with decimation 4 from 200 Hz physics). |
| **Task 2** | **Zero-I/O Observation & Action Engine** (`edge/policy_builder.py`, `tests/test_policy_contract.py`) | Pure observation vector construction matching `env.yaml` exact scales: base gyro ($0.25\times$), projected gravity ($1.00\times$), command vel ($1.00\times$), joint pos ($1.00\times$), joint vel ($0.10\times$), and previous actions ($1.00\times$). Action setpoint blending: $q_{\text{des}} = q_{\text{default}} + 0.25 \times a$. |
| **Task 3** | **Integrated Policy Controller** (`edge/policy.py`, `edge/sim.py`, `edge/core.py`) | Thread-safe ONNX Runtime inference embedded in `SimBackend.step()`. Transitions `STAND` $\to$ `MOVE/POLICY` upon `balance()` or velocity command. Auto-releases virtual gantry for unconstrained walking. |
| **Task 4** | **Stepping Dynamics & Matrix Analysis** (`tools/test_stepping_matrix.py`, `reports/raw/task3_stepping_matrix.csv`) | Characterized stationary standing attractor vs. walking initiation: verified stepping behavior across velocities $v_x \in [0.1, 0.4]\text{ m/s}$ and defined minimal threshold criteria. |
| **Task 5** | **Full SDK & CLI Policy Parity** (`tools/test_sdk_policy.py`, `reports/raw/sdk_policy_test.json`) | End-to-end verification using unmodified `menlo.asimov.Robot` and `menlo` CLI: 10s balance (max tilt $3.92^\circ$), 10s walk at $v_x = 0.4\text{ m/s}$ ($2.916\text{ m}$ forward displacement without falling), velocity staleness watchdog (auto-zero hold after 2.0s), and fall protection recovery. |
| **Task 6** | **Fast-Time & Real-Time Performance** | Real-Time Factor (RTF) measured at **7.77x** in headless fast mode and exact **1.00x** in realtime mode, confirming high-headroom execution for real-time edge streaming. |

---

## 2. In-Depth Subsystem Breakdown

### Task 1 & 2: Policy Architecture & Observation Pipeline
- **Observation Contract (78-D):**
  1. `base_ang_vel` (3-D): Gyro angular velocity $[\omega_x, \omega_y, \omega_z]$ in pelvis frame, scaled by $0.25$.
  2. `projected_gravity` (3-D): World gravity $[0, 0, -1]$ rotated into pelvis frame, scaled by $1.00$.
  3. `command` (3-D): Commanded velocities $[v_x, v_y, \omega_z]$ clipped to $[-0.6, 0.8]\text{ m/s}$ ($v_x$), $[-0.5, 0.5]\text{ m/s}$ ($v_y$), $[-0.8, 0.8]\text{ rad/s}$ ($\omega_z$).
  4. `joint_pos` (23-D across slots 01, 23, 45): Deviation from default pose $(q - q_{\text{default}})$, scaled by $1.00$.
  5. `joint_vel` (23-D across slots 01, 23, 45): Joint angular velocities $\dot{q}$, scaled by $0.10$.
  6. `previous_actions` (23-D): Prior step action setpoint offsets in $[-1.0, 1.0]$, scaled by $1.00$.
- **Action Mapping (23-D $\to$ 25 Sim Actuators):**
  - Desired joint position: $q_{\text{des}} = q_{\text{default}} + 0.25 \times a$.
  - Legs (12 DOFs) + Torso (1 DOF) + Arms (10 DOFs).
  - Neck joints (23: `Neck_Yaw`, 24: `Neck_Pitch`) are held at nominal zero with $K_p = 40.0, K_d = 2.0$.

### Task 3: Edge State Machine Integration
- **Transitions:**
  - `STAND` $\to$ `MOVE/POLICY`: Triggered by `robot.balance()` (zero velocity) or `robot.set_velocity(vx, vy, vyaw)`.
  - Gantry Management: Virtual pelvis gantry is automatically released upon entering `MOVE/POLICY`, enabling free bipedal locomotion. Manual toggles via HTTP API `:8852` are respected with thread-safe `RLock` re-anchoring.
  - Staleness Watchdog: If no velocity command is received for $> 2.0\text{ s}$ during `MOVE/POLICY`, edge automatically zeroes target velocity $[0, 0, 0]$ to hold stationary balance without falling.

### Task 4 & 5: SDK & CLI Locomotion Validation (`sdk_policy_test.json`)

From 10-second continuous evaluation runs:
* **Balance Mode (10.0s):**
  - Mean Pelvis Height: $0.6178\text{ m}$ (nominal upright: $0.6096\text{ m}$ to $0.6390\text{ m}$).
  - Maximum Body Tilt: $3.92^\circ$ (well within $60.0^\circ$ fall threshold).
  - Verdict: **SURVIVED (100% stable)**.
* **Walking Mode ($v_x = 0.4\text{ m/s}$, 10.0s):**
  - Forward Displacement: **$2.9162\text{ m}$**.
  - Mean Actual Velocity: $0.3234\text{ m/s}$ (forward tracking error: $0.0766\text{ m/s}$).
  - Body Tilt: $4.65^\circ \pm 0.15^\circ$.
  - Sole Contact Alternation: Periodic bilateral foot contact confirmed.
  - Verdict: **SURVIVED (Continuous forward walking)**.
* **Official CLI Commands:**
  - `menlo balance -y`: Exit code 0, transitions robot to `MOVE`, armed, balancing in place.
  - `menlo walk -y --vx 0.4 --duration 3.0`: Exit code 0, walks forward $0.808\text{ m}$ over 3 seconds and settles back into balance.

---

## 3. Test Suite Verification (57/57 Tests Passing)

Full test discovery results (`.venv/bin/python -m unittest discover tests/ -v`):
```
Ran 57 tests in 20.488s
OK
```
- `tests/test_adapter.py`: 8 tests (joint adapter & differential ankle kinematics roundtrips).
- `tests/test_ankle_map.py`: 5 tests (parallel pushrod geometry & virtual work conservation).
- `tests/test_battery.py`: 4 tests (virtual battery state, preflight checks, discharge).
- `tests/test_edge_core.py`: 9 tests (FSM states, transition ramp, drop rules, watchdogs).
- `tests/test_edge_proto_roundtrip.py`: 3 tests (telemetry, diagnostics, fault injection).
- `tests/test_enums.py`: 4 tests (wire protocol numbers & inversion trap verification).
- `tests/test_ground_truth_and_dashboard.py`: 3 tests (50 Hz WebSocket & dashboard endpoints).
- `tests/test_motor_model.py`: 5 tests (L0-L4 motor dynamics & back-EMF envelopes).
- `tests/test_policy_contract.py`: 4 tests (ONNX I/O contracts, synthetic states, action parity).
- `tests/test_push_metrics.py`: 3 tests (perturbation survival & tilt angle definitions).
- `tests/test_sdk_integration.py`: 9 tests (unmodified `Robot` class e2e integration, posture transitions, balance/policy, ankle trajectory, and gantry behavior).

---

## 4. Next Steps: Roadmap to Step 6 (Final Polish & Cloud/LiveKit Stubs)

With Step 5 complete, all core simulation, motor modeling, wire protocols, and locomotion policy controls are verified. The project moves to the final phase:
- **Step 6: LiveKit WebRTC & Cloud Protocol Integration**
  - Implement WebRTC media/data track streaming stubs for remote cloud teleoperation.
  - Clean packaging, documentation finalization, and end-to-end demonstration scripts.
