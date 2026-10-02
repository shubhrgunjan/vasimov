# Virtual Asimov 1 — Live Interactive Demonstration Script

This document provides a concise, reproducible sequence for presenting the **Virtual Asimov 1 in MuJoCo** prototype.

---

## 1. Quick Launch

Launch the interactive console with the native MuJoCo passive viewer:

```bash
./run_sim.sh
```

*(On headless systems without a desktop display, add `--no-viewer`)*.

---

## 2. Interactive Presentation Sequence

Follow these 16 steps during a live demonstration:

### Step 1: Inspect Boot State
```text
asimov> state
```
- **Observe**: Robot floating base at settled height $z = 0.6096\,\text{m}$.
- **Observe**: Virtual gantry is `ACTIVE (RIGID HOLD)` to prevent unpowered falls before arming.
- **Observe**: 8 foot contact points established with ground floor.

---

### Step 2: Arm and Stand
```text
asimov> stand
```
- **Observe**: Firmware transitions into `STAND`.
- **Observe**: Robot joints execute a smooth 2.0s ramp from unpowered compliance to upright standing pose.
- **Observe**: After ramp completes, virtual gantry auto-releases for unconstrained free physics.

---

### Step 3: Command Locomotion Policy
```text
asimov> walk 0.4
```
- **Observe Terminal**:
  ```text
  USER INPUT      : walk 0.4
  PARSED COMMAND  : vx=+0.400, vy=+0.000, wz=+0.000
  CONTROL COMMAND : command_vel = [+0.400, +0.000, +0.000] -> 78-D Policy
  SIMULATOR RESULT: mode = MOVE/POLICY, gantry = RELEASED
  ```
- **Observe Viewer**: The biped dynamically strides forward along the corridor.

---

### Step 4: Verify Policy Input/Output Pipeline
```text
asimov> io
```
- Displays the complete mathematical transformation:
  1. **78-D Observation Vector**:
     - Base angular velocity (body frame, $\times 0.25$)
     - Projected gravity vector ($R^T [0, 0, -1]$)
     - Commanded velocity $[v_x, v_y, \omega_z]$
     - 23 Joint positions relative to default standing pose
     - 23 Joint velocities ($\times 0.10$)
     - 23 Previous action history
  2. **ONNX Evaluation**:
     - Evaluates official Hugging Face policy checkpoint (`47f70691...`).
  3. **23-D Joint Target Synthesis**:
     - $q_{\text{des}} = q_{\text{default}} + 0.25 \times a_{\text{raw}}$
     - Evaluates MotorModel PD law to generate joint torques.

---

### Step 5: Foot Contacts & Ground Reaction Forces
```text
asimov> contacts
```
- Shows alternating ground reaction forces ($F_z \approx 140\text{--}300\,\text{N}$) and contact constraint counts.

---

### Step 6: Full 25-Joint Telemetry Inspection
```text
asimov> joints
```
- Shows tabular telemetry for all 25 joints:
  - Positions ($q$), Velocities ($\dot{q}$), Targets ($q_{\text{des}}$)
  - Applied torques ($\tau$), Effort limits, and Limit violation status.

---

### Step 7: Frame-by-Frame Precision Debugging
```text
asimov> pause
asimov> step 1
asimov> step 5
asimov> resume
```
- Pauses physics stepping.
- Advances by exact control ticks ($0.020\,\text{s}$ per tick at 50 Hz).
- Resumes smooth continuous simulation.

---

### Step 8: Hold Position
```text
asimov> stop
```
- Commands zero velocity. Robot settles into upright stationary balance.

---

### Step 9: Manual Joint Manipulation
```text
asimov> mode manual
asimov> select 4
asimov> add 0.10
asimov> sub 0.05
asimov> set 0.50
```
- Direct control over individual joint targets.

---

### Step 10: Hybrid Mode (Policy + Manual Override)
```text
asimov> mode hybrid
asimov> select 4
asimov> set 0.55
asimov> walk 0.3
```
- Locomotion policy controls 22 joints while the selected joint (e.g. `left_knee_joint`) holds the user's manual override!

---

### Step 11: Real-time Disturbance Push
```text
asimov> push 60 x 0.1
```
- Applies a $60\,\text{N}$ horizontal push to the pelvis link for $0.1\,\text{s}$.
- Observe robot bipedal recovery.

---

### Step 12: Load Procedural Obstacle Scene
```text
asimov> env load obstacle_basic
asimov> reset 42
```
- Dynamically injects step blocks into the corridor without manual XML editing.
- Resets simulation to deterministic initial condition.

---

### Step 13: Flight Data Recording
```text
asimov> record start presentation_run
asimov> walk 0.35
asimov> state
asimov> record stop
```
- Records full state trajectory to `reports/raw/presentation_run.json` and `reports/raw/presentation_run.csv`.

---

### Step 14: Flight Data Replay
```text
asimov> replay presentation_run
```
- Displays recording metadata, captured frames, and duration.

---

### Step 15: Single-Key Interactive Keyboard Navigation
- In the console, you can also steer interactively using:
  - `w` / `s`: Forward / backward velocity ($0.05\,\text{m/s}$ steps)
  - `a` / `d`: Lateral velocity ($0.05\,\text{m/s}$ steps)
  - `q` / `e`: Yaw angular velocity ($0.10\,\text{rad/s}$ steps)
  - `space`: Emergency / safe hold
  - `p`: Pause / resume
  - `r`: Reset

---

### Step 16: Clean Exit
```text
asimov> quit
```
- Closes viewer and terminates simulation threads cleanly.
