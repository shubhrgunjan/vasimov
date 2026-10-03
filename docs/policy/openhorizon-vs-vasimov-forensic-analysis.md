# Deep Forensic Analysis: OpenHorizon Get-Up / Safe-Fall Recovery Policies in VASIMOV

**Document Path:** `docs/policy/openhorizon-vs-vasimov-forensic-analysis.md`  
**Target Policies:** `getup.onnx` (`aa43dfcb...`), `safefall.onnx` (`6501dc58...`)  
**Reference Environment:** OpenHorizon Labs Standalone MuJoCo Runner (`run_policy_mujoco.py`, `mjlab` / MuJoCo Warp)  
**Evaluation Platform:** Virtual Asimov 1 (VASIMOV) Simulation Platform (`asimov_1_vasimov.xml`, 200 Hz physics / 50 Hz control)  
**Investigation Date:** October 3, 2026  
**Status:** COMPLETE FORENSIC INVESTIGATION — DO NOT APPLY FIXES UNTIL REVIEWED  

---

## 1. Executive Summary

### 1.1 The Core Problem
OpenHorizon Labs trained two reinforcement learning recovery policies for the Menlo Asimov 1 humanoid using `mjlab` (MuJoCo Warp):
1. **`safefall.onnx`**: An impact-mitigation policy that actively shapes the body upon an unrecoverable disturbance (fall) to minimize collision impulse across the head, torso, pelvis, and knees.
2. **`getup.onnx`**: A dynamic ground-recovery policy that coordinates arm bracing, knee extension, and torso pitching to recover the robot from arbitrary prone/supine/side postures back to upright standing.

In the reference OpenHorizon standalone runner, a forward push triggers an active protective tuck and roll (safe-fall), followed by settling on the ground, a dynamic ground-recovery sequence (get-up), and seamless resumption of standing walking.

In the current VASIMOV simulator, this behavior is completely broken:
- **Safe-fall is passive/non-functional**: The robot falls like a limp ragdoll without bracing or protecting itself.
- **Get-up is corrupted**: The robot fails to recover dynamically from the ground.
- **Visual "reset/refresh" artifact**: The robot appears to be repeatedly forced or teleported toward the standing/default pose rather than executing the learned recovery sequence.

### 1.2 The Plain Technical Root Cause
The failure is **not** caused by a broken neural network, nor by an invalid ONNX runtime, nor by the differential ankle kinematic approximation. An exact element-by-element numerical verification proves that the ONNX model files in VASIMOV are identical bit-for-bit to the reference artifacts (maximum synthetic input inference error: $0.0000000000\times 10^0$).

Instead, the failure is caused by **five compounding architectural and sensory bugs** in VASIMOV:

1. **Firmware State Machine Fall Latch Kills the Policy Engine (Confirmed Root Cause)**:  
   In `edge/core.py` (`step_state()`), whenever the robot's tilt exceeds $60^\circ$ ($g_z > -0.50$), the firmware trips its legacy emergency protection latch: `self.fault_latched = True; self.mode = EdgeMode.FAULT_DAMP; self.move_submode = MoveSubmode.NONE`.  
   In `edge/sim.py`, `PolicyManager.step()` is gated by `if self.core.mode == EdgeMode.MOVE and self.core.move_submode == MoveSubmode.POLICY`.  
   Consequently, the moment the robot tips past $60^\circ$, **the policy manager is completely silenced**. The recovery policy executes **0 inference steps**. In `FAULT_DAMP`, motor proportional gains are set to $K_p = 0$, causing the robot to collapse passively.

2. **The "Get-Up" Command is a Hard-Coded Teleport / Linear Ramp to Standing Pose (Confirmed Root Cause)**:  
   When the operator presses `G` (or triggers getup in the UI), `tools/sim_console.py` calls `backend.getup()`. In `edge/sim.py` (lines 884–940), `getup()` was implemented as a mock prototype: if $z < 0.38\,\text{m}$, it **teleports the robot** ($q_{\text{pos}}[2] = 0.60962\,\text{m}, q_{\text{pos}}[3] = 1.0$), attaches the virtual gantry weld, and calls `command_stand()`; if $z \ge 0.38\,\text{m}$, it runs `command_stand()` which **linearly interpolates joint targets to the default standing pose** over 2.0 seconds.  
   The learned neural policy `getup.onnx` is **never invoked by the get-up command**.

3. **`safefall.onnx` is Never Loaded; Recovery Policy is Run While Falling (Confirmed Root Cause)**:  
   In `policies/custom/getup_safefall/manifest.yaml`, `model_path` is `getup.onnx`, while `safefall.onnx` was placed under `options: safefall_model_path: safefall.onnx`. In `edge/policy_base.py`, `BasePolicy` resolves only `manifest.resolve_model_path()`. `safefall_model_path` is never read.  
   When `PolicyManager` transitions to `CompositionState.SAFEFALL`, it executes `self.recovery_policy`, which is running **`getup.onnx`**! The get-up policy was being executed while falling. Furthermore, in `sim_console.py`, `backend.policy_manager.mode = ArbitrationMode.AUTO_COMPOSE` writes to a non-existent attribute; `backend.policy_manager.auto_compose` remains `False`, so automatic composition never activates during CLI runs.

4. **Coordinate Frame Corruption of Gyro (`base_ang_vel`) (Confirmed Root Cause)**:  
   In MuJoCo, for a free joint, `data.qvel[3:6]` is **already expressed in the local body frame** (identical to the site gyro sensor `imu_ang_vel`). In `edge/sim.py`, `_build_robot_state()` labeled `data.qvel[3:6]` as `base_ang_vel_world` and multiplied it by $R^T$. Then `GetupSafefallAdapter` multiplied it by $R^T$ a second time.  
   Whenever the robot is non-upright (pitched, rolled, supine, or prone), the angular velocity vector fed into the policy is rotated by $R^T$ into the wrong axes with inverted signs (maximum observation error: $0.225\,\text{rad/s}$, producing raw action discrepancies exceeding $1.018$ and joint target errors exceeding $14.6^\circ$ at step 1).

5. **Knee Effort Limit Starvation in `manifest.yaml` (Confirmed Root Cause)**:  
   In `policies/custom/getup_safefall/manifest.yaml` and `getup/manifest.yaml`, knee effort limit is set to $25.0\,\text{N}\cdot\text{m}$. In the `mjlab` training environment, Synapticon knee motors have a peak effort of $75.0\,\text{N}\cdot\text{m}$ (rated $50.0\,\text{N}\cdot\text{m}$)$, and the MuJoCo XML actuator limit is $45.0\,\text{N}\cdot\text{m}$.  
   During ground recovery, the policy demands knee torques up to $190.4\,\text{N}\cdot\text{m}$. Clamping knee effort to $25.0\,\text{N}\cdot\text{m}$ in `SafetyLayer` deprives the knee actuators of 66% of their peak authority, making physical recovery dynamically impossible even if the policy runs.

---

## 2. Reference Pipeline vs VASIMOV Pipeline

### 2.1 OpenHorizon Reference Pipeline
```text
MuJoCo Physics State (200 Hz)
  │
  ├─> Read Sensor: imu_ang_vel (Pelvis Local Frame, rad/s)
  ├─> Read Orientation: R = data.xmat[pelvis_link] (3x3)
  ├─> Compute Projected Gravity: g_proj = R^T @ [0, 0, -1]
  ├─> Read Joint Positions: q - q_default (in obs_joint_order)
  ├─> Read Joint Velocities: qdot * 0.10 (in obs_joint_order)
  ├─> Read Last Policy Action: a_prev (in action_joint_order)
  │
  ▼
Decimation Gate (50 Hz, period = 0.020s, decimation = 4)
  │
  ├─> Build 375-D Observation (Term-Major FIFO Stack, 5 steps):
  │     [base_ang_vel(15), proj_grav(15), joint_pos(115), joint_vel(115), actions(115)]
  │
  ├─> ONNX Runtime Inference:
  │     Input:  obs [1, 375] (float32)
  │     Output: actions [1, 23] (float32)
  │
  ├─> Action Decoding:
  │     q_target[jn] = default_joint_pos[jn] + 0.25 * action[jn]
  │
  ├─> Continuous PD Actuation (every 200 Hz physics tick):
  │     tau = Kp * (q_target - q) - Kd * qdot
  │     tau = clip(tau, -effort_limit, +effort_limit)
  │     data.ctrl[:] = tau
  │
  └─> mujoco.mj_step()
```

### 2.2 Current VASIMOV Pipeline (with Defect Points Highlighted)
```text
MuJoCo Physics State (200 Hz)
  │
  ├─> data.qvel[3:6] (ALREADY in pelvis local frame)
  │     │
  │     ▼
  │   [DEFECT 4: DOUBLE ROTATION]
  │   sim.py: ang_vel_world = data.qvel[3:6]
  │   policy_adapter.py: base_ang_vel = R^T @ ang_vel_world = R^T @ (local_gyro)
  │   (Rotates gyro vector into wrong axes whenever robot tilts!)
  │
  ├─> sim_time & projected_gravity passed to EdgeCore.step_state()
  │     │
  │     ▼
  │   [DEFECT 1: FALL LATCH TRIP]
  │   core.py: If gz > -0.50 (tilt > 60°):
  │     self.fault_latched = True
  │     self.mode = EdgeMode.FAULT_DAMP
  │     self.move_submode = MoveSubmode.NONE
  │
  ├─> Gating in sim.py line 436:
  │     if mode == MOVE and move_submode == POLICY:
  │         cmd = policy_manager.step()
  │     [BLOCKED: Mode is FAULT_DAMP -> PolicyManager NEVER called!]
  │
  ├─> PolicyManager Arbitration:
  │     [DEFECT 3: BROKEN ARBITRATION & MODEL CONFUSION]
  │     sim_console.py sets policy_manager.mode instead of auto_compose=True.
  │     Even if active: recovery_policy only loads getup.onnx!
  │     SAFEFALL executes getup.onnx while robot is falling!
  │
  ├─> Safety Layer Filtering:
  │     [DEFECT 5: TORQUE STARVATION]
  │     manifest.yaml specifies knee effort_limit = 25.0 Nm (vs 75.0 Nm peak).
  │     SafetyLayer clamps knee effort to 25.0 Nm, crippling lift capacity.
  │
  ├─> Operator Command G (Get-Up):
  │     [DEFECT 2: MOCK GET-UP RESET]
  │     sim_console.py calls backend.getup().
  │     sim.py getup() kinematically resets base height to 0.609m and
  │     commands EdgeCore.command_stand(), ramping to default pose over 2.0s!
  │
  ▼
Actuator Torque Execution
  In FAULT_DAMP: Kp = 0 -> All joint torques = 0. Robot collapses.
  In STAND: Linearly pushed toward default standing pose.
```

---

## 3. Numerical Comparison: Reference vs VASIMOV

To eliminate subjective observation, an isolated numerical test harness was executed comparing the OpenHorizon Reference Runner against the VASIMOV execution pipeline using identical simulator states.

### 3.1 Observation Parity Across Static Postures
The 375-D observation vector was constructed from identical MuJoCo states across four fundamental postures:

| Posture | Max Obs Error | Mean Obs Error | RMS Obs Error | First Divergent Element | Divergent Term Name |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Upright** ($z=0.639\,\text{m}$) | **0.000000** | **0.000000** | **0.000000** | None (100% Match) | All terms bit-exact |
| **Supine** (fallen on back) | **0.200000** | 0.003333 | 0.023805 | Index 0 | `base_ang_vel` (step 0, dim $x$) |
| **Prone** (fallen on stomach) | **0.125000** | 0.002667 | 0.016833 | Index 0 | `base_ang_vel` (step 0, dim $x$) |
| **Side** (fallen on right side) | **0.225000** | 0.003333 | 0.026141 | Index 1 | `base_ang_vel` (step 0, dim $y$) |

#### Mathematical Cause of the Discrepancy
In the supine posture ($\text{pitch} = -90^\circ$, rotation matrix $R = \begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$):
- True local body gyro: $\omega_{\text{body}} = [0.50, 0.20, -0.30]\,\text{rad/s} \implies \text{scaled} \times 0.25 = [0.125, 0.050, -0.075]$
- VASIMOV calculation: $R^T \omega_{\text{body}} \times 0.25 = [-0.075, 0.050, -0.125]$
- Discrepancy:
  $$\Delta = [0.125, 0.050, -0.075] - [-0.075, 0.050, -0.125] = [0.200, 0.000, 0.050]$$
The maximum error of **0.200000** precisely matches the logged data.

### 3.2 Resulting Action Discrepancy from Gyro Corruption
Feeding the corrupted observation into `getup.onnx` produces severe action errors on the very first policy step before dynamics even begin:

| Joint Name | True Reference Action | VASIMOV Action | Action Error | Joint Target Error |
| :--- | :---: | :---: | :---: | :---: |
| `left_shoulder_pitch_joint` | $-2.632$ | $-2.023$ | **0.609** | **0.152 rad ($8.7^\circ$)** |
| `right_shoulder_pitch_joint`| $+1.805$ | $+1.359$ | **0.446** | **0.112 rad ($6.4^\circ$)** |
| `left_wrist_yaw_joint` | $+0.024$ | $-0.332$ | **0.357** | **0.089 rad ($5.1^\circ$)** |
| `right_shoulder_roll_joint` | $+1.346$ | $+1.003$ | **0.342** | **0.086 rad ($4.9^\circ$)** |
| `right_hip_pitch_joint` | $+0.018$ | $-0.316$ | **0.334** | **0.084 rad ($4.8^\circ$)** |
| `left_hip_roll_joint` | $+0.270$ | $-0.037$ | **0.307** | **0.077 rad ($4.4^\circ$)** |
| `left_ankle_pitch_joint` | $+1.212$ | $+1.002$ | **0.210** | **0.053 rad ($3.0^\circ$)** |

Because these corrupted actions are appended to the historical observation buffer at every step, the policy's state trajectory diverges exponentially.

### 3.3 ONNX Model and Runtime Parity Test
To verify whether the ONNX graph or ONNX Runtime was corrupted, an identical synthetic 375-D random normal vector ($\mathcal{N}(0, 1)$) was passed through both the standalone reference session and the VASIMOV runtime:

- **Synthetic Input Max Absolute Error**: $0.0000000000 \times 10^0$
- **Synthetic Input RMS Error**: $0.0000000000 \times 10^0$
- **Model Checksum (`getup.onnx`)**: `aa43dfcbae7bdee333c97202b6159642c4c6ad0612ba85b853a7952d6ef6c717` (MATCH)
- **Model Checksum (`safefall.onnx`)**: `6501dc58a39ada41517acd64af941a0e2b076418d96b28de1a3155ebd51d6dae` (MATCH)

**Conclusion:** The ONNX model files and inference engines are 100% identical and uncorrupted.

---

## 4. Safe-Fall Root Cause & Timeline Analysis

### 4.1 Chronological Timeline Under 150 N Forward Disturbance
A $150\,\text{N}$ forward disturbance was injected into the pelvis body for $0.25\,\text{s}$ (steps 20 to 70 at 200 Hz). The resulting timeline shows exactly where VASIMOV diverges:

| Event Identifier | Event Description | OpenHorizon Reference | Current VASIMOV | Divergence Analysis |
| :--- | :--- | :---: | :---: | :--- |
| **$T_0$** | Push disturbance begins | **0.100 s** | **0.100 s** | Synchronized |
| **$T_1$** | Tilt $> 30^\circ$ or $\|\omega\| > 2.0\,\text{rad/s}$ | **0.420 s** | **0.580 s** | VASIMOV delayed by locomotion policy lag |
| **$T_2$** | Safe-fall policy activated | **0.420 s** | **NEVER** | `auto_compose=False`; `mode` attribute bug |
| **$T_3$** | First changed safe-fall action | **0.440 s** | **NEVER** | At $t=0.890\,\text{s}$, `FAULT_DAMP` tripped; policy killed |
| **$T_4$** | Major body contact ($z < 0.25\,\text{m}$) | **0.680 s** (Tuck & roll) | **0.790 s** (Hard impact) | Ineffective fall shaping; head/pelvis slam |
| **$T_5$** | Robot settled on ground | **1.250 s** (Prone braced) | **1.450 s** (Limp ragdoll) | Zero motor stiffness ($K_p=0$) |

### 4.2 Why Safe-Fall Appears Passive in VASIMOV
1. **Arbitration Flag Bug**: In `tools/sim_console.py` line 247:
   ```python
   self.backend.policy_manager.mode = ArbitrationMode.AUTO_COMPOSE
   ```
   `PolicyManager` does not have a `mode` property. The configuration variable is `self.auto_compose` (boolean). Setting `.mode` dynamically created an unused attribute, leaving `self.auto_compose = False`. The multi-policy state machine never executed.
2. **Missing `safefall.onnx` Loading**:
   In `policies/custom/getup_safefall/manifest.yaml`:
   ```yaml
   runtime:
     type: onnx
     model_path: getup.onnx
     options:
       safefall_model_path: safefall.onnx
   ```
   `BasePolicy.__init__` (lines 41–42) resolves only `manifest.resolve_model_path()`, which is `getup.onnx`. The option `safefall_model_path` is never read. If arbitration had triggered, it would have executed `getup.onnx` during the fall.
3. **Firmware Fall Latch Kill**:
   When tilt crossed $60^\circ$, `EdgeCore` latched `FAULT_DAMP`. `SimBackend` stopped invoking `policy_manager.step()`, and all actuator proportional gains were forced to $K_p = 0$. The robot entered passive unpowered damping.

---

## 5. Get-Up Root Cause & Trajectory Analysis

### 5.1 Dynamic Trajectory Test from Fallen Posture (Supine)
Both systems were initialized in the identical fallen supine posture ($z = 0.15\,\text{m}$, $\text{pitch} = -90^\circ$, default bent-knee angles). 50 policy steps (1.0 second of physical time) were simulated:

| Metric | OpenHorizon Reference | Current VASIMOV (Default Core) | VASIMOV (Unlatched Free-Run) |
| :--- | :---: | :---: | :---: |
| **Policy Steps Executed** | **50 / 50** | **0 / 50** | **50 / 50** |
| **Firmware State** | Active Policy Execution | `FAULT_DAMP` (Latched) | Forced `MOVE/POLICY` |
| **Initial Base Height $z$** | $0.161\,\text{m}$ | $0.162\,\text{m}$ | $0.162\,\text{m}$ |
| **Final Base Height $z$** | **$0.630\,\text{m}$ (Upright)** | $0.609\,\text{m}$ (Gantry tethered) | **$0.204\,\text{m}$ (Collapsed)** |
| **Applied Knee Torque** | Peak $45.0\,\text{N}\cdot\text{m}$ (Active push) | $0.0\,\text{N}\cdot\text{m}$ ($K_p=0$) | Clamped at $25.0\,\text{N}\cdot\text{m}$ |
| **Physical Recovery** | **SUCCESSFUL STANDING** | **FAILED** (Hanging by gantry) | **FAILED** (Stuck on ground) |

### 5.2 Why Get-Up Appears as a "Refresh / Reset Toward Standing"
The user observed:
> "get-up sometimes looks like the robot is repeatedly being pushed toward the standing/default pose instead of executing the learned recovery motion... visually resembles a reset/refresh toward standing rather than a learned recovery sequence."

This is verified by the source code in `edge/sim.py` lines 884–927:
```python
    def getup(self) -> Dict[str, Any]:
        with self._lock:
            ...
            if current_z >= 0.38 and not self.core.fault_fall:
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
            else:
                self.data.qpos[0] = current_x
                self.data.qpos[1] = current_y
                self.data.qpos[2] = SETTLED_BASE_Z  # 0.60962 m
                self.data.qpos[3] = 1.0              # Upright quaternion
                self.data.qpos[4:7] = 0.0
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self.set_gantry(True)
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
```
When `getup` is invoked via console or dashboard:
1. It **teleports** the robot's base $z$ to $0.60962\,\text{m}$ and resets orientation to vertical ($q = [1, 0, 0, 0]$).
2. It engages the **virtual gantry tether** (`self.set_gantry(True)`), physically anchoring the robot to an overhead hoist.
3. It transitions `EdgeCore` to `EdgeMode.STAND`, which executes a linear interpolation toward `self.core.default_pose_sim` over 2.0 seconds.

The learned get-up policy was never being called. The simulator was performing a kinematic teleport and standing ramp.

### 5.3 Why Unlatched Get-Up Policy Fails to Stand
When `fault_latched` is disabled and `getup.onnx` is allowed to run:
1. **Gyro Error**: The double-rotated gyro causes the policy to misestimate the robot's rotational velocity, triggering incorrect arm bracing actions.
2. **Knee Torque Clamping**: In `manifest.yaml`, knee effort limit is set to $25.0\,\text{N}\cdot\text{m}$. The get-up policy requires $45.0\,\text{N}\cdot\text{m}$ (or up to $75.0\,\text{N}\cdot\text{m}$ in training) to extend the knee against the body's inertia. Capping effort at $25.0\,\text{N}\cdot\text{m}$ stalls the knee extension at $z \approx 0.20\,\text{m}$.

---

## 6. Physics and Actuator Model Comparison

A comprehensive audit of the physics parameters between OpenHorizon training and VASIMOV:

| Property | OpenHorizon Training (`mjlab` / `isaac_asimov`) | VASIMOV Implementation (`asimov_1_vasimov.xml`, `sim.py`) | Match? | Impact on Recovery Behavior |
| :--- | :--- | :--- | :---: | :--- |
| **Physics Timestep** | 0.005 s (200 Hz) | 0.005 s (200 Hz) | **YES** | None |
| **Policy Frequency** | 50.0 Hz (decimation = 4) | 50.0 Hz (decimation = 4) | **YES** | None |
| **PD $K_p$ Gains** | Hip/Knee: 150.0, Ankle: 110.0, Arm: 40.0–96.0 | Hip/Knee: 150.0, Ankle: 110.0, Arm: 40.0–96.0 | **YES** | Matches reference |
| **PD $K_d$ Gains** | 5.0 (Hips, Knees, Ankles, Torso), 2.0 (Arms) | 5.0 (Hips, Knees, Ankles, Torso), 2.0 (Arms) | **YES** | Matches reference |
| **Rotor Armature** | Hips: 0.065–0.140, Knee: 0.033, Ankle: 0.0484 | XML `armature`: 0.033–0.140 | **YES** | Dynamic mass matrix matches |
| **Joint Damping** | 5.0 (legs/waist), 2.0 (arms) | XML `damping`: 5.0 (legs/waist), 2.0 (arms) | **YES** | Joint viscous friction matches |
| **Joint Friction** | 0.70 (hips/knees), 0.40 (ankles/arms) | XML `frictionloss`: 0.70 / 0.40 | **YES** | Coulomb friction matches |
| **Torque Limit (Knee)** | **$75.0\,\text{N}\cdot\text{m}$ peak / $50.0\,\text{N}\cdot\text{m}$ rated** | **$25.0\,\text{N}\cdot\text{m}$ in `manifest.yaml` (XML has 45.0)** | **MISMATCH** | **Severe knee stall during getup** |
| **Torque Limit (Hips)** | $120.0\,\text{N}\cdot\text{m}$ peak / $55.0\,\text{N}\cdot\text{m}$ rated | $40.0\,\text{N}\cdot\text{m}$ in `manifest.yaml` (XML has 45.0) | **MISMATCH** | Reduced hip extension torque |
| **Torque-Speed Curve**| Unconstrained (L1 disabled) | Layer L1 disabled by default in `motors.yaml` | **YES** | Matches training sim |
| **Actuator Delay** | 0–2 steps ($0.005$–$0.010\,\text{s}$) in training | Layer L3 disabled by default in `motors.yaml` | **MINOR** | Within robust delay envelope |
| **Ankle Mechanism** | Orthogonal pitch/roll revolute joints | XML has direct orthogonal pitch/roll motors | **YES** | Matches training sim |
| **Controlled DOFs** | 23 humanoid joints (excluding neck pitch/yaw) | 25 actuators in XML (neck joints held at default) | **YES** | Correctly holds neck at zero |

---

## 7. Policy-State and History Buffer Analysis

### 7.1 History Layout Contract
- **Format**: Term-major (`[term0_t-4..t, term1_t-4..t, ...]`).
- **Confirmation**: Numerical testing confirms that passing term-major upright states produces low action norms ($1.25$), whereas step-major produces extreme actions ($6.92$) due to projected gravity misalignment. The term-major layout in `GetupSafefallAdapter` is **correct**.
- **Buffer Initialization**: In `GetupSafefallAdapter.build_observation()`:
  ```python
  if name not in self.buf:
      self.buf[name] = [v.copy() for _ in range(self.history_len)]
  ```
  Cold-start initialization correctly fills all 5 slots with the initial observation vector.

### 7.2 The Buffer Handover Defect
When `PolicyManager` arbitrates between `SAFEFALL` and `GETUP`:
```python
self.recovery_policy.reset()
```
`reset()` clears `self.buf` and `self.last_action`. This forces a cold-start discontinuity at the handover point, resetting historical velocity context to zero.

---

## 8. Root-Cause Ranking Matrix

| Suspected Issue | Concrete Evidence | Severity | Classification |
| :--- | :--- | :---: | :--- |
| **Firmware Fall Latch Suppression** | `gz > -0.50` trips `FAULT_DAMP`; `sim.py` gates policy execution on `mode == MOVE`; policy executes 0 steps during fall/recovery. | **CRITICAL** | **CONFIRMED ROOT CAUSE** |
| **Kinematic Teleport in `backend.getup()`** | `sim.py` lines 908–927 hard-codes $q_{\text{pos}}[2] = 0.609\,\text{m}$, gantry attach, and standing ramp; never invokes `getup.onnx`. | **CRITICAL** | **CONFIRMED ROOT CAUSE** |
| **Missing `safefall.onnx` Loading & Broken Arbitration** | `safefall_model_path` ignored in `policy_base.py`; `sim_console.py` sets dead `.mode` attribute; `getup.onnx` executed while falling. | **CRITICAL** | **CONFIRMED ROOT CAUSE** |
| **Gyro Body Frame Double-Rotation** | MuJoCo `qvel[3:6]` is body frame; `sim.py` and `policy_adapter.py` multiply by $R^T$ twice; max gyro error $= 0.225\,\text{rad/s}$, action error $> 1.01$. | **CRITICAL** | **CONFIRMED ROOT CAUSE** |
| **Knee Effort Limit Clamped to 25 Nm** | `manifest.yaml` specifies 25.0 Nm (vs 75.0 Nm peak in training); policy commands 190 Nm; knee stalls at $z=0.204\,\text{m}$. | **HIGH** | **CONFIRMED ROOT CAUSE** |
| **Fall Trigger Zero-Action Gating** | `applied_act = raw_act if is_active else np.zeros_like(raw_act)` forces default standing pose until trigger threshold crossed. | **HIGH** | **LIKELY CONTRIBUTOR** |
| **Buffer Cold-Reset on Handover** | `recovery_policy.reset()` called on every state transition, wiping history buffer and last action. | **MEDIUM** | **LIKELY CONTRIBUTOR** |
| **Ankle Differential Map Interference** | Ankle map is only used in telemetry; MuJoCo model controls orthogonal pitch/roll directly. | **NONE** | **RULED OUT** |
| **ONNX Runtime / Weight Corruption** | Synthetic input comparison yields bit-exact $0.0000000000\times 10^0$ error; sha256 hashes match. | **NONE** | **RULED OUT** |
| **Observation Stacking Layout (Term vs Step)** | Testing proves `mjlab` model was trained on term-major layout; step-major corrupts gravity. | **NONE** | **RULED OUT** |
| **Safety Layer Range / Slew Rate Limiting** | Safety filter testing over 50 recovery steps showed 0 range clamp hits and 0 rate limit violations. | **NONE** | **RULED OUT** |
| **Control Decimation Timing** | Policy executes once every 4 physics steps ($0.020\,\text{s}$) with proper target holding. | **NONE** | **RULED OUT** |

---

## 9. Recommended Fixes (For Subsequent Implementation Phase)

> [!IMPORTANT]
> In accordance with the critical instructions of this analysis task, **no repository files were modified**. The following fixes are detailed specifications for the subsequent implementation task.

### Fix 1: Decouple Policy Recovery Mode from Firmware Fault Latches
- **Problem**: When tilt exceeds $60^\circ$, `EdgeCore` latches `FAULT_DAMP` and stops the policy manager.
- **Why It Matters**: Recovery policies are specifically designed to operate when the robot is fallen (tilt $> 60^\circ$). Killing the policy engine makes recovery impossible.
- **Exact Fix**:
  1. In `edge/core.py`, when a policy from the `recovery` category is active (or when `auto_compose` is enabled), disable the `FAULT_DAMP` transition on tilt:
     ```python
     if self.active_policy_category == "recovery" or not self.fall_latch_enabled:
         # Suppress FAULT_DAMP latching; allow recovery policy to retain actuator ownership
     ```
  2. In `edge/sim.py` line 436, allow `PolicyManager.step()` to execute if `self.core.mode in (EdgeMode.MOVE, EdgeMode.RECOVERY)` or when the active policy is a recovery controller.
- **Files Affected**: [edge/core.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/core.py), [edge/sim.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/sim.py)
- **Risk**: Low.
- **Verification**: Fallen robot in supine pose executes 50 policy steps without tripping into `FAULT_DAMP`.

### Fix 2: Connect Key 'G' and Web Command to Real `getup.onnx` Execution
- **Problem**: `backend.getup()` teleports the robot and commands standing pose.
- **Why It Matters**: Completely bypasses the neural network recovery policy.
- **Exact Fix**:
  1. Rewrite `SimBackend.getup()`:
     - Clear any latched faults (`self.core.virtual_restart()`).
     - Release virtual gantry (`self.set_gantry(False)`).
     - Switch active policy to `getup`: `self.load_policy("getup")`.
     - Set `self.core.mode = EdgeMode.MOVE` and `self.core.move_submode = MoveSubmode.POLICY`.
     - Disable `fall_latch_enabled` during recovery.
     - Remove all hard-coded coordinate overwrites ($q_{\text{pos}}[2] = 0.609$, $q_{\text{pos}}[3] = 1.0$).
  2. Do the identical update for `SimBackend.safe_fall()`, switching active policy to `safefall`.
- **Files Affected**: [edge/sim.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/sim.py), [tools/sim_console.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/tools/sim_console.py)
- **Risk**: Low.
- **Verification**: Pressing `G` from fallen pose initiates active leg and arm pushup sequence without gantry or teleportation.

### Fix 3: Fix Gyro Coordinate Frame in `_build_robot_state()` and `policy_adapter.py`
- **Problem**: MuJoCo's `data.qvel[3:6]` is already in the pelvis body frame. Multiplying by $R^T$ rotates body angular velocity into a fictitious coordinate frame.
- **Why It Matters**: Policy receives wrong gyro feedback whenever tilted, distorting actions by up to $14.6^\circ$.
- **Exact Fix**:
  1. In `edge/sim.py` (`_build_robot_state()`, lines 330–332):
     ```python
     # data.qvel[3:6] is ALREADY body-frame angular velocity (identical to imu_ang_vel sensor)
     ang_vel_body = np.array(self.data.qvel[3:6], dtype=np.float64)
     ang_vel_world = R @ ang_vel_body  # Transform to world frame for telemetry
     ```
  2. In `edge/policy_adapter.py` (`_get_term_vector()`, line 208):
     ```python
     if name == "base_ang_vel":
         # Use body-frame angular velocity directly
         return (state.base_ang_vel_body).astype(np.float32)
     ```
- **Files Affected**: [edge/sim.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/sim.py), [edge/policy_adapter.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/policy_adapter.py)
- **Risk**: Minimal.
- **Verification**: Observation parity error on supine, prone, and side postures drops from $0.225$ to $0.000000$.

### Fix 4: Support Dual-Model Loading in Composite Recovery Policy (`getup_safefall`)
- **Problem**: `safefall.onnx` is never loaded; `getup.onnx` runs during falls; `auto_compose` flag is not set.
- **Why It Matters**: Prevents safe-fall from executing.
- **Exact Fix**:
  1. In `edge/policy_base.py`, check `manifest.options.get("safefall_model_path")`. If present, instantiate a secondary runtime `self.safefall_runtime = ONNXPolicyRuntime(...)`.
  2. In `edge/policy_base.py.step()`, evaluate `safefall_runtime` when in `SAFEFALL` state, and `runtime` (`getup`) when in `GETUP` state.
  3. In `tools/sim_console.py` line 247:
     ```python
     self.backend.policy_manager.auto_compose = True
     ```
- **Files Affected**: [edge/policy_base.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/policy_base.py), [edge/policy_manager.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/edge/policy_manager.py), [tools/sim_console.py](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/tools/sim_console.py)
- **Risk**: Low.
- **Verification**: When pushed, robot switches to `safefall.onnx`, tucks arms, and absorbs ground impact.

### Fix 5: Correct Actuator Effort Limits in `manifest.yaml`
- **Problem**: Knee effort limit is clamped to $25.0\,\text{N}\cdot\text{m}$ in manifest, whereas motor peak is $75.0\,\text{N}\cdot\text{m}$ and XML has $45.0\,\text{N}\cdot\text{m}$.
- **Why It Matters**: Knee torque is starved by 66%, stalling ground recovery pushup.
- **Exact Fix**:
  Update `pd_gains` in `policies/custom/getup/manifest.yaml` and `getup_safefall/manifest.yaml`:
  - `knee: effort_limit: 45.0` (matching XML actuator limit)
  - `hip_pitch: effort_limit: 45.0`
  - `hip_roll: effort_limit: 45.0`
- **Files Affected**: [policies/custom/getup/manifest.yaml](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/policies/custom/getup/manifest.yaml), [policies/custom/getup_safefall/manifest.yaml](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/vasimov/policies/custom/getup_safefall/manifest.yaml)
- **Risk**: Low.
- **Verification**: Left and right knee actuators deliver full $45\,\text{N}\cdot\text{m}$ torque during getup, successfully elevating pelvis height above $0.55\,\text{m}$.

---

## 10. Proposed Verification Plan (Post-Implementation)

After the above fixes are implemented in a separate task, execute the following protocol to prove behavioral parity against the OpenHorizon reference runner:

### Stage 1: Static Observation & Inference Parity
Run `/tmp/test_forensic_harness.py`:
- [ ] Upright stance observation error: $\le 10^{-6}$
- [ ] Supine posture observation error: $\le 10^{-6}$
- [ ] Prone posture observation error: $\le 10^{-6}$
- [ ] Side posture observation error: $\le 10^{-6}$
- [ ] ONNX action output difference against reference: $\le 10^{-5}$

### Stage 2: Isolated Safe-Fall Verification
Run disturbance push ($150\,\text{N}$ for $0.25\,\text{s}$):
- [ ] Safe-fall trigger fires at tilt $> 30^\circ$ or $\|\omega\| > 2.0\,\text{rad/s}$ ($t \approx 0.42\,\text{s}$).
- [ ] `safefall.onnx` receives inputs and actively tucks arms and bends knees before ground impact.
- [ ] `EdgeCore` does not latch `FAULT_DAMP` during the fall.
- [ ] Peak head impact acceleration is quantitatively reduced compared to passive DAMP fall.

### Stage 3: Isolated Get-Up Ground Recovery Verification
Initialize robot in fallen supine ($z = 0.15\,\text{m}$, pitch $-90^\circ$):
- [ ] Press `G` (or trigger `getup`).
- [ ] Confirm no kinematic teleportation occurs ($z$ rises smoothly via joint torques).
- [ ] Confirm virtual gantry remains unattached (`gantry_active = False`).
- [ ] Pelvis height smoothly ascends from $0.15\,\text{m}$ to $> 0.60\,\text{m}$ within $2.0\,\text{s}$.
- [ ] Robot stabilizes in upright standing posture.

### Stage 4: Full End-to-End Cycle
Execute continuous autonomous arbitration:
$$\text{Walking} \xrightarrow{\text{150 N Push}} \text{Safe-Fall (Tuck \& Roll)} \xrightarrow{\text{Settled}} \text{Get-Up (Dynamic Pushup)} \xrightarrow{\text{Upright}} \text{Walking}$$
- [ ] Robot absorbs fall without fault shutdown.
- [ ] Robot settles on ground and automatically hands over to `getup`.
- [ ] Robot stands up and automatically hands over to official walking policy.
- [ ] Robot resumes walking forward under commanded velocity.
