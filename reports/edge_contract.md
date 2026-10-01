# Asimov Edge Client Contract & Wire Protocol Report

**Document Status:** VERIFIED against official `menlo-sdk` source and `asimov-protocol` bindings.  
**SDK Version:** `menlo` v0.1.0 (`menlo.asimov`)  
**Protocol Version:** `robots.PROTOCOL_VERSION = 1` (`asimov.io` v1)

---

## 1. Supported Transports & Local No-Cloud Connectivity (Question a)

### (i) Transports Supported by the SDK
The official `menlo-sdk` supports three distinct connection modes (`menlo.asimov.connection.ConnectMode`, lines 42–47):
1. **`"udp"`:**
   - **Source:** [`menlo/asimov/transport/udp.py`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/udp.py#L1-L21)
   - **Wire:** 
     - Commands $\rightarrow$ UDP `<host>:8850` (`COMMAND_PORT = 8850`).
     - State $\leftarrow$ UDP `:<bind_port>` (`STATE_PORT = 8851`), pushed from Edge at the firmware telemetry rate (10 Hz).
   - **Capabilities:** `frozenset({"drive", "state"})` (no camera or audio).
2. **`"hybrid"`:**
   - **Source:** [`menlo/asimov/transport/livekit.py`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/livekit.py#L7)
   - **Wire:** Commands over UDP :8850, state over UDP :8851, camera and audio over LiveKit WebRTC.
3. **`"livekit"`:**
   - **Source:** [`menlo/asimov/transport/livekit.py`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/livekit.py#L8-L15)
   - **Wire:** Reliable data packet on topic `"commands"`; state frames on data track `"state"`; video/audio tracks in the room.

### (ii) Local, No-Cloud Connection Feasibility
**VERIFIED:** A completely local, no-cloud connection is **100% supported** out-of-the-box by the official, unmodified `menlo-sdk` using the **`udp` transport mode**.
- **Citation:** [`menlo/asimov/connection.py`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/connection.py#L62-L66) & [`UdpConfig`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/connection.py#L70-L82):
  ```python
  cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1"))
  robot = Robot(cfg)
  robot.connect("udp")
  ```
- **Authentication:** None. Datagrams are unauthenticated ([`udp.py` line 6](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/udp.py#L6)). No API key, token, or cloud manager is required.
- **CLI Support:** Supported via `menlo robots add local --mode udp --udp 127.0.0.1` and `menlo status --mode udp --udp 127.0.0.1` ([`menlo/cli/_robots.py` lines 94–105](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/cli/_robots.py#L94-L105)).

---

## 2. Protobuf Messages, Channels & Telemetry Requirements (Question b)

### (i) Wire Protobufs per Transport
- **On the UDP Wire (`udp` & `hybrid` control):**
  - **Commands:** Serialized bare `asimov.io.RobotCommand` per datagram to port `8850` ([`menlo/asimov/transport/_wire.py` lines 44–79](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/_wire.py#L44-L79)).
    - Fields: `protocol_version=1`, `sequence`, `timestamp_us`, `mode` (`CONTROL_MODE_STAND=1`, `CONTROL_MODE_DAMP=0`, `CONTROL_MODE_MOVE=2`), `command_control` (`COMMAND_CONTROL_POLICY=1`, `COMMAND_CONTROL_TRAJECTORY=2`), `policy` (`vx, vy, vyaw`), `all_trajectory` (`positions`, `kp`, `kd`).
  - **State / Telemetry:** Serialized bare `asimov.io.RobotState` per datagram to port `8851` ([`menlo/asimov/transport/_wire.py` lines 81–125](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/_wire.py#L81-L125)).
    - Fields: `protocol_version=1`, `sequence`, `timestamp_us`, `current_mode` (`CONTROL_MODE_DAMP=0`, `STAND=1`, `MOVE=2`, `FAULT_DAMP=5`), `joint_pos[25]`, `joint_vel[25]`, `joint_current[25]`, `joint_temp[25]`, `base_ang_vel[3]`, `projected_gravity[3]`, `base_quat[4]`, `error_flags`, `active_alerts`, `battery` (optional).
- **On the Edge-Cloud WebRTC Data Channels (LiveKit):**
  - LiveKit data channel uses the exact same bare `RobotCommand` and `RobotState` payloads on topic `"commands"` and track `"state"` ([`menlo/asimov/transport/livekit.py` lines 10–16](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/livekit.py#L10-L16)).
  - The `edge_cloud.proto` messages (`menlo.edge.CloudCommand`, `menlo.edge.EdgeTelemetry`, `menlo.edge.EdgeEvent`) represent the Edge-to-Cloud backend signaling schema ([`asimov_protocol/v1/edge_cloud_pb2.py`](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/asimov_protocol/v1/edge_cloud_pb2.py)).

### (ii) What the SDK Requires from Telemetry
1. **Handshake Criteria (`Robot._handshake`, lines 496–525):**
   - `protocol_version == 1` (matches `robots.PROTOCOL_VERSION`; skew throws `ProtocolMismatchError`).
   - `len(joint_pos) == 25` (must match `len(ASIMOV_1_BIPED_JOINTS)`).
2. **Freshness & Age (`_preflight.py` lines 58, 231–238, `robot.py` line 1434):**
   - `MAX_STATE_AGE_S = 0.5 s`: States older than 0.5 s trigger `stale_state` blocking error in preflight.
   - `link_timeout = 2.0 s`: 2.0 s without a fresh state packet triggers `LinkLostError`.
3. **Sequence & Ordering (`Robot._classify_sample`, lines 1646–1681):**
   - Unstamped or monotonically increasing sequence accepted as `"newer"`.
   - Repeated sequence + identical timestamp dropped as `"duplicate"`.
   - Reordered past packets dropped as `"stale"`.
   - Sequence jump backwards > `STATE_REORDER_WINDOW` (128) or timestamp jump back > 1.0 s classified as `"restart"` (causes hold reset and fence).
4. **Preflight OK Requirements (`_preflight.py` lines 214–294):**
   - `not_connected`: session must be open with active link.
   - `no_state` / `stale_state`: fresh state within 0.5 s.
   - `faulted`: `error_flags == 0` and mode not `FAULT_DAMP` (alert severity != 0).
   - `wrong_mode`:
     - For `stand()`: robot can be in DAMP or STAND; cannot be in MOVE (`wrong_mode`).
     - For `set_velocity()`: robot must be in MOVE; in STAND it raises `wrong_mode`.
     - For `balance()`: can transition from STAND (if armed) to MOVE.
     - For `trajectory()`: robot must be in MOVE or armed STAND.
   - `not_armed`: in STAND, projected gravity $z \le -0.87$ held continuously for $\ge 0.5\text{ s}$ (`ARM_HOLD_S`).
   - `joint_hot`: all reported joint temperatures must be $< 60^\circ\text{C}$ (`JOINT_HOT_C = 60.0`).
   - `battery_low`: state of charge $\ge 20\%$ (`BATTERY_LOW_PERCENT = 20.0`).

---

## 3. Client-Side Watchdogs, Keepalives & Error Handling (Question c)

### (i) Keepalives & Watchdogs
- **SDK Keepalive Loop (`Robot._keepalive_loop`, lines 1421–1468):**
  - Runs at `KEEPALIVE_HZ = 10.0 Hz` (period 100 ms).
  - Actively re-sends latched velocity commands (`set_velocity(hold=True)`).
  - Checks state stream liveness against `link_timeout` (default 2.0 s).
- **Edge-Side Velocity Timeout:**
  - If Edge receives no nonzero velocity commands for 2.0 s, Edge zeroes velocity and holds position in MOVE ([`menlo/asimov/robot.py` lines 47–48](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/robot.py#L47-L48)).
- **Edge-Side Trajectory Watchdog:**
  - In `MOVE/TRAJECTORY`, if 2.0 s pass without a trajectory setpoint packet, Edge automatically triggers **auto-DAMP** ([`menlo/asimov/robot.py` lines 68–69](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/robot.py#L68-L69)).

### (ii) Error Handling & Unknown Codes
- **`EdgeError` & Outcome Handling:**
  - On UDP transport, `UdpTransport.subscribe_outcome` is a no-op; all command outcomes return `Unknown` ([`udp.py` lines 144–145](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/udp.py#L144-L145)).
  - On LiveKit / Cloud, if an `EdgeError` is received, it maps to `Refused` outcome.
- **Alert Codes (`asimov_protocol/alerts.py` & `_state.py` lines 75–112):**
  - Unknown alert IDs are gracefully mapped to `"ALERT_<id>"` ([`_state.py` line 70](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/_state.py#L70)).
  - Severity 0 alerts mark `critical=True` and trip `faulted=True` in SDK state.

---

## 4. Controller Arbiter Contract (Question d)

### (i) Priority Hierarchy
Documented in [`menlo/asimov/transport/udp.py` lines 3–6](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/transport/udp.py#L3-L6) and [`menlo/asimov/robot.py` lines 999–1001](file:///Users/admin/Projects/Asimov-Reverse-Engeneering/asminov-test/.venv/lib/python3.12/site-packages/menlo/asimov/robot.py#L999-L1001):
1. **E-Stop / Fault Protection (Highest)**: Firmware fault latching (fall, overtemp).
2. **Manager Cockpit / Web UI**: High priority override.
3. **Paired Gamepad / BLE**: Manual operator control.
4. **External API / UDP Control (SDK)**: Lowest priority.

### (ii) Behavior & SDK Expectations
- Commands from an inactive or outranked controller are dropped or rejected (`EDGE_ROBOT_COMMAND_RESULT_REJECTED_CONTROLLER_INACTIVE`).
- When controller ownership transfers, an `EdgeEvent` / `ControllerEvent` (`previous`, `current`, `reason`) is dispatched, calling `robot.on_controller_change`.
- On UDP, commands from the SDK directly feed the Edge arbiter; if another controller is active, mode changes and setpoints are ignored without crashing the SDK.

---

## 5. Official Poses, Gains & Safety Thresholds (Question e)

### (i) Standing Pose & Gait Configuration
- **Official Init State (VERIFIED):**
  - Source: `isaac_asimov/source/cyclotron/cyclotron/assets/robots/asimov_1.py`, lines 167–190 (`ASIMOV_1_STANDING_INIT_STATE`).
  - Pelvis initial height: `root_z = 0.639 m`.
  - Pelvis settled height (VERIFIED Task 0.5): **`0.60962 m`** (drop = 29.38 mm).
  - Joint positions: Left Hip Pitch -0.15 rad, Right Hip Pitch +0.15 rad, Left Knee +0.45 rad, Right Knee -0.45 rad, Left Ankle Pitch -0.30 rad, Right Ankle Pitch +0.30 rad, Left Shoulder Pitch -0.25 rad, Right Shoulder Pitch +0.25 rad, Left Elbow +0.40 rad, Right Elbow -0.40 rad. (All rolls/yaws 0.0 rad).
- **Ankle Coupling Kinematics (VERIFIED):**
  - Source: `asimov-firmware` `policy_thread.c` cited in `menlo/asimov/robots.py` lines 41–55:
    - $K_\text{pitch} = 2.02$, $K_\text{roll} = 0.8$.
    - Motor positions from joint: $A = K_p \theta_p - K_r \theta_r$, $B = -K_p \theta_p - K_r \theta_r$.
    - Joint from motor positions: $\theta_p = \frac{A - B}{2 K_p}$, $\theta_r = -\frac{A + B}{2 K_r}$.
    - Ankle limits: pitch $\pm 0.35\text{ rad}$, roll $\pm 0.10\text{ rad}$.

### (ii) Default Gains per Mode
- **Ranges (VERIFIED):** `docs.menlo.ai/asimov/1/program/api/robot-control#pd-gains`:
  - $K_p \in [0, 500]\text{ N}\cdot\text{m/rad}$, $K_d \in [0, 5.0]\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
  - Typical hardware: $K_p \in [40, 150]\text{ N}\cdot\text{m/rad}$, $K_d \in [2.0, 5.0]\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
- **DAMP Mode:**
  - $K_p = 0.0\text{ N}\cdot\text{m/rad}$ for all 25 joints.
  - $K_d = 2.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$ (ASSUMED damping constant to prevent rapid joint snapping).
- **STAND & TRAJECTORY Mode:**
  - Hardware training sim gains (`vasimov/config/gains.yaml`):
    - Legs (Hips, Knees, Ankles): $K_p = 250.0\text{ N}\cdot\text{m/rad}$, $K_d = 5.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
    - Torso (Waist): $K_p = 100.0\text{ N}\cdot\text{m/rad}$, $K_d = 4.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
    - Arms (Shoulders, Elbows, Wrists): $K_p = 80.0\text{ N}\cdot\text{m/rad}$, $K_d = 3.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$.
    - Neck (Yaw, Pitch): $K_p = 40.0\text{ N}\cdot\text{m/rad}$, $K_d = 2.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$ (ASSUMED).

### (iii) Unified Wire Protocol Enum Mapping (Single Source of Truth)

| State / Intent | `asimov.io.ControlMode` (Wire / Firmware) | `edge_cloud.Mode` (Command) | `edge_cloud.FirmwareMode` (Telemetry) | Official Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`STAND`** | **`1`** (`CONTROL_MODE_STAND`) | **`0`** (`MODE_STAND`) | **`1`** (`FW_MODE_STAND`) | Robot holds or ramps into upright standing pose. |
| **`DAMP`** | **`0`** (`CONTROL_MODE_DAMP`) | **`1`** (`MODE_DAMP`) | **`0`** (`FW_MODE_DAMP`) | Actuators compliant ($K_p=0$). |
| **`MOVE`** | **`2`** (`CONTROL_MODE_MOVE`) | *N/A (via policy/trajectory)* | **`2`** (`FW_MODE_MOVE`) | Active motion control (trajectory streaming or policy). |
| **`FAULT_DAMP`** | **`5`** (`CONTROL_MODE_FAULT_DAMP`) | *N/A (latched fault)* | *Reports 0 (DAMP)* | Latched safety fault (fall/overtemp); STAND refused. |

### (iii) Safety & Fall Thresholds (VERIFIED)
- **Upright / Arming:**
  - Arming threshold: Projected gravity $g_z < -0.87$ (tilt $< 29.5^\circ$) held in STAND for $\ge 0.5\text{ s}$ (`ARM_HOLD_S = 0.5 s`, `_preflight.py` lines 63–66).
  - General upright check: $g_z < -0.80$ (tilt $< 36.8^\circ$, `_state.py` lines 207–213).
- **Fall Detection (Trip Threshold):**
  - Trip condition: Projected gravity $g_z > -0.50$ (tilt $> 60.0^\circ$, `_state.py` line 210).
  - Latches `error_flags` bit 0 (latched fault) and bit 8 (`1 << (1 + 7)` for alert ID 7: `FALL_DETECTED`), setting `error_flags = 0x101`.
  - Mode switches immediately to `FAULT_DAMP` (`5`).
  - **Latch clearance:** Latched permanently until virtual firmware restart (`_preflight.py` lines 248–250).
- **Temperature Thresholds:**
  - Warning: $T \ge 60^\circ\text{C}$ (`MOTOR_TEMP_HIGH`, alert ID 17, `_preflight.py` line 62).
  - Critical trip: $T \ge 80^\circ\text{C}$ (`MOTOR_OVERTEMP`, alert ID 2, latches `FAULT_DAMP`).
  - Clearance hysteresis: Clears when temperature drops below $70^\circ\text{C}$.
