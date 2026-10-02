"""
vasimov/edge/telemetry.py
Telemetry Engine, State Formatting, and Recording/Replay for Virtual Asimov 1.

Provides:
- Configurable telemetry verbosity (minimal, normal, verbose, raw).
- Configurable broadcast rate (1 Hz, 10 Hz, 50 Hz).
- Deep inspection tables for joints, contacts, torques, simulated IMU, and 78->23 policy I/O.
- Deterministic session recording and CSV/JSON export.
- Replay facility with metadata preservation.
"""

from __future__ import annotations
import csv
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger("vasimov.edge.telemetry")

_REPORTS_RAW_DIR = Path(__file__).resolve().parent.parent / "reports" / "raw"


class TelemetryEngine:
    """Manages telemetry formatting, live terminal dashboards, and flight-data recording."""

    def __init__(
        self,
        verbosity: str = "normal",
        enabled: bool = True,
        rate_hz: float = 2.0,
    ):
        self.verbosity = verbosity.lower()
        self.enabled = enabled
        self.rate_hz = rate_hz
        self.last_print_sim_time: float = -1.0
        self.last_print_wall_time: float = -1.0

        # Recording state
        self.recording_active: bool = False
        self.current_recording_name: Optional[str] = None
        self.recording_buffer: List[Dict[str, Any]] = []
        self.recording_metadata: Dict[str, Any] = {}

        # Gait tracking counters
        self.last_left_contact: bool = False
        self.last_right_contact: bool = False
        self.step_transitions: int = 0
        self.last_step_sim_time: float = 0.0
        self.stride_times: List[float] = []

    def set_verbosity(self, verbosity: str) -> None:
        """Set verbosity level: minimal, normal, verbose, or raw."""
        valid = ("minimal", "normal", "verbose", "raw")
        v = verbosity.lower().strip()
        if v not in valid:
            raise ValueError(f"Invalid verbosity '{verbosity}'. Options: {valid}")
        self.verbosity = v

    def set_rate(self, rate_hz: float) -> None:
        """Set periodic display rate in Hz."""
        if rate_hz <= 0:
            raise ValueError("Rate must be positive Hz")
        self.rate_hz = float(rate_hz)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable live terminal telemetry output."""
        self.enabled = bool(enabled)

    # ── Formatting APIs ──────────────────────────────────────────────────────

    def format_minimal(self, state: Dict[str, Any]) -> str:
        """One-line compact telemetry summary."""
        sim_t = state.get("sim_time", 0.0)
        rtf = state.get("rtf", 1.0)
        mode = f"{state.get('edge_mode', '')}/{state.get('move_submode', '')}"
        ctrl_mode = state.get("control_mode", "policy").upper()
        cmd = state.get("command", {})
        vx, vy, yaw = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("vyaw", 0.0)
        base = state.get("base", {})
        bx, by, bz = base.get("x", 0.0), base.get("y", 0.0), base.get("z", 0.0)
        bvx = base.get("body_vx", 0.0)
        contacts = state.get("contacts", {})
        l_c = "ON" if contacts.get("left_foot", {}).get("contact") else "OFF"
        r_c = "ON" if contacts.get("right_foot", {}).get("contact") else "OFF"
        gantry = "G-ON" if state.get("gantry_active") else "G-OFF"

        return (
            f"[{sim_t:6.2f}s | RTF {rtf:4.2f}] {ctrl_mode}:{mode} {gantry} | "
            f"CMD:({vx:+.2f}, {vy:+.2f}, {yaw:+.2f}) | "
            f"POS:({bx:.2f}, {by:.2f}, {bz:.2f}) | BVX:{bvx:+.2f} | "
            f"FEET:[L:{l_c} R:{r_c}]"
        )

    def format_status_block(self, state: Dict[str, Any]) -> str:
        """Multi-line structured dashboard status block."""
        sim_t = state.get("sim_time", 0.0)
        rtf = state.get("rtf", 1.0)
        mode = f"{state.get('edge_mode', '')}/{state.get('move_submode', '')}"
        ctrl_mode = state.get("control_mode", "policy").upper()
        gantry_str = "ACTIVE (RIGID HOLD)" if state.get("gantry_active") else "RELEASED (FREE LOCOMOTION)"

        cmd = state.get("command", {})
        vx, vy, yaw = cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("vyaw", 0.0)

        base = state.get("base", {})
        bx, by, bz = base.get("x", 0.0), base.get("y", 0.0), base.get("z", 0.0)
        r_deg, p_deg, y_deg = base.get("roll_deg", 0.0), base.get("pitch_deg", 0.0), base.get("yaw_deg", 0.0)
        b_vx, b_vy, b_vz = base.get("body_vx", 0.0), base.get("body_vy", 0.0), base.get("body_vz", 0.0)

        contacts = state.get("contacts", {})
        lf = contacts.get("left_foot", {})
        rf = contacts.get("right_foot", {})
        lf_str = "CONTACT" if lf.get("contact") else "SWING  "
        rf_str = "CONTACT" if rf.get("contact") else "SWING  "
        lf_force = lf.get("force_z", 0.0)
        rf_force = rf.get("force_z", 0.0)

        # Gait metrics
        avg_stride = (sum(self.stride_times[-5:]) / len(self.stride_times[-5:])) if self.stride_times else 0.0

        # Limits count
        violations = sum(1 for j in state.get("joints", []) if j.get("limit_status") == "VIOLATION_POS")
        saturations = sum(1 for j in state.get("joints", []) if j.get("limit_status") == "SATURATED")

        # Policy state
        policy_obs = state.get("policy_obs", {})
        obs_dim = policy_obs.get("dim", 78)
        act_dim = state.get("policy_act", {}).get("dim", 23)

        env_cfg = state.get("environment", {})
        env_name = env_cfg.get("preset", "flat")
        friction = env_cfg.get("ground_friction", 1.0)

        lines = [
            "═" * 68,
            f" ASIMOV 1 STATUS — SIM TIME: {sim_t:7.3f}s   RTF: {rtf:5.3f}   ENV: {env_name} (mu={friction:.1f})",
            "═" * 68,
            f" CONTROL MODE : {ctrl_mode:<12} EDGE MODE : {mode:<18} GANTRY : {gantry_str}",
            f" COMMAND      : vx={vx:+.3f} m/s   vy={vy:+.3f} m/s   wz={yaw:+.3f} rad/s",
            f" BODY VEL     : vx={b_vx:+.3f} m/s   vy={b_vy:+.3f} m/s   vz={b_vz:+.3f} m/s",
            f" POSE (WORLD) : x={bx:+.3f} m   y={by:+.3f} m   z={bz:+.3f} m",
            f" ATTITUDE     : roll={r_deg:+.1f}°   pitch={p_deg:+.1f}°   yaw={y_deg:+.1f}°",
            "─" * 68,
            f" LEFT FOOT    : {lf_str}   Fz={lf_force:6.1f} N   (touch={lf.get('touch', 0.0):.2f})",
            f" RIGHT FOOT   : {rf_str}   Fz={rf_force:6.1f} N   (touch={rf.get('touch', 0.0):.2f})",
            f" GAIT         : transitions={self.step_transitions:<4}  stride_period={avg_stride:5.2f}s  contacts={contacts.get('contact_count', 0)}",
            "─" * 68,
            f" LIMITS       : violations={violations}   torque_saturations={saturations}",
            f" POLICY I/O   : inputs={obs_dim} (78-D)   outputs={act_dim} (23-D)   action_scale=0.25",
            "═" * 68,
        ]
        return "\n".join(lines)

    def format_joint_table(self, state: Dict[str, Any]) -> str:
        """Format 25-joint telemetry table."""
        joints = state.get("joints", [])
        if not joints:
            return "No joint telemetry available."

        header = (
            f"{'#':<3} {'Joint Name':<26} {'Pos (rad)':>9} {'Vel (r/s)':>9} "
            f"{'Target':>9} {'Torque(Nm)':>10} {'Limit':>11} {'Status':<10}"
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]

        for j in joints:
            idx = j.get("index", 0)
            name = j.get("name", "")
            pos = j.get("pos", 0.0)
            vel = j.get("vel", 0.0)
            tgt = j.get("target", 0.0)
            tau = j.get("torque", 0.0)
            l_min = j.get("limit_min", -3.14)
            l_max = j.get("limit_max", 3.14)
            status = j.get("limit_status", "OK")
            if j.get("is_overridden"):
                status = "OVERRIDE"
            elif j.get("is_selected"):
                status = "SELECTED"

            lim_str = f"[{l_min:+.1f},{l_max:+.1f}]"
            lines.append(
                f"{idx:<3} {name:<26} {pos:>+9.4f} {vel:>+9.3f} "
                f"{tgt:>+9.4f} {tau:>+10.2f} {lim_str:>11} {status:<10}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def format_contacts(self, state: Dict[str, Any]) -> str:
        """Format contact details and ground reaction forces."""
        contacts = state.get("contacts", {})
        sim_t = state.get("sim_time", 0.0)
        lf = contacts.get("left_foot", {})
        rf = contacts.get("right_foot", {})
        other = contacts.get("other_contacts", [])

        lines = [
            f"CONTACT TELEMETRY @ sim={sim_t:.3f}s",
            "─" * 48,
            f"LEFT FOOT  : {'CONTACT' if lf.get('contact') else 'SWING  '}  Fz={lf.get('force_z', 0.0):.2f} N  touch={lf.get('touch', 0.0):.2f}",
            f"RIGHT FOOT : {'CONTACT' if rf.get('contact') else 'SWING  '}  Fz={rf.get('force_z', 0.0):.2f} N  touch={rf.get('touch', 0.0):.2f}",
            f"TOTAL CONTACT CONSTRAINTS : {contacts.get('contact_count', 0)}",
            f"OTHER COLLISIONS          : {', '.join(other) if other else 'NONE'}",
        ]
        return "\n".join(lines)

    def format_policy_io(self, state: Dict[str, Any]) -> str:
        """Format 78-D observation and 23-D action breakdown."""
        obs = state.get("policy_obs", {})
        act = state.get("policy_act", {})

        if not obs.get("available") or not act.get("available"):
            return "Policy I/O not active or robot not in MOVE/POLICY mode."

        lines = [
            "═" * 70,
            " POLICY INPUT / OUTPUT PIPELINE (78 Obs -> 23 Act)",
            "═" * 70,
            "1. OBSERVATION VECTOR (78 dimensions):",
            f"   - Base Ang Vel (scaled 0.25): {[round(x, 4) for x in obs.get('base_ang_vel', [])]}",
            f"   - Projected Gravity (body)  : {[round(x, 4) for x in obs.get('projected_gravity', [])]}",
            f"   - Velocity Command [vx,vy,w]: {[round(x, 4) for x in obs.get('command', [])]}",
            f"   - Joint Positions (slot01/23/45): {len(obs.get('joint_pos', []))} values",
            f"   - Joint Velocities (scaled 0.1) : {len(obs.get('joint_vel', []))} values",
            f"   - Previous Action Vector        : {len(obs.get('prev_actions', []))} values",
            "─" * 70,
            "2. ACTION VECTOR & TARGET SYNTHESIS (23 dimensions):",
            f"   Transformation: q_des = q_default + 0.25 * raw_action",
            f"{'#':<3} {'Policy Joint':<24} {'Raw Act':>9} {'Scaled':>9} {'Default':>9} {'q_des (rad)':>11}",
            "─" * 70,
        ]

        for i, it in enumerate(act.get("items", [])):
            jn = it.get("joint", "")
            raw = it.get("raw_action", 0.0)
            scaled = it.get("scaled_delta", 0.0)
            d_pos = it.get("default_pos", 0.0)
            q_des = it.get("desired_target", 0.0)
            lines.append(
                f"{i+1:<3} {jn:<24} {raw:>+9.4f} {scaled:>+9.4f} {d_pos:>+9.4f} {q_des:>+11.4f}"
            )
        lines.append("═" * 70)
        return "\n".join(lines)

    def format_sensors(self, state: Dict[str, Any]) -> str:
        """Format simulated sensor outputs (IMU, base, battery)."""
        imu = state.get("imu", {})
        base = state.get("base", {})
        sim_t = state.get("sim_time", 0.0)

        lines = [
            f"SIMULATED SENSORS @ sim={sim_t:.3f}s",
            "─" * 56,
            "NOTE: Quantities are derived from MuJoCo physics (sim-only).",
            f"BASE GYRO (body frame) : {[round(x, 4) for x in imu.get('angular_velocity_body', [])]} rad/s",
            f"PROJECTED GRAVITY       : {[round(x, 4) for x in imu.get('projected_gravity', [])]}",
            f"ORIENTATION QUAT        : {[round(x, 4) for x in imu.get('orientation_quat', [])]} (w,x,y,z)",
            f"EULER ANGLES (deg)      : roll={base.get('roll_deg', 0.0):+.1f}°  pitch={base.get('pitch_deg', 0.0):+.1f}°  yaw={base.get('yaw_deg', 0.0):+.1f}°",
            f"LINEAR ACCEL (body)     : {[round(x, 4) for x in imu.get('linear_acceleration_body', [])]} m/s²",
        ]
        return "\n".join(lines)

    # ── Gait Tracking ────────────────────────────────────────────────────────

    def update_gait_metrics(self, state: Dict[str, Any]) -> None:
        """Update step transitions and stride duration based on foot contact changes."""
        contacts = state.get("contacts", {})
        sim_t = state.get("sim_time", 0.0)
        l_c = bool(contacts.get("left_foot", {}).get("contact", False))
        r_c = bool(contacts.get("right_foot", {}).get("contact", False))

        # Detect transition: either foot changed contact state
        if l_c != self.last_left_contact or r_c != self.last_right_contact:
            self.step_transitions += 1
            if self.last_step_sim_time > 0:
                dt = sim_t - self.last_step_sim_time
                if dt >= 0.1:  # Filter bounce chatter
                    self.stride_times.append(dt)
                    if len(self.stride_times) > 20:
                        self.stride_times.pop(0)
            self.last_step_sim_time = sim_t
            self.last_left_contact = l_c
            self.last_right_contact = r_c

    # ── Recording / Replay / Export ──────────────────────────────────────────

    def start_recording(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start capturing simulation frames into recording buffer."""
        self.recording_active = True
        self.current_recording_name = name.strip().replace(" ", "_")
        self.recording_buffer = []
        self.recording_metadata = {
            "name": self.current_recording_name,
            "start_wall_time": time.time(),
            "custom_metadata": metadata or {},
        }
        log.info("[RECORD] Started recording flight data session: '%s'", self.current_recording_name)

    def record_frame(self, state: Dict[str, Any]) -> None:
        """Append one frame to the recording buffer if active."""
        if not self.recording_active:
            return
        # Compact copy of frame for storage
        self.recording_buffer.append({
            "sim_time": state.get("sim_time", 0.0),
            "wall_time": state.get("wall_time", time.time()),
            "rtf": state.get("rtf", 1.0),
            "control_mode": state.get("control_mode", "policy"),
            "edge_mode": state.get("edge_mode", ""),
            "command": state.get("command", {}),
            "base_pos": state.get("base", {}).get("pos", []),
            "base_quat": state.get("base", {}).get("quat", []),
            "body_lin_vel": state.get("base", {}).get("lin_vel_body", []),
            "body_ang_vel": state.get("base", {}).get("ang_vel_body", []),
            "contacts": {
                "l_contact": state.get("contacts", {}).get("left_foot", {}).get("contact", False),
                "r_contact": state.get("contacts", {}).get("right_foot", {}).get("contact", False),
                "l_force_z": state.get("contacts", {}).get("left_foot", {}).get("force_z", 0.0),
                "r_force_z": state.get("contacts", {}).get("right_foot", {}).get("force_z", 0.0),
            },
            "joint_positions": [j.get("pos", 0.0) for j in state.get("joints", [])],
            "joint_velocities": [j.get("vel", 0.0) for j in state.get("joints", [])],
            "joint_targets": [j.get("target", 0.0) for j in state.get("joints", [])],
            "joint_torques": [j.get("torque", 0.0) for j in state.get("joints", [])],
            "policy_raw_actions": state.get("policy_act", {}).get("raw", []),
            "policy_obs_preview": state.get("policy_obs", {}).get("raw", [])[:10],
        })

    def stop_recording(self) -> Tuple[Path, Path]:
        """Stop recording and save JSON and CSV artifacts to reports/raw/."""
        if not self.recording_active or not self.current_recording_name:
            raise RuntimeError("No recording is currently active")

        self.recording_active = False
        name = self.current_recording_name
        _REPORTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

        json_path = _REPORTS_RAW_DIR / f"{name}.json"
        csv_path = _REPORTS_RAW_DIR / f"{name}.csv"

        payload = {
            "metadata": {
                **self.recording_metadata,
                "end_wall_time": time.time(),
                "frame_count": len(self.recording_buffer),
            },
            "frames": self.recording_buffer,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Export CSV summary
        if self.recording_buffer:
            fieldnames = [
                "sim_time", "control_mode", "edge_mode",
                "cmd_vx", "cmd_vy", "cmd_wz",
                "base_x", "base_y", "base_z",
                "body_vx", "body_vy", "body_wz",
                "left_contact", "right_contact",
                "left_force_z", "right_force_z",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for fr in self.recording_buffer:
                    cmd = fr.get("command", {})
                    bpos = fr.get("base_pos", [0, 0, 0])
                    blin = fr.get("body_lin_vel", [0, 0, 0])
                    bang = fr.get("body_ang_vel", [0, 0, 0])
                    cnt = fr.get("contacts", {})
                    writer.writerow({
                        "sim_time": f"{fr.get('sim_time', 0.0):.4f}",
                        "control_mode": fr.get("control_mode", ""),
                        "edge_mode": fr.get("edge_mode", ""),
                        "cmd_vx": f"{cmd.get('vx', 0.0):.4f}",
                        "cmd_vy": f"{cmd.get('vy', 0.0):.4f}",
                        "cmd_wz": f"{cmd.get('vyaw', 0.0):.4f}",
                        "base_x": f"{bpos[0] if len(bpos)>0 else 0.0:.4f}",
                        "base_y": f"{bpos[1] if len(bpos)>1 else 0.0:.4f}",
                        "base_z": f"{bpos[2] if len(bpos)>2 else 0.0:.4f}",
                        "body_vx": f"{blin[0] if len(blin)>0 else 0.0:.4f}",
                        "body_vy": f"{blin[1] if len(blin)>1 else 0.0:.4f}",
                        "body_wz": f"{bang[2] if len(bang)>2 else 0.0:.4f}",
                        "left_contact": int(cnt.get("l_contact", False)),
                        "right_contact": int(cnt.get("r_contact", False)),
                        "left_force_z": f"{cnt.get('l_force_z', 0.0):.2f}",
                        "right_force_z": f"{cnt.get('r_force_z', 0.0):.2f}",
                    })

        log.info("[RECORD] Saved %d frames to %s and %s", len(self.recording_buffer), json_path, csv_path)
        return json_path, csv_path

    @staticmethod
    def load_recording(name_or_path: Union[str, Path]) -> Dict[str, Any]:
        """Load recording data and metadata from JSON file."""
        p = Path(name_or_path)
        if not p.exists():
            p = _REPORTS_RAW_DIR / f"{name_or_path}.json"
        if not p.exists():
            raise FileNotFoundError(f"Recording file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
