#!/usr/bin/env python3
"""
vasimov/edge/sim.py
MuJoCo Simulation Backend with Virtual Gantry and Motor Model Integration.

Features:
- Steps MuJoCo at the physics rate (200 Hz, dt=0.005 s).
- Pacing: realtime wall-clock pacing (--realtime) or max-speed (--fast) for tests.
- Uses Step-2 MotorModel (vasimov/motor_model.py) with pd_update="per_physics_step".
- Gains per mode:
    * DAMP: kp=0, kd=2.0 (ASSUMED damping).
    * STAND & TRAJECTORY: default gains from vasimov/config/gains.yaml.
- Virtual Gantry:
    * Weld equality constraint on pelvis at settled height (0.60962 m).
    * Active at boot; auto-released 1.0 s after STAND ramp finishes (3.0 s after boot).
    * Re-engageable / releasable via control interface or Python API.
    * In DAMP with gantry OFF: robot collapses realistically under gravity.
- JSON Control Interface:
    * HTTP / JSON server on localhost:8852 (gantry on|off, push, inject, restart, status).
"""

from __future__ import annotations
import http.server
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml
import mujoco

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

import ankle_map
from adapter import JointAdapter, SIM_JOINTS, FIRMWARE_JOINTS
from edge.core import EdgeCore, EdgeMode
from motor_model import MotorModel, MotorState

log = logging.getLogger("vasimov.edge.sim")

# Settled base height from Task 0.5 [VERIFIED]
SETTLED_BASE_Z: float = 0.60962
GANTRY_AUTO_RELEASE_DELAY_S: float = 1.0  # 1s settle after 2s STAND ramp


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """
    Project world gravity vector [0, 0, -1] into body frame using base quat [w, x, y, z].
    """
    w, x, y, z = base_quat
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    a = v * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c


class SimBackend:
    """
    Manages the MuJoCo simulation environment, motor dynamics, and virtual gantry.
    """

    def __init__(
        self,
        core: EdgeCore,
        model_path: Optional[Union[str, Path]] = None,
        gains_path: Optional[Union[str, Path]] = None,
        pd_update: str = "per_physics_step",
        auto_gantry: bool = True,
        enable_l1: bool = False,
    ):
        self.core = core
        if model_path is None:
            model_path = _VASIMOV_DIR / "model" / "asimov_1_vasimov.xml"
        if gains_path is None:
            gains_path = _VASIMOV_DIR / "config" / "gains.yaml"

        self.model_path = Path(model_path)
        self.gains_path = Path(gains_path)
        self.pd_update = pd_update
        self.auto_gantry = auto_gantry
        self.ground_truth_server = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # 0.005 s (200 Hz)

        # Load joints config for telemetry
        with open(_VASIMOV_DIR / "config" / "joints.yaml", "r", encoding="utf-8") as f:
            self.joint_configs = yaml.safe_load(f)["joints"]

        # Foot contact sensor addresses (from derived model)
        def get_sensor(name: str) -> Tuple[int, int]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                return int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])
            return -1, 0

        self.sensor_left_touch = get_sensor("left_foot_touch")
        self.sensor_right_touch = get_sensor("right_foot_touch")
        self.sensor_left_force = get_sensor("left_foot_force")
        self.sensor_right_force = get_sensor("right_foot_force")

        # Ground truth streaming & RTF tracking
        self.step_count: int = 0
        self.last_rtf_wall_time: float = time.perf_counter()
        self.last_rtf_sim_time: float = 0.0
        self.current_rtf: float = 1.0
        self.last_ground_truth_frame: Optional[Dict[str, Any]] = None

        # Initialize MotorModel
        self.motor_model = MotorModel(
            motors_yaml_path=_VASIMOV_DIR / "config" / "motors.yaml",
            joints_yaml_path=_VASIMOV_DIR / "config" / "joints.yaml",
            gains_yaml_path=self.gains_path,
        )
        self.motor_model.set_layer_l1(enable_l1)

        # Find pelvis body id and virtual gantry equality constraint id
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        if self.pelvis_body_id == -1:
            self.pelvis_body_id = 1

        self.gantry_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "virtual_gantry")

        # Map actuators to qpos/qvel indices
        self.actuator_qposadr = []
        self.actuator_dofadr = []
        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self.actuator_qposadr.append(self.model.jnt_qposadr[jnt_id])
            self.actuator_dofadr.append(self.model.jnt_dofadr[jnt_id])

        # Push disturbance tracking
        self.push_remaining_steps: int = 0
        self.push_force_vec = np.zeros(6, dtype=np.float64)

        # Gantry state
        self.gantry_active: bool = True
        self.gantry_released_by_auto: bool = False

        # Reset robot to settled pose with gantry active
        self.reset()

    def reset(self) -> None:
        """Reset simulation state and re-engage gantry."""
        mujoco.mj_resetData(self.model, self.data)

        # Set floating base at settled height
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = SETTLED_BASE_Z
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0
        self.data.qvel[:] = 0.0

        # Initialize joint angles to default standing pose
        for i in range(self.model.nu):
            qpos_adr = self.actuator_qposadr[i]
            self.data.qpos[qpos_adr] = self.core.default_pose_sim[i]

        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # Re-engage gantry
        self.set_gantry(True)
        self.gantry_released_by_auto = False

        mujoco.mj_forward(self.model, self.data)

    def set_gantry(self, active: bool) -> None:
        """Toggle virtual gantry weld equality constraint."""
        self.gantry_active = bool(active)
        if self.gantry_eq_id != -1:
            self.data.eq_active[self.gantry_eq_id] = 1 if self.gantry_active else 0
        log.info("[SIM] Virtual Gantry %s", "ENGAGED (pelvis locked at %.4fm)" % SETTLED_BASE_Z if self.gantry_active else "RELEASED (free motion)")

    def apply_push(self, force_n: float, direction: str = "x", duration_s: float = 0.1) -> None:
        """Apply a disturbance force to the pelvis."""
        steps = max(1, int(round(duration_s / self.dt)))
        self.push_remaining_steps = steps
        self.push_force_vec[:] = 0.0
        idx = 0 if direction.lower() == "x" else (1 if direction.lower() == "y" else 2)
        self.push_force_vec[idx] = float(force_n)
        log.warning("[SIM] Applying push: %.1f N in '%s' for %.2fs (%d steps)", force_n, direction, duration_s, steps)

    def step(self, realtime: bool = False) -> None:
        """Step physics by one integration step (200 Hz, 0.005 s)."""
        sim_time = self.data.time

        # Apply push disturbance if active
        if self.push_remaining_steps > 0:
            self.data.xfrc_applied[self.pelvis_body_id, :] = self.push_force_vec
            self.push_remaining_steps -= 1
        else:
            self.data.xfrc_applied[self.pelvis_body_id, :] = 0.0

        # Auto-release gantry: 1.0s after STAND ramp finishes
        if self.auto_gantry and self.gantry_active and not self.gantry_released_by_auto:
            if self.core.mode == EdgeMode.STAND and self.core.stand_settled:
                ramp_finished_time = self.core.stand_ramp_start_time + 2.0
                if sim_time >= ramp_finished_time + GANTRY_AUTO_RELEASE_DELAY_S:
                    log.info("[SIM] STAND ramp + 1.0s settle elapsed. Auto-releasing virtual gantry.")
                    self.set_gantry(False)
                    self.gantry_released_by_auto = True

        # Read current sensor state
        current_pos = [float(self.data.qpos[adr]) for adr in self.actuator_qposadr]
        current_vel = [float(self.data.qvel[adr]) for adr in self.actuator_dofadr]
        base_quat = np.array([self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]], dtype=np.float64)
        base_gyro = [float(self.data.qvel[3]), float(self.data.qvel[4]), float(self.data.qvel[5])]
        projected_gravity = compute_projected_gravity(base_quat)

        # Step EdgeCore timers, watchdogs & safety checks
        self.core.step_state(sim_time, projected_gravity)

        # Get target positions and gains from EdgeCore
        targets, kp_vals, kd_vals = self.core.get_control_targets(current_pos, sim_time)

        # Compute torques via MotorModel
        q_arr = np.array(current_pos, dtype=np.float64)
        qdot_arr = np.array(current_vel, dtype=np.float64)
        q_des_arr = np.array(targets, dtype=np.float64)
        qdot_des_arr = np.zeros(self.model.nu, dtype=np.float64)

        if self.core.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
            effective_kp = [0.0] * 25
            effective_kd = [2.0] * 25
        else:
            effective_kp = kp_vals
            effective_kd = kd_vals

        applied_torques, _ = self.motor_model.step(
            q=q_arr,
            qdot=qdot_arr,
            q_des=q_des_arr,
            qdot_des=qdot_des_arr,
            dt=self.dt,
            kp_override=effective_kp,
            kd_override=effective_kd,
        )
        self.data.ctrl[:] = applied_torques

        # Step physics
        mujoco.mj_step(self.model, self.data)

        # Step counter and RTF tracking
        self.step_count += 1
        if self.step_count % 50 == 0:
            now_wall = time.perf_counter()
            sim_elapsed = self.data.time - self.last_rtf_sim_time
            wall_elapsed = now_wall - self.last_rtf_wall_time
            if wall_elapsed > 0:
                self.current_rtf = max(0.01, sim_elapsed / wall_elapsed)
            self.last_rtf_wall_time = now_wall
            self.last_rtf_sim_time = self.data.time

        # Ground-truth stream at ~50 Hz (every 4 steps of 200 Hz = 50 Hz)
        if self.step_count % 4 == 0 and self.ground_truth_server is not None:
            frame = self.get_ground_truth_frame()
            self.ground_truth_server.broadcast(frame)

    def get_ground_truth_frame(self, rtf: Optional[float] = None) -> Dict[str, Any]:
        """Construct the complete ~50 Hz sim-only ground-truth payload."""
        sim_time = float(self.data.time)
        effective_rtf = float(rtf) if rtf is not None else self.current_rtf

        # Pelvis 6-DoF pose, velocity & tilt
        pelvis_pos = [float(self.data.qpos[i]) for i in range(3)]
        pelvis_quat = [float(self.data.qpos[i]) for i in range(3, 7)]
        pelvis_lin_vel = [float(self.data.qvel[i]) for i in range(3)]
        pelvis_ang_vel = [float(self.data.qvel[i]) for i in range(3, 6)]
        proj_grav = compute_projected_gravity(np.array(pelvis_quat))
        norm_g = math.sqrt(sum(g * g for g in proj_grav)) or 1.0
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -proj_grav[2] / norm_g))))

        # Foot contacts
        def read_sensor(adr_dim: Tuple[int, int]) -> List[float]:
            adr, dim = adr_dim
            if adr != -1:
                return [float(x) for x in self.data.sensordata[adr:adr + dim]]
            return [0.0] * dim

        left_touch = read_sensor(self.sensor_left_touch)
        right_touch = read_sensor(self.sensor_right_touch)
        left_force = read_sensor(self.sensor_left_force)
        right_force = read_sensor(self.sensor_right_force)

        f_l_z = abs(left_force[2]) if len(left_force) >= 3 else 0.0
        f_r_z = abs(right_force[2]) if len(right_force) >= 3 else 0.0

        # Ankles: Joint Pitch/Roll vs Pushrod Motors A/B
        q_pitch_l = float(self.data.qpos[self.actuator_qposadr[4]])
        q_roll_l = float(self.data.qpos[self.actuator_qposadr[5]])
        tau_pitch_l = float(self.data.ctrl[4])
        tau_roll_l = float(self.data.ctrl[5])
        m_a_l, m_b_l = ankle_map.pr_to_ab(q_pitch_l, q_roll_l)
        tau_a_l, tau_b_l = ankle_map.torque_pr_to_ab(tau_pitch_l, tau_roll_l)

        q_pitch_r = float(self.data.qpos[self.actuator_qposadr[10]])
        q_roll_r = float(self.data.qpos[self.actuator_qposadr[11]])
        tau_pitch_r = float(self.data.ctrl[10])
        tau_roll_r = float(self.data.ctrl[11])
        m_a_r, m_b_r = ankle_map.pr_to_ab(q_pitch_r, q_roll_r)
        tau_a_r, tau_b_r = ankle_map.torque_pr_to_ab(tau_pitch_r, tau_roll_r)

        # Targets and active gains
        current_sim_pos = [float(self.data.qpos[adr]) for adr in self.actuator_qposadr]
        targets, kp_vals, kd_vals = self.core.get_control_targets(current_sim_pos, sim_time)

        # 25 Joints in firmware order (0..24)
        joints_payload = []
        for j_cfg in self.joint_configs:
            idx = j_cfg["index"]
            fw_name = j_cfg["firmware_name"]
            sim_name = j_cfg["sim_joint"]
            pos_min, pos_max = j_cfg["sim_range"]
            effort_limit = float(j_cfg["effort_limit"])

            sim_idx = SIM_JOINTS.index(sim_name)
            pos = float(self.data.qpos[self.actuator_qposadr[sim_idx]])
            vel = float(self.data.qvel[self.actuator_dofadr[sim_idx]])
            tgt = float(targets[sim_idx])
            err = tgt - pos
            tau = float(self.data.ctrl[sim_idx])
            pct_eff = min(100.0, (abs(tau) / effort_limit) * 100.0) if effort_limit > 0 else 0.0
            rng = pos_max - pos_min
            pct_rng = min(100.0, max(0.0, ((pos - pos_min) / rng) * 100.0)) if rng > 0 else 50.0

            joints_payload.append({
                "index": idx,
                "firmware_name": fw_name,
                "sim_name": sim_name,
                "pos": pos,
                "vel": vel,
                "target": tgt,
                "error": err,
                "torque": tau,
                "effort_limit": effort_limit,
                "torque_pct_effort": pct_eff,
                "range": [pos_min, pos_max],
                "pos_pct_range": pct_rng,
                "kp": float(kp_vals[sim_idx]),
                "kd": float(kd_vals[sim_idx]),
            })

        frame = {
            "stream_metadata": {
                "sim_only": True,
                "hardware_equivalent": False,
                "description": "Virtual Asimov sim-only ground-truth stream. Has no hardware equivalent.",
            },
            "sim_time": sim_time,
            "real_time_factor": effective_rtf,
            "mode": self.core.mode.name,
            "gantry_active": self.gantry_active,
            "fault_state": {
                "latched": self.core.fault_latched,
                "fall": self.core.fault_fall,
                "overtemp": self.core.fault_overtemp,
                "can": self.core.fault_can,
                "error_flags": self.core.error_flags,
            },
            "battery": {
                "soc_percent": self.core.battery_soc,
                "voltage_v": self.core.battery_voltage,
                "discharge_rate_pct_s": self.core.battery_discharge_rate,
            },
            "pelvis": {
                "pos": pelvis_pos,
                "quat": pelvis_quat,
                "tilt_deg": tilt_deg,
                "lin_vel": pelvis_lin_vel,
                "ang_vel": pelvis_ang_vel,
            },
            "contacts": {
                "left_foot": {
                    "contact": bool((left_touch[0] > 0.5 if left_touch else False) or f_l_z > 5.0),
                    "touch": left_touch[0] if left_touch else 0.0,
                    "force": left_force,
                    "normal_force": f_l_z,
                },
                "right_foot": {
                    "contact": bool((right_touch[0] > 0.5 if right_touch else False) or f_r_z > 5.0),
                    "touch": right_touch[0] if right_touch else 0.0,
                    "force": right_force,
                    "normal_force": f_r_z,
                },
            },
            "ankles": {
                "left": {
                    "pitch_rad": q_pitch_l,
                    "roll_rad": q_roll_l,
                    "motor_a_rad": m_a_l,
                    "motor_b_rad": m_b_l,
                    "motor_a_torque_nm": tau_a_l,
                    "motor_b_torque_nm": tau_b_l,
                },
                "right": {
                    "pitch_rad": q_pitch_r,
                    "roll_rad": q_roll_r,
                    "motor_a_rad": m_a_r,
                    "motor_b_rad": m_b_r,
                    "motor_a_torque_nm": tau_a_r,
                    "motor_b_torque_nm": tau_b_r,
                },
            },
            "joints": joints_payload,
        }
        self.last_ground_truth_frame = frame
        return frame


# ── JSON Control Server (localhost:8852) ─────────────────────────────────────
class SimControlServer:
    """Lightweight HTTP server on localhost:8852 for sim control and dashboard."""

    def __init__(self, backend: SimBackend, port: int = 8852):
        self.backend = backend
        self.port = port
        self.server: Optional[http.server.HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        backend = self.backend

        class RequestHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Keep quiet

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html", "/dashboard"):
                    dash_path = _VASIMOV_DIR / "dashboard" / "index.html"
                    if dash_path.exists():
                        payload = dash_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    else:
                        self.send_error(404, "Dashboard HTML file not found")
                        return

                elif self.path == "/ground_truth":
                    frame = backend.get_ground_truth_frame()
                    payload = json.dumps(frame).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                elif self.path == "/status":
                    sim_z = float(backend.data.qpos[2])
                    quat = [float(backend.data.qpos[3]), float(backend.data.qpos[4]), float(backend.data.qpos[5]), float(backend.data.qpos[6])]
                    grav = compute_projected_gravity(np.array(quat))
                    data = {
                        "mode": backend.core.mode.name,
                        "gantry_active": backend.gantry_active,
                        "base_z": sim_z,
                        "projected_gravity_z": float(grav[2]),
                        "fault_latched": backend.core.fault_latched,
                        "error_flags": backend.core.error_flags,
                        "active_controller": backend.core.active_controller,
                        "battery_soc": backend.core.battery_soc,
                        "battery_voltage": backend.core.battery_voltage,
                        "battery_discharge_rate": backend.core.battery_discharge_rate,
                    }
                    payload = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if self.path == "/control":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length).decode("utf-8")
                    try:
                        req = json.loads(body)
                    except Exception:
                        self.send_error(400, "Invalid JSON")
                        return

                    cmd = req.get("command", "").lower()
                    if cmd == "gantry":
                        state = req.get("state", "off").lower()
                        backend.set_gantry(state == "on")
                        resp = {"status": "ok", "gantry": backend.gantry_active}
                    elif cmd == "push":
                        force = float(req.get("force", 40.0))
                        direction = str(req.get("dir", "x"))
                        duration = float(req.get("duration", 0.1))
                        backend.apply_push(force, direction, duration)
                        resp = {"status": "ok", "pushed": force}
                    elif cmd == "inject":
                        fault = str(req.get("fault", "")).lower()
                        if fault == "fall":
                            backend.core.inject_fall()
                        elif fault == "overtemp":
                            backend.core.inject_overtemp(float(req.get("temp", 85.0)))
                        elif fault == "can":
                            backend.core.inject_can_fault()
                        resp = {"status": "ok", "injected": fault}
                    elif cmd in ("set_battery", "battery"):
                        soc = req.get("soc", req.get("percentage"))
                        rate = req.get("rate", req.get("discharge_rate"))
                        voltage = req.get("voltage")
                        backend.core.set_battery(soc=soc, discharge_rate=rate, voltage=voltage)
                        resp = {"status": "ok", "battery_soc": backend.core.battery_soc, "discharge_rate": backend.core.battery_discharge_rate}
                    elif cmd == "restart":
                        backend.core.virtual_restart()
                        backend.reset()
                        resp = {"status": "ok", "restarted": True}
                    else:
                        resp = {"status": "error", "message": f"Unknown command '{cmd}'"}

                    payload = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)

        try:
            self.server = http.server.HTTPServer(("127.0.0.1", self.port), RequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="vasimov-sim-control")
            self.thread.start()
            log.info("[SIM CONTROL] Listening on http://127.0.0.1:%d", self.port)
        except Exception as e:
            log.warning("[SIM CONTROL] Could not bind port %d (%s); control server disabled", self.port, e)

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
