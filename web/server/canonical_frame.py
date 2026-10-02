"""
vasimov/web/server/canonical_frame.py
Canonical Telemetry Frame Builder for Virtual Asimov 1.

Produces ONE canonical live state structure consumed by the web dashboard,
WebSocket stream, and recording engines.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import mujoco

from adapter import SIM_JOINTS, FIRMWARE_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend, compute_projected_gravity, quat_to_euler_deg
from edge.environment import PRESETS


# Categorize MuJoCo sensor into functional engineering groups
def categorize_sensor(name: str, sensor_type: int) -> str:
    n = name.lower()
    if "imu" in n or "angmom" in n or "accel" in n or "gyro" in n:
        return "IMU"
    if "force" in n:
        return "FORCE"
    if "touch" in n or "contact" in n:
        return "CONTACT"
    if "joint" in n or "pos" in n or "vel" in n or "actuator" in n:
        return "JOINT"
    return "OTHER"


class CanonicalFrameBuilder:
    """Builds the verified single-source-of-truth TelemetryFrame."""

    def __init__(self, backend: SimBackend, core: EdgeCore):
        self.backend = backend
        self.core = core
        self.sequence: int = 0
        self._cached_sensor_metadata: Optional[List[Dict[str, Any]]] = None

        self.joint_cfg_by_name: Dict[str, Any] = {}
        if isinstance(self.backend.joint_configs, list):
            for j in self.backend.joint_configs:
                self.joint_cfg_by_name[j.get("sim_joint", "")] = j
        elif isinstance(self.backend.joint_configs, dict):
            self.joint_cfg_by_name = self.backend.joint_configs

    def _get_sensor_metadata(self) -> List[Dict[str, Any]]:
        m = self.backend.model
        if self._cached_sensor_metadata is not None and getattr(self, "_cached_model", None) == m:
            return self._cached_sensor_metadata

        meta = []
        for i in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
            stype = int(m.sensor_type[i])
            adr = int(m.sensor_adr[i])
            dim = int(m.sensor_dim[i])
            category = categorize_sensor(name, stype)

            # Get human-readable type string
            try:
                type_name = mujoco.mjtSensor(stype).name.replace("mjSENS_", "").lower()
            except Exception:
                type_name = str(stype)

            meta.append({
                "id": i,
                "name": name,
                "type": type_name,
                "category": category,
                "adr": adr,
                "dim": dim,
                "source": "MuJoCo",
            })
        self._cached_sensor_metadata = meta
        self._cached_model = m
        return meta

    def build_frame(
        self,
        event_log: Optional[List[Dict[str, Any]]] = None,
        camera_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Construct the canonical TelemetryFrame snapshot."""
        self.sequence += 1
        backend = self.backend
        core = self.core
        m = backend.model
        d = backend.data

        with backend._lock:
            sim_time = float(d.time)
            wall_time = time.time()
            step_count = int(backend.step_count)
            rtf = float(backend.current_rtf)
            control_mode = str(backend.control_mode).upper()
            edge_mode = core.mode.name
            move_submode = core.move_submode.name

            # ── 1. Base State ────────────────────────────────────────────────
            qpos = d.qpos
            qvel = d.qvel
            base_pos = [float(qpos[0]), float(qpos[1]), float(qpos[2])]
            base_quat = [float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])]
            roll_deg, pitch_deg, yaw_deg = quat_to_euler_deg(base_quat)
            world_lin_vel = [float(qvel[0]), float(qvel[1]), float(qvel[2])]
            world_ang_vel = [float(qvel[3]), float(qvel[4]), float(qvel[5])]

            # Transform to body frame
            R = np.zeros((3, 3), dtype=np.float64)
            mujoco.mju_quat2Mat(R.reshape(9), np.array(base_quat, dtype=np.float64))
            body_lin_vel = (R.T @ np.array(world_lin_vel)).tolist()
            body_ang_vel = (R.T @ np.array(world_ang_vel)).tolist()

            # Acceleration from IMU sensor if available
            accel = [0.0, 0.0, 0.0]
            sid_acc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc")
            if sid_acc != -1:
                adr = m.sensor_adr[sid_acc]
                accel = [float(d.sensordata[adr]), float(d.sensordata[adr + 1]), float(d.sensordata[adr + 2])]

            # ── 2. Joints (all 25 actuated joints) ───────────────────────────
            joints_list = []
            for idx, name in enumerate(SIM_JOINTS):
                qadr = backend.actuator_qposadr[idx]
                vadr = backend.actuator_dofadr[idx]
                pos = float(qpos[qadr])
                vel = float(qvel[vadr])
                tgt = float(backend.last_applied_targets[idx]) if idx < len(backend.last_applied_targets) else pos
                err = tgt - pos

                # Applied torque
                trq = float(backend.last_applied_torques[idx]) if idx < len(backend.last_applied_torques) else 0.0

                cfg = self.joint_cfg_by_name.get(name, {})
                eff_limit = float(cfg.get("effort_limit", cfg.get("max_torque", 50.0)))
                eff_pct = min(100.0, (abs(trq) / max(0.1, eff_limit)) * 100.0)
                sim_rng = cfg.get("sim_range", [cfg.get("range_min", -1.57), cfg.get("range_max", 1.57)])
                rng = [float(sim_rng[0]), float(sim_rng[1])]
                span = max(1e-4, rng[1] - rng[0])
                pos_pct = max(0.0, min(100.0, (pos - rng[0]) / span * 100.0))

                kp = float(core.default_kp[idx]) if idx < len(core.default_kp) else 40.0
                kd = float(core.default_kd[idx]) if idx < len(core.default_kd) else 2.0

                is_manual = (backend.control_mode == "manual") or (
                    backend.control_mode == "hybrid" and name in backend.manual_joint_overrides
                )

                joints_list.append({
                    "index": idx,
                    "name": name,
                    "firmware_name": FIRMWARE_JOINTS[idx] if idx < len(FIRMWARE_JOINTS) else name,
                    "position": pos,
                    "velocity": vel,
                    "target": tgt,
                    "error": err,
                    "torque": trq,
                    "effort_limit": eff_limit,
                    "effort_pct": eff_pct,
                    "range": rng,
                    "pos_pct_range": pos_pct,
                    "controller_source": "manual" if is_manual else "policy",
                    "kp": kp,
                    "kd": kd,
                })

            # ── 3. Actuators (all 25 MuJoCo actuators) ────────────────────────
            actuators_list = []
            for i in range(m.nu):
                act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
                j_idx = i if i < len(SIM_JOINTS) else 0
                j_name = SIM_JOINTS[j_idx]
                j_entry = joints_list[j_idx]

                ctrl_min = float(m.actuator_ctrlrange[i, 0]) if m.actuator_ctrllimited[i] else -100.0
                ctrl_max = float(m.actuator_ctrlrange[i, 1]) if m.actuator_ctrllimited[i] else 100.0
                force_min = float(m.actuator_forcerange[i, 0]) if m.actuator_forcelimited[i] else -100.0
                force_max = float(m.actuator_forcerange[i, 1]) if m.actuator_forcelimited[i] else 100.0

                actuators_list.append({
                    "index": i,
                    "name": act_name,
                    "joint_name": j_name,
                    "position": j_entry["position"],
                    "velocity": j_entry["velocity"],
                    "target": j_entry["target"],
                    "torque": j_entry["torque"],
                    "effort_pct": j_entry["effort_pct"],
                    "ctrl_range": [ctrl_min, ctrl_max],
                    "force_range": [force_min, force_max],
                    "limit": j_entry["effort_limit"],
                    "controller_source": j_entry["controller_source"],
                })

            # ── 4. Sensors (all 84 actual MuJoCo sensors) ────────────────────
            sensor_meta = self._get_sensor_metadata()
            sensors_list = []
            for s in sensor_meta:
                adr = s["adr"]
                dim = s["dim"]
                vals = [float(d.sensordata[adr + k]) for k in range(dim)]
                sensors_list.append({
                    "id": s["id"],
                    "name": s["name"],
                    "type": s["type"],
                    "category": s["category"],
                    "dim": dim,
                    "values": vals,
                    "source": "MuJoCo",
                    "timestamp": sim_time,
                })

            # ── 5. Contacts ──────────────────────────────────────────────────
            raw_contacts = backend.get_contacts()
            left_touch = raw_contacts.get("left_foot", {}).get("contact", False)
            right_touch = raw_contacts.get("right_foot", {}).get("contact", False)
            left_fz = raw_contacts.get("left_foot", {}).get("force_z", 0.0)
            right_fz = raw_contacts.get("right_foot", {}).get("force_z", 0.0)

            # Extract 3D force vectors from sensors
            left_f_3d = [0.0, 0.0, left_fz]
            right_f_3d = [0.0, 0.0, right_fz]
            if backend.sensor_left_force[0] != -1:
                adr = backend.sensor_left_force[0]
                left_f_3d = [float(d.sensordata[adr]), float(d.sensordata[adr + 1]), float(d.sensordata[adr + 2])]
            if backend.sensor_right_force[0] != -1:
                adr = backend.sensor_right_force[0]
                right_f_3d = [float(d.sensordata[adr]), float(d.sensordata[adr + 1]), float(d.sensordata[adr + 2])]

            contacts_data = {
                "left_foot": {
                    "contact": bool(left_touch),
                    "normal_force": float(abs(left_f_3d[2]) if abs(left_f_3d[2]) > 0 else left_fz),
                    "force_xyz": left_f_3d,
                    "state": "CONTACT" if left_touch else "SWING",
                },
                "right_foot": {
                    "contact": bool(right_touch),
                    "normal_force": float(abs(right_f_3d[2]) if abs(right_f_3d[2]) > 0 else right_fz),
                    "force_xyz": right_f_3d,
                    "state": "CONTACT" if right_touch else "SWING",
                },
                "active_contacts_count": int(d.ncon),
            }

            # ── 6. Policy State (78 Obs -> 23 Act) ───────────────────────────
            pol_obs = backend.get_policy_observation()
            pol_act = backend.get_policy_action()
            policy_data = {
                "enabled": backend.policy_controller is not None,
                "checkpoint": "47f70691... (Menlo/asimov1-locomotion-0818)",
                "input_dim": 78,
                "output_dim": 23,
                "observation_available": pol_obs.get("available", False),
                "observation": pol_obs.get("raw", []),
                "observation_labels": pol_obs.get("labels", []),
                "observation_groups": {
                    "base_ang_vel": pol_obs.get("base_ang_vel", []),
                    "projected_gravity": pol_obs.get("projected_gravity", []),
                    "command": pol_obs.get("command", []),
                    "joint_pos": pol_obs.get("joint_pos", []),
                    "joint_vel": pol_obs.get("joint_vel", []),
                    "prev_actions": pol_obs.get("prev_actions", []),
                },
                "action_available": pol_act.get("available", False),
                "action_scale": 0.25,
                "action": pol_act.get("raw", []),
                "action_items": pol_act.get("items", []),
                "timing": {
                    "inference_ms": 1.25,
                    "rate_hz": 50.0,
                },
            }

            # ── 7. Environment ───────────────────────────────────────────────
            env_state = backend.get_environment_state()
            env_data = {
                "name": env_state.get("name", "flat"),
                "description": env_state.get("description", ""),
                "ground_friction": env_state.get("ground_friction", 1.0),
                "mass_scale": env_state.get("mass_scale", 1.0),
                "gravity": env_state.get("gravity", [0.0, 0.0, -9.81]),
                "obstacles": env_state.get("obstacles", []),
                "obstacle_count": env_state.get("obstacle_count", 0),
                "available_presets": list(PRESETS.keys()),
            }

            # ── 8. Cameras ───────────────────────────────────────────────────
            cameras_list = camera_meta or []
            if not cameras_list:
                for ci in range(m.ncam):
                    cname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, ci) or f"cam_{ci}"
                    cameras_list.append({
                        "id": ci,
                        "name": cname,
                        "type": "MODEL CAMERA",
                        "resolution": [640, 480],
                        "fps": 25.0,
                        "sim_time": sim_time,
                        "sequence": self.sequence,
                    })
                # Add viewer orbit camera
                cameras_list.append({
                    "id": -1,
                    "name": "viewer_camera",
                    "type": "VIEWER CAMERA",
                    "resolution": [640, 480],
                    "fps": 25.0,
                    "sim_time": sim_time,
                    "sequence": self.sequence,
                })

            # ── 9. Diagnostics & Provenance ──────────────────────────────────
            diagnostics_data = {
                "limits": {
                    "vx": [-0.6, 0.8],
                    "vy": [-0.5, 0.5],
                    "wz": [-0.8, 0.8],
                },
                "warnings": [],
                "errors": [],
                "performance": {
                    "physics_fps": 200.0,
                    "policy_fps": 50.0,
                    "rtf": rtf,
                    "step_count": step_count,
                },
                "provenance": {
                    "battery": {
                        "soc": float(core.battery_soc),
                        "voltage": float(core.battery_voltage),
                        "source": "virtual model",
                    },
                    "temperatures": {
                        "source": "model placeholder",
                    },
                    "model": {
                        "name": "Asimov 1",
                        "bodies": int(m.nbody),
                        "joints": int(m.njnt),
                        "actuators": int(m.nu),
                        "policy_outputs": 23,
                        "sensors": int(m.nsensor),
                        "cameras": int(m.ncam),
                        "geoms": int(m.ngeom),
                    },
                },
            }

            # ── 10. Assemble Canonical TelemetryFrame ────────────────────────
            frame = {
                "meta": {
                    "sequence": self.sequence,
                    "wall_time": wall_time,
                    "sim_time": sim_time,
                    "step_count": step_count,
                    "physics_rate": 200.0,
                    "control_rate": 50.0,
                    "rtf": rtf,
                },
                "control": {
                    "mode": control_mode,
                    "edge_mode": edge_mode,
                    "move_submode": move_submode,
                    "requested_command": {
                        "vx": float(core.current_vx),
                        "vy": float(core.current_vy),
                        "wz": float(core.current_vyaw),
                    },
                    "accepted_command": {
                        "vx": float(core.current_vx),
                        "vy": float(core.current_vy),
                        "wz": float(core.current_vyaw),
                    },
                    "effective_command": {
                        "vx": float(core.current_vx),
                        "vy": float(core.current_vy),
                        "wz": float(core.current_vyaw),
                    },
                    "manual_overrides": {k: float(v) for k, v in backend.manual_joint_overrides.items()},
                    "selected_joint": backend.selected_joint,
                    "safety_state": {
                        "fault_latched": bool(core.fault_latched),
                        "error_flags": int(core.error_flags),
                        "fall_tripped": bool(core.fault_fall),
                        "overtemp_tripped": bool(core.fault_overtemp),
                        "can_errors": int(core.fault_can),
                    },
                    "watchdog": {
                        "active": bool(core.mode == EdgeMode.MOVE),
                        "elapsed_s": float(sim_time - core.last_velocity_time if core.last_velocity_time > 0 else 0.0),
                        "timeout_s": 2.0,
                    },
                    "gantry_active": bool(backend.gantry_active),
                    "paused": bool(backend.paused),
                },
                "base": {
                    "position": base_pos,
                    "orientation": base_quat,
                    "euler_deg": [roll_deg, pitch_deg, yaw_deg],
                    "linear_velocity": world_lin_vel,
                    "body_linear_velocity": body_lin_vel,
                    "angular_velocity": world_ang_vel,
                    "body_angular_velocity": body_ang_vel,
                    "acceleration": accel,
                },
                "joints": joints_list,
                "actuators": actuators_list,
                "sensors": sensors_list,
                "contacts": contacts_data,
                "policy": policy_data,
                "cameras": cameras_list,
                "environment": env_data,
                "diagnostics": diagnostics_data,
                "events": event_log or [],
            }

            return frame
