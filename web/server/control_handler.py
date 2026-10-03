"""
vasimov/web/server/control_handler.py
Command Lifecycle Engine and Control Dispatcher for Virtual Asimov 1.

Implements the mandatory lifecycle:
USER INPUT -> PARSED -> VALIDATED -> ACCEPTED / REJECTED -> APPLIED -> SIMULATOR RESULT
"""

from __future__ import annotations
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from adapter import SIM_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.policy import VX_MIN, VX_MAX, VY_MIN, VY_MAX, WZ_MIN, WZ_MAX
from edge.environment import PRESETS

log = logging.getLogger("vasimov.web.control")


class ControlHandler:
    """Dispatches web commands through validated lifecycle with acknowledgement."""

    def __init__(self, backend, core: EdgeCore, telemetry_engine=None, replay_manager=None):
        self.backend = backend
        self.core = core
        self.telemetry = telemetry_engine
        self.replay_manager = replay_manager
        self.command_id: int = 0
        self.recent_events: List[Dict[str, Any]] = []

    def record_event(self, text: str, level: str = "info") -> None:
        """Add to internal timestamped event log (capped at 50)."""
        sim_t = float(self.backend.data.time)
        entry = {
            "time": sim_t,
            "wall_time": time.time(),
            "event": text,
            "level": level,
        }
        self.recent_events.append(entry)
        if len(self.recent_events) > 50:
            self.recent_events.pop(0)

    def handle_command(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute command with full explicit lifecycle tracking.
        """
        self.command_id += 1
        cmd_id = self.command_id
        action = action.lower().strip()
        t0 = time.time()

        ack: Dict[str, Any] = {
            "command_id": cmd_id,
            "action": action,
            "timestamp": t0,
            "sim_time": float(self.backend.data.time),
            "status": "pending",
            "lifecycle": {
                "user_input": f"{action} {params}",
                "parsed": {},
                "validation": "OK",
                "clipped": False,
                "reason": "",
                "applied": "",
                "simulator_result": "",
            },
        }

        # ── 1. STAND ─────────────────────────────────────────────────────────
        if action in ("stand", "arm"):
            ack["lifecycle"]["parsed"] = {"action": "STAND"}
            self.core.set_active_controller("sdk")
            if self.core.fault_latched:
                self.core.clear_fault("web_cmd")

            cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
            sim_t = float(self.backend.data.time)
            ok = self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
            if ok:
                self.core.stand_settled = True
                if getattr(self.backend, "gantry_active", False):
                    self.backend.set_gantry(False)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "EdgeCore.command_stand -> STAND active"
                ack["lifecycle"]["simulator_result"] = f"mode = STAND, base_z = {self.backend.data.qpos[2]:.4f}m"
                self.record_event(f"Command STAND: standing pose engaged (sim_t={sim_t:.2f}s)")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = "REJECTED: command_stand failed"

        # ── 1b. DAMP ─────────────────────────────────────────────────────────
        elif action in ("damp", "park", "disarm"):
            ack["lifecycle"]["parsed"] = {"action": "DAMP"}
            self.core.set_active_controller("sdk")
            sim_t = float(self.backend.data.time)
            ok = self.core.command_damp("sdk", current_time=sim_t)
            if ok:
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "EdgeCore.command_damp -> DAMP mode"
                ack["lifecycle"]["simulator_result"] = f"mode = DAMP, base_z = {self.backend.data.qpos[2]:.4f}m"
                self.record_event(f"Command DAMP: compliant damping engaged (sim_t={sim_t:.2f}s)")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = "REJECTED: command_damp failed"

        # ── 1c. GETUP / RECOVER ──────────────────────────────────────────────
        elif action in ("getup", "recover", "standup"):
            ack["lifecycle"]["parsed"] = {"action": "RECOVER / GETUP"}
            self.core.set_active_controller("sdk")
            if self.core.fault_latched:
                self.core.clear_fault("web_cmd")

            if hasattr(self.backend, "getup"):
                res = self.backend.getup()
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = res.get("message", "Neural recovery getup initiated")
                ack["lifecycle"]["simulator_result"] = f"mode = {res.get('edge_mode')}, base_z = {res.get('base_z', 0.0):.4f}m"
                self.record_event(f"Command RECOVER: {res.get('message', 'neural recovery')}")
            else:
                cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                sim_t = float(self.backend.data.time)
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
                self.core.stand_settled = True
                if getattr(self.backend, "gantry_active", False):
                    self.backend.set_gantry(False)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "Recovered to STAND pose"
                self.record_event("Command RECOVER: recovered to STAND")

        # ── 1d. FALL / SAFE FALL ─────────────────────────────────────────────
        elif action in ("fall", "safefall", "safe_fall"):
            ack["lifecycle"]["parsed"] = {"action": "SAFE FALL"}
            self.core.set_active_controller("sdk")
            if hasattr(self.backend, "safe_fall"):
                res = self.backend.safe_fall()
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = res.get("message", "Safe fall policy active")
                ack["lifecycle"]["simulator_result"] = f"mode = {res.get('edge_mode')}"
                self.record_event("Command SAFE FALL executed")
            else:
                sim_t = float(self.backend.data.time)
                self.core.command_damp("sdk", current_time=sim_t)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "Compliant fall damping active"
                self.record_event("Command SAFE FALL: compliant damping engaged")

        # ── 2. WALK ──────────────────────────────────────────────────────────
        elif action == "walk":
            try:
                req_vx = float(params.get("vx", 0.0))
                req_vy = float(params.get("vy", 0.0))
                req_wz = float(params.get("wz", params.get("vyaw", 0.0)))
            except (ValueError, TypeError) as e:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"REJECTED: Invalid parameters ({e})"
                return ack

            ack["lifecycle"]["parsed"] = {"requested": {"vx": req_vx, "vy": req_vy, "wz": req_wz}}

            # Contract clipping
            cvx = float(np.clip(req_vx, VX_MIN, VX_MAX))
            cvy = float(np.clip(req_vy, VY_MIN, VY_MAX))
            cwz = float(np.clip(req_wz, WZ_MIN, WZ_MAX))

            clip_reasons = []
            if not math.isclose(cvx, req_vx, abs_tol=1e-4):
                clip_reasons.append(f"vx {req_vx:+.2f} clipped to [{VX_MIN}, {VX_MAX}]")
            if not math.isclose(cvy, req_vy, abs_tol=1e-4):
                clip_reasons.append(f"vy {req_vy:+.2f} clipped to [{VY_MIN}, {VY_MAX}]")
            if not math.isclose(cwz, req_wz, abs_tol=1e-4):
                clip_reasons.append(f"wz {req_wz:+.2f} clipped to [{WZ_MIN}, {WZ_MAX}]")

            if clip_reasons:
                ack["lifecycle"]["clipped"] = True
                ack["lifecycle"]["reason"] = "; ".join(clip_reasons)

            ack["lifecycle"]["effective"] = {"vx": cvx, "vy": cvy, "wz": cwz}

            # Apply
            sim_t = float(self.backend.data.time)
            self.core.set_active_controller("sdk")
            if self.core.fault_latched:
                self.core.clear_fault("web_cmd")

            # Auto-arm from DAMP / SETTLE into STAND if user commands velocity
            from edge.core import EdgeMode, MoveSubmode
            if self.core.mode not in (EdgeMode.STAND, EdgeMode.MOVE):
                cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
                self.core.stand_settled = True

            if self.core.mode == EdgeMode.STAND and not self.core.stand_settled:
                self.core.stand_settled = True

            # Release virtual gantry for policy walking
            if getattr(self.backend, "gantry_active", False):
                self.backend.set_gantry(False)

            ok = self.core.command_velocity(cvx, cvy, cwz, controller="sdk", current_time=sim_t)
            if ok:
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"MOVE/POLICY vx={cvx:+.2f}, vy={cvy:+.2f}, wz={cwz:+.2f}"
                ack["lifecycle"]["simulator_result"] = f"mode = {self.core.mode.name}/{self.core.move_submode.name}"
                self.record_event(f"Command WALK: vx={cvx:+.2f}, vy={cvy:+.2f}, wz={cwz:+.2f}")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"REJECTED by EdgeCore (mode={self.core.mode.name})"

        # ── 3. STOP ──────────────────────────────────────────────────────────
        elif action == "stop":
            ack["lifecycle"]["parsed"] = {"action": "STOP (zero velocity hold)"}
            self.core.set_active_controller("sdk")
            sim_t = float(self.backend.data.time)
            from edge.core import EdgeMode
            if self.core.mode == EdgeMode.MOVE:
                self.core.command_velocity(0.0, 0.0, 0.0, controller="sdk", current_time=sim_t)
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = "Zero commanded velocity"
            ack["lifecycle"]["simulator_result"] = f"mode = {self.core.mode.name}, body_vx = {self.backend.data.qvel[0]:.3f} m/s"
            self.record_event("Command STOP: velocity zeroed")

        # ── 4. RESET ─────────────────────────────────────────────────────────
        elif action == "reset":
            seed = params.get("seed", 42)
            try:
                seed = int(seed) if seed is not None else 42
            except ValueError:
                seed = 42

            ack["lifecycle"]["parsed"] = {"seed": seed}
            self.backend.reset(seed=seed)
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = f"SimBackend.reset(seed={seed})"
            ack["lifecycle"]["simulator_result"] = f"sim_time = 0.000s, base_z = {self.backend.data.qpos[2]:.4f}m, gantry = {self.backend.gantry_active}"
            self.record_event(f"Command RESET: seed={seed}, sim_time=0.000s")

        # ── 5. PAUSE / RESUME ────────────────────────────────────────────────
        elif action == "pause":
            self.backend.paused = True
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = "Simulation PAUSED"
            ack["lifecycle"]["simulator_result"] = f"paused = True at sim_time = {self.backend.data.time:.3f}s"
            self.record_event(f"Simulation PAUSED at {self.backend.data.time:.2f}s")

        elif action == "resume":
            self.backend.paused = False
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = "Simulation RESUMED"
            ack["lifecycle"]["simulator_result"] = f"paused = False at sim_time = {self.backend.data.time:.3f}s"
            self.record_event(f"Simulation RESUMED at {self.backend.data.time:.2f}s")

        # ── 6. STEP ──────────────────────────────────────────────────────────
        elif action == "step":
            count = int(params.get("n", 1))
            self.backend.paused = True
            # Step forward n control intervals (2 physics ticks each)
            with self.backend._lock:
                for _ in range(count * 2):
                    self.backend.step()
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = f"Advanced {count} control step(s)"
            ack["lifecycle"]["simulator_result"] = f"sim_time = {self.backend.data.time:.4f}s"
            self.record_event(f"Simulation STEP: {count} steps -> {self.backend.data.time:.3f}s")

        # ── 7. MODE ──────────────────────────────────────────────────────────
        elif action == "mode":
            mode_name = str(params.get("mode", "policy")).lower().strip()
            if mode_name in ("policy", "manual", "hybrid"):
                self.backend.set_control_mode(mode_name)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"Switched control mode to {mode_name.upper()}"
                ack["lifecycle"]["simulator_result"] = f"backend.control_mode = {self.backend.control_mode}"
                self.record_event(f"Control mode -> {mode_name.upper()}")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"Invalid mode '{mode_name}'. Valid: policy, manual, hybrid"

        # ── 8. JOINT TARGET ──────────────────────────────────────────────────
        elif action == "joint":
            joint = str(params.get("joint", "")).strip()
            if not joint and "index" in params:
                try:
                    idx = int(params["index"])
                    if 0 <= idx < len(SIM_JOINTS):
                        joint = SIM_JOINTS[idx]
                except ValueError:
                    pass

            if joint not in SIM_JOINTS:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"Unknown joint '{joint}'"
                return ack

            self.backend.select_joint(joint)

            if "target" in params:
                req_val = float(params["target"])
                idx = SIM_JOINTS.index(joint)
                jnt_id = self.backend.model.actuator_trnid[idx, 0]
                rmin = float(self.backend.model.jnt_range[jnt_id, 0])
                rmax = float(self.backend.model.jnt_range[jnt_id, 1])
                cval = float(np.clip(req_val, rmin, rmax))

                if not math.isclose(cval, req_val, abs_tol=1e-4):
                    ack["lifecycle"]["clipped"] = True
                    ack["lifecycle"]["reason"] = f"Target clipped to range [{rmin:.2f}, {rmax:.2f}]"

                self.backend.set_manual_joint_target(joint, cval)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"{joint} target -> {cval:.4f} rad"
                ack["lifecycle"]["simulator_result"] = f"target = {cval:.4f} rad"
                self.record_event(f"Joint {joint} -> {cval:.3f} rad")
            else:
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"Selected {joint}"

        # ── 9. ENVIRONMENT PRESET ────────────────────────────────────────────
        elif action == "environment":
            preset = str(params.get("preset", "flat")).strip()
            if preset in PRESETS:
                self.backend.load_environment(preset)
                if hasattr(self, "camera_streamer") and self.camera_streamer:
                    try:
                        self.camera_streamer._sync_model_if_changed()
                    except Exception:
                        pass
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"Loaded preset '{preset}'"
                ack["lifecycle"]["simulator_result"] = f"env = {preset}, friction = {PRESETS[preset]['ground_friction']}"
                self.record_event(f"Environment loaded: '{preset}'")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}"

        # ── 10. GANTRY ───────────────────────────────────────────────────────
        elif action == "gantry":
            state = bool(params.get("state", True))
            self.backend.set_gantry(state)
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = f"Gantry {'ENGAGED' if state else 'RELEASED'}"
            ack["lifecycle"]["simulator_result"] = f"gantry_active = {self.backend.gantry_active}"
            self.record_event(f"Virtual Gantry -> {'ENGAGED' if state else 'RELEASED'}")

        # ── 11. PUSH DISTURBANCE ─────────────────────────────────────────────
        elif action == "push":
            force = float(params.get("force", 40.0))
            direction = str(params.get("dir", "x"))
            duration = float(params.get("duration", 0.1))
            self.backend.apply_push(force, direction, duration)
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = f"Pelvis push {force}N along {direction} for {duration}s"
            ack["lifecycle"]["simulator_result"] = "Force injected into pelvis body"
            self.record_event(f"Pelvis push: {force}N along {direction} ({duration}s)")

        # ── 12. INJECT FAULT ─────────────────────────────────────────────────
        elif action == "inject":
            fault = str(params.get("fault", "")).lower()
            if fault == "fall":
                self.core.inject_fall()
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "Injected simulated fall fault"
                self.record_event("FAULT INJECTED: Simulated Fall (tilt > 60 deg)", level="warn")
            elif fault == "overtemp":
                temp = float(params.get("temp", 85.0))
                self.core.inject_overtemp(temp)
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = f"Injected overtemperature ({temp} C)"
                self.record_event(f"FAULT INJECTED: Motor Overtemp ({temp} C)", level="warn")
            elif fault == "can":
                self.core.inject_can_fault()
                ack["status"] = "accepted"
                ack["lifecycle"]["applied"] = "Injected synthetic CAN bus error"
                self.record_event("FAULT INJECTED: CAN bus error", level="warn")
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"Unknown fault type '{fault}'"

        # ── 12b. SAFE FALL ───────────────────────────────────────────────────
        elif action in ("fall", "safefall", "safe_fall"):
            res = self.backend.safe_fall()
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = "Controlled safe fall -> compliant DAMP"
            ack["lifecycle"]["simulator_result"] = f"mode = {res.get('edge_mode')}, gantry = {res.get('gantry')}"
            self.record_event("Safe fall triggered: compliant damping engaged", level="warn")

        # ── 12c. GETUP / RECOVERY ────────────────────────────────────────────
        elif action in ("getup", "recover", "standup"):
            res = self.backend.getup()
            ack["status"] = "accepted"
            ack["lifecycle"]["applied"] = "Getup recovery -> cleared faults, base restored, STAND armed"
            ack["lifecycle"]["simulator_result"] = f"base_z = {res.get('base_z'):.4f}m, mode = {res.get('edge_mode')}"
            self.record_event(f"Getup recovery complete: base_z={res.get('base_z', 0.61):.3f}m, STAND armed")

        # ── 12d. EMOTES & GESTURES ───────────────────────────────────────────
        elif action == "emote" or action in ("hello", "wave", "bow", "squat", "crouch", "cheer", "dance", "nod", "shake"):
            emote_name = action if action != "emote" else str(params.get("name", params.get("emote", "hello"))).lower().strip()
            ok = self.backend.play_emote(emote_name)
            if ok:
                ack["status"] = "accepted"
                dur = self.backend.emote_controller.active_emote.duration if self.backend.emote_controller.active_emote else 3.0
                ack["lifecycle"]["applied"] = f"Emote '{emote_name}' active ({dur:.1f}s)"
                ack["lifecycle"]["simulator_result"] = f"Playing emote animation '{emote_name}'"
                self.record_event(f"Emote triggered: '{emote_name}' ({dur:.1f}s)")
            else:
                ack["status"] = "rejected"
                from edge.gestures import EMOTE_CATALOG
                ack["lifecycle"]["validation"] = f"Unknown emote '{emote_name}'. Available: {list(EMOTE_CATALOG.keys())}"

        # ── 12e. DYNAMIC OBSTACLE SPAWN ──────────────────────────────────────
        elif action in ("spawn", "spawn_obstacle", "drop_obstacle"):
            dist = float(params.get("distance", 0.8))
            res = self.backend.spawn_dynamic_obstacle(distance=dist)
            ack["status"] = "accepted" if res.get("status") == "spawned" else "error"
            ack["lifecycle"]["applied"] = res.get("message", "Spawned obstacle")
            self.record_event(f"Dynamic obstacle: {res.get('message')}")

        # ── 13. RECORDING ────────────────────────────────────────────────────
        elif action == "record":
            sub = str(params.get("command", "start")).lower()
            if sub == "start":
                name = str(params.get("name", f"web_session_{int(time.time())}"))
                if self.telemetry:
                    self.telemetry.start_recording(name=name, metadata={"source": "web_dashboard", "start_time": time.time()})
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = f"Recording started: {name}"
                    self.record_event(f"Recording started: '{name}'")
                else:
                    ack["status"] = "rejected"
                    ack["lifecycle"]["validation"] = "Telemetry engine not available"
            elif sub == "stop":
                if self.telemetry and self.telemetry.recording_active:
                    json_p, csv_p = self.telemetry.stop_recording()
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = "Recording stopped"
                    ack["lifecycle"]["simulator_result"] = f"Saved: {json_p.name}, {csv_p.name}"
                    ack["files"] = {"json": str(json_p), "csv": str(csv_p)}
                    self.record_event(f"Recording saved: {json_p.name}")
                else:
                    ack["status"] = "rejected"
                    ack["lifecycle"]["validation"] = "No recording currently active"

        # ── 14. REPLAY ───────────────────────────────────────────────────────
        elif action == "replay":
            cmd = str(params.get("command", "load")).lower()
            if cmd == "load":
                name = str(params.get("name", "")).strip()
                if self.replay_manager and self.replay_manager.load_recording(name):
                    ack["status"] = "accepted"
                    ack["frames_count"] = len(self.replay_manager.frames)
                    ack["lifecycle"]["applied"] = f"Loaded recording '{name}' ({len(self.replay_manager.frames)} frames)"
                    ack["lifecycle"]["simulator_result"] = f"Replay ready: {len(self.replay_manager.frames)} frames"
                    self.record_event(f"Replay loaded: '{name}'")
                else:
                    ack["status"] = "rejected"
                    ack["lifecycle"]["validation"] = f"Failed to load recording '{name}'"
            elif cmd == "play":
                if self.replay_manager:
                    self.replay_manager.play()
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = "Replay playback started"
            elif cmd == "pause":
                if self.replay_manager:
                    self.replay_manager.pause()
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = "Replay playback paused"
            elif cmd == "stop":
                if self.replay_manager:
                    self.replay_manager.stop_replay()
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = "Replay playback stopped"
            elif cmd == "step":
                if self.replay_manager:
                    n = int(params.get("n", 1))
                    self.replay_manager.step(n)
                    ack["status"] = "accepted"
                    ack["lifecycle"]["applied"] = f"Replay stepped by {n}"
            else:
                ack["status"] = "rejected"
                ack["lifecycle"]["validation"] = f"Unknown replay sub-command '{cmd}'"

        else:
            ack["status"] = "rejected"
            ack["lifecycle"]["validation"] = f"Unrecognized action '{action}'"

        return ack
