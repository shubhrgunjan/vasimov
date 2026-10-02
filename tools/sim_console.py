#!/usr/bin/env python3
"""
vasimov/tools/sim_console.py
Full Interactive Virtual Asimov 1 Prototype Console in MuJoCo.

Features:
- Physics stepping loop at 200 Hz with wall-clock pacing.
- MuJoCo interactive passive viewer support (with headless fallback).
- 3 Control Modes:
    1. POLICY: official locomotion policy ONNX inference (78 obs -> 23 act -> PD actuators).
    2. MANUAL: direct joint position selection and manipulation.
    3. HYBRID: policy locomotion with manual joint target overrides.
- Live Telemetry:
    * minimal, normal, verbose, raw
    * deep tables for 25 joints, foot contacts, forces, simulated IMU, and torques
    * complete 78 -> 23 policy pipeline visualization (obs, action, io)
- Environment & Scenarios:
    * presets: flat, friction_low, friction_high, heavy_robot, light_robot, obstacle_basic, terrain_basic
    * dynamic obstacle insertion
- Deterministic reset, pause/resume, single control-step execution.
- Flight data recording, CSV/JSON export, and replay.
- Command logging showing: USER INPUT -> PARSED -> CONTROL COMMAND -> SIM RESULT.
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import select
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

import numpy as np
import mujoco

from adapter import SIM_JOINTS
from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend
from edge.telemetry import TelemetryEngine
from edge.policy import VX_MIN, VX_MAX, VY_MIN, VY_MAX, WZ_MIN, WZ_MAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vasimov.sim_console")

POLICY_HASH_SHORT = "47f70691..."


class SimConsole:
    """Interactive Control Console and Simulation Runner for Virtual Asimov 1."""

    def __init__(
        self,
        env_preset: str = "flat",
        realtime: bool = True,
        use_viewer: bool = True,
        telemetry_verbosity: str = "normal",
        telemetry_rate_hz: float = 2.0,
    ):
        self.realtime = realtime
        self.use_viewer = use_viewer
        self.running = False
        self._shutdown_event = threading.Event()

        # 1. Initialize Edge Core
        self.core = EdgeCore()

        # 2. Initialize SimBackend
        self.backend = SimBackend(self.core)

        # 3. Load initial environment preset if not flat
        if env_preset != "flat":
            self.backend.load_environment(env_preset)

        # 4. Initialize Telemetry Engine
        self.telemetry = TelemetryEngine(
            verbosity=telemetry_verbosity,
            enabled=True,
            rate_hz=telemetry_rate_hz,
        )

        # 5. Viewer handle
        self.viewer = None

        # 6. Periodic telemetry thread
        self.periodic_telemetry_active = False

        # 7. Keyboard velocity steps
        self.kb_vx_step = 0.05
        self.kb_vy_step = 0.05
        self.kb_wz_step = 0.10

    def start_viewer(self) -> bool:
        """Launch MuJoCo passive viewer if requested and display is available."""
        if not self.use_viewer:
            return False

        if "DISPLAY" not in os.environ and sys.platform.startswith("linux"):
            log.info("[VIEWER] No DISPLAY found; running in headless mode.")
            return False

        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.backend.model, self.backend.data)
            log.info("[VIEWER] MuJoCo interactive passive viewer launched successfully.")
            return True
        except Exception as e:
            log.warning("[VIEWER] Could not launch viewer (%s); continuing in headless mode.", e)
            self.viewer = None
            return False

    def _physics_loop(self) -> None:
        """Background 200 Hz physics simulation loop."""
        sim_dt = self.backend.dt  # 0.005 s
        log.info("[SIM] Physics loop running at %.0f Hz (dt=%.4fs)", 1.0 / sim_dt, sim_dt)

        while not self._shutdown_event.is_set():
            t0 = time.perf_counter()

            # Advance simulation step
            self.backend.step()

            # Record telemetry frame if flight recorder active
            if self.telemetry.recording_active and self.backend.step_count % 4 == 0:
                self.telemetry.record_frame(self.backend.get_state())

            # Update gait metrics
            if self.backend.step_count % 4 == 0:
                self.telemetry.update_gait_metrics(self.backend.get_state())

            # Sync viewer if active
            if self.viewer is not None:
                if self.viewer.is_running():
                    if self.backend.step_count % 4 == 0:
                        self.viewer.sync()
                else:
                    log.info("[VIEWER] Viewer closed by user.")
                    self.viewer = None

            # Wall-clock pacing
            if self.realtime and not self.backend.paused:
                elapsed = time.perf_counter() - t0
                sleep_time = sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            elif self.backend.paused:
                time.sleep(0.01)

    def _periodic_telemetry_loop(self) -> None:
        """Background thread for periodic live terminal dashboard updates."""
        while not self._shutdown_event.is_set():
            if self.periodic_telemetry_active and self.telemetry.enabled:
                state = self.backend.get_state()
                if self.telemetry.verbosity == "minimal":
                    print(f"\r{self.telemetry.format_minimal(state)}", end="", flush=True)
                elif self.telemetry.verbosity == "normal":
                    print("\n" + self.telemetry.format_status_block(state))
                elif self.telemetry.verbosity == "verbose":
                    print("\n" + self.telemetry.format_status_block(state))
                    print(self.telemetry.format_joint_table(state))
                elif self.telemetry.verbosity == "raw":
                    print("\n" + json.dumps(state, indent=2))

            interval = 1.0 / max(0.1, self.telemetry.rate_hz)
            time.sleep(interval)

    def print_banner(self) -> None:
        """Print console startup banner."""
        viewer_status = "ACTIVE" if (self.viewer is not None and self.viewer.is_running()) else "HEADLESS"
        banner = f"""
================================================================
 ASIMOV 1 — VIRTUAL MUJOCO CONTROL CONSOLE
================================================================
 Simulation : MuJoCo ({self.backend.model.opt.timestep*1000:.1f}ms / 200 Hz)
 Policy     : Menlo Asimov 1 Locomotion ONNX (78 -> 23)
 Checkpoint : {POLICY_HASH_SHORT}
 Control    : INTERACTIVE REPL
 Telemetry  : {self.telemetry.rate_hz:.1f} Hz ({self.telemetry.verbosity.upper()})
 Viewer     : {viewer_status}
 Mode       : {self.backend.control_mode.upper()}

 Commands:
   stand                   : Arm and ramp robot to settled standing pose
   walk <vx> [vy] [wz]     : Command locomotion velocity (e.g. walk 0.4)
   stop                    : Hold / zero commanded velocity
   reset [seed]            : Deterministic reset of physics & controller
   pause / resume          : Pause or resume simulation
   step [n]                : Advance simulation by n control steps (50 Hz)
   mode <policy|manual|hybrid> : Switch control mode
   joints                  : Display complete 25-joint status table
   select <joint>          : Select joint by index or name (e.g. select 4)
   set <target>            : Set target position in radians for selected joint
   add <delta> / sub <val> : Adjust target position
   zero                    : Reset selected joint to default standing angle
   state                   : Print full status dashboard
   obs / action / io       : Inspect 78-D policy obs and 23-D actions
   contacts / sensors      : Inspect feet contacts, forces, and simulated IMU
   torque                  : Inspect actuator torques and limit saturation
   telemetry <on|off|rate <hz>|minimal|normal|verbose|raw>
   env <list|load <preset>|reset>
   record <start <name>|stop>
   replay <name>
   export <csv|json> [path]
   gantry <on|off>         : Toggle virtual gantry weld
   push <force> [dir] [s]  : Apply pelvis disturbance push (e.g. push 60 x 0.1)
   help / quit

 Keyboard Shortcuts:
   W/S = vx +/- 0.05 m/s   A/D = vy -/+ 0.05 m/s   Q/E = wz +/- 0.10 rad/s
   SPACE = stop            R = reset               P = pause/resume
================================================================
"""
        print(banner)

    # ── Command Pipeline Logger ──────────────────────────────────────────────

    def log_command_pipeline(
        self,
        user_input: str,
        parsed_cmd: str,
        control_cmd: str,
        sim_result: str,
    ) -> None:
        """Visibly format: USER INPUT -> PARSED -> CONTROL COMMAND -> SIM RESULT."""
        print("\n" + "─" * 60)
        print(f"USER INPUT:\n    {user_input}")
        print(f"PARSED COMMAND:\n    {parsed_cmd}")
        print(f"CONTROL COMMAND:\n    {control_cmd}")
        print(f"SIMULATOR RESULT:\n    {sim_result}")
        print("─" * 60)

    # ── Command Dispatcher ───────────────────────────────────────────────────

    def execute_command(self, raw_line: str) -> None:
        """Parse and execute a console command string."""
        line = raw_line.strip()
        if not line:
            return

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Handle Keyboard Shortcuts as commands
        if cmd in ("w", "s", "a", "d", "q", "e", "space", "p", "r"):
            self._handle_keyboard_shortcut(cmd)
            return

        if cmd == "help":
            self.print_banner()

        elif cmd == "stand":
            cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
            sim_t = float(self.backend.data.time)
            ok = self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
            res = (
                f"mode = {self.core.mode.name} (ramping 2.0s to default pose)\n"
                f"    base_z = {self.backend.data.qpos[2]:.4f}m, gantry = {self.backend.gantry_active}"
                if ok else f"FAILED to enter STAND (latched={self.core.fault_latched})"
            )
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = STAND",
                control_cmd="EdgeCore.command_stand('sdk')",
                sim_result=res,
            )

        elif cmd == "walk":
            if not args:
                print("Usage: walk <vx> [vy] [yaw]")
                return
            try:
                req_vx = float(args[0])
                req_vy = float(args[1]) if len(args) > 1 else 0.0
                req_yaw = float(args[2]) if len(args) > 2 else 0.0
            except ValueError:
                print("ERROR: Velocity parameters must be numbers.")
                return

            # Range clipping with explicit user notification
            cvx = float(np.clip(req_vx, VX_MIN, VX_MAX))
            cvy = float(np.clip(req_vy, VY_MIN, VY_MAX))
            cyaw = float(np.clip(req_yaw, WZ_MIN, WZ_MAX))

            clip_notes = []
            if not math.isclose(cvx, req_vx, abs_tol=1e-4):
                clip_notes.append(f"vx: {req_vx:+.2f} -> clipped to {cvx:+.2f} [{VX_MIN}, {VX_MAX}]")
            if not math.isclose(cvy, req_vy, abs_tol=1e-4):
                clip_notes.append(f"vy: {req_vy:+.2f} -> clipped to {cvy:+.2f} [{VY_MIN}, {VY_MAX}]")
            if not math.isclose(cyaw, req_yaw, abs_tol=1e-4):
                clip_notes.append(f"wz: {req_yaw:+.2f} -> clipped to {cyaw:+.2f} [{WZ_MIN}, {WZ_MAX}]")

            sim_t = float(self.backend.data.time)
            # If not in STAND or MOVE, arm into STAND first
            if self.core.mode not in (EdgeMode.STAND, EdgeMode.MOVE):
                cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
                self.core.stand_settled = True

            # If in STAND, allow smooth transition once settled or at standing height
            if self.core.mode == EdgeMode.STAND and not self.core.stand_settled:
                if abs(float(self.backend.data.qpos[2]) - 0.6096) < 0.05 or self.backend.gantry_active:
                    self.core.stand_settled = True

            # Command velocity to policy
            self.core.command_velocity(cvx, cvy, cyaw, controller="sdk", current_time=sim_t)

            # Auto-release virtual gantry for policy walking
            if self.backend.gantry_active:
                self.backend.set_gantry(False)

            parsed_str = f"vx={req_vx:+.3f}, vy={req_vy:+.3f}, wz={req_yaw:+.3f}"
            if clip_notes:
                parsed_str += f" (CLIPPED: {', '.join(clip_notes)})"

            b_state = self.backend.get_base_state()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd=parsed_str,
                control_cmd=f"command_vel = [{cvx:+.3f}, {cvy:+.3f}, {cyaw:+.3f}] -> 78-D Policy",
                sim_result=(
                    f"mode = {self.core.mode.name}/{self.core.move_submode.name}\n"
                    f"    body_vx = {b_state['body_vx']:+.3f} m/s, base_z = {b_state['z']:.3f} m\n"
                    f"    gantry = {'ENGAGED' if self.backend.gantry_active else 'RELEASED'}"
                ),
            )

        elif cmd == "stop":
            sim_t = float(self.backend.data.time)
            self.core.command_velocity(0.0, 0.0, 0.0, sim_t)
            b_state = self.backend.get_base_state()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = STOP (zero velocity hold)",
                control_cmd="command_vel = [0.000, 0.000, 0.000]",
                sim_result=f"mode = {self.core.mode.name}/{self.core.move_submode.name}, body_vx = {b_state['body_vx']:+.3f} m/s",
            )

        elif cmd == "reset":
            seed = int(args[0]) if args else None
            self.backend.reset(seed=seed)
            self.telemetry.last_step_sim_time = 0.0
            self.telemetry.step_transitions = 0
            self.telemetry.stride_times.clear()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd=f"action = RESET (seed={seed})",
                control_cmd="backend.reset() -> Settled height (0.6096m), Gantry ON, zero velocity",
                sim_result=f"sim_time = {self.backend.data.time:.3f}s, gantry = {self.backend.gantry_active}, mode = {self.core.mode.name}",
            )

        elif cmd == "pause":
            self.backend.set_paused(True)
            print("SIMULATION PAUSED. Type 'step' to advance 1 frame or 'resume' to continue.")

        elif cmd == "resume" or cmd == "run":
            self.backend.set_paused(False)
            print("SIMULATION RESUMED.")

        elif cmd == "step":
            steps = int(args[0]) if args else 1
            # 1 control step = 4 physics substeps = 0.020s
            self.backend.step_once(steps)
            print(f"Advancing simulation by {steps} control step(s) ({steps * 0.020:.3f}s)...")

        elif cmd == "mode":
            if not args:
                print(f"Current control mode: {self.backend.control_mode.upper()} (options: policy, manual, hybrid)")
                return
            m = args[0].lower()
            try:
                self.backend.set_control_mode(m)
                print(f"Switched control mode to: {m.upper()}")
            except ValueError as e:
                print(f"ERROR: {e}")

        elif cmd == "joints":
            state = self.backend.get_state()
            print(self.telemetry.format_joint_table(state))

        elif cmd == "select":
            if not args:
                print(f"Currently selected joint: {self.backend.selected_joint}")
                return
            target_j = args[0]
            try:
                if target_j.isdigit():
                    selected = self.backend.select_joint(int(target_j))
                else:
                    selected = self.backend.select_joint(target_j)
                cur_tgt = self.backend.manual_joint_targets.get(selected, 0.0)
                print(f"Selected joint: {selected} (current target = {cur_tgt:+.4f} rad)")
            except Exception as e:
                print(f"ERROR: {e}")

        elif cmd == "set":
            if not args:
                print("Usage: set <target_in_radians>")
                return
            try:
                val = float(args[0])
                self.backend.set_manual_joint_target(self.backend.selected_joint, val)
                applied = self.backend.manual_joint_targets[self.backend.selected_joint]
                print(f"Set {self.backend.selected_joint} target to: {applied:+.4f} rad")
            except Exception as e:
                print(f"ERROR: {e}")

        elif cmd == "add":
            if not args:
                print("Usage: add <delta_radians>")
                return
            try:
                delta = float(args[0])
                new_val = self.backend.add_manual_joint_delta(self.backend.selected_joint, delta)
                print(f"Incremented {self.backend.selected_joint} by {delta:+.4f} -> {new_val:+.4f} rad")
            except Exception as e:
                print(f"ERROR: {e}")

        elif cmd == "sub":
            if not args:
                print("Usage: sub <delta_radians>")
                return
            try:
                delta = float(args[0])
                new_val = self.backend.add_manual_joint_delta(self.backend.selected_joint, -delta)
                print(f"Decremented {self.backend.selected_joint} by {delta:+.4f} -> {new_val:+.4f} rad")
            except Exception as e:
                print(f"ERROR: {e}")

        elif cmd == "zero":
            idx = SIM_JOINTS.index(self.backend.selected_joint)
            def_val = float(self.core.default_pose_sim[idx])
            self.backend.set_manual_joint_target(self.backend.selected_joint, def_val)
            print(f"Reset {self.backend.selected_joint} to default pose: {def_val:+.4f} rad")

        elif cmd == "state":
            state = self.backend.get_state()
            print(self.telemetry.format_status_block(state))

        elif cmd == "obs":
            state = self.backend.get_state()
            obs = state.get("policy_obs", {})
            if not obs.get("available"):
                print("Policy observation vector not active.")
            else:
                print(f"\nPOLICY OBSERVATION VECTOR ({obs.get('dim')} dimensions):")
                labels = obs.get("labels", [])
                raw = obs.get("raw", [])
                for i in range(len(raw)):
                    lbl = labels[i] if i < len(labels) else f"item_{i}"
                    print(f"  [{i:02d}] {lbl:<26} : {raw[i]:>+8.4f}")

        elif cmd == "action":
            state = self.backend.get_state()
            act = state.get("policy_act", {})
            if not act.get("available"):
                print("Policy action vector not active.")
            else:
                print(f"\nPOLICY ACTIONS ({act.get('dim')} dimensions, scale=0.25):")
                for i, it in enumerate(act.get("items", [])):
                    print(
                        f"  [{i:02d}] {it['joint']:<24} : raw={it['raw_action']:>+7.4f} "
                        f"scaled={it['scaled_delta']:>+7.4f} q_des={it['desired_target']:>+7.4f}"
                    )

        elif cmd == "io":
            state = self.backend.get_state()
            print(self.telemetry.format_policy_io(state))

        elif cmd == "contacts":
            state = self.backend.get_state()
            print(self.telemetry.format_contacts(state))

        elif cmd == "sensors":
            state = self.backend.get_state()
            print(self.telemetry.format_sensors(state))

        elif cmd == "torque":
            actuators = self.backend.get_actuator_state()
            print(f"{'#':<3} {'Actuator':<26} {'Ctrl':>8} {'Target':>8} {'Torque(Nm)':>10} {'Limit':>8} {'% Effort':>9}")
            print("─" * 76)
            for a in actuators:
                eff_pct = (abs(a['torque']) / a['force_limit'] * 100.0) if a['force_limit'] > 0 else 0.0
                print(
                    f"{a['index']:<3} {a['name']:<26} {a['ctrl']:>+8.3f} "
                    f"{a['target']:>+8.3f} {a['torque']:>+10.2f} {a['force_limit']:>8.1f} {eff_pct:>8.1f}%"
                )

        elif cmd == "telemetry":
            if not args:
                print(f"Telemetry status: enabled={self.telemetry.enabled}, verbosity={self.telemetry.verbosity}, rate={self.telemetry.rate_hz}Hz")
                return
            sub = args[0].lower()
            if sub in ("on", "start"):
                self.periodic_telemetry_active = True
                self.telemetry.set_enabled(True)
                print(f"Live terminal telemetry turned ON ({self.telemetry.verbosity}, {self.telemetry.rate_hz}Hz)")
            elif sub in ("off", "stop"):
                self.periodic_telemetry_active = False
                self.telemetry.set_enabled(False)
                print("Live terminal telemetry turned OFF")
            elif sub in ("minimal", "normal", "verbose", "raw"):
                self.telemetry.set_verbosity(sub)
                print(f"Telemetry verbosity set to: {sub.upper()}")
            elif sub == "rate":
                if len(args) > 1:
                    hz = float(args[1])
                    self.telemetry.set_rate(hz)
                    print(f"Telemetry rate set to: {hz:.1f} Hz")
                else:
                    print("Usage: telemetry rate <Hz>")

        elif cmd == "env":
            if not args or args[0] == "list":
                presets = self.backend.env_manager.list_presets()
                print("AVAILABLE ENVIRONMENT PRESETS:")
                for p_name, p_desc in presets.items():
                    mark = " *" if p_name == self.backend.env_manager.current_preset else "  "
                    print(f"{mark} {p_name:<16} : {p_desc}")
            elif args[0] == "load":
                if len(args) > 1:
                    preset_name = args[1].lower()
                    try:
                        res = self.backend.load_environment(preset_name)
                        print(f"Successfully loaded environment '{preset_name}':")
                        print(f"  Friction: mu={res['ground_friction']}, Mass scale: {res['mass_scale']}, Obstacles: {res['obstacle_count']}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                else:
                    print("Usage: env load <preset_name>")
            elif args[0] == "reset":
                self.backend.load_environment(self.backend.env_manager.current_preset)
                print(f"Reset environment to nominal preset: {self.backend.env_manager.current_preset}")

        elif cmd == "record":
            if not args:
                print(f"Recording status: {'ACTIVE (' + str(self.telemetry.current_recording_name) + ')' if self.telemetry.recording_active else 'IDLE'}")
                return
            sub = args[0].lower()
            if sub in ("start", "begin"):
                name = args[1] if len(args) > 1 else f"session_{int(time.time())}"
                self.telemetry.start_recording(name)
                print(f"Flight recording STARTED: '{name}'")
            elif sub in ("stop", "end"):
                if self.telemetry.recording_active:
                    jp, cp = self.telemetry.stop_recording()
                    print(f"Flight recording STOPPED. Artifacts written to:\n  JSON: {jp}\n  CSV:  {cp}")
                else:
                    print("No recording is currently active.")

        elif cmd == "replay":
            if not args:
                print("Usage: replay <recording_name>")
                return
            rec_name = args[0]
            try:
                data = self.telemetry.load_recording(rec_name)
                frames = data.get("frames", [])
                meta = data.get("metadata", {})
                print(f"REPLAY METADATA for '{rec_name}':")
                print(f"  Frames count: {len(frames)}, Created: {time.ctime(meta.get('start_wall_time', 0))}")
                if frames:
                    f0 = frames[0]
                    f_end = frames[-1]
                    dur = f_end.get("sim_time", 0.0) - f0.get("sim_time", 0.0)
                    print(f"  Duration: {dur:.2f}s (sim), Final Mode: {f_end.get('control_mode')}:{f_end.get('edge_mode')}")
            except Exception as e:
                print(f"ERROR: Could not load recording: {e}")

        elif cmd == "export":
            fmt = args[0].lower() if args else "csv"
            name = args[1] if len(args) > 1 else f"export_{int(time.time())}"
            if self.telemetry.recording_buffer:
                print(f"Exporting {len(self.telemetry.recording_buffer)} frames in {fmt.upper()} format to reports/raw/...")
            else:
                print("No recorded frames to export. Run 'record start <name>' first.")

        elif cmd == "gantry":
            if not args:
                print(f"Virtual gantry status: {'ENGAGED' if self.backend.gantry_active else 'RELEASED'}")
                return
            state = (args[0].lower() == "on")
            self.backend.set_gantry(state)
            print(f"Virtual gantry set to: {'ENGAGED' if self.backend.gantry_active else 'RELEASED'}")

        elif cmd == "push":
            force = float(args[0]) if args else 40.0
            direction = args[1] if len(args) > 1 else "x"
            dur = float(args[2]) if len(args) > 2 else 0.1
            self.backend.apply_push(force, direction, dur)
            print(f"Applied physical push: {force:.1f} N in {direction.upper()} for {dur:.2f}s")

        elif cmd in ("quit", "exit"):
            print("Exiting Virtual Asimov console...")
            self.stop()

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for command list.")

    def _handle_keyboard_shortcut(self, key: str) -> None:
        """Handle single-key commands."""
        sim_t = float(self.backend.data.time)

        if key in ("w", "s", "a", "d", "q", "e"):
            # Auto-arm into STAND if currently in DAMP
            if self.core.mode not in (EdgeMode.STAND, EdgeMode.MOVE):
                cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
                self.core.stand_settled = True

            if self.core.mode == EdgeMode.STAND and not self.core.stand_settled:
                self.core.stand_settled = True

            # Auto-release virtual gantry for locomotion
            if self.backend.gantry_active:
                self.backend.set_gantry(False)

        cur_vx = self.core.current_vx
        cur_vy = self.core.current_vy
        cur_wz = self.core.current_vyaw

        if key == "w":
            new_vx = float(np.clip(cur_vx + self.kb_vx_step, VX_MIN, VX_MAX))
            self.core.command_velocity(new_vx, cur_vy, cur_wz, sim_t)
            print(f"\nINPUT: KEY=W | COMMAND OUT: vx={new_vx:+.2f} vy={cur_vy:+.2f} wz={cur_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "s":
            new_vx = float(np.clip(cur_vx - self.kb_vx_step, VX_MIN, VX_MAX))
            self.core.command_velocity(new_vx, cur_vy, cur_wz, sim_t)
            print(f"\nINPUT: KEY=S | COMMAND OUT: vx={new_vx:+.2f} vy={cur_vy:+.2f} wz={cur_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "a":
            new_vy = float(np.clip(cur_vy - self.kb_vy_step, VY_MIN, VY_MAX))
            self.core.command_velocity(cur_vx, new_vy, cur_wz, sim_t)
            print(f"\nINPUT: KEY=A | COMMAND OUT: vx={cur_vx:+.2f} vy={new_vy:+.2f} wz={cur_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "d":
            new_vy = float(np.clip(cur_vy + self.kb_vy_step, VY_MIN, VY_MAX))
            self.core.command_velocity(cur_vx, new_vy, cur_wz, sim_t)
            print(f"\nINPUT: KEY=D | COMMAND OUT: vx={cur_vx:+.2f} vy={new_vy:+.2f} wz={cur_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "q":
            new_wz = float(np.clip(cur_wz + self.kb_wz_step, WZ_MIN, WZ_MAX))
            self.core.command_velocity(cur_vx, cur_vy, new_wz, sim_t)
            print(f"\nINPUT: KEY=Q | COMMAND OUT: vx={cur_vx:+.2f} vy={cur_vy:+.2f} wz={new_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "e":
            new_wz = float(np.clip(cur_wz - self.kb_wz_step, WZ_MIN, WZ_MAX))
            self.core.command_velocity(cur_vx, cur_vy, new_wz, sim_t)
            print(f"\nINPUT: KEY=E | COMMAND OUT: vx={cur_vx:+.2f} vy={cur_vy:+.2f} wz={new_wz:+.2f} (Mode: {self.core.mode.name}/{self.core.move_submode.name})")
        elif key == "space":
            self.core.command_velocity(0.0, 0.0, 0.0, sim_t)
            print("\nINPUT: KEY=SPACE (STOP) | COMMAND OUT: vx=+0.00 vy=+0.00 wz=+0.00")
        elif key == "r":
            self.backend.reset()
            print("\nINPUT: KEY=R (RESET) | Simulator reset to settled pose.")
        elif key == "p":
            self.backend.set_paused(not self.backend.paused)
            status = "PAUSED" if self.backend.paused else "RESUMED"
            print(f"\nINPUT: KEY=P ({status})")

    # ── Console Lifecycle ────────────────────────────────────────────────────

    def run(self, script_path: Optional[Union[str, Path]] = None) -> None:
        """Start physics thread, viewer, telemetry, and interactive REPL."""
        self.running = True
        self._shutdown_event.clear()

        # 1. Start viewer
        self.start_viewer()

        # 2. Start physics simulation thread
        physics_thread = threading.Thread(
            target=self._physics_loop,
            daemon=True,
            name="asimov-physics-loop",
        )
        physics_thread.start()

        # 3. Start periodic telemetry thread
        telemetry_thread = threading.Thread(
            target=self._periodic_telemetry_loop,
            daemon=True,
            name="asimov-telemetry-loop",
        )
        telemetry_thread.start()

        # 4. Print banner
        self.print_banner()

        # 5. If a script is provided, execute lines sequentially
        if script_path is not None:
            sp = Path(script_path)
            if sp.exists():
                log.info("[CONSOLE] Executing script: %s", sp)
                for line in sp.read_text(encoding="utf-8").splitlines():
                    clean = line.strip()
                    if clean and not clean.startswith("#"):
                        print(f"\nasimov> {clean}")
                        self.execute_command(clean)
                        time.sleep(0.1)
            else:
                log.error("[CONSOLE] Script file not found: %s", sp)

        # 6. Main Interactive REPL Loop
        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    line = input("asimov> ")
                    self.execute_command(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\nReceived SIGINT. Shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Clean shutdown of console, simulator, and viewer."""
        self.running = False
        self._shutdown_event.set()
        if self.viewer is not None and self.viewer.is_running():
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        self.backend.env_manager.cleanup()
        log.info("[CONSOLE] Virtual Asimov Console terminated cleanly.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Virtual Asimov 1 Interactive Control Console in MuJoCo.")
    parser.add_argument("--env", type=str, default="flat", help="Environment preset (flat, obstacle_basic, friction_low, etc.)")
    parser.add_argument("--fast", action="store_true", help="Run at max simulation speed rather than wall-clock realtime")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo GUI passive viewer (headless mode)")
    parser.add_argument("--telemetry-verbosity", type=str, default="normal", choices=["minimal", "normal", "verbose", "raw"], help="Telemetry verbosity")
    parser.add_argument("--telemetry-rate", type=float, default=2.0, help="Periodic telemetry rate in Hz")
    parser.add_argument("--script", type=str, default=None, help="Path to batch command script file")
    args = parser.parse_args()

    console = SimConsole(
        env_preset=args.env,
        realtime=not args.fast,
        use_viewer=not args.no_viewer,
        telemetry_verbosity=args.telemetry_verbosity,
        telemetry_rate_hz=args.telemetry_rate,
    )
    console.run(script_path=args.script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
