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

if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
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
from edge.gestures import EmoteController, EMOTE_CATALOG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vasimov.sim_console")

POLICY_HASH_SHORT = "47f70691..."

# ── Terminal Styling & ANSI Badges ──────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[1;36m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_BLUE = "\033[1;34m"
C_MAGENTA = "\033[1;35m"
C_RED = "\033[1;31m"
C_WHITE = "\033[1;37m"
C_GRAY = "\033[90m"

COMMANDS_LIST = [
    "drive", "game", "teleop", "live", "spawn",
    "stand", "walk", "stop", "fall", "safefall", "getup", "recover", "standup",
    "hello", "wave", "bow", "squat", "crouch", "cheer", "dance", "nod", "shake", "emote",
    "policy",
    "reset", "pause", "resume", "step", "mode",
    "joints", "select", "set", "add", "sub", "zero",
    "state", "obs", "action", "io", "contacts", "sensors", "torque", "telemetry",
    "camera", "cam", "env", "record", "replay", "export", "gantry", "push", "help", "quit", "exit",
]

SUBCOMMANDS = {
    "mode": ["policy", "manual", "hybrid"],
    "policy": ["list", "select", "info", "validate", "active"],
    "telemetry": ["on", "off", "minimal", "normal", "verbose", "raw", "rate"],
    "camera": ["list", "chase", "fpv", "fpv_behind", "front", "side", "back", "free", "track"],
    "cam": ["list", "chase", "fpv", "fpv_behind", "front", "side", "back", "free", "track"],
    "env": ["list", "load", "reset"],
    "env load": [
        "flat", "obstacles", "playground", "physics_obstacles",
        "friction_low", "friction_high", "heavy_robot", "light_robot",
        "obstacle_basic", "terrain_basic", "corridor", "arena", "obstacle_course",
    ],
    "record": ["start", "stop"],
    "gantry": ["on", "off"],
    "emote": ["hello", "wave", "bow", "squat", "crouch", "cheer", "dance", "nod", "shake"],
}



def setup_readline() -> None:
    """Configure readline tab auto-completion for console commands."""
    try:
        import readline
        def completer(text: str, state: int) -> Optional[str]:
            buffer = readline.get_line_buffer()
            tokens = buffer.split()
            if not tokens or (len(tokens) == 1 and not buffer.endswith(" ")):
                matches = [c for c in COMMANDS_LIST if c.startswith(text)]
            else:
                cmd = tokens[0].lower()
                if cmd in SUBCOMMANDS and (len(tokens) == 1 or (len(tokens) == 2 and not buffer.endswith(" "))):
                    matches = [s for s in SUBCOMMANDS[cmd] if s.startswith(text)]
                elif cmd == "select":
                    matches = [j for j in SIM_JOINTS if j.startswith(text)]
                elif cmd == "env" and len(tokens) >= 2 and tokens[1].lower() == "load":
                    matches = [p for p in SUBCOMMANDS["env load"] if p.startswith(text)]
                else:
                    matches = []
            if state < len(matches):
                return matches[state]
            return None

        readline.set_completer(completer)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


class _RawTerminalMode:
    """Cross-platform context manager for character-by-character raw terminal mode."""
    def __init__(self):
        self.is_posix = sys.platform != "win32" and sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.is_posix:
            import termios
            import tty
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                self.old_settings = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_posix and self.old_settings is not None:
            import termios
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass


def _read_key_nonblocking() -> Optional[str]:
    """Read a single key or escape sequence without blocking (cross-platform)."""
    if sys.platform == "win32":
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    ch2 = msvcrt.getch()
                    arrow_map = {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}
                    return arrow_map.get(ch2, None)
                if ch == b"\x1b":
                    return "esc"
                try:
                    return ch.decode("utf-8", errors="ignore").lower()
                except Exception:
                    return None
        except Exception:
            return None
    else:
        if not sys.stdin.isatty():
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r2:
                    rest = sys.stdin.read(2)
                    if rest == "[A":
                        return "up"
                    elif rest == "[B":
                        return "down"
                    elif rest == "[C":
                        return "right"
                    elif rest == "[D":
                        return "left"
                return "esc"
            return ch.lower()
    return None


class SimConsole:
    """Interactive Control Console and Simulation Runner for Virtual Asimov 1."""

    def __init__(
        self,
        env_preset: str = "flat",
        realtime: bool = True,
        use_viewer: bool = True,
        telemetry_verbosity: str = "normal",
        telemetry_rate_hz: float = 2.0,
        enable_web: bool = False,
        web_host: str = "0.0.0.0",
        web_port: int = 8852,
        ws_port: int = 8854,
        policy_name: Optional[str] = None,
        locomotion_policy: Optional[str] = None,
        balance_policy: Optional[str] = None,
        recovery_policy: Optional[str] = None,
        debug_policy: bool = False,
        camera: str = "chase",
    ):
        self.realtime = realtime
        self.use_viewer = use_viewer
        self.enable_web = enable_web
        self.web_host = web_host
        self.web_port = web_port
        self.ws_port = ws_port
        self.debug_policy = debug_policy
        self.camera_name = camera
        self.active_camera = camera
        self.running = False
        self._shutdown_event = threading.Event()

        # 1. Initialize Edge Core
        self.core = EdgeCore()

        # 2. Initialize SimBackend
        self.backend = SimBackend(self.core)

        # 2b. Pluggable Policy configuration
        if policy_name:
            self.backend.load_policy(policy_name)
        if locomotion_policy or balance_policy or recovery_policy:
            self.backend.policy_manager.auto_compose = True
            if locomotion_policy:
                self.backend.policy_manager.register_policy(
                    self.backend.policy_registry.instantiate(locomotion_policy),
                    category="locomotion"
                )
            if balance_policy:
                self.backend.policy_manager.register_policy(
                    self.backend.policy_registry.instantiate(balance_policy),
                    category="balance"
                )
            if recovery_policy:
                if recovery_policy in ("openhorizon", "openhorizon_recovery", "getup_safefall"):
                    self.backend.policy_manager.configure_multi_policy(
                        locomotion=locomotion_policy or "official_locomotion",
                        recovery="openhorizon_recovery",
                        auto_compose=True,
                    )
                else:
                    self.backend.policy_manager.register_policy(
                        self.backend.policy_registry.instantiate(recovery_policy),
                        category="recovery"
                    )
            self.core.recovery_mode_active = True
            self.core.fall_latch_enabled = False

        # 3. Load initial environment preset if not flat
        if env_preset != "flat":
            self.backend.load_environment(env_preset)

        # 4. Initialize Telemetry Engine
        self.telemetry = TelemetryEngine(
            verbosity=telemetry_verbosity,
            enabled=True,
            rate_hz=telemetry_rate_hz,
        )

        # 5. Optional Web Dashboard Gateway
        self.gateway = None
        if self.enable_web:
            try:
                from web.server.gateway import WebGateway
                self.gateway = WebGateway(
                    backend=self.backend,
                    core=self.core,
                    telemetry_engine=self.telemetry,
                    http_port=self.web_port,
                    ws_port=self.ws_port,
                    host=self.web_host,
                )
                self.gateway.start()
            except Exception as e:
                log.error("[CONSOLE] Could not start WebGateway: %s", e)

        # 6. Viewer handle
        self.viewer = None

        # 7. Periodic telemetry thread
        self.periodic_telemetry_active = False

        # 8. Keyboard velocity steps
        self.kb_vx_step = 0.05
        self.kb_vy_step = 0.05
        self.kb_wz_step = 0.10

        self.physics_thread: Optional[threading.Thread] = None
        self.telemetry_thread: Optional[threading.Thread] = None

    CAMERA_ALIASES: Dict[str, str] = {
        "chase": "chase_camera",
        "chase_camera": "chase_camera",
        "behind": "chase_camera",
        "fpv_behind": "first_person_behind",
        "first_person_behind": "first_person_behind",
        "follow": "first_person_behind",
        "fpv": "first_person_camera",
        "first_person": "first_person_camera",
        "eyes": "first_person_camera",
        "head": "first_person_camera",
        "front": "front_camera",
        "front_camera": "front_camera",
        "side": "side_camera",
        "side_camera": "side_camera",
        "back": "back_camera",
        "back_camera": "back_camera",
        "direct_side": "direct_side_camera",
        "direct_behind": "direct_behind_camera",
        "free": "free",
        "track": "track",
    }

    def get_available_cameras(self) -> List[str]:
        """Return list of camera names available in current MuJoCo model."""
        cams = []
        if hasattr(self.backend, "model") and self.backend.model is not None:
            for i in range(self.backend.model.ncam):
                name = mujoco.mj_id2name(self.backend.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    cams.append(name)
        return cams

    def set_camera(self, camera_name: str) -> Tuple[bool, str]:
        """
        Configure the 3D passive viewer camera view.
        Supports:
          - 'chase': 3rd person follow camera behind robot (turns and moves with pelvis)
          - 'fpv_behind': close follow camera behind robot
          - 'fpv': head eye-level first-person view
          - 'front', 'side', 'back': tracking perspectives
          - 'free': orbit free camera
          - 'track': center-tracking robot floating base
        """
        raw = camera_name.strip().lower()
        resolved = self.CAMERA_ALIASES.get(raw, raw)
        self.active_camera = resolved

        if self.viewer is None:
            return True, f"Active camera set to '{resolved}' (will apply when viewer opens)"

        try:
            if resolved == "free":
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                msg = "Viewer camera switched to FREE orbit camera"
            elif resolved == "track":
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = self.backend.pelvis_body_id
                msg = "Viewer camera switched to TRACKING robot base"
            else:
                cam_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_CAMERA, resolved)
                if cam_id != -1:
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = cam_id
                    msg = f"Viewer camera switched to '{resolved}'"
                else:
                    return False, f"Camera '{resolved}' not found in MuJoCo model. Available: {self.get_available_cameras()} + ['free', 'track']"

            if hasattr(self.viewer, "sync"):
                self.viewer.sync()
            return True, msg
        except Exception as e:
            return False, f"Failed to set camera: {e}"

    def cycle_camera(self) -> str:
        """Cycle through primary camera views (Chase ➔ FPV Behind ➔ FPV Head ➔ Free Orbit)."""
        cycle_order = ["chase_camera", "first_person_behind", "first_person_camera", "free"]
        cur = self.CAMERA_ALIASES.get(self.active_camera, self.active_camera)
        try:
            idx = cycle_order.index(cur)
            next_cam = cycle_order[(idx + 1) % len(cycle_order)]
        except ValueError:
            next_cam = cycle_order[0]
        self.set_camera(next_cam)
        return next_cam

    def start_viewer(self, timeout: float = 3.0) -> bool:
        """Launch MuJoCo passive viewer if requested and display is available."""
        if not self.use_viewer:
            return False

        if "DISPLAY" not in os.environ and sys.platform.startswith("linux"):
            log.info("[VIEWER] No DISPLAY found; running in headless mode.")
            return False

        deadline = time.time() + timeout
        while True:
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.backend.model, self.backend.data)
                log.info("[VIEWER] MuJoCo interactive passive viewer launched successfully.")
                self.set_camera(self.active_camera)
                return True
            except Exception as e:
                err_str = str(e).lower()
                # On macOS Cocoa, previous window teardown is asynchronous; retry until UI thread releases viewer
                if "another mujoco viewer is already open" in err_str and time.time() < deadline:
                    time.sleep(0.08)
                    continue
                log.warning("[VIEWER] Could not launch viewer (%s); continuing in headless mode.", e)
                if sys.platform == "darwin" and "mjpython" in err_str:
                    log.info("[VIEWER] On macOS, launch via './run_sim.sh' or '.venv/bin/mjpython' for native Cocoa 3D GUI.")
                self.viewer = None
                return False

    def restart_viewer(self, timeout: float = 3.0) -> bool:
        """Safely close existing viewer, wait for native window teardown, and relaunch with updated model."""
        if not self.use_viewer:
            return False

        if self.viewer is not None:
            old_viewer = self.viewer
            self.viewer = None
            try:
                old_viewer.close()
            except Exception:
                pass
            # Wait for previous viewer to finish closing on UI thread
            deadline = time.time() + 1.5
            while time.time() < deadline:
                try:
                    if not old_viewer.is_running():
                        break
                except Exception:
                    break
                time.sleep(0.05)
            # Brief settling pause for Cocoa window cleanup on macOS
            time.sleep(0.15)

        return self.start_viewer(timeout=timeout)

    def reload_viewer_model(self, gen_path: Optional[str] = None) -> bool:
        """Dynamically update existing viewer window with new model/data in-place."""
        if not self.use_viewer:
            return False
        if self.viewer is None or not self.viewer.is_running():
            return self.start_viewer()
        try:
            sim = self.viewer._get_sim()
            if sim is not None and hasattr(sim, "load"):
                model_path = str(gen_path or getattr(self.backend.env_manager, "generated_model_path", ""))
                # Note: Simulate::Load acquires its non-recursive mutex internally in C++.
                # Do NOT wrap sim.load in self.viewer.lock() as that causes self-deadlock on the UI thread.
                sim.load(self.backend.model, self.backend.data, model_path)
                if hasattr(sim, "load_message_clear"):
                    sim.load_message_clear()
                # Update camera tracking body to pelvis_link in new model
                pelvis_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
                if pelvis_id != -1 and self.viewer.cam is not None:
                    self.viewer.cam.trackbodyid = pelvis_id
                self.set_camera(self.active_camera)
                self.viewer.sync()
                log.info("[VIEWER] Live viewer model updated to '%s' in-place.", model_path)
                return True
        except Exception as e:
            log.warning("[VIEWER] Dynamic in-place model reload failed: %s; attempting restart.", e)
        return self.restart_viewer()

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
                try:
                    if self.viewer.is_running():
                        if self.backend.step_count % 4 == 0:
                            self.viewer.sync()
                    else:
                        log.info("[VIEWER] Viewer closed by user.")
                        self.viewer = None
                except Exception:
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

    def _get_prompt(self) -> str:
        """Construct interactive status prompt with ANSI indicators."""
        try:
            ctrl_m = self.backend.control_mode.upper()
            edge_m = self.core.mode.name
            z = float(self.backend.data.qpos[2])
            t = float(self.backend.data.time)
            paused = f"{C_YELLOW}⏸ {C_RESET}" if self.backend.paused else ""
            emote = ""
            if hasattr(self.backend, "emote_controller") and self.backend.emote_controller.is_active:
                ename = self.backend.emote_controller.current_emote_name or "EMOTE"
                prog = int(self.backend.emote_controller.progress * 100)
                emote = f"|{C_MAGENTA}🎭{ename.upper()} {prog}%{C_RESET}"

            gantry_pill = f"{C_CYAN}G{C_RESET}" if self.backend.gantry_active else f"{C_GRAY}g{C_RESET}"

            return (
                f"{paused}{C_CYAN}[{C_RESET}"
                f"{C_GREEN}{ctrl_m}{C_RESET}|"
                f"{C_YELLOW}{edge_m}{C_RESET}|"
                f"{C_WHITE}z={z:.2f}m{C_RESET}|"
                f"{C_BLUE}{t:.1f}s{C_RESET}|"
                f"{gantry_pill}"
                f"{emote}"
                f"{C_CYAN}]{C_RESET} "
                f"{C_MAGENTA}asimov{C_RESET}{C_BOLD}❯{C_RESET} "
            )
        except Exception:
            return "asimov> "

    def print_banner(self) -> None:
        """Print high-tech console startup banner."""
        viewer_status = f"{C_GREEN}● ACTIVE{C_RESET}" if (self.viewer is not None and self.viewer.is_running()) else f"{C_YELLOW}○ HEADLESS{C_RESET}"
        web_info = f"\n  {C_CYAN}🌐 Web Dashboard{C_RESET} : http://127.0.0.1:{self.web_port}\n  {C_CYAN}📡 Telemetry WS  {C_RESET} : ws://127.0.0.1:{self.ws_port}" if self.enable_web else ""
        active_pol = self.backend.policy_manager.get_active_policy() if hasattr(self.backend, "policy_manager") else None
        pol_name_str = active_pol.name if active_pol else "None"
        pol_spec_str = f"{active_pol.manifest.observation.dim}-D obs ➔ {active_pol.manifest.action.dim}-D acts" if active_pol else "N/A"
        pol_freq_str = f"{active_pol.manifest.control.frequency_hz:.0f} Hz" if active_pol else "N/A"
        banner = f"""
{C_CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                    ASIMOV 1 — VIRTUAL MUJOCO CONTROL CONSOLE                ║
╚══════════════════════════════════════════════════════════════════════════════╝{C_RESET}
  {C_CYAN}Physics Engine{C_RESET}   : MuJoCo ({self.backend.model.opt.timestep*1000:.1f}ms / 200 Hz loop)
  {C_CYAN}Active Policy{C_RESET}    : {pol_name_str} ({pol_spec_str} @ {pol_freq_str})
  {C_CYAN}Policy Category{C_RESET}  : {active_pol.category.upper() if active_pol else 'NONE'}
  {C_CYAN}Live 3D Viewer{C_RESET}   : {viewer_status}
  {C_CYAN}Camera View{C_RESET}      : {self.active_camera} (Chase / FPV / Free)
  {C_CYAN}Environment{C_RESET}      : {self.backend.env_manager.current_preset}
  {C_CYAN}Telemetry Engine{C_RESET} : {self.telemetry.rate_hz:.1f} Hz ({self.telemetry.verbosity.upper()})
  {C_CYAN}Active Mode{C_RESET}      : {C_BOLD}{self.backend.control_mode.upper()}{C_RESET}{web_info}

{C_CYAN}┌── COMMANDS ──────────────────────────────────────────────────────────────────┐{C_RESET}
  {C_GREEN}🕹️  Live Game Teleoperation{C_RESET}
    {C_BOLD}drive / game / teleop{C_RESET}    Direct keyboard game mode (WASD / Space, NO Enter needed, 25Hz HUD)
    {C_BOLD}spawn [dist]{C_RESET}             Drop dynamic physics obstacle in front of robot (real physics)

  {C_GREEN}🚶 Locomotion & Posture{C_RESET}
    {C_BOLD}stand{C_RESET}                   Arm & ramp robot to settled standing pose
    {C_BOLD}walk <vx> [vy] [wz]{C_RESET}     Command velocity in m/s and rad/s (e.g. walk 0.4)
    {C_BOLD}stop{C_RESET}                    Zero commanded velocity & hold active balance
    {C_BOLD}fall / safefall{C_RESET}         Controlled safe collapse into compliant DAMP mode
    {C_BOLD}getup / recover{C_RESET}         Clear faults, restore base height & arm standing

  {C_CYAN}🧠 Pluggable Policies{C_RESET}
    {C_BOLD}policy list{C_RESET}             List all discovered policies in catalog
    {C_BOLD}policy select <name>{C_RESET}    Hot-swap active policy (locomotion, getup, safefall)
    {C_BOLD}policy info [name]{C_RESET}      Display policy manifest specifications
    {C_BOLD}policy validate [name]{C_RESET}  Run deep validation on policy weights & kinematic spec
    {C_BOLD}policy active{C_RESET}            Show runtime status and inference telemetry of active policy

  {C_MAGENTA}🎭 Gestures & Emotes{C_RESET}
    {C_BOLD}hello / wave{C_RESET}            Raise right arm & wave greeting
    {C_BOLD}bow{C_RESET}                     Polite humanoid bow with tilted head
    {C_BOLD}squat / crouch{C_RESET}          Deep knee/hip crouch & smooth return
    {C_BOLD}cheer{C_RESET}                   Double-arm overhead victory celebration
    {C_BOLD}dance{C_RESET}                   Rhythmic waist sway & arm groove
    {C_BOLD}nod / shake{C_RESET}             Affirmation nod or negation head shake
    {C_BOLD}emote <name>{C_RESET}            List or trigger gesture animation

  {C_BLUE}🔬 Telemetry & Diagnostics{C_RESET}
    {C_BOLD}state{C_RESET}                   Full status dashboard (base, IMU, battery, mode)
    {C_BOLD}joints{C_RESET}                  Detailed 25-joint table (pos, target, vel, effort)
    {C_BOLD}sensors / contacts{C_RESET}      Ground contacts, normal forces, and simulated IMU
    {C_BOLD}torque{C_RESET}                  Motor torques, saturation %, and limit margins
    {C_BOLD}obs / action / io{C_RESET}       Inspect 78-D policy inputs & 23-D scaled actions
    {C_BOLD}telemetry <opts>{C_RESET}        on|off|minimal|normal|verbose|raw|rate <hz>

  {C_YELLOW}🎮 Manual Joint Control{C_RESET}
    {C_BOLD}mode <policy|manual|hybrid>{C_RESET} Switch control mode
    {C_BOLD}select <joint>{C_RESET}          Select joint by name or 1-25 index
    {C_BOLD}set <val> / add <delta>{C_RESET} Set target angle in radians
    {C_BOLD}zero{C_RESET}                    Reset selected joint to default standing angle

  {C_WHITE}🌍 Sim & World Controls{C_RESET}
    {C_BOLD}camera <name|list|cycle>{C_RESET} Switch 3D follow / FPV camera view (chase, fpv, free)
    {C_BOLD}env <list|load <p>|reset>{C_RESET} Switch environment presets & custom files (YAML/XML)
    {C_BOLD}reset [seed]{C_RESET}            Deterministic reset to initial settled stance
    {C_BOLD}pause / resume / step{C_RESET}   Freeze physics or step n control cycles
    {C_BOLD}gantry <on|off>{C_RESET}         Toggle virtual gantry weld
    {C_BOLD}push <force> [dir] [s]{C_RESET}  Inject pelvis disturbance force
    {C_BOLD}record / replay / export{C_RESET} Flight data recording & replay

{C_CYAN}├── KEYBOARD SHORTCUTS ────────────────────────────────────────────────────────┤{C_RESET}
  {C_BOLD}W / S{C_RESET} : vx +/- 0.05 m/s    {C_BOLD}A / D{C_RESET} : vy +/- 0.05 m/s    {C_BOLD}Q / E{C_RESET} : wz +/- 0.10 rad/s
  {C_BOLD}SPACE{C_RESET} : stop (zero vel)   {C_BOLD}F{C_RESET}     : safe fall           {C_BOLD}G{C_RESET}     : getup recovery
  {C_BOLD}1 - 4{C_RESET} : pushes 40-250N    {C_BOLD}O{C_RESET}     : spawn obstacle      {C_BOLD}L{C_RESET}     : toggle fall latch
  {C_BOLD}H / B{C_RESET} : wave / bow        {C_BOLD}C / K{C_RESET} : crouch / cheer      {C_BOLD}T{C_RESET}     : dance groove
  {C_BOLD}V{C_RESET}     : cycle camera view {C_BOLD}P / R{C_RESET} : pause / reset     {C_BOLD}TAB{C_RESET}   : auto-complete
{C_CYAN}└──────────────────────────────────────────────────────────────────────────────┘{C_RESET}
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
        print(f"\n{C_CYAN}┌─ COMMAND PIPELINE ───────────────────────────────────────────────────────────┐{C_RESET}")
        print(f"│ {C_BOLD}USER INPUT{C_RESET}      : {user_input}")
        print(f"│ {C_BOLD}PARSED COMMAND{C_RESET}  : {parsed_cmd}")
        print(f"│ {C_BOLD}CONTROL COMMAND{C_RESET} : {control_cmd}")
        lines = sim_result.splitlines()
        first_line = lines[0] if lines else ""
        print(f"│ {C_BOLD}SIMULATOR RESULT{C_RESET}: {C_GREEN}{first_line}{C_RESET}")
        for l in lines[1:]:
            print(f"│                   {C_GREEN}{l}{C_RESET}")
        print(f"{C_CYAN}└──────────────────────────────────────────────────────────────────────────────┘{C_RESET}")

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
        if cmd in ("w", "s", "a", "d", "q", "e", "space", "p", "r", "f", "g", "h", "b", "c", "k", "t", "v"):
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
            self.backend.stop_emote()
            b_state = self.backend.get_base_state()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = STOP (zero velocity hold)",
                control_cmd="command_vel = [0.000, 0.000, 0.000]",
                sim_result=f"mode = {self.core.mode.name}/{self.core.move_submode.name}, body_vx = {b_state['body_vx']:+.3f} m/s",
            )

        elif cmd in ("fall", "safefall", "safe_fall"):
            res = self.backend.safe_fall()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = SAFE_FALL (OpenHorizon neural impact mitigation)",
                control_cmd="backend.safe_fall() -> safefall.onnx active in MOVE/POLICY",
                sim_result=f"mode = {res.get('edge_mode')}, gantry = {res.get('gantry')}\n    {res.get('message')}",
            )

        elif cmd in ("getup", "recover", "standup"):
            res = self.backend.getup()
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = GETUP (OpenHorizon neural ground recovery)",
                control_cmd="backend.getup() -> getup.onnx dynamic pushup in MOVE/POLICY",
                sim_result=f"mode = {res.get('edge_mode')}, base_z = {res.get('base_z'):.4f}m, gantry = {res.get('gantry')}\n    {res.get('message')}",
            )

        elif cmd == "policy":
            self._handle_policy_cmd(args)

        elif cmd in ("drive", "game", "teleop", "live"):
            try:
                self.enter_drive_mode()
            except Exception as e:
                log.error("[CONSOLE] Error during live game teleoperation: %s", e)
                print(f"\n{C_RED}[ERROR] Drive mode failed: {e}{C_RESET}")

        elif cmd == "spawn":
            dist = float(args[0]) if args else 0.8
            res = self.backend.spawn_dynamic_obstacle(distance=dist)
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd=f"action = SPAWN_DYNAMIC_OBSTACLE (distance = {dist:.2f}m)",
                control_cmd=f"backend.spawn_dynamic_obstacle(distance={dist})",
                sim_result=res.get("message", "Dynamic obstacle dropped"),
            )

        elif cmd in ("hello", "wave"):
            self.backend.play_emote("hello")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: HELLO / WAVE",
                control_cmd="backend.play_emote('hello') -> 3.0s right arm greeting wave",
                sim_result="Playing gesture: hello (3.0s)",
            )

        elif cmd == "bow":
            self.backend.play_emote("bow")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: BOW",
                control_cmd="backend.play_emote('bow') -> 3.0s respectful bipedal bow",
                sim_result="Playing gesture: bow (3.0s)",
            )

        elif cmd in ("squat", "crouch"):
            self.backend.play_emote("squat")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: SQUAT / CROUCH",
                control_cmd="backend.play_emote('squat') -> 3.5s controlled knee/hip squat & return",
                sim_result="Playing gesture: squat (3.5s)",
            )

        elif cmd in ("cheer", "celebrate"):
            self.backend.play_emote("cheer")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: CHEER",
                control_cmd="backend.play_emote('cheer') -> 3.0s victory double-arm overhead raise",
                sim_result="Playing gesture: cheer (3.0s)",
            )

        elif cmd == "dance":
            self.backend.play_emote("dance")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: DANCE",
                control_cmd="backend.play_emote('dance') -> 4.0s rhythmic waist sway & arm groove",
                sim_result="Playing gesture: dance (4.0s)",
            )

        elif cmd == "nod":
            self.backend.play_emote("nod")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: NOD",
                control_cmd="backend.play_emote('nod') -> 2.0s head pitch affirmation nod",
                sim_result="Playing gesture: nod (2.0s)",
            )

        elif cmd == "shake":
            self.backend.play_emote("shake")
            self.log_command_pipeline(
                user_input=line,
                parsed_cmd="action = EMOTE: SHAKE",
                control_cmd="backend.play_emote('shake') -> 2.0s head yaw negation shake",
                sim_result="Playing gesture: shake (2.0s)",
            )

        elif cmd == "emote":
            if not args:
                print(f"\n{C_MAGENTA}AVAILABLE EMOTES:{C_RESET}")
                for e in EmoteController.list_emotes():
                    print(f"  • {C_BOLD}{e['name']:<10}{C_RESET} : {e['description']} ({e['duration']:.1f}s)")
                print(f"\nUsage: emote <name>  (or type the emote directly, e.g. 'hello', 'bow', 'squat')")
                return
            ename = args[0].lower()
            ok = self.backend.play_emote(ename)
            if ok:
                dur = self.backend.emote_controller.active_emote.duration if self.backend.emote_controller.active_emote else 3.0
                self.log_command_pipeline(
                    user_input=line,
                    parsed_cmd=f"action = EMOTE: {ename.upper()}",
                    control_cmd=f"backend.play_emote('{ename}') -> {dur:.1f}s",
                    sim_result=f"Playing gesture: {ename} ({dur:.1f}s)",
                )
            else:
                print(f"{C_RED}ERROR: Unknown emote '{ename}'. Type 'emote' for full list.{C_RESET}")

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
            header = f"{'#':<3} {'Actuator':<26} {'Ctrl':>8} {'Target':>8} {'Torque(Nm)':>10} {'Limit':>8} {'Effort%':>9} {'Load':<8}"
            print(f"\n{C_CYAN}{'─' * len(header)}{C_RESET}")
            print(f"{C_BOLD}{header}{C_RESET}")
            print(f"{C_CYAN}{'─' * len(header)}{C_RESET}")
            for a in actuators:
                eff_pct = (abs(a['torque']) / a['force_limit'] * 100.0) if a['force_limit'] > 0 else 0.0
                if eff_pct > 90.0:
                    eff_color = C_RED
                    load_badge = "CRIT"
                elif eff_pct > 65.0:
                    eff_color = C_YELLOW
                    load_badge = "HIGH"
                elif eff_pct > 30.0:
                    eff_color = C_BLUE
                    load_badge = "MED"
                else:
                    eff_color = C_GREEN
                    load_badge = "NOM"
                print(
                    f"{a['index']:<3} {a['name']:<26} {a['ctrl']:>+8.3f} "
                    f"{a['target']:>+8.3f} {a['torque']:>+10.2f} {a['force_limit']:>8.1f} {eff_color}{eff_pct:>8.1f}%{C_RESET} {eff_color}{load_badge:<8}{C_RESET}"
                )
            print(f"{C_CYAN}{'─' * len(header)}{C_RESET}")

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

        elif cmd in ("camera", "cam"):
            if not args or args[0] in ("list", "info"):
                cams = self.get_available_cameras() + ["free", "track"]
                active = self.active_camera
                print(f"\n{C_CYAN}AVAILABLE 3D VIEWER CAMERAS:{C_RESET}")
                for c in cams:
                    is_active = (c == active or self.CAMERA_ALIASES.get(active) == c)
                    mark = f"{C_GREEN} ●{C_RESET}" if is_active else "  "
                    tag = ""
                    if c in ("chase_camera", "chase"):
                        tag = f" {C_YELLOW}(3rd person follow behind robot, moves & rotates with robot){C_RESET}"
                    elif c in ("first_person_behind", "fpv_behind"):
                        tag = f" {C_YELLOW}(Close over-the-shoulder follow behind robot){C_RESET}"
                    elif c in ("first_person_camera", "fpv"):
                        tag = f" {C_YELLOW}(Head eye-level FPV looking forward){C_RESET}"
                    elif c == "free":
                        tag = " (Interactive orbit view)"
                    elif c == "track":
                        tag = " (Center tracking robot floating base)"
                    print(f"{mark} {C_BOLD}{c:<22}{C_RESET}{tag}")
                print(f"\nActive Camera: {C_CYAN}{active}{C_RESET}")
                print("Switch camera: 'camera <name>' (e.g. 'camera chase', 'camera fpv', 'camera free')")
                print("Cycle camera : 'camera cycle' (or press 'V' in live game mode)")
            elif args[0] in ("cycle", "next"):
                new_cam = self.cycle_camera()
                print(f"{C_GREEN}[CAMERA]{C_RESET} Cycled view to '{new_cam}'")
            else:
                target = args[0]
                ok, msg = self.set_camera(target)
                if ok:
                    print(f"{C_GREEN}[CAMERA]{C_RESET} {msg}")
                else:
                    print(f"{C_RED}[CAMERA ERROR]{C_RESET} {msg}")

        elif cmd == "env":
            if not args or args[0] == "list":
                presets = self.backend.env_manager.list_presets()
                print(f"\n{C_CYAN}AVAILABLE ENVIRONMENT PRESETS & CUSTOM FILES:{C_RESET}")
                for p_name, p_desc in presets.items():
                    mark = f"{C_GREEN} ●{C_RESET}" if p_name == self.backend.env_manager.current_preset else "  "
                    print(f"{mark} {C_BOLD}{p_name:<18}{C_RESET} : {p_desc}")
                print("\nLoad any preset or custom file with: 'env load <name_or_path>'")
            elif args[0] == "load":
                if len(args) > 1:
                    preset_arg = " ".join(args[1:]).strip()
                    try:
                        with self.backend._lock:
                            res = self.backend.load_environment(preset_arg)
                            if self.use_viewer:
                                self.reload_viewer_model(res.get("generated_model_path"))
                            if self.gateway and hasattr(self.gateway, "camera_streamer") and self.gateway.camera_streamer:
                                self.gateway.camera_streamer._sync_model_if_changed()
                        print(f"Successfully loaded environment '{preset_arg}':")
                        print(f"  Friction: mu={res['ground_friction']}, Mass scale: {res['mass_scale']}, Obstacles: {res['obstacle_count']}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                else:
                    print("Usage: env load <preset_or_file_path>")
            elif args[0] == "reset":
                with self.backend._lock:
                    res = self.backend.load_environment(self.backend.env_manager.current_preset)
                    if self.use_viewer:
                        self.reload_viewer_model(res.get("generated_model_path"))
                    if self.gateway and hasattr(self.gateway, "camera_streamer") and self.gateway.camera_streamer:
                        self.gateway.camera_streamer._sync_model_if_changed()
                print(f"Reset environment to nominal: {self.backend.env_manager.current_preset}")


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

    def _handle_policy_cmd(self, args: List[str]) -> None:
        """Handle policy inspection, selection, and validation."""
        sub = args[0].lower() if args else "list"
        if sub in ("list", "ls", "catalog"):
            print(self.backend.policy_registry.format_policy_catalog())
        elif sub in ("select", "load", "set", "use"):
            if len(args) < 2:
                print(f"{C_RED}Usage: policy select <name_or_dir>{C_RESET}")
                return
            pol_name = args[1]
            try:
                self.backend.load_policy(pol_name)
                pol = self.backend.policy_manager.get_active_policy()
                dim_str = f"{pol.manifest.observation.dim}-D obs ➔ {pol.manifest.action.dim}-D acts" if pol else "N/A"
                freq_str = f"{pol.manifest.control.frequency_hz:.0f} Hz" if pol else "N/A"
                print(f"{C_GREEN}Successfully loaded policy '{pol_name}' ({dim_str} @ {freq_str}){C_RESET}")
            except Exception as e:
                print(f"{C_RED}Failed to load policy '{pol_name}': {e}{C_RESET}")
        elif sub in ("info", "inspect"):
            target = args[1] if len(args) > 1 else None
            pol = self.backend.policy_manager.get_active_policy()
            if target:
                try:
                    manifest = self.backend.policy_registry.get_manifest(target)
                except Exception as e:
                    print(f"{C_RED}Error: {e}{C_RESET}")
                    return
            elif pol:
                manifest = pol.manifest
            else:
                print(f"{C_RED}No policy specified and no active policy.{C_RESET}")
                return
            print(f"\n{C_CYAN}── POLICY MANIFEST: {manifest.name} ─────────────────────────────{C_RESET}")
            print(f"  Category    : {manifest.category}")
            print(f"  Version     : {manifest.version}")
            print(f"  Runtime     : {manifest.runtime.type}")
            print(f"  Model Path  : {manifest.model.path}")
            print(f"  Frequency   : {manifest.control.frequency_hz} Hz")
            print(f"  Observation : {manifest.observation.dim}-D ({manifest.observation.dtype})")
            print(f"  Action      : {manifest.action.dim}-D ({manifest.action.dtype}, {manifest.action.type})")
            print(f"  Actuator Q  : {len(manifest.action.joint_order)} joints")
            if manifest.description:
                print(f"  Description : {manifest.description}")
        elif sub in ("validate", "check"):
            target = args[1] if len(args) > 1 else (self.backend.policy_manager.active_policy_name or "official_locomotion")
            valid, checks = self.backend.policy_registry.validate_policy(target, self.backend.model)
            print(f"\n{C_CYAN}── POLICY VALIDATION: {target} ──────────────────────────────{C_RESET}")
            for c in checks:
                if c.startswith("✓"):
                    print(f"  {C_GREEN}{c}{C_RESET}")
                else:
                    print(f"  {C_RED}{c}{C_RESET}")
            if valid:
                print(f"\n{C_GREEN}✓ POLICY VALIDATION PASSED{C_RESET}")
            else:
                print(f"\n{C_RED}✗ POLICY VALIDATION FAILED{C_RESET}")
        elif sub in ("active", "status"):
            pol = self.backend.policy_manager.get_active_policy()
            if pol:
                print(f"\n{C_CYAN}Active Policy:{C_RESET} {pol.name}")
                print(f"  Category      : {pol.category}")
                print(f"  Obs / Act Dim : {pol.manifest.observation.dim} / {pol.manifest.action.dim}")
                print(f"  Frequency     : {pol.manifest.control.frequency_hz} Hz")
                print(f"  Step Count    : {pol.step_counter}")
                print(f"  Last Latency  : {pol.last_inference_time_ms:.2f} ms")
            else:
                print("No active policy.")
        else:
            print(f"{C_RED}Unknown policy subcommand '{sub}'. Available: list, select, info, validate, active{C_RESET}")

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
            print(f"\n{C_CYAN}INPUT: KEY=R (RESET) | Simulator reset to settled pose.{C_RESET}")
        elif key == "p":
            self.backend.set_paused(not self.backend.paused)
            status = "PAUSED" if self.backend.paused else "RESUMED"
            print(f"\n{C_YELLOW}INPUT: KEY=P ({status}){C_RESET}")
        elif key in ("f", "fall", "safefall"):
            res = self.backend.safe_fall()
            print(f"\n{C_YELLOW}INPUT: KEY=F (OPENHORIZON SAFE-FALL) | {res.get('message')}{C_RESET}")
        elif key in ("g", "getup", "recover"):
            res = self.backend.getup()
            print(f"\n{C_GREEN}INPUT: KEY=G (OPENHORIZON GET-UP RECOVERY) | {res.get('message')}{C_RESET}")
        elif key in ("h", "hello", "wave"):
            self.backend.play_emote("hello")
            print(f"\n{C_MAGENTA}INPUT: KEY=H (HELLO EMOTE) | Playing wave greeting (3.0s){C_RESET}")
        elif key in ("b", "bow"):
            self.backend.play_emote("bow")
            print(f"\n{C_MAGENTA}INPUT: KEY=B (BOW EMOTE) | Playing respectful bow (3.0s){C_RESET}")
        elif key in ("c", "crouch", "squat"):
            self.backend.play_emote("squat")
            print(f"\n{C_MAGENTA}INPUT: KEY=C (SQUAT EMOTE) | Playing crouch/squat (3.5s){C_RESET}")
        elif key in ("k", "cheer"):
            self.backend.play_emote("cheer")
            print(f"\n{C_MAGENTA}INPUT: KEY=K (CHEER EMOTE) | Playing victory cheer (3.0s){C_RESET}")
        elif key in ("t", "dance"):
            self.backend.play_emote("dance")
            print(f"\n{C_MAGENTA}INPUT: KEY=T (DANCE EMOTE) | Playing dance groove (4.0s){C_RESET}")
        elif key == "1":
            self.backend.apply_push(40.0, "x", 0.15)
            print(f"\n{C_YELLOW}INPUT: KEY=1 (40N PUSH DISTURBANCE){C_RESET}")
        elif key == "2":
            self.backend.apply_push(80.0, "x", 0.20)
            print(f"\n{C_YELLOW}INPUT: KEY=2 (80N PUSH DISTURBANCE){C_RESET}")
        elif key == "3":
            self.backend.apply_push(150.0, "x", 0.25)
            print(f"\n{C_YELLOW}INPUT: KEY=3 (150N STRONG PUSH DISTURBANCE){C_RESET}")
        elif key == "4":
            self.backend.apply_push(250.0, "x", 0.30)
            print(f"\n{C_RED}INPUT: KEY=4 (250N KNOCK-DOWN IMPULSE){C_RESET}")
        elif key == "o":
            res = self.backend.spawn_dynamic_obstacle(0.8)
            print(f"\n{C_GREEN}INPUT: KEY=O (SPAWN OBSTACLE) | {res.get('message')}{C_RESET}")
        elif key == "l":
            self.core.fall_latch_enabled = not self.core.fall_latch_enabled
            state_str = "ENABLED" if self.core.fall_latch_enabled else "DISABLED (Policy free-run)"
            print(f"\n{C_CYAN}INPUT: KEY=L (TOGGLE FALL LATCH) | Fall latch: {state_str}{C_RESET}")
        elif key == "v":
            new_cam = self.cycle_camera()
            print(f"\n{C_CYAN}INPUT: KEY=V (CYCLE CAMERA) | Active: {new_cam}{C_RESET}")


    def enter_drive_mode(self) -> None:
        """
        Interactive live game teleoperation mode.
        Full keyboard control with instantaneous response without pressing Enter.
        Renders live ANSI HUD refreshing at 25 Hz reflecting official ONNX policy and MuJoCo multi-body physics.
        """
        saved_periodic = self.periodic_telemetry_active
        self.periodic_telemetry_active = False

        if not sys.stdin.isatty():
            print(f"\n{C_YELLOW}[DRIVE MODE] Non-interactive environment detected (TTY required for live game controls).{C_RESET}")
            self.periodic_telemetry_active = saved_periodic
            return

        self.backend.control_mode = "policy"
        sim_t = float(self.backend.data.time)
        if self.core.mode == EdgeMode.DAMP:
            cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
            self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
            self.core.stand_settled = True

        last_action_msg = "Game teleoperation engaged. Use WASD to steer, Space to brake."

        try:
            with _RawTerminalMode():
                while self.running and not self._shutdown_event.is_set():
                    loop_start = time.time()
                    sim_t = float(self.backend.data.time)

                    key = _read_key_nonblocking()
                    if key:
                        cur_vx = self.core.current_vx
                        cur_vy = self.core.current_vy
                        cur_wz = self.core.current_vyaw

                        if key in ("esc", "m"):
                            break
                        elif key in ("w", "up"):
                            new_vx = float(np.clip(cur_vx + 0.05, VX_MIN, VX_MAX))
                            self.core.command_velocity(new_vx, cur_vy, cur_wz, sim_t)
                            last_action_msg = f"Accelerate forward: vx={new_vx:+.2f} m/s"
                        elif key in ("s", "down"):
                            new_vx = float(np.clip(cur_vx - 0.05, VX_MIN, VX_MAX))
                            self.core.command_velocity(new_vx, cur_vy, cur_wz, sim_t)
                            last_action_msg = f"Decelerate / reverse: vx={new_vx:+.2f} m/s"
                        elif key in ("a", "left"):
                            new_vy = float(np.clip(cur_vy - 0.05, VY_MIN, VY_MAX))
                            self.core.command_velocity(cur_vx, new_vy, cur_wz, sim_t)
                            last_action_msg = f"Strafe left: vy={new_vy:+.2f} m/s"
                        elif key in ("d", "right"):
                            new_vy = float(np.clip(cur_vy + 0.05, VY_MIN, VY_MAX))
                            self.core.command_velocity(cur_vx, new_vy, cur_wz, sim_t)
                            last_action_msg = f"Strafe right: vy={new_vy:+.2f} m/s"
                        elif key == "q":
                            new_wz = float(np.clip(cur_wz + 0.10, WZ_MIN, WZ_MAX))
                            self.core.command_velocity(cur_vx, cur_vy, new_wz, sim_t)
                            last_action_msg = f"Turn CCW / left: vyaw={new_wz:+.2f} rad/s"
                        elif key == "e":
                            new_wz = float(np.clip(cur_wz - 0.10, WZ_MIN, WZ_MAX))
                            self.core.command_velocity(cur_vx, cur_vy, new_wz, sim_t)
                            last_action_msg = f"Turn CW / right: vyaw={new_wz:+.2f} rad/s"
                        elif key in (" ", "x"):
                            self.core.command_velocity(0.0, 0.0, 0.0, sim_t)
                            last_action_msg = "BRAKE: Commanded velocity zeroed."
                        elif key == "1":
                            self.backend.apply_push(40.0, "x", 0.15)
                            last_action_msg = "Disturbance: 40 N nudge push injected."
                        elif key == "2":
                            self.backend.apply_push(80.0, "x", 0.20)
                            last_action_msg = "Disturbance: 80 N medium push injected."
                        elif key == "3":
                            self.backend.apply_push(150.0, "x", 0.25)
                            last_action_msg = "Disturbance: 150 N strong shove injected."
                        elif key == "4":
                            self.backend.apply_push(250.0, "x", 0.30)
                            last_action_msg = "Disturbance: 250 N heavy knock-down impulse injected!"
                        elif key == "o":
                            res = self.backend.spawn_dynamic_obstacle(distance=0.8)
                            last_action_msg = res.get("message", "Dynamic obstacle dropped in front of robot.")
                        elif key == "f":
                            res = self.backend.safe_fall()
                            last_action_msg = res.get("message", "Controlled safe fall initiated.")
                        elif key == "g":
                            res = self.backend.getup()
                            last_action_msg = res.get("message", "Real policy getup recovery initiated.")
                        elif key == "l":
                            self.core.fall_latch_enabled = not self.core.fall_latch_enabled
                            state_str = "ENABLED" if self.core.fall_latch_enabled else "DISABLED (Policy free-run)"
                            last_action_msg = f"Fall Latch toggled -> {state_str}"
                        elif key == "r":
                            cur_pos = [float(self.backend.data.qpos[adr]) for adr in self.backend.actuator_qposadr]
                            self.core.command_stand("sdk", current_sim_pos=cur_pos, current_time=sim_t)
                            last_action_msg = "STAND commanded: official policy stance armed."
                        elif key == "v":
                            new_cam = self.cycle_camera()
                            last_action_msg = f"Camera view cycled -> {new_cam}"
                        elif key == "p":
                            self.backend.set_paused(not self.backend.paused)
                            status = "PAUSED" if self.backend.paused else "RESUMED"
                            last_action_msg = f"Simulator {status}."

                    # Render ANSI HUD
                    b_state = self.backend.get_base_state()
                    contacts = self.backend.get_contacts()
                    try:
                        torques = self.backend.get_joint_torques()
                        max_tau = max(abs(v) for v in torques.values()) if torques else 0.0
                    except Exception:
                        max_tau = 0.0

                    lf = contacts.get("left_foot", {})
                    rf = contacts.get("right_foot", {})
                    l_touch = lf.get("contact", False)
                    r_touch = rf.get("contact", False)
                    l_force = float(lf.get("force_z", 0.0))
                    r_force = float(rf.get("force_z", 0.0))

                    l_badge = f"{C_GREEN}[TOUCH {l_force:4.1f}N]{C_RESET}" if l_touch else f"{C_GRAY}[AIR   {l_force:4.1f}N]{C_RESET}"
                    r_badge = f"{C_GREEN}[TOUCH {r_force:4.1f}N]{C_RESET}" if r_touch else f"{C_GRAY}[AIR   {r_force:4.1f}N]{C_RESET}"

                    mode_str = f"{self.core.mode.name}/{self.core.move_submode.name}"
                    latch_badge = f"{C_GREEN}ENABLED{C_RESET}" if self.core.fall_latch_enabled else f"{C_YELLOW}DISABLED (Direct Policy){C_RESET}"
                    gantry_badge = f"{C_YELLOW}ENGAGED{C_RESET}" if self.backend.gantry_active else f"{C_GREEN}FREE{C_RESET}"
                    sim_status = f"{C_YELLOW}PAUSED{C_RESET}" if self.backend.paused else f"{C_GREEN}RUNNING{C_RESET}"

                    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -b_state.get("gravity_z", -1.0)))))
                    pos_x = b_state.get("pos_x", b_state.get("x", 0.0))
                    pos_y = b_state.get("pos_y", b_state.get("y", 0.0))
                    pos_z = b_state.get("pos_z", b_state.get("z", 0.61))
                    speed = b_state.get("speed", 0.0)

                    hud = (
                        f"\033[H\033[J"
                        f"{C_CYAN}╔════════════════════════════════════════════════════════════════════════════════════════════════╗{C_RESET}\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}{C_WHITE}🕹️  VIRTUAL ASIMOV 1 — REAL-TIME GAME TELEOPERATION (OFFICIAL POLICY){C_RESET}           {C_CYAN}║{C_RESET}\n"
                        f"{C_CYAN}╠════════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}SIM STATUS{C_RESET} : {sim_status} | Time: {sim_t:7.2f}s | Mode: {C_YELLOW}{mode_str:<12}{C_RESET} | Cam: {C_CYAN}{self.active_camera:<14}{C_RESET} | Latch: {latch_badge}\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}COMMANDS{C_RESET}   : Vx: {C_BOLD}{self.core.current_vx:+.2f}{C_RESET} m/s | Vy: {C_BOLD}{self.core.current_vy:+.2f}{C_RESET} m/s | Vyaw: {C_BOLD}{self.core.current_vyaw:+.2f}{C_RESET} rad/s\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}ODOMETRY{C_RESET}   : X: {pos_x:+6.2f}m | Y: {pos_y:+6.2f}m | Z: {pos_z:6.3f}m | Speed: {speed:4.2f} m/s\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}ATTITUDE{C_RESET}   : Roll: {b_state.get('roll_deg', 0.0):+5.1f}° | Pitch: {b_state.get('pitch_deg', 0.0):+5.1f}° | Yaw: {b_state.get('yaw_deg', 0.0):+5.1f}° | Tilt: {tilt_deg:4.1f}°\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}CONTACTS{C_RESET}   : Left Foot: {l_badge} | Right Foot: {r_badge} | Max Tau: {max_tau:4.1f} N·m\n"
                        f"{C_CYAN}╠════════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}\n"
                        f"{C_CYAN}║{C_RESET} {C_BOLD}LIVE KEYBOARD CONTROLS (NO ENTER NEEDED):{C_RESET}\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[W] / [S]{C_RESET} : Forward / Backward (±0.05)   {C_BOLD}[SPACE]{C_RESET} : Emergency Brake (Zero Velocity)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[A] / [D]{C_RESET} : Strafe Left / Right (±0.05)   {C_BOLD}[X]{C_RESET}     : Zero Commanded Velocity\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[Q] / [E]{C_RESET} : Rotate CCW / CW (±0.10 rad/s) {C_BOLD}[R]{C_RESET}     : Arm STAND Pose\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[1] - [4]{C_RESET} : Push Disturbances (40N, 80N, 150N, 250N)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[O]{C_RESET}       : Spawn / Drop Dynamic Physics Obstacle in front of robot\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[F]{C_RESET}       : OpenHorizon Safe-Fall (safefall.onnx active impact mitigation)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[G]{C_RESET}       : OpenHorizon Get-Up (getup.onnx learned dynamic pushup)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[L]{C_RESET}       : Toggle Fall Latch (Allow policy to run through falls)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[V]{C_RESET}       : Cycle Camera (Chase ➔ FPV Behind ➔ FPV Head ➔ Free Orbit)\n"
                        f"{C_CYAN}║{C_RESET}   {C_BOLD}[P]{C_RESET}       : Pause / Resume Simulation      {C_BOLD}[ESC / M]{C_RESET} : Exit to Main Console\n"

                        f"{C_CYAN}╠════════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}\n"
                        f"{C_CYAN}║{C_RESET} {C_MAGENTA}FEEDBACK{C_RESET}   : {last_action_msg:<86}\n"
                        f"{C_CYAN}╚════════════════════════════════════════════════════════════════════════════════════════════════╝{C_RESET}\n"
                    )
                    sys.stdout.write(hud)
                    sys.stdout.flush()

                    elapsed = time.time() - loop_start
                    sleep_time = max(0.01, 0.04 - elapsed)
                    time.sleep(sleep_time)

        except Exception as e:
            log.error("[DRIVE] Exception in game teleoperation loop: %s", e)
            print(f"\n{C_RED}[ERROR in Drive Mode] {e}{C_RESET}")
        finally:
            self.periodic_telemetry_active = saved_periodic
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()
            print(f"\n{C_GREEN}Exited game teleoperation mode. Returned to Asimov interactive console.{C_RESET}")
            print(f"{C_DIM}Type 'help' for command catalog or 'drive' to re-enter live game control.{C_RESET}\n")

    # ── Console Lifecycle ────────────────────────────────────────────────────

    def run(self, script_path: Optional[Union[str, Path]] = None, start_in_drive_mode: bool = False) -> None:
        """Start physics thread, viewer, telemetry, and interactive REPL."""
        self.running = True
        self._shutdown_event.clear()

        # 1. Start viewer
        self.start_viewer()

        # 2. Start physics simulation thread
        self.physics_thread = threading.Thread(
            target=self._physics_loop,
            daemon=True,
            name="asimov-physics-loop",
        )
        self.physics_thread.start()

        # 3. Start periodic telemetry thread
        self.telemetry_thread = threading.Thread(
            target=self._periodic_telemetry_loop,
            daemon=True,
            name="asimov-telemetry-loop",
        )
        self.telemetry_thread.start()

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

        # 5b. Enter live game drive mode immediately if requested
        if start_in_drive_mode:
            self.enter_drive_mode()

        # 6. Main Interactive REPL Loop with Readline Auto-completion
        setup_readline()
        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    prompt = self._get_prompt()
                    line = input(prompt)
                    self.execute_command(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Received SIGINT. Shutting down...{C_RESET}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Clean shutdown of console, simulator, and viewer."""
        self.running = False
        self._shutdown_event.set()
        if self.physics_thread is not None and self.physics_thread.is_alive():
            self.physics_thread.join(timeout=1.0)
            self.physics_thread = None
        if self.telemetry_thread is not None and self.telemetry_thread.is_alive():
            self.telemetry_thread.join(timeout=1.0)
            self.telemetry_thread = None
        if self.gateway is not None:
            try:
                self.gateway.stop()
            except Exception:
                pass
            self.gateway = None
        if self.viewer is not None:
            v = self.viewer
            self.viewer = None
            try:
                v.close()
            except BaseException:
                pass
        self.backend.env_manager.cleanup()
        log.info("[CONSOLE] Virtual Asimov Console terminated cleanly.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Virtual Asimov 1 Interactive Control Console in MuJoCo.")
    parser.add_argument("--env", type=str, default="flat", help="Environment preset (flat, obstacles, playground, friction_low, etc.)")
    parser.add_argument("--fast", action="store_true", help="Run at max simulation speed rather than wall-clock realtime")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo GUI passive viewer (headless mode)")
    parser.add_argument("--telemetry-verbosity", type=str, default="normal", choices=["minimal", "normal", "verbose", "raw"], help="Telemetry verbosity")
    parser.add_argument("--telemetry-rate", type=float, default=2.0, help="Periodic telemetry rate in Hz")
    parser.add_argument("--script", type=str, default=None, help="Path to batch command script file")
    parser.add_argument("--web", action=argparse.BooleanOptionalAction, default=True, help="Launch live web dashboard gateway alongside interactive console (default: True)")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="HTTP & WS host to bind gateway (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=8852, help="HTTP web port for dashboard (default: 8852)")
    parser.add_argument("--ws-port", type=int, default=8854, help="WebSocket telemetry port (default: 8854)")
    parser.add_argument("--drive", "--game", "--teleop", dest="drive_mode", action="store_true", help="Launch directly into live keyboard game teleoperation mode (WASD keys)")

    # Pluggable Policy CLI extensions
    parser.add_argument("--policy", type=str, default=None, help="Policy name or profile to load: 'official_locomotion' (Menlo locomotion only), 'openhorizon_recovery' (suite), 'openhorizon_safefall', 'openhorizon_getup'. (Default: Menlo locomotion + OpenHorizon recovery suite)")
    parser.add_argument("--list-policies", action="store_true", help="List all discovered policies in catalog and exit")
    parser.add_argument("--validate", action="store_true", help="Validate specified policy contract, weights, and joint kinematics and exit")
    parser.add_argument("--debug-policy", action="store_true", help="Enable verbose debug telemetry for policy observations and actions")
    parser.add_argument("--locomotion", type=str, default=None, help="Specify locomotion policy for multi-policy composition (default: official_locomotion)")
    parser.add_argument("--balance", type=str, default=None, help="Specify balance policy for multi-policy composition")
    parser.add_argument("--recovery", type=str, default=None, help="Specify recovery policy or suite ('openhorizon', 'openhorizon_safefall', 'openhorizon_getup')")
    parser.add_argument(
        "--camera",
        type=str,
        default="chase",
        help="Initial 3D viewer camera view: 'chase' (behind robot, follows & turns), 'fpv_behind', 'fpv' (head eye-level), 'front', 'side', 'free', etc. (default: chase)",
    )
    args = parser.parse_args()

    # macOS Cocoa GUI Support: MuJoCo passive viewer requires running under `mjpython`
    if (
        sys.platform == "darwin"
        and "MJPYTHON_BIN" not in os.environ
        and not args.no_viewer
        and not args.list_policies
        and not args.validate
    ):
        mjpy = Path(sys.executable).parent / "mjpython"
        if mjpy.exists() and os.access(mjpy, os.X_OK):
            os.execv(str(mjpy), [str(mjpy)] + sys.argv)

    # If --list-policies requested, discover and print catalog, then exit
    if args.list_policies:
        from edge.policy_registry import PolicyRegistry
        reg = PolicyRegistry()
        reg.discover()
        print(reg.format_policy_catalog())
        return 0

    # If --validate requested, run validation on policy and exit
    if args.validate:
        from edge.policy_registry import PolicyRegistry
        reg = PolicyRegistry()
        reg.discover()
        target_policy = args.policy or "official_locomotion"
        valid, checks = reg.validate_policy(target_policy)
        print(f"\n{C_CYAN}── POLICY VALIDATION REPORT: {target_policy} ──{C_RESET}")
        for c in checks:
            if c.startswith("✓"):
                print(f"  {C_GREEN}{c}{C_RESET}")
            else:
                print(f"  {C_RED}{c}{C_RESET}")
        if valid:
            print(f"\n{C_GREEN}✓ POLICY VALIDATION PASSED — Ready for simulation.{C_RESET}\n")
            return 0
        else:
            print(f"\n{C_RED}✗ POLICY VALIDATION FAILED — Cannot run in simulation.{C_RESET}\n")
            return 1

    # CLI Default Arbitration:
    # If user ran without --policy, or specified openhorizon recovery:
    # Default is Official Menlo locomotion + OpenHorizon Recovery Suite (Safe-Fall + Get-Up)
    # If user specified --policy official_locomotion: Menlo-only mode (auto_compose=False).
    if args.policy in ("official_locomotion", "official") and args.recovery is None:
        target_policy = "official_locomotion"
        target_loco = None
        target_rec = None
    elif args.policy in ("openhorizon_safefall", "openhorizon_getup", "safefall", "getup") and args.recovery is None:
        target_policy = args.policy
        target_loco = None
        target_rec = None
    else:
        # Default profile: Official Menlo locomotion + OpenHorizon Recovery Suite
        target_policy = None
        target_loco = args.locomotion or (args.policy if args.policy and args.policy not in ("openhorizon_recovery", "getup_safefall") else "official_locomotion")
        target_rec = args.recovery or "openhorizon_recovery"

    console = SimConsole(
        env_preset=args.env,
        realtime=not args.fast,
        use_viewer=not args.no_viewer,
        telemetry_verbosity=args.telemetry_verbosity,
        telemetry_rate_hz=args.telemetry_rate,
        enable_web=args.web,
        web_host=args.web_host,
        web_port=args.web_port,
        ws_port=args.ws_port,
        policy_name=target_policy,
        locomotion_policy=target_loco,
        balance_policy=args.balance,
        recovery_policy=target_rec,
        debug_policy=args.debug_policy,
        camera=args.camera,
    )
    console.run(script_path=args.script, start_in_drive_mode=args.drive_mode)
    return 0



if __name__ == "__main__":
    sys.exit(main())
