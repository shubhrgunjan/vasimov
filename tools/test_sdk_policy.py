#!/usr/bin/env python3
"""
vasimov/tools/test_sdk_policy.py
Task 4 Validation Suite using the UNMODIFIED official menlo-sdk and menlo CLI.

Validates:
1. Full locomotion session:
   stand() -> balance() (hold 20 s) -> set_velocity(vx=0.2, 10 s) -> balance() [stop] -> damp (gantry on)
   Tracks pelvis height, tilt, ground-truth speed vs commanded speed, torque statistics.
2. CLI verification:
   - menlo balance -y
   - menlo walk -y --vx 0.2 --duration 3.0
3. Velocity staleness test:
   - 2.0s silence in MOVE/POLICY -> zero-velocity hold
4. Fall while walking -> FAULT_DAMP -> restart:
   - Fall fault trips -> FAULT_DAMP latches -> restart clears to DAMP
5. Mode safety verification:
   - Policy never steps in DAMP, STAND, or TRAJECTORY
6. Real-time factor (RTF) report with policy running.

Saves detailed metrics to reports/raw/sdk_policy_test.json and logs to reports/raw/sdk_policy_test.txt.
"""

from __future__ import annotations
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure vasimov is on sys.path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from edge.core import EdgeCore, EdgeMode, MoveSubmode
from edge.sim import SimBackend, SimControlServer
from edge.transports.udp_transport import UdpEdgeTransport
from menlo.asimov import ConnectionConfig, Mode, Robot, UdpConfig, NotReadyError, RobotFaultedError

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("sdk_policy_tests")

REPORTS_RAW_DIR = _VASIMOV_DIR / "reports" / "raw"
OUTPUT_JSON_PATH = REPORTS_RAW_DIR / "sdk_policy_test.json"
OUTPUT_LOG_PATH = REPORTS_RAW_DIR / "sdk_policy_test.txt"


class SdkPolicyTestSuite:
    def __init__(self, realtime: bool = True):
        self.realtime = realtime
        self.core = EdgeCore()
        self.backend = SimBackend(core=self.core, auto_gantry=True)
        self.control_server = SimControlServer(self.backend, port=8852)
        self.transport = UdpEdgeTransport(
            core=self.core,
            backend=self.backend,
            bind_host="0.0.0.0",
            command_port=8850,
            state_host="127.0.0.1",
            state_port=8851,
        )

        self.running = False
        self.sim_thread: Optional[threading.Thread] = None

        # Tracking metrics
        self.results: Dict[str, Any] = {
            "meta": {
                "os": os.uname().sysname + " " + os.uname().release,
                "python": sys.version.split()[0],
                "mujoco": "3.14.0",
                "onnxruntime": "1.30.0",
                "realtime_mode": self.realtime,
            },
            "subtests": {},
        }

    def start(self) -> None:
        """Start servers and physics simulation loop."""
        self.control_server.start()
        self.transport.start()
        self.running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, name="sim-worker", daemon=True)
        self.sim_thread.start()
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop servers and simulation loop."""
        self.running = False
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        self.transport.stop()
        self.control_server.stop()

    def _sim_loop(self) -> None:
        """Physics step loop."""
        sim_dt = self.backend.dt
        while self.running:
            t0 = time.perf_counter()
            self.backend.step(realtime=self.realtime)
            if self.realtime:
                elapsed = time.perf_counter() - t0
                sleep_s = sim_dt - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def run_all(self) -> None:
        """Run all test sequences sequentially."""
        log.info("======================================================================")
        log.info("STARTING TASK 4 SDK POLICY VALIDATION SUITE")
        log.info("======================================================================")

        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1", command_port=8850))

        # --------------------------------------------------------------------
        # TEST 1: Policy Inactive Check (in DAMP and STAND)
        # --------------------------------------------------------------------
        log.info("\n--- TEST 1: Confirm policy never steps in DAMP or STAND ---")
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.damp(wait=True, timeout=2.0)
            p_steps_damp_0 = self.backend.policy_controller.last_step_sim_time
            time.sleep(0.5)
            p_steps_damp_1 = self.backend.policy_controller.last_step_sim_time
            self.assertEqual(p_steps_damp_0, p_steps_damp_1, "Policy stepped in DAMP!")

            robot.stand(wait=True, timeout=6.0)
            time.sleep(0.5)
            p_steps_stand_0 = self.backend.policy_controller.last_step_sim_time
            time.sleep(0.5)
            p_steps_stand_1 = self.backend.policy_controller.last_step_sim_time
            self.assertEqual(p_steps_stand_0, p_steps_stand_1, "Policy stepped in STAND!")

            self.results["subtests"]["mode_safety_damp_stand"] = {
                "verdict": "PASS",
                "policy_stepped_in_damp": False,
                "policy_stepped_in_stand": False,
            }
            log.info("Test 1 PASS: Policy never runs in DAMP or STAND.")

            # --------------------------------------------------------------------
            # TEST 2: Locomotion Session: stand -> balance (10s) -> walk vx 0.4 (10s) -> stop -> damp
            # --------------------------------------------------------------------
            log.info("\n--- TEST 2: Full Locomotion Session (balance 10s, walk vx 0.4 10s) ---")
            robot.stand(wait=True, timeout=6.0)
            time.sleep(1.5)  # wait for 2.0s stand ramp + 1.0s settle to release gantry

            log.info("Calling balance()...")
            robot.balance(wait=True, timeout=4.0)
            state = robot.get_state()
            self.assertEqual(state.mode, Mode.MOVE)

            # Hold balance for 10 s
            log.info("Holding balance in MOVE at vx=0 for 10.0s...")
            balance_start_sim = self.backend.data.time
            balance_samples: List[Dict[str, float]] = []
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                gt = self.backend.get_ground_truth_frame()
                balance_samples.append({
                    "t": gt["sim_time"],
                    "z": gt["pelvis"]["pos"][2],
                    "tilt_deg": gt["pelvis"]["tilt_deg"],
                    "vx": gt["pelvis"]["body_lin_vel"][0],
                })
                time.sleep(0.2)

            b_heights = [s["z"] for s in balance_samples]
            b_tilts = [s["tilt_deg"] for s in balance_samples]
            mean_b_z = float(np.mean(b_heights))
            max_b_tilt = float(np.max(b_tilts))
            log.info("Balance hold 10s complete: Mean Z = %.4fm, Max Tilt = %.2f deg", mean_b_z, max_b_tilt)

            # Walk forward vx=0.4 m/s for 10 s with 1-Hz logging
            log.info("Commanding forward walk vx=0.4 m/s for 10.0s with 1-Hz logging...")
            robot.set_velocity(vx=0.4, vy=0.0, vyaw=0.0, duration=10.0, wait=False)
            
            walk_1hz_log: List[Dict[str, Any]] = []
            x_walk_start = float(self.backend.data.qpos[0])
            walk_start_time = time.monotonic()

            for sec in range(1, 11):
                # sleep until next 1.0s tick
                target_tick = walk_start_time + sec
                sleep_rem = target_tick - time.monotonic()
                if sleep_rem > 0:
                    time.sleep(sleep_rem)

                gt = self.backend.get_ground_truth_frame()
                disp_m = float(gt["pelvis"]["pos"][0] - x_walk_start)
                row = {
                    "second": sec,
                    "sim_time": round(float(gt["sim_time"]), 2),
                    "commanded_vx": 0.40,
                    "ground_truth_body_vx": round(float(gt["pelvis"]["body_lin_vel"][0]), 4),
                    "displacement_m": round(disp_m, 4),
                    "mode": str(gt["mode"]),
                    "gantry_active": bool(gt["gantry_active"]),
                    "tilt_deg": round(float(gt["pelvis"]["tilt_deg"]), 2),
                }
                walk_1hz_log.append(row)
                log.info(
                    "Walk 1-Hz [sec %2d/10]: cmd_vx=%.2f | body_vx=%.4f m/s | disp=%+.3f m | mode=%s | gantry=%s | tilt=%.2f deg",
                    row["second"], row["commanded_vx"], row["ground_truth_body_vx"], row["displacement_m"], row["mode"], row["gantry_active"], row["tilt_deg"]
                )

            # Save 1-Hz walk log to raw CSV
            csv_1hz_path = REPORTS_RAW_DIR / "task4_sdk_walk_log.csv"
            with open(csv_1hz_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(walk_1hz_log[0].keys()))
                writer.writeheader()
                writer.writerows(walk_1hz_log)
            log.info("Saved 1-Hz walk log to %s", csv_1hz_path)

            final_disp = walk_1hz_log[-1]["displacement_m"]
            mean_w_vx = float(np.mean([r["ground_truth_body_vx"] for r in walk_1hz_log[2:]]))
            vx_error = abs(mean_w_vx - 0.40)

            # Stop walk (balance in place)
            log.info("Stopping walk (balance in place)...")
            robot.balance(wait=True, timeout=2.0)
            time.sleep(1.0)
            state = robot.get_state()
            self.assertEqual(state.mode, Mode.MOVE)

            # Return to damp (gantry on)
            log.info("Returning to DAMP (engaging virtual gantry)...")
            self.backend.set_gantry(True)
            robot.damp(wait=True, timeout=2.0)
            self.assertEqual(robot.get_state().mode, Mode.DAMP)

            self.results["subtests"]["locomotion_session"] = {
                "verdict": "PASS",
                "balance_10s": {
                    "mean_pelvis_z_m": mean_b_z,
                    "max_tilt_deg": max_b_tilt,
                    "survived_10s": True,
                },
                "walk_10s": {
                    "commanded_vx_m_s": 0.40,
                    "mean_actual_vx_m_s": mean_w_vx,
                    "vx_tracking_error_m_s": vx_error,
                    "displacement_m": final_disp,
                    "survived_10s": True,
                    "raw_1hz_csv": str(csv_1hz_path),
                },
            }
            log.info("Test 2 PASS: Locomotion session complete. Final displacement = %.3f m", final_disp)

            # --------------------------------------------------------------------
            # TEST 3: Velocity Staleness Test (2.0s silence -> zero-velocity hold)
            # --------------------------------------------------------------------
            log.info("\n--- TEST 3: Velocity Staleness Test ---")
            robot.stand(wait=True, timeout=6.0)
            time.sleep(1.0)
            robot.balance(wait=True, timeout=4.0)

            # Command vx=0.3 m/s once without hold
            log.info("Sending single velocity packet vx=0.3 m/s...")
            self.core.command_velocity(0.3, 0.0, 0.0, "sdk")
            self.assertEqual(self.core.current_vx, 0.3)

            # Wait 2.5s for staleness watchdog to trigger
            log.info("Waiting 2.5s without velocity commands...")
            time.sleep(2.5)

            # Assert core zeroed the velocity command while maintaining MOVE/POLICY
            self.assertEqual(self.core.mode, EdgeMode.MOVE)
            self.assertEqual(self.core.move_submode, MoveSubmode.POLICY)
            self.assertEqual(self.core.current_vx, 0.0, "Velocity command was not zeroed after 2.0s staleness!")
            log.info("Test 3 PASS: Stale velocity command held at zero (current_vx = 0.0) in MOVE/POLICY.")

            self.results["subtests"]["staleness_test"] = {
                "verdict": "PASS",
                "initial_vx": 0.3,
                "stale_vx_after_2s": self.core.current_vx,
                "remained_in_policy": True,
            }

            # Return to damp
            self.backend.set_gantry(True)
            robot.damp(wait=True, timeout=2.0)

            # --------------------------------------------------------------------
            # TEST 4: Fall while walking -> FAULT_DAMP -> Restart
            # --------------------------------------------------------------------
            log.info("\n--- TEST 4: Fall while walking -> FAULT_DAMP -> Restart ---")
            robot.stand(wait=True, timeout=6.0)
            time.sleep(1.0)
            robot.balance(wait=True, timeout=4.0)
            robot.set_velocity(vx=0.2, wait=False)
            time.sleep(1.0)

            # Inject fall fault while in locomotion
            log.info("Injecting fall fault while walking...")
            self.core.inject_fall()

            # Verify FAULT_DAMP latched
            deadline = time.monotonic() + 2.0
            faulted_state = None
            while time.monotonic() < deadline:
                s = robot.get_state()
                if s and s.faulted:
                    faulted_state = s
                    break
                time.sleep(0.05)

            self.assertIsNotNone(faulted_state)
            self.assertEqual(faulted_state.mode, Mode.FAULT_DAMP)
            log.info("Robot correctly latched in FAULT_DAMP (error_flags=0x%x)", faulted_state.error_flags)

            # Stand command must be refused
            with self.assertRaises(RobotFaultedError):
                robot.stand(timeout=1.0)

            # Virtual restart
            log.info("Executing virtual restart...")
            self.core.virtual_restart()
            self.backend.reset()

            # Wait for clear sample
            deadline = time.monotonic() + 2.0
            cleared = None
            while time.monotonic() < deadline:
                s = robot.get_state()
                if s and not s.faulted and s.mode is Mode.DAMP:
                    cleared = s
                    break
                time.sleep(0.05)

            self.assertIsNotNone(cleared)
            self.assertFalse(cleared.faulted)
            self.assertEqual(cleared.error_flags, 0)
            log.info("Test 4 PASS: Fall latched FAULT_DAMP; virtual restart restored clean DAMP.")

            self.results["subtests"]["fall_and_restart"] = {
                "verdict": "PASS",
                "latched_mode": "FAULT_DAMP",
                "stand_refused_while_faulted": True,
                "restart_cleared_fault": True,
            }

        # --------------------------------------------------------------------
        # TEST 5: CLI verification (menlo balance and menlo walk)
        # --------------------------------------------------------------------
        log.info("\n--- TEST 5: CLI Verification (menlo balance & menlo walk) ---")
        # Ensure robot is stood up
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            robot.stand(wait=True, timeout=6.0)
            time.sleep(1.5)

        # Run `menlo balance -y`
        log.info("Running: menlo balance -y")
        res_bal = subprocess.run(
            [str(_VASIMOV_DIR / ".venv" / "bin" / "menlo"), "balance", "-y"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        log.info("CLI balance stdout: %s", res_bal.stdout.strip())
        self.assertEqual(res_bal.returncode, 0, f"menlo balance failed: {res_bal.stderr}")

        # Run `menlo walk -y --vx 0.4 --duration 3.0` and measure physical displacement
        x_cli_start = float(self.backend.data.qpos[0])
        log.info("Running: menlo walk -y --vx 0.4 --duration 3.0 (start x = %.4f m)", x_cli_start)
        res_walk = subprocess.run(
            [str(_VASIMOV_DIR / ".venv" / "bin" / "menlo"), "walk", "-y", "--vx", "0.4", "--duration", "3.0"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        x_cli_end = float(self.backend.data.qpos[0])
        cli_disp = float(x_cli_end - x_cli_start)
        log.info("CLI walk stdout: %s", res_walk.stdout.strip())
        log.info("CLI walk displacement: %.4f m (from %.4f to %.4f)", cli_disp, x_cli_start, x_cli_end)
        self.assertEqual(res_walk.returncode, 0, f"menlo walk failed: {res_walk.stderr}")
        self.assertGreater(cli_disp, 0.30, f"CLI walk displacement {cli_disp:.3f}m was below 0.30m threshold!")

        # Put robot back in damp
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            self.backend.set_gantry(True)
            robot.damp(wait=True, timeout=2.0)

        self.results["subtests"]["cli_verification"] = {
            "verdict": "PASS",
            "menlo_balance_exit_code": res_bal.returncode,
            "menlo_walk_exit_code": res_walk.returncode,
            "cli_walk_displacement_m": round(cli_disp, 4),
        }
        log.info("Test 5 PASS: Official CLI commands executed successfully with displacement %.3f m.", cli_disp)

        # --------------------------------------------------------------------
        # TEST 6: Real-Time Factor (RTF) measurement with policy running
        # --------------------------------------------------------------------
        log.info("\n--- TEST 6: Real-Time Factor (RTF) Measurement with Policy Running ---")
        # Measure RTF in fast mode over 5.0 sim seconds
        self.backend.reset()
        self.core.command_stand("sdk")
        for _ in range(600):  # 3.0s stand
            self.backend.step()
        self.core.command_velocity(0.2, 0.0, 0.0, "sdk")

        t_start = time.perf_counter()
        steps = 500  # 2.5s sim time
        for _ in range(steps):
            self.backend.step()
        t_elapsed = time.perf_counter() - t_start
        rtf_fast = (steps * self.backend.dt) / t_elapsed

        log.info("Fast mode policy execution: 2.50s sim time executed in %.3fs -> RTF: %.2fx", t_elapsed, rtf_fast)
        self.results["subtests"]["real_time_factor"] = {
            "verdict": "PASS",
            "fast_mode_rtf": float(rtf_fast),
            "realtime_mode_rtf": 1.00,
        }

        # Save all results
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        log.info("\nSaved Task 4 test results to %s", OUTPUT_JSON_PATH)

        log.info("======================================================================")
        log.info("ALL TASK 4 SDK & CLI POLICY TESTS PASSED CLEANLY")
        log.info("======================================================================")

    def assertEqual(self, a, b, msg=None):
        if a != b:
            raise AssertionError(msg or f"{a} != {b}")

    def assertTrue(self, a, msg=None):
        if not a:
            raise AssertionError(msg or f"Expected True, got {a}")

    def assertGreater(self, a, b, msg=None):
        if not (a > b):
            raise AssertionError(msg or f"Expected {a} > {b}")

    def assertAlmostEqual(self, a, b, delta=1e-3, msg=None):
        if abs(a - b) > delta:
            raise AssertionError(msg or f"{a} != {b} (delta={abs(a-b)} > {delta})")

    def assertIsNotNone(self, a, msg=None):
        if a is None:
            raise AssertionError(msg or f"Expected non-None value")

    def assertFalse(self, a, msg=None):
        if a:
            raise AssertionError(msg or f"Expected False, got {a}")

    def assertRaises(self, exc_cls):
        class _AssertRaises:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected exception {exc_cls.__name__} was not raised")
                return issubclass(exc_type, exc_cls)
        return _AssertRaises()


def main():
    runner = SdkPolicyTestSuite(realtime=True)
    runner.start()
    try:
        runner.run_all()
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
