#!/usr/bin/env python3
"""Run menlo balance and menlo walk against NoPolicy stub and log exact output."""

import os
import subprocess
import sys
import time
from pathlib import Path

VASIMOV_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = VASIMOV_DIR / "reports" / "raw" / "nopolicy_cli_outcome.log"
PYTHON = sys.executable
MENLO_BIN = str(Path(PYTHON).parent / "menlo")


def main():
    print("Starting Virtual Edge...")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(VASIMOV_DIR)

    edge_proc = subprocess.Popen(
        [PYTHON, "-m", "edge", "--transport", "udp", "--fast"],
        cwd=str(VASIMOV_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)

    log_lines = []
    try:
        def run_cli(cmd_args):
            full_cmd = [MENLO_BIN] + cmd_args
            cmd_str = " ".join(full_cmd)
            print(f"\n>>> Running: {cmd_str}")
            log_lines.append(f"$ {cmd_str}\n")
            t0 = time.time()
            res = subprocess.run(
                full_cmd,
                cwd=str(VASIMOV_DIR),
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            output = res.stdout + res.stderr
            print(f"Elapsed: {elapsed:.2f}s | Exit code: {res.returncode}")
            print(output)
            log_lines.append(f"[Elapsed: {elapsed:.2f}s | Exit code: {res.returncode}]\n")
            log_lines.append(output + "\n\n")
            return res.returncode

        # 1. Stand robot
        run_cli(["stand", "-y"])

        # 2. Balance against NoPolicy stub
        run_cli(["balance", "-y"])

        # 3. Walk against NoPolicy stub
        run_cli(["walk", "--vx", "0.2", "--duration", "2", "-y"])

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        print(f"\nSaved NoPolicy CLI transcript to: {LOG_PATH}")

    finally:
        print("Stopping Virtual Edge...")
        edge_proc.terminate()
        try:
            edge_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            edge_proc.kill()
        print("Virtual Edge stopped.")


if __name__ == "__main__":
    main()
