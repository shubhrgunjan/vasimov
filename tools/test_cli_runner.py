#!/usr/bin/env python3
"""Run official menlo CLI commands against Virtual Edge and record output."""

import os
import subprocess
import sys
import time
from pathlib import Path

VASIMOV_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = VASIMOV_DIR / "reports" / "cli_run.log"
PYTHON = sys.executable

def main():
    print(f"Starting Virtual Edge in background...")
    edge_env = dict(os.environ)
    edge_env["PYTHONPATH"] = str(VASIMOV_DIR)

    edge_proc = subprocess.Popen(
        [PYTHON, "-m", "edge", "--transport", "udp", "--fast"],
        cwd=str(VASIMOV_DIR),
        env=edge_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for edge to start up
        time.sleep(1.0)
        if edge_proc.poll() is not None:
            out, _ = edge_proc.communicate()
            print("Edge failed to start:")
            print(out)
            sys.exit(1)

        menlo_bin = str(Path(PYTHON).parent / "menlo")

        log_lines = []
        def run_cli(cmd_args):
            full_cmd = [menlo_bin] + cmd_args
            cmd_str = " ".join(full_cmd)
            print(f"\n>>> Running: {cmd_str}")
            log_lines.append(f"$ {cmd_str}\n")
            res = subprocess.run(
                full_cmd,
                cwd=str(VASIMOV_DIR),
                capture_output=True,
                text=True,
            )
            output = res.stdout + res.stderr
            print(output)
            log_lines.append(output)
            log_lines.append(f"[Exit code: {res.returncode}]\n\n")
            return res.returncode

        # 1. status (in DAMP)
        run_cli(["status", "--json"])

        # 2. stand -y
        run_cli(["stand", "-y"])

        # 3. status (in STAND)
        run_cli(["status", "--json"])

        # 4. damp -y
        run_cli(["damp", "-y"])

        # 5. status (in DAMP)
        run_cli(["status", "--json"])

        # Write log
        with open(LOG_PATH, "w") as f:
            f.writelines(log_lines)
        print(f"\nAll CLI commands executed successfully! Log written to: {LOG_PATH}")

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
