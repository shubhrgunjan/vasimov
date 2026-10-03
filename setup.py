#!/usr/bin/env python3
"""
Virtual Asimov 1 (vAsimov) — OS-Independent Setup Script
Works natively across Linux, macOS (Apple Silicon / Intel), and Windows (x86_64 / ARM64).

Usage:
    python setup.py
    python3 setup.py
    python setup.py --skip-tests
"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"\033[1;34m[vasimov-setup]\033[0m {msg}" if sys.stdout.isatty() else f"[vasimov-setup] {msg}")


def log_success(msg: str) -> None:
    print(f"\033[1;32m[vasimov-setup]\033[0m {msg}" if sys.stdout.isatty() else f"[vasimov-setup] {msg}")


def log_warn(msg: str) -> None:
    print(f"\033[1;33m[vasimov-setup]\033[0m WARNING: {msg}" if sys.stdout.isatty() else f"[vasimov-setup] WARNING: {msg}")


def log_error(msg: str) -> None:
    print(f"\033[1;31m[vasimov-setup]\033[0m ERROR: {msg}" if sys.stdout.isatty() else f"[vasimov-setup] ERROR: {msg}", file=sys.stderr)


def get_venv_python(venv_dir: Path) -> Path:
    """Return the platform-specific path to the python binary inside the virtualenv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def get_venv_pip(venv_dir: Path) -> Path:
    """Return the platform-specific path to the pip binary inside the virtualenv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """Run a shell command with real-time output streaming."""
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"Running: {cmd_str}")
    res = subprocess.run(cmd, cwd=cwd)
    if check and res.returncode != 0:
        log_error(f"Command failed with exit code {res.returncode}: {cmd_str}")
        sys.exit(res.returncode)
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform setup for Virtual Asimov 1.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running the test suite after installation.")
    parser.add_argument("--venv", type=str, default=".venv", help="Virtual environment directory (default: .venv).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print("======================================================================")
    print("       Virtual Asimov 1 (vAsimov) — OS-Independent Setup")
    print("======================================================================")

    # 1. Determine Operating System & Architecture
    system_os = platform.system()
    machine = platform.machine()
    log(f"[1/5] Detected Platform: {system_os} ({machine}) running Python {platform.python_version()}")

    # 2. Check Python Version (Must be >= 3.10)
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 10):
        log_error(f"Python 3.10+ is required, but found Python {major}.{minor}. Please install Python 3.10 or newer.")
        return 1
    log(f"[2/5] Host Python version verified: {major}.{minor}")

    # 3. Create or verify Virtual Environment
    venv_dir = project_root / args.venv
    log(f"[3/5] Setting up virtual environment at: {venv_dir}")
    if not venv_dir.exists():
        import venv
        log("Creating fresh virtual environment...")
        venv.create(venv_dir, with_pip=True)
    else:
        log("Existing virtual environment detected.")

    venv_python = get_venv_python(venv_dir)
    venv_pip = get_venv_pip(venv_dir)

    if not venv_python.exists():
        log_error(f"Virtualenv python executable not found at: {venv_python}")
        return 1

    # Upgrade pip, setuptools, wheel inside venv
    log("Upgrading pip, setuptools, and wheel in virtualenv...")
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install requirements with --pre (required for asimov-protocol and menlo-sdk rc releases)
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        log("Installing dependencies from requirements.txt (with pre-releases enabled)...")
        run_command([str(venv_pip), "install", "--pre", "-r", str(req_file)])

        # macOS compatibility check: on macOS < 13.4, onnxruntime > 1.20.1 lacks std::to_chars in libc++
        if sys.platform == "darwin":
            try:
                mac_ver_str = platform.mac_ver()[0]
                mac_parts = [int(p) for p in mac_ver_str.split(".") if p.isdigit()]
                if len(mac_parts) >= 2 and (mac_parts[0] < 13 or (mac_parts[0] == 13 and mac_parts[1] < 4)):
                    log("Detected macOS < 13.4: ensuring compatible onnxruntime<=1.20.1 is installed...")
                    run_command([str(venv_pip), "install", "onnxruntime<=1.20.1"])
            except Exception as e:
                log_warn(f"Could not check macOS version for onnxruntime pin: {e}")
    else:
        log_warn("requirements.txt not found. Skipping dependency installation.")

    # Platform-specific compatibility links (POSIX mjpython alias on Linux)
    # Note: On macOS, mujoco installs a native Cocoa trampoline binary `mjpython`.
    # On Linux, MuJoCo does not provide mjpython, so we create an alias to python.
    if sys.platform.startswith("linux"):
        mjpython = venv_dir / "bin" / "mjpython"
        if not mjpython.exists():
            try:
                mjpython.symlink_to("python")
                log("Created mjpython symlink in virtual environment.")
            except Exception as e:
                log_warn(f"Could not create mjpython symlink: {e}")

    # 4. Check & Link Upstream Robot Assets
    log("[4/5] Checking upstream/asimov-1 robot assets and meshes...")
    upstream_dir = project_root / "upstream"
    upstream_dir.mkdir(exist_ok=True)
    target_asimov = upstream_dir / "asimov-1"
    meshes_dir = target_asimov / "sim-model" / "assets" / "meshes"

    if not meshes_dir.exists():
        log("Looking for local asimov-1 repository copies...")
        candidates = [
            project_root.parent.parent / "asimov-v1-policies" / "third_party" / "asimov-1",
            project_root.parent / "asimov-v1-policies" / "third_party" / "asimov-1",
            Path.home() / "Projects" / "asimov-stack" / "official-asimov-1",
        ]
        found_source = None
        for cand in candidates:
            if (cand / "sim-model" / "assets" / "meshes").exists():
                found_source = cand
                break

        if found_source:
            log(f"Found local asset repository at: {found_source}")
            if target_asimov.exists() or target_asimov.is_symlink():
                if target_asimov.is_symlink() or target_asimov.is_file():
                    target_asimov.unlink()
                else:
                    shutil.rmtree(target_asimov)

            try:
                target_asimov.symlink_to(found_source, target_is_directory=True)
                log(f"Linked {target_asimov} -> {found_source}")
            except OSError as e:
                log_warn(f"Symlink creation failed ({e}). Copying meshes directory instead...")
                shutil.copytree(found_source, target_asimov)
        else:
            log("Cloning official asimov-1 repository from GitHub (shallow clone)...")
            try:
                run_command(["git", "clone", "--depth", "1", "https://github.com/menloresearch/asimov-1.git", str(target_asimov)])
            except Exception as e:
                log_warn(f"git clone failed: {e}. Model meshes may need to be placed manually in upstream/asimov-1.")

    if meshes_dir.exists():
        log_success("Upstream asimov-1 assets and meshes verified.")
    else:
        log_warn("upstream/asimov-1/sim-model/assets/meshes not found. Robot simulations may fall back to simplified geoms.")

    # 5. Verification Test Suite
    if not args.skip_tests:
        log("[5/5] Running test suite verification...")
        test_env = os.environ.copy()
        if sys.platform == "darwin":
            default_gl = "cgl"
        elif sys.platform.startswith("linux"):
            default_gl = "egl"
        else:
            default_gl = "osmesa"
        test_env.setdefault("MUJOCO_GL", default_gl)
        res = subprocess.run([str(venv_python), "-m", "unittest", "discover", "-s", "tests", "-v"], env=test_env)
        if res.returncode == 0:
            log_success("All tests passed successfully!")
        else:
            log_warn(f"Some tests failed (code {res.returncode}). Check output above.")
    else:
        log("[5/5] Skipping test suite verification (--skip-tests passed).")

    print("\n" + "=" * 70)
    log_success("Virtual Asimov 1 (vAsimov) Setup Completed Successfully!")
    print("=" * 70)

    # OS-Specific Launch Instructions
    if sys.platform == "win32":
        print("\nTo launch on Windows (Command Prompt or PowerShell):")
        print("  1. Interactive Console (Default CLI):")
        print("     .\\run_sim.bat")
        print("     or: .\\.venv\\Scripts\\python -m tools.sim_console")
        print("  2. Live Web Robot Dashboard:")
        print("     .\\run_dashboard.bat")
        print("     or: .\\.venv\\Scripts\\python -m tools.sim_console --web")
        print("     Browser: http://127.0.0.1:8852\n")
    elif sys.platform == "darwin":
        print("\nTo launch on macOS:")
        print("  1. Interactive Console (Default CLI):")
        print("     ./run_sim.sh")
        print("     or: .venv/bin/python -m tools.sim_console")
        print("  2. Live Web Robot Dashboard:")
        print("     ./run_dashboard.sh")
        print("     or: .venv/bin/python -m tools.sim_console --web")
        print("     Browser: http://127.0.0.1:8852\n")
    else:  # Linux
        print("\nTo launch on Linux:")
        print("  1. Interactive Console (Default CLI):")
        print("     ./run_sim.sh")
        print("     or: .venv/bin/python -m tools.sim_console")
        print("  2. Live Web Robot Dashboard:")
        print("     ./run_dashboard.sh")
        print("     or: .venv/bin/python -m tools.sim_console --web")
        print("     Browser: http://127.0.0.1:8852\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
