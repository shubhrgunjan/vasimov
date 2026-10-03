#!/usr/bin/env bash
# ==============================================================================
# run_sim.sh — Virtual Asimov 1 Interactive MuJoCo Console Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Select python virtualenv (on macOS prefer mjpython for interactive Cocoa 3D GUI)
if [ "$(uname -s)" = "Darwin" ] && [ -x "$SCRIPT_DIR/.venv/bin/mjpython" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/mjpython"
elif [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python3 not found." >&2
    exit 1
fi

echo "======================================================================"
echo " Launching Virtual Asimov 1 Interactive Console in MuJoCo"
echo " Python: $PYTHON_BIN"
echo "======================================================================"

exec "$PYTHON_BIN" -m tools.sim_console "$@"
