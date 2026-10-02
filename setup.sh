#!/usr/bin/env bash
# ==============================================================================
# Virtual Asimov 1 (vAsimov) - Cross-Platform Setup Script
# Works on Linux (Ubuntu/Debian, Fedora, Arch) and macOS (Apple Silicon / Intel)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    Virtual Asimov 1 (vAsimov) Environment Setup"
echo "=========================================================="

# 1. Determine OS
OS_TYPE="$(uname -s)"
echo "[1/5] Detected Operating System: $OS_TYPE ($(uname -m))"

# 2. Check Python 3
PYTHON_BIN=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PY_VER="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        PY_MAJOR="$("$cmd" -c 'import sys; print(sys.version_info.major)')"
        PY_MINOR="$("$cmd" -c 'import sys; print(sys.version_info.minor)')"
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_BIN="$(command -v "$cmd")"
            echo "[2/5] Using Python: $PYTHON_BIN (version $PY_VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.10+ is required but not found in PATH." >&2
    exit 1
fi

# 3. Create or verify .venv
echo "[3/5] Setting up virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

# Ensure pip is up to date
.venv/bin/pip install --upgrade pip setuptools wheel

# Install dependencies with pre-releases allowed (required for asimov-protocol & menlo-sdk rc releases)
echo "Installing dependencies from requirements.txt..."
.venv/bin/pip install --pre -r requirements.txt

# On Linux, provide mjpython symlink to python for cross-platform command parity with macOS
if [ "$OS_TYPE" = "Linux" ] && [ ! -f ".venv/bin/mjpython" ]; then
    ln -sf python .venv/bin/mjpython
fi

# 4. Ensure upstream/asimov-1 assets exist
echo "[4/5] Checking upstream/asimov-1 assets..."
mkdir -p upstream
if [ ! -d "upstream/asimov-1/sim-model/assets/meshes" ]; then
    echo "Looking for local asimov-1 repositories..."
    FOUND_ASIMOV=""
    # Check common sibling locations
    for cand in "../../asimov-v1-policies/third_party/asimov-1" "../asimov-v1-policies/third_party/asimov-1" "/home/shubhr/Projects/asimov-stack/official-asimov-1"; do
        if [ -d "$cand/sim-model/assets/meshes" ]; then
            FOUND_ASIMOV="$cand"
            break
        fi
    done

    if [ -n "$FOUND_ASIMOV" ]; then
        echo "Found local asimov-1 repository at $FOUND_ASIMOV; linking..."
        ln -sfn "$FOUND_ASIMOV" upstream/asimov-1
    else
        echo "Cloning official asimov-1 model from GitHub..."
        git clone --depth 1 https://github.com/menloresearch/asimov-1.git upstream/asimov-1
    fi
fi

if [ -d "upstream/asimov-1/sim-model/assets/meshes" ]; then
    echo "Upstream assets verified."
else
    echo "WARNING: upstream/asimov-1/sim-model/assets/meshes not found. Some simulations may fail to load meshes." >&2
fi

# 5. Verification
echo "[5/5] Running test suite verification..."
.venv/bin/python -m unittest discover tests/ -v

echo "=========================================================="
echo "    Setup Complete! Environment is ready."
echo "=========================================================="
echo "To interact with the robot:"
echo "  1. Standalone simulation:"
echo "     .venv/bin/python stand.py"
echo "  2. 3D Viewer (Linux / macOS):"
echo "     .venv/bin/python stand.py --viewer"
echo "  3. Full Virtual Edge stack (UDP + Web Dashboard + 3D Viewer):"
echo "     .venv/bin/python run_edge.py --realtime --viewer"
echo "=========================================================="
