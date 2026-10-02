#!/usr/bin/env bash
# ==============================================================================
# Virtual Asimov 1 (vAsimov) — Linux & macOS Setup Script
# Invokes cross-platform setup.py
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=""
for cmd in python3.12 python3.11 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PY_MAJOR="$("$cmd" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
        PY_MINOR="$("$cmd" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_BIN="$(command -v "$cmd")"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.10+ is required but not found in PATH." >&2
    echo "Please install Python 3.10 or newer." >&2
    exit 1
fi

exec "$PYTHON_BIN" setup.py "$@"
