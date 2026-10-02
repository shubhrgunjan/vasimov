#!/usr/bin/env bash
# ==============================================================================
# run_dashboard.sh — Virtual Asimov 1 Web Robot Dashboard Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Launching Virtual Asimov 1 Live Web Dashboard in MuJoCo"
echo " Web UI   : http://127.0.0.1:8852"
echo " WebSocket: ws://127.0.0.1:8854"
echo "======================================================================"

exec "$SCRIPT_DIR/run_sim.sh" --web "$@"
