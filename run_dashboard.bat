@echo off
rem ==============================================================================
rem run_dashboard.bat — Virtual Asimov 1 Live Web Dashboard Launcher (Windows)
rem ==============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ======================================================================
echo  Launching Virtual Asimov 1 Live Web Dashboard in MuJoCo (Windows)
echo  Web UI   : http://127.0.0.1:8852
echo  WebSocket: ws://127.0.0.1:8854
echo ======================================================================

call "%SCRIPT_DIR%\run_sim.bat" --web %*
endlocal
