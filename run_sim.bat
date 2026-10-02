@echo off
rem ==============================================================================
rem run_sim.bat — Virtual Asimov 1 Interactive MuJoCo Console Launcher (Windows)
rem ==============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%\.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_BIN=python"
    ) else (
        echo ERROR: Virtualenv python not found. Please run setup.bat first.
        exit /b 1
    )
)

echo ======================================================================
echo  Launching Virtual Asimov 1 Interactive Console in MuJoCo (Windows)
echo  Python: %PYTHON_BIN%
echo ======================================================================

"%PYTHON_BIN%" -m tools.sim_console %*
endlocal
