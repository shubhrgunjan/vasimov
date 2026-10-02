@echo off
rem ==============================================================================
rem Virtual Asimov 1 (vAsimov) — Windows Setup Batch Script
rem ==============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ======================================================================
echo    Virtual Asimov 1 (vAsimov) — Windows Environment Setup
echo ======================================================================

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3"
    goto :found_python
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

echo ERROR: Python 3.10+ is required but not found in PATH.
echo Please install Python 3.10+ from https://www.python.org/
exit /b 1

:found_python
%PYTHON_EXE% setup.py %*
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Setup failed with exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

endlocal
