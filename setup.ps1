# ==============================================================================
# Virtual Asimov 1 (vAsimov) — Windows PowerShell Setup Script
# ==============================================================================
[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "    Virtual Asimov 1 (vAsimov) — Windows PowerShell Setup" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PythonExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $ArgsList = @("-3", "setup.py")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
    $ArgsList = @("setup.py")
} else {
    Write-Error "Python 3.10+ is required but not found in PATH. Please install Python 3.10+ from https://www.python.org/"
    exit 1
}

if ($SkipTests) {
    $ArgsList += "--skip-tests"
}

& $PythonExe @ArgsList
if ($LASTEXITCODE -ne 0) {
    Write-Error "Setup failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
