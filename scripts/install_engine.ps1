# Registers the Complexity Engine background service with Windows Task Scheduler.
# Run as Administrator.
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$XmlPath  = Join-Path $RepoRoot "engine\scheduler\ComplexityEngine.xml"
$TaskName = "ComplexityEngine"

if (-not (Test-Path $XmlPath)) {
    Write-Error "Task XML not found: $XmlPath"
    exit 1
}

$venvPython = Join-Path $RepoRoot "engine\.venv\Scripts\pythonw.exe"
if (-not (Test-Path $venvPython)) {
    Write-Warning "venv pythonw.exe not found at $venvPython — task will fail until venv is created."
}

Write-Host "Registering scheduled task '$TaskName' from $XmlPath"
schtasks.exe /Create /TN $TaskName /XML $XmlPath /F | Write-Host

Write-Host "Task installed. To start now: schtasks /Run /TN $TaskName"
Write-Host "To uninstall: scripts\uninstall_engine.ps1"
