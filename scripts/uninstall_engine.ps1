# Removes the Complexity Engine scheduled task. Run as Administrator.
$ErrorActionPreference = "Stop"
$TaskName = "ComplexityEngine"
schtasks.exe /Delete /TN $TaskName /F | Write-Host
Write-Host "Scheduled task '$TaskName' removed."
