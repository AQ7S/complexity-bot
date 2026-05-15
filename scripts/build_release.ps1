# Build the full Complexity Engine installer (.exe).
#
# Steps:
#   1. PyInstaller bundles the Python engine -> engine/dist/engine/engine.exe
#   2. Vite + tsc build the renderer + electron main/preload -> ui/dist + ui/dist-electron
#   3. electron-builder packs everything (engine bundle + renderer + main process)
#      into ui/release/Complexity Engine-Setup-<ver>.exe
#
# Run from repo root:
#   .\scripts\build_release.ps1
# Optional flags:
#   -SkipEngine   reuse the existing PyInstaller output (much faster iteration)
#   -SkipUi       only rebuild the engine bundle
#   -Clean        wipe engine/build, engine/dist, ui/release before building

param(
    [switch]$SkipEngine,
    [switch]$SkipUi,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

$EngineDir   = Join-Path $RepoRoot "engine"
$EngineVenv  = Join-Path $EngineDir ".venv\Scripts"
$EnginePy    = Join-Path $EngineVenv "python.exe"
$UiDir       = Join-Path $RepoRoot "ui"

if (-not (Test-Path $EnginePy)) {
    throw "Engine venv not found at $EnginePy. Run: python -m venv engine\.venv && pip install -r engine\requirements.txt -r engine\requirements-dev.txt"
}

if ($Clean) {
    Write-Host "[clean] removing prior build artefacts" -ForegroundColor Yellow
    Remove-Item -Recurse -Force (Join-Path $EngineDir "build")    -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $EngineDir "dist")     -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $UiDir     "release")  -ErrorAction SilentlyContinue
}

if (-not $SkipEngine) {
    Write-Host "[1/3] PyInstaller bundling engine ..." -ForegroundColor Cyan
    & $EnginePy -m PyInstaller (Join-Path $EngineDir "build_engine.spec") --noconfirm --clean --distpath (Join-Path $EngineDir "dist") --workpath (Join-Path $EngineDir "build")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }
    $bundled = Join-Path $EngineDir "dist\engine\engine.exe"
    if (-not (Test-Path $bundled)) { throw "Expected $bundled - PyInstaller did not produce the engine binary." }
    Write-Host "[1/3] engine bundle ready at $bundled" -ForegroundColor Green
} else {
    Write-Host "[1/3] skipping engine bundle (using existing engine\dist\engine\)" -ForegroundColor DarkYellow
}

if (-not $SkipUi) {
    Write-Host "[2/3] vite + tsc UI build ..." -ForegroundColor Cyan
    Push-Location $UiDir
    try {
        & pnpm run build
        if ($LASTEXITCODE -ne 0) { throw "UI build failed (exit $LASTEXITCODE)" }
    } finally { Pop-Location }

    Write-Host "[3/3] electron-builder packaging installer ..." -ForegroundColor Cyan
    Push-Location $UiDir
    try {
        & pnpm exec electron-builder --win nsis
        if ($LASTEXITCODE -ne 0) { throw "electron-builder failed (exit $LASTEXITCODE)" }
    } finally { Pop-Location }

    $installer = Get-ChildItem (Join-Path $UiDir "release") -Filter "*Setup*.exe" | Select-Object -First 1
    if ($installer) {
        Write-Host ""
        Write-Host "DONE. Installer: $($installer.FullName)" -ForegroundColor Green
        Write-Host "  size: $([math]::Round($installer.Length / 1MB, 1)) MB" -ForegroundColor Green
    } else {
        Write-Warning "Build finished but no *Setup*.exe found in ui\release\."
    }
} else {
    Write-Host "[2/3] [3/3] skipping UI + electron-builder" -ForegroundColor DarkYellow
}
