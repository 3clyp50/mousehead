param(
    [ValidateSet("AUTO", "FFMPEG", "DSHOW", "MSMF", "ANY")]
    [string]$CameraBackend = "AUTO",
    [string]$CameraDeviceName = "",
    [switch]$ListCameras
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$requirementsFile = Join-Path $projectRoot "requirements.txt"
$bootstrapOk = $false

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment..."
    py -3.12 -m venv $venvPath
}

Write-Host "Checking Python dependencies..."
& $pythonExe -c "import imageio_ffmpeg, mediapipe, numpy; print('Python dependencies OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required dependencies..."
    & $pythonExe -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
} else {
    $bootstrapOk = $true
}

if (-not $bootstrapOk) {
    Write-Host "Dependencies installed successfully."
}

if ($CameraDeviceName) {
    $env:HEAD_MOUSE_CAMERA_DEVICE_NAME = $CameraDeviceName
} elseif (Test-Path Env:HEAD_MOUSE_CAMERA_DEVICE_NAME) {
    Remove-Item Env:HEAD_MOUSE_CAMERA_DEVICE_NAME
}

if ($ListCameras) {
    Write-Host "Listing webcams via FFMPEG backend..."
    & $pythonExe (Join-Path $projectRoot "head_mouse.py") --list-cameras
    exit $LASTEXITCODE
}

if ($CameraBackend -eq "AUTO") {
    if (Test-Path Env:HEAD_MOUSE_CAMERA_BACKEND) {
        Remove-Item Env:HEAD_MOUSE_CAMERA_BACKEND
    }
    Write-Host "Starting Head Mouse with auto camera backend selection..."
} else {
    $env:HEAD_MOUSE_CAMERA_BACKEND = $CameraBackend
    Write-Host "Starting Head Mouse with camera backend $CameraBackend..."
}
& $pythonExe (Join-Path $projectRoot "head_mouse.py")
