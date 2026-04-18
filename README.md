# mousehead (Webcam-to-Mouse Head Control)

![mousehead thumbnail](thumb.png)

Stop controlling in-game cameras with your mouse. This project turns a webcam into a head-tracking mouse input for Windows.

- Look straight, the pointer stays still (or at least it tries to).
- Move your head, the pointer moves with smooth dead-zone and response controls.
- Works with a global preview window and a separate live-controls panel.

## Quick Start

From PowerShell in this folder:

```powershell
.\run_head_mouse.ps1
```

The script will:

1. Create `.venv` if missing.
2. Install dependencies from `requirements.txt`.
3. Start webcam capture and the MediaPipe face-landmarker model (downloaded automatically on first run).

## Requirements

- Windows 10/11 x86-64
- Webcam with direct access
- Python dependencies from `requirements.txt`

## Hotkeys

- `Ctrl+F6`: Toggle **Game Mode**
- `Ctrl+F7`: Toggle preview window
- `Ctrl+F8`: Toggle tracking
- `Ctrl+F9`: Recalibrate current neutral pose
- `Ctrl+F10`: Exit
- `R`: Reset tuning controls to defaults
- `C`: Manual recalibrate (shortcut in-app)

### Camera backend options

```powershell
.\run_head_mouse.ps1 -CameraBackend AUTO
.\run_head_mouse.ps1 -CameraBackend FFMPEG
.\run_head_mouse.ps1 -CameraBackend MSMF
.\run_head_mouse.ps1 -CameraBackend DSHOW
.\run_head_mouse.ps1 -CameraBackend ANY
```

### List available webcams (FFMPEG)

```powershell
.\run_head_mouse.ps1 -ListCameras
```

### Select a webcam by name (FFMPEG)

```powershell
.\run_head_mouse.ps1 -CameraBackend FFMPEG -CameraDeviceName "USB Camera"
```

## Live Controls and Defaults

The control panel stores values in `head_mouse_settings.json` and loads them at start.

Default values shipped with this repo are:

- Speed X: **1500**
- Speed Y: **1500**
- Deadzone X: **70** (shown as 70, internally 0.070)
- Deadzone Y: **70** (shown as 70, internally 0.070)
- Smoothing: **5** (shown as 5, internally 0.05)
- Curve: **20** (shown as 20, internally 2.0)
- Auto-center seconds: **40**
- Idle threshold: **20** (shown as 20, internally 0.020)
- Face min: **23** (shown as 23, internally 0.023)

These defaults are intentionally editable and safe for sharing.

## Manual run (no PowerShell wrapper)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\head_mouse.py
```

## Notes on behavior

- Auto-centering is triggered after inactivity, using the configured seconds and movement threshold.
- `FFMPEG` backend tries `640x480` first to avoid unstable high-resolution negotiation on older webcams.
- The preview window and control panel can be hidden to reduce latency during gameplay.
- Tuning values are stored in `head_mouse_settings.json` and can be safely version-controlled or edited manually.
- Tested on Logitech C270 from 2010 on Windows 10 x86 :-)
