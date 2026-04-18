from __future__ import annotations

import ctypes
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np


IS_WINDOWS = platform.system() == "Windows"


if IS_WINDOWS:
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
else:
    wintypes = None
    user32 = None
    kernel32 = None


VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F7 = 0x76
VK_F6 = 0x75
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"
HEAD_MOUSE_WINDOW = "Head Mouse"
CONTROLS_WINDOW = "Head Mouse Controls"
TUNING_FILENAME = "head_mouse_settings.json"


@dataclass
class Settings:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    camera_warmup_frames: int = 40
    ffmpeg_warmup_frames: int = 8
    ffmpeg_frame_wait_seconds: float = 0.2
    camera_probe_timeout_seconds: float = 10.0
    camera_backend_attempts: int = 3
    camera_retry_delay_seconds: float = 0.6
    blank_frame_mean_threshold: float = 8.0
    blank_frame_p95_threshold: float = 8.0
    deadzone_x: float = 0.07
    deadzone_y: float = 0.07
    response_power: float = 2.0
    max_speed_x: float = 1500.0
    max_speed_y: float = 1500.0
    smoothing: float = 0.05
    tracking_hz: float = 18.0
    control_loop_hz: float = 144.0
    prediction_gain: float = 1.0
    prediction_max_seconds: float = 0.045
    idle_recenter_seconds: float = 40.0
    idle_motion_threshold: float = 0.020
    min_tracking_scale: float = 0.023
    min_face_detection_confidence: float = 0.35
    min_face_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    lost_tracking_grace_frames: int = 18
    preview_width: int = 220
    preview_height: int = 124
    preview_padding: float = 0.18
    draw_overlay: bool = True


@dataclass
class PoseSample:
    nose_x: float
    nose_y: float
    face_scale: float


@dataclass
class CameraInfo:
    backend_id: int
    backend_name: str


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    sequence: int
    captured_at: float


@dataclass(frozen=True)
class SliderSpec:
    label: str
    attr_name: str
    min_value: float
    max_value: float
    slider_scale: int
    precision: int
    value_display_scale: float = 1.0
    value_display_precision: int | None = None

    @property
    def slider_max(self) -> int:
        return int(round(self.max_value * self.slider_scale))

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.min_value, self.max_value))

    def slider_from_value(self, value: float) -> int:
        return int(round(self.clamp(value) * self.slider_scale))

    def value_from_slider(self, slider_value: int) -> float:
        return self.clamp(slider_value / self.slider_scale)

    def format_value(self, value: float) -> str:
        precision = (
            self.value_display_precision
            if self.value_display_precision is not None
            else self.precision
        )
        display_value = value * self.value_display_scale
        if precision <= 0:
            return str(int(round(display_value)))
        return f"{display_value:.{precision}f}"


class FFmpegCamera:
    def __init__(
        self,
        process: subprocess.Popen,
        width: int,
        height: int,
        device_name: str,
    ) -> None:
        self.process = process
        self.width = width
        self.height = height
        self.device_name = device_name
        self.frame_bytes = width * height * 3
        self._running = True
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_ready = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="ffmpeg-camera", daemon=True)
        self._reader_thread.start()

    def isOpened(self) -> bool:
        return self.process.poll() is None

    def read(self, timeout_seconds: float = 0.6) -> tuple[bool, np.ndarray | None]:
        if not self._frame_ready.wait(timeout=timeout_seconds):
            return False, None
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._running = False
        self._frame_ready.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

    def _reader_loop(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            return
        while self._running and self.process.poll() is None:
            chunk = stdout.read(self.frame_bytes)
            if not chunk or len(chunk) != self.frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            with self._frame_lock:
                self._latest_frame = frame
            self._frame_ready.set()
        self._running = False
        self._frame_ready.set()


def tunable_setting_specs() -> tuple[SliderSpec, ...]:
    return (
        SliderSpec("Speed X", "max_speed_x", 400.0, 4000.0, 1, 0),
        SliderSpec("Speed Y", "max_speed_y", 400.0, 4000.0, 1, 0),
        SliderSpec("Deadzone X", "deadzone_x", 0.0, 0.120, 1000, 0, 1000, 0),
        SliderSpec("Deadzone Y", "deadzone_y", 0.0, 0.120, 1000, 0, 1000, 0),
        SliderSpec("Smoothing", "smoothing", 0.0, 1.0, 100, 0, 100, 0),
        SliderSpec("Curve", "response_power", 0.5, 6.0, 10, 1, 10, 0),
        SliderSpec("Auto-center s", "idle_recenter_seconds", 0.0, 120.0, 10, 1),
        SliderSpec("Idle threshold", "idle_motion_threshold", 0.002, 0.060, 1000, 3, 1000, 0),
        SliderSpec("Face min", "min_tracking_scale", 0.020, 0.120, 1000, 3, 1000, 0),
    )


def tuning_file_path() -> Path:
    return Path(__file__).resolve().parent / TUNING_FILENAME


def hotkey_help_text(include_reset: bool = False) -> str:
    parts = [
        "Ctrl+F6 game",
        "Ctrl+F7 preview",
        "Ctrl+F8 toggle",
        "Ctrl+F9 calibrate",
    ]
    if include_reset:
        parts.append("R reset controls")
    parts.append("Ctrl+F10 exit")
    return " | ".join(parts)


def settings_storage_dict(settings: Settings) -> dict[str, float]:
    return {
        spec.attr_name: float(getattr(settings, spec.attr_name))
        for spec in tunable_setting_specs()
    }


def apply_saved_tuning(settings: Settings, raw_values: dict[str, float]) -> None:
    for spec in tunable_setting_specs():
        raw_value = raw_values.get(spec.attr_name)
        if not isinstance(raw_value, (int, float)):
            continue
        setattr(settings, spec.attr_name, spec.clamp(float(raw_value)))


def load_tunable_settings(settings: Settings, path: Path) -> None:
    if not path.exists():
        return
    try:
        raw_values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(raw_values, dict):
        apply_saved_tuning(settings, raw_values)


def save_tunable_settings(settings: Settings, path: Path) -> None:
    try:
        path.write_text(
            json.dumps(settings_storage_dict(settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def noop_trackbar(_value: int) -> None:
    return


class LiveControls:
    def __init__(self, settings: Settings, tuning_path: Path) -> None:
        self.settings = settings
        self.tuning_path = tuning_path
        self.defaults = Settings()
        self.window_name = CONTROLS_WINDOW
        self._created = False
        self._last_saved = settings_storage_dict(settings)
        self._last_rendered: dict[str, float] | None = None

    def create(self) -> None:
        if self._created:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 620, 520)
        for spec in tunable_setting_specs():
            cv2.createTrackbar(
                spec.label,
                self.window_name,
                spec.slider_from_value(getattr(self.settings, spec.attr_name)),
                spec.slider_max,
                noop_trackbar,
            )
        self._created = True
        self._render()

    def close(self) -> None:
        if not self._created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self._created = False

    def sync_from_settings(self) -> None:
        if not self._created:
            return
        for spec in tunable_setting_specs():
            cv2.setTrackbarPos(
                spec.label,
                self.window_name,
                spec.slider_from_value(getattr(self.settings, spec.attr_name)),
            )

    def refresh(self) -> bool:
        if not self._created:
            return False
        try:
            for spec in tunable_setting_specs():
                slider_value = cv2.getTrackbarPos(spec.label, self.window_name)
                setattr(self.settings, spec.attr_name, spec.value_from_slider(slider_value))
        except cv2.error:
            self._created = False
            return False

        snapshot = settings_storage_dict(self.settings)
        if snapshot != self._last_rendered:
            self._render()
        if snapshot != self._last_saved:
            save_tunable_settings(self.settings, self.tuning_path)
            self._last_saved = snapshot
            return True
        return False

    def reset(self) -> None:
        apply_saved_tuning(self.settings, settings_storage_dict(self.defaults))
        if self._created:
            self.sync_from_settings()
            self._render()
        save_tunable_settings(self.settings, self.tuning_path)
        self._last_saved = settings_storage_dict(self.settings)

    def _render(self) -> None:
        panel = np.full((280, 620, 3), 16, dtype=np.uint8)
        cv2.putText(panel, "Live controls", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Ctrl+F6 game | Ctrl+F7 preview | R reset | Ctrl+F8 toggle", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Settings auto-saved. Auto-center 0.0s disables recentering.", (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        y = 118
        for spec in tunable_setting_specs():
            value = getattr(self.settings, spec.attr_name)
            cv2.putText(
                panel,
                f"{spec.label}: {spec.format_value(value)}",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 24

        cv2.imshow(self.window_name, panel)
        self._last_rendered = settings_storage_dict(self.settings)


class GlobalHotkeys:
    def __init__(self, command_queue: queue.SimpleQueue[str]) -> None:
        self.command_queue = command_queue
        self._thread: threading.Thread | None = None
        self._thread_ready = threading.Event()
        self._thread_id: int | None = None
        self._running = threading.Event()
        self._registrations = {
            1: ("toggle_game_mode", MOD_CONTROL | MOD_NOREPEAT, VK_F6),
            2: ("toggle_preview", MOD_CONTROL | MOD_NOREPEAT, VK_F7),
            3: ("toggle", MOD_CONTROL | MOD_NOREPEAT, VK_F8),
            4: ("calibrate", MOD_CONTROL | MOD_NOREPEAT, VK_F9),
            5: ("quit", MOD_CONTROL | MOD_NOREPEAT, VK_F10),
        }

    def start(self) -> None:
        if not IS_WINDOWS:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        self._thread_ready.wait(timeout=2.0)

    def stop(self) -> None:
        if not IS_WINDOWS or self._thread_id is None:
            return
        self._running.clear()
        user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        assert IS_WINDOWS
        self._thread_id = kernel32.GetCurrentThreadId()
        success = True
        for hotkey_id, (_, modifiers, vk_code) in self._registrations.items():
            if not user32.RegisterHotKey(None, hotkey_id, modifiers, vk_code):
                success = False
        self._thread_ready.set()
        if not success:
            self.command_queue.put("register_hotkey_failed")
            return

        msg = wintypes.MSG()
        while self._running.is_set():
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result <= 0:
                break
            if msg.message == WM_HOTKEY:
                hotkey_name = self._registrations.get(msg.wParam, (None,))[0]
                if hotkey_name:
                    self.command_queue.put(hotkey_name)

        for hotkey_id in self._registrations:
            user32.UnregisterHotKey(None, hotkey_id)


class MouseMover:
    def __init__(self) -> None:
        self._carry_x = 0.0
        self._carry_y = 0.0

    def move_relative(self, dx: float, dy: float) -> None:
        if not IS_WINDOWS:
            return

        self._carry_x += dx
        self._carry_y += dy
        step_x = int(round(self._carry_x))
        step_y = int(round(self._carry_y))
        self._carry_x -= step_x
        self._carry_y -= step_y

        if step_x == 0 and step_y == 0:
            return

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.WPARAM),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("mi", MOUSEINPUT),
            ]

        command = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=step_x,
                dy=step_y,
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE,
                time=0,
                dwExtraInfo=0,
            ),
        )
        user32.SendInput(1, ctypes.byref(command), ctypes.sizeof(INPUT))

    def move_to_center(self) -> None:
        if not IS_WINDOWS:
            return
        self._carry_x = 0.0
        self._carry_y = 0.0
        center_x = int(user32.GetSystemMetrics(0) // 2)
        center_y = int(user32.GetSystemMetrics(1) // 2)
        user32.SetCursorPos(center_x, center_y)


def landmark_mean(landmarks: Iterable, indices: tuple[int, ...]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in indices]
    ys = [landmarks[index].y for index in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def extract_pose(face_landmarks) -> PoseSample | None:
    nose = face_landmarks[1]
    left_anchor = landmark_mean(face_landmarks, (234, 33, 133))
    right_anchor = landmark_mean(face_landmarks, (454, 362, 263))
    face_scale = abs(right_anchor[0] - left_anchor[0])
    if face_scale <= 0:
        return None
    return PoseSample(nose_x=nose.x, nose_y=nose.y, face_scale=face_scale)


def apply_deadzone(value: float, deadzone: float, power: float) -> float:
    distance = abs(value)
    if distance <= deadzone:
        return 0.0
    usable = (distance - deadzone) / max(1e-6, 1.0 - deadzone)
    curved = usable ** power
    return np.sign(value) * float(np.clip(curved, 0.0, 1.0))


def pose_movement_magnitude(previous: PoseSample, current: PoseSample) -> float:
    scale = max((previous.face_scale + current.face_scale) * 0.5, 1e-6)
    delta_x = (current.nose_x - previous.nose_x) / scale
    delta_y = (current.nose_y - previous.nose_y) / scale
    return float(np.hypot(delta_x, delta_y))


class IdleRecentering:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_pose: PoseSample | None = None
        self.last_large_movement_at = time.perf_counter()
        self._recentered_since_movement = False

    def reset(self, now: float | None = None) -> None:
        timestamp = time.perf_counter() if now is None else now
        self.last_pose = None
        self.last_large_movement_at = timestamp
        self._recentered_since_movement = False

    def observe(self, pose: PoseSample | None, now: float) -> float:
        if pose is None:
            self.last_pose = None
            return 0.0
        if self.last_pose is None:
            self.last_pose = pose
            self.last_large_movement_at = now
            self._recentered_since_movement = False
            return 0.0
        movement = pose_movement_magnitude(self.last_pose, pose)
        self.last_pose = pose
        if movement >= self.settings.idle_motion_threshold:
            self.last_large_movement_at = now
            self._recentered_since_movement = False
        return movement

    def should_recenter(self, active: bool, pose: PoseSample | None, now: float) -> bool:
        if not active or pose is None:
            return False
        if self.settings.idle_recenter_seconds <= 0:
            return False
        if self._recentered_since_movement:
            return False
        return (now - self.last_large_movement_at) >= self.settings.idle_recenter_seconds

    def mark_recentered(self, pose: PoseSample, now: float) -> None:
        self.last_pose = pose
        self.last_large_movement_at = now
        self._recentered_since_movement = True


def put_status(frame, text: str, line: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (18, 28 + line * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )


def sync_preview_window(preview_enabled: bool, preview_window_open: bool) -> bool:
    if preview_enabled and not preview_window_open:
        cv2.namedWindow(HEAD_MOUSE_WINDOW, cv2.WINDOW_NORMAL)
        return True
    if not preview_enabled and preview_window_open:
        try:
            cv2.destroyWindow(HEAD_MOUSE_WINDOW)
        except cv2.error:
            pass
        return False
    return preview_window_open


def read_camera_source(camera) -> tuple[bool, np.ndarray | None]:
    if isinstance(camera, FFmpegCamera):
        return camera.read(timeout_seconds=0.35)
    return camera.read()


class LatestFrameCamera:
    def __init__(self, source_camera) -> None:
        self.source_camera = source_camera
        self._running = threading.Event()
        self._running.set()
        self._frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: FramePacket | None = None
        self._sequence = 0
        self._thread = threading.Thread(target=self._reader_loop, name="latest-camera", daemon=True)
        self._thread.start()

    def read(self, timeout_seconds: float = 0.40) -> tuple[bool, FramePacket | None]:
        if not self._frame_ready.wait(timeout=timeout_seconds):
            return False, None
        with self._frame_lock:
            packet = self._latest_frame
        if packet is None:
            return False, None
        return True, packet

    def release(self) -> None:
        self._running.clear()
        try:
            self.source_camera.release()
        except Exception:
            pass
        self._frame_ready.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _reader_loop(self) -> None:
        # Keep only the newest frame so tracking never burns time on stale data.
        while self._running.is_set():
            ok, frame = read_camera_source(self.source_camera)
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                if not self._running.is_set():
                    break
                time.sleep(0.005)
                continue
            packet = FramePacket(frame=frame, sequence=self._sequence, captured_at=time.perf_counter())
            self._sequence += 1
            with self._frame_lock:
                self._latest_frame = packet
            self._frame_ready.set()


class PosePredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.previous_pose: PoseSample | None = None
        self.previous_at: float = 0.0
        self.current_pose: PoseSample | None = None
        self.current_at: float = 0.0

    def reset(self) -> None:
        self.previous_pose = None
        self.previous_at = 0.0
        self.current_pose = None
        self.current_at = 0.0

    def update(self, pose: PoseSample, observed_at: float) -> None:
        if self.current_pose is not None:
            self.previous_pose = self.current_pose
            self.previous_at = self.current_at
        self.current_pose = pose
        self.current_at = observed_at

    def hold(self) -> PoseSample | None:
        return self.current_pose

    def predict(self, now: float) -> PoseSample | None:
        if self.current_pose is None:
            return None
        if self.previous_pose is None or self.current_at <= self.previous_at:
            return self.current_pose

        dt = self.current_at - self.previous_at
        if dt <= 1e-6:
            return self.current_pose

        lead_seconds = max(0.0, now - self.current_at)
        lead_seconds = min(lead_seconds, self.settings.prediction_max_seconds)
        lead_seconds *= self.settings.prediction_gain

        vx = (self.current_pose.nose_x - self.previous_pose.nose_x) / dt
        vy = (self.current_pose.nose_y - self.previous_pose.nose_y) / dt
        predicted_x = float(np.clip(self.current_pose.nose_x + vx * lead_seconds, 0.0, 1.0))
        predicted_y = float(np.clip(self.current_pose.nose_y + vy * lead_seconds, 0.0, 1.0))

        return PoseSample(
            nose_x=predicted_x,
            nose_y=predicted_y,
            face_scale=self.current_pose.face_scale,
        )


def draw_crosshair(frame, sample: PoseSample, neutral: PoseSample | None) -> None:
    height, width = frame.shape[:2]
    x = int(sample.nose_x * width)
    y = int(sample.nose_y * height)
    cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
    cv2.line(frame, (x - 16, y), (x + 16, y), (255, 255, 255), 1)
    cv2.line(frame, (x, y - 16), (x, y + 16), (255, 255, 255), 1)
    if neutral is not None:
        nx = int(neutral.nose_x * width)
        ny = int(neutral.nose_y * height)
        cv2.circle(frame, (nx, ny), 10, (80, 200, 120), 2)


def compute_face_bbox(face_landmarks, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    xs = [landmark.x for landmark in face_landmarks]
    ys = [landmark.y for landmark in face_landmarks]
    min_x = max(0, min(int(min(xs) * frame_width), frame_width - 1))
    max_x = max(0, min(int(max(xs) * frame_width), frame_width - 1))
    min_y = max(0, min(int(min(ys) * frame_height), frame_height - 1))
    max_y = max(0, min(int(max(ys) * frame_height), frame_height - 1))
    return min_x, min_y, max_x, max_y


def expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_width, x2 + pad_x),
        min(frame_height, y2 + pad_y),
    )


def draw_preview_panel(
    frame,
    preview_source,
    settings: Settings,
    tracking_state: str,
    face_scale: float | None,
    camera_label: str,
    brightness_mean: float | None,
) -> None:
    panel_w = settings.preview_width
    panel_h = settings.preview_height
    x2 = frame.shape[1] - 18
    y1 = 18
    x1 = x2 - panel_w
    y2 = y1 + panel_h

    thumbnail = cv2.resize(preview_source, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    frame[y1:y2, x1:x2] = thumbnail
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (245, 245, 245), 1)
    cv2.rectangle(frame, (x1, y2), (x2, y2 + 56), (18, 18, 18), -1)
    cv2.rectangle(frame, (x1, y2), (x2, y2 + 56), (90, 90, 90), 1)
    cv2.putText(frame, "Preview", (x1 + 10, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    color = (80, 220, 120) if tracking_state == "OK" else (120, 200, 255) if tracking_state == "SEARCHING" else (110, 150, 255)
    details = f"Tracking: {tracking_state}"
    if face_scale is not None:
        details += f" | scale {face_scale:.3f}"
    cv2.putText(frame, details, (x1 + 10, y2 + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    footer = camera_label
    if brightness_mean is not None:
        footer += f" | lum {brightness_mean:.0f}"
    cv2.putText(frame, footer, (x1 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)


def beep(frequency: int, duration_ms: int) -> None:
    if not IS_WINDOWS:
        return
    ctypes.windll.kernel32.Beep(frequency, duration_ms)


def configure_camera(
    camera: cv2.VideoCapture,
    settings: Settings,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = 30,
    fourcc_name: str | None = "MJPG",
) -> None:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width or settings.frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height or settings.frame_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if fps is not None:
        camera.set(cv2.CAP_PROP_FPS, fps)
    if IS_WINDOWS and fourcc_name:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_name))


def camera_backend_candidates() -> list[CameraInfo]:
    if not IS_WINDOWS:
        return [CameraInfo(cv2.CAP_ANY, "ANY")]

    backend_override = os.environ.get("HEAD_MOUSE_CAMERA_BACKEND", "").strip().upper()
    backend_map = {
        "FFMPEG": CameraInfo(-1, "FFMPEG"),
        "DSHOW": CameraInfo(cv2.CAP_DSHOW, "DSHOW"),
        "MSMF": CameraInfo(cv2.CAP_MSMF, "MSMF"),
        "ANY": CameraInfo(cv2.CAP_ANY, "ANY"),
    }
    if backend_override in backend_map:
        return [backend_map[backend_override]]

    candidates = [
        CameraInfo(cv2.CAP_DSHOW, "DSHOW"),
        CameraInfo(-1, "FFMPEG"),
        CameraInfo(cv2.CAP_MSMF, "MSMF"),
        CameraInfo(cv2.CAP_ANY, "ANY"),
    ]

    unique: list[CameraInfo] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate.backend_id in seen:
            continue
        seen.add(candidate.backend_id)
        unique.append(candidate)
    return unique


def camera_backend_map() -> dict[str, CameraInfo]:
    return {candidate.backend_name: candidate for candidate in camera_backend_candidates()}


def frame_stats(frame) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.percentile(gray, 95)),
    )


def is_blank_frame(frame, settings: Settings) -> bool:
    mean_value, _std_value, p95_value = frame_stats(frame)
    return (
        mean_value <= settings.blank_frame_mean_threshold
        and p95_value <= settings.blank_frame_p95_threshold
    )


def create_no_window_kwargs() -> dict:
    if not IS_WINDOWS:
        return {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": creationflags}


def create_probe_subprocess_kwargs() -> dict:
    if not IS_WINDOWS:
        return {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": creationflags}


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            **create_no_window_kwargs(),
        )
    else:
        process.kill()


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_capture_profiles(settings: Settings) -> list[tuple[int | None, int | None, int | None, str | None]]:
    # Some Logitech C270 driver states become unstable after high-resolution
    # negotiation attempts. Keep automatic probing on a conservative 640x480
    # profile and only widen the search if we ever add an explicit override.
    return [
        (settings.frame_width, settings.frame_height, 30, "mjpeg"),
        (640, 480, 30, "mjpeg"),
        (640, 480, 30, None),
        (None, None, None, None),
    ]


def opencv_capture_profiles(settings: Settings) -> list[tuple[int, int, int | None, str | None]]:
    if not IS_WINDOWS:
        return [(settings.frame_width, settings.frame_height, None, None)]

    preferred = [
        (settings.frame_width, settings.frame_height, 30, "YUY2"),
        (640, 480, 30, "MJPG"),
        (640, 480, 30, None),
    ]
    unique: list[tuple[int, int, int | None, str | None]] = []
    seen: set[tuple[int, int, int | None, str | None]] = set()
    for profile in preferred:
        if profile in seen:
            continue
        seen.add(profile)
        unique.append(profile)
    return unique


def capture_profile_label(
    width: int | None,
    height: int | None,
    fps: int | None,
    fourcc_name: str | None,
) -> str:
    label = f"{width or 'auto'}x{height or 'auto'}"
    if fps:
        label += f"@{fps}"
    if fourcc_name:
        label += f"/{fourcc_name}"
    return label


def list_ffmpeg_dshow_video_devices(ffmpeg_exe: str) -> list[str]:
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-f",
        "dshow",
        "-list_devices",
        "true",
        "-i",
        "dummy",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        **create_no_window_kwargs(),
    )
    output = (result.stderr or "") + "\n" + (result.stdout or "")

    devices: list[str] = []
    in_video_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if in_video_section and "DirectShow audio devices" in line:
            break
        if "Alternative name" in line:
            continue
        # FFmpeg output format varies by build/version. Some builds print a
        # "DirectShow video devices" header, while newer builds may emit only
        # lines like: [dshow @ ...] "Logi C270 HD WebCam" (video)
        is_video_line = "(video)" in line or in_video_section
        match = re.search(r'"([^"]+)"', line)
        if is_video_line and match:
            devices.append(match.group(1))
    return devices


def preferred_ffmpeg_device_name(devices: list[str]) -> str | None:
    override = os.environ.get("HEAD_MOUSE_CAMERA_DEVICE_NAME", "").strip()
    if override:
        for device in devices:
            if device.lower() == override.lower():
                return device
        return None
    return devices[0] if devices else None


def create_ffmpeg_camera(
    settings: Settings,
    device_name: str,
    width: int | None,
    height: int | None,
    fps: int | None,
    input_codec: str | None,
) -> FFmpegCamera:
    ffmpeg_exe = ffmpeg_executable()
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "dshow",
        "-rtbufsize",
        "256M",
    ]
    if fps is not None:
        command.extend(["-framerate", str(fps)])
    if width is not None and height is not None:
        command.extend(["-video_size", f"{width}x{height}"])
    if input_codec is not None:
        command.extend(["-vcodec", input_codec])
    command.extend([
        "-i",
        f"video={device_name}",
        "-an",
        "-pix_fmt",
        "bgr24",
        "-vcodec",
        "rawvideo",
        "-f",
        "rawvideo",
        "pipe:1",
    ])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=(width or settings.frame_width) * (height or settings.frame_height) * 3 * 2,
        **create_no_window_kwargs(),
    )
    return FFmpegCamera(
        process=process,
        width=width or settings.frame_width,
        height=height or settings.frame_height,
        device_name=device_name,
    )


def try_open_ffmpeg_camera(settings: Settings) -> tuple[FFmpegCamera | None, str]:
    ffmpeg_exe = ffmpeg_executable()
    devices = list_ffmpeg_dshow_video_devices(ffmpeg_exe)
    if not devices:
        return None, "no ffmpeg dshow video devices"

    device_name = preferred_ffmpeg_device_name(devices)
    if device_name is None:
        return None, "requested ffmpeg device not found"

    failures: list[str] = []
    for width, height, fps, input_codec in ffmpeg_capture_profiles(settings):
        camera = create_ffmpeg_camera(settings, device_name, width, height, fps, input_codec)
        usable_frame = None
        saw_nonblank = False
        for _ in range(settings.ffmpeg_warmup_frames):
            ok, frame = camera.read(timeout_seconds=settings.ffmpeg_frame_wait_seconds)
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.03)
                continue
            usable_frame = frame
            if not is_blank_frame(frame, settings):
                saw_nonblank = True
                break
            time.sleep(0.03)

        if saw_nonblank:
            actual_width = width or camera.width
            actual_height = height or camera.height
            actual_fps = fps or 0
            profile_desc = f"{actual_width}x{actual_height}"
            if actual_fps:
                profile_desc += f"@{actual_fps}"
            if input_codec:
                profile_desc += f"/{input_codec}"
            return camera, f"ok:{device_name}:{profile_desc}"

        reason = "black frames" if usable_frame is not None else "no frames"
        profile_desc = f"{width or 'auto'}x{height or 'auto'}"
        if fps:
            profile_desc += f"@{fps}"
        if input_codec:
            profile_desc += f"/{input_codec}"
        failures.append(profile_desc + ":" + reason)
        camera.release()

    return None, "; ".join(failures)


def try_open_camera_backend(
    settings: Settings,
    candidate: CameraInfo,
) -> tuple[cv2.VideoCapture | FFmpegCamera | None, str]:
    if candidate.backend_name == "FFMPEG":
        return try_open_ffmpeg_camera(settings)

    failures: list[str] = []
    for width, height, fps, fourcc_name in opencv_capture_profiles(settings):
        camera = cv2.VideoCapture(settings.camera_index, candidate.backend_id)
        if not camera.isOpened():
            return None, "open failed"

        configure_camera(
            camera,
            settings,
            width=width,
            height=height,
            fps=fps,
            fourcc_name=fourcc_name,
        )
        usable_frame = None
        saw_nonblank = False
        for _ in range(settings.camera_warmup_frames):
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.03)
                continue
            usable_frame = frame
            if not is_blank_frame(frame, settings):
                saw_nonblank = True
                break
            time.sleep(0.03)

        profile_label = capture_profile_label(width, height, fps, fourcc_name)
        if saw_nonblank:
            return camera, f"ok:{profile_label}"

        reason = "black frames" if usable_frame is not None else "no frames"
        failures.append(f"{profile_label}:{reason}")
        camera.release()

    return None, "; ".join(failures)


def camera_success_info(candidate: CameraInfo, reason: str) -> CameraInfo:
    if not reason.startswith("ok:"):
        return candidate

    details = reason[3:]
    if candidate.backend_name == "FFMPEG":
        parts = details.split(":")
        details = parts[-1] if len(parts) >= 2 else details
    return CameraInfo(candidate.backend_id, f"{candidate.backend_name} {details}")


def probe_camera_backend_subprocess(settings: Settings, candidate: CameraInfo) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-camera",
        candidate.backend_name,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **create_probe_subprocess_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=settings.camera_probe_timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return False, "timeout"

    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    details = stdout or stderr or f"exit {process.returncode}"
    return process.returncode == 0, details


def open_camera(settings: Settings) -> tuple[cv2.VideoCapture, CameraInfo]:
    failures: list[str] = []
    for candidate in camera_backend_candidates():
        if candidate.backend_name == "FFMPEG":
            print(f"Trying camera backend {candidate.backend_name} ...", flush=True)
            camera, reason = try_open_camera_backend(settings, candidate)
            if camera is not None:
                camera_info = camera_success_info(candidate, reason)
                print(f"Camera ready with backend {camera_info.backend_name}.", flush=True)
                return camera, camera_info

            failures.append(f"{candidate.backend_name}: {reason}")
            print(f"Backend {candidate.backend_name} not usable: {reason}", flush=True)
            continue

        for attempt_index in range(settings.camera_backend_attempts):
            attempt_label = f"{attempt_index + 1}/{settings.camera_backend_attempts}"
            print(f"Trying camera backend {candidate.backend_name} (attempt {attempt_label}) ...", flush=True)

            probe_ok, details = probe_camera_backend_subprocess(settings, candidate)
            if not probe_ok:
                if attempt_index < settings.camera_backend_attempts - 1:
                    print(
                        f"Backend {candidate.backend_name} not ready: {details}. "
                        f"Waiting {settings.camera_retry_delay_seconds:.1f}s and retrying...",
                        flush=True,
                    )
                    time.sleep(settings.camera_retry_delay_seconds)
                    continue

                failures.append(f"{candidate.backend_name}: {details}")
                print(f"Backend {candidate.backend_name} not usable: {details}", flush=True)
                break

            camera, reason = try_open_camera_backend(settings, candidate)
            if camera is not None:
                camera_info = camera_success_info(candidate, reason)
                print(f"Camera ready with backend {camera_info.backend_name}.", flush=True)
                return camera, camera_info

            if attempt_index < settings.camera_backend_attempts - 1:
                print(
                    f"Backend {candidate.backend_name} not ready: {reason}. "
                    f"Waiting {settings.camera_retry_delay_seconds:.1f}s and retrying...",
                    flush=True,
                )
                time.sleep(settings.camera_retry_delay_seconds)
                continue

            failures.append(f"{candidate.backend_name}: {reason}")
            print(f"Backend {candidate.backend_name} not usable: {reason}", flush=True)

    raise RuntimeError(
        "Unable to open a valid camera feed. Attempts: " + ", ".join(failures)
    )


def probe_camera_main(backend_name: str) -> int:
    settings = Settings()
    backend = camera_backend_map().get(backend_name.upper())
    if backend is None:
        print("Unknown backend", flush=True)
        return 2

    camera = None
    try:
        camera, reason = try_open_camera_backend(settings, backend)
        if camera is None:
            print(reason, flush=True)
            return 1
        print("ok", flush=True)
        return 0
    except Exception as exc:
        print(str(exc), flush=True)
        return 1
    finally:
        if camera is not None:
            camera.release()


def list_cameras_main() -> int:
    try:
        devices = list_ffmpeg_dshow_video_devices(ffmpeg_executable())
    except Exception as exc:
        print(f"Webcam list error: {exc}", flush=True)
        return 1

    if not devices:
        print("No webcams detected via FFMPEG dshow.", flush=True)
        return 1

    print("Webcams detected via FFMPEG dshow:", flush=True)
    for index, device in enumerate(devices, start=1):
        print(f"{index}. {device}", flush=True)
    return 0


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / MODEL_FILENAME


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"Downloading MediaPipe model to {model_path} ...")
    with urlopen(MODEL_URL, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with model_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = 100.0 * downloaded / total_size
                    print(f"\rModel download: {percent:5.1f}%", end="", flush=True)
    if total_size:
        print()
    return model_path


def create_landmarker(model_path: Path, settings: Settings):
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=settings.min_face_detection_confidence,
        min_face_presence_confidence=settings.min_face_presence_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def main() -> int:
    if not IS_WINDOWS:
        print("This script must run on Windows.", file=sys.stderr)
        return 1

    settings = Settings()
    tuning_path = tuning_file_path()
    load_tunable_settings(settings, tuning_path)
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    hotkeys = GlobalHotkeys(commands)
    mouse = MouseMover()
    idle_recentering = IdleRecentering(settings)
    pose_predictor = PosePredictor(settings)

    try:
        source_camera, camera_info = open_camera(settings)
    except Exception as exc:
        print(f"Unable to open webcam: {exc}", file=sys.stderr)
        return 1

    try:
        model_path = ensure_model(default_model_path())
        landmarker = create_landmarker(model_path, settings)
    except Exception as exc:
        source_camera.release()
        print(f"Unable to initialize FaceLandmarker: {exc}", file=sys.stderr)
        return 1

    camera = LatestFrameCamera(source_camera)

    active = False
    game_mode = False
    preview_enabled = True
    preview_window_open = False
    neutral_pose: PoseSample | None = None
    smoothed_x = 0.0
    smoothed_y = 0.0
    status_message = hotkey_help_text(include_reset=True)
    status_deadline = time.monotonic() + 5.0
    last_frame_at = time.perf_counter()
    last_timestamp_ms = 0
    last_face_bbox: tuple[int, int, int, int] | None = None
    lost_tracking_frames = settings.lost_tracking_grace_frames + 1
    brightness_mean: float | None = None
    last_inferred_sequence = -1
    last_inference_at = 0.0
    inference_period = (1.0 / settings.tracking_hz) if settings.tracking_hz > 0 else 0.0
    control_period = (1.0 / settings.control_loop_hz) if settings.control_loop_hz > 0 else 0.0

    hotkeys.start()
    controls = LiveControls(settings, tuning_path)
    controls.create()
    preview_window_open = sync_preview_window(preview_enabled, preview_window_open)

    try:
        while True:
            loop_started_at = time.perf_counter()
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break

                if command == "toggle_game_mode":
                    game_mode = not game_mode
                    preview_window_open = sync_preview_window(preview_enabled and not game_mode, preview_window_open)
                    if game_mode:
                        controls.close()
                        status_message = "Game Mode active: windows disabled, tracking-only mode."
                        beep(540, 90)
                    else:
                        controls.create()
                        status_message = "Game Mode deactivated."
                        beep(960, 90)
                    status_deadline = time.monotonic() + 2.5
                elif command == "toggle_preview":
                    preview_enabled = not preview_enabled
                    preview_window_open = sync_preview_window(preview_enabled and not game_mode, preview_window_open)
                    if preview_enabled:
                        status_message = "Preview enabled."
                        beep(980, 90)
                    else:
                        status_message = "Preview disabled: tracking only for lower latency."
                        beep(680, 90)
                    status_deadline = time.monotonic() + 2.5
                elif command == "toggle":
                    active = not active
                    idle_recentering.reset(time.perf_counter())
                    if active and neutral_pose is None:
                        status_message = "Active: center your face and press Ctrl+F9 to recalibrate."
                    else:
                        status_message = "Tracking ON" if active else "Tracking OFF"
                    status_deadline = time.monotonic() + 2.5
                    beep(1120 if active else 560, 120)
                elif command == "calibrate":
                    neutral_pose = None
                    idle_recentering.reset(time.perf_counter())
                    status_message = "Calibration requested: stay still with face in the center."
                    status_deadline = time.monotonic() + 2.5
                    beep(860, 90)
                elif command == "quit":
                    return 0
                elif command == "register_hotkey_failed":
                    status_message = "Global hotkeys not registered: close apps using Ctrl+F6/F7/F8/F9/F10."
                    status_deadline = time.monotonic() + 6.0

            if not game_mode and controls._created:
                controls_changed = controls.refresh()
                if controls_changed:
                    status_message = "Controls updated."
                    status_deadline = time.monotonic() + 1.5

            ok, packet = camera.read(timeout_seconds=0.45)
            if not ok or packet is None:
                status_message = "Webcam frame unavailable."
                status_deadline = time.monotonic() + 2.5
                if control_period > 0:
                    remaining = control_period - (time.perf_counter() - loop_started_at)
                    if remaining > 0:
                        time.sleep(remaining)
                continue

            frame = cv2.flip(packet.frame, 1)
            visuals_enabled = preview_enabled and not game_mode and settings.draw_overlay
            raw_frame = frame.copy() if visuals_enabled else None
            brightness_mean = None
            if raw_frame is not None:
                brightness_mean, _brightness_std, _brightness_p95 = frame_stats(raw_frame)

            inferred_pose: PoseSample | None = None
            inference_ran = False
            # Landmarker inference is the expensive part, so we run it on fresh
            # frames at a capped rate and predict in between.
            should_infer = (
                pose_predictor.current_pose is None
                or (
                    packet.sequence != last_inferred_sequence
                    and (inference_period <= 0 or (time.perf_counter() - last_inference_at) >= inference_period)
                )
            )
            if should_infer:
                inference_ran = True
                last_inference_at = time.perf_counter()
                last_inferred_sequence = packet.sequence
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp_ms = int(packet.captured_at * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.face_landmarks:
                    face_landmarks = results.face_landmarks[0]
                    inferred_pose = extract_pose(face_landmarks)
                    last_face_bbox = compute_face_bbox(face_landmarks, frame.shape[1], frame.shape[0])
                    if inferred_pose is not None and inferred_pose.face_scale >= settings.min_tracking_scale:
                        pose_predictor.update(inferred_pose, packet.captured_at)
                        lost_tracking_frames = 0
                    else:
                        lost_tracking_frames += 1
                else:
                    lost_tracking_frames += 1

            now = time.perf_counter()
            dt = min(now - last_frame_at, 0.1)
            last_frame_at = now

            if lost_tracking_frames == 0:
                pose = pose_predictor.predict(now)
            elif lost_tracking_frames <= settings.lost_tracking_grace_frames:
                pose = pose_predictor.hold()
            else:
                pose_predictor.reset()
                pose = None

            face_scale = pose.face_scale if pose is not None else inferred_pose.face_scale if inferred_pose is not None else None

            if pose and pose.face_scale >= settings.min_tracking_scale:
                idle_recentering.observe(pose, now)
                if neutral_pose is None:
                    neutral_pose = pose
                    status_message = "Calibrated."
                    status_deadline = time.monotonic() + 1.8

                just_recentered = False
                if idle_recentering.should_recenter(active, pose, now):
                    mouse.move_to_center()
                    neutral_pose = pose
                    smoothed_x = 0.0
                    smoothed_y = 0.0
                    idle_recentering.mark_recentered(pose, now)
                    just_recentered = True
                    status_message = "Mouse centered after inactivity; new neutral point captured."
                    status_deadline = time.monotonic() + 2.2
                    beep(740, 70)

                raw_x = (pose.nose_x - neutral_pose.nose_x) / pose.face_scale
                raw_y = (pose.nose_y - neutral_pose.nose_y) / pose.face_scale
                smoothed_x = (1.0 - settings.smoothing) * smoothed_x + settings.smoothing * raw_x
                smoothed_y = (1.0 - settings.smoothing) * smoothed_y + settings.smoothing * raw_y

                response_x = apply_deadzone(smoothed_x, settings.deadzone_x, settings.response_power)
                response_y = apply_deadzone(smoothed_y, settings.deadzone_y, settings.response_power)

                if active and not just_recentered:
                    dx = response_x * settings.max_speed_x * dt
                    dy = response_y * settings.max_speed_y * dt
                    mouse.move_relative(dx, dy)

                if visuals_enabled:
                    draw_crosshair(frame, pose, neutral_pose)
                    cv2.rectangle(frame, (10, 10), (530, 145), (15, 15, 15), -1)
                    cv2.rectangle(frame, (10, 10), (530, 145), (90, 90, 90), 1)
                    put_status(frame, "Head Mouse", 0, (255, 255, 255))
                    put_status(frame, f"State: {'TRACKING' if active else 'PAUSED'}", 1, (80, 220, 120) if active else (120, 180, 255))
                    put_status(frame, f"Offset: X {smoothed_x:+.3f} | Y {smoothed_y:+.3f}", 2, (220, 220, 220))
                    if time.monotonic() < status_deadline:
                        put_status(frame, status_message, 3, (255, 210, 120))
            else:
                idle_recentering.observe(None, now)
                smoothed_x *= 0.88
                smoothed_y *= 0.88
                if visuals_enabled:
                    cv2.rectangle(frame, (10, 10), (620, 120), (15, 15, 15), -1)
                    cv2.rectangle(frame, (10, 10), (620, 120), (90, 90, 90), 1)
                    if raw_frame is not None and is_blank_frame(raw_frame, settings):
                        put_status(frame, "Camera feed looks dark. Try another webcam app or backend.", 0, (110, 150, 255))
                        put_status(frame, f"Current backend: {camera_info.backend_name}", 1, (220, 220, 220))
                    elif inference_ran and lost_tracking_frames <= settings.lost_tracking_grace_frames:
                        put_status(frame, "Tracking unstable: holding and searching for a stronger pose.", 0, (120, 200, 255))
                        put_status(frame, "Move slowly and keep your face in frame for a moment.", 1, (220, 220, 220))
                    else:
                        put_status(frame, "Face not tracked. Press Ctrl+F9 and stay in the frame.", 0, (120, 180, 255))
                        put_status(frame, hotkey_help_text(), 1, (220, 220, 220))
                    if time.monotonic() < status_deadline:
                        put_status(frame, status_message, 2, (255, 210, 120))

            if visuals_enabled and last_face_bbox is not None and raw_frame is not None:
                x1, y1, x2, y2 = expand_bbox(
                    last_face_bbox,
                    frame.shape[1],
                    frame.shape[0],
                    settings.preview_padding,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120) if pose else (120, 180, 255), 2)
                preview_source = raw_frame[y1:y2, x1:x2]
                if preview_source.size == 0:
                    preview_source = raw_frame
            elif visuals_enabled and raw_frame is not None:
                preview_source = raw_frame

            if visuals_enabled:
                tracking_state = "OK" if pose and pose.face_scale >= settings.min_tracking_scale else "SEARCHING" if lost_tracking_frames <= settings.lost_tracking_grace_frames else "LOST"
                draw_preview_panel(
                    frame,
                    preview_source,
                    settings,
                    tracking_state,
                    face_scale,
                    camera_info.backend_name,
                    brightness_mean,
                )

            preview_window_open = sync_preview_window(preview_enabled and not game_mode, preview_window_open)
            if preview_enabled and not game_mode:
                cv2.imshow(HEAD_MOUSE_WINDOW, frame)
            if preview_window_open or controls._created:
                wait_ms = 1
                if control_period > 0:
                    remaining = control_period - (time.perf_counter() - loop_started_at)
                    if remaining > 0:
                        wait_ms = max(1, int(round(remaining * 1000)))
                key = cv2.waitKey(wait_ms) & 0xFF
            else:
                if control_period > 0:
                    remaining = control_period - (time.perf_counter() - loop_started_at)
                    if remaining > 0:
                        time.sleep(remaining)
                key = -1
            if key == 27:
                return 0
            if key in (ord("c"), ord("C")):
                neutral_pose = pose if pose else None
                idle_recentering.reset(time.perf_counter())
                status_message = "Manual calibration."
                status_deadline = time.monotonic() + 1.8
            if key in (ord("r"), ord("R")):
                controls.reset()
                status_message = "Controls reset to defaults."
                status_deadline = time.monotonic() + 2.0
    finally:
        save_tunable_settings(settings, tuning_path)
        controls.close()
        hotkeys.stop()
        landmarker.close()
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--probe-camera":
        raise SystemExit(probe_camera_main(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--list-cameras":
        raise SystemExit(list_cameras_main())
    raise SystemExit(main())
