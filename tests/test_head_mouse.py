from __future__ import annotations

import os
import queue
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

import head_mouse


@pytest.fixture(autouse=True)
def clear_backend_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEAD_MOUSE_CAMERA_BACKEND", raising=False)


def test_camera_backend_candidates_default_order() -> None:
    candidates = head_mouse.camera_backend_candidates()
    names = [candidate.backend_name for candidate in candidates]
    assert names == ["DSHOW", "FFMPEG", "MSMF", "ANY"]


def test_camera_backend_candidates_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEAD_MOUSE_CAMERA_BACKEND", "MSMF")
    candidates = head_mouse.camera_backend_candidates()
    assert [candidate.backend_name for candidate in candidates] == ["MSMF"]


def test_list_ffmpeg_dshow_video_devices_parses_names(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = """
    [dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
    [dshow @ 000001]  "USB Camera"
    [dshow @ 000001]     Alternative name "@device_pnp_foo"
    [dshow @ 000001]  "Virtual Camera"
    [dshow @ 000001] DirectShow audio devices
    """

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="", stderr=stderr)

    monkeypatch.setattr(head_mouse.subprocess, "run", fake_run)
    devices = head_mouse.list_ffmpeg_dshow_video_devices("ffmpeg.exe")
    assert devices == ["USB Camera", "Virtual Camera"]


def test_ffmpeg_capture_profiles_prefers_safe_640p_mjpeg() -> None:
    settings = head_mouse.Settings()
    profiles = head_mouse.ffmpeg_capture_profiles(settings)
    assert profiles[0] == (640, 480, 30, "mjpeg")


def test_opencv_capture_profiles_prefers_safe_640p_yuy2() -> None:
    settings = head_mouse.Settings()
    profiles = head_mouse.opencv_capture_profiles(settings)
    assert profiles[0] == (640, 480, 30, "YUY2")


def test_global_hotkeys_require_ctrl_modifier() -> None:
    hotkeys = head_mouse.GlobalHotkeys(queue.SimpleQueue())
    registrations = list(hotkeys._registrations.values())
    assert registrations
    commands = {name for name, _modifiers, _vk_code in registrations}
    assert commands == {"toggle_game_mode", "toggle_preview", "toggle", "calibrate", "quit"}
    for _name, modifiers, _vk_code in registrations:
        assert modifiers & head_mouse.MOD_CONTROL
        assert modifiers & head_mouse.MOD_NOREPEAT


def test_hotkey_help_text_mentions_preview() -> None:
    text = head_mouse.hotkey_help_text(include_reset=True)
    assert "Ctrl+F6 game" in text
    assert "Ctrl+F7 preview" in text
    assert "Ctrl+F8 toggle" in text


def test_slider_spec_roundtrip() -> None:
    spec = next(s for s in head_mouse.tunable_setting_specs() if s.attr_name == "deadzone_x")
    slider_value = spec.slider_from_value(0.07)
    assert spec.value_from_slider(slider_value) == pytest.approx(0.07, abs=0.001)


def test_apply_saved_tuning_clamps_values() -> None:
    settings = head_mouse.Settings()
    head_mouse.apply_saved_tuning(
        settings,
        {
            "max_speed_x": 99999,
            "deadzone_x": -1,
            "smoothing": 0.45,
        },
    )
    assert settings.max_speed_x == 4000.0
    assert settings.deadzone_x == 0.0
    assert settings.smoothing == 0.45


def test_default_tunable_settings_match_public_defaults() -> None:
    settings = head_mouse.Settings()
    assert settings.max_speed_x == 1500.0
    assert settings.max_speed_y == 1500.0
    assert settings.deadzone_x == pytest.approx(0.07)
    assert settings.deadzone_y == pytest.approx(0.07)
    assert settings.smoothing == pytest.approx(0.05)
    assert settings.response_power == pytest.approx(2.0)
    assert settings.idle_recenter_seconds == 40.0
    assert settings.idle_motion_threshold == pytest.approx(0.020)
    assert settings.min_tracking_scale == pytest.approx(0.023)


def test_save_and_load_tunable_settings_roundtrip(tmp_path: Path) -> None:
    settings = head_mouse.Settings(
        max_speed_x=2300.0,
        deadzone_x=0.031,
        smoothing=0.33,
        idle_recenter_seconds=9.5,
        idle_motion_threshold=0.014,
    )
    path = tmp_path / "head_mouse_settings.json"
    head_mouse.save_tunable_settings(settings, path)

    loaded = head_mouse.Settings()
    head_mouse.load_tunable_settings(loaded, path)
    assert loaded.max_speed_x == 2300.0
    assert loaded.deadzone_x == pytest.approx(0.031)
    assert loaded.smoothing == pytest.approx(0.33)
    assert loaded.idle_recenter_seconds == pytest.approx(9.5)
    assert loaded.idle_motion_threshold == pytest.approx(0.014)


def test_idle_recentering_triggers_once_after_timeout() -> None:
    settings = head_mouse.Settings(idle_recenter_seconds=7.0, idle_motion_threshold=0.010)
    tracker = head_mouse.IdleRecentering(settings)
    pose = head_mouse.PoseSample(nose_x=0.50, nose_y=0.50, face_scale=0.25)

    assert tracker.observe(pose, now=1.0) == 0.0
    assert tracker.should_recenter(True, pose, now=7.9) is False
    assert tracker.should_recenter(True, pose, now=8.1) is True

    tracker.mark_recentered(pose, now=8.1)
    assert tracker.should_recenter(True, pose, now=20.0) is False


def test_idle_recentering_rearms_after_large_motion() -> None:
    settings = head_mouse.Settings(idle_recenter_seconds=7.0, idle_motion_threshold=0.010)
    tracker = head_mouse.IdleRecentering(settings)
    pose_a = head_mouse.PoseSample(nose_x=0.50, nose_y=0.50, face_scale=0.25)
    pose_b = head_mouse.PoseSample(nose_x=0.506, nose_y=0.50, face_scale=0.25)

    tracker.observe(pose_a, now=1.0)
    tracker.mark_recentered(pose_a, now=8.0)
    movement = tracker.observe(pose_b, now=9.0)

    assert movement > settings.idle_motion_threshold
    assert tracker.should_recenter(True, pose_b, now=16.2) is True


def test_pose_predictor_extrapolates_forward() -> None:
    settings = head_mouse.Settings(prediction_gain=1.0, prediction_max_seconds=0.050)
    predictor = head_mouse.PosePredictor(settings)
    pose_a = head_mouse.PoseSample(nose_x=0.50, nose_y=0.50, face_scale=0.25)
    pose_b = head_mouse.PoseSample(nose_x=0.55, nose_y=0.52, face_scale=0.25)

    predictor.update(pose_a, observed_at=1.00)
    predictor.update(pose_b, observed_at=1.10)
    predicted = predictor.predict(now=1.14)

    assert predicted is not None
    assert predicted.nose_x > pose_b.nose_x
    assert predicted.nose_y > pose_b.nose_y
    assert predicted.face_scale == pose_b.face_scale


def test_latest_frame_camera_reads_recent_frame() -> None:
    class FakeSource:
        def __init__(self) -> None:
            self.frames = [
                np.full((2, 2, 3), 10, dtype=np.uint8),
                np.full((2, 2, 3), 20, dtype=np.uint8),
                np.full((2, 2, 3), 30, dtype=np.uint8),
            ]
            self.released = False

        def read(self):
            if self.frames:
                frame = self.frames.pop(0)
                time.sleep(0.01)
                return True, frame
            time.sleep(0.01)
            return False, None

        def release(self) -> None:
            self.released = True

    source = FakeSource()
    camera = head_mouse.LatestFrameCamera(source)
    try:
        time.sleep(0.05)
        ok, packet = camera.read(timeout_seconds=0.2)
        assert ok is True
        assert packet is not None
        assert packet.sequence >= 1
        assert int(packet.frame[0, 0, 0]) in {20, 30}
    finally:
        camera.release()
    assert source.released is True


def test_sync_preview_window_creates_and_destroys(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(head_mouse.cv2, "namedWindow", lambda name, _flags: events.append(("open", name)))
    monkeypatch.setattr(head_mouse.cv2, "destroyWindow", lambda name: events.append(("close", name)))

    open_state = head_mouse.sync_preview_window(True, False)
    closed_state = head_mouse.sync_preview_window(False, True)

    assert open_state is True
    assert closed_state is False
    assert events == [
        ("open", head_mouse.HEAD_MOUSE_WINDOW),
        ("close", head_mouse.HEAD_MOUSE_WINDOW),
    ]


def test_probe_camera_backend_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings(camera_probe_timeout_seconds=0.01)
    candidate = head_mouse.CameraInfo(700, "DSHOW")

    class FakeProcess:
        pid = 4242

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="python child", timeout=timeout)

        @property
        def returncode(self):
            return None

    monkeypatch.setattr(head_mouse.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(head_mouse, "terminate_process_tree", lambda process: None)

    ok, details = head_mouse.probe_camera_backend_subprocess(settings, candidate)
    assert ok is False
    assert details == "timeout"


def test_open_camera_skips_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings()
    ffmpeg = head_mouse.CameraInfo(-1, "FFMPEG")
    msmf = head_mouse.CameraInfo(1400, "MSMF")
    fake_camera = object()

    monkeypatch.setattr(head_mouse, "camera_backend_candidates", lambda: [ffmpeg, msmf])

    probe_results = {
        "FFMPEG": (False, "timeout"),
        "MSMF": (True, "ok"),
    }
    monkeypatch.setattr(
        head_mouse,
        "probe_camera_backend_subprocess",
        lambda settings_arg, candidate: probe_results[candidate.backend_name],
    )

    open_results = {
        "MSMF": (fake_camera, "ok"),
    }
    monkeypatch.setattr(
        head_mouse,
        "try_open_camera_backend",
        lambda settings_arg, candidate: open_results.get(candidate.backend_name, (None, "open failed")),
    )

    camera, info = head_mouse.open_camera(settings)
    assert camera is fake_camera
    assert info.backend_name == "MSMF"


def test_open_camera_retries_timed_out_probe_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings(camera_backend_attempts=3, camera_retry_delay_seconds=0.0)
    dshow = head_mouse.CameraInfo(700, "DSHOW")
    fake_camera = object()
    probe_calls: list[str] = []
    responses = [(False, "timeout"), (False, "timeout"), (True, "ok")]

    monkeypatch.setattr(head_mouse, "camera_backend_candidates", lambda: [dshow])

    def fake_probe(_settings, candidate):
        probe_calls.append(candidate.backend_name)
        return responses[len(probe_calls) - 1]

    monkeypatch.setattr(head_mouse, "probe_camera_backend_subprocess", fake_probe)
    monkeypatch.setattr(head_mouse, "try_open_camera_backend", lambda _settings, _candidate: (fake_camera, "ok:640x480@30/YUY2"))
    monkeypatch.setattr(head_mouse.time, "sleep", lambda *_args, **_kwargs: None)

    camera, info = head_mouse.open_camera(settings)
    assert camera is fake_camera
    assert info.backend_name == "DSHOW 640x480@30/YUY2"
    assert probe_calls == ["DSHOW", "DSHOW", "DSHOW"]


def test_open_camera_retries_backend_open_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings(camera_backend_attempts=3, camera_retry_delay_seconds=0.0)
    dshow = head_mouse.CameraInfo(700, "DSHOW")
    fake_camera = object()
    open_calls: list[int] = []

    monkeypatch.setattr(head_mouse, "camera_backend_candidates", lambda: [dshow])
    monkeypatch.setattr(head_mouse, "probe_camera_backend_subprocess", lambda _settings, _candidate: (True, "ok"))

    def fake_open(_settings, _candidate):
        open_calls.append(len(open_calls))
        if len(open_calls) < 3:
            return None, "no frames"
        return fake_camera, "ok:640x480@30/YUY2"

    monkeypatch.setattr(head_mouse, "try_open_camera_backend", fake_open)
    monkeypatch.setattr(head_mouse.time, "sleep", lambda *_args, **_kwargs: None)

    camera, info = head_mouse.open_camera(settings)
    assert camera is fake_camera
    assert info.backend_name == "DSHOW 640x480@30/YUY2"
    assert len(open_calls) == 3


def test_probe_camera_main_unknown_backend() -> None:
    assert head_mouse.probe_camera_main("NOPE") == 2


def test_list_cameras_main_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(head_mouse, "ffmpeg_executable", lambda: "ffmpeg.exe")
    monkeypatch.setattr(head_mouse, "list_ffmpeg_dshow_video_devices", lambda _exe: ["USB Camera"])
    assert head_mouse.list_cameras_main() == 0
    output = capsys.readouterr().out
    assert "USB Camera" in output


def test_try_open_camera_backend_returns_black_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings(camera_warmup_frames=2)
    candidate = head_mouse.CameraInfo(700, "DSHOW")

    class FakeCapture:
        def __init__(self, *args, **kwargs) -> None:
            self.released = False
            self.frames = [
                SimpleNamespace(size=1),
                SimpleNamespace(size=1),
            ]

        def isOpened(self) -> bool:
            return True

        def set(self, prop, value) -> bool:
            return True

        def read(self):
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(head_mouse.cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture())
    monkeypatch.setattr(head_mouse, "is_blank_frame", lambda frame, settings_arg: True)
    monkeypatch.setattr(head_mouse.time, "sleep", lambda *_args, **_kwargs: None)

    camera, reason = head_mouse.try_open_camera_backend(settings, candidate)
    assert camera is None
    assert "640x480@30/YUY2:black frames" in reason


def test_try_open_camera_backend_falls_back_to_second_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings(camera_warmup_frames=1)
    candidate = head_mouse.CameraInfo(700, "DSHOW")
    profiles = [(640, 480, 30, "YUY2"), (640, 480, 30, "MJPG")]
    created: list[FakeCapture] = []

    class FakeCapture:
        def __init__(self, *args, **kwargs) -> None:
            self.released = False
            self.profile = profiles[len(created)]
            self.frames = [SimpleNamespace(size=1)]
            created.append(self)

        def isOpened(self) -> bool:
            return True

        def set(self, prop, value) -> bool:
            return True

        def read(self):
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(head_mouse, "opencv_capture_profiles", lambda _settings: profiles)
    monkeypatch.setattr(head_mouse.cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture())
    monkeypatch.setattr(
        head_mouse,
        "is_blank_frame",
        lambda frame, settings_arg: created[-1].profile[3] == "YUY2",
    )
    monkeypatch.setattr(head_mouse.time, "sleep", lambda *_args, **_kwargs: None)

    camera, reason = head_mouse.try_open_camera_backend(settings, candidate)
    assert camera is created[1]
    assert reason == "ok:640x480@30/MJPG"
    assert created[0].released is True


def test_is_blank_frame_rejects_sparse_noise() -> None:
    settings = head_mouse.Settings()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[0, :, 1] = 200
    assert head_mouse.is_blank_frame(frame, settings) is True


def test_is_blank_frame_accepts_real_image() -> None:
    settings = head_mouse.Settings()
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert head_mouse.is_blank_frame(frame, settings) is False


def test_try_open_camera_backend_uses_ffmpeg_path(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = head_mouse.Settings()
    candidate = head_mouse.CameraInfo(-1, "FFMPEG")
    fake_camera = object()

    monkeypatch.setattr(head_mouse, "try_open_ffmpeg_camera", lambda settings_arg: (fake_camera, "ok:USB Camera"))
    camera, reason = head_mouse.try_open_camera_backend(settings, candidate)
    assert camera is fake_camera
    assert reason == "ok:USB Camera"
