from pathlib import Path

import pytest

from thoughtography import media

PROBE_STDERR = """ffmpeg version 7.0.2
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:01:00.13, start: 0.000000, bitrate: 267 kb/s
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 640x360 [SAR 1:1 DAR 16:9], 191 kb/s, 29.97 fps
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 65 kb/s
At least one output file must be specified
"""


def _fake_ffmpeg_run(stderr: str):
    def fake_run(command, **kwargs):  # noqa: ANN001 - subprocess signature
        return media.subprocess.CompletedProcess(command, 1, "", stderr)

    return fake_run


def test_probe_video_fallback_parses_full_ffmpeg_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media, "find_ffprobe", lambda: None)
    monkeypatch.setattr(media, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", _fake_ffmpeg_run(PROBE_STDERR))

    info = media.probe_video(tmp_path / "input.mp4")

    assert info.duration == pytest.approx(60.13)
    assert info.width == 640
    assert info.height == 360
    assert info.fps == pytest.approx(29.97)
    assert info.has_audio is True


def test_probe_video_fallback_rejects_invalid_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media, "find_ffprobe", lambda: None)
    monkeypatch.setattr(media, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", _fake_ffmpeg_run("Invalid data found when processing input\n"))

    with pytest.raises(media.FFmpegError, match="无法读取视频信息"):
        media.probe_video(tmp_path / "input.mp4")
