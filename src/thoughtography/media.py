from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .models import PreviewFrame, VideoInfo


class FFmpegError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    override = os.environ.get("FFMPEG_BIN")
    if override:
        return override
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on local setup
        raise FFmpegError(
            "找不到 ffmpeg。请安装 ffmpeg，或设置环境变量 FFMPEG_BIN。"
        ) from exc


def find_ffprobe() -> str | None:
    override = os.environ.get("FFPROBE_BIN")
    if override:
        return override
    return shutil.which("ffprobe")


def run_command(command: list[str], *, timeout: float | None = None) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = "\n".join(detail[-8:]) if detail else "未知错误"
        raise FFmpegError(f"命令执行失败: {' '.join(command)}\n{tail}")
    return completed.stdout


_FPS_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*fps")
_DURATION_RE = re.compile(r"Duration:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)")
_SIZE_RE = re.compile(r"(?P<w>\d{2,5})x(?P<h>\d{2,5})")


def _parse_ffmpeg_probe_text(stderr: str) -> tuple[float, int, int, float | None, bool]:
    duration = 0.0
    match = _DURATION_RE.search(stderr)
    if match:
        duration = (
            int(match.group("h")) * 3600
            + int(match.group("m")) * 60
            + float(match.group("s"))
        )
    width = height = 0
    size_match = _SIZE_RE.search(stderr)
    if size_match:
        width = int(size_match.group("w"))
        height = int(size_match.group("h"))
    fps = None
    fps_match = _FPS_RE.search(stderr)
    if fps_match:
        fps = float(fps_match.group("num"))
    has_audio = "Audio:" in stderr
    return duration, width, height, fps, has_audio


def probe_video(path: Path) -> VideoInfo:
    ffprobe = find_ffprobe()
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        raw = run_command(command, timeout=60)
        data: dict[str, Any] = json.loads(raw)
        streams = data.get("streams") or []
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise FFmpegError(f"视频没有视频流: {path}")
        vs = video_streams[0]
        duration = 0.0
        for candidate in (vs.get("duration"), (data.get("format") or {}).get("duration")):
            if candidate is not None:
                try:
                    duration = float(candidate)
                    break
                except (TypeError, ValueError):
                    pass
        fps = None
        for key in ("avg_frame_rate", "r_frame_rate"):
            value = vs.get(key)
            if value and value != "0/0":
                try:
                    num, den = value.split("/")
                    fps = float(num) / float(den) if float(den) else None
                    break
                except (ValueError, ZeroDivisionError):
                    pass
        if fps is None:
            fps_match = _FPS_RE.search(json.dumps(vs))
            if fps_match:
                fps = float(fps_match.group("num"))
        return VideoInfo(
            path=path,
            duration=duration,
            width=int(vs.get("width") or 0),
            height=int(vs.get("height") or 0),
            fps=fps,
            has_audio=any(s.get("codec_type") == "audio" for s in streams),
        )

    # Fallback: parse `ffmpeg -i` stderr if ffprobe is unavailable.
    command = [find_ffmpeg(), "-hide_banner", "-i", str(path)]
    try:
        run_command(command, timeout=60)
    except FFmpegError as exc:
        text = str(exc)
        duration, width, height, fps, has_audio = _parse_ffmpeg_probe_text(text)
        return VideoInfo(
            path=path,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            has_audio=has_audio,
        )
    raise FFmpegError(f"无法读取视频信息: {path}")


def extract_preview_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float = 2.0,
    width: int = 640,
) -> list[PreviewFrame]:
    if fps <= 0:
        raise ValueError("fps 必须大于 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    pattern = str(out_dir / "frame_%06d.jpg")
    command = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-q:v",
        "4",
        "-start_number",
        "0",
        pattern,
    ]
    run_command(command, timeout=60 * 30)
    files = sorted(out_dir.glob("frame_*.jpg"))
    return [
        PreviewFrame(index=index, time=index / fps, path=path)
        for index, path in enumerate(files)
    ]


def format_timestamp(seconds: float) -> str:
    total = max(0.0, seconds)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total - hours * 3600 - minutes * 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".rstrip("0").rstrip(".")
    return f"{minutes:02d}:{secs:06.3f}".rstrip("0").rstrip(".")
