from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from thoughtography.models import PreviewFrame, VideoInfo
from thoughtography.sampling import apply_keyframe_budget, detect_keyframes


def _make_frame(path: Path, text: str, color: str = "white") -> None:
    image = Image.new("RGB", (320, 180), color)
    draw = ImageDraw.Draw(image)
    draw.text((20, 130), text, fill="black")
    image.save(path, quality=90)


def test_detect_keyframes_splits_on_visual_change(tmp_path: Path) -> None:
    frames: list[PreviewFrame] = []
    for index, text in enumerate(["AAAA", "AAAA", "BBBB", "BBBB"]):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _make_frame(path, text)
        frames.append(PreviewFrame(index=index, time=index * 0.5, path=path))

    keyframes = detect_keyframes(
        frames,
        VideoInfo(path=Path("input.mp4"), duration=2.0, width=320, height=180),
        max_stable_seconds=20.0,
    )
    assert len(keyframes) == 2
    assert keyframes[0].start == 0.0
    assert keyframes[0].end == 1.0
    assert keyframes[1].start == 1.0
    assert keyframes[1].end == 2.0
    assert keyframes[0].path.name in {"frame_000000.jpg", "frame_000001.jpg"}


def test_apply_keyframe_budget_merges_lowest_score(tmp_path: Path) -> None:
    frames: list[PreviewFrame] = []
    for index, text in enumerate(["A", "B", "C"]):
        path = tmp_path / f"frame_{index:06d}.jpg"
        _make_frame(path, text)
        frames.append(PreviewFrame(index=index, time=index * 2.0, path=path))
    keyframes = detect_keyframes(
        frames,
        VideoInfo(path=Path("input.mp4"), duration=6.0, width=320, height=180),
        max_stable_seconds=20.0,
    )
    assert len(keyframes) == 3
    limited = apply_keyframe_budget(keyframes, max_keyframes=2)
    assert len(limited) == 2
    assert limited[0].start == 0.0
    assert limited[-1].end == 6.0
