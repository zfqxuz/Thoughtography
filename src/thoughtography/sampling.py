from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .models import Keyframe, PreviewFrame, VideoInfo


def _dhash(gray: Image.Image, *, width: int = 17, height: int = 16) -> bytes:
    small = gray.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(small, dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    return np.packbits(bits).tobytes()


def _crop_band(gray: Image.Image, band: str) -> Image.Image:
    width, height = gray.size
    third = max(1, height // 3)
    if band == "top":
        return gray.crop((0, 0, width, third))
    if band == "middle":
        return gray.crop((0, third, width, min(height, third * 2)))
    if band == "bottom":
        return gray.crop((0, min(height, third * 2), width, height))
    return gray


def frame_signature(path: Path) -> tuple[tuple[bytes, ...], np.ndarray]:
    """Return (per-band dHash, low-resolution grayscale)."""
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        hashes = tuple(
            _dhash(_crop_band(gray, band))
            for band in ("full", "top", "middle", "bottom")
        )
        low = gray.resize((160, 90), Image.Resampling.BILINEAR)
        low_array = np.asarray(low, dtype=np.uint8)
        return hashes, low_array


def hamming_distance(first: bytes, second: bytes) -> int:
    return sum((a ^ b).bit_count() for a, b in zip(first, second))


def signature_distance(
    previous_hash: tuple[bytes, ...],
    current_hash: tuple[bytes, ...],
    previous_low: np.ndarray,
    current_low: np.ndarray,
) -> tuple[float, float, float, int]:
    difference = np.abs(
        previous_low.astype(np.int16) - current_low.astype(np.int16)
    )
    global_diff = float(np.mean(difference))
    height = current_low.shape[0]
    bottom = difference[int(height * 0.55) :, :]
    bottom_diff = float(np.mean(bottom))
    # Mean diff is diluted by the large static background. The strongest 0.5%
    # of pixels catch small text-box changes while still being insensitive to
    # ordinary JPEG noise.
    flat = difference.reshape(-1)
    top_count = max(1, int(flat.size * 0.005))
    local_diff = float(np.sort(flat)[-top_count:].mean())
    hash_distance = max(
        hamming_distance(a, b) for a, b in zip(previous_hash, current_hash)
    )
    return global_diff, bottom_diff, local_diff, hash_distance


def detect_keyframes(
    frames: list[PreviewFrame],
    video: VideoInfo,
    *,
    global_threshold: float = 5.0,
    bottom_threshold: float = 3.0,
    local_threshold: float = 5.0,
    hash_threshold: int = 28,
    min_segment_seconds: float = 0.5,
    max_stable_seconds: float = 20.0,
) -> list[Keyframe]:
    """Detect stable visual/text states and return representative keyframes.

    Completeness strategy:
    - A new keyframe is emitted whenever global/bottom-region pixels or dHash
      change beyond the configured thresholds.
    - Even when the image is completely static, a fresh keyframe is forced
      every ``max_stable_seconds``. This prevents a subtle gradual text change
      from being missed by a static diff.
    """
    if not frames:
        return []
    if len(frames) == 1:
        return [
            Keyframe(
                index=0,
                start=frames[0].time,
                end=max(video.duration, frames[0].time + 1.0),
                path=frames[0].path,
            )
        ]

    frame_step = max(0.01, frames[1].time - frames[0].time)
    keyframes: list[Keyframe] = []

    segment_start = 0
    previous_hash, previous_low = frame_signature(frames[0].path)
    frames[0].band_hashes = previous_hash

    def finalize(end_index: int, boundary_score: float, forced: bool) -> None:
        # The last frame before the next change is usually the most complete
        # state (important for typewriter-style text animation).
        representative_index = end_index
        segment_start_time = frames[segment_start].time
        segment_end_time = min(
            video.duration if video.duration > 0 else float("inf"),
            frames[end_index].time + frame_step,
        )
        keyframes.append(
            Keyframe(
                index=len(keyframes),
                start=segment_start_time,
                end=segment_end_time,
                path=frames[representative_index].path,
                boundary_score=boundary_score,
                forced=forced,
            )
        )

    for index in range(1, len(frames)):
        current_hash, current_low = frame_signature(frames[index].path)
        frames[index].band_hashes = current_hash

        global_diff, bottom_diff, local_diff, hash_distance = signature_distance(
            previous_hash, current_hash, previous_low, current_low
        )
        frames[index].pixel_diff = global_diff
        frames[index].bottom_diff = bottom_diff
        frames[index].local_diff = local_diff

        segment_duration = frames[index].time - frames[segment_start].time
        changed = (
            global_diff >= global_threshold
            or bottom_diff >= bottom_threshold
            or local_diff >= local_threshold
            or hash_distance >= hash_threshold
        )
        forced = segment_duration >= max_stable_seconds and not changed

        if changed and segment_duration < min_segment_seconds:
            # Very short transient frames (usually fades/transitions) are not
            # promoted to standalone states.
            changed = False
            forced = False

        if changed or forced:
            score = max(
                global_diff,
                bottom_diff,
                local_diff,
                float(hash_distance) / 10.0,
            )
            finalize(index - 1, score, forced)
            segment_start = index

        previous_hash, previous_low = current_hash, current_low

    score = max(frames[-1].pixel_diff, frames[-1].bottom_diff)
    finalize(len(frames) - 1, score, False)
    return keyframes


def apply_keyframe_budget(
    keyframes: list[Keyframe],
    *,
    max_keyframes: int | None,
) -> list[Keyframe]:
    """Merge adjacent keyframes when the configured budget is exceeded.

    The lowest boundary-score states are merged first. This is an explicit
    cost/completeness trade-off; without a budget every detected state is kept.
    """
    if not max_keyframes or len(keyframes) <= max_keyframes:
        return keyframes

    merged = list(keyframes)
    while len(merged) > max_keyframes:
        merge_at = min(
            range(1, len(merged)),
            key=lambda i: (merged[i].boundary_score, -merged[i].duration),
        )
        previous = merged[merge_at - 1]
        current = merged[merge_at]
        merged[merge_at - 1] = replace(
            previous,
            end=current.end,
            boundary_score=previous.boundary_score,
        )
        del merged[merge_at]
        for index, keyframe in enumerate(merged):
            merged[index] = replace(keyframe, index=index)
    return merged
