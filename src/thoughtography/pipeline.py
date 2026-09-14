from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .config import ProviderConfig
from .fountain import render_fountain, render_text
from .media import extract_preview_frames, probe_video
from .models import (
    Character,
    Keyframe,
    Scene,
    ScriptLine,
    ScriptProject,
    WindowAnalysis,
)
from .sampling import apply_keyframe_budget, detect_keyframes
from .vision import VisionClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractionResult:
    project: ScriptProject
    keyframes: list[Keyframe]
    vision_calls: int = 0
    errors: list[str] | None = None


def _normalise_text(text: str) -> str:
    return "".join(text.split()).casefold()


_DUPLICATE_SIMILARITY_THRESHOLD = 0.70
_DUPLICATE_MERGE_GAP_SECONDS = 3.0
_DUPLICATE_CROSS_CLASS_MIN_LENGTH = 4


def _text_similarity(first: str, second: str) -> float:
    """Fuzzy similarity for typewriter-style repeated dialogue variants."""
    first_norm = _normalise_text(first)
    second_norm = _normalise_text(second)
    if not first_norm or not second_norm:
        return 0.0
    if first_norm == second_norm:
        return 1.0
    shorter, longer = sorted((first_norm, second_norm), key=len)
    # Partial text can appear while text is still being typed/translated.
    if len(shorter) >= 3 and shorter in longer:
        return max(0.9, len(shorter) / len(longer))
    return SequenceMatcher(None, first_norm, second_norm).ratio()


def _same_classification(first: ScriptLine, second: ScriptLine) -> bool:
    return (
        first.kind == second.kind
        and first.speaker.strip() == second.speaker.strip()
    )


def _lines_are_duplicates(first: ScriptLine, second: ScriptLine) -> bool:
    if first.start > second.end + _DUPLICATE_MERGE_GAP_SECONDS:
        return False

    first_norm = _normalise_text(first.display_text)
    second_norm = _normalise_text(second.display_text)
    if not first_norm or not second_norm:
        return False

    same_speaker = first.speaker.strip() == second.speaker.strip()
    same_kind = first.kind == second.kind
    min_length = min(len(first_norm), len(second_norm))
    if first_norm == second_norm:
        if same_speaker and same_kind:
            return True
        # The model may switch the same on-screen line between narration and
        # dialogue. Only merge exact long text across that boundary.
        return min_length >= _DUPLICATE_CROSS_CLASS_MIN_LENGTH

    if not same_speaker:
        return False

    similarity = _text_similarity(first.display_text, second.display_text)
    if min_length >= _DUPLICATE_CROSS_CLASS_MIN_LENGTH:
        # Same speaker can still be classified as narration in one keyframe and
        # dialogue in another; allow fuzzy merge when the text is long enough.
        return similarity >= _DUPLICATE_SIMILARITY_THRESHOLD
    return same_kind and similarity >= _DUPLICATE_SIMILARITY_THRESHOLD


def _deduplicate_lines(lines: list[ScriptLine]) -> list[ScriptLine]:
    """Merge repeated/typewriter/overlapping state lines across keyframes."""
    ordered = sorted(
        (line for line in lines if line.display_text.strip()),
        key=lambda line: (line.start, line.end, line.speaker),
    )
    if not ordered:
        return []

    # Pairwise union-find keeps transitivity: A~B and B~C should collapse even
    # when A and C are slightly further apart in time.
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for index, line in enumerate(ordered):
        end_limit = line.end + _DUPLICATE_MERGE_GAP_SECONDS
        other = index + 1
        while other < len(ordered) and ordered[other].start <= end_limit:
            if _lines_are_duplicates(line, ordered[other]):
                union(index, other)
            other += 1

    groups: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        groups.setdefault(find(index), []).append(index)

    merged: list[ScriptLine] = []
    for indexes in groups.values():
        # Keep the most complete text version. On equal length prefer higher
        # confidence, then dialogue, then the earlier occurrence.
        representative = max(
            indexes,
            key=lambda item: (
                len(_normalise_text(ordered[item].display_text)),
                ordered[item].confidence,
                ordered[item].kind == "dialogue",
                -ordered[item].start,
            ),
        )
        best = ordered[representative]
        merged.append(
            replace(
                best,
                start=min(ordered[item].start for item in indexes),
                end=max(ordered[item].end for item in indexes),
                confidence=max(ordered[item].confidence for item in indexes),
            )
        )

    return sorted(merged, key=lambda line: (line.start, line.end, line.speaker))




def _build_scenes(
    analyses: list[WindowAnalysis],
    lines: list[ScriptLine],
    *,
    unlabelled_scene_seconds: float = 60.0,
) -> list[Scene]:
    scenes: list[Scene] = []
    for analysis in analyses:
        heading = analysis.heading.strip()
        summary = analysis.summary.strip()
        start = analysis.keyframe.start
        end = analysis.keyframe.end

        if scenes:
            previous = scenes[-1]
            both_named = bool(previous.heading) and bool(heading)
            merge_named = both_named and previous.heading == heading
            merge_unnamed = (
                not previous.heading
                and not heading
                and end - previous.start <= unlabelled_scene_seconds
            )
            if merge_named or merge_unnamed:
                previous.end = max(previous.end, end)
                if summary and summary not in previous.summary:
                    previous.summary = f"{previous.summary}\n{summary}".strip()
                continue

        scenes.append(
            Scene(
                index=len(scenes) + 1,
                start=start,
                end=end,
                heading=heading,
                summary=summary,
            )
        )

    if not scenes:
        return scenes

    scene_index = 0
    for line in sorted(lines, key=lambda item: item.start):
        midpoint = (line.start + line.end) / 2
        while scene_index + 1 < len(scenes) and midpoint >= scenes[scene_index].end:
            scene_index += 1
        scenes[scene_index].lines.append(line)

    return scenes


def _build_characters(analyses: list[WindowAnalysis], lines: list[ScriptLine]) -> list[Character]:
    descriptions: dict[str, str] = {}
    for analysis in analyses:
        for item in analysis.characters:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            if name:
                descriptions.setdefault(name, description)

    registry: dict[str, Character] = {}
    for line in lines:
        if line.kind == "narration":
            continue
        name = line.speaker.strip()
        if not name or name == "未知角色":
            continue
        key = _normalise_text(name)
        if key not in registry:
            registry[key] = Character(
                id=f"C{len(registry) + 1}",
                name=name,
                description=descriptions.get(name, ""),
                appearances=[line.start],
            )
        elif line.start not in registry[key].appearances:
            registry[key].appearances.append(line.start)
    return list(registry.values())


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _coverage_report(video_duration: float, keyframes: list[Keyframe]) -> dict[str, object]:
    if not keyframes:
        return {"duration": video_duration, "keyframe_count": 0, "gaps": []}
    gaps: list[dict[str, float]] = []
    cursor = 0.0
    for keyframe in sorted(keyframes, key=lambda item: item.start):
        if keyframe.start > cursor + 0.001:
            gaps.append({"start": cursor, "end": keyframe.start})
        cursor = max(cursor, keyframe.end)
    if video_duration > cursor + 0.001:
        gaps.append({"start": cursor, "end": video_duration})
    return {
        "duration": video_duration,
        "keyframe_count": len(keyframes),
        "covered_until": cursor,
        "gaps": gaps,
    }


def run_extract(
    video: Path,
    output_dir: Path,
    *,
    title: str | None = None,
    user_hint: str = "",
    provider_config: ProviderConfig | None = None,
    preview_fps: float = 2.0,
    preview_width: int = 640,
    global_threshold: float = 5.0,
    bottom_threshold: float = 3.0,
    local_threshold: float = 5.0,
    hash_threshold: int = 28,
    min_segment_seconds: float = 0.5,
    max_stable_seconds: float = 20.0,
    max_keyframes: int | None = None,
    include_timestamps: bool = True,
    dry_run: bool = False,
) -> ExtractionResult:
    video = video.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    frames_dir = cache_dir / "frames"
    vision_cache_dir = cache_dir / "vision"

    logger.info("读取视频信息: %s", video)
    media = probe_video(video)
    _write_json(output_dir / "media.json", media.to_dict())

    logger.info("抽取预览帧: fps=%s width=%s", preview_fps, preview_width)
    frames = extract_preview_frames(
        video,
        frames_dir,
        fps=preview_fps,
        width=preview_width,
    )
    if not frames:
        raise RuntimeError("未能从视频中抽取任何预览帧")

    logger.info("检测内容变化: %s 张预览帧", len(frames))
    keyframes = detect_keyframes(
        frames,
        media,
        global_threshold=global_threshold,
        bottom_threshold=bottom_threshold,
        local_threshold=local_threshold,
        hash_threshold=hash_threshold,
        min_segment_seconds=min_segment_seconds,
        max_stable_seconds=max_stable_seconds,
    )
    keyframes = apply_keyframe_budget(keyframes, max_keyframes=max_keyframes)
    _write_json(output_dir / "keyframes.json", [item.to_dict() for item in keyframes])
    _write_json(output_dir / "coverage.json", _coverage_report(media.duration, keyframes))
    logger.info("检测到关键帧: %s", len(keyframes))

    project = ScriptProject(
        title=title or video.stem,
        source=video,
        metadata={
            "video": media.to_dict(),
            "keyframe_count": len(keyframes),
            "preview_fps": preview_fps,
            "user_hint": user_hint.strip(),
            "provider": provider_config.public_dict() if provider_config else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
        },
    )

    if dry_run:
        (output_dir / "sampling_report.txt").write_text(
            "\n".join(
                f"{item.index:03d}  "
                f"{item.start:8.3f} - {item.end:8.3f}  "
                f"score={item.boundary_score:7.3f}  "
                f"{'forced' if item.forced else 'change'}"
                for item in keyframes
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractionResult(project=project, keyframes=keyframes)

    if provider_config is None:
        raise RuntimeError("非 dry-run 模式必须提供 provider_config")

    analyses: list[WindowAnalysis] = []
    errors: list[str] = []
    known_characters: list[str] = []
    vision_calls = 0

    with VisionClient(provider_config) as client:
        for keyframe in keyframes:
            try:
                logger.info(
                    "分析关键帧 %s/%s  %s - %s",
                    keyframe.index + 1,
                    len(keyframes),
                    keyframe.start,
                    keyframe.end,
                )
                analysis = client.analyze(
                    keyframe,
                    media,
                    user_hint=user_hint,
                    known_characters=known_characters,
                    cache_dir=vision_cache_dir,
                )
                vision_calls += 1
                analyses.append(analysis)
                for item in analysis.characters:
                    if isinstance(item, dict):
                        name = str(item.get("name") or "").strip()
                        if name and name not in known_characters:
                            known_characters.append(name)
            except Exception as exc:  # noqa: BLE001 - continue other keyframes
                message = f"关键帧 {keyframe.index} ({keyframe.start:.3f}s) 分析失败: {exc}"
                logger.warning(message)
                errors.append(message)

    if not analyses:
        raise RuntimeError("所有关键帧分析都失败了，请检查模型配置。\n" + "\n".join(errors))

    lines: list[ScriptLine] = []
    for analysis in analyses:
        lines.extend(analysis.lines)
    lines = _deduplicate_lines(lines)
    scenes = _build_scenes(analyses, lines)
    characters = _build_characters(analyses, lines)

    project.scenes = scenes
    project.characters = characters
    project.metadata["vision_calls"] = vision_calls
    project.metadata["errors"] = errors

    _write_json(output_dir / "analysis.json", project.to_dict())
    (output_dir / "script.fountain").write_text(
        render_fountain(project, include_timestamps=include_timestamps),
        encoding="utf-8",
    )
    (output_dir / "script.txt").write_text(
        render_text(project, include_timestamps=include_timestamps),
        encoding="utf-8",
    )
    return ExtractionResult(
        project=project,
        keyframes=keyframes,
        vision_calls=vision_calls,
        errors=errors,
    )
