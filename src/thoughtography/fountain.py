from __future__ import annotations

from typing import Iterable

from .media import format_timestamp
from .models import Scene, ScriptLine, ScriptProject


def _cue_name(line: ScriptLine) -> str:
    if line.kind == "narration":
        return line.speaker or "旁白"
    return (line.speaker or "未知角色").strip()


def _dialogue_text(line: ScriptLine) -> str:
    text = line.display_text.strip()
    return text if text else "（画面文字未能识别）"


def _timestamp_note(start: float, end: float) -> str:
    return f"[[{format_timestamp(start)}-{format_timestamp(end)}]]"


def _append_stanza(output: list[str], lines: Iterable[str]) -> None:
    for line in lines:
        output.append(line)
    output.append("")


def render_fountain(
    project: ScriptProject,
    *,
    include_timestamps: bool = True,
) -> str:
    output: list[str] = [
        f"Title: {project.title}",
        "Credit: 基于视频画面文字提取整理",
        f"Source: {project.source.name}",
        "",
    ]
    if project.metadata.get("user_hint"):
        output.append(f"Note: 用户提示词：{project.metadata['user_hint']}")
        output.append("")

    for scene in project.scenes:
        heading = f"# 场景 {scene.index}"
        if scene.heading:
            heading += f"：{scene.heading}"
        _append_stanza(output, [heading])

        if include_timestamps:
            _append_stanza(output, [_timestamp_note(scene.start, scene.end)])

        if scene.heading:
            forced_heading = scene.heading
            if not forced_heading.startswith("."):
                forced_heading = f".{forced_heading}"
            _append_stanza(output, [forced_heading])

        if scene.summary:
            _append_stanza(output, [scene.summary.strip()])

        for line in scene.lines:
            if include_timestamps:
                _append_stanza(output, [_timestamp_note(line.start, line.end)])
            cue = f"@{_cue_name(line)}"
            _append_stanza(output, [cue, _dialogue_text(line)])

    return "\n".join(output).rstrip() + "\n"


def render_text(
    project: ScriptProject,
    *,
    include_timestamps: bool = True,
) -> str:
    output: list[str] = [project.title, "=" * len(project.title), ""]
    for scene in project.scenes:
        header = f"场景 {scene.index}"
        if scene.heading:
            header += f"：{scene.heading}"
        output.append(header)
        if scene.summary:
            output.append(scene.summary.strip())
        output.append("")
        for line in scene.lines:
            timestamp = ""
            if include_timestamps:
                timestamp = (
                    f"[{format_timestamp(line.start)}-{format_timestamp(line.end)}] "
                )
            output.append(f"{timestamp}{_cue_name(line)}：{_dialogue_text(line)}")
        output.append("")
    return "\n".join(output).rstrip() + "\n"
