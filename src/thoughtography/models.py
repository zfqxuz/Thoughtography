from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float | None = None
    has_audio: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_audio": self.has_audio,
        }


@dataclass(slots=True)
class PreviewFrame:
    """One low-rate frame extracted from the video."""

    index: int
    time: float
    path: Path
    band_hashes: tuple[bytes, ...] | None = None
    pixel_diff: float = 0.0
    bottom_diff: float = 0.0
    local_diff: float = 0.0
    boundary_score: float = 0.0


@dataclass(slots=True)
class Keyframe:
    """A representative frame for one stable visual/text state."""

    index: int
    start: float
    end: float
    path: Path
    boundary_score: float = 0.0
    forced: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "path": str(self.path),
            "boundary_score": self.boundary_score,
            "forced": self.forced,
        }


@dataclass(slots=True)
class ScriptLine:
    start: float
    end: float
    speaker: str = "未知角色"
    kind: Literal["dialogue", "narration"] = "dialogue"
    text_original: str = ""
    text_zh: str = ""
    confidence: float = 1.0
    source_index: int = 0

    @property
    def display_text(self) -> str:
        return self.text_zh or self.text_original

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "kind": self.kind,
            "text_original": self.text_original,
            "text_zh": self.text_zh,
            "confidence": self.confidence,
            "source_index": self.source_index,
        }


@dataclass(slots=True)
class Character:
    id: str
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    appearances: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "aliases": self.aliases,
            "appearances": self.appearances,
        }


@dataclass(slots=True)
class Scene:
    index: int
    start: float
    end: float
    heading: str = ""
    summary: str = ""
    lines: list[ScriptLine] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "heading": self.heading,
            "summary": self.summary,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass(slots=True)
class ScriptProject:
    title: str
    source: Path
    scenes: list[Scene] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": str(self.source),
            "metadata": self.metadata,
            "characters": [c.to_dict() for c in self.characters],
            "scenes": [s.to_dict() for s in self.scenes],
        }


@dataclass(slots=True)
class WindowAnalysis:
    """Structured response produced by the vision model for one keyframe."""

    keyframe: Keyframe
    heading: str = ""
    summary: str = ""
    characters: list[dict[str, Any]] = field(default_factory=list)
    lines: list[ScriptLine] = field(default_factory=list)
