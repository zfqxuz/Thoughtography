from pathlib import Path

from thoughtography.fountain import render_fountain, render_text
from thoughtography.models import Scene, ScriptLine, ScriptProject


def make_project() -> ScriptProject:
    scene = Scene(
        index=1,
        start=0.0,
        end=12.5,
        heading="教室 - 日",
        summary="窗外下着雨。",
        lines=[
            ScriptLine(
                start=1.0,
                end=3.2,
                speaker="小林",
                kind="dialogue",
                text_original="おはよう",
                text_zh="早上好。",
            ),
            ScriptLine(
                start=3.2,
                end=5.0,
                speaker="旁白",
                kind="narration",
                text_original="那是最后一次见面。",
                text_zh="那是最后一次见面。",
            ),
        ],
    )
    return ScriptProject(title="测试台本", source=Path("input.mp4"), scenes=[scene])


def test_render_fountain_contains_cues_and_timestamps() -> None:
    text = render_fountain(make_project(), include_timestamps=True)
    assert "Title: 测试台本" in text
    assert "# 场景 1：教室 - 日" in text
    assert ".教室 - 日" in text
    assert "@小林" in text
    assert "@旁白" in text
    assert "[[00:01-00:03.2]]" in text
    assert "早上好。" in text


def test_render_text_contains_speaker_and_timestamp() -> None:
    text = render_text(make_project(), include_timestamps=True)
    assert "[00:01-00:03.2] 小林：早上好。" in text
    assert "[00:03.2-00:05] 旁白：那是最后一次见面。" in text
