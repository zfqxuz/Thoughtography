from thoughtography.models import ScriptLine
from thoughtography.pipeline import _deduplicate_lines


def _line(start: float, end: float, text: str, speaker: str = "旁白", kind: str = "narration") -> ScriptLine:
    return ScriptLine(
        start=start,
        end=end,
        speaker=speaker,
        kind=kind,  # type: ignore[arg-type]
        text_original=text,
        text_zh=text,
    )


def test_deduplicate_merges_typewriter_variants() -> None:
    lines = [
        _line(0.0, 0.5, "这是一个非常残酷的故事呢"),
        _line(0.5, 1.0, "这是一个残酷的故事"),
        _line(1.0, 4.0, "那可真是一个残酷的故事呢"),
    ]

    merged = _deduplicate_lines(lines)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 4.0
    assert merged[0].display_text == "这是一个非常残酷的故事呢"


def test_deduplicate_keeps_distinct_similar_speakers() -> None:
    lines = [
        _line(8.5, 12.0, "我不是杀人者", "未知角色", "dialogue"),
        _line(12.0, 16.0, "我不是被害者", "未知角色", "dialogue"),
    ]

    merged = _deduplicate_lines(lines)

    assert [line.display_text for line in merged] == ["我不是杀人者", "我不是被害者"]


def test_deduplicate_prefers_more_complete_text() -> None:
    lines = [
        _line(25.5, 28.0, "来，看看我的心吧"),
        _line(28.0, 31.0, "来，看看我的心吧，那里有着一切的真相"),
    ]

    merged = _deduplicate_lines(lines)

    assert len(merged) == 1
    assert merged[0].start == 25.5
    assert merged[0].end == 31.0
    assert merged[0].display_text == "来，看看我的心吧，那里有着一切的真相"


def test_deduplicate_merges_warning_variants() -> None:
    lines = [
        _line(0.0, 0.5, "※轻微猎奇注意"),
        _line(0.5, 1.0, "※轻微血腥注意"),
        _line(1.0, 4.0, "※轻微猎奇血腥注意"),
    ]

    merged = _deduplicate_lines(lines)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 4.0


def test_deduplicate_merges_exact_text_across_classification() -> None:
    lines = [
        ScriptLine(
            start=28.0,
            end=31.0,
            speaker="旁白",
            kind="narration",
            text_original="将情报传达给能解决事件的人",
            text_zh="将情报传达给能解决事件的人",
            confidence=0.9,
        ),
        ScriptLine(
            start=31.0,
            end=35.5,
            speaker="未知角色",
            kind="dialogue",
            text_original="将情报传达给能解决事件的人",
            text_zh="将情报传达给能解决事件的人",
            confidence=0.95,
        ),
    ]

    merged = _deduplicate_lines(lines)

    assert len(merged) == 1
    assert merged[0].start == 28.0
    assert merged[0].end == 35.5
    assert merged[0].speaker == "未知角色"
    assert merged[0].kind == "dialogue"
    assert merged[0].confidence == 0.95


def test_deduplicate_keeps_short_exact_text_across_speakers() -> None:
    lines = [
        _line(0.0, 0.5, "好", "旁白", "narration"),
        _line(0.5, 1.0, "好", "未知角色", "dialogue"),
    ]

    merged = _deduplicate_lines(lines)

    assert [line.display_text for line in merged] == ["好", "好"]
