from pathlib import Path

from thoughtography.bili_ocr import _acceptable_ocr_text, _box_center, _dedupe_frame_detections


def test_acceptable_ocr_text_keeps_chinese_and_english() -> None:
    assert _acceptable_ocr_text("这是一个残酷的故事")
    assert _acceptable_ocr_text("HELLO WORLD")


def test_acceptable_ocr_text_drops_kana_noise() -> None:
    assert not _acceptable_ocr_text("あいうえおかきくけこ")


def test_box_center() -> None:
    cx, cy, width, height = _box_center([[10, 20], [50, 20], [50, 80], [10, 80]])
    assert (cx, cy) == (30.0, 50.0)
    assert (width, height) == (40.0, 60.0)


def test_dedupe_frame_detections_prefers_higher_score() -> None:
    detections = [
        {"text": "这是一个残酷的故事", "score": 0.9, "cx": 100.0, "cy": 100.0, "bh": 40.0},
        {"text": "这是一个残酷的故事", "score": 0.7, "cx": 102.0, "cy": 101.0, "bh": 40.0},
        {"text": "另一句", "score": 0.8, "cx": 300.0, "cy": 100.0, "bh": 40.0},
    ]
    kept = _dedupe_frame_detections(detections)
    assert [item["text"] for item in kept] == ["这是一个残酷的故事", "另一句"]
    assert kept[0]["score"] == 0.9


def test_line_tag_uses_speaker_and_narration() -> None:
    from thoughtography.bili_ocr import OcrLine, _line_tag

    def line(speaker: str, kind: str, box_h: float = 120.0) -> OcrLine:
        return OcrLine(
            start=0.0,
            end=1.0,
            text="测试",
            score=1.0,
            center_x=10.0,
            center_y=20.0,
            box_w=20.0,
            box_h=box_h,
            speaker=speaker,
            speaker_kind=kind,
        )

    assert _line_tag(line("蕾米莉亚", "dialogue")) == "【蕾米莉亚】"
    assert _line_tag(line("红发少女", "narration")) == "【红发少女·旁白】"
    assert _line_tag(line("旁白", "narration")) == "【旁白】"
    assert _line_tag(line("", "dialogue")) == "【对白】"
