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
