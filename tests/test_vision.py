import json
from pathlib import Path

from PIL import Image

from thoughtography.config import ProviderConfig
from thoughtography.models import Keyframe, VideoInfo
from thoughtography.vision import VisionClient


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        content = {
            "scene": {"heading": "教室 - 日", "summary": "两个人在教室里。"},
            "characters": [{"name": "小林", "description": "蓝发少女"}],
            "lines": [
                {
                    "speaker": "小林",
                    "type": "dialogue",
                    "text_original": "おはよう",
                    "text_zh": "早上好",
                    "confidence": 0.95,
                },
                {
                    "speaker": "",
                    "type": "narration",
                    "text_original": "朝の光が差し込む。",
                    "text_zh": "晨光洒了进来。",
                    "confidence": 0.8,
                },
            ],
        }
        return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}


class FakeHTTPClient:
    def __init__(self) -> None:
        self.payload = None

    def post(self, url: str, json: dict):  # noqa: A002 - matches httpx API
        self.payload = (url, json)
        return FakeResponse()

    def close(self) -> None:
        return None


def test_vision_client_parses_and_clamps_lines(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "white").save(frame)
    keyframe = Keyframe(index=0, start=10.0, end=12.5, path=frame)
    config = ProviderConfig(
        name="test",
        base_url="http://example.test/v1",
        api_key="sk-test",
        model="test-vision",
        chat_url="http://example.test/v1/chat/completions",
        timeout_sec=5,
        json_mode=True,
    )
    client = VisionClient(config)
    fake = FakeHTTPClient()
    client._client = fake  # type: ignore[assignment]

    analysis = client.analyze(
        keyframe,
        VideoInfo(path=tmp_path / "input.mp4", duration=20, width=64, height=64),
        user_hint="角色叫小林",
        known_characters=["小林"],
    )

    assert analysis.heading == "教室 - 日"
    assert len(analysis.lines) == 2
    dialogue, narration = analysis.lines
    assert dialogue.speaker == "小林"
    assert dialogue.kind == "dialogue"
    assert dialogue.text_zh == "早上好"
    assert dialogue.start == 10.0
    assert dialogue.end == 12.5
    assert narration.kind == "narration"
    assert narration.speaker == "旁白"

    assert fake.payload is not None
    url, payload = fake.payload
    assert url == "http://example.test/v1/chat/completions"
    assert payload["model"] == "test-vision"
    assert payload["response_format"] == {"type": "json_object"}
    image_part = payload["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
