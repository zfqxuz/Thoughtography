from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import ProviderConfig
from .media import format_timestamp
from .models import Keyframe, ScriptLine, VideoInfo, WindowAnalysis

logger = logging.getLogger(__name__)


class VisionError(RuntimeError):
    pass


SYSTEM_PROMPT = """你是一个视觉小说/手书视频的对白与旁白提取助手。
你的目标是提取画面中实际存在的对白和旁白文字，不要描述画面。

必须遵守：
1. 只输出画面中实际可见的文字：对话框、旁白框、心声框、人物名牌、字幕等。
2. 不要在 lines 里写动作、镜头、氛围、拟声说明等描述；lines 必须是画面文字本身。
3. scene.heading 只填地点和大致时间（例如“红魔馆 - 夜晚”）；无法判断时留空。scene.summary 永远填空字符串。
4. 认真区分说话者：
   - 优先使用画面可见的角色名或人物名牌；
   - 没有名字时，根据说话框/气泡的位置、颜色、尾巴方向，或角色外观特征命名，例如“橙发少女”“紫发墨镜少女”“红发少女”；同一角色在全片必须保持同一个名字；
   - 画面只有一名角色时，绝不写“未知角色”，直接用已确认名字或外观命名；
   - 已提供全片确认角色名单时，优先使用名单中的名字。
5. 判断每句是 dialogue（有明确说话框/气泡的角色对白）还是 narration（旁白、文字卡、字幕、叙事句）。
6. 原文是日文或英文时翻译成简体中文；原文是中文时 text_zh 填原文。

只输出一个 JSON 对象，不要输出 Markdown、解释或代码块。
JSON 格式：
{
  "scene": {"heading": "地点 - 时间，无法判断时留空", "summary": ""},
  "characters": [{"name": "角色名", "description": "极简外观特征"}],
  "lines": [
    {
      "speaker": "角色名或旁白",
      "type": "dialogue",
      "text_original": "画面原始文字",
      "text_zh": "简体中文翻译",
      "confidence": 0.9
    }
  ]
}
如果没有文字，lines 返回空数组。
"""


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise VisionError(f"模型未返回 JSON: {content[:300]}")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VisionError(f"模型 JSON 解析失败: {exc}\n{content[:500]}") from exc
    if not isinstance(data, dict):
        raise VisionError("模型返回的 JSON 不是对象")
    return data


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class VisionClient:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            timeout=config.timeout_sec,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VisionClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _prompt(self, keyframe: Keyframe, user_hint: str, known_characters: list[str]) -> str:
        known_text = "、".join(known_characters) if known_characters else "暂无"
        hint = user_hint.strip() if user_hint.strip() else "用户未提供额外提示。"
        return (
            f"该关键帧代表视频时间段："
            f"{format_timestamp(keyframe.start)} - {format_timestamp(keyframe.end)}。\n"
            f"全片已确认角色（必须优先使用这些名字）：{known_text}\n"
            f"用户辅助提示：\n{hint}\n\n"
            "请只提取这一张画面中实际可见的对白/旁白文字。"
            "不要描述动作、镜头或氛围；不要在台词里添加不存在的说明。"
            "如果画面只有一名角色，请直接使用其名字或外观命名，不要写“未知角色”。"
        )

    def analyze(
        self,
        keyframe: Keyframe,
        video: VideoInfo,
        *,
        user_hint: str = "",
        known_characters: list[str] | None = None,
        cache_dir: Path | None = None,
    ) -> WindowAnalysis:
        known_characters = known_characters or []
        cache_file = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"keyframe_{keyframe.index:04d}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._parse_response(
                    cached.get("parsed") or cached,
                    keyframe,
                    video,
                )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._prompt(keyframe, user_hint, known_characters),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(keyframe.path)},
                        },
                    ],
                },
            ],
            "temperature": 0.1,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(self.config.chat_url, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise VisionError(
                        f"模型服务暂时不可用 ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            raise VisionError(f"调用视觉模型失败: {last_error}") from last_error

        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise VisionError(f"无法解析模型响应: {str(data)[:500]}")
        parsed = _extract_json(content)
        analysis = self._parse_response(parsed, keyframe, video)

        if cache_file is not None:
            cache_file.write_text(
                json.dumps(
                    {
                        "keyframe": keyframe.to_dict(),
                        "parsed": parsed,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return analysis

    def _parse_response(
        self,
        data: dict[str, Any],
        keyframe: Keyframe,
        video: VideoInfo,
    ) -> WindowAnalysis:
        scene = data.get("scene") or {}
        heading = str(scene.get("heading") or data.get("heading") or "").strip()
        summary = str(scene.get("summary") or data.get("summary") or "").strip()
        characters = data.get("characters") or []
        if not isinstance(characters, list):
            characters = []

        raw_lines = data.get("lines") or []
        if not isinstance(raw_lines, list):
            raw_lines = []

        lines: list[ScriptLine] = []
        for item in raw_lines:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or item.get("kind") or "dialogue").lower()
            if kind not in {"dialogue", "narration", "旁白", "对白"}:
                kind = "dialogue"
            resolved_kind = "narration" if kind in {"narration", "旁白"} else "dialogue"

            text_original = str(
                item.get("text_original") or item.get("original") or item.get("text") or ""
            ).strip()
            text_zh = str(
                item.get("text_zh") or item.get("translation") or item.get("zh") or ""
            ).strip()
            if not text_original and text_zh:
                text_original = text_zh
            if not text_zh:
                text_zh = text_original
            if not text_original and not text_zh:
                continue

            speaker = str(item.get("speaker") or "").strip()
            if resolved_kind == "narration":
                speaker = speaker or "旁白"

            start = item.get("start")
            end = item.get("end")
            try:
                start_value = float(start) if start is not None else keyframe.start
            except (TypeError, ValueError):
                start_value = keyframe.start
            try:
                end_value = float(end) if end is not None else keyframe.end
            except (TypeError, ValueError):
                end_value = keyframe.end
            start_value = _clamp(start_value, keyframe.start, keyframe.end)
            end_value = _clamp(max(start_value, end_value), start_value, keyframe.end)

            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = _clamp(confidence, 0.0, 1.0)

            lines.append(
                ScriptLine(
                    start=start_value,
                    end=end_value,
                    speaker=speaker or "未知角色",
                    kind=resolved_kind,
                    text_original=text_original,
                    text_zh=text_zh,
                    confidence=confidence,
                    source_index=keyframe.index,
                )
            )

        return WindowAnalysis(
            keyframe=keyframe,
            heading=heading,
            summary=summary,
            characters=characters,
            lines=lines,
        )
