from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from .media import find_ffmpeg, format_timestamp

PEANUT_API = "https://xapi.peanutdl.com/beibei"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


class BiliOcrError(RuntimeError):
    pass


@dataclass(slots=True)
class BiliVideo:
    source_url: str
    title: str
    part_title: str
    author: str
    bvid: str
    page: int
    duration: float
    size_bytes: int
    quality: str
    player_url: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def display_title(self) -> str:
        part = self.part_title.strip()
        base = self.title.strip() or self.bvid
        if part and part != base:
            return f"{base} / {part}"
        return base


@dataclass(slots=True)
class OcrLine:
    start: float
    end: float
    text: str
    score: float
    center_x: float
    center_y: float
    box_w: float
    box_h: float
    hits: int = 1

    @property
    def vertical(self) -> bool:
        return self.box_h > self.box_w * 1.35

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "score": round(self.score, 4),
            "center": [round(self.center_x, 1), round(self.center_y, 1)],
            "box": [round(self.box_w, 1), round(self.box_h, 1)],
            "vertical": self.vertical,
            "hits": self.hits,
        }


def resolve_bilibili(url: str, *, timeout: float = 60.0) -> BiliVideo:
    """Resolve a Bilibili link through peanutdl's public parser API."""
    payload = {
        "api": "BEIBEI",
        "url": url.strip(),
        "captcha": "",
        "id": "",
        "captcha_code": "",
        "captcha_id": "",
        "turnstile_token": "",
        "captcha_type": "",
    }
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://peanutdl.com",
        "Referer": "https://peanutdl.com/zh/bilibili",
        "User-Agent": BROWSER_UA,
    }
    try:
        response = httpx.post(PEANUT_API, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise BiliOcrError(f"peanutdl 解析失败: {exc}") from exc

    if not isinstance(data, dict) or data.get("status") not in (0, "0", None):
        raise BiliOcrError(f"peanutdl 返回异常: {json.dumps(data, ensure_ascii=False)[:500]}")

    info = data.get("info") or {}
    mp4 = data.get("mp4") or {}
    player_url = mp4.get("player_url")
    if not isinstance(player_url, str) or not player_url.strip():
        raise BiliOcrError(f"peanutdl 未返回播放地址: {json.dumps(data, ensure_ascii=False)[:500]}")

    urlend = str(info.get("Urlend") or "")
    page_match = re.search(r"[?&]p=(\d+)", urlend)
    page = int(page_match.group(1)) if page_match else 1
    duration_ms = float(mp4.get("info_length") or 0)
    return BiliVideo(
        source_url=url,
        title=str(info.get("Title") or ""),
        part_title=str(info.get("Titlep") or ""),
        author=str(info.get("Name") or ""),
        bvid=str(info.get("Bvid") or ""),
        page=page,
        duration=duration_ms / 1000.0,
        size_bytes=int(mp4.get("info_size") or 0),
        quality=str(mp4.get("info_description") or ""),
        player_url=player_url,
        raw=data,
    )


def download_video(
    video: BiliVideo,
    dest: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    timeout: float = 120.0,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": BROWSER_UA,
        "Referer": "https://www.bilibili.com/",
    }
    downloaded = 0
    try:
        with httpx.stream(
            "GET",
            video.player_url,
            headers=headers,
            follow_redirects=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or video.size_bytes or 0)
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
    except Exception as exc:  # noqa: BLE001
        if dest.exists():
            dest.unlink()
        raise BiliOcrError(f"视频下载失败: {exc}") from exc
    return dest


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float = 1.0,
    timeout: float = 1800.0,
) -> list[tuple[float, Path]]:
    if fps <= 0:
        raise ValueError("fps 必须大于 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    pattern = str(out_dir / "frame_%06d.jpg")
    command = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "3",
        "-start_number",
        "0",
        pattern,
    ]
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - ffmpeg failure
        raise BiliOcrError(f"抽帧失败: {exc}") from exc
    files = sorted(out_dir.glob("frame_*.jpg"))
    return [(index / fps, path) for index, path in enumerate(files)]


def _acceptable_ocr_text(text: str) -> bool:
    """Drop OCR noise that is mostly Japanese kana/roumaji fragments.

    RapidOCR uses Chinese+English models here.  Chinese subtitles are the
    target; vertical Japanese dialogue often comes back as kana gibberish and
    would only pollute the transcript.
    """
    value = str(text or "").strip()
    if len(value) < 2:
        return False
    kana = sum(1 for char in value if 0x3040 <= ord(char) <= 0x30FF)
    han = sum(1 for char in value if 0x4E00 <= ord(char) <= 0x9FFF)
    latin = sum(1 for char in value if char.isascii() and char.isalpha())
    if kana and kana / max(1, len(value)) > 0.25 and han < kana:
        return False
    return han > 0 or latin >= 2


def _normalise_text(text: str) -> str:
    return "".join(str(text).split()).casefold()


def _text_similarity(first: str, second: str) -> float:
    a = _normalise_text(first)
    b = _normalise_text(second)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 2 and shorter in longer:
        return max(0.92, len(shorter) / len(longer))
    return SequenceMatcher(None, a, b).ratio()


def _box_center(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return (left + right) / 2.0, (top + bottom) / 2.0, max(1.0, right - left), max(1.0, bottom - top)


def _detections_from_result(result: Iterable[Any] | None) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for item in result or []:
        try:
            box, text, score = item
        except (TypeError, ValueError):
            continue
        text = str(text or "").strip()
        if not text:
            continue
        center_x, center_y, box_w, box_h = _box_center(box)
        detections.append(
            {
                "text": text,
                "score": float(score or 0.0),
                "box": box,
                "cx": center_x,
                "cy": center_y,
                "bw": box_w,
                "bh": box_h,
            }
        )
    return detections


def _dedupe_frame_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        duplicate = False
        for existing in kept:
            if _text_similarity(detection["text"], existing["text"]) < 0.8:
                continue
            distance = (
                (detection["cx"] - existing["cx"]) ** 2
                + (detection["cy"] - existing["cy"]) ** 2
            ) ** 0.5
            if distance <= max(30.0, 0.5 * max(detection["bh"], existing["bh"])):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def extract_ocr_lines(
    frames: list[tuple[float, Path]],
    *,
    fps: float = 1.0,
    min_score: float = 0.6,
    progress: Callable[[int, int], None] | None = None,
) -> list[OcrLine]:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:  # pragma: no cover
        raise BiliOcrError(
            "未安装 OCR 引擎，请先执行: pip install rapidocr_onnxruntime"
        ) from exc

    ocr = RapidOCR()
    active: list[OcrLine] = []
    finished: list[OcrLine] = []
    frame_interval = 1.0 / fps if fps > 0 else 1.0
    max_gap = max(frame_interval * 3.0, 2.5)

    def finalize_before(now: float) -> None:
        nonlocal active
        still_active: list[OcrLine] = []
        for line in active:
            if now - line.end > max_gap:
                line.end = min(line.end + frame_interval, now)
                finished.append(line)
            else:
                still_active.append(line)
        active = still_active

    for index, (timestamp, frame) in enumerate(frames):
        try:
            result, _ = ocr(str(frame))
        except Exception:  # noqa: BLE001 - a bad frame should not kill the job
            continue
        detections = [
            item
            for item in _dedupe_frame_detections(_detections_from_result(result))
            if item["score"] >= min_score and _acceptable_ocr_text(item["text"])
        ]

        matched_ids: set[int] = set()
        for detection in detections:
            best_index: int | None = None
            best_score = -1.0
            for line_index, line in enumerate(active):
                if line_index in matched_ids:
                    continue
                if timestamp - line.end > max_gap:
                    continue
                similarity = _text_similarity(detection["text"], line.text)
                if similarity < 0.6:
                    continue
                distance = (
                    (detection["cx"] - line.center_x) ** 2
                    + (detection["cy"] - line.center_y) ** 2
                ) ** 0.5
                if distance > max(80.0, 0.9 * max(detection["bh"], line.box_h)):
                    continue
                score = similarity - distance / 2000.0
                if score > best_score:
                    best_index = line_index
                    best_score = score

            if best_index is None:
                active.append(
                    OcrLine(
                        start=timestamp,
                        end=timestamp,
                        text=detection["text"],
                        score=detection["score"],
                        center_x=detection["cx"],
                        center_y=detection["cy"],
                        box_w=detection["bw"],
                        box_h=detection["bh"],
                    )
                )
                matched_ids.add(len(active) - 1)
                continue

            line = active[best_index]
            matched_ids.add(best_index)
            if len(_normalise_text(detection["text"])) > len(_normalise_text(line.text)):
                line.text = detection["text"]
                line.center_x = detection["cx"]
                line.center_y = detection["cy"]
                line.box_w = detection["bw"]
                line.box_h = detection["bh"]
            line.end = timestamp
            line.score = max(line.score, detection["score"])
            line.hits += 1

        finalize_before(timestamp)
        if progress is not None and (index % 20 == 0 or index + 1 == len(frames)):
            progress(index + 1, len(frames))

    finalize_before(float("inf"))
    for line in active:
        finished.append(line)

    # Drop OCR flickers and merge near-identical adjacent lines.
    cleaned = [line for line in finished if line.end - line.start >= 0.4]
    cleaned.sort(key=lambda line: (line.start, line.center_x, line.center_y))
    merged: list[OcrLine] = []
    for line in cleaned:
        if merged:
            previous = merged[-1]
            similarity = _text_similarity(previous.text, line.text)
            close_in_time = line.start <= previous.end + max_gap
            same_area = (
                abs(previous.center_x - line.center_x) < 90
                and abs(previous.center_y - line.center_y) < 120
            )
            if similarity >= 0.82 and close_in_time and same_area:
                if len(_normalise_text(line.text)) > len(_normalise_text(previous.text)):
                    previous.text = line.text
                    previous.center_x = line.center_x
                    previous.center_y = line.center_y
                    previous.box_w = line.box_w
                    previous.box_h = line.box_h
                previous.end = max(previous.end, line.end)
                previous.score = max(previous.score, line.score)
                previous.hits += line.hits
                continue
        merged.append(line)
    return merged


def write_transcript(
    video: BiliVideo,
    lines: list[OcrLine],
    dest: Path,
    *,
    source_url: str,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "=" * 80,
        video.display_title,
        f"来源：{source_url}（UP主：{video.author}）",
        f"视频时长：{format_timestamp(video.duration)}",
        "说明：本文件由 RapidOCR 逐帧识别画面文字生成；【对白】表示竖排对话框，",
        "      【字幕】表示横排字幕/文字卡。OCR 无法可靠判断具体说话角色。",
        "=" * 80,
        "",
    ]
    body: list[str] = []
    last_end = -1.0
    for line in lines:
        if line.end <= line.start:
            continue
        if body and line.start - last_end >= 30.0:
            body.append("")
        start = format_timestamp(line.start)
        tag = "【对白】" if line.vertical else "【字幕】"
        body.append(f"{start} {tag}{line.text}")
        last_end = line.end
    dest.write_text("\n".join(header + body).rstrip() + "\n", encoding="utf-8")
    return dest


def run_bili_ocr(
    url: str,
    output_dir: Path,
    *,
    fps: float = 1.0,
    min_score: float = 0.6,
    skip_download: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[BiliVideo, Path, Path, list[OcrLine]]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"解析链接: {url}")
    video = resolve_bilibili(url)
    log(f"标题: {video.display_title}")
    log(f"作者: {video.author}  分P: P{video.page}  时长: {format_timestamp(video.duration)}  画质: {video.quality}")

    video_path = output_dir / f"{video.bvid or 'bilibili'}-p{video.page}.mp4"
    if skip_download and video_path.exists():
        log(f"复用已下载视频: {video_path}")
    else:
        log(f"下载视频: {video_path}")
        last_report = {"value": 0}

        def report(done: int, total: int) -> None:
            if done - last_report["value"] < 5 * 1024 * 1024 and done < total:
                return
            last_report["value"] = done
            if total:
                log(f"  下载进度: {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB")
            else:
                log(f"  已下载: {done / 1024 / 1024:.1f} MiB")

        download_video(video, video_path, progress=report)

    frames_dir = output_dir / "frames"
    log(f"抽帧: fps={fps}")
    frames = extract_frames(video_path, frames_dir, fps=fps)
    log(f"抽帧完成: {len(frames)} 帧")
    if not frames:
        raise BiliOcrError("未能抽取任何视频帧")

    def ocr_progress(done: int, total: int) -> None:
        log(f"  OCR: {done}/{total}")

    log("RapidOCR 识别中...")
    lines = extract_ocr_lines(frames, fps=fps, min_score=min_score, progress=ocr_progress)
    log(f"识别到文字行: {len(lines)}")

    json_path = output_dir / "ocr_lines.json"
    json_path.write_text(
        json.dumps([line.to_dict() for line in lines], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_path = output_dir / "对话与旁白_OCR.txt"
    write_transcript(video, lines, text_path, source_url=url)
    log(f"输出: {text_path}")
    log(f"结构化: {json_path}")
    return video, video_path, text_path, lines
