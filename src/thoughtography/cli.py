from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from .bili_ocr import run_bili_ocr
from .config import ConfigError, load_provider_config, resolve_chat_url
from .media import find_ffmpeg, find_ffprobe
from .pipeline import run_extract

app = typer.Typer(
    name="thoughtography",
    help="从手书/视觉小说视频的画面文字中提取角色对白与旁白，输出 Fountain 台本。",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def check() -> None:
    """检查 ffmpeg / ffprobe 和模型配置是否就绪。"""
    load_dotenv()
    typer.echo(f"ffmpeg:  {find_ffmpeg()}")
    typer.echo(f"ffprobe: {find_ffprobe() or '未找到（将回退解析 ffmpeg -i）'}")

    env_names = [
        "THOUGHTOGRAPHY_PROVIDER",
        "DSH_BASE_URL",
        "DSH_API_KEY",
        "DSH_VISION_MODEL",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_VISION_MODEL",
        "VISION_JSON_MODE",
    ]
    typer.echo("")
    typer.echo("环境变量:")
    for name in env_names:
        value = os.environ.get(name)
        if name.endswith("_API_KEY"):
            shown = "已设置 (***)" if value else "未设置"
        else:
            shown = value or "未设置"
        typer.echo(f"  {name}: {shown}")

    base = os.environ.get("DSH_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
    if base:
        typer.echo("")
        typer.echo(f"解析出的 chat completions 地址示例: {resolve_chat_url(base)}")


@app.command()
def extract(
    video: Path = typer.Argument(..., exists=True, readable=True, help="输入视频文件"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="输出目录"),
    hint: Path | None = typer.Option(None, "--hint", "-h", help="辅助提示词文本文件"),
    provider: str = typer.Option(
        "deepseek",
        "--provider",
        "-p",
        envvar="THOUGHTOGRAPHY_PROVIDER",
        help="模型提供方：dsh / deepseek / 自定义名称",
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="模型 API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="模型 API key"),
    model: str | None = typer.Option(None, "--model", help="Vision 模型名称"),
    title: str | None = typer.Option(None, "--title", help="台本标题，默认使用视频文件名"),
    preview_fps: float = typer.Option(2.0, "--preview-fps", help="预览帧采样率"),
    preview_width: int = typer.Option(640, "--preview-width", help="预览帧宽度"),
    global_threshold: float = typer.Option(
        5.0, "--global-threshold", help="全画面像素平均差阈值"
    ),
    bottom_threshold: float = typer.Option(
        3.0, "--bottom-threshold", help="画面下方文字区域像素差阈值"
    ),
    local_threshold: float = typer.Option(
        5.0, "--local-threshold", help="最显著 0.5% 像素的平均差阈值"
    ),
    hash_threshold: int = typer.Option(
        28, "--hash-threshold", help="dHash 汉明距离阈值"
    ),
    min_segment_seconds: float = typer.Option(
        0.5, "--min-segment-seconds", help="忽略短于此长度的瞬时状态"
    ),
    max_stable_seconds: float = typer.Option(
        20.0,
        "--max-stable-seconds",
        help="画面完全静止时，最长多久强制补一个关键帧",
    ),
    max_keyframes: int | None = typer.Option(
        None, "--max-keyframes", help="关键帧预算；超过后优先合并低变化边界"
    ),
    include_timestamps: bool = typer.Option(
        True,
        "--timestamps/--no-timestamps",
        help="是否在台本中输出 [[开始-结束]] 时间戳",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只做采样并输出 keyframes.json，不调用模型"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="输出调试日志"),
) -> None:
    """提取视频画面文字，生成 Fountain 台本。"""
    load_dotenv()
    _setup_logging(verbose)

    user_hint = ""
    if hint is not None:
        if not hint.exists():
            raise typer.BadParameter(f"提示词文件不存在: {hint}")
        user_hint = hint.read_text(encoding="utf-8")

    provider_config = None
    if dry_run:
        typer.echo("dry-run 模式：只进行视频采样，不调用模型。")
    else:
        try:
            provider_config = load_provider_config(
                provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        except ConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc

    try:
        result = run_extract(
            video,
            output,
            title=title,
            user_hint=user_hint,
            provider_config=provider_config,
            preview_fps=preview_fps,
            preview_width=preview_width,
            global_threshold=global_threshold,
            bottom_threshold=bottom_threshold,
            local_threshold=local_threshold,
            hash_threshold=hash_threshold,
            min_segment_seconds=min_segment_seconds,
            max_stable_seconds=max_stable_seconds,
            max_keyframes=max_keyframes,
            include_timestamps=include_timestamps,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        typer.secho(f"失败: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("完成", fg=typer.colors.GREEN)
    typer.echo(f"关键帧数量: {len(result.keyframes)}")
    typer.echo(f"输出目录: {output.resolve()}")
    if dry_run:
        typer.echo("已输出 sampling_report.txt / keyframes.json / coverage.json")
    else:
        typer.echo(f"模型调用: {result.vision_calls}")
        typer.echo("已输出 script.fountain / script.txt / analysis.json")
    if result.errors:
        typer.echo(f"失败的关键帧: {len(result.errors)} 个，详见 analysis.json")


@app.command()
def bili(
    url: str = typer.Argument(..., help="Bilibili / b23.tv 链接"),
    output: Path = typer.Option(
        Path("output/bili-ocr"), "--output", "-o", help="下载、抽帧和文本输出目录"
    ),
    fps: float = typer.Option(1.0, "--fps", help="OCR 抽帧频率"),
    min_score: float = typer.Option(0.6, "--min-score", help="OCR 置信度下限"),
    skip_download: bool = typer.Option(
        False, "--skip-download", help="输出目录已有视频时跳过下载"
    ),
) -> None:
    """用 peanutdl 解析 B 站视频，并用 RapidOCR 逐帧提取画面字幕。"""
    try:
        run_bili_ocr(
            url,
            output,
            fps=fps,
            min_score=min_score,
            skip_download=skip_download,
            log=typer.echo,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        typer.secho(f"失败: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
