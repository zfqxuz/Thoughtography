# Thoughtography

从静态手书 / 视觉小说视频的画面文字中提取角色对白和旁白，整理为带时间戳的
Fountain 剧本。

## 环境依赖

- Python 3.11+
- `ffmpeg` 与 `ffprobe`（也支持通过 `imageio-ffmpeg` 自带的 ffmpeg 作为回退）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# 填写 DSH_* 或 DEEPSEEK_* 配置
```

模型配置三项分别是什么：

- `*_BASE_URL`：服务商给的 API 地址（控制台/文档里的“接口地址”或 “Base URL”）。
- `*_API_KEY`：服务商控制台复制的密钥。
- `*_VISION_MODEL`：要调用的视觉模型 ID（模型列表里的名字）。

如果服务商是 OpenAI 兼容接口，通常可以直接使用；如果接口路径特殊，把完整
`.../chat/completions` 地址填进 `*_BASE_URL` 也能识别。

## 使用

```bash
thoughtography check
thoughtography extract input.mp4 --output output/ --hint hints.txt --provider dsh
```

如果只想验证采样和覆盖率、不调用视觉模型：

```bash
thoughtography extract input.mp4 --output output/ --dry-run
```

输出文件：

- `script.fountain`：Fountain 格式台本，带 `[[开始-结束]]` 时间戳
- `script.txt`：纯文本台本
- `analysis.json`：结构化对白 / 旁白 / 角色 / 场景
- `keyframes.json`：关键帧采样与变化分数
