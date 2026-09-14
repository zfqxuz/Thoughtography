---
name: bilibili-ocr
description: 解析 Bilibili / b23.tv 视频，下载后使用 RapidOCR 逐帧提取画面字幕，输出按时间排序的中文对白与旁白文本。
whenToUse: 当用户给出 B 站或 b23.tv 链接，并希望提取视频画面字幕、台词、对话、旁白时。
---

# Bilibili OCR 字幕提取

## 何时使用

用户给出 Bilibili 链接（`bilibili.com/video/...`、`b23.tv/...`）并要求：

- 提取视频里的台词 / 对话 / 旁白；
- 把画面字幕转成文字；
- 不希望通过视觉大模型逐帧识别，优先使用本地 OCR。

## 前置条件

- 本仓库已安装 `rapidocr_onnxruntime`（在 Thoughtography 虚拟环境中）。
- 使用项目虚拟环境中的 `thoughtography` 命令。

## 使用方式

```bash
cd /home/zfq/Thoughtography
.venv/bin/thoughtography bili '<B站或b23链接>' \
  --output output/bili-ocr/<任务名> \
  --fps 1
```

参数：

- `--fps 1`：每秒 1 帧，适合视觉小说式字幕；
- `--min-score 0.6`：OCR 置信度下限；
- `--skip-download`：输出目录已有视频时复用，不重新下载。

## 工作流程

1. 调用 `https://xapi.peanutdl.com/beibei` 解析 B 站链接（peanutdl 的接口），获得分 P 标题、时长和 CDN 播放地址。
2. 下载视频到输出目录。
3. 用 ffmpeg 抽帧。
4. 用 RapidOCR 识别每帧文字，按文字内容和位置跨帧合并，生成时间区间。
5. 输出：
   - `对话与旁白_OCR.txt`：按时间排序的文本结果；
   - `ocr_lines.json`：结构化时间、文本、置信度、框位置；
   - `frames/`：抽帧图片；
   - `*.mp4`：下载的原视频。

## 输出格式

```text
00:01 【字幕】这是一个残酷的故事
00:25 【对白】把情报传达给能解决事件的人
```

- `【对白】`：竖排对话框识别结果；
- `【字幕】`：横排字幕、文字卡、标题说明等。

OCR 只能识别文字，无法可靠判断具体说话角色。需要角色区分时，再在 OCR 结果上叠加人工或视觉模型标注。

## 失败处理

- peanutdl 接口返回验证码字段时，需要先通过浏览器访问 `https://peanutdl.com/zh/bilibili` 完成验证，再重试；
- 若 CDN 直链过期，重新执行解析即可；
- 若 `rapidocr_onnxruntime` 未安装：`pip install rapidocr_onnxruntime`。
