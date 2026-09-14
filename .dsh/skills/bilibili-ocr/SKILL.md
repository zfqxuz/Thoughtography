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
  --fps 1 \
  --label-speakers \
  --speaker-concurrency 4
```

参数：

- `--fps 1`：每秒 1 帧，适合视觉小说式字幕；
- `--min-score 0.6`：OCR 置信度下限；
- `--skip-download`：输出目录已有视频时复用，不重新下载；
- `--label-speakers / --no-label-speakers`：是否用视觉模型给文字框标注说话人；
- `--speaker-seed <file>`：已知角色名单文件，每行一个名字；
- `--speaker-concurrency 4`：说话人标注并发数。

## 工作流程

1. 调用 `https://xapi.peanutdl.com/beibei` 解析 B 站链接（peanutdl 的接口），获得分 P 标题、时长和 CDN 播放地址。
2. 下载视频到输出目录。
3. 用 ffmpeg 抽帧。
4. 用 RapidOCR 识别每帧文字，按文字内容和位置跨帧合并，生成时间区间。
5. 输出：
   - `对话与旁白_OCR.txt`：按时间排序的文本结果；
   - `对话与旁白_OCR_精选.txt`：过滤 OP/STAFF 后的精简结果；
   - `ocr_lines.json`：结构化时间、文本、置信度、框位置；
   - `speaker_cache/`：说话人标注的视觉模型缓存（可复用，避免重复调用）；
   - `frames/`：抽帧图片；
   - `*.mp4`：下载的原视频。

## 输出格式

```text
00:01 【旁白】这是一个残酷的故事
00:25 【蕾米莉亚】把情报传达给能解决事件的人
00:30 【帕琪·旁白】只要对这类魔导书注入强大的魔力
```

- `【角色名】`：视觉模型根据编号文字框标注的说话人；
- `【角色名·旁白】`：角色独白/画外音；
- `【旁白】`：没有具体说话人的旁白/文字卡；
- `【对白】`/`【字幕】`：未成功标注时的回退标签。

说话人标注只判断“谁在说”，文字仍完全来自 OCR。

## 失败处理

- peanutdl 接口返回验证码字段时，需要先通过浏览器访问 `https://peanutdl.com/zh/bilibili` 完成验证，再重试；
- 若 CDN 直链过期，重新执行解析即可；
- 若 `rapidocr_onnxruntime` 未安装：`pip install rapidocr_onnxruntime`。
