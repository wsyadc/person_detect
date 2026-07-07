# person-detect

macOS 摄像头目标人物检测与跟踪工具。程序每 500ms 处理并显示一帧摄像头画面，用 YOLO 检测人体，用 InsightFace 将画面中的人脸和给定目标照片做相似度匹配。目标人物锚定后，会持续输出人体框，并在 OpenCV 窗口中同时绘制 `1x` 和 `1.5x` 两个框。离席、回座和行为识别统一按 6 帧窗口判定，默认约 3 秒一个窗口。

## 环境安装

本项目使用 `uv` 和 Python 3.11。依赖安装使用清华 PyPI 镜像：

```bash
cd /Users/tal/Desktop/person_detect
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --python 3.11
```

`pyproject.toml` 里也配置了 `tool.uv.index-url`，日常运行时可以直接使用：

```bash
uv run person-detect --face ./face.jpg
```

首次运行会自动下载 `yolov8n.pt` 和 InsightFace `buffalo_s` 模型权重；之后会复用本地缓存。人脸模型默认会优先从 HuggingFace 镜像 `https://hf-mirror.com/vladmandic/insightface-faceanalysis/resolve/main/buffalo_s.zip` 下载，避免 InsightFace 默认 GitHub release 下载过慢。

## 运行

```bash
uv run person-detect --face ./face.jpg --camera 0
```

常用参数：

```bash
uv run person-detect \
  --face ./face.jpg \
  --camera 0 \
  --interval-ms 500 \
  --lost-seconds 5 \
  --face-threshold 0.38 \
  --iou-threshold 0.25 \
  --edge-iou-margin-ratio 0.01 \
  --behavior-window-size 6 \
  --log-root ./log \
  --face-model buffalo_s \
  --face-model-mirror https://hf-mirror.com
```

`--lost-seconds` 保留用于兼容旧命令；当前离席与回座由 `--behavior-window-size` 控制，默认 6 帧，即 2fps 下约 3 秒。

如果想禁用 HuggingFace 镜像预下载，改回 InsightFace 内置的 GitHub 下载源：

```bash
uv run person-detect --face ./face.jpg --face-model-mirror ""
```

无窗口模式：

```bash
uv run person-detect --face ./face.jpg --no-window
```

开启目标人物 crop 行为识别：

```bash
./run_person_detect.sh --behavior-enable
```

行为识别默认连接 OpenAI-compatible vLLM 服务 `http://10.198.106.42:8011/v1`，模型名 `Qwen`，每 6 个目标人物 crop 调用一次大模型。当前默认 2fps 下，相当于约每 3 秒做一次行为判断：

```bash
./run_person_detect.sh \
  --behavior-enable \
  --behavior-base-url http://10.198.106.42:8011/v1 \
  --behavior-api-key EMPTY \
  --behavior-model Qwen \
  --behavior-window-size 6 \
  --behavior-crop-scale 1.5
```

行为模型结果会以解析后的 JSON 打印到终端，同时写入 `events.jsonl`。

每次运行都会创建一个审计目录，默认在 `log/{YYYYmmdd_HHMMSS}/`。其中：

- `events.jsonl`：每行一个 3 秒窗口记录，包括行为 JSON、离席/回座固定事件、图片路径和原始模型输出。
- `*.jpg`：该窗口实际判定使用的 6 张图片。行为和回座记录保存目标 crop；离席记录保存原始摄像头帧。
- 当 VLM 没有输出合法 JSON 时，行为字段会兜底为 `{"is_abnormal": false, "behavior_type": "", "behavior_name": "", "evidence": ""}`，并在 `raw_model_output` 保留原始输出。

如果 macOS 弹出相机权限请求，请允许当前终端或 Codex 访问相机。也可以在“系统设置 -> 隐私与安全性 -> 相机”里手动授权。

## 终端输出

首次检测到目标人物：

```text
锚定成功
```

连续 6 帧未找到目标人物：

```text
完全离席
```

目标人物重新回到画面并被人脸匹配确认：

```text
回到座位
```

回座需要先重新通过人脸匹配确认目标，再连续累计满 6 帧目标框；这个回座确认窗口只写审计日志，不调用 VLM。下一个完整目标窗口才会调用 VLM 做行为判断。

每次成功跟踪时输出两组框，坐标格式为 `(x1,y1,x2,y2)`，左上角为原点：

```text
[BOX] base=(40,50,80,110) scale1.5=(30,35,90,125)
```

## 实现原理

- YOLO 只检测 COCO 的 `person` 类，运行时强制 `device="cpu"`。
- InsightFace 默认使用较轻的 `buffalo_s`，启动前会先尝试从 `hf-mirror.com` 预下载并解压到 `~/.insightface/models/buffalo_s`；模型推理通过 `CPUExecutionProvider` 运行，目标照片会提取一条归一化 embedding。
- 未锚定或已离席时，必须通过人脸相似度重新确认目标。
- 已锚定且未离席时，如果脸短暂不可见，会用上一帧人体框和当前人体框的 IoU 做短期跟踪。
- IoU-only 跟踪会默认忽略左右边缘 1% 范围内的贴边框，避免目标离开画面后把边缘误检框继续当作目标；如需关闭可传 `--edge-iou-margin-ratio 0`。
- 连续 `--behavior-window-size` 帧没有目标框时，写入 `完全离席`；持续离席期间每个窗口继续写 jsonl，但终端不重复刷屏。
- 离席后第 1 帧必须人脸匹配目标，随后连续满一个窗口才写入 `回到座位`；该窗口不会调用 VLM。
- 框扩大以原始框中心为基准，按宽高同比例扩大，并裁剪到图像边界内。
- OpenCV 窗口的显示节奏与 `--interval-ms` 对齐；默认 `500ms` 即约 `2fps`。
- 行为识别默认关闭；开启后仅分析当前已跟踪到的目标人物 crop，不参与身份匹配。
- 行为识别默认使用 `1.5x` crop，以保留手部、桌面、电子设备和玩具等上下文；可用 `--behavior-crop-scale 1.0` 切换为原始目标框。

## 测试

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --python 3.11 pytest -q
```
