# 单帧人物锚定与行为准确率评估实验

本文档说明 `experiment/single-frame-vlm-selection` 分支新增的实验 pipeline。该策略不依赖目标人脸照片，也不修改现有实时摄像头检测逻辑。

## 运行命令

```bash
uv run python -m person_detect.single_frame \
  --jsonl test_dataset/test_filtered.jsonl \
  --image-dir test \
  --output-dir eval_outputs/single_frame \
  --crop-scale 1.5
```

默认 VLM 配置复用当前项目设置：

```text
base-url: http://10.198.106.42:8011/v1
api-key: EMPTY
model: Qwen
```

可按需覆盖：

```bash
uv run python -m person_detect.single_frame \
  --base-url http://10.198.106.42:8011/v1 \
  --api-key EMPTY \
  --model Qwen \
  --crop-scale 1.0
```

或直接使用脚本：

```bash
./run_single_frame_eval.sh
WORKERS=2 ./run_single_frame_eval.sh
SAVE_AUDIT_IMAGES=1 ./run_single_frame_eval.sh
```

## Pipeline

1. 用现有 `PersonDetector` 在 CPU 上运行 YOLO person 检测。
2. 如果没有检测到人物框，直接预测 `完全离席`，不调用 VLM。
3. 如果只有一个人物框，直接按 `--crop-scale` 裁剪该人物，并送入行为 VLM。
4. 如果有多个人物框，先在原图上绘制编号框 `1..N`，送入 VLM 做目标选择。
5. 目标选择要求 VLM 只返回 JSON：`{"box_id": 1, "reason": "..."}`。
6. 如果选择 JSON 无法解析、为空或越界，则 fallback 到中心点最接近图像中心的人物框。
7. 对最终选中的人物 crop 调用行为 VLM，解析 JSON 中的 `behavior_name` 作为预测标签。

多人选择规则为：第一优先级选择更像孩子/学生的人物，第二优先级选择更靠近图像中心的人物。

## 行为判定

行为判定使用单帧 JSON prompt。输入已经是选中目标人物的 crop，因此 VLM 只需要判断该人物行为，不需要重新做人脸锚定或多目标跟踪。

模型必须输出严格 JSON：

```json
{
  "evidence": ["基于图片可验证的视觉事实，1-2条"],
  "behavior_name": "趴桌懈怠"
}
```

允许的行为标签为：`趴桌懈怠`、`摆弄玩具`、`摆弄电子设备`、`双手托腮`、`揉眼睛`、`打哈欠`、`无异常`。如果模型输出不可解析 JSON、字段缺失、字段类型错误或行为名越界，预测兜底为 `无异常`。

`--crop-scale` 支持：

- `1.0`：只截取检测框本身。
- `1.5`：以检测框中心扩展到 1.5 倍，并裁剪到图像边界内。

送入 VLM 前，图片会统一 resize 到 `--frame-width x --frame-height`，默认 `640x480`。

## 过滤测试集

默认评测文件为：

```text
test_dataset/test_filtered.jsonl
```

它由原始 `test_dataset/test.jsonl` 生成，过滤唯一标准是 `ground_truth` 归一化后是否属于以下 8 类：

```text
趴桌懈怠、摆弄玩具、摆弄电子设备、双手托腮、揉眼睛、打哈欠、无异常、完全离席
```

生成命令：

```bash
uv run python -m person_detect.single_frame \
  --build-filtered-jsonl \
  --jsonl test_dataset/test.jsonl \
  --filtered-jsonl test_dataset/test_filtered.jsonl
```

过滤后的每行只保留：

```json
{"image_name": "example.jpg", "ground_truth": "无异常"}
```

## 输出

每次运行会创建：

```text
eval_outputs/single_frame/{YYYYmmdd_HHMMSS}/
```

其中：

- `predictions.jsonl`：逐图预测明细。
- `summary.json`：整体准确率、按标签 precision/recall/support、混淆矩阵和检测框统计。
- `audit/`：仅在 `--save-audit-images` 或 `SAVE_AUDIT_IMAGES=1` 开启时生成的过程图片。

`predictions.jsonl` 每行包含：

```json
{
  "sample_index": 0,
  "image_name": "example.jpg",
  "ground_truth": "无异常",
  "num_boxes": 1,
  "boxes": [[10, 20, 200, 300]],
  "selected_box_id": 1,
  "selected_box": [10, 20, 200, 300],
  "selection_raw": "",
  "selection_source": "",
  "selection_fallback": false,
  "behavior_raw": "{\"evidence\": [], \"behavior_name\": \"无异常\"}",
  "behavior_result": {"evidence": [], "behavior_name": "无异常"},
  "behavior_parse_ok": true,
  "predicted_label": "无异常",
  "correct": true,
  "audit_images": {},
  "error": ""
}
```

开启过程图片后，单样本目录位于：

```text
eval_outputs/single_frame/{YYYYmmdd_HHMMSS}/audit/samples/{sample_index}_{image_stem}/
```

可能包含：

- `detector_input.jpg`：进入人体检测的原图。
- `selection_input_resized.jpg`：多人框场景下送入选框 VLM 的编号图。
- `selection_result.jpg`：多人框场景下最终选中框复查图。
- `behavior_crop_resized.jpg`：最终送入行为 VLM 的 resize 后 crop。

## 评估口径

主指标是行为名 exact match accuracy：

- 标注或模型输出为 `无任何输出` 时，兼容归一化为 `无异常`。
- 标准 `with` 格式如 `{课堂表现}with{双手托腮}with{...}` 会抽取中间行为名 `双手托腮`。
- 模型空输出视为 `无异常`。
- 准确率只比较归一化后的行为名。
- `summary.json` 同时输出整体准确率和 8 类 `per_label_accuracy`。

## 数据提交约定

- `test_dataset/test.jsonl` 可以提交。
- `test_dataset/test_filtered.jsonl` 可以提交，作为默认评测集。
- `test/` 图片目录不提交。
- `test_dataset/*.xlsx` 不提交。
- `eval_outputs/` 评估输出不提交。

## 限制

该策略只基于单帧，没有时间连续性，因此对“揉眼睛”等需要动作轨迹的行为只能依赖单帧视觉线索。多人场景下的目标选择由 VLM 决定，VLM 失败时使用图像中心 fallback，保证整批评估不中断。
