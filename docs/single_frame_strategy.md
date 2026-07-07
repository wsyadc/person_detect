# 单帧人物锚定与行为准确率评估实验

本文档说明 `experiment/single-frame-vlm-selection` 分支新增的实验 pipeline。该策略不依赖目标人脸照片，也不修改现有实时摄像头检测逻辑。

## 运行命令

```bash
uv run python -m person_detect.single_frame \
  --jsonl test_dataset/test.jsonl \
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

## Pipeline

1. 用现有 `PersonDetector` 在 CPU 上运行 YOLO person 检测。
2. 如果没有检测到人物框，直接预测 `完全离席`，不调用 VLM。
3. 如果只有一个人物框，直接按 `--crop-scale` 裁剪该人物，并送入行为 VLM。
4. 如果有多个人物框，先在原图上绘制编号框 `1..N`，送入 VLM 做目标选择。
5. 目标选择要求 VLM 只返回 JSON：`{"box_id": 1, "reason": "..."}`。
6. 如果选择 JSON 无法解析、为空或越界，则 fallback 到中心点最接近图像中心的人物框。
7. 对最终选中的人物 crop 调用行为 VLM，抽取行为名作为预测标签。

多人选择规则为：第一优先级选择更像孩子/学生的人物，第二优先级选择更靠近图像中心的人物。

## 行为判定

行为判定使用单帧 prompt。输入已经是选中目标人物的 crop，因此 VLM 只需要判断该人物行为，不需要重新做人脸锚定或多目标跟踪。

`--crop-scale` 支持：

- `1.0`：只截取检测框本身。
- `1.5`：以检测框中心扩展到 1.5 倍，并裁剪到图像边界内。

送入 VLM 前，图片会统一 resize 到 `--frame-width x --frame-height`，默认 `640x480`。

## 输出

每次运行会创建：

```text
eval_outputs/single_frame/{YYYYmmdd_HHMMSS}/
```

其中：

- `predictions.jsonl`：逐图预测明细。
- `summary.json`：整体准确率、按标签 precision/recall/support、混淆矩阵和检测框统计。

`predictions.jsonl` 每行包含：

```json
{
  "id": 1,
  "image_name": "example.jpg",
  "ground_truth_raw": "无任何输出",
  "ground_truth_label": "无任何输出",
  "num_boxes": 1,
  "boxes": [[10, 20, 200, 300]],
  "selected_box_id": 1,
  "selected_box": [10, 20, 200, 300],
  "selection_raw": "",
  "selection_fallback": false,
  "behavior_raw": "",
  "predicted_label": "无任何输出",
  "correct": true,
  "error": ""
}
```

## 评估口径

主指标是行为名 exact match accuracy：

- 标注或模型输出为 `无任何输出` 时，标签归一化为 `无任何输出`。
- 标准 `with` 格式如 `{课堂表现}with{双手托腮}with{...}` 会抽取中间行为名 `双手托腮`。
- 模型空输出视为 `无任何输出`。
- 准确率只比较归一化后的行为名。

## 数据提交约定

- `test_dataset/test.jsonl` 可以提交。
- `test/` 图片目录不提交。
- `test_dataset/*.xlsx` 不提交。
- `eval_outputs/` 评估输出不提交。

## 限制

该策略只基于单帧，没有时间连续性，因此对“揉眼睛”等需要动作轨迹的行为只能依赖单帧视觉线索。多人场景下的目标选择由 VLM 决定，VLM 失败时使用图像中心 fallback，保证整批评估不中断。
