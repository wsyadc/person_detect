# 原生 VLM 全图行为识别评测

该功能用于评估 VLM 直接看完整课堂图片时的行为识别能力。它不做人体检测、不选框、不裁剪，也不使用 `test_filtered.jsonl`。

## 运行命令

```bash
uv run python -m person_detect.native_vlm \
  --jsonl test_dataset/test.jsonl \
  --image-dir test \
  --output-dir eval_outputs/native_vlm
```

或使用脚本：

```bash
./run_native_vlm_eval.sh
WORKERS=2 ./run_native_vlm_eval.sh
SAVE_AUDIT_IMAGES=1 ./run_native_vlm_eval.sh
```

默认 VLM 配置：

```text
base-url: http://10.198.106.42:8011/v1
api-key: EMPTY
model: Qwen
```

## Pipeline

1. 从 `test_dataset/test.jsonl` 读取样本。
2. 从 `test/` 读取整张图片。
3. 将整图 resize 到 `--frame-width x --frame-height`，默认 `640x480`。
4. 将 resize 后整图直接送入 VLM。
5. 解析模型输出中的行为名并与 `ground_truth` 做 exact match。

## 输出解析

正常输出兼容：

- 空输出
- `无任何输出`
- `无异常`

这些都会归一为 `无异常`。

异常输出使用 `with` 格式解析：

```text
{行为类型}with{具体行为名称}with{视觉证据}
```

可识别标签：

```text
完全离席、趴桌懈怠、摆弄玩具、摆弄电子设备、遮挡面部、仰头、东张西望、双手托腮、揉眼睛、打哈欠、举手行为、无异常
```

无法解析成上述标签时，预测兜底为 `无异常`，并设置 `parse_ok=false`。

## 输出文件

每次运行创建：

```text
eval_outputs/native_vlm/{YYYYmmdd_HHMMSS}/
```

其中：

- `predictions.jsonl`：逐图预测明细。
- `summary.json`：整体准确率、每类准确率、precision/recall/support 和混淆矩阵。
- `audit/`：仅在 `--save-audit-images` 或 `SAVE_AUDIT_IMAGES=1` 开启时生成。

`predictions.jsonl` 每行包含：

```json
{
  "sample_index": 0,
  "image_name": "example.jpg",
  "ground_truth": "无异常",
  "predicted_label": "无异常",
  "correct": true,
  "raw_model_output": "无任何输出",
  "parse_ok": true,
  "error": "",
  "audit_images": {}
}
```

开启过程图片后，每个样本保存：

```text
audit/samples/{sample_index}_{image_stem}/vlm_input_resized.jpg
```

该图片就是实际送入 VLM 的 resize 后全图。

## 与单帧检测策略的区别

- `native_vlm`：直接评估 VLM 看全图的能力。
- `single_frame`：先用 YOLO 检测人体，多人时让 VLM 选目标，再对目标 crop 做行为判断。

两者输出目录不同，可以并行保留结果用于对比。
