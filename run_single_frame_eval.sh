#!/usr/bin/env bash
set -euo pipefail

# Single-frame VLM selection and behavior evaluation runner.
#
# Basic usage:
#   ./run_single_frame_eval.sh
#
# Override common options with environment variables:
#   CROP_SCALE=1.0 ./run_single_frame_eval.sh
#   BASE_URL=http://10.198.106.42:8011/v1 MODEL=Qwen ./run_single_frame_eval.sh
#   JSONL_PATH=./test_dataset/test.jsonl IMAGE_DIR=./test ./run_single_frame_eval.sh
#
# Pass any supported module argument after the script; later argparse values win:
#   ./run_single_frame_eval.sh --crop-scale 1.0
#   ./run_single_frame_eval.sh --output-dir ./eval_outputs/single_frame_debug
#   ./run_single_frame_eval.sh --help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

JSONL_PATH="${JSONL_PATH:-./test_dataset/test.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-./test}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/single_frame}"

DETECTOR_MODEL="${DETECTOR_MODEL:-yolov8n.pt}"
DET_CONFIDENCE="${DET_CONFIDENCE:-0.25}"
DET_IMAGE_SIZE="${DET_IMAGE_SIZE:-640}"

BASE_URL="${BASE_URL:-http://10.198.106.42:8011/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen}"

CROP_SCALE="${CROP_SCALE:-1.5}"
FRAME_WIDTH="${FRAME_WIDTH:-640}"
FRAME_HEIGHT="${FRAME_HEIGHT:-480}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"

uv run python -m person_detect.single_frame \
    --jsonl "$JSONL_PATH" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --detector-model "$DETECTOR_MODEL" \
    --det-confidence "$DET_CONFIDENCE" \
    --det-image-size "$DET_IMAGE_SIZE" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --crop-scale "$CROP_SCALE" \
    --frame-width "$FRAME_WIDTH" \
    --frame-height "$FRAME_HEIGHT" \
    --jpeg-quality "$JPEG_QUALITY" \
    "$@"
