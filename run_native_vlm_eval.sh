#!/usr/bin/env bash
set -euo pipefail

# Native full-image VLM behavior evaluation runner.
#
# Basic usage:
#   ./run_native_vlm_eval.sh
#
# Override common options with environment variables:
#   BASE_URL=http://10.198.106.42:8011/v1 MODEL=Qwen ./run_native_vlm_eval.sh
#   JSONL_PATH=./test_dataset/test.jsonl IMAGE_DIR=./test ./run_native_vlm_eval.sh
#   WORKERS=2 ./run_native_vlm_eval.sh
#   SAVE_AUDIT_IMAGES=1 ./run_native_vlm_eval.sh
#
# Pass any supported module argument after the script; later argparse values win:
#   ./run_native_vlm_eval.sh --workers 1
#   ./run_native_vlm_eval.sh --output-dir ./eval_outputs/native_vlm_debug
#   ./run_native_vlm_eval.sh --save-audit-images
#   ./run_native_vlm_eval.sh --help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

JSONL_PATH="${JSONL_PATH:-./test_dataset/test.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-./test}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/native_vlm}"

BASE_URL="${BASE_URL:-http://10.198.106.42:8011/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen}"

FRAME_WIDTH="${FRAME_WIDTH:-640}"
FRAME_HEIGHT="${FRAME_HEIGHT:-480}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
WORKERS="${WORKERS:-4}"
SAVE_AUDIT_IMAGES="${SAVE_AUDIT_IMAGES:-0}"

EXTRA_ARGS=()
SAVE_AUDIT_IMAGES_NORMALIZED="$(printf '%s' "$SAVE_AUDIT_IMAGES" | tr '[:upper:]' '[:lower:]')"
case "$SAVE_AUDIT_IMAGES_NORMALIZED" in
    1|true|yes|on)
        EXTRA_ARGS+=(--save-audit-images)
        ;;
esac

uv run python -m person_detect.native_vlm \
    --jsonl "$JSONL_PATH" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --frame-width "$FRAME_WIDTH" \
    --frame-height "$FRAME_HEIGHT" \
    --jpeg-quality "$JPEG_QUALITY" \
    --workers "$WORKERS" \
    "${EXTRA_ARGS[@]}" \
    "$@"
