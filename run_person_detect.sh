#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

FACE_IMAGE="${FACE_IMAGE:-./face.jpg}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"

# Examples:
#   ./run_person_detect.sh
#   ./run_person_detect.sh --behavior-enable
#   ./run_person_detect.sh --log-root ./log
#   FACE_IMAGE=/path/to/face.jpg CAMERA_INDEX=1 ./run_person_detect.sh --behavior-enable
uv run person-detect \
    --face "$FACE_IMAGE" \
    --camera "$CAMERA_INDEX" \
    --behavior-enable \
    --behavior-base-url http://10.198.106.42:8011/v1 \
    --behavior-api-key EMPTY \
    --behavior-model Qwen \
    --behavior-window-size 6 \
    --behavior-crop-scale 1 \
    "$@"
