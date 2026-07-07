"""Audit logging helpers for window-level tracking and behavior records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from person_detect.boxes import Box

DEFAULT_BEHAVIOR_RESULT: dict[str, Any] = {
    "is_abnormal": False,
    "behavior_type": "",
    "behavior_name": "",
    "evidence": "",
}
ALLOWED_BEHAVIOR_TYPES = {"", "课堂表现", "健康状态"}
ALLOWED_BEHAVIOR_NAMES = {
    "",
    "趴桌懈怠",
    "摆弄玩具",
    "摆弄电子设备",
    "双手托腮",
    "举手行为",
    "打哈欠",
    "揉眼睛",
}


def parse_behavior_output(raw_output: str) -> tuple[dict[str, Any], bool]:
    """Parse the model JSON output, returning a safe default on any mismatch."""

    try:
        parsed = json.loads(raw_output.strip())
    except (json.JSONDecodeError, AttributeError):
        return dict(DEFAULT_BEHAVIOR_RESULT), False

    if not _is_valid_behavior_result(parsed):
        return dict(DEFAULT_BEHAVIOR_RESULT), False
    return {
        "is_abnormal": parsed["is_abnormal"],
        "behavior_type": parsed["behavior_type"],
        "behavior_name": parsed["behavior_name"],
        "evidence": parsed["evidence"],
    }, True


def _is_valid_behavior_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"is_abnormal", "behavior_type", "behavior_name", "evidence"}
    if set(value) != required:
        return False
    if not isinstance(value["is_abnormal"], bool):
        return False
    if not isinstance(value["behavior_type"], str):
        return False
    if not isinstance(value["behavior_name"], str):
        return False
    if not isinstance(value["evidence"], str):
        return False
    if value["behavior_type"] not in ALLOWED_BEHAVIOR_TYPES:
        return False
    return value["behavior_name"] in ALLOWED_BEHAVIOR_NAMES


class AuditLogger:
    """Persist window evidence images and append structured JSONL records."""

    def __init__(self, *, log_root: str | Path, started_at: datetime) -> None:
        self.log_root = Path(log_root)
        self.run_dir = self.log_root / started_at.strftime("%Y%m%d_%H%M%S")
        self.events_path = self.run_dir / "events.jsonl"
        self._lock = Lock()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(
        self,
        frame,
        *,
        kind: str,
        window_index: int,
        frame_index: int,
        timestamp: datetime,
        box: Box | None = None,
        jpeg_quality: int = 80,
    ) -> dict[str, Any]:
        """Save one JPEG frame and return the JSONL image metadata."""

        import cv2

        filename = _image_filename(
            timestamp=timestamp,
            window_index=window_index,
            frame_index=frame_index,
            kind=kind,
        )
        path = self.run_dir / filename
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"无法保存审计图片: {path}")

        return {
            "path": filename,
            "kind": kind,
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "box": list(box) if box is not None else None,
        }

    def write_record(self, record: dict[str, Any]) -> None:
        """Append one JSON object to the run's JSONL file."""

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


def _image_filename(
    *,
    timestamp: datetime,
    window_index: int,
    frame_index: int,
    kind: str,
) -> str:
    milliseconds = timestamp.microsecond // 1000
    return (
        f"{timestamp:%Y%m%d_%H%M%S}_{milliseconds:03d}_"
        f"w{window_index:03d}_f{frame_index:02d}_{kind}.jpg"
    )
