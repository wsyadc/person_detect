import json
from datetime import datetime, timezone

import numpy as np

from person_detect.audit import (
    DEFAULT_BEHAVIOR_RESULT,
    AuditLogger,
    parse_behavior_output,
)


def test_parse_behavior_output_accepts_valid_json() -> None:
    parsed, ok = parse_behavior_output(
        '{"is_abnormal": true, "behavior_type": "课堂表现", '
        '"behavior_name": "摆弄电子设备", "evidence": "手部持有手机。"}'
    )

    assert ok is True
    assert parsed == {
        "is_abnormal": True,
        "behavior_type": "课堂表现",
        "behavior_name": "摆弄电子设备",
        "evidence": "手部持有手机。",
    }


def test_parse_behavior_output_falls_back_for_invalid_model_outputs() -> None:
    invalid_outputs = [
        "",
        "课堂表现 with 摆弄电子设备 with 手部持有手机",
        '```json\n{"is_abnormal": false}\n```',
        '{"is_abnormal": false, "behavior_type": "", "behavior_name": ""}',
        '{"is_abnormal": "false", "behavior_type": "", "behavior_name": "", "evidence": ""}',
    ]

    for raw in invalid_outputs:
        parsed, ok = parse_behavior_output(raw)
        assert ok is False
        assert parsed == DEFAULT_BEHAVIOR_RESULT


def test_audit_logger_saves_images_and_appends_jsonl(tmp_path) -> None:
    started_at = datetime(2026, 7, 6, 10, 57, 7, tzinfo=timezone.utc)
    logger = AuditLogger(log_root=tmp_path, started_at=started_at)
    frame = np.full((12, 16, 3), 127, dtype=np.uint8)
    timestamp = datetime(2026, 7, 6, 10, 57, 7, 123000, tzinfo=timezone.utc)

    image = logger.save_frame(
        frame,
        kind="crop",
        window_index=4,
        frame_index=2,
        timestamp=timestamp,
        box=(1, 2, 3, 4),
        jpeg_quality=80,
    )
    logger.write_record(
        {
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "window_index": 4,
            "record_type": "behavior",
            **DEFAULT_BEHAVIOR_RESULT,
            "images": [image],
            "model_invoked": True,
            "model_parse_ok": False,
            "raw_model_output": "bad",
        }
    )

    assert logger.run_dir == tmp_path / "20260706_105707"
    assert (logger.run_dir / image["path"]).is_file()
    assert image == {
        "path": "20260706_105707_123_w004_f02_crop.jpg",
        "kind": "crop",
        "timestamp": "2026-07-06T10:57:07.123+00:00",
        "box": [1, 2, 3, 4],
    }
    lines = logger.events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["images"] == [image]
