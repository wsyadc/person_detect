import json
from datetime import datetime, timedelta, timezone

import numpy as np

from person_detect.audit import AuditLogger
from person_detect.session import SeatWindowCoordinator
from person_detect.tracking import TargetTracker, TrackingResult


class ImmediateFuture:
    def __init__(self, result_text: str) -> None:
        self.result_text = result_text

    def result(self) -> str:
        return self.result_text

    def add_done_callback(self, callback) -> None:
        callback(self)


class StubBehaviorWindow:
    def __init__(self, result_text: str | None = None) -> None:
        self.result_text = result_text or (
            '{"is_abnormal": false, "behavior_type": "", '
            '"behavior_name": "", "evidence": ""}'
        )
        self.submissions: list[list[str]] = []

    def submit(self, frame_urls: list[str]) -> ImmediateFuture:
        self.submissions.append(frame_urls)
        return ImmediateFuture(self.result_text)


def _frame(value: int = 0):
    return np.full((30, 40, 3), value, dtype=np.uint8)


def _time(offset: int) -> datetime:
    base = datetime(2026, 7, 6, 10, 57, 0, tzinfo=timezone.utc)
    return base + timedelta(milliseconds=500 * offset)


def _records(logger: AuditLogger) -> list[dict]:
    if not logger.events_path.exists():
        return []
    return [
        json.loads(line)
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
    ]


def test_absent_window_logs_raw_images_and_marks_tracker_absent(tmp_path) -> None:
    logger = AuditLogger(log_root=tmp_path, started_at=_time(0))
    tracker = TargetTracker()
    events: list[str] = []
    coordinator = SeatWindowCoordinator(
        window_size=2,
        audit_logger=logger,
        behavior_window=None,
        on_event=events.append,
    )

    coordinator.update(_frame(1), TrackingResult(target_box=None), _time(1), tracker)
    coordinator.update(_frame(2), TrackingResult(target_box=None), _time(2), tracker)

    assert events == ["完全离席"]
    assert tracker.is_absent is True
    records = _records(logger)
    assert [record["record_type"] for record in records] == ["fully_absent"]
    assert records[0]["behavior_name"] == "完全离席"
    assert records[0]["model_invoked"] is False
    assert [image["kind"] for image in records[0]["images"]] == ["raw", "raw"]


def test_continuous_absence_logs_every_window_without_reprinting_event(tmp_path) -> None:
    logger = AuditLogger(log_root=tmp_path, started_at=_time(0))
    tracker = TargetTracker()
    events: list[str] = []
    coordinator = SeatWindowCoordinator(
        window_size=2,
        audit_logger=logger,
        behavior_window=None,
        on_event=events.append,
    )

    for index in range(4):
        coordinator.update(_frame(index), TrackingResult(target_box=None), _time(index), tracker)

    assert events == ["完全离席"]
    records = _records(logger)
    assert [record["record_type"] for record in records] == [
        "fully_absent",
        "fully_absent",
    ]


def test_return_requires_face_first_and_skips_vlm_for_return_window(tmp_path) -> None:
    logger = AuditLogger(log_root=tmp_path, started_at=_time(0))
    tracker = TargetTracker()
    behavior = StubBehaviorWindow()
    events: list[str] = []
    coordinator = SeatWindowCoordinator(
        window_size=2,
        audit_logger=logger,
        behavior_window=behavior,
        on_event=events.append,
    )
    for index in range(2):
        coordinator.update(_frame(index), TrackingResult(target_box=None), _time(index), tracker)

    coordinator.update(
        _frame(2),
        TrackingResult(target_box=(2, 2, 20, 24), matched_by="iou"),
        _time(2),
        tracker,
    )
    coordinator.update(
        _frame(3),
        TrackingResult(target_box=(2, 2, 20, 24), matched_by="face"),
        _time(3),
        tracker,
    )
    coordinator.update(
        _frame(4),
        TrackingResult(target_box=(3, 2, 21, 24), matched_by="iou"),
        _time(4),
        tracker,
    )

    assert events == ["完全离席", "回到座位"]
    assert behavior.submissions == []
    records = _records(logger)
    assert [record["record_type"] for record in records] == [
        "fully_absent",
        "returned",
    ]
    assert records[1]["behavior_name"] == "回到座位"
    assert [image["kind"] for image in records[1]["images"]] == ["crop", "crop"]


def test_behavior_window_after_return_invokes_vlm_and_logs_parsed_json(tmp_path) -> None:
    logger = AuditLogger(log_root=tmp_path, started_at=_time(0))
    tracker = TargetTracker()
    behavior = StubBehaviorWindow(
        '{"is_abnormal": true, "behavior_type": "课堂表现", '
        '"behavior_name": "双手托腮", "evidence": "双手托举下巴。"}'
    )
    behavior_records: list[dict] = []
    coordinator = SeatWindowCoordinator(
        window_size=2,
        audit_logger=logger,
        behavior_window=behavior,
        on_behavior=behavior_records.append,
    )
    for index in range(2):
        coordinator.update(_frame(index), TrackingResult(target_box=None), _time(index), tracker)
    coordinator.update(
        _frame(2),
        TrackingResult(target_box=(2, 2, 20, 24), matched_by="face"),
        _time(2),
        tracker,
    )
    coordinator.update(
        _frame(3),
        TrackingResult(target_box=(3, 2, 21, 24), matched_by="iou"),
        _time(3),
        tracker,
    )

    coordinator.update(
        _frame(4),
        TrackingResult(target_box=(4, 2, 22, 24), matched_by="iou"),
        _time(4),
        tracker,
    )
    coordinator.update(
        _frame(5),
        TrackingResult(target_box=(5, 2, 23, 24), matched_by="iou"),
        _time(5),
        tracker,
    )

    assert len(behavior.submissions) == 1
    assert len(behavior_records) == 1
    records = _records(logger)
    assert [record["record_type"] for record in records] == [
        "fully_absent",
        "returned",
        "behavior",
    ]
    assert records[2]["is_abnormal"] is True
    assert records[2]["behavior_name"] == "双手托腮"
    assert records[2]["model_parse_ok"] is True
    assert [image["kind"] for image in records[2]["images"]] == ["crop", "crop"]
