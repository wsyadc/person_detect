import argparse

import numpy as np

from person_detect.cli import (
    build_parser,
    _create_behavior_window,
    _print_result,
    _process_frame,
)
from person_detect.detector import PersonDetection
from person_detect.tracking import PersonCandidate, TargetTracker, TrackingResult


class FakeDetector:
    """Detector double returning one configured person box."""

    def detect(self, frame):
        return [PersonDetection(box=(10, 10, 100, 200), confidence=0.91)]


class FakeMatcher:
    """Face matcher double assigning a configured target score."""

    def __init__(self, score: float | None) -> None:
        self.score = score

    def annotate_candidates(self, frame, detections):
        return [
            PersonCandidate(box=detection.box, face_score=self.score)
            for detection in detections
        ]


def test_process_frame_wires_detection_identity_and_tracking() -> None:
    tracker = TargetTracker(face_threshold=0.38, iou_threshold=0.25, lost_seconds=5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = _process_frame(
        frame,
        FakeDetector(),
        FakeMatcher(score=0.7),
        tracker,
        now=1.0,
    )

    assert result.event == "锚定成功"
    assert result.target_box == (10, 10, 100, 200)
    assert result.matched_by == "face"


def test_print_result_emits_event_and_scaled_box_line(capsys) -> None:
    _print_result(
        TrackingResult(
            target_box=(40, 50, 80, 110),
            event="锚定成功",
            matched_by="face",
            score=0.8,
        ),
        (160, 120, 3),
    )

    assert capsys.readouterr().out == (
        "锚定成功\n"
        "[BOX] base=(40,50,80,110) scale1.5=(30,35,90,125)\n"
    )


def test_create_behavior_window_returns_none_when_behavior_is_disabled() -> None:
    args = argparse.Namespace(behavior_enable=False)

    assert _create_behavior_window(args) is None


def test_parser_exposes_log_root_and_shared_window_size() -> None:
    help_text = build_parser().format_help()

    assert "--log-root" in help_text
    assert "--behavior-window-size" in help_text
