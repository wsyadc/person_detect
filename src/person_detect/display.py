"""OpenCV drawing helpers for the live tracking window."""

from __future__ import annotations

from person_detect.boxes import Box, scale_boxes
from person_detect.tracking import TargetTracker, TrackingResult

WINDOW_NAME = "person-detect"
BOX_DRAW_STYLES = [
    (1.5, (0, 180, 255), "1.5x"),
    (1.0, (0, 220, 0), "1x"),
]


def draw_tracking_overlay(
    frame,
    result: TrackingResult,
    tracker: TargetTracker,
) :
    """Draw the latest target box and status text on a video frame."""

    if result.target_box is not None:
        _draw_scaled_boxes(frame, result.target_box)
    _draw_status(frame, _status_text(result, tracker))
    return frame


def _draw_scaled_boxes(frame, box: Box) -> None:
    import cv2

    height, width = frame.shape[:2]
    scaled = scale_boxes(box, (width, height))
    for scale, color, label in BOX_DRAW_STYLES:
        x1, y1, x2, y2 = scaled[scale]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_status(frame, text: str) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _status_text(result: TrackingResult, tracker: TargetTracker) -> str:
    if result.target_box is not None and result.matched_by:
        return f"TRACKING/{result.matched_by.upper()}"
    if tracker.is_absent:
        return "ABSENT"
    if tracker.is_anchored:
        return "TEMPORARILY LOST"
    return "SEARCHING"
