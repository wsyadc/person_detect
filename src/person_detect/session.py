"""Window-level seat-state coordination and behavior audit orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal

from person_detect.audit import (
    DEFAULT_BEHAVIOR_RESULT,
    AuditLogger,
    parse_behavior_output,
)
from person_detect.behavior import BehaviorWindow, image_file_to_data_url, resize_frame
from person_detect.boxes import Box, expand_box
from person_detect.tracking import FULLY_ABSENT, RETURNED, TargetTracker, TrackingResult

SeatState = Literal["present", "absent", "return_confirming"]


@dataclass
class FrameSnapshot:
    """One retained frame plus the tracking metadata needed for window logging."""

    frame: Any
    timestamp: datetime
    target_box: Box | None
    matched_by: str | None


@dataclass(frozen=True)
class BehaviorContext:
    """Metadata kept until an asynchronous VLM request completes."""

    timestamp: datetime
    window_index: int
    images: list[dict[str, Any]]


class SeatWindowCoordinator:
    """Convert per-frame tracker results into 3-second state and behavior records."""

    def __init__(
        self,
        *,
        window_size: int,
        audit_logger: AuditLogger,
        behavior_window: BehaviorWindow | None,
        crop_scale: float = 1.5,
        jpeg_quality: int = 80,
        frame_width: int | None = None,
        frame_height: int | None = None,
        on_event: Callable[[str], None] | None = None,
        on_behavior: Callable[[dict[str, Any]], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.window_size = window_size
        self.audit_logger = audit_logger
        self.behavior_window = behavior_window
        self.crop_scale = crop_scale
        self.jpeg_quality = jpeg_quality
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.on_event = on_event
        self.on_behavior = on_behavior
        self.on_error = on_error

        self._state: SeatState = "present"
        self._missing_frames: list[FrameSnapshot] = []
        self._present_frames: list[FrameSnapshot] = []
        self._return_frames: list[FrameSnapshot] = []
        self._window_index = 0
        self._has_reported_absent = False

    def update(
        self,
        frame,
        result: TrackingResult,
        timestamp: datetime,
        tracker: TargetTracker,
    ) -> None:
        """Consume one frame and emit/log a full window when enough evidence exists."""

        snapshot = FrameSnapshot(
            frame=frame.copy(),
            timestamp=timestamp,
            target_box=result.target_box,
            matched_by=result.matched_by,
        )
        if result.target_box is None:
            self._handle_missing(snapshot, tracker)
            return

        if self._state == "absent":
            if result.matched_by == "face":
                self._start_return_confirmation(snapshot, tracker)
            else:
                self._handle_missing(snapshot, tracker)
            return

        if self._state == "return_confirming":
            self._continue_return_confirmation(snapshot, tracker)
            return

        self._handle_present(snapshot)

    def _handle_missing(
        self,
        snapshot: FrameSnapshot,
        tracker: TargetTracker,
    ) -> None:
        self._present_frames.clear()
        self._return_frames.clear()
        if self._state == "return_confirming":
            self._state = "absent"

        missing_snapshot = FrameSnapshot(
            frame=snapshot.frame,
            timestamp=snapshot.timestamp,
            target_box=None,
            matched_by=None,
        )
        self._missing_frames.append(missing_snapshot)
        if len(self._missing_frames) < self.window_size:
            return

        window = self._missing_frames[: self.window_size]
        self._missing_frames = self._missing_frames[self.window_size :]
        self._write_absent_window(window, tracker)

    def _handle_present(self, snapshot: FrameSnapshot) -> None:
        self._missing_frames.clear()
        self._present_frames.append(snapshot)
        if len(self._present_frames) < self.window_size:
            return

        window = self._present_frames[: self.window_size]
        self._present_frames = self._present_frames[self.window_size :]
        if self.behavior_window is not None:
            self._submit_behavior_window(window)

    def _start_return_confirmation(
        self,
        snapshot: FrameSnapshot,
        tracker: TargetTracker,
    ) -> None:
        self._state = "return_confirming"
        self._missing_frames.clear()
        self._present_frames.clear()
        self._return_frames = [snapshot]
        if len(self._return_frames) >= self.window_size:
            self._write_return_window(self._return_frames, tracker)

    def _continue_return_confirmation(
        self,
        snapshot: FrameSnapshot,
        tracker: TargetTracker,
    ) -> None:
        self._missing_frames.clear()
        self._return_frames.append(snapshot)
        if len(self._return_frames) < self.window_size:
            return
        window = self._return_frames[: self.window_size]
        self._return_frames = []
        self._write_return_window(window, tracker)

    def _write_absent_window(
        self,
        window: list[FrameSnapshot],
        tracker: TargetTracker,
    ) -> None:
        window_index = self._next_window_index()
        images = self._save_raw_window(window, window_index)
        record = self._fixed_record(
            timestamp=window[-1].timestamp,
            window_index=window_index,
            record_type="fully_absent",
            behavior={
                "is_abnormal": True,
                "behavior_type": "课堂表现",
                "behavior_name": FULLY_ABSENT,
                "evidence": f"连续{self.window_size}帧未检测到目标人物。",
            },
            images=images,
        )
        self.audit_logger.write_record(record)
        tracker.mark_absent()
        self._state = "absent"
        if not self._has_reported_absent:
            self._has_reported_absent = True
            self._emit_event(FULLY_ABSENT)

    def _write_return_window(
        self,
        window: list[FrameSnapshot],
        tracker: TargetTracker,
    ) -> None:
        window_index = self._next_window_index()
        images = self._save_crop_window(window, window_index)
        record = self._fixed_record(
            timestamp=window[-1].timestamp,
            window_index=window_index,
            record_type="returned",
            behavior={
                "is_abnormal": False,
                "behavior_type": "",
                "behavior_name": RETURNED,
                "evidence": (
                    f"连续{self.window_size}帧重新检测并跟踪到目标人物，"
                    "本窗口未调用行为模型。"
                ),
            },
            images=images,
        )
        self.audit_logger.write_record(record)
        self._state = "present"
        self._has_reported_absent = False
        self._present_frames.clear()
        self._missing_frames.clear()
        tracker._absent = False
        self._emit_event(RETURNED)

    def _submit_behavior_window(self, window: list[FrameSnapshot]) -> None:
        window_index = self._next_window_index()
        images = self._save_crop_window(window, window_index)
        frame_urls = [
            image_file_to_data_url(self.audit_logger.run_dir / image["path"])
            for image in images
        ]
        context = BehaviorContext(
            timestamp=window[-1].timestamp,
            window_index=window_index,
            images=images,
        )
        future = self.behavior_window.submit(frame_urls)
        future.add_done_callback(
            lambda done_future, done_context=context: self._complete_behavior_window(
                done_future,
                done_context,
            )
        )

    def _complete_behavior_window(self, future, context: BehaviorContext) -> None:
        try:
            raw_output = future.result()
        except Exception as exc:
            raw_output = f"ERROR: {exc}"
            behavior = dict(DEFAULT_BEHAVIOR_RESULT)
            parse_ok = False
            self._emit_error(f"[行为错误] {exc}")
        else:
            behavior, parse_ok = parse_behavior_output(raw_output)

        record = {
            "timestamp": context.timestamp.isoformat(timespec="milliseconds"),
            "window_index": context.window_index,
            "record_type": "behavior",
            **behavior,
            "images": context.images,
            "model_invoked": True,
            "model_parse_ok": parse_ok,
            "raw_model_output": raw_output,
        }
        self.audit_logger.write_record(record)
        if self.on_behavior is not None:
            self.on_behavior(record)

    def _fixed_record(
        self,
        *,
        timestamp: datetime,
        window_index: int,
        record_type: str,
        behavior: dict[str, Any],
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "window_index": window_index,
            "record_type": record_type,
            **behavior,
            "images": images,
            "model_invoked": False,
            "model_parse_ok": True,
            "raw_model_output": "",
        }

    def _save_raw_window(
        self,
        window: list[FrameSnapshot],
        window_index: int,
    ) -> list[dict[str, Any]]:
        return [
            self.audit_logger.save_frame(
                snapshot.frame,
                kind="raw",
                window_index=window_index,
                frame_index=frame_index,
                timestamp=snapshot.timestamp,
                jpeg_quality=self.jpeg_quality,
            )
            for frame_index, snapshot in enumerate(window)
        ]

    def _save_crop_window(
        self,
        window: list[FrameSnapshot],
        window_index: int,
    ) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for frame_index, snapshot in enumerate(window):
            if snapshot.target_box is None:
                continue
            height, width = snapshot.frame.shape[:2]
            crop_box = expand_box(snapshot.target_box, self.crop_scale, (width, height))
            x1, y1, x2, y2 = crop_box
            crop = snapshot.frame[y1:y2, x1:x2].copy()
            if self.frame_width is not None and self.frame_height is not None:
                crop = resize_frame(crop, self.frame_width, self.frame_height)
            images.append(
                self.audit_logger.save_frame(
                    crop,
                    kind="crop",
                    window_index=window_index,
                    frame_index=frame_index,
                    timestamp=snapshot.timestamp,
                    box=crop_box,
                    jpeg_quality=self.jpeg_quality,
                )
            )
        return images

    def _next_window_index(self) -> int:
        window_index = self._window_index
        self._window_index += 1
        return window_index

    def _emit_event(self, event: str) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _emit_error(self, error: str) -> None:
        if self.on_error is not None:
            self.on_error(error)
