"""Target-person tracking state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from person_detect.boxes import Box, ImageSize, iou

ANCHOR_SUCCESS = "锚定成功"
FULLY_ABSENT = "完全离席"
RETURNED = "回到座位"
DEFAULT_EDGE_IOU_MARGIN_RATIO = 0.01

MatchSource = Literal["face", "iou"]


@dataclass(frozen=True)
class PersonCandidate:
    """A detected person body, optionally annotated with target-face similarity."""

    box: Box
    face_score: float | None = None


@dataclass(frozen=True)
class TrackingResult:
    """Result produced by one tracker update."""

    target_box: Box | None
    event: str | None = None
    matched_by: MatchSource | None = None
    score: float | None = None


class TargetTracker:
    """Track one anchored target person across webcam frames.

    The tracker uses target-face similarity to anchor or recover the person.
    While the target is present, it can bridge short face occlusions with body
    box IoU. After full absence is declared, IoU is intentionally ignored until
    the target face is matched again.
    """

    def __init__(
        self,
        *,
        face_threshold: float = 0.38,
        iou_threshold: float = 0.25,
        lost_seconds: float = 5.0,
        edge_iou_margin_ratio: float = DEFAULT_EDGE_IOU_MARGIN_RATIO,
    ) -> None:
        if edge_iou_margin_ratio < 0:
            raise ValueError("edge_iou_margin_ratio must be non-negative")

        self.face_threshold = face_threshold
        self.iou_threshold = iou_threshold
        self.lost_seconds = lost_seconds
        self.edge_iou_margin_ratio = edge_iou_margin_ratio
        self.last_box: Box | None = None
        self.last_seen: float | None = None
        self._anchored = False
        self._absent = False

    @property
    def is_absent(self) -> bool:
        """Whether the target has crossed the full-absence threshold."""

        return self._absent

    @property
    def is_anchored(self) -> bool:
        """Whether the target has ever been identified by face matching."""

        return self._anchored

    def update(
        self,
        candidates: list[PersonCandidate],
        *,
        now: float,
        image_size: ImageSize | None = None,
    ) -> TrackingResult:
        """Update tracker state from one frame of person candidates."""

        face_match = self._best_face_match(candidates)
        if face_match is not None:
            return self._accept_face_match(face_match, now)

        if self._anchored and not self._absent:
            iou_match = self._best_iou_match(candidates, image_size=image_size)
            if iou_match is not None:
                candidate, match_score = iou_match
                self.last_box = candidate.box
                self.last_seen = now
                return TrackingResult(
                    target_box=candidate.box,
                    matched_by="iou",
                    score=match_score,
                )

        return self._mark_lost_if_needed(now)

    def mark_absent(self) -> None:
        """Force the tracker into absence mode until the target face reappears."""

        self._absent = True

    def _best_face_match(
        self,
        candidates: list[PersonCandidate],
    ) -> PersonCandidate | None:
        scored = [
            candidate
            for candidate in candidates
            if candidate.face_score is not None
            and candidate.face_score >= self.face_threshold
        ]
        if not scored:
            return None
        return max(scored, key=lambda candidate: candidate.face_score or 0.0)

    def _best_iou_match(
        self,
        candidates: list[PersonCandidate],
        *,
        image_size: ImageSize | None = None,
    ) -> tuple[PersonCandidate, float] | None:
        if self.last_box is None:
            return None
        if image_size is not None:
            candidates = [
                candidate
                for candidate in candidates
                if not self._is_horizontal_edge_clipped(candidate.box, image_size)
            ]
        ranked = [
            (iou(self.last_box, candidate.box), candidate)
            for candidate in candidates
        ]
        if not ranked:
            return None
        best_score, best_candidate = max(ranked, key=lambda item: item[0])
        if best_score < self.iou_threshold:
            return None
        return best_candidate, best_score

    def _is_horizontal_edge_clipped(self, box: Box, image_size: ImageSize) -> bool:
        """Whether a no-face IoU candidate is likely an entering/leaving fragment."""

        if self.edge_iou_margin_ratio == 0:
            return False
        width, _height = image_size
        x1, _y1, x2, _y2 = box
        margin = max(1, round(width * self.edge_iou_margin_ratio))
        return x1 <= margin or x2 >= width - margin

    def _accept_face_match(
        self,
        candidate: PersonCandidate,
        now: float,
    ) -> TrackingResult:
        event = None
        if not self._anchored:
            event = ANCHOR_SUCCESS

        self._anchored = True
        self._absent = False
        self.last_box = candidate.box
        self.last_seen = now
        return TrackingResult(
            target_box=candidate.box,
            event=event,
            matched_by="face",
            score=candidate.face_score,
        )

    def _mark_lost_if_needed(self, now: float) -> TrackingResult:
        return TrackingResult(target_box=None)
