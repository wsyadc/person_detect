"""Single-frame person selection and behavior classification pipeline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from person_detect.behavior import (
    DEFAULT_BEHAVIOR_API_KEY,
    DEFAULT_BEHAVIOR_BASE_URL,
    DEFAULT_BEHAVIOR_FRAME_HEIGHT,
    DEFAULT_BEHAVIOR_FRAME_WIDTH,
    DEFAULT_BEHAVIOR_JPEG_QUALITY,
    DEFAULT_BEHAVIOR_MODEL,
    encode_frame_to_data_url,
)
from person_detect.boxes import Box, expand_box
from person_detect.detector import PersonDetection
from person_detect.single_frame.prompts import BEHAVIOR_PROMPT, TARGET_SELECTION_PROMPT

NORMAL_LABEL = "无任何输出"
BEHAVIOR_PRIORITY = [
    "完全离席",
    "趴桌懈怠",
    "摆弄玩具",
    "摆弄电子设备",
    "遮挡面部",
    "仰头",
    "东张西望",
    "双手托腮",
    "揉眼睛",
    "打哈欠",
    "举手行为",
]
KNOWN_LABELS = {NORMAL_LABEL, *BEHAVIOR_PRIORITY}


class DetectorLike(Protocol):
    """Minimal detector interface used by the experiment pipeline."""

    def detect(self, frame) -> list[PersonDetection]:
        """Return person detections for one BGR frame."""


class VlmLike(Protocol):
    """Minimal VLM interface used by the experiment pipeline."""

    def select_target(self, image_url: str, detections: list[PersonDetection]) -> str:
        """Return a JSON target selection string."""

    def classify_behavior(self, image_url: str) -> str:
        """Return the behavior classification text."""


@dataclass(frozen=True)
class SingleFrameResult:
    """Prediction result for one image."""

    image_name: str = ""
    num_boxes: int = 0
    boxes: list[Box] | None = None
    selected_box_id: int | None = None
    selected_box: Box | None = None
    selection_raw: str = ""
    selection_fallback: bool = False
    behavior_raw: str = ""
    predicted_label: str = NORMAL_LABEL
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert the result into a JSON-serializable prediction record fragment."""

        return {
            "image_name": self.image_name,
            "num_boxes": self.num_boxes,
            "boxes": [list(box) for box in (self.boxes or [])],
            "selected_box_id": self.selected_box_id,
            "selected_box": list(self.selected_box) if self.selected_box else None,
            "selection_raw": self.selection_raw,
            "selection_fallback": self.selection_fallback,
            "behavior_raw": self.behavior_raw,
            "predicted_label": self.predicted_label,
            "error": self.error,
        }


class OpenAICompatibleVLM:
    """OpenAI-compatible VLM client for target selection and behavior classification."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BEHAVIOR_BASE_URL,
        api_key: str = DEFAULT_BEHAVIOR_API_KEY,
        model: str = DEFAULT_BEHAVIOR_MODEL,
        client=None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai is not installed. Run `uv sync --python 3.11` first."
                ) from exc
            client = OpenAI(base_url=base_url, api_key=api_key)

        self.client = client
        self.model = model

    def select_target(self, image_url: str, detections: list[PersonDetection]) -> str:
        """Ask the VLM to select one numbered box from an annotated full image."""

        box_lines = "\n".join(
            f"{index}: box={detection.box}, confidence={detection.confidence:.3f}"
            for index, detection in enumerate(detections, start=1)
        )
        prompt = f"{TARGET_SELECTION_PROMPT}\n\n候选框列表：\n{box_lines}"
        return self._chat(prompt=prompt, image_url=image_url)

    def classify_behavior(self, image_url: str) -> str:
        """Ask the VLM to classify behavior for a selected target crop."""

        return self._chat(prompt=BEHAVIOR_PROMPT, image_url=image_url)

    def _chat(self, *, prompt: str, image_url: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        return (response.choices[0].message.content or "").strip()


class SingleFramePipeline:
    """Run single-frame detection, optional VLM target selection, and behavior VLM."""

    def __init__(
        self,
        *,
        detector: DetectorLike,
        vlm: VlmLike,
        crop_scale: float = 1.5,
        frame_width: int = DEFAULT_BEHAVIOR_FRAME_WIDTH,
        frame_height: int = DEFAULT_BEHAVIOR_FRAME_HEIGHT,
        jpeg_quality: int = DEFAULT_BEHAVIOR_JPEG_QUALITY,
    ) -> None:
        if crop_scale not in (1.0, 1.5):
            raise ValueError("crop_scale must be 1.0 or 1.5")
        self.detector = detector
        self.vlm = vlm
        self.crop_scale = crop_scale
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality

    def process_image(self, image_path: str | Path) -> SingleFrameResult:
        """Read an image from disk and process it."""

        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            return SingleFrameResult(
                image_name=image_path.name,
                predicted_label=NORMAL_LABEL,
                error=f"无法读取图片: {image_path}",
            )
        return self.process_frame(frame, image_name=image_path.name)

    def process_frame(self, frame, *, image_name: str = "") -> SingleFrameResult:
        """Process one BGR frame."""

        detections = self.detector.detect(frame)
        boxes = [detection.box for detection in detections]
        if not detections:
            return SingleFrameResult(
                image_name=image_name,
                num_boxes=0,
                boxes=[],
                predicted_label="完全离席",
            )

        selection_raw = ""
        selection_fallback = False
        if len(detections) == 1:
            selected_box_id = 1
        else:
            annotated = draw_numbered_boxes(frame, detections)
            image_url = encode_frame_to_data_url(
                annotated,
                jpeg_quality=self.jpeg_quality,
                width=self.frame_width,
                height=self.frame_height,
            )
            selection_raw = self.vlm.select_target(image_url, detections)
            selected_box_id = parse_selection_box_id(selection_raw, len(detections))
            if selected_box_id is None:
                height, width = frame.shape[:2]
                selected_box_id = center_fallback_box_id(detections, (width, height))
                selection_fallback = True

        selected_detection = detections[selected_box_id - 1]
        crop = crop_detection(frame, selected_detection.box, self.crop_scale)
        crop_url = encode_frame_to_data_url(
            crop,
            jpeg_quality=self.jpeg_quality,
            width=self.frame_width,
            height=self.frame_height,
        )
        behavior_raw = self.vlm.classify_behavior(crop_url)
        return SingleFrameResult(
            image_name=image_name,
            num_boxes=len(detections),
            boxes=boxes,
            selected_box_id=selected_box_id,
            selected_box=selected_detection.box,
            selection_raw=selection_raw,
            selection_fallback=selection_fallback,
            behavior_raw=behavior_raw,
            predicted_label=normalize_behavior_label(behavior_raw),
        )


def normalize_behavior_label(text: str | None) -> str:
    """Normalize model or ground-truth text into a comparable behavior label."""

    raw = (text or "").strip()
    if not raw or raw == NORMAL_LABEL:
        return NORMAL_LABEL

    parts = re.split(r"\s*with\s*", raw, flags=re.IGNORECASE)
    if len(parts) >= 3:
        candidate = _strip_braces(parts[1])
        if candidate:
            return candidate

    stripped = _strip_braces(raw)
    if stripped in KNOWN_LABELS:
        return stripped

    compact = re.sub(r"\s+", "", raw)
    for label in BEHAVIOR_PRIORITY:
        if label in compact:
            return label
    return raw


def parse_selection_box_id(raw: str, num_boxes: int) -> int | None:
    """Parse the VLM's target selection JSON and return a 1-based box id."""

    try:
        parsed = json.loads(_extract_json_object(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    box_id = parsed.get("box_id") if isinstance(parsed, dict) else None
    if not isinstance(box_id, int):
        return None
    if box_id < 1 or box_id > num_boxes:
        return None
    return box_id


def center_fallback_box_id(
    detections: list[PersonDetection],
    image_size: tuple[int, int],
) -> int:
    """Return the 1-based id of the candidate closest to the image center."""

    width, height = image_size
    center_x = width / 2
    center_y = height / 2
    ranked = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        box_x = (x1 + x2) / 2
        box_y = (y1 + y2) / 2
        distance = math.hypot(box_x - center_x, box_y - center_y)
        ranked.append((distance, index))
    return min(ranked, key=lambda item: item[0])[1]


def crop_detection(frame, box: Box, crop_scale: float):
    """Crop a detection box after optional center expansion."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, crop_scale, (width, height))
    return frame[y1:y2, x1:x2].copy()


def draw_numbered_boxes(frame, detections: list[PersonDetection]):
    """Draw 1-based candidate ids on a copy of the image."""

    import cv2

    annotated = frame.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = str(index)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return annotated


def _strip_braces(text: str) -> str:
    return text.strip().strip("{}").strip()


def _extract_json_object(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text
