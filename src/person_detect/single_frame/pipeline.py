"""Single-frame person selection and behavior classification pipeline."""

from __future__ import annotations

import base64
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
)
from person_detect.boxes import Box, expand_box
from person_detect.detector import PersonDetection
from person_detect.single_frame.prompts import BEHAVIOR_PROMPT, TARGET_SELECTION_PROMPT

NORMAL_LABEL = "无异常"
LEGACY_NORMAL_LABEL = "无任何输出"
ABSENT_LABEL = "完全离席"
BEHAVIOR_LABELS = (
    "趴桌懈怠",
    "摆弄玩具",
    "摆弄电子设备",
    "双手托腮",
    "揉眼睛",
    "打哈欠",
    NORMAL_LABEL,
)
TARGET_LABELS = (*BEHAVIOR_LABELS, ABSENT_LABEL)
BEHAVIOR_LABEL_SET = set(BEHAVIOR_LABELS)
TARGET_LABEL_SET = set(TARGET_LABELS)
DEFAULT_BEHAVIOR_RESULT: dict[str, Any] = {
    "evidence": [],
    "behavior_name": NORMAL_LABEL,
}


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
    selection_source: str = ""
    selection_fallback: bool = False
    behavior_raw: str = ""
    behavior_result: dict[str, Any] | None = None
    behavior_parse_ok: bool = False
    predicted_label: str = NORMAL_LABEL
    audit_images: dict[str, str] | None = None
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert the result into a JSON-serializable prediction record fragment."""

        return {
            "image_name": self.image_name,
            "predicted_label": self.predicted_label,
            "num_boxes": self.num_boxes,
            "boxes": [list(box) for box in (self.boxes or [])],
            "selected_box_id": self.selected_box_id,
            "selected_box": list(self.selected_box) if self.selected_box else None,
            "selection_raw": self.selection_raw,
            "selection_source": self.selection_source,
            "selection_fallback": self.selection_fallback,
            "behavior_raw": self.behavior_raw,
            "behavior_result": self.behavior_result or dict(DEFAULT_BEHAVIOR_RESULT),
            "behavior_parse_ok": self.behavior_parse_ok,
            "audit_images": self.audit_images or {},
            "error": self.error,
        }


@dataclass(frozen=True)
class SingleFrameAudit:
    """Optional per-sample image audit writer for the single-frame experiment."""

    run_dir: Path
    sample_index: int
    image_name: str
    enabled: bool = False
    jpeg_quality: int = DEFAULT_BEHAVIOR_JPEG_QUALITY

    def save_image(self, filename: str, frame) -> str:
        """Save an audit image and return its path relative to the run directory."""

        if not self.enabled:
            return ""

        import cv2

        sample_dir = (
            self.run_dir
            / "audit"
            / "samples"
            / f"{self.sample_index:05d}_{_safe_stem(self.image_name)}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / filename
        ok = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"无法保存单帧审计图片: {path}")
        return path.relative_to(self.run_dir).as_posix()


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

    def process_image(
        self,
        image_path: str | Path,
        *,
        sample_index: int = 0,
        audit: SingleFrameAudit | None = None,
    ) -> SingleFrameResult:
        """Read an image from disk and process it."""

        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            return SingleFrameResult(
                image_name=image_path.name,
                predicted_label=NORMAL_LABEL,
                behavior_result=dict(DEFAULT_BEHAVIOR_RESULT),
                error=f"无法读取图片: {image_path}",
            )
        if audit is None:
            audit = SingleFrameAudit(
                run_dir=Path("."),
                sample_index=sample_index,
                image_name=image_path.name,
                enabled=False,
                jpeg_quality=self.jpeg_quality,
            )
        return self.process_frame(frame, image_name=image_path.name, audit=audit)

    def process_frame(
        self,
        frame,
        *,
        image_name: str = "",
        audit: SingleFrameAudit | None = None,
    ) -> SingleFrameResult:
        """Process one BGR frame."""

        audit_images: dict[str, str] = {}
        if audit is not None:
            _maybe_store(audit_images, "detector_input", audit.save_image("detector_input.jpg", frame))

        detections = self.detector.detect(frame)
        boxes = [detection.box for detection in detections]
        if not detections:
            return SingleFrameResult(
                image_name=image_name,
                num_boxes=0,
                boxes=[],
                predicted_label=ABSENT_LABEL,
                behavior_result={"evidence": [], "behavior_name": ABSENT_LABEL},
                behavior_parse_ok=True,
                audit_images=audit_images,
            )

        selection_raw = ""
        selection_source = ""
        selection_fallback = False
        if len(detections) == 1:
            selected_box_id = 1
        else:
            annotated = draw_numbered_boxes(frame, detections)
            selection_frame, image_url = prepare_vlm_image(
                annotated,
                jpeg_quality=self.jpeg_quality,
                width=self.frame_width,
                height=self.frame_height,
            )
            if audit is not None:
                _maybe_store(
                    audit_images,
                    "selection_input",
                    audit.save_image("selection_input_resized.jpg", selection_frame),
                )
            selection_raw = self.vlm.select_target(image_url, detections)
            selected_box_id = parse_selection_box_id(selection_raw, len(detections))
            if selected_box_id is None:
                height, width = frame.shape[:2]
                selected_box_id = center_fallback_box_id(detections, (width, height))
                selection_source = "center_fallback"
                selection_fallback = True
            else:
                selection_source = "vlm"

            selection_result = draw_numbered_boxes(
                frame,
                detections,
                selected_box_id=selected_box_id,
                show_count=True,
            )
            if audit is not None:
                _maybe_store(
                    audit_images,
                    "selection_result",
                    audit.save_image("selection_result.jpg", selection_result),
                )

        selected_detection = detections[selected_box_id - 1]
        crop = crop_detection(frame, selected_detection.box, self.crop_scale)
        crop_frame, crop_url = prepare_vlm_image(
            crop,
            jpeg_quality=self.jpeg_quality,
            width=self.frame_width,
            height=self.frame_height,
        )
        if audit is not None:
            _maybe_store(
                audit_images,
                "behavior_crop",
                audit.save_image("behavior_crop_resized.jpg", crop_frame),
            )

        behavior_raw = self.vlm.classify_behavior(crop_url)
        behavior_result, behavior_parse_ok = parse_behavior_output(behavior_raw)
        return SingleFrameResult(
            image_name=image_name,
            num_boxes=len(detections),
            boxes=boxes,
            selected_box_id=selected_box_id,
            selected_box=selected_detection.box,
            selection_raw=selection_raw,
            selection_source=selection_source,
            selection_fallback=selection_fallback,
            behavior_raw=behavior_raw,
            behavior_result=behavior_result,
            behavior_parse_ok=behavior_parse_ok,
            predicted_label=behavior_result["behavior_name"],
            audit_images=audit_images,
        )


def parse_behavior_output(raw: str | None) -> tuple[dict[str, Any], bool]:
    """Parse behavior VLM JSON, returning a safe normal result on any mismatch."""

    try:
        parsed = json.loads(_extract_json_object(raw or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return dict(DEFAULT_BEHAVIOR_RESULT), False

    if not isinstance(parsed, dict):
        return dict(DEFAULT_BEHAVIOR_RESULT), False

    evidence = parsed.get("evidence")
    behavior_name = parsed.get("behavior_name")
    if not isinstance(evidence, list):
        return dict(DEFAULT_BEHAVIOR_RESULT), False
    if not all(isinstance(item, str) for item in evidence):
        return dict(DEFAULT_BEHAVIOR_RESULT), False
    if not isinstance(behavior_name, str) or behavior_name not in BEHAVIOR_LABEL_SET:
        return dict(DEFAULT_BEHAVIOR_RESULT), False
    return {"evidence": evidence, "behavior_name": behavior_name}, True


def normalize_behavior_label(text: str | None) -> str:
    """Normalize model or ground-truth text into a comparable behavior label."""

    raw = (text or "").strip()
    if not raw or raw in {NORMAL_LABEL, LEGACY_NORMAL_LABEL}:
        return NORMAL_LABEL

    behavior_result, parse_ok = parse_behavior_output(raw)
    if parse_ok:
        return behavior_result["behavior_name"]

    parts = re.split(r"\s*with\s*", raw, flags=re.IGNORECASE)
    if len(parts) >= 3:
        candidate = _strip_braces(parts[1])
        if candidate:
            return candidate

    stripped = _strip_braces(raw)
    if stripped in TARGET_LABEL_SET:
        return stripped

    compact = re.sub(r"\s+", "", raw)
    for label in TARGET_LABELS:
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


def prepare_vlm_image(
    frame,
    *,
    jpeg_quality: int,
    width: int,
    height: int,
):
    """Resize a BGR frame once and encode that exact image as a JPEG data URL."""

    import cv2

    resized = cv2.resize(frame, (width, height))
    ok, buffer = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        raise RuntimeError("无法编码 VLM 输入图片")
    encoded = base64.b64encode(buffer).decode("ascii")
    return resized, f"data:image/jpeg;base64,{encoded}"


def draw_numbered_boxes(
    frame,
    detections: list[PersonDetection],
    *,
    selected_box_id: int | None = None,
    show_count: bool = False,
):
    """Draw 1-based candidate ids on a copy of the image."""

    import cv2

    annotated = frame.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        is_selected = index == selected_box_id
        color = (0, 255, 0) if is_selected else (0, 220, 255)
        thickness = 4 if is_selected else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = str(index)
        if is_selected:
            label = f"{index} selected"
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
    if show_count:
        text = f"boxes={len(detections)} selected={selected_box_id}"
        cv2.putText(
            annotated,
            text,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _maybe_store(images: dict[str, str], key: str, path: str) -> None:
    if path:
        images[key] = path


def _safe_stem(image_name: str) -> str:
    stem = Path(image_name).stem or "image"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


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
