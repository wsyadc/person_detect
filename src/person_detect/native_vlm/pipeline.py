"""Native full-image VLM behavior evaluation pipeline."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from person_detect.behavior import (
    DEFAULT_BEHAVIOR_API_KEY,
    DEFAULT_BEHAVIOR_BASE_URL,
    DEFAULT_BEHAVIOR_FRAME_HEIGHT,
    DEFAULT_BEHAVIOR_FRAME_WIDTH,
    DEFAULT_BEHAVIOR_JPEG_QUALITY,
    DEFAULT_BEHAVIOR_MODEL,
)
from person_detect.native_vlm.prompts import BEHAVIOR_PROMPT

NORMAL_LABEL = "无异常"
LEGACY_NORMAL_LABEL = "无任何输出"
NATIVE_VLM_LABELS = (
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
    NORMAL_LABEL,
)
NATIVE_VLM_LABEL_SET = set(NATIVE_VLM_LABELS)


class NativeVlmLike(Protocol):
    """Minimal VLM interface used by the native full-image baseline."""

    def classify_behavior(self, image_url: str) -> str:
        """Return the behavior classification text."""


@dataclass(frozen=True)
class NativeVlmResult:
    """Prediction result for one full-image VLM evaluation sample."""

    image_name: str = ""
    predicted_label: str = NORMAL_LABEL
    raw_model_output: str = ""
    parse_ok: bool = False
    audit_images: dict[str, str] | None = None
    error: str = ""

    def to_record(self) -> dict[str, object]:
        """Convert the result into a JSON-serializable prediction record fragment."""

        return {
            "image_name": self.image_name,
            "predicted_label": self.predicted_label,
            "raw_model_output": self.raw_model_output,
            "parse_ok": self.parse_ok,
            "audit_images": self.audit_images or {},
            "error": self.error,
        }


@dataclass(frozen=True)
class NativeVlmAudit:
    """Optional per-sample image audit writer for native full-image evaluation."""

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
            raise RuntimeError(f"无法保存原生 VLM 审计图片: {path}")
        return path.relative_to(self.run_dir).as_posix()


class OpenAICompatibleNativeVLM:
    """OpenAI-compatible VLM client for native full-image behavior classification."""

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

    def classify_behavior(self, image_url: str) -> str:
        """Ask the VLM to classify behavior from the full input image."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BEHAVIOR_PROMPT},
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


class NativeVlmPipeline:
    """Run native full-image VLM behavior classification without detection."""

    def __init__(
        self,
        *,
        vlm: NativeVlmLike,
        frame_width: int = DEFAULT_BEHAVIOR_FRAME_WIDTH,
        frame_height: int = DEFAULT_BEHAVIOR_FRAME_HEIGHT,
        jpeg_quality: int = DEFAULT_BEHAVIOR_JPEG_QUALITY,
    ) -> None:
        self.vlm = vlm
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality

    def process_image(
        self,
        image_path: str | Path,
        *,
        sample_index: int = 0,
        audit: NativeVlmAudit | None = None,
    ) -> NativeVlmResult:
        """Read a full image from disk and process it."""

        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            return NativeVlmResult(
                image_name=image_path.name,
                predicted_label=NORMAL_LABEL,
                error=f"无法读取图片: {image_path}",
            )
        if audit is None:
            audit = NativeVlmAudit(
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
        audit: NativeVlmAudit | None = None,
    ) -> NativeVlmResult:
        """Process one BGR frame as a full-image VLM input."""

        resized, image_url = prepare_native_vlm_image(
            frame,
            jpeg_quality=self.jpeg_quality,
            width=self.frame_width,
            height=self.frame_height,
        )
        audit_images = {}
        if audit is not None:
            saved = audit.save_image("vlm_input_resized.jpg", resized)
            if saved:
                audit_images["vlm_input"] = saved

        raw_output = self.vlm.classify_behavior(image_url)
        predicted_label, parse_ok = parse_native_behavior_output(raw_output)
        return NativeVlmResult(
            image_name=image_name,
            predicted_label=predicted_label,
            raw_model_output=raw_output,
            parse_ok=parse_ok,
            audit_images=audit_images,
        )


def parse_native_behavior_output(raw: str | None) -> tuple[str, bool]:
    """Parse native VLM text output into a known label and parse status."""

    text = _strip_code_fence((raw or "").strip())
    if not text:
        return NORMAL_LABEL, True
    if text in {LEGACY_NORMAL_LABEL, NORMAL_LABEL}:
        return NORMAL_LABEL, True

    parts = re.split(r"\s*with\s*", text, flags=re.IGNORECASE)
    if len(parts) >= 3:
        candidate = _strip_braces(parts[1])
        if candidate in NATIVE_VLM_LABEL_SET:
            return candidate, True
        return NORMAL_LABEL, False

    stripped = _strip_braces(text)
    if stripped in NATIVE_VLM_LABEL_SET:
        return stripped, True

    compact = re.sub(r"\s+", "", text)
    for label in NATIVE_VLM_LABELS:
        if label in compact:
            return label, True
    return NORMAL_LABEL, False


def normalize_native_label(raw: str | None) -> str:
    """Normalize model output or ground truth into a comparable native label."""

    label, _parse_ok = parse_native_behavior_output(raw)
    return label


def prepare_native_vlm_image(
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
        raise RuntimeError("无法编码原生 VLM 输入图片")
    encoded = base64.b64encode(buffer).decode("ascii")
    return resized, f"data:image/jpeg;base64,{encoded}"


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_braces(text: str) -> str:
    return text.strip().strip("{}").strip()


def _safe_stem(image_name: str) -> str:
    stem = Path(image_name).stem or "image"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
