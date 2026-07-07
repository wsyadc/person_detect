"""YOLO person detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from person_detect.boxes import Box


@dataclass(frozen=True)
class PersonDetection:
    """A single detected person body box."""

    box: Box
    confidence: float


class PersonDetector:
    """Detect COCO ``person`` objects with YOLO on CPU."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        *,
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `uv sync --python 3.11` first."
            ) from exc

        self.model_name = model_name
        self.confidence = confidence
        self.image_size = image_size
        self._model = YOLO(model_name)

    def detect(self, frame) -> list[PersonDetection]:
        """Return person detections for a BGR OpenCV frame."""

        results = self._model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            device="cpu",
            verbose=False,
        )

        detections: list[PersonDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for coords, score, class_id in zip(xyxy, confidences, classes, strict=False):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = (round(float(value)) for value in coords)
                detections.append(
                    PersonDetection(
                        box=(x1, y1, x2, y2),
                        confidence=float(score),
                    )
                )
        return detections
