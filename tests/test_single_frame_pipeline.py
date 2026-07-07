import numpy as np

from person_detect.detector import PersonDetection
from person_detect.single_frame.pipeline import (
    SingleFramePipeline,
    center_fallback_box_id,
    crop_detection,
    normalize_behavior_label,
    parse_selection_box_id,
)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame):
        return [
            PersonDetection(box=box, confidence=0.9)
            for box in self.boxes
        ]


class FakeVLM:
    def __init__(self, *, selection="{}", behavior="") -> None:
        self.selection = selection
        self.behavior = behavior
        self.selection_calls = 0
        self.behavior_calls = 0

    def select_target(self, image_url, detections):
        self.selection_calls += 1
        return self.selection

    def classify_behavior(self, image_url):
        self.behavior_calls += 1
        return self.behavior


def test_normalize_behavior_label_handles_empty_with_format_and_known_names() -> None:
    assert normalize_behavior_label("") == "无任何输出"
    assert normalize_behavior_label("无任何输出") == "无任何输出"
    assert normalize_behavior_label("{课堂表现}with{完全离席}with{无人}") == "完全离席"
    assert normalize_behavior_label("{课堂表现} with {摆弄电子设备} with {手机}") == "摆弄电子设备"
    assert normalize_behavior_label("课堂表现with双手托腮with检测到托腮") == "双手托腮"
    assert normalize_behavior_label("双手托腮") == "双手托腮"
    assert normalize_behavior_label("not a behavior") == "not a behavior"


def test_parse_selection_box_id_accepts_valid_json_and_rejects_invalid_ids() -> None:
    assert parse_selection_box_id('{"box_id": 2, "reason": "child"}', 3) == 2
    assert parse_selection_box_id('```json\n{"box_id": 1}\n```', 3) == 1
    assert parse_selection_box_id('{"box_id": 4}', 3) is None
    assert parse_selection_box_id('not json', 3) is None


def test_center_fallback_selects_box_closest_to_image_center() -> None:
    detections = [
        PersonDetection(box=(0, 0, 20, 20), confidence=0.9),
        PersonDetection(box=(45, 35, 65, 55), confidence=0.9),
        PersonDetection(box=(80, 70, 99, 99), confidence=0.9),
    ]

    assert center_fallback_box_id(detections, image_size=(100, 100)) == 2


def test_crop_detection_respects_scale_and_image_bounds() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    scale_1 = crop_detection(frame, (20, 20, 60, 60), 1.0)
    scale_15 = crop_detection(frame, (20, 20, 60, 60), 1.5)
    clipped = crop_detection(frame, (0, 0, 20, 20), 1.5)

    assert scale_1.shape[:2] == (40, 40)
    assert scale_15.shape[:2] == (60, 60)
    assert clipped.shape[:2] == (25, 25)


def test_pipeline_no_box_returns_absent_without_calling_vlm() -> None:
    pipeline = SingleFramePipeline(detector=FakeDetector([]), vlm=FakeVLM())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)

    assert result.predicted_label == "完全离席"
    assert result.num_boxes == 0
    assert result.selected_box_id is None
    assert pipeline.vlm.behavior_calls == 0
    assert pipeline.vlm.selection_calls == 0


def test_pipeline_single_box_skips_selection_and_classifies_crop() -> None:
    vlm = FakeVLM(behavior="{课堂表现}with{遮挡面部}with{手遮挡眼部}")
    pipeline = SingleFramePipeline(
        detector=FakeDetector([(10, 10, 50, 60)]),
        vlm=vlm,
        crop_scale=1.0,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)

    assert result.predicted_label == "遮挡面部"
    assert result.selected_box_id == 1
    assert result.selected_box == (10, 10, 50, 60)
    assert vlm.selection_calls == 0
    assert vlm.behavior_calls == 1


def test_pipeline_multi_box_uses_selection_fallback_then_classifies() -> None:
    vlm = FakeVLM(selection="bad", behavior="")
    pipeline = SingleFramePipeline(
        detector=FakeDetector([(0, 0, 20, 20), (45, 35, 65, 55)]),
        vlm=vlm,
        crop_scale=1.5,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)

    assert result.predicted_label == "无任何输出"
    assert result.selected_box_id == 2
    assert result.selection_fallback is True
    assert vlm.selection_calls == 1
    assert vlm.behavior_calls == 1
