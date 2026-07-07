import numpy as np
import pytest

from person_detect.detector import PersonDetection
from person_detect.single_frame.pipeline import (
    BEHAVIOR_LABELS,
    SingleFrameAudit,
    SingleFramePipeline,
    center_fallback_box_id,
    crop_detection,
    normalize_behavior_label,
    parse_behavior_output,
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


def test_normalize_behavior_label_handles_new_normal_and_legacy_formats() -> None:
    assert normalize_behavior_label("") == "无异常"
    assert normalize_behavior_label("无异常") == "无异常"
    assert normalize_behavior_label("无任何输出") == "无异常"
    assert normalize_behavior_label("{课堂表现}with{完全离席}with{无人}") == "完全离席"
    assert normalize_behavior_label("{课堂表现} with {摆弄电子设备} with {手机}") == "摆弄电子设备"
    assert normalize_behavior_label("课堂表现with双手托腮with检测到托腮") == "双手托腮"
    assert normalize_behavior_label("双手托腮") == "双手托腮"
    assert normalize_behavior_label("not a behavior") == "not a behavior"


@pytest.mark.parametrize(
    ("raw", "expected", "parse_ok"),
    [
        (
            '{"evidence":["手部接触眼部"],"behavior_name":"揉眼睛"}',
            {"evidence": ["手部接触眼部"], "behavior_name": "揉眼睛"},
            True,
        ),
        (
            '```json\n{"evidence":["嘴巴张大"],"behavior_name":"打哈欠"}\n```',
            {"evidence": ["嘴巴张大"], "behavior_name": "打哈欠"},
            True,
        ),
        ("", {"evidence": [], "behavior_name": "无异常"}, False),
        ("无异常", {"evidence": [], "behavior_name": "无异常"}, False),
        ('{"behavior_name":"双手托腮"}', {"evidence": [], "behavior_name": "无异常"}, False),
        ('{"evidence":"托腮","behavior_name":"双手托腮"}', {"evidence": [], "behavior_name": "无异常"}, False),
        ('{"evidence":[],"behavior_name":"遮挡面部"}', {"evidence": [], "behavior_name": "无异常"}, False),
    ],
)
def test_parse_behavior_output_returns_json_or_safe_normal(raw, expected, parse_ok) -> None:
    assert parse_behavior_output(raw) == (expected, parse_ok)


def test_behavior_label_set_is_limited_to_json_prompt_labels() -> None:
    assert BEHAVIOR_LABELS == (
        "趴桌懈怠",
        "摆弄玩具",
        "摆弄电子设备",
        "双手托腮",
        "揉眼睛",
        "打哈欠",
        "无异常",
    )


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


def test_pipeline_single_box_skips_selection_and_classifies_resized_crop(tmp_path) -> None:
    vlm = FakeVLM(behavior='{"evidence":["单手托腮"],"behavior_name":"双手托腮"}')
    pipeline = SingleFramePipeline(
        detector=FakeDetector([(10, 10, 50, 60)]),
        vlm=vlm,
        crop_scale=1.0,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    audit = SingleFrameAudit(
        run_dir=tmp_path,
        sample_index=0,
        image_name="single.jpg",
        enabled=True,
    )

    result = pipeline.process_frame(frame, audit=audit)

    assert result.predicted_label == "双手托腮"
    assert result.selected_box_id == 1
    assert result.selected_box == (10, 10, 50, 60)
    assert result.behavior_result == {"evidence": ["单手托腮"], "behavior_name": "双手托腮"}
    assert result.behavior_parse_ok is True
    assert result.audit_images["detector_input"].endswith("detector_input.jpg")
    assert result.audit_images["behavior_crop"].endswith("behavior_crop_resized.jpg")
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

    assert result.predicted_label == "无异常"
    assert result.selected_box_id == 2
    assert result.selection_fallback is True
    assert result.selection_source == "center_fallback"
    assert vlm.selection_calls == 1
    assert vlm.behavior_calls == 1


def test_pipeline_multi_box_saves_selection_images_when_audit_enabled(tmp_path) -> None:
    import cv2

    vlm = FakeVLM(
        selection='{"box_id":1,"reason":"更像孩子"}',
        behavior='{"evidence":[],"behavior_name":"无异常"}',
    )
    pipeline = SingleFramePipeline(
        detector=FakeDetector([(0, 0, 20, 20), (45, 35, 65, 55)]),
        vlm=vlm,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    audit = SingleFrameAudit(
        run_dir=tmp_path,
        sample_index=3,
        image_name="multi.jpg",
        enabled=True,
    )

    result = pipeline.process_frame(frame, audit=audit)

    assert result.selection_source == "vlm"
    assert set(result.audit_images) == {
        "detector_input",
        "selection_input",
        "selection_result",
        "behavior_crop",
    }
    crop_path = tmp_path / result.audit_images["behavior_crop"]
    selection_input_path = tmp_path / result.audit_images["selection_input"]
    assert cv2.imread(str(crop_path)).shape[:2] == (48, 64)
    assert cv2.imread(str(selection_input_path)).shape[:2] == (48, 64)


def test_pipeline_does_not_save_audit_images_when_disabled(tmp_path) -> None:
    pipeline = SingleFramePipeline(detector=FakeDetector([]), vlm=FakeVLM())
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    audit = SingleFrameAudit(
        run_dir=tmp_path,
        sample_index=1,
        image_name="none.jpg",
        enabled=False,
    )

    result = pipeline.process_frame(frame, audit=audit)

    assert result.audit_images == {}
    assert list(tmp_path.iterdir()) == []
