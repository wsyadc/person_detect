import numpy as np

from person_detect.native_vlm.pipeline import (
    NATIVE_VLM_LABELS,
    NativeVlmPipeline,
    NativeVlmAudit,
    normalize_native_label,
    parse_native_behavior_output,
)


class FakeVLM:
    def __init__(self, output="") -> None:
        self.output = output
        self.calls: list[str] = []

    def classify_behavior(self, image_url):
        self.calls.append(image_url)
        return self.output


def test_native_label_set_contains_full_prompt_labels() -> None:
    assert NATIVE_VLM_LABELS == (
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
        "无异常",
    )


def test_parse_native_behavior_output_handles_empty_with_and_unknown() -> None:
    assert parse_native_behavior_output("") == ("无异常", True)
    assert parse_native_behavior_output("无任何输出") == ("无异常", True)
    assert parse_native_behavior_output("无异常") == ("无异常", True)
    assert parse_native_behavior_output("{课堂表现}with{东张西望}with{朝左}") == ("东张西望", True)
    assert parse_native_behavior_output("{课堂表现} with {摆弄电子设备} with {手机}") == ("摆弄电子设备", True)
    assert parse_native_behavior_output("```text\n{健康状态}with{打哈欠}with{嘴张大}\n```") == ("打哈欠", True)
    assert parse_native_behavior_output("not a behavior") == ("无异常", False)


def test_normalize_native_label_supports_ground_truth_formats() -> None:
    assert normalize_native_label("无任何输出") == "无异常"
    assert normalize_native_label("课堂表现with双手托腮with检测到托腮") == "双手托腮"
    assert normalize_native_label("完全离席") == "完全离席"


def test_native_pipeline_sends_resized_full_image_and_no_box_fields(tmp_path) -> None:
    vlm = FakeVLM("{课堂表现}with{仰头}with{面部朝上}")
    pipeline = NativeVlmPipeline(
        vlm=vlm,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame, image_name="full.jpg")

    assert result.predicted_label == "仰头"
    assert result.raw_model_output == "{课堂表现}with{仰头}with{面部朝上}"
    assert result.parse_ok is True
    assert result.audit_images == {}
    assert vlm.calls[0].startswith("data:image/jpeg;base64,")
    record = result.to_record()
    assert "boxes" not in record
    assert "selected_box" not in record


def test_native_pipeline_audit_saves_resized_full_image(tmp_path) -> None:
    import cv2

    vlm = FakeVLM("")
    pipeline = NativeVlmPipeline(
        vlm=vlm,
        frame_width=64,
        frame_height=48,
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    audit = NativeVlmAudit(
        run_dir=tmp_path,
        sample_index=2,
        image_name="full.jpg",
        enabled=True,
    )

    result = pipeline.process_frame(frame, image_name="full.jpg", audit=audit)

    saved_path = tmp_path / result.audit_images["vlm_input"]
    assert saved_path.name == "vlm_input_resized.jpg"
    assert cv2.imread(str(saved_path)).shape[:2] == (48, 64)
