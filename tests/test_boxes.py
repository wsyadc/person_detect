from person_detect.boxes import (
    DEFAULT_BOX_SCALES,
    expand_box,
    format_box,
    format_scaled_boxes,
    iou,
    scale_boxes,
)


def test_expand_box_scales_from_center_and_clips_to_image_bounds() -> None:
    assert expand_box((40, 50, 80, 110), 1.5, (120, 160)) == (30, 35, 90, 125)
    assert expand_box((0, 0, 20, 20), 2.0, (30, 30)) == (0, 0, 30, 30)


def test_scale_boxes_returns_required_box_sizes() -> None:
    scaled = scale_boxes((40, 50, 80, 110), (120, 160))

    assert DEFAULT_BOX_SCALES == (1.0, 1.5)
    assert scaled[1.0] == (40, 50, 80, 110)
    assert scaled[1.5] == (30, 35, 90, 125)
    assert 2.0 not in scaled


def test_iou_handles_overlap_no_overlap_and_containment() -> None:
    assert iou((10, 10, 30, 30), (20, 20, 40, 40)) == 100 / 700
    assert iou((10, 10, 30, 30), (40, 40, 60, 60)) == 0.0
    assert iou((10, 10, 50, 50), (20, 20, 30, 30)) == 100 / 1600


def test_box_terminal_format_matches_runtime_contract() -> None:
    assert format_box((1, 2, 3, 4)) == "(1,2,3,4)"
    assert (
        format_scaled_boxes((40, 50, 80, 110), (120, 160))
        == "[BOX] base=(40,50,80,110) scale1.5=(30,35,90,125)"
    )
