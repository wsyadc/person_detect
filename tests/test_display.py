from person_detect.display import BOX_DRAW_STYLES


def test_display_draw_styles_only_include_base_and_one_point_five_scales() -> None:
    assert [scale for scale, _color, _label in BOX_DRAW_STYLES] == [1.5, 1.0]
    assert all(label != "2x" for _scale, _color, label in BOX_DRAW_STYLES)
