"""Geometry helpers for person boxes.

All boxes use image pixel coordinates in ``(x1, y1, x2, y2)`` format. The
origin is the top-left corner, and ``x2``/``y2`` are the lower-right edge.
"""

from __future__ import annotations

from typing import Iterable, TypeAlias

Box: TypeAlias = tuple[int, int, int, int]
ImageSize: TypeAlias = tuple[int, int]
DEFAULT_BOX_SCALES = (1.0, 1.5)


def clamp_box(box: Box, image_size: ImageSize) -> Box:
    """Clip a box so every coordinate stays inside the image bounds."""

    width, height = image_size
    x1, y1, x2, y2 = box
    clipped_x1 = max(0, min(width, x1))
    clipped_y1 = max(0, min(height, y1))
    clipped_x2 = max(0, min(width, x2))
    clipped_y2 = max(0, min(height, y2))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def expand_box(box: Box, scale: float, image_size: ImageSize) -> Box:
    """Expand ``box`` around its center and clip the result to the image."""

    if scale <= 0:
        raise ValueError("scale must be positive")

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    scaled_width = (x2 - x1) * scale
    scaled_height = (y2 - y1) * scale

    expanded = (
        round(center_x - scaled_width / 2),
        round(center_y - scaled_height / 2),
        round(center_x + scaled_width / 2),
        round(center_y + scaled_height / 2),
    )
    return clamp_box(expanded, image_size)


def scale_boxes(
    box: Box,
    image_size: ImageSize,
    scales: Iterable[float] = DEFAULT_BOX_SCALES,
) -> dict[float, Box]:
    """Return expanded boxes for each requested scale."""

    return {scale: expand_box(box, scale, image_size) for scale in scales}


def format_box(box: Box) -> str:
    """Format a box as compact terminal-friendly pixel coordinates."""

    x1, y1, x2, y2 = box
    return f"({x1},{y1},{x2},{y2})"


def format_scaled_boxes(box: Box, image_size: ImageSize) -> str:
    """Format the required 1x and 1.5x boxes for terminal output."""

    scaled = scale_boxes(box, image_size)
    return (
        f"[BOX] base={format_box(scaled[1.0])} "
        f"scale1.5={format_box(scaled[1.5])}"
    )


def iou(a: Box, b: Box) -> float:
    """Compute intersection-over-union for two boxes."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
