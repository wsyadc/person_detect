"""CLI entrypoint for the native full-image VLM baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from person_detect.behavior import (
    DEFAULT_BEHAVIOR_API_KEY,
    DEFAULT_BEHAVIOR_BASE_URL,
    DEFAULT_BEHAVIOR_FRAME_HEIGHT,
    DEFAULT_BEHAVIOR_FRAME_WIDTH,
    DEFAULT_BEHAVIOR_JPEG_QUALITY,
    DEFAULT_BEHAVIOR_MODEL,
)
from person_detect.native_vlm.eval import NativeVlmEvaluationRunner
from person_detect.native_vlm.pipeline import OpenAICompatibleNativeVLM, NativeVlmPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the native full-image VLM baseline."""

    parser = argparse.ArgumentParser(
        description="Evaluate native full-image VLM behavior classification.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("test_dataset/test.jsonl"),
        help="Path to evaluation JSONL annotations.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("test"),
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/native_vlm"),
        help="Directory where timestamped evaluation outputs are written.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BEHAVIOR_BASE_URL,
        help="OpenAI-compatible VLM base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_BEHAVIOR_API_KEY,
        help="OpenAI-compatible VLM API key.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BEHAVIOR_MODEL,
        help="OpenAI-compatible VLM model name.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_BEHAVIOR_FRAME_WIDTH,
        help="Width used when resizing images sent to the VLM.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_BEHAVIOR_FRAME_HEIGHT,
        help="Height used when resizing images sent to the VLM.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_BEHAVIOR_JPEG_QUALITY,
        help="JPEG quality for VLM image data URLs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of sample-level evaluation workers.",
    )
    parser.add_argument(
        "--save-audit-images",
        action="store_true",
        help="Save per-sample resized VLM input images for review.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the native VLM baseline from command-line arguments."""

    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    def make_pipeline() -> NativeVlmPipeline:
        vlm = OpenAICompatibleNativeVLM(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        return NativeVlmPipeline(
            vlm=vlm,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            jpeg_quality=args.jpeg_quality,
        )

    runner = NativeVlmEvaluationRunner(
        jsonl_path=args.jsonl,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pipeline_factory=make_pipeline,
        workers=args.workers,
        save_audit_images=args.save_audit_images,
    )
    run_dir = runner.run()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Output: {run_dir}")
    print(
        f"Accuracy: {summary['accuracy']:.4f} "
        f"({summary['correct']}/{summary['total']})"
    )
    print("Per-label accuracy:")
    for label in summary["label_set"]:
        item = summary["per_label_accuracy"][label]
        print(
            f"  {label}: {item['accuracy']:.4f} "
            f"({item['correct']}/{item['total']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
