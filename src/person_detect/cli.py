"""Command-line entrypoint for live target-person detection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from person_detect.audit import AuditLogger
from person_detect.boxes import format_scaled_boxes
from person_detect.behavior import (
    DEFAULT_BEHAVIOR_API_KEY,
    DEFAULT_BEHAVIOR_BASE_URL,
    DEFAULT_BEHAVIOR_CROP_SCALE,
    DEFAULT_BEHAVIOR_FRAME_HEIGHT,
    DEFAULT_BEHAVIOR_FRAME_WIDTH,
    DEFAULT_BEHAVIOR_JPEG_QUALITY,
    DEFAULT_BEHAVIOR_MODEL,
    DEFAULT_BEHAVIOR_WINDOW_SIZE,
    BehaviorAnalyzer,
    BehaviorWindow,
)
from person_detect.display import WINDOW_NAME, draw_tracking_overlay
from person_detect.identity import DEFAULT_FACE_MODEL_MIRROR, DEFAULT_FACE_MODEL_REPO
from person_detect.session import SeatWindowCoordinator
from person_detect.tracking import TargetTracker, TrackingResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "log"


@dataclass
class FrameCadence:
    """Keep capture, processing, and display aligned to one frame interval."""

    interval_seconds: float
    next_frame_at: float = 0.0

    def wait(self, *, now_fn=time.monotonic, sleep_fn=time.sleep) -> float:
        """Sleep until the next frame slot and return that slot's timestamp."""

        now = now_fn()
        if now < self.next_frame_at:
            sleep_fn(self.next_frame_at - now)
            now = now_fn()
        self.next_frame_at = now + self.interval_seconds
        return now


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Detect a target person from a webcam using CPU inference.",
    )
    parser.add_argument("--face", required=True, type=Path, help="目标人脸照片路径")
    parser.add_argument("--camera", default=0, type=int, help="摄像头编号，默认 0")
    parser.add_argument("--interval-ms", default=500, type=int, help="处理间隔毫秒数")
    parser.add_argument("--lost-seconds", default=5.0, type=float, help="离席判定秒数")
    parser.add_argument(
        "--face-threshold",
        default=0.38,
        type=float,
        help="目标人脸 cosine 相似度阈值",
    )
    parser.add_argument(
        "--iou-threshold",
        default=0.25,
        type=float,
        help="短暂遮脸时的人体框 IoU 阈值",
    )
    parser.add_argument(
        "--edge-iou-margin-ratio",
        default=0.01,
        type=float,
        help="IoU-only 跟踪时忽略左右边缘贴边框的画面宽度比例，传 0 可关闭",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO 模型路径或名称")
    parser.add_argument(
        "--face-model",
        default="buffalo_s",
        help="InsightFace 模型名，默认 buffalo_s",
    )
    parser.add_argument(
        "--face-model-mirror",
        default=DEFAULT_FACE_MODEL_MIRROR,
        help="InsightFace 模型下载镜像，传空字符串可禁用镜像预下载",
    )
    parser.add_argument(
        "--face-model-repo",
        default=DEFAULT_FACE_MODEL_REPO,
        help="HuggingFace 镜像仓库，默认 vladmandic/insightface-faceanalysis",
    )
    parser.add_argument(
        "--behavior-enable",
        action="store_true",
        help="开启目标人物 crop 行为识别",
    )
    parser.add_argument(
        "--behavior-base-url",
        default=DEFAULT_BEHAVIOR_BASE_URL,
        help="OpenAI-compatible vLLM 服务地址",
    )
    parser.add_argument(
        "--behavior-api-key",
        default=DEFAULT_BEHAVIOR_API_KEY,
        help="OpenAI-compatible API key",
    )
    parser.add_argument(
        "--behavior-model",
        default=DEFAULT_BEHAVIOR_MODEL,
        help="行为识别模型名",
    )
    parser.add_argument(
        "--behavior-window-size",
        default=DEFAULT_BEHAVIOR_WINDOW_SIZE,
        type=int,
        help="行为、离席、回座共享窗口帧数",
    )
    parser.add_argument(
        "--behavior-crop-scale",
        default=DEFAULT_BEHAVIOR_CROP_SCALE,
        type=float,
        choices=(1.0, 1.5),
        help="行为识别 crop 框倍率",
    )
    parser.add_argument(
        "--behavior-jpeg-quality",
        default=DEFAULT_BEHAVIOR_JPEG_QUALITY,
        type=int,
        help="审计图片 JPEG 质量",
    )
    parser.add_argument(
        "--behavior-frame-width",
        default=DEFAULT_BEHAVIOR_FRAME_WIDTH,
        type=int,
        help="行为识别输入宽度",
    )
    parser.add_argument(
        "--behavior-frame-height",
        default=DEFAULT_BEHAVIOR_FRAME_HEIGHT,
        type=int,
        help="行为识别输入高度",
    )
    parser.add_argument(
        "--behavior-print-normal",
        action="store_true",
        help="兼容旧参数；行为 JSON 结果现在默认都会打印",
    )
    parser.add_argument(
        "--log-root",
        default=DEFAULT_LOG_ROOT,
        type=Path,
        help="审计日志根目录，默认项目 log 目录",
    )
    parser.add_argument("--no-window", action="store_true", help="不显示 OpenCV 窗口")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line program and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n已停止", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


def run(args: argparse.Namespace) -> None:
    """Initialize models and run the webcam processing loop."""

    import cv2
    from person_detect.detector import PersonDetector
    from person_detect.identity import TargetFaceMatcher

    detector = PersonDetector(args.model)
    matcher = TargetFaceMatcher(
        args.face,
        model_name=args.face_model,
        model_mirror=args.face_model_mirror or None,
        model_repo=args.face_model_repo,
    )
    tracker = TargetTracker(
        face_threshold=args.face_threshold,
        iou_threshold=args.iou_threshold,
        lost_seconds=args.lost_seconds,
        edge_iou_margin_ratio=args.edge_iou_margin_ratio,
    )

    capture = _open_camera(args.camera)
    interval_seconds = max(0.001, args.interval_ms / 1000)
    cadence = FrameCadence(interval_seconds=interval_seconds)
    audit_logger = AuditLogger(
        log_root=args.log_root,
        started_at=datetime.now().astimezone(),
    )
    behavior_window = _create_behavior_window(args)
    coordinator = SeatWindowCoordinator(
        window_size=args.behavior_window_size,
        audit_logger=audit_logger,
        behavior_window=behavior_window,
        crop_scale=args.behavior_crop_scale,
        jpeg_quality=args.behavior_jpeg_quality,
        frame_width=args.behavior_frame_width,
        frame_height=args.behavior_frame_height,
        on_event=lambda event: print(event, flush=True),
        on_behavior=lambda record: _print_behavior_record(record, args.behavior_print_normal),
        on_error=lambda error: print(error, file=sys.stderr, flush=True),
    )

    try:
        while True:
            now = cadence.wait()
            frame_timestamp = datetime.now().astimezone()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("无法从摄像头读取画面，请检查权限或摄像头占用。")

            result = _process_frame(frame, detector, matcher, tracker, now)
            _print_result(result, frame.shape)
            coordinator.update(frame, result, frame_timestamp, tracker)

            if not args.no_window:
                display_frame = draw_tracking_overlay(frame.copy(), result, tracker)
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        if behavior_window is not None:
            behavior_window.shutdown()
        capture.release()
        if not args.no_window:
            cv2.destroyAllWindows()


def _process_frame(
    frame,
    detector: PersonDetector,
    matcher: TargetFaceMatcher,
    tracker: TargetTracker,
    now: float,
) -> TrackingResult:
    detections = detector.detect(frame)
    candidates = matcher.annotate_candidates(frame, detections)
    height, width = frame.shape[:2]
    return tracker.update(candidates, now=now, image_size=(width, height))


def _print_result(result: TrackingResult, frame_shape: tuple[int, ...]) -> None:
    if result.event is not None:
        print(result.event, flush=True)
    if result.target_box is not None:
        height, width = frame_shape[:2]
        print(format_scaled_boxes(result.target_box, (width, height)), flush=True)


def _create_behavior_window(args: argparse.Namespace) -> BehaviorWindow | None:
    if not getattr(args, "behavior_enable", False):
        return None

    analyzer = BehaviorAnalyzer(
        base_url=args.behavior_base_url,
        api_key=args.behavior_api_key,
        model=args.behavior_model,
    )
    return BehaviorWindow(window_size=args.behavior_window_size, analyzer=analyzer)


def _print_behavior_record(record: dict, print_normal: bool) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "is_abnormal": record["is_abnormal"],
        "behavior_type": record["behavior_type"],
        "behavior_name": record["behavior_name"],
        "evidence": record["evidence"],
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    print(f"[行为 {timestamp}] {text}", flush=True)


def _open_camera(camera_index: int):
    import cv2

    if sys.platform == "darwin":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {camera_index}，请在系统设置中允许终端访问相机。"
        )
    return capture


if __name__ == "__main__":
    raise SystemExit(main())
