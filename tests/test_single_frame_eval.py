import json
import threading
from pathlib import Path

import numpy as np

from person_detect.single_frame.eval import (
    EvaluationRunner,
    build_filtered_jsonl,
    compute_summary,
)
from person_detect.single_frame.pipeline import SingleFrameResult


class FakePipeline:
    def __init__(self, labels, lock=None, calls=None):
        self.labels = labels
        self.lock = lock
        self.calls = calls if calls is not None else []

    def process_image(self, image_path, *, sample_index=0, audit=None):
        with self.lock or _NullLock():
            self.calls.append(Path(image_path))
        label = self.labels[sample_index]
        return SingleFrameResult(
            image_name=Path(image_path).name,
            num_boxes=1,
            boxes=[(1, 2, 3, 4)],
            selected_box_id=1,
            selected_box=(1, 2, 3, 4),
            selection_raw="",
            selection_source="",
            selection_fallback=False,
            behavior_raw=label,
            behavior_result={"evidence": [], "behavior_name": label},
            behavior_parse_ok=True,
            predicted_label=label,
            audit_images={},
            error="",
        )


class _NullLock:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_compute_summary_reports_accuracy_per_label_and_confusion_matrix() -> None:
    rows = [
        {"ground_truth": "无异常", "predicted_label": "无异常", "correct": True, "num_boxes": 0, "selection_fallback": False},
        {"ground_truth_label": "双手托腮", "predicted_label": "双手托腮", "correct": True, "num_boxes": 1, "selection_fallback": False},
        {"ground_truth": "双手托腮", "predicted_label": "无异常", "correct": False, "num_boxes": 2, "selection_fallback": True},
        {"ground_truth": "揉眼睛", "predicted_label": "打哈欠", "correct": False, "num_boxes": 1, "selection_fallback": False},
    ]

    summary = compute_summary(rows)

    assert summary["total"] == 4
    assert summary["correct"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["label_set"] == [
        "趴桌懈怠",
        "摆弄玩具",
        "摆弄电子设备",
        "双手托腮",
        "揉眼睛",
        "打哈欠",
        "无异常",
        "完全离席",
    ]
    assert summary["no_box_count"] == 1
    assert summary["single_box_count"] == 2
    assert summary["multi_box_count"] == 1
    assert summary["selection_fallback_count"] == 1
    assert summary["per_label"]["双手托腮"]["support"] == 2
    assert summary["per_label"]["双手托腮"]["recall"] == 0.5
    assert summary["per_label_accuracy"]["双手托腮"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert summary["confusion_matrix"]["双手托腮"]["无异常"] == 1


def test_build_filtered_jsonl_keeps_only_target_ground_truth_labels(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "filtered.jsonl"
    rows = [
        {"id": 1, "image_name": "a.jpg", "model_answer": "x", "ground_truth": "无异常"},
        {"id": 2, "image_name": "b.jpg", "model_answer": "x", "ground_truth": "东张西望"},
        {"id": 3, "image_name": "c.jpg", "model_answer": "x", "ground_truth": "课堂表现with双手托腮with检测到托腮"},
        {"id": 4, "image_name": "d.jpg", "model_answer": "x", "ground_truth": "完全离席"},
        {"id": 5, "image_name": "e.jpg", "model_answer": "x", "ground_truth": "遮挡面部"},
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    stats = build_filtered_jsonl(source, target)

    filtered = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    assert stats == {"input": 5, "kept": 3, "filtered_out": 2}
    assert filtered == [
        {"image_name": "a.jpg", "ground_truth": "无异常"},
        {"image_name": "c.jpg", "ground_truth": "双手托腮"},
        {"image_name": "d.jpg", "ground_truth": "完全离席"},
    ]


def test_evaluation_runner_writes_predictions_and_summary(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["one.jpg", "two.jpg"]:
        import cv2

        cv2.imwrite(str(image_dir / name), np.zeros((10, 10, 3), dtype=np.uint8))

    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "image_name": "one.jpg", "ground_truth": "无任何输出"}, ensure_ascii=False),
                json.dumps({"id": 2, "image_name": "two.jpg", "ground_truth": "{课堂表现}with{双手托腮}with{托腮}"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = []
    runner = EvaluationRunner(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        output_dir=output_dir,
        pipeline_factory=lambda: FakePipeline(["无异常", "双手托腮"], calls=calls),
        workers=2,
        save_audit_images=False,
    )

    run_dir = runner.run()

    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert [row["correct"] for row in predictions] == [True, True]
    assert [row["sample_index"] for row in predictions] == [0, 1]
    assert predictions[0]["ground_truth"] == "无异常"
    assert predictions[0]["audit_images"] == {}
    assert summary["accuracy"] == 1.0
    assert len(list(run_dir.iterdir())) == 2


def test_evaluation_runner_uses_parallel_factory_and_preserves_output_order(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["one.jpg", "two.jpg", "three.jpg"]:
        import cv2

        cv2.imwrite(str(image_dir / name), np.zeros((10, 10, 3), dtype=np.uint8))

    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        "\n".join(
            json.dumps(
                {"image_name": name, "ground_truth": label},
                ensure_ascii=False,
            )
            for name, label in [
                ("one.jpg", "无异常"),
                ("two.jpg", "双手托腮"),
                ("three.jpg", "打哈欠"),
            ]
        ),
        encoding="utf-8",
    )
    calls = []
    lock = threading.Lock()

    runner = EvaluationRunner(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        output_dir=tmp_path / "out",
        pipeline_factory=lambda: FakePipeline(
            ["无异常", "双手托腮", "打哈欠"],
            lock=lock,
            calls=calls,
        ),
        workers=4,
        save_audit_images=False,
    )

    run_dir = runner.run()

    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_index"] for row in predictions] == [0, 1, 2]
    assert [row["image_name"] for row in predictions] == ["one.jpg", "two.jpg", "three.jpg"]
    assert len(calls) == 3
