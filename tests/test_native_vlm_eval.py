import json
import threading
from pathlib import Path

import numpy as np

from person_detect.native_vlm.eval import NativeVlmEvaluationRunner, compute_native_summary
from person_detect.native_vlm.pipeline import NativeVlmResult


class FakePipeline:
    def __init__(self, labels, lock=None, calls=None) -> None:
        self.labels = labels
        self.lock = lock
        self.calls = calls if calls is not None else []

    def process_image(self, image_path, *, sample_index=0, audit=None):
        with self.lock or _NullLock():
            self.calls.append(Path(image_path))
        label = self.labels[sample_index]
        return NativeVlmResult(
            image_name=Path(image_path).name,
            predicted_label=label,
            raw_model_output=label,
            parse_ok=True,
            audit_images={},
            error="",
        )


class _NullLock:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_compute_native_summary_reports_per_label_accuracy() -> None:
    rows = [
        {"ground_truth": "无异常", "predicted_label": "无异常", "correct": True},
        {"ground_truth": "东张西望", "predicted_label": "东张西望", "correct": True},
        {"ground_truth": "东张西望", "predicted_label": "无异常", "correct": False},
    ]

    summary = compute_native_summary(rows)

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == 2 / 3
    assert summary["label_set"][-1] == "无异常"
    assert summary["per_label_accuracy"]["东张西望"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert summary["confusion_matrix"]["东张西望"]["无异常"] == 1


def test_native_eval_runner_writes_ordered_predictions_and_summary(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["one.jpg", "two.jpg", "three.jpg"]:
        import cv2

        cv2.imwrite(str(image_dir / name), np.zeros((10, 10, 3), dtype=np.uint8))

    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        "\n".join(
            json.dumps({"image_name": name, "ground_truth": gt}, ensure_ascii=False)
            for name, gt in [
                ("one.jpg", "无异常"),
                ("two.jpg", "课堂表现with双手托腮with托腮"),
                ("three.jpg", "打哈欠"),
            ]
        ),
        encoding="utf-8",
    )
    calls = []
    lock = threading.Lock()
    runner = NativeVlmEvaluationRunner(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        output_dir=tmp_path / "out",
        pipeline_factory=lambda: FakePipeline(
            ["无异常", "双手托腮", "无异常"],
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
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [row["sample_index"] for row in predictions] == [0, 1, 2]
    assert [row["ground_truth"] for row in predictions] == ["无异常", "双手托腮", "打哈欠"]
    assert [row["correct"] for row in predictions] == [True, True, False]
    assert summary["accuracy"] == 2 / 3
    assert len(calls) == 3
