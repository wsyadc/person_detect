import json
from pathlib import Path

import numpy as np

from person_detect.single_frame.eval import (
    EvaluationRunner,
    compute_summary,
)
from person_detect.single_frame.pipeline import SingleFrameResult


class FakePipeline:
    def __init__(self, labels):
        self.labels = labels
        self.calls: list[Path] = []

    def process_image(self, image_path):
        self.calls.append(Path(image_path))
        label = self.labels[len(self.calls) - 1]
        return SingleFrameResult(
            image_name=Path(image_path).name,
            num_boxes=1,
            boxes=[(1, 2, 3, 4)],
            selected_box_id=1,
            selected_box=(1, 2, 3, 4),
            selection_raw="",
            selection_fallback=False,
            behavior_raw=label,
            predicted_label=label,
            error="",
        )


def test_compute_summary_reports_accuracy_per_label_and_confusion_matrix() -> None:
    rows = [
        {"ground_truth_label": "无任何输出", "predicted_label": "无任何输出", "correct": True, "num_boxes": 0, "selection_fallback": False},
        {"ground_truth_label": "双手托腮", "predicted_label": "双手托腮", "correct": True, "num_boxes": 1, "selection_fallback": False},
        {"ground_truth_label": "双手托腮", "predicted_label": "无任何输出", "correct": False, "num_boxes": 2, "selection_fallback": True},
    ]

    summary = compute_summary(rows)

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == 2 / 3
    assert summary["no_box_count"] == 1
    assert summary["single_box_count"] == 1
    assert summary["multi_box_count"] == 1
    assert summary["selection_fallback_count"] == 1
    assert summary["per_label"]["双手托腮"]["support"] == 2
    assert summary["per_label"]["双手托腮"]["recall"] == 0.5
    assert summary["confusion_matrix"]["双手托腮"]["无任何输出"] == 1


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
    runner = EvaluationRunner(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        output_dir=output_dir,
        pipeline=FakePipeline(["无任何输出", "双手托腮"]),
    )

    run_dir = runner.run()

    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert [row["correct"] for row in predictions] == [True, True]
    assert predictions[1]["ground_truth_label"] == "双手托腮"
    assert summary["accuracy"] == 1.0
    assert len(list(run_dir.iterdir())) == 2
