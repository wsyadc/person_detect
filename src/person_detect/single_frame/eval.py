"""Batch evaluation for the single-frame VLM selection experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from person_detect.single_frame.pipeline import (
    SingleFramePipeline,
    normalize_behavior_label,
)


@dataclass
class EvaluationRunner:
    """Run a single-frame pipeline over a JSONL dataset and write metrics."""

    jsonl_path: Path
    image_dir: Path
    output_dir: Path
    pipeline: SingleFramePipeline

    def run(self) -> Path:
        """Run evaluation and return the timestamped run directory."""

        run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        predictions_path = run_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as output:
            for sample in _read_jsonl(self.jsonl_path):
                record = self._evaluate_sample(sample)
                rows.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = compute_summary(rows)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_dir

    def _evaluate_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        image_name = sample["image_name"]
        ground_truth_raw = sample.get("ground_truth", "")
        ground_truth_label = normalize_behavior_label(ground_truth_raw)
        image_path = self.image_dir / image_name

        try:
            result = self.pipeline.process_image(image_path)
            result_record = result.to_record()
        except Exception as exc:
            result_record = {
                "image_name": image_name,
                "num_boxes": 0,
                "boxes": [],
                "selected_box_id": None,
                "selected_box": None,
                "selection_raw": "",
                "selection_fallback": False,
                "behavior_raw": "",
                "predicted_label": f"ERROR:{type(exc).__name__}",
                "error": str(exc),
            }

        predicted_label = result_record["predicted_label"]
        return {
            "id": sample.get("id"),
            "image_name": image_name,
            "ground_truth_raw": ground_truth_raw,
            "ground_truth_label": ground_truth_label,
            **result_record,
            "correct": predicted_label == ground_truth_label,
        }


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-match and per-label metrics for prediction rows."""

    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    labels = sorted(
        {
            row["ground_truth_label"]
            for row in rows
        }
        | {
            row["predicted_label"]
            for row in rows
        }
    )
    confusion = {
        label: {predicted: 0 for predicted in labels}
        for label in labels
    }
    for row in rows:
        confusion[row["ground_truth_label"]][row["predicted_label"]] += 1

    per_label = {}
    for label in labels:
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in labels)
        true_positive = confusion[label][label]
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "support": support,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "no_box_count": sum(1 for row in rows if row.get("num_boxes") == 0),
        "single_box_count": sum(1 for row in rows if row.get("num_boxes") == 1),
        "multi_box_count": sum(1 for row in rows if row.get("num_boxes", 0) >= 2),
        "selection_fallback_count": sum(
            1 for row in rows if row.get("selection_fallback")
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
