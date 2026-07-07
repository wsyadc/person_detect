"""Batch evaluation for the single-frame VLM selection experiment."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from person_detect.single_frame.pipeline import (
    TARGET_LABELS,
    TARGET_LABEL_SET,
    SingleFrameAudit,
    SingleFramePipeline,
    normalize_behavior_label,
)


@dataclass
class EvaluationRunner:
    """Run a single-frame pipeline over a JSONL dataset and write metrics."""

    jsonl_path: Path
    image_dir: Path
    output_dir: Path
    pipeline_factory: Callable[[], SingleFramePipeline] | None = None
    pipeline: SingleFramePipeline | None = None
    workers: int = 4
    save_audit_images: bool = False

    def __post_init__(self) -> None:
        if self.pipeline_factory is None and self.pipeline is None:
            raise ValueError("pipeline_factory or pipeline is required")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        self._thread_local = threading.local()

    def run(self) -> Path:
        """Run evaluation and return the timestamped run directory."""

        run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        samples = _read_jsonl(self.jsonl_path)
        rows = self._evaluate_samples(samples, run_dir)
        rows.sort(key=lambda row: row["sample_index"])

        predictions_path = run_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as output:
            for record in rows:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = compute_summary(rows)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_dir

    def _evaluate_samples(
        self,
        samples: list[dict[str, Any]],
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        if self.workers == 1:
            return [
                self._evaluate_sample(index, sample, run_dir)
                for index, sample in enumerate(samples)
            ]

        rows = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self._evaluate_sample, index, sample, run_dir)
                for index, sample in enumerate(samples)
            ]
            for future in as_completed(futures):
                rows.append(future.result())
        return rows

    def _pipeline_for_thread(self) -> SingleFramePipeline:
        if self.pipeline_factory is None:
            if self.pipeline is None:
                raise ValueError("pipeline is not configured")
            return self.pipeline

        pipeline = getattr(self._thread_local, "pipeline", None)
        if pipeline is None:
            pipeline = self.pipeline_factory()
            self._thread_local.pipeline = pipeline
        return pipeline

    def _evaluate_sample(
        self,
        sample_index: int,
        sample: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        image_name = sample["image_name"]
        ground_truth = normalize_behavior_label(sample.get("ground_truth", ""))
        image_path = self.image_dir / image_name
        audit = SingleFrameAudit(
            run_dir=run_dir,
            sample_index=sample_index,
            image_name=image_name,
            enabled=self.save_audit_images,
        )

        try:
            result = self._pipeline_for_thread().process_image(
                image_path,
                sample_index=sample_index,
                audit=audit,
            )
            result_record = result.to_record()
        except Exception as exc:
            result_record = {
                "image_name": image_name,
                "predicted_label": f"ERROR:{type(exc).__name__}",
                "num_boxes": 0,
                "boxes": [],
                "selected_box_id": None,
                "selected_box": None,
                "selection_raw": "",
                "selection_source": "",
                "selection_fallback": False,
                "behavior_raw": "",
                "behavior_result": {"evidence": [], "behavior_name": "无异常"},
                "behavior_parse_ok": False,
                "audit_images": {},
                "error": str(exc),
            }

        predicted_label = result_record["predicted_label"]
        return {
            "sample_index": sample_index,
            "image_name": image_name,
            "ground_truth": ground_truth,
            **result_record,
            "correct": predicted_label == ground_truth,
        }


def build_filtered_jsonl(source_path: Path, target_path: Path) -> dict[str, int]:
    """Write a filtered JSONL containing only target labels and required fields."""

    rows = _read_jsonl(source_path)
    kept = []
    for row in rows:
        label = normalize_behavior_label(row.get("ground_truth", ""))
        if label not in TARGET_LABEL_SET:
            continue
        kept.append(
            {
                "image_name": row["image_name"],
                "ground_truth": label,
            }
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as output:
        for row in kept:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input": len(rows),
        "kept": len(kept),
        "filtered_out": len(rows) - len(kept),
    }


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-match and per-label metrics for prediction rows."""

    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    fixed_labels = list(TARGET_LABELS)
    observed_labels = {
        _ground_truth_label(row)
        for row in rows
    } | {
        row["predicted_label"]
        for row in rows
    }
    labels = fixed_labels + sorted(observed_labels - set(fixed_labels))
    confusion = {
        label: {predicted: 0 for predicted in labels}
        for label in labels
    }
    for row in rows:
        confusion[_ground_truth_label(row)][row["predicted_label"]] += 1

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

    per_label_accuracy = {}
    for label in fixed_labels:
        label_rows = [row for row in rows if _ground_truth_label(row) == label]
        label_correct = sum(1 for row in label_rows if row["predicted_label"] == label)
        label_total = len(label_rows)
        per_label_accuracy[label] = {
            "correct": label_correct,
            "total": label_total,
            "accuracy": label_correct / label_total if label_total else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_label_accuracy": per_label_accuracy,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "label_set": fixed_labels,
        "no_box_count": sum(1 for row in rows if row.get("num_boxes") == 0),
        "single_box_count": sum(1 for row in rows if row.get("num_boxes") == 1),
        "multi_box_count": sum(1 for row in rows if row.get("num_boxes", 0) >= 2),
        "selection_fallback_count": sum(
            1 for row in rows if row.get("selection_fallback")
        ),
    }


def _ground_truth_label(row: dict[str, Any]) -> str:
    return normalize_behavior_label(
        row.get("ground_truth", row.get("ground_truth_label", ""))
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
