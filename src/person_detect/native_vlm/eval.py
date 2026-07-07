"""Batch evaluation for the native full-image VLM baseline."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from person_detect.native_vlm.pipeline import (
    NATIVE_VLM_LABELS,
    NativeVlmAudit,
    NativeVlmPipeline,
    normalize_native_label,
)


@dataclass
class NativeVlmEvaluationRunner:
    """Run native full-image VLM evaluation over a JSONL dataset."""

    jsonl_path: Path
    image_dir: Path
    output_dir: Path
    pipeline_factory: Callable[[], NativeVlmPipeline] | None = None
    pipeline: NativeVlmPipeline | None = None
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

        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
            for record in rows:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = compute_native_summary(rows)
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

    def _pipeline_for_thread(self) -> NativeVlmPipeline:
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
        ground_truth = normalize_native_label(sample.get("ground_truth", ""))
        image_path = self.image_dir / image_name
        audit = NativeVlmAudit(
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
                "predicted_label": "无异常",
                "raw_model_output": "",
                "parse_ok": False,
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


def compute_native_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-match and per-label metrics for native VLM predictions."""

    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    fixed_labels = list(NATIVE_VLM_LABELS)
    observed_labels = {
        row["ground_truth"]
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
        confusion[row["ground_truth"]][row["predicted_label"]] += 1

    per_label = {}
    for label in labels:
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in labels)
        true_positive = confusion[label][label]
        per_label[label] = {
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / support if support else 0.0,
            "support": support,
        }

    per_label_accuracy = {}
    for label in fixed_labels:
        label_rows = [row for row in rows if row["ground_truth"] == label]
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
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
