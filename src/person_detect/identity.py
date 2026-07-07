"""Target face embedding and person-candidate identity scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np

from person_detect.boxes import Box, clamp_box
from person_detect.detector import PersonDetection
from person_detect.tracking import PersonCandidate

DEFAULT_FACE_MODEL_MIRROR = "https://hf-mirror.com"
DEFAULT_FACE_MODEL_REPO = "vladmandic/insightface-faceanalysis"


class TargetFaceMatcher:
    """Match faces inside person boxes against one target face photo."""

    def __init__(
        self,
        face_image_path: str | Path,
        *,
        model_name: str = "buffalo_s",
        model_mirror: str | None = DEFAULT_FACE_MODEL_MIRROR,
        model_repo: str = DEFAULT_FACE_MODEL_REPO,
        model_root: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            import cv2
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface/opencv/onnxruntime dependencies are missing. "
                "Run `uv sync --python 3.11` first."
            ) from exc

        self.face_image_path = Path(face_image_path)
        self._cv2 = cv2
        root = _model_root(model_root)
        _remove_corrupt_model_zip(model_name, root=root)
        if model_mirror:
            _ensure_model_from_hf_mirror(
                model_name,
                endpoint=model_mirror,
                repo=model_repo,
                root=root,
            )
        self._app = FaceAnalysis(
            name=model_name,
            root=str(root),
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)
        self.target_embedding = self._load_target_embedding(self.face_image_path)

    def annotate_candidates(
        self,
        frame,
        detections: Iterable[PersonDetection],
    ) -> list[PersonCandidate]:
        """Attach target-face similarity scores to detected person boxes."""

        frame_height, frame_width = frame.shape[:2]
        candidates: list[PersonCandidate] = []
        for detection in detections:
            box = clamp_box(detection.box, (frame_width, frame_height))
            crop = _crop_box(frame, box)
            score = self.score_image(crop) if crop is not None else None
            candidates.append(PersonCandidate(box=box, face_score=score))
        return candidates

    def score_image(self, image) -> float | None:
        """Return the best target-face similarity score found in ``image``."""

        faces = self._app.get(image)
        if not faces:
            return None
        scores = [
            float(np.dot(self.target_embedding, _normalize(face.embedding)))
            for face in faces
            if getattr(face, "embedding", None) is not None
        ]
        return max(scores) if scores else None

    def _load_target_embedding(self, path: Path) -> np.ndarray:
        image = self._cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取目标人脸照片: {path}")

        faces = self._app.get(image)
        if not faces:
            raise ValueError("目标照片未检测到人脸，请换一张更清晰的正脸照片。")

        face = max(faces, key=_face_area)
        if getattr(face, "embedding", None) is None:
            raise ValueError("目标照片的人脸缺少 embedding，请换一张更清晰的照片。")
        return _normalize(face.embedding)


def _normalize(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _face_area(face) -> float:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _crop_box(frame, box: Box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _remove_corrupt_model_zip(model_name: str, *, root: str | Path | None = None) -> None:
    """Delete an interrupted InsightFace model zip before the library reads it."""

    base_dir = _model_root(root)
    zip_path = base_dir / "models" / f"{model_name}.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()


def _ensure_model_from_hf_mirror(
    model_name: str,
    *,
    endpoint: str,
    repo: str,
    root: str | Path | None = None,
    downloader=None,
) -> Path:
    """Download and extract an InsightFace model zip from a HuggingFace mirror."""

    base_dir = _model_root(root)
    model_dir = base_dir / "models" / model_name
    if _has_onnx_files(model_dir):
        return model_dir

    zip_path = base_dir / "models" / f"{model_name}.zip"
    if not zipfile.is_zipfile(zip_path):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        download = downloader or _download_file
        download(_build_hf_model_url(endpoint=endpoint, repo=repo, model_name=model_name), zip_path)

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"模型压缩包下载不完整或损坏: {zip_path}")

    _extract_model_zip(zip_path, model_dir)
    if not _has_onnx_files(model_dir):
        raise RuntimeError(f"模型解压后未找到 ONNX 文件: {model_dir}")
    return model_dir


def _build_hf_model_url(*, endpoint: str, repo: str, model_name: str) -> str:
    """Build a HuggingFace-style model zip URL."""

    return f"{endpoint.rstrip('/')}/{repo.strip('/')}/resolve/main/{model_name}.zip"


def _download_file(url: str, destination: str | Path) -> None:
    """Download a file atomically so interrupted downloads do not poison the cache."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        if shutil.which("curl") is not None:
            _download_with_curl(url, tmp_path)
        else:
            _download_with_requests(url, tmp_path)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_with_curl(url: str, destination: str | Path) -> None:
    """Download a URL with curl, which handles hf-mirror redirects well on macOS."""

    destination = Path(destination)
    print(f"Downloading {destination} from {url}...")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--connect-timeout",
            "20",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def _download_with_requests(url: str, destination: str | Path) -> None:
    """Download a URL with requests when curl is unavailable."""

    import requests

    destination = Path(destination)
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        next_report = 20 * 1024 * 1024
        print(f"Downloading {destination} from {url}...")
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_report:
                    percent = downloaded / total * 100
                    downloaded_mb = downloaded / 1024 / 1024
                    total_mb = total / 1024 / 1024
                    print(f"  {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
                    next_report += 20 * 1024 * 1024


def _extract_model_zip(zip_path: Path, model_dir: Path) -> None:
    """Extract a model zip into the exact layout expected by InsightFace."""

    models_dir = model_dir.parent
    extract_dir = models_dir / f".{model_dir.name}_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    onnx_parent = _find_onnx_parent(extract_dir)
    if onnx_parent is None:
        shutil.rmtree(extract_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    for child in onnx_parent.iterdir():
        shutil.move(str(child), model_dir / child.name)
    shutil.rmtree(extract_dir)


def _find_onnx_parent(path: Path) -> Path | None:
    if list(path.glob("*.onnx")):
        return path
    for child in path.iterdir():
        if child.is_dir():
            found = _find_onnx_parent(child)
            if found is not None:
                return found
    return None


def _has_onnx_files(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.onnx"))


def _model_root(root: str | Path | None) -> Path:
    return Path(root).expanduser() if root is not None else Path.home() / ".insightface"
