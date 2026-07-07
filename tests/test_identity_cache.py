import zipfile
import shutil

import pytest

from person_detect.identity import (
    DEFAULT_FACE_MODEL_REPO,
    _build_hf_model_url,
    _download_with_curl,
    _ensure_model_from_hf_mirror,
    _remove_corrupt_model_zip,
)


def test_remove_corrupt_model_zip_deletes_partial_download(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    partial_zip = model_dir / "buffalo_s.zip"
    partial_zip.write_bytes(b"not a complete zip")

    _remove_corrupt_model_zip("buffalo_s", root=tmp_path)

    assert not partial_zip.exists()


def test_remove_corrupt_model_zip_keeps_valid_zip(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    valid_zip = model_dir / "buffalo_s.zip"
    with zipfile.ZipFile(valid_zip, "w") as archive:
        archive.writestr("model.txt", "ok")

    _remove_corrupt_model_zip("buffalo_s", root=tmp_path)

    assert valid_zip.exists()


def test_build_hf_model_url_uses_endpoint_repo_and_model_name() -> None:
    assert _build_hf_model_url(
        endpoint="https://hf-mirror.com/",
        repo=DEFAULT_FACE_MODEL_REPO,
        model_name="buffalo_s",
    ) == (
        "https://hf-mirror.com/"
        "vladmandic/insightface-faceanalysis/resolve/main/buffalo_s.zip"
    )


def test_ensure_model_from_hf_mirror_downloads_and_extracts_zip(tmp_path) -> None:
    def fake_download(url: str, destination) -> None:
        assert url.endswith("/buffalo_s.zip")
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("det_500m.onnx", "detector")
            archive.writestr("w600k_mbf.onnx", "recognizer")

    model_dir = _ensure_model_from_hf_mirror(
        "buffalo_s",
        endpoint="https://hf-mirror.com",
        repo=DEFAULT_FACE_MODEL_REPO,
        root=tmp_path,
        downloader=fake_download,
    )

    assert model_dir == tmp_path / "models" / "buffalo_s"
    assert (model_dir / "det_500m.onnx").read_text() == "detector"
    assert (model_dir / "w600k_mbf.onnx").read_text() == "recognizer"


def test_ensure_model_from_hf_mirror_skips_existing_model_dir(tmp_path) -> None:
    model_dir = tmp_path / "models" / "buffalo_s"
    model_dir.mkdir(parents=True)
    (model_dir / "det_500m.onnx").write_text("already here")

    def should_not_download(url: str, destination) -> None:
        raise AssertionError("existing model should not be downloaded")

    assert _ensure_model_from_hf_mirror(
        "buffalo_s",
        endpoint="https://hf-mirror.com",
        repo=DEFAULT_FACE_MODEL_REPO,
        root=tmp_path,
        downloader=should_not_download,
    ) == model_dir


def test_download_with_curl_supports_file_url(tmp_path) -> None:
    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")

    source = tmp_path / "source.zip"
    source.write_bytes(b"zip bytes")
    destination = tmp_path / "destination.zip"

    _download_with_curl(source.as_uri(), destination)

    assert destination.read_bytes() == b"zip bytes"
