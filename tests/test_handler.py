"""Runpod handler tests. These never load Cog, YOLO, or the Runpod SDK."""
import base64
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import handler
from main import ProcessingError


@pytest.fixture(autouse=True)
def isolated_output(monkeypatch, tmp_path):
    for name in (*handler.BUCKET_ENV_VARS, "BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(handler, "VOLUME_ROOT", tmp_path / "no-volume")


def _write_output(tmp_path, payload, capture=None):
    def _run(video_path, options):
        if capture is not None:
            capture["video"] = video_path
            capture["video_bytes"] = video_path.read_bytes()
            capture["options"] = options
        work = tmp_path / "autocrop-out"
        work.mkdir()
        out = work / "vertical.mp4"
        out.write_bytes(payload)
        return out

    return _run


def test_platform_limits_are_seven_days():
    assert handler.EXECUTION_TIMEOUT_MS == 604_800_000
    assert handler.JOB_TTL_MS == 604_800_000


def test_missing_input_object():
    assert handler.handler({}) == {"error": "Job input must be an object."}
    assert handler.handler({"input": "video"})["error"]


def test_missing_video(monkeypatch, tmp_path):
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"x"))
    result = handler.handler({"input": {}})
    assert result == {"error": "Input video is required."}


def test_defaults_match_replicate_model(monkeypatch, tmp_path):
    source = tmp_path / "horizontal.mp4"
    source.write_bytes(b"source")
    capture = {}
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out", capture))

    result = handler.handler({"id": "job-1", "input": {"video": str(source)}})

    assert capture["options"] == {
        "ratio": "9:16",
        "quality": "balanced",
        "crf": -1,
        "preset": "auto",
        "frame_skip": 0,
        "downscale": 0,
        "encoder": "hw",
    }
    assert capture["video_bytes"] == b"source"
    assert result == {"video_base64": base64.b64encode(b"out").decode("ascii")}
    assert source.is_file()
    assert not (tmp_path / "autocrop-out").exists()


def test_explicit_options_are_forwarded(monkeypatch, tmp_path):
    source = tmp_path / "clip.mov"
    source.write_bytes(b"source")
    capture = {}
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out", capture))

    handler.handler({
        "input": {
            "video": str(source),
            "ratio": "4:5",
            "quality": "high",
            "crf": "20",
            "preset": "medium",
            "frame_skip": 2.0,
            "downscale": 1,
            "encoder": "hevc",
        }
    })

    assert capture["options"] == {
        "ratio": "4:5",
        "quality": "high",
        "crf": 20,
        "preset": "medium",
        "frame_skip": 2,
        "downscale": 1,
        "encoder": "hevc",
    }


@pytest.mark.parametrize("crf", [True, 1.5, "nope", [20]])
def test_rejects_non_integer_crf(monkeypatch, tmp_path, crf):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))
    result = handler.handler({"input": {"video": str(source), "crf": crf}})
    assert result == {"error": "crf must be an integer."}


def test_https_video_is_downloaded(monkeypatch, tmp_path):
    capture = {}

    def fake_download(url, dest):
        assert url == "https://cdn.example.com/in.mov"
        dest.write_bytes(b"downloaded")

    monkeypatch.setattr(handler, "download_https", fake_download)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out", capture))

    handler.handler({"input": {"video": "https://cdn.example.com/in.mov"}})
    assert capture["video_bytes"] == b"downloaded"
    assert capture["video"].suffix == ".mov"


def test_download_failure_is_an_error(monkeypatch, tmp_path):
    def fail_download(url, dest):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(handler, "download_https", fail_download)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))
    result = handler.handler({"input": {"video": "https://cdn.example.com/in.mp4"}})
    assert result["error"].startswith("Could not download video:")


def test_rejects_non_https_urls(monkeypatch, tmp_path):
    def fail_download(url, dest):
        raise AssertionError(f"should not download {url}")

    monkeypatch.setattr(handler, "download_https", fail_download)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))
    result = handler.handler({"input": {"video": "http://cdn.example.com/in.mp4"}})
    assert "https" in result["error"]


def test_redirect_handler_rejects_non_https():
    redirector = handler._HttpsOnlyRedirect()
    request = urllib.request.Request("https://cdn.example.com/in.mp4")
    with pytest.raises(ValueError, match="non-https"):
        redirector.redirect_request(
            request, None, 302, "Found", {}, "http://cdn.example.com/in.mp4"
        )


def test_base64_and_data_uri(monkeypatch, tmp_path):
    payload = base64.b64encode(b"frame-bytes").decode("ascii")
    capture = {}
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out", capture))

    handler.handler({"input": {"video": payload}})
    assert capture["video_bytes"] == b"frame-bytes"

    wrapped = f"data:video/mp4;base64,{payload}"
    handler.handler({"input": {"video": wrapped}})
    assert capture["video_bytes"] == b"frame-bytes"


def test_processing_error_becomes_error_dict(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")

    def fail(video_path, options):
        raise ProcessingError("Quality must be one of: fast, balanced, high")

    monkeypatch.setattr(handler, "run_prediction", fail)
    result = handler.handler({"input": {"video": str(source), "quality": "ultra"}})
    assert result == {"error": "Quality must be one of: fast, balanced, high"}


def test_return_base64_too_large(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    payload = b"x" * (handler.BASE64_MAX_BYTES + 1)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, payload))
    result = handler.handler({"input": {"video": str(source), "return_base64": True}})
    assert "BUCKET_ENDPOINT_URL" in result["error"]
    assert "/runpod-volume" in result["error"]
    assert "return_base64" in result["error"]


def test_small_file_without_storage_is_base64(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"tiny"))
    result = handler.handler({"input": {"video": str(source)}})
    assert result == {"video_base64": base64.b64encode(b"tiny").decode("ascii")}


def test_large_file_without_storage_explains_options(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    payload = b"x" * (handler.BASE64_MAX_BYTES + 1)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, payload))
    result = handler.handler({"input": {"video": str(source)}})
    assert "BUCKET_SECRET_ACCESS_KEY" in result["error"]
    assert "/runpod-volume" in result["error"]


def test_bucket_upload_wins_over_volume(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    volume = tmp_path / "runpod-volume"
    volume.mkdir()
    monkeypatch.setattr(handler, "VOLUME_ROOT", volume)
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("BUCKET_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("BUCKET_NAME", "autocrop")
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))

    def fake_upload(job_id, file_path):
        assert job_id == "job-9"
        assert file_path.read_bytes() == b"out"
        return "https://s3.example.com/autocrop/job-9.mp4"

    monkeypatch.setattr(handler, "upload_output", fake_upload)
    result = handler.handler({"id": "job-9", "input": {"video": str(source)}})
    assert result == {"video_url": "https://s3.example.com/autocrop/job-9.mp4"}
    assert list(volume.iterdir()) == []


def test_return_base64_skips_bucket(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("BUCKET_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))

    def fail_upload(job_id, file_path):
        raise AssertionError("bucket should not be used")

    monkeypatch.setattr(handler, "upload_output", fail_upload)
    result = handler.handler({
        "input": {"video": str(source), "return_base64": "true"}
    })
    assert result == {"video_base64": base64.b64encode(b"out").decode("ascii")}


def test_network_volume_copy(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    volume = tmp_path / "runpod-volume"
    volume.mkdir()
    monkeypatch.setattr(handler, "VOLUME_ROOT", volume)
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"rendered"))

    result = handler.handler({"id": "abc/1", "input": {"video": str(source)}})
    dest = Path(result["video_path"])
    assert dest == volume / "outputs" / "abc-1.mp4"
    assert dest.read_bytes() == b"rendered"


def test_bucket_upload_without_url_is_an_error(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("BUCKET_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(handler, "run_prediction", _write_output(tmp_path, b"out"))
    monkeypatch.setattr(handler, "upload_output", lambda job_id, file_path: "local_upload/out.mp4")
    result = handler.handler({"input": {"video": str(source)}})
    assert "BUCKET_" in result["error"]
