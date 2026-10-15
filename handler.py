"""Runpod Serverless handler for AutoCrop-Vertical.

Replicate keeps using predict.py through Cog. This module loads that same
Predictor once per worker and runs one job at a time. It does not impose its
own deadline: the endpoint execution timeout is the only wall clock.
"""
from __future__ import annotations

import base64
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from main import ProcessingError

# Runpod's platform maximum. The worker cannot raise this itself. Set the
# endpoint execution timeout and job TTL to these values. See the README.
EXECUTION_TIMEOUT_MS = 604_800_000
JOB_TTL_MS = 604_800_000

# /run responses are capped at 10 MB. Base64 expands by 4/3, so 7 MB of video
# stays under that cap.
BASE64_MAX_BYTES = 7 * 1024 * 1024

VOLUME_ROOT = Path("/runpod-volume")
BUCKET_ENV_VARS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
)

_predictor = None


def get_predictor():
    """Load YOLO and BlazeFace once for this process."""
    global _predictor
    if _predictor is None:
        from predict import Predictor

        _predictor = Predictor()
        _predictor.setup()
    return _predictor


def bucket_configured() -> bool:
    return all(os.environ.get(name) for name in BUCKET_ENV_VARS)


def _as_int(value: object, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or (
        isinstance(value, float) and not value.is_integer()
    ):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
    raise ValueError(f"{name} must be an integer.")


def build_options(job_input: dict) -> dict:
    """Map a Runpod input object onto Predictor.predict keyword arguments.

    Every argument is explicit. Cog's Input() defaults are not applied when
    predict() is called from plain Python.
    """
    return {
        "ratio": job_input.get("ratio") or "9:16",
        "quality": job_input.get("quality") or "balanced",
        "crf": _as_int(job_input.get("crf"), -1, "crf"),
        "preset": job_input.get("preset") or "auto",
        "frame_skip": _as_int(job_input.get("frame_skip"), 0, "frame_skip"),
        "downscale": _as_int(job_input.get("downscale"), 0, "downscale"),
        "encoder": job_input.get("encoder") or "hw",
    }


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("Video URL redirected to a non-https address.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_https(url: str, dest: Path) -> None:
    """Download a source video. No socket timeout; the platform limit applies."""
    opener = urllib.request.build_opener(_HttpsOnlyRedirect)
    with opener.open(url, timeout=None) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _decode_base64(payload: str) -> bytes:
    compact = "".join(payload.split())
    try:
        data = base64.b64decode(compact, validate=True)
    except Exception as exc:
        raise ValueError(
            "video must be an https URL, a base64 string, or an existing file path."
        ) from exc
    if not data:
        raise ValueError("video base64 payload is empty.")
    return data


def materialize_video(video: object, dest_dir: Path) -> Path:
    """Turn a job's video value into a local file path."""
    if not isinstance(video, str) or not video.strip():
        raise ValueError("Input video is required.")
    video = video.strip()
    if "://" in video:
        parsed = urlparse(video)
        if parsed.scheme != "https":
            raise ValueError(
                f"Unsupported video URL scheme '{parsed.scheme}'. Use an https URL."
            )
        suffix = Path(parsed.path).suffix or ".mp4"
        dest = dest_dir / f"input{suffix}"
        try:
            download_https(video, dest)
        except urllib.error.URLError as exc:
            raise ValueError(f"Could not download video: {exc}") from exc
        if dest.stat().st_size == 0:
            raise ValueError("Downloaded video is empty.")
        return dest
    if video.startswith("data:"):
        header, sep, payload = video.partition(",")
        if not sep or ";base64" not in header:
            raise ValueError("data: video inputs must be base64 encoded.")
        dest = dest_dir / "input.mp4"
        dest.write_bytes(_decode_base64(payload))
        return dest
    path = Path(video)
    if path.is_file():
        return path
    dest = dest_dir / "input.mp4"
    dest.write_bytes(_decode_base64(video))
    return dest


def wants_base64(job_input: dict) -> bool:
    flag = job_input.get("return_base64")
    if isinstance(flag, str):
        return flag.strip().lower() == "true"
    return flag is True


def _too_large_message(size: int) -> str:
    return (
        "Output video is too large to return as base64 "
        f"({size} bytes, limit {BASE64_MAX_BYTES}). "
        "Pass return_base64 only for small files, set BUCKET_ENDPOINT_URL, "
        "BUCKET_ACCESS_KEY_ID, and BUCKET_SECRET_ACCESS_KEY, or mount /runpod-volume."
    )


def encode_base64(path: Path) -> dict:
    size = path.stat().st_size
    if size > BASE64_MAX_BYTES:
        raise ValueError(_too_large_message(size))
    return {"video_base64": base64.b64encode(path.read_bytes()).decode("ascii")}


def upload_output(job_id: str, file_path: Path) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket

    return upload_file_to_bucket(
        file_name=f"{job_id}.mp4",
        file_location=str(file_path),
        bucket_name=os.environ.get("BUCKET_NAME") or None,
        extra_args={"ContentType": "video/mp4"},
    )


def copy_to_volume(job_id: str, file_path: Path) -> Path:
    dest_dir = VOLUME_ROOT / "outputs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{job_id}.mp4"
    shutil.copy(file_path, dest)
    return dest


def deliver_output(job_id: str, job_input: dict, output_path: Path) -> dict:
    if wants_base64(job_input):
        return encode_base64(output_path)
    if bucket_configured():
        url = upload_output(job_id, output_path)
        if not str(url).startswith(("https://", "http://")):
            raise RuntimeError(
                "Bucket upload did not return a URL. Check the BUCKET_* settings."
            )
        return {"video_url": url}
    if VOLUME_ROOT.is_dir():
        return {"video_path": str(copy_to_volume(job_id, output_path))}
    return encode_base64(output_path)


def run_prediction(video_path: Path, options: dict) -> Path:
    from cog import Path as CogPath

    result = get_predictor().predict(video=CogPath(str(video_path)), **options)
    return Path(str(result))


def _safe_job_id(job: dict) -> str:
    raw = str(job.get("id") or "job")
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return cleaned or "job"


def _cleanup_work_dir(path: Path | None) -> None:
    if path is None:
        return
    parent = path.parent
    if parent.name.startswith("autocrop-"):
        shutil.rmtree(parent, ignore_errors=True)


def handler(job: dict) -> dict:
    job_input = job.get("input") if isinstance(job, dict) else None
    if not isinstance(job_input, dict):
        return {"error": "Job input must be an object."}

    job_id = _safe_job_id(job)
    download_dir: Path | None = None
    output_path: Path | None = None
    try:
        options = build_options(job_input)
        download_dir = Path(tempfile.mkdtemp(prefix="autocrop-src-"))
        video_path = materialize_video(job_input.get("video"), download_dir)
        output_path = run_prediction(video_path, options)
        if not output_path.is_file():
            raise RuntimeError("Processing finished without writing an output video.")
        return deliver_output(job_id, job_input, output_path)
    except (ProcessingError, ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    finally:
        if download_dir is not None:
            shutil.rmtree(download_dir, ignore_errors=True)
        _cleanup_work_dir(output_path)


def main() -> None:
    import runpod

    get_predictor()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
