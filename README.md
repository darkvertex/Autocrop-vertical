# AutoCrop-Vertical: Smart Video Cropper for Social Media

![Demo of AutoCrop-Vertical](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

Automatically converts horizontal videos into vertical format for TikTok, Instagram Reels, and YouTube Shorts.

Instead of a static center crop, the script follows people through each scene and, when a shot is too wide to crop, keeps the full picture sharp on a blurred fill of the same frame.

---

### Quick Start

```bash
git clone https://github.com/darkvertex/Autocrop-vertical.git
cd Autocrop-vertical
uv sync
uv run python main.py -i video.mp4 -o vertical.mp4
```

Install [uv](https://docs.astral.sh/uv/) if you don't have it. `uv sync` creates `.venv` and installs the dependencies from `pyproject.toml` / `uv.lock`, including CPU PyTorch wheels.

The `yolov8n.pt` model weights are downloaded automatically on first run.

**Prerequisites:** Python 3.12+ and [FFmpeg](https://ffmpeg.org/) (`ffmpeg` + `ffprobe`) in your PATH.

Hosted runs go through [Replicate](#replicate-cog) or [Runpod Serverless](#runpod-serverless) and do not need a local Python install.

---

### Usage Examples

```bash
# Basic — 9:16 vertical, balanced quality
uv run python main.py -i video.mp4 -o vertical.mp4

# Instagram feed (4:5) with high quality
uv run python main.py -i video.mp4 -o vertical.mp4 --ratio 4:5 --quality high

# Fast encode, square format
uv run python main.py -i video.mp4 -o vertical.mp4 --ratio 1:1 --quality fast

# Preview the processing plan without encoding
uv run python main.py -i video.mp4 -o vertical.mp4 --plan-only

# Full control over encoding parameters
uv run python main.py -i video.mp4 -o vertical.mp4 --crf 20 --preset medium

# Use a hardware H.264 encoder when that device is present
uv run python main.py -i video.mp4 -o vertical.mp4 --encoder hw

# HEVC / H.265 (libx265). Smaller files than H.264 at the same quality name
uv run python main.py -i video.mp4 -o vertical.mp4 --encoder hevc --crf 26 --preset medium

# Maximum accuracy scene detection (slower)
uv run python main.py -i video.mp4 -o vertical.mp4 --frame-skip 0
```

---

### All Flags

**Output:**

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | Path to input video |
| `-o`, `--output` | *(required)* | Path to output video (`.mp4` appended if no extension) |
| `--ratio` | `9:16` | Output aspect ratio. Examples: `9:16`, `4:5`, `1:1` |

**Encoding quality:**

| Flag | Default | Description |
|------|---------|-------------|
| `--quality` | `balanced` | Preset: `fast`, `balanced`, or `high` (see table below) |
| `--encoder` | `auto` | `auto` = libx264. `hevc` = libx265. `hw` / `hevc_hw` = H.264 or HEVC hardware when that device is present. Or an explicit name: `libx264`, `libx265`, `h264_videotoolbox`, `h264_nvenc`, `hevc_videotoolbox`, `hevc_nvenc` |
| `--crf` | *(from quality)* | Override quality directly, 0-51, lower = better. CRF for libx264 and libx265, CQ for NVENC |
| `--preset` | *(from quality)* | Override the x264/x265 preset: `ultrafast`..`veryslow` |

**Quality presets:**

HEVC uses a CRF about 5 higher than H.264, which keeps the same quality name in a similar visual band. HEVC MP4 output is tagged `hvc1` so Apple players can open it. NVENC uses those same numbers as CQ, with unconstrained VBR (`-rc vbr -cq N -b:v 0`).

| `--quality` | H.264 CRF | HEVC CRF | Preset | Typical use |
|-------------|-----------|----------|--------|-------------|
| `fast` | 28 | 33 | veryfast | Quick previews, drafts |
| `balanced` | 23 | 28 | fast | Good quality, reasonable speed |
| `high` | 18 | 23 | slow | Best quality, largest file, slowest |

**Scene detection tuning:**

| Flag | Default | Description |
|------|---------|-------------|
| `--frame-skip` | `0` | Frames to skip during scene detection. `0` = every frame (most accurate). `1` = every other frame (~2x faster). Higher = faster but may miss cuts |
| `--downscale` | `0` (auto) | Downscale factor for scene detection. `0` = auto. `2`-`4` = faster but may miss subtle cuts |

**Other:**

| Flag | Default | Description |
|------|---------|-------------|
| `--plan-only` | off | Run scene detection + analysis only, print the plan, exit without encoding |

---

### Key Features

*   **Content-Aware Cropping:** BlazeFace follows the subject through the scene. YOLOv8 is the fallback when no face is found. The camera holds still inside a center safe zone, ignores a single wild detection, and cuts (rather than pans) onto the first face after a scene change.
*   **Wide shots:** When nobody fits in the vertical crop, the full picture stays sharp and centered over an intentionally blurred copy of the same frame. That blur is a background treatment, not a quality loss from scaling.
*   **Tight groups:** Two people who still fit in the vertical window are kept together. The camera follows the centre of the group.
*   **Scene-by-Scene Processing:** Five samples per scene choose the layout. Near-black fade frames are ignored. A short scene whose neighbors agree is kept on that layout so the frame does not flap.
*   **Delivery size:** The crop is scaled up to 1080px wide when it would otherwise be narrower, which is the usual Shorts upload size. Sources that are already wider are not scaled down. A source that is already vertical is passed through.
*   **Frame-Accurate Processing:** Every frame is processed individually with the correct per-scene strategy — no timestamp rounding or scene boundary drift.
*   **H.264 and HEVC:** `--encoder hevc` writes H.265 with libx265. `--encoder hw` and `--encoder hevc_hw` use VideoToolbox or NVENC when a one-frame session opens, and otherwise fall back to libx264 or libx265.
*   **VFR Handling:** Variable frame rate sources are automatically normalized before processing.
*   **Audio Sync:** Non-zero stream start times are detected and compensated to keep audio/video aligned.

---

### How It Works

```
Input Video
    |
    v
+-------------------------------+
| 1. Scene Detection            |  PySceneDetect splits the video into scenes
|    (--frame-skip, --downscale)|
+---------------+---------------+
                |
                v
+-------------------------------+
| 2. Content Analysis           |  Five frames per scene. YOLOv8 person
|                               |  boxes decide whether the group fits;
|                               |  BlazeFace covers a missed face
+---------------+---------------+
                |
                v
+-------------------------------+
| 3. Strategy Decision          |  Per scene: TRACK (follow subject or
|                               |  a tight group) or WIDE (sharp picture
|                               |  on a blurred fill)
+---------------+---------------+
                |
                v
+-------------------------------+
| 4. Frame Processing           |  Smoothed crop, or the wide-shot
|    (--quality, --encoder)     |  composite, piped to FFmpeg
+---------------+---------------+
                |
                v
+-------------------------------+
| 5-6. Audio extract + merge    |  Audio synced with start-time offset
+---------------+---------------+
                |
                v
          Output Video
```

Steps 1-3 are the "planning" phase (Python + AI). Step 4 applies the plan frame-by-frame and encodes via FFmpeg.

---

### Performance

Benchmarks on Apple M1 MacBook Pro (AC power):

| Resolution | Duration | Total time | Speed |
|-----------|----------|-----------|-------|
| 1280x720 | 49s | ~6s | 8.3x real-time |
| 1920x1080 | 12 min | ~51s | 13.7x real-time |

Scene detection is the dominant bottleneck (~50% of total time). Those timings are for a static crop. Following the subject also runs face detection while frames are encoded, so a tracked scene takes longer.

---

### Technical Details

This script is built on a pipeline that uses specialized libraries for each step:

*   **Core Libraries:**
    *   `PySceneDetect`: For accurate, content-aware scene cut detection.
    *   `Ultralytics (YOLOv8)`: Person detection for the fit test, and a fallback when no face is visible.
    *   `MediaPipe BlazeFace`: Face tracking while the camera is following someone.
    *   `OpenCV`: Frame reads, the wide-shot composite, and video properties.
    *   `FFmpeg` / `ffprobe`: The backbone of video encoding, audio extraction, and media stream analysis.
    *   `tqdm`: For clean and informative progress bars in the console.

*   **Processing Pipeline:**
    1.  **(Pre-processing)** If the source is VFR, it is normalized to constant frame rate.
    2.  `PySceneDetect` scans the video and returns a list of scene timestamps.
    3.  For each scene, five frames are sampled. `YOLOv8` measures whether the people fit in the vertical crop. A frame with no person falls back to `BlazeFace`. Near-black frames are skipped.
    4.  A set of rules chooses `TRACK` (one person, or a group that fits) or `WIDE` (nobody, or a group that does not fit).
    5.  OpenCV reads every frame sequentially. `TRACK` scenes follow the subject with a smoothed camera (BlazeFace every 4th frame, YOLOv8 when no face is found). `WIDE` scenes keep a sharp picture on a blurred fill. Frames are piped as raw pixels to FFmpeg. Scene boundaries stay on exact frame numbers.
    6.  Audio is extracted separately (with start-time offset correction), then merged with the processed video.

---

### Replicate (Cog)

This fork publishes a Cog image at [replicate.com/darkvertex/autocrop-vertical](https://replicate.com/darkvertex/autocrop-vertical). Local `cog predict` and that hosted model return the same vertical mp4.

The image is a GPU build (`gpu: true`, CUDA 12.8) so FFmpeg can open NVENC. PyTorch stays on the CPU wheel: YOLOv8n and BlazeFace run on CPU, and the GPU is there for `h264_nvenc` and `hevc_nvenc`. Set the model hardware to an Nvidia T4 or L40S. Those cards have an NVENC engine. A100 and H100 have decode only, and `hw` / `hevc_hw` fall back to libx264 / libx265 after a failed session probe. The build downloads `yolov8n.pt` in `cog.yaml`, so cold boots do no network I/O. `predict.py` sets `YOLO_OFFLINE=1` and refuses to start if those weights are missing.

Install the [Cog CLI](https://cog.run/getting-started/) and Docker.

**Published image** (no local `cog build`):

```bash
cog predict r8.im/darkvertex/autocrop-vertical -i video=@./horizontal.mp4 -o ./vertical.mp4
```

**Build from this checkout**, then predict with the image named in `cog.yaml`:

```bash
cog build
cog predict -i video=@./horizontal.mp4 -o ./vertical.mp4
```

Optional inputs use the same names as the Replicate API:

```bash
cog predict r8.im/darkvertex/autocrop-vertical \
  -i video=@./horizontal.mp4 \
  -i ratio=4:5 \
  -i quality=high \
  -i crf=20 \
  -i preset=medium \
  -i encoder=hevc \
  -o ./vertical.mp4
```

`quality`, `crf`, `preset`, and `encoder` are the same encoding controls as the CLI. `crf` and `preset` replace the values chosen by `quality` for libx264 and libx265. `-1` and `auto` leave that preset alone. `encoder=hevc` writes H.265.

| Input | Default | Meaning |
| --- | --- | --- |
| `video` | (required) | Horizontal video. Prefix local files with `@`. |
| `ratio` | `9:16` | Output aspect ratio as `W:H` (`4:5`, `1:1`, …). |
| `quality` | `balanced` | `fast` (CRF 28, veryfast), `balanced` (CRF 23, fast), or `high` (CRF 18, slow). |
| `crf` | `-1` | libx264 or libx265 CRF override, 0–51, lower is higher quality. `-1` keeps the quality preset. |
| `preset` | `auto` | libx264 or libx265 preset override (`ultrafast` … `veryslow`). `auto` keeps the quality preset. |
| `encoder` | `hw` | `hw` = H.264 NVENC on a T4 or L40S, otherwise libx264. `hevc_hw` = HEVC NVENC, otherwise libx265. `auto` / `libx264` = software H.264. `hevc` / `libx265` = software H.265. |
| `frame_skip` | `0` | Frames skipped during scene detection. |
| `downscale` | `0` | Scene-detection downscale. `0` is automatic. |

### Push your own build

After `cog login` (CLI auth token from https://replicate.com/auth/token, not an `r8_…` API token):

```bash
cog push r8.im/darkvertex/autocrop-vertical
```

Or another destination: `cog push r8.im/<your-username>/autocrop-vertical`.

From a checkout with [uv](https://docs.astral.sh/uv/) and Cog on your PATH:

```bash
uvx --from poethepoet poe cog-deploy
```

### GitHub Actions

- `.github/workflows/cog-build.yml` runs on every pull request: `pytest` and `cog build` plus a `cog predict` smoke test, in parallel. Nothing is pushed. The GitHub runner has no NVIDIA driver, so the default `hw` encoder falls back to libx264 there.
- `.github/workflows/cog-release.yml` runs on every push to `main` (or manually): `cog build` then `cog push`. It runs in the `production` GitHub Environment and needs the secret `REPLICATE_CLI_AUTH_TOKEN` defined there (Settings > Environments > production). The value must be the CLI auth token copied from https://replicate.com/auth/token, not an API token (`r8_...`) from the account settings page: `cog login` rejects API tokens. The job fails early with a clear message if the secret is missing or is an API token.

### Runpod Serverless

The root `Dockerfile` is a Runpod worker. It calls the same `Predictor` as the Cog model, in-process, with the same input names and defaults. `cog build` and `cog push` do not use that Dockerfile, so the Replicate image is unchanged.

Replicate stops a prediction at 10 minutes. Runpod's endpoint default is also 10 minutes, and the default job TTL is 24 hours. A stock endpoint will kill a long encode the same way. In the endpoint's Advanced settings, set both of these to the platform maximum, 7 days (`604800000` milliseconds):

- **Execution timeout** — how long the worker may run after it picks up the job.
- **Job TTL** — lifespan from submit, including time spent in the queue. If TTL expires mid-encode, the job is deleted.

Leave the idle timeout at 5 seconds. That only scales an idle worker down. It does not stop a running job.

Use a GPU with an NVENC engine (Nvidia T4 or L40S). A100 and H100 have decode only, and `hw` / `hevc_hw` fall back to libx264 / libx265 after a failed session probe, same as on Replicate.

Build and push a `linux/amd64` image, then create a Queue endpoint from it:

```bash
docker build --platform linux/amd64 -t your-registry/autocrop-vertical:runpod .
docker push your-registry/autocrop-vertical:runpod
```

Submit with `/run`, then poll `/status`. `/runsync` holds the HTTP connection open and only keeps the result for about a minute, so it is the wrong call for a long encode. A request `policy` replaces the endpoint timeouts for that job, so set both to 7 days there too:

```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "authorization: Bearer RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "video": "https://example.com/horizontal.mp4",
      "ratio": "9:16"
    },
    "policy": {
      "executionTimeout": 604800000,
      "ttl": 604800000
    }
  }'
```

`video` is an `https` URL, a base64 string (or a `data:...;base64,` URI), or a file path on a mounted network volume. Optional fields match the [Replicate inputs](#replicate-cog): `ratio`, `quality`, `crf`, `preset`, `frame_skip`, `downscale`, and `encoder`.

`/run` results are kept for 30 minutes and the response body is capped at 10 MB, so a finished mp4 is returned in one of three ways:

| Result | When |
| --- | --- |
| `video_url` | `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, and `BUCKET_SECRET_ACCESS_KEY` are set on the endpoint. Optional `BUCKET_NAME` selects the bucket; otherwise the Runpod SDK uses a month-year name. |
| `video_path` | A network volume is mounted at `/runpod-volume`. The file is copied to `/runpod-volume/outputs/<job_id>.mp4`. |
| `video_base64` | Neither of the above is set and the file is at most 7 MB, or the input sets `return_base64` to `true` for a file under that size. Larger files return an error naming the other two options. |

The handler does not add its own time limit around the download or the encode.

---

### Changelog

#### Vertical tracking

*   **The crop follows the subject.** BlazeFace runs every 4th frame. The camera stays still while the person remains near the centre, waits for a large jump to repeat before following it, and cuts to the new subject at a scene change. YOLOv8 still decides whether a group fits, and fills in when no face is found.
*   **Wide shots use a blurred fill.** The sharp picture sits on a blurred copy of the same frame. The blur is drawn as the background. It is not caused by scaling.
*   **1080-wide delivery.** A crop narrower than 1080px is scaled up. Wider sources are left alone. A video that is already vertical is passed through.

#### v1.5.0 — Replicate model and uv

*   **Hosted Cog model.** `cog.yaml` and `predict.py` run the same pipeline as the CLI. The image is a GPU build so NVENC can run on a T4 or L40S. Weights are baked into the image. CI builds the image and smoke-tests a short vertical encode; pushes to `main` publish `r8.im/darkvertex/autocrop-vertical`.
*   **uv project.** `requirements.txt` is replaced by `pyproject.toml` and `uv.lock`. Dependencies install with `uv sync`. PyTorch resolves from the CPU wheel index.
*   **Single-shot videos.** Scene detection now treats a video with no cuts as one scene, so a clip without a camera change still gets cropped.
*   **Encoding controls on the API.** `quality`, `crf`, `preset`, and `encoder` are inputs on the Replicate model, matching the CLI flags.
*   **HEVC output.** `--encoder hevc` (and `encoder=hevc` on Replicate) encodes with libx265 and tags the MP4 `hvc1`. `hevc_hw` uses HEVC VideoToolbox or NVENC when a one-frame session opens.
*   **NVENC on the hosted model.** `encoder` defaults to `hw`. On a T4 or L40S that selects `h264_nvenc`. On an A100, H100, or a machine with no NVIDIA device, it stays on libx264.



#### v1.4.1 (2026-02-15) — Cropping Accuracy Fix

*   **Fixed incorrect crop/letterbox decisions near scene boundaries.** The v1.3 `filter_complex` pipeline used seconds-based `trim` filters, which caused floating-point misalignment with the frame-based scene boundaries from PySceneDetect. This led to frames at scene transitions receiving the wrong strategy (e.g., a properly tracked person switching to letterbox mid-scene, or a group shot being cropped instead of letterboxed). Restored the original frame-by-frame processing pipeline which uses exact frame numbers for scene boundary matching, guaranteeing frame-accurate results.

#### v1.4.0 (2026-02-15) — Hardware Encoding & Scene Detection Tuning

**New Features:**

*   **Hardware encoder support (`--encoder`).** New flag with three modes: `auto` (libx264, default for best quality/compatibility), `hw` (auto-detect VideoToolbox on macOS or NVENC on NVIDIA), or an explicit encoder name. Quality presets (`--quality`) map automatically per encoder type.
*   **Configurable scene detection (`--frame-skip`, `--downscale`).** Power users can tune the speed/accuracy trade-off. Default `--frame-skip 0` processes every frame for maximum accuracy; increase for faster detection on longer videos.

#### v1.3.0 (2026-02-15) — Configurable Output & Quality Presets

**New Features:**

*   **Configurable aspect ratio (`--ratio`).** Output is no longer locked to 9:16. Use `--ratio 4:5` for Instagram feed, `--ratio 1:1` for square, or any custom W:H ratio.
*   **Quality presets (`--quality`).** Choose between `fast` (CRF 28, veryfast), `balanced` (CRF 23, fast — default), or `high` (CRF 18, slow). Power users can override directly with `--crf` and `--preset`.
*   **Dry-run mode (`--plan-only`).** Runs scene detection and analysis only, prints the processing plan, and exits without encoding. Useful for previewing decisions before committing to a long encode.
*   **Fixed output pixel format.** Encoder now outputs `yuv420p` instead of `yuv444p`, which is compatible with all players and platforms and produces smaller files.
*   **Improved logging and progress reporting.** Input file summary upfront (resolution, duration, fps, codec, file size, frame count), progress bars on all slow operations, and a final summary with output size, compression ratio, and processing speed.

#### v1.1.0 (2026-02-14)

**Bug Fixes:**

*   **Fixed audio/video desynchronization.** This was caused by two separate issues:
    *   The frame rate was being read from PySceneDetect while frames were read by OpenCV. A mismatch between the two (e.g. 29.97 vs 30.0) caused the encoded video duration to drift from the audio. FPS is now read from OpenCV (the same backend that reads the frames) with explicit `-vsync cfr` enforcement.
    *   Many source files (especially YouTube downloads) have a non-zero `start_time` on the video stream (e.g. audio at 0.0s, video at 1.8s). The script now detects this offset via `ffprobe` and trims the extracted audio to match, so the two streams stay aligned.
*   **Fixed crash on videos without an audio stream.** The script now detects whether audio exists using `ffprobe` and skips the audio extraction/merge steps gracefully.
*   **Fixed hardcoded `.aac` temp audio file.** The temp audio container is now `.mkv`, which accepts any audio codec. Previously, source files with non-AAC audio (MP3, Opus, AC3, etc.) could fail or produce corrupt output.
*   **Fixed crash when output path has no file extension.** The script now auto-appends `.mp4` if no extension is provided.
*   **Fixed orphaned temp files on failure.** Temporary files are now cleaned up on all exit paths, not just on success.

**Improvements:**

*   **Variable frame rate (VFR) handling.** Phone-recorded videos often use VFR, which caused frame timing drift. The script now detects VFR sources via `ffprobe` and normalizes them to constant frame rate before processing.
*   **Corrupt frame resilience.** If a frame fails to process (bad crop, corrupt data), it is duplicated from the previous good frame instead of being dropped. This preserves the total frame count and prevents audio drift.
*   **Lazy model loading.** YOLO and Haar cascade models are now loaded on first use instead of at import time. Heavy library imports (`torch`, `ultralytics`, `cv2`, etc.) are deferred until after argument parsing, so `--help` is instant.
*   **Pinned dependency versions.** `requirements.txt` now specifies compatible version ranges to prevent breakage from upstream changes.
*   **Replaced `exit()` with `sys.exit(1)`.** Ensures proper exit codes and reliable behavior in all environments.
