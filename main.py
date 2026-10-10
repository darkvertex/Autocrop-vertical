import sys
import time
import subprocess
import argparse
import os

# --- Constants ---
ASPECT_RATIO = 9 / 16
X264_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)

# Lazy-loaded models — initialized on first use so that importing the module
# or running --help doesn't trigger heavyweight model loading.
_model = None
_model_path = None


class ProcessingError(RuntimeError):
    """User-facing failure. The CLI prints this and exits; Cog surfaces it as a failed prediction."""


def parse_aspect_ratio(ratio):
    """Parse a W:H ratio string into a width/height float."""
    try:
        width_s, height_s = ratio.split(":")
        width = int(width_s)
        height = int(height_s)
        if width <= 0 or height <= 0:
            raise ValueError
        return width / height
    except (ValueError, AttributeError):
        raise ProcessingError(
            f"Invalid aspect ratio '{ratio}'. Use format W:H (e.g. 9:16, 4:5, 1:1)"
        ) from None


def get_yolo_model(weights_path=None):
    """Load YOLOv8 once.

    `weights_path` wins, then the YOLO_WEIGHTS environment variable, then
    `yolov8n.pt` in the working directory (downloaded by Ultralytics if missing).
    """
    global _model, _model_path
    path = weights_path or os.environ.get("YOLO_WEIGHTS") or "yolov8n.pt"
    if _model is None or _model_path != path:
        from ultralytics import YOLO
        _model = YOLO(path)
        _model_path = path
    return _model


def detect_scenes(video_path, downscale=0, frame_skip=0):
    """Detect scene boundaries.

    Args:
        video_path: Path to the video file.
        downscale: Downscale factor for processing (0 = auto-detect based on
                   resolution).  Higher values are faster but may miss subtle cuts.
        frame_skip: Number of frames to skip between each processed frame.
                    0 = process every frame (default, most accurate).
    """
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    if downscale > 0:
        scene_manager.auto_downscale = False
        scene_manager.downscale = downscale
    scene_manager.detect_scenes(video=video, show_progress=True, frame_skip=frame_skip)
    # A video with no cuts is one scene. Without start_in_scene, PySceneDetect
    # returns an empty list and a single-shot clip cannot be cropped.
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    return scene_list, video.frame_rate

def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height, aspect_ratio=ASPECT_RATIO):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * aspect_ratio
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None

def calculate_crop_box(target_box, frame_width, frame_height, aspect_ratio=ASPECT_RATIO):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * aspect_ratio)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2

def get_video_properties(video_path):
    """Returns (width, height, fps) from OpenCV — the same backend that reads frames."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def get_media_info(video_path):
    """Returns a dict with human-readable info about the input file."""
    info = {}
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration,size',
             '-show_entries', 'stream=codec_name,codec_type,width,height,r_frame_rate',
             '-of', 'json', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            fmt = data.get('format', {})
            info['duration'] = float(fmt.get('duration', 0))
            info['size_bytes'] = int(fmt.get('size', 0))
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and 'video_codec' not in info:
                    info['video_codec'] = stream.get('codec_name', 'unknown')
                    info['width'] = stream.get('width', 0)
                    info['height'] = stream.get('height', 0)
                    rate = stream.get('r_frame_rate', '0/1')
                    parts = rate.split('/')
                    if len(parts) == 2 and int(parts[1]) != 0:
                        info['fps'] = round(int(parts[0]) / int(parts[1]), 2)
                    else:
                        info['fps'] = float(parts[0])
                elif stream.get('codec_type') == 'audio' and 'audio_codec' not in info:
                    info['audio_codec'] = stream.get('codec_name', 'unknown')
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return info

def format_duration(seconds):
    """Formats seconds into a human-readable string like '1h 32m 15s'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def format_file_size(size_bytes):
    """Formats bytes into a human-readable string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"

def has_audio_stream(video_path):
    """Uses ffprobe to check whether the file contains an audio stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        return result.returncode == 0 and 'audio' in result.stdout
    except FileNotFoundError:
        # ffprobe not available — assume audio exists and let ffmpeg handle it
        return True

def get_stream_start_time(video_path, stream_type='v:0'):
    """Returns the start_time of a stream in seconds (0.0 if unavailable)."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', stream_type,
             '-show_entries', 'stream=start_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0

def is_variable_frame_rate(video_path):
    """Uses ffprobe to check if the video has a variable frame rate."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
        # ffprobe returns "num/den" for both r_frame_rate and avg_frame_rate
        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return False
        def parse_rate(s):
            nums = s.strip().split('/')
            if len(nums) == 2 and int(nums[1]) != 0:
                return int(nums[0]) / int(nums[1])
            return float(nums[0])
        r_fps = parse_rate(parts[0])
        avg_fps = parse_rate(parts[1])
        # If the real frame rate and average frame rate differ significantly, it's VFR
        return abs(r_fps - avg_fps) > 0.5
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        return False

def run_ffmpeg_with_progress(command, total_duration, desc="Processing"):
    """Runs an FFmpeg command and shows a tqdm progress bar based on stderr output.
    Returns (returncode, stderr_text) so callers can print errors."""
    from tqdm import tqdm
    import re
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    pbar = tqdm(total=int(total_duration), desc=desc, unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]')
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+)\.(\d+)')
    last_seconds = 0
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        match = time_pattern.search(line)
        if match:
            h, m, s, _ = match.groups()
            current_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            if current_seconds > last_seconds:
                pbar.update(current_seconds - last_seconds)
                last_seconds = current_seconds
    pbar.update(max(0, int(total_duration) - last_seconds))
    pbar.close()
    process.wait()
    return process.returncode, ''.join(stderr_lines)

def normalize_to_cfr(video_path, output_path, total_duration=0):
    """Re-muxes a VFR video to constant frame rate."""
    print("  Normalizing variable frame rate to constant frame rate...")
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vsync', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', output_path
    ]
    if total_duration > 0:
        returncode, stderr_text = run_ffmpeg_with_progress(command, total_duration, desc="VFR → CFR")
    else:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Warning: VFR normalization failed, proceeding with original file.")
            print("  Stderr:", e.stderr.decode())
            return False
    if returncode != 0:
        print(f"  Warning: VFR normalization failed, proceeding with original file.")
        return False
    return True

# FFmpeg encoder name -> preset family used by build_encoder_args.
ENCODER_FAMILIES = {
    'libx264': 'libx264',
    'h264_videotoolbox': 'videotoolbox',
    'h264_nvenc': 'nvenc',
    'libx265': 'libx265',
    'hevc_videotoolbox': 'hevc_videotoolbox',
    'hevc_nvenc': 'hevc_nvenc',
}
H264_HW_CANDIDATES = ('h264_videotoolbox', 'h264_nvenc')
HEVC_HW_CANDIDATES = ('hevc_videotoolbox', 'hevc_nvenc')
_ffmpeg_encoders = None


def ffmpeg_encoders():
    """Return the set of video encoder names compiled into this FFmpeg."""
    global _ffmpeg_encoders
    if _ffmpeg_encoders is None:
        found = set()
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    # Encoder rows look like " V....D libx265  libx265 H.265 / HEVC".
                    if (len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] == 'V'
                            and parts[1].replace('_', '').isalnum()):
                        found.add(parts[1])
        except FileNotFoundError:
            pass
        _ffmpeg_encoders = found
    return _ffmpeg_encoders


_nvenc_session_cache = {}


def nvenc_device_present():
    """True when the NVIDIA driver has exposed a GPU device node."""
    return os.path.exists('/dev/nvidia0') or os.path.exists('/dev/nvidiactl')


def nvenc_session_opens(encoder_name, run=None):
    """True when FFmpeg can open a one-frame NVENC session.

    A100 and H100 expose a device node and still have no NVENC engine, so the
    device check alone would select an encoder that then fails the job.
    """
    if run is None and encoder_name in _nvenc_session_cache:
        return _nvenc_session_cache[encoder_name]
    if run is None:
        run = subprocess.run
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'color=black:s=64x64:r=30:d=0.04',
        '-frames:v', '1',
        '-c:v', encoder_name,
        '-f', 'null', '-',
    ]
    try:
        result = run(command, capture_output=True, timeout=30)
        ok = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if run is subprocess.run:
        _nvenc_session_cache[encoder_name] = ok
    return ok


def hardware_encoder_usable(encoder_name, device_present=None, session_opens=None):
    """True when this hardware encoder can encode a frame on this machine.

    FFmpeg lists NVENC and VideoToolbox whenever they were compiled in, even
    on a machine that cannot run them.
    """
    if encoder_name.endswith('_videotoolbox'):
        return sys.platform == 'darwin'
    if encoder_name.endswith('_nvenc'):
        present = nvenc_device_present if device_present is None else device_present
        if not present():
            return False
        opens = nvenc_session_opens if session_opens is None else session_opens
        return bool(opens(encoder_name))
    return False


def first_usable_encoder(candidates, fallback, available=None, usable=None):
    """Pick the first compiled, usable encoder, or `fallback`."""
    if available is None:
        available = ffmpeg_encoders()
    if usable is None:
        usable = hardware_encoder_usable
    for name in candidates:
        if name in available and usable(name):
            return name
    return fallback


def resolve_encoder(requested, available=None, usable=None):
    """Resolve a user encoder choice to (ffmpeg_encoder_name, preset_family).

    `auto` is libx264. `hw` is a usable H.264 hardware encoder, or libx264.
    `hevc` is libx265. `hevc_hw` is a usable HEVC hardware encoder, or libx265.
    Explicit names such as `h264_nvenc` and `hevc_videotoolbox` are also accepted.
    """
    if available is None:
        available = ffmpeg_encoders()
    choice = (requested or 'auto').strip()
    if choice == 'auto':
        return 'libx264', 'libx264'
    if choice == 'hw':
        name = first_usable_encoder(H264_HW_CANDIDATES, 'libx264', available, usable)
        return name, ENCODER_FAMILIES[name]
    if choice in ('hevc', 'libx265'):
        return 'libx265', 'libx265'
    if choice == 'hevc_hw':
        name = first_usable_encoder(HEVC_HW_CANDIDATES, 'libx265', available, usable)
        return name, ENCODER_FAMILIES[name]
    if choice in ENCODER_FAMILIES:
        return choice, ENCODER_FAMILIES[choice]
    known = ', '.join(['auto', 'hw', 'hevc', 'hevc_hw', *ENCODER_FAMILIES])
    raise ProcessingError(f"Unsupported encoder '{requested}'. Use one of: {known}.")


def build_encoder_args(encoder_type, quality_level, crf_override=None, preset_override=None):
    """Returns a list of FFmpeg encoder arguments for the given encoder and quality.

    quality_level is one of 'fast', 'balanced', 'high'.
    crf_override applies to libx264, libx265, and NVENC (as CQ).
    preset_override applies to libx264 and libx265.
    HEVC CRF targets sit about 5 higher than H.264 so the same quality name
    stays in a similar visual band. HEVC MP4 output is tagged hvc1.
    NVENC uses unconstrained VBR so CQ, not a default bitrate cap, sets the size.
    """
    presets = {
        'libx264': {
            'fast':     ['-crf', '28', '-preset', 'veryfast'],
            'balanced': ['-crf', '23', '-preset', 'fast'],
            'high':     ['-crf', '18', '-preset', 'slow'],
        },
        'libx265': {
            'fast':     ['-crf', '33', '-preset', 'veryfast', '-tag:v', 'hvc1'],
            'balanced': ['-crf', '28', '-preset', 'fast', '-tag:v', 'hvc1'],
            'high':     ['-crf', '23', '-preset', 'slow', '-tag:v', 'hvc1'],
        },
        'videotoolbox': {
            'fast':     ['-b:v', '3M', '-allow_sw', '1', '-realtime', '0'],
            'balanced': ['-b:v', '6M', '-allow_sw', '1', '-realtime', '0'],
            'high':     ['-b:v', '12M', '-allow_sw', '1', '-realtime', '0'],
        },
        'hevc_videotoolbox': {
            'fast':     ['-b:v', '2M', '-allow_sw', '1', '-realtime', '0', '-tag:v', 'hvc1'],
            'balanced': ['-b:v', '4M', '-allow_sw', '1', '-realtime', '0', '-tag:v', 'hvc1'],
            'high':     ['-b:v', '8M', '-allow_sw', '1', '-realtime', '0', '-tag:v', 'hvc1'],
        },
        'nvenc': {
            'fast':     ['-rc', 'vbr', '-cq', '28', '-b:v', '0', '-preset', 'p1'],
            'balanced': ['-rc', 'vbr', '-cq', '23', '-b:v', '0', '-preset', 'p4'],
            'high':     ['-rc', 'vbr', '-cq', '18', '-b:v', '0', '-preset', 'p7'],
        },
        'hevc_nvenc': {
            'fast':     ['-rc', 'vbr', '-cq', '33', '-b:v', '0', '-preset', 'p1', '-tag:v', 'hvc1'],
            'balanced': ['-rc', 'vbr', '-cq', '28', '-b:v', '0', '-preset', 'p4', '-tag:v', 'hvc1'],
            'high':     ['-rc', 'vbr', '-cq', '23', '-b:v', '0', '-preset', 'p7', '-tag:v', 'hvc1'],
        },
    }

    args = list(presets[encoder_type][quality_level])

    # libx264 and libx265 share CRF plus the ultrafast..veryslow preset names.
    # NVENC uses the same 0-51 number as CQ. x264 preset names are not NVENC presets.
    if encoder_type in ('libx264', 'libx265'):
        if crf_override is not None:
            args[args.index('-crf') + 1] = str(crf_override)
        if preset_override is not None:
            args[args.index('-preset') + 1] = preset_override
    elif encoder_type in ('nvenc', 'hevc_nvenc') and crf_override is not None:
        args[args.index('-cq') + 1] = str(crf_override)

    return args


def _layout_label(strategy):
    """User-facing name. LETTERBOX is the wide shot: sharp picture on a blurred fill."""
    if strategy == "LETTERBOX":
        return "WIDE"
    return strategy


def _render_frame(
    frame, strategy, mode, frame_number, scene_start,
    cameraman, speaker_tracker, frame_width,
    output_width, output_height,
):
    """Crop with the smoothed camera, or compose the wide-shot backdrop."""
    import cv2
    import tracking

    if strategy != "TRACK":
        cameraman.current_center_x = frame_width / 2
        cameraman.target_center_x = frame_width / 2
        return tracking.render_wide_frame(frame, output_width, output_height)

    if mode == "passthrough":
        if frame.shape[1] == output_width and frame.shape[0] == output_height:
            return frame
        return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    is_scene_start = frame_number == scene_start
    detect_now = frame_number % tracking.DETECT_STRIDE == 0 or (is_scene_start and tracking.SCENE_CUT_RESET)
    if is_scene_start and tracking.SCENE_CUT_RESET:
        speaker_tracker.reset()
        cameraman.begin_scene()

    if detect_now:
        candidates = tracking.detect_face_candidates(frame)
        yolo_now = frame_number % tracking.YOLO_FALLBACK_STRIDE == 0 or (is_scene_start and tracking.SCENE_CUT_RESET)
        if mode == "group":
            target_box = tracking.group_target(candidates)
            if target_box is None and yolo_now:
                target_box = tracking.largest_person_upper(frame)
            if target_box:
                cameraman.update_target(target_box)
        else:
            target_box = speaker_tracker.get_target(candidates, frame_number, frame_width)
            if target_box:
                cameraman.update_target(target_box)
            elif yolo_now:
                person_box = tracking.largest_person_upper(frame)
                if person_box:
                    cameraman.update_target(person_box)

    x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
    if y2 > y1 and x2 > x1:
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def process_video(
    input_path,
    output_path,
    ratio="9:16",
    quality="balanced",
    crf=None,
    preset=None,
    plan_only=False,
    frame_skip=0,
    downscale=0,
    encoder="auto",
):
    """Crop a horizontal video into a vertical one.

    Returns the output path, or None when plan_only is set.
    Raises ProcessingError for expected failures (bad input, ffmpeg errors).
    """
    aspect_ratio = parse_aspect_ratio(ratio)
    if quality not in ("fast", "balanced", "high"):
        raise ProcessingError("Quality must be one of: fast, balanced, high")
    if crf is not None and not (0 <= int(crf) <= 51):
        raise ProcessingError("CRF must be between 0 and 51")
    if preset is not None and preset not in X264_PRESETS:
        raise ProcessingError(
            "Preset must be an x264/x265 preset: " + ", ".join(X264_PRESETS)
        )
    if not os.path.isfile(input_path):
        raise ProcessingError(f"Input video not found: {input_path}")

    # auto/hw stay on H.264. hevc and hevc_hw select H.265.
    encoder_name, encoder_type = resolve_encoder(encoder)
    try:
        enc_args = build_encoder_args(encoder_type, quality,
                                      crf_override=crf, preset_override=preset)
    except KeyError:
        raise ProcessingError(f"Unsupported encoder '{encoder}'.") from None

    # Defer heavy imports until after arg parsing so --help is instant
    import cv2
    import numpy as np
    from tqdm import tqdm

    script_start_time = time.time()

    input_video = input_path
    final_output_video = output_path

    # Ensure the output filename has a video extension so FFmpeg can determine the format
    _, ext = os.path.splitext(final_output_video)
    if not ext:
        final_output_video += '.mp4'
    out_dir = os.path.dirname(os.path.abspath(final_output_video))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.mkv"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"
    
    def cleanup_temp_files():
        """Remove any leftover temporary files."""
        for f in [temp_video_output, temp_audio_output, temp_cfr_input]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # Clean up previous temp files if they exist
    cleanup_temp_files()
    if os.path.exists(final_output_video): os.remove(final_output_video)

    try:
        # Print input file summary
        media_info = get_media_info(input_video)
        if media_info:
            print(f"\n📄 Input: {os.path.basename(input_path)}")
            parts = []
            if 'width' in media_info:
                parts.append(f"{media_info['width']}x{media_info['height']}")
            if 'fps' in media_info:
                parts.append(f"{media_info['fps']} fps")
            if 'video_codec' in media_info:
                parts.append(media_info['video_codec'])
            if 'audio_codec' in media_info:
                parts.append(media_info['audio_codec'])
            if 'duration' in media_info:
                parts.append(format_duration(media_info['duration']))
            if 'size_bytes' in media_info:
                parts.append(format_file_size(media_info['size_bytes']))
            print(f"   {' | '.join(parts)}")
            total_frames_est = int(media_info.get('duration', 0) * media_info.get('fps', 0))
            if total_frames_est > 0:
                print(f"   ~{total_frames_est:,} frames to process")
        enc_label = f"{encoder_name} ({' '.join(enc_args)})"
        print(f"   Ratio: {ratio} | Quality: {quality} | Encoder: {enc_label}")
        print()

        # Pre-processing: normalize VFR to CFR if needed
        if is_variable_frame_rate(input_video):
            print("⚠️  Variable frame rate detected — normalizing to constant frame rate first...")
            duration = media_info.get('duration', 0) if media_info else 0
            if normalize_to_cfr(input_video, temp_cfr_input, total_duration=duration):
                input_video = temp_cfr_input
                print("✅ VFR normalization complete.")
            else:
                print("⚠️  Proceeding with original VFR file (audio sync may be affected).")

        print("🎬 Step 1: Detecting scenes...")
        step_start_time = time.time()
        scenes, _ = detect_scenes(input_video, downscale=downscale, frame_skip=frame_skip)
        step_end_time = time.time()
    
        if not scenes:
            raise ProcessingError("No scenes were detected. Aborting.")
    
        print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
        for i, (start, end) in enumerate(scenes):
            print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")


        print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
        step_start_time = time.time()
        # Get fps from OpenCV — the same backend that reads the frames — to avoid
        # frame-rate mismatches between the reader and encoder that cause audio drift.
        original_width, original_height, fps = get_video_properties(input_video)
    
        import tracking
        # Scale a narrow crop up to 1080px wide. Wider sources stay at the crop size.
        OUTPUT_WIDTH, OUTPUT_HEIGHT = tracking.delivery_size(
            original_width, original_height, aspect_ratio
        )
        passthrough = tracking.source_already_fits(original_width, original_height, aspect_ratio)

        scenes_analysis = []
        if passthrough:
            print("   Source already matches the target aspect — passing it through.")
            for start_time, end_time in scenes:
                scenes_analysis.append({
                    'start_frame': start_time.get_frames(),
                    'end_frame': end_time.get_frames(),
                    'start_seconds': start_time.get_seconds(),
                    'end_seconds': end_time.get_seconds(),
                    'strategy': 'TRACK',
                    'mode': 'passthrough',
                    'person_count': 0,
                })
        else:
            layouts = tracking.analyze_video_layouts(
                input_video, scenes, original_height, aspect_ratio, decide_cropping_strategy,
            )
            for (start_time, end_time), layout in zip(scenes, layouts):
                scenes_analysis.append({
                    'start_frame': start_time.get_frames(),
                    'end_frame': end_time.get_frames(),
                    'start_seconds': start_time.get_seconds(),
                    'end_seconds': end_time.get_seconds(),
                    'strategy': layout['strategy'],
                    'mode': layout['mode'],
                    'person_count': layout['person_count'],
                })
        step_end_time = time.time()
        print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

        print("\n📋 Step 3: Generated Processing Plan")
        for i, scene_data in enumerate(scenes_analysis):
            num_people = scene_data['person_count']
            strategy = _layout_label(scene_data['strategy'])
            start_time = scenes[i][0].get_timecode()
            end_time = scenes[i][1].get_timecode()
            print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

        if plan_only:
            track_count = sum(1 for s in scenes_analysis if s['strategy'] == 'TRACK')
            wide_count = sum(1 for s in scenes_analysis if s['strategy'] == 'LETTERBOX')
            elapsed = time.time() - script_start_time
            print(f"\n📊 Plan summary: {track_count} TRACK / {wide_count} WIDE scenes")
            print(f"⏱️  Analysis took {elapsed:.1f}s. Run without --plan-only to encode.")
            return None

        print("\n✂️ Step 4: Processing video frames...")
        step_start_time = time.time()

        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
            '-r', str(fps), '-i', '-',
            '-c:v', encoder_name, *enc_args,
            '-pix_fmt', 'yuv420p',
            '-r', str(fps), '-vsync', 'cfr',
            '-an', temp_video_output
        ]

        ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_number = 0
        current_scene_index = 0
        dropped_frames = 0
        last_output_frame = None
        cameraman = tracking.SmoothedCameraman(
            OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=aspect_ratio,
        )
        speaker_tracker = tracking.SpeakerTracker(cooldown_frames=30)

        num_scenes = len(scenes_analysis)
        with tqdm(total=total_frames, desc=f"Processing [scene 1/{num_scenes}]",
                  unit="fr", dynamic_ncols=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_scene_index < len(scenes_analysis) - 1 and \
                   frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                    current_scene_index += 1
                    pbar.set_description(f"Processing [scene {current_scene_index + 1}/{num_scenes}]")

                scene_data = scenes_analysis[current_scene_index]
                strategy = scene_data['strategy']
                mode = scene_data['mode']

                try:
                    output_frame = _render_frame(
                        frame, strategy, mode, frame_number, scene_data['start_frame'],
                        cameraman, speaker_tracker, original_width,
                        OUTPUT_WIDTH, OUTPUT_HEIGHT,
                    )
                    last_output_frame = output_frame
                except Exception:
                    dropped_frames += 1
                    if last_output_frame is not None:
                        output_frame = last_output_frame
                    else:
                        output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

                ffmpeg_process.stdin.write(output_frame.tobytes())
                frame_number += 1
                pbar.update(1)

        if dropped_frames > 0:
            print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated from the previous frame.")

        ffmpeg_process.stdin.close()
        stderr_output = ffmpeg_process.stderr.read().decode()
        ffmpeg_process.wait()
        cap.release()

        if ffmpeg_process.returncode != 0:
            raise ProcessingError(f"FFmpeg frame processing failed.\nStderr: {stderr_output}")
        step_end_time = time.time()
        print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

        input_has_audio = has_audio_stream(input_video)

        if input_has_audio:
            print("\n🔊 Step 5: Extracting original audio...")
            step_start_time = time.time()

            # Some files have a non-zero video start_time (e.g. audio starts at 0s
            # but video starts at 1.8s). OpenCV ignores this offset and reads frames
            # from the first video frame, so the processed video starts at 0s.
            # We must trim the audio to match: skip audio before the video started,
            # and limit to the video's duration.
            video_start = get_stream_start_time(input_video, 'v:0')
            audio_extract_command = [
                'ffmpeg', '-y', '-ss', str(video_start),
                '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
            ]
            try:
                subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                step_end_time = time.time()
                print(f"✅ Audio extracted in {step_end_time - step_start_time:.2f}s.")
            except subprocess.CalledProcessError as e:
                detail = e.stderr.decode() if e.stderr else ""
                raise ProcessingError(f"Audio extraction failed.\nStderr: {detail}") from None

            print("\n✨ Step 6: Merging video and audio...")
            step_start_time = time.time()
            merge_command = [
                'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
                '-c:v', 'copy', '-c:a', 'copy', '-shortest', final_output_video
            ]
            try:
                subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                step_end_time = time.time()
                print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
            except subprocess.CalledProcessError as e:
                detail = e.stderr.decode() if e.stderr else ""
                raise ProcessingError(f"Final merge failed.\nStderr: {detail}") from None

            cleanup_temp_files()
        else:
            print("\n🔇 Step 5: No audio stream detected, skipping audio extraction.")
            # Just rename the temp video as the final output
            os.rename(temp_video_output, final_output_video)
            cleanup_temp_files()

        script_end_time = time.time()
        total_time = script_end_time - script_start_time

        # Final summary
        print(f"\n{'─' * 50}")
        print(f"🎉 All done! Final video saved to {final_output_video}")
        print(f"{'─' * 50}")
        output_info = get_media_info(final_output_video)
        if output_info:
            out_parts = []
            if 'width' in output_info:
                out_parts.append(f"{output_info['width']}x{output_info['height']}")
            if 'duration' in output_info:
                out_parts.append(format_duration(output_info['duration']))
            if 'size_bytes' in output_info:
                out_parts.append(format_file_size(output_info['size_bytes']))
            print(f"   Output: {' | '.join(out_parts)}")
        if media_info and output_info and media_info.get('size_bytes') and output_info.get('size_bytes'):
            size_pct = output_info['size_bytes'] / media_info['size_bytes'] * 100
            print(f"   Size:   {format_file_size(media_info['size_bytes'])} → {format_file_size(output_info['size_bytes'])} ({size_pct:.0f}% of original)")
        print(f"   Time:   {format_duration(total_time)} ({total_time:.1f}s)")
        if media_info and media_info.get('duration'):
            speed = media_info['duration'] / total_time if total_time > 0 else 0
            print(f"   Speed:  {speed:.1f}x real-time")
        return final_output_video
    finally:
        cleanup_temp_files()


def build_parser():
    parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    parser.add_argument('--ratio', type=str, default='9:16',
                        help="Output aspect ratio as W:H (default: 9:16). Examples: 9:16, 4:5, 1:1")
    parser.add_argument('--quality', type=str, default='balanced', choices=['fast', 'balanced', 'high'],
                        help="Encoding quality preset (default: balanced). fast=quick encode, balanced=good quality, high=best quality/slow")
    parser.add_argument('--crf', type=int, default=None,
                        help="Override CRF directly (0-51, lower=better quality) for libx264 and libx265. Overrides --quality.")
    parser.add_argument('--preset', type=str, default=None,
                        help="Override the x264/x265 preset directly (ultrafast..veryslow). Overrides --quality.")
    parser.add_argument('--plan-only', action='store_true',
                        help="Only run scene detection and analysis (Steps 1-3), then print the processing plan without encoding.")
    parser.add_argument('--frame-skip', type=int, default=0,
                        help="Frames to skip during scene detection (default: 0 = every frame, most accurate). "
                             "1 = every other frame (~2x faster). Higher = faster but may miss quick cuts.")
    parser.add_argument('--downscale', type=int, default=0,
                        help="Downscale factor for scene detection (default: 0 = auto). "
                             "Higher values (2-4) are faster but may miss subtle scene changes.")
    parser.add_argument('--encoder', type=str, default='auto',
                        help="Video encoder: 'auto' (libx264), 'hevc' (libx265), "
                             "'hw' or 'hevc_hw' (hardware H.264 or HEVC when that device is present), "
                             "or an explicit name: libx264, libx265, h264_videotoolbox, h264_nvenc, "
                             "hevc_videotoolbox, hevc_nvenc.")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    try:
        process_video(
            input_path=args.input,
            output_path=args.output,
            ratio=args.ratio,
            quality=args.quality,
            crf=args.crf,
            preset=args.preset,
            plan_only=args.plan_only,
            frame_skip=args.frame_skip,
            downscale=args.downscale,
            encoder=args.encoder,
        )
    except ProcessingError as exc:
        print(f"❌ {exc}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
