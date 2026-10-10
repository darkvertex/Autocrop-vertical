"""Replicate Cog predictor for AutoCrop-Vertical.

The image build downloads yolov8n.pt. setup() loads it once per worker so
predictions do not touch the network.
"""
import os
import shutil
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path

from main import get_face_cascade, get_yolo_model, process_video

WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS", "/src/models/yolov8n.pt")
X264_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load YOLO and the face cascade. Weights must already be on disk."""
        os.environ["YOLO_OFFLINE"] = "1"
        if not os.path.isfile(WEIGHTS_PATH):
            raise RuntimeError(
                f"YOLO weights not found at {WEIGHTS_PATH}. "
                "The image build should have downloaded them."
            )
        os.environ["YOLO_WEIGHTS"] = WEIGHTS_PATH
        get_yolo_model(WEIGHTS_PATH)
        get_face_cascade()

    def predict(
        self,
        video: Path = Input(description="Horizontal input video (mp4, mov, mkv, and other FFmpeg formats)."),
        ratio: str = Input(
            description="Output aspect ratio as W:H. Examples: 9:16 (Shorts/Reels/TikTok), 4:5 (Instagram feed), 1:1.",
            default="9:16",
        ),
        quality: str = Input(
            description="Encoding quality preset.",
            default="balanced",
            choices=["fast", "balanced", "high"],
        ),
        crf: int = Input(
            description="libx264 CRF override, 0-51, lower is higher quality. -1 keeps the quality preset.",
            default=-1,
            ge=-1,
            le=51,
        ),
        preset: str = Input(
            description="libx264 preset override. 'auto' keeps the quality preset.",
            default="auto",
            choices=["auto", *X264_PRESETS],
        ),
        frame_skip: int = Input(
            description="Frames to skip during scene detection. 0 processes every frame.",
            default=0,
            ge=0,
            le=30,
        ),
        downscale: int = Input(
            description="Scene-detection downscale factor. 0 lets PySceneDetect choose.",
            default=0,
            ge=0,
            le=8,
        ),
    ) -> Path:
        """Crop the video to a vertical frame, tracking people when they fit."""
        suffix = LocalPath(str(video)).suffix or ".mp4"
        input_path = LocalPath("/tmp/autocrop-input").with_suffix(suffix)
        shutil.copy(str(video), input_path)
        output_path = LocalPath("/tmp/vertical.mp4")
        if output_path.exists():
            output_path.unlink()

        result = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            ratio=ratio,
            quality=quality,
            crf=None if crf < 0 else int(crf),
            preset=None if preset == "auto" else preset,
            frame_skip=int(frame_skip),
            downscale=int(downscale),
            encoder="auto",
        )
        if not result or not os.path.isfile(result):
            raise RuntimeError("Processing finished without writing an output video.")
        return Path(result)
