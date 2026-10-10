import subprocess
import sys

import pytest

from main import (
    ProcessingError,
    calculate_crop_box,
    decide_cropping_strategy,
    detect_scenes,
    parse_aspect_ratio,
    process_video,
)


def test_parse_aspect_ratio_common_formats():
    assert parse_aspect_ratio("9:16") == pytest.approx(9 / 16)
    assert parse_aspect_ratio("4:5") == pytest.approx(0.8)
    assert parse_aspect_ratio("1:1") == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", ["", "9", "9:", ":16", "0:16", "9:0", "a:b", "9/16"])
def test_parse_aspect_ratio_rejects_bad_values(ratio):
    with pytest.raises(ProcessingError):
        parse_aspect_ratio(ratio)


def test_no_people_letterboxes():
    strategy, box = decide_cropping_strategy([], 1080, 9 / 16)
    assert strategy == "LETTERBOX"
    assert box is None


def test_single_person_prefers_face_box():
    analysis = [{"person_box": [0, 0, 100, 200], "face_box": [10, 20, 40, 60]}]
    strategy, box = decide_cropping_strategy(analysis, 1080, 9 / 16)
    assert strategy == "TRACK"
    assert box == [10, 20, 40, 60]


def test_single_person_falls_back_to_person_box():
    analysis = [{"person_box": [5, 6, 70, 80], "face_box": None}]
    strategy, box = decide_cropping_strategy(analysis, 1080, 9 / 16)
    assert strategy == "TRACK"
    assert box == [5, 6, 70, 80]


def test_tight_group_is_tracked():
    analysis = [
        {"person_box": [100, 0, 200, 400], "face_box": None},
        {"person_box": [250, 0, 400, 400], "face_box": None},
    ]
    strategy, box = decide_cropping_strategy(analysis, 1080, 9 / 16)
    assert strategy == "TRACK"
    assert box == [100, 0, 400, 400]


def test_spread_group_letterboxes():
    analysis = [
        {"person_box": [0, 0, 100, 400], "face_box": None},
        {"person_box": [1500, 0, 1700, 400], "face_box": None},
    ]
    strategy, box = decide_cropping_strategy(analysis, 1080, 9 / 16)
    assert strategy == "LETTERBOX"
    assert box is None


def test_crop_box_stays_inside_frame():
    # Subject on the left edge would otherwise produce a negative x.
    x1, y1, x2, y2 = calculate_crop_box([0, 0, 10, 10], 1920, 1080, 9 / 16)
    assert x1 == 0
    assert y1 == 0
    assert y2 == 1080
    assert x2 - x1 == int(1080 * 9 / 16)
    assert x2 <= 1920


def test_crop_box_clamps_right_edge():
    x1, y1, x2, y2 = calculate_crop_box([1910, 0, 1920, 10], 1920, 1080, 9 / 16)
    assert y1 == 0 and y2 == 1080
    assert x2 == 1920
    assert x1 == 1920 - int(1080 * 9 / 16)


def test_process_video_missing_input(tmp_path):
    with pytest.raises(ProcessingError, match="not found"):
        process_video(str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4"))


def test_single_shot_is_one_scene(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "shot.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:] = (40, 80, 120)
    for _ in range(10):
        writer.write(frame)
    writer.release()

    scenes, _fps = detect_scenes(str(path), downscale=1, frame_skip=0)
    assert len(scenes) == 1
    assert scenes[0][0].get_frames() == 0
    assert scenes[0][1].get_frames() == 10


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--ratio" in result.stdout
    assert "--plan-only" in result.stdout


def test_cli_rejects_bad_ratio(tmp_path):
    result = subprocess.run(
        [sys.executable, "main.py", "-i", "x.mp4", "-o", str(tmp_path / "out.mp4"), "--ratio", "nope"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Invalid aspect ratio" in result.stdout
