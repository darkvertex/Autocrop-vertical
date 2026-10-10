import numpy as np

from tracking import (
    SpeakerTracker,
    SmoothedCameraman,
    apply_hysteresis,
    delivery_size,
    render_wide_frame,
    sample_frame_indices,
    source_already_fits,
    vote_layout,
)


def _cameraman():
    # 1920x1080 landscape into 9:16. Crop width is int(1080 * 9/16) == 607.
    return SmoothedCameraman(1080, 1920, 1920, 1080, 9 / 16)


def _box_at(center_x, size=10):
    return [int(center_x - size / 2), 0, size, size]


def test_camera_holds_inside_the_safe_zone():
    cameraman = _cameraman()
    cameraman.update_target(_box_at(1000))
    assert cameraman.target_center_x == 1000
    x1, y1, x2, y2 = cameraman.get_crop_box()
    assert cameraman.current_center_x == 960
    assert y1 == 0 and y2 == 1080
    assert x2 - x1 == 607


def test_large_jump_waits_for_three_confirmations():
    cameraman = _cameraman()
    jumped = _box_at(1360)
    cameraman.update_target(jumped)
    cameraman.update_target(jumped)
    assert cameraman.target_center_x == 960
    cameraman.update_target(jumped)
    assert cameraman.target_center_x == 1360
    # The confirmed move is more than half the crop, so the pan is the fast step.
    cameraman.get_crop_box()
    assert cameraman.current_center_x == 960 + 15


def test_scene_cut_snaps_to_the_new_subject():
    cameraman = _cameraman()
    cameraman.update_target(_box_at(1000))
    cameraman.begin_scene()
    cameraman.update_target(_box_at(1400))
    assert cameraman.current_center_x == 1400
    assert cameraman.target_center_x == 1400
    cameraman.get_crop_box(force_snap=True)
    assert cameraman.current_center_x == 1400


def test_speaker_stays_put_during_cooldown_and_forgets_on_reset():
    tracker = SpeakerTracker(cooldown_frames=30)
    width = 1920
    first = {"box": [100, 0, 80, 80], "score": 80 * 80}
    other = {"box": [900, 0, 200, 200], "score": 200 * 200}

    assert tracker.get_target([first], 0, width) == first["box"]
    # A much larger face appears immediately. The cooldown keeps the first face.
    assert tracker.get_target([first, other], 5, width) == first["box"]
    # The locked face blinked. Hold the camera instead of jumping.
    assert tracker.get_target([other], 6, width) is None

    tracker.reset()
    assert tracker.get_target([other], 7, width) == other["box"]


def test_speaker_can_switch_after_the_cooldown():
    tracker = SpeakerTracker(cooldown_frames=30)
    width = 1920
    first = {"box": [100, 0, 40, 40], "score": 40 * 40}
    other = {"box": [900, 0, 180, 180], "score": 180 * 180}
    tracker.get_target([first], 0, width)
    assert tracker.get_target([other], 40, width) == other["box"]


def test_vote_layout_thresholds():
    assert vote_layout([]) == ("LETTERBOX", "wide")
    assert vote_layout([("TRACK", 1), ("TRACK", 1), ("LETTERBOX", 0)]) == ("TRACK", "speaker")
    assert vote_layout([("LETTERBOX", 3), ("LETTERBOX", 3), ("TRACK", 1)]) == ("LETTERBOX", "wide")
    assert vote_layout([("TRACK", 2), ("TRACK", 2), ("TRACK", 1)]) == ("TRACK", "group")


def test_hysteresis_flips_only_a_short_scene():
    strategies = ["TRACK", "LETTERBOX", "TRACK"]
    assert apply_hysteresis(strategies, [90, 10, 90], fps=30) == ["TRACK", "TRACK", "TRACK"]
    assert apply_hysteresis(strategies, [90, 90, 90], fps=30) == strategies


def test_sample_indices_stay_inside_the_scene():
    indices = sample_frame_indices(0, 10, samples=5)
    assert indices
    assert min(indices) >= 0
    assert max(indices) < 10


def test_delivery_size_upscales_narrow_crops_and_leaves_wide_ones():
    assert delivery_size(1920, 1080, 9 / 16) == (1080, 1920)
    assert delivery_size(1280, 720, 9 / 16) == (1080, 1920)
    # 2160 * 9/16 = 1215, rounded up to an even width. Not scaled down to 1080.
    assert delivery_size(3840, 2160, 9 / 16) == (1216, 2160)
    assert source_already_fits(1080, 1920, 9 / 16)
    assert source_already_fits(1920, 1080, 9 / 16) is False


def test_process_video_wide_shot_scales_to_1080_without_black_bars(tmp_path):
    import cv2

    from main import process_video

    src = tmp_path / "wide.avi"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    assert writer.isOpened()
    frame = np.full((180, 320, 3), (20, 40, 200), dtype=np.uint8)
    for _ in range(8):
        writer.write(frame)
    writer.release()

    out = tmp_path / "out.mp4"
    result = process_video(str(src), str(out), ratio="9:16", quality="fast", encoder="libx264")
    assert result == str(out)
    cap = cv2.VideoCapture(str(out))
    ok, rendered = cap.read()
    cap.release()
    assert ok
    assert rendered.shape == (1920, 1080, 3)
    # The corner is the blurred fill of the picture. A black letterbox would be 0.
    assert int(rendered[0, 0].max()) > 0


def test_process_video_portrait_source_passes_through(tmp_path):
    import cv2

    from main import process_video

    src = tmp_path / "portrait.avi"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (180, 320))
    assert writer.isOpened()
    frame = np.full((320, 180, 3), (30, 60, 180), dtype=np.uint8)
    for _ in range(8):
        writer.write(frame)
    writer.release()

    out = tmp_path / "out.mp4"
    result = process_video(str(src), str(out), ratio="9:16", quality="fast", encoder="libx264")
    assert result == str(out)
    cap = cv2.VideoCapture(str(out))
    ok, rendered = cap.read()
    cap.release()
    assert ok
    assert rendered.shape == (1920, 1080, 3)


def test_wide_frame_is_a_sharp_picture_on_a_blurred_fill():
    frame = np.full((90, 160, 3), (10, 20, 200), dtype=np.uint8)
    # A hard white speck. The blurred fill smears it; a black letterbox would not.
    frame[0, 0] = (255, 255, 255)
    composed = render_wide_frame(frame, 90, 160)
    assert composed.shape == (160, 90, 3)
    assert int(composed[0, 0].sum()) > 0
    center = composed[80, 45]
    assert abs(int(center[0]) - 10) < 40
    assert abs(int(center[2]) - 200) < 40
    # The top row is the blurred fill, so the speck is not a single white pixel.
    assert int(composed[0].max()) < 255 or int((composed[0] == 255).sum()) != 3
