"""Smoothed vertical camera.

Adapted from OpenShorts (MIT). Copyright (c) 2024 OpenShorts.

The camera holds still while the subject stays near the center, ignores a
single wild detection, and cuts to a new subject at a scene boundary. Wide
shots are a separate layout: a sharp picture over an intentionally blurred
copy of the same frame.
"""

import os
import threading

# Consecutive detections a large target move must survive before the camera
# follows it. 1 restores an immediate jump.
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)

# Reset the tracker and the cameraman at every scene cut so the first face in
# the new shot is framed immediately. 0 keeps the previous shot's damping.
SCENE_CUT_RESET = os.environ.get("SCENE_CUT_RESET", "1") != "0"

# Detectors never need a full-resolution frame. Boxes are scaled back up.
DETECT_MAX_WIDTH = 640
# Detect every Nth frame. The cameraman interpolates between updates.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
# YOLO is heavier than BlazeFace, so the person fallback runs less often.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2

# Shorts uploads are treated as low quality below 1080px wide. The crop itself
# still comes from the source; this only scales the finished frame up.
DELIVERY_MIN_WIDTH = 1080

# Share of the output height occupied by the sharp wide-shot picture.
# 0.42 keeps most of a 16:9 frame while making the picture larger than a
# full-width letterbox. The blurred layer behind it is a designed fill.
GENERAL_CONTENT_HEIGHT_RATIO = 0.42

_face_detector = None
_detect_lock = threading.Lock()


def delivery_size(orig_w, orig_h, aspect_ratio):
    """Output (width, height) for a reframe of this source.

    Both dimensions are even. A crop narrower than ``DELIVERY_MIN_WIDTH`` is
    scaled up. A source that is already wider is not scaled down.
    """
    out_h = int(orig_h)
    out_w = int(out_h * aspect_ratio)
    if out_w > orig_w:
        out_w = int(orig_w)
        out_h = int(out_w / aspect_ratio)

    if out_w < DELIVERY_MIN_WIDTH:
        out_w = DELIVERY_MIN_WIDTH
        out_h = int(round(out_w / aspect_ratio))

    return out_w + (out_w % 2), out_h + (out_h % 2)


def source_already_fits(orig_w, orig_h, aspect_ratio, tol=0.01):
    """True when the source is already at or narrower than the target aspect."""
    if orig_h <= 0:
        return False
    return orig_w / float(orig_h) <= aspect_ratio * (1 + tol)


def full_width_content_height(orig_w, orig_h, out_w):
    """Even height of the source when its full width fills ``out_w``."""
    if orig_w <= 0:
        return 2
    fg_h = int(round(out_w * orig_h / float(orig_w)))
    return fg_h + (fg_h % 2)


def render_wide_frame(frame, output_width, output_height, content_height_ratio=GENERAL_CONTENT_HEIGHT_RATIO):
    """Sharp picture centered on a blurred fill of the same frame.

    The blur is drawn on purpose. It is not a resize artifact. The sharp layer
    is at least ``content_height_ratio`` of the output height, and never
    shorter than the height that keeps the source's full width.
    """
    import cv2

    orig_h, orig_w = frame.shape[:2]
    bg_w = max(int(orig_w * (output_height / float(orig_h))), 1)
    background = cv2.resize(frame, (bg_w, output_height), interpolation=cv2.INTER_LINEAR)
    start_x = max(0, (background.shape[1] - output_width) // 2)
    background = background[:, start_x:start_x + output_width]
    if background.shape[1] != output_width or background.shape[0] != output_height:
        background = cv2.resize(background, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Blur a quarter-resolution copy. A full-frame Gaussian of this strength
    # looks the same and costs much more.
    small = cv2.resize(
        background,
        (max(output_width // 4, 2), max(output_height // 4, 2)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (13, 13), 0)
    background = cv2.resize(small, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    fg_h = int(output_height * content_height_ratio)
    fg_h = max(fg_h, full_width_content_height(orig_w, orig_h, output_width))
    fg_h += fg_h % 2
    fg_w = max(int(round(orig_w * fg_h / float(orig_h))), 2)
    fg_w += fg_w % 2
    foreground = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR)
    if foreground.shape[0] > output_height:
        top = (foreground.shape[0] - output_height) // 2
        foreground = foreground[top:top + output_height, :]
    if foreground.shape[1] > output_width:
        left = (foreground.shape[1] - output_width) // 2
        foreground = foreground[:, left:left + output_width]

    y_offset = (output_height - foreground.shape[0]) // 2
    x_offset = (output_width - foreground.shape[1]) // 2
    composed = background.copy()
    composed[
        y_offset:y_offset + foreground.shape[0],
        x_offset:x_offset + foreground.shape[1],
    ] = foreground
    return composed


class SmoothedCameraman:
    """Heavy-tripod camera. It pans only after the subject leaves the safe zone."""

    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio

        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
            self.crop_width = video_width
            self.crop_height = int(self.crop_width / aspect_ratio)

        # Stay put while the subject remains inside the middle half of the crop.
        self.safe_zone_radius = self.crop_width * 0.25
        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = False

    def begin_scene(self):
        """Drop the previous shot so the next accepted target is a cut, not a pan."""
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True

    def update_target(self, box):
        """Update the target centre from an xywh box, ignoring a lone big jump."""
        if not box:
            return
        x, _y, w, _h = box
        new_center = x + w / 2

        if self._snap_pending:
            self._snap_pending = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center
            self.current_center_x = new_center
            return

        if abs(new_center - self.target_center_x) > self.safe_zone_radius:
            if (self._pending_target is not None
                    and abs(new_center - self._pending_target) <= self.safe_zone_radius):
                self._pending_count += 1
            else:
                self._pending_target = new_center
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = new_center

    def get_crop_box(self, force_snap=False):
        """Return (x1, y1, x2, y2) for the current frame, clamped to the source."""
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.safe_zone_radius:
                direction = 1 if diff > 0 else -1
                speed = 15.0 if abs(diff) > self.crop_width * 0.5 else 3.0
                self.current_center_x += direction * speed
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x

        half_crop = self.crop_width / 2
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop

        x1 = max(0, int(self.current_center_x - half_crop))
        x2 = min(self.video_width, int(self.current_center_x + half_crop))
        # Landscape crops use the full frame height. A shorter crop is centered
        # so the bottom of a tall source is not thrown away.
        y1 = max(0, (self.video_height - self.crop_height) // 2)
        y2 = min(self.video_height, y1 + self.crop_height)
        return x1, y1, x2, y2


class SpeakerTracker:
    """Keep the crop on one face instead of bouncing to every nearby detection."""

    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.stabilization_threshold = stabilization_frames
        self.switch_cooldown = cooldown_frames
        self.last_switch_frame = -1000
        self.next_id = 0
        self.known_faces = []

    def reset(self):
        """Forget every speaker at a scene cut."""
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.last_switch_frame = -1000
        self.known_faces = []

    def get_target(self, face_candidates, frame_number, width):
        """Return the xywh box to follow, or None to keep the current camera target."""
        current_candidates = []
        for face in face_candidates:
            x, _y, w, _h = face["box"]
            center_x = x + w / 2
            best_match_id = -1
            min_dist = width * 0.15
            for known in self.known_faces:
                if frame_number - known["last_frame"] > 30:
                    continue
                dist = abs(center_x - known["center"])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = known["id"]
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            self.known_faces = [known for known in self.known_faces if known["id"] != best_match_id]
            self.known_faces.append({"id": best_match_id, "center": center_x, "last_frame": frame_number})
            current_candidates.append({
                "id": best_match_id,
                "box": face["box"],
                "score": face["score"],
            })

        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand["id"]
            raw_score = cand["score"] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        if not current_candidates:
            return None

        best_candidate = None
        max_score = -1
        for cand in current_candidates:
            total_score = self.speaker_scores.get(cand["id"], 0)
            if cand["id"] == self.active_speaker_id:
                total_score *= 3.0
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate is None:
            return None

        target_id = best_candidate["id"]
        if target_id == self.active_speaker_id:
            self.locked_counter += 1
            return best_candidate["box"]

        if frame_number - self.last_switch_frame < self.switch_cooldown:
            old = next((c for c in current_candidates if c["id"] == self.active_speaker_id), None)
            return old["box"] if old else None

        self.active_speaker_id = target_id
        self.last_switch_frame = frame_number
        self.locked_counter = 0
        return best_candidate["box"]


def sample_frame_indices(start_f, end_f, samples=5):
    """Up to ``samples`` frame indices spread through [start_f, end_f)."""
    span = end_f - start_f
    if span <= 0:
        return [max(0, start_f)]
    margin = min(2, max(0, (span - 1) // 2))
    first = start_f + margin
    last = end_f - 1 - margin
    if last < first:
        last = first
    if samples <= 1 or first == last:
        return [int(first)]
    return sorted({
        int(round(first + i * (last - first) / (samples - 1)))
        for i in range(samples)
    })


def is_near_black(frame, threshold=16):
    """True for fades and cut-to-black frames, which carry no faces."""
    return float(frame.mean()) < threshold


def vote_layout(sample_strategies):
    """Pick TRACK or LETTERBOX from per-sample (strategy, person_count) votes.

    A tie goes to TRACK. TRACK follows the group centre when more samples are
    multi-person shots than single-person shots, and one speaker otherwise.
    LETTERBOX is the wide-shot layout (sharp picture on a blurred fill).
    """
    if not sample_strategies:
        return "LETTERBOX", "wide"
    track_votes = sum(1 for strategy, _count in sample_strategies if strategy == "TRACK")
    if track_votes * 2 >= len(sample_strategies):
        group_votes = sum(1 for strategy, count in sample_strategies if strategy == "TRACK" and count > 1)
        speaker_votes = track_votes - group_votes
        mode = "group" if group_votes > speaker_votes else "speaker"
        return "TRACK", mode
    return "LETTERBOX", "wide"


def apply_hysteresis(strategies, scene_lengths, fps):
    """Flip a short scene when both neighbors already agree on the other layout.

    A layout change is a full-frame jump, so a one-frame sampling miss should
    not flap. Returns a new list.
    """
    chosen = list(strategies)
    max_flip_frames = int(2.0 * float(fps))
    for i in range(1, len(chosen) - 1):
        if (scene_lengths[i] < max_flip_frames
                and chosen[i - 1] == chosen[i + 1] != chosen[i]):
            chosen[i] = chosen[i - 1]
    return chosen


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]


def enclosing_xywh(boxes):
    """Enclosing xywh box around xywh boxes."""
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def upper_body_xywh(person_xyxy):
    """Top 40% of a person box, which is where the face sits when BlazeFace misses."""
    x1, y1, x2, y2 = person_xyxy
    height = max(1, y2 - y1)
    return [int(x1), int(y1), int(max(1, x2 - x1)), int(height * 0.4)]


def group_target(face_candidates):
    """One xywh box for a group: the enclosing box when several faces are visible."""
    if not face_candidates:
        return None
    boxes = [cand["box"] for cand in face_candidates]
    if len(boxes) == 1:
        return boxes[0]
    return enclosing_xywh(boxes)


def _detection_frame(frame):
    import cv2

    height, width = frame.shape[:2]
    if width <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = width / DETECT_MAX_WIDTH
    small_w = DETECT_MAX_WIDTH
    small_h = max(int(height / scale), 2)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return small, scale


def get_face_detector():
    """Load BlazeFace once. The graph ships inside the mediapipe package."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
    return _face_detector


def detect_face_candidates(frame):
    """Return BlazeFace boxes as {'box': [x, y, w, h], 'score': area} in source pixels."""
    import cv2

    height, width = frame.shape[:2]
    small, _scale = _detection_frame(frame)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with _detect_lock:
        results = get_face_detector().process(rgb)
    if not results.detections:
        return []

    candidates = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        if w < 2 or h < 2:
            continue
        candidates.append({"box": [x, y, w, h], "score": w * h})
    return candidates


def detect_people(frame):
    """Return full-body person boxes as xyxy in source pixels. Empty when nobody is found."""
    from main import get_yolo_model

    small, scale = _detection_frame(frame)
    with _detect_lock:
        results = get_yolo_model()(small, verbose=False, classes=[0])
    people = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in box.xyxy[0]]
            if x2 > x1 and y2 > y1:
                people.append([x1, y1, x2, y2])
    return people


def largest_person_upper(frame):
    """xywh upper body of the largest detected person, or None."""
    people = detect_people(frame)
    if not people:
        return None
    largest = max(people, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    return upper_body_xywh(largest)


def analysis_for_frame(frame):
    """Person boxes for ``decide_cropping_strategy``.

    YOLO person boxes preserve the existing fit test. BlazeFace is used only
    when YOLO finds nobody, so a face the person detector missed can still track.
    """
    people = detect_people(frame)
    if people:
        return [{"person_box": box, "face_box": None} for box in people]
    return [
        {"person_box": xywh_to_xyxy(face["box"]), "face_box": xywh_to_xyxy(face["box"])}
        for face in detect_face_candidates(frame)
    ]


def analyze_video_layouts(video_path, scenes, frame_height, aspect_ratio, decide):
    """Sample each scene and return layout dicts with strategy, mode, and person_count.

    ``decide`` is ``decide_cropping_strategy``. Near-black samples are ignored.
    A short scene flanked by the opposite layout is flipped to match its neighbors.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    layouts = []
    lengths = []
    try:
        for start, end in scenes:
            start_f = start.get_frames()
            end_f = end.get_frames()
            lengths.append(max(0, end_f - start_f))
            votes = []
            counts = []
            for index in sample_frame_indices(start_f, end_f):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None or is_near_black(frame):
                    continue
                analysis = analysis_for_frame(frame)
                strategy, _box = decide(analysis, frame_height, aspect_ratio)
                votes.append((strategy, len(analysis)))
                counts.append(len(analysis))
            strategy, mode = vote_layout(votes)
            layouts.append({
                "strategy": strategy,
                "mode": mode,
                "person_count": max(counts) if counts else 0,
            })
    finally:
        cap.release()

    smoothed = apply_hysteresis([item["strategy"] for item in layouts], lengths, fps)
    for item, strategy in zip(layouts, smoothed):
        if strategy != item["strategy"]:
            item["strategy"] = strategy
            item["mode"] = "speaker" if strategy == "TRACK" else "wide"
    return layouts
