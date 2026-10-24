import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from hand_raise import hand_raised
from head_pose import classify_head, face_metrics
from models import ensure_models
from phone_detect import assign_phones
from scoring import class_percent, student_level
from tracker import StudentTracker

COLORS = {"engaged": (94, 197, 34), "partial": (11, 158, 245), "disengaged": (68, 68, 239)} # BGR


def _match_face(pt, bboxes):
    """Index of the face box that contains this body point, or None."""
    best, best_d = None, 1e18
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        w, h = x1 - x0, y1 - y0
        if x0 - 0.5 * w <= pt[0] <= x1 + 0.5 * w and y0 - 0.5 * h <= pt[1] <= y1 + 0.5 * h:
            d = ((x0 + x1) / 2 - pt[0]) ** 2 + ((y0 + y1) / 2 - pt[1]) ** 2
            if d < best_d:
                best, best_d = i, d
    return best


class Analyzer:
    def __init__(self, max_students=config.MAX_STUDENTS):
        paths = ensure_models()
        # VIDEO mode lets MediaPipe track faces between frames, which is steadier than IMAGE mode
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=max_students))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths[config.POSE_MODEL]),
            running_mode=vision.RunningMode.VIDEO, num_poses=max_students))
        self.phone = None
        self.phone_error = None
        self.reset()

    def reset(self):
        """Forget everything seen so far (call when a session or the camera starts)."""
        self.tracker = StudentTracker()
        self.frame_i = 0
        self._ts = getattr(self, "_ts", 0)
        self.phone_seen_at = 0.0
        self.faces, self.poses, self.phone_boxes = [], [], []

    def _load_phone(self):
        if self.phone is None and self.phone_error is None:
            try:
                from phone_detect import PhoneDetector
                self.phone = PhoneDetector()
            except Exception as e: # ultralytics missing, no internet, etc.
                self.phone_error = str(e)

    # ---- 1. run the heavy detectors (not on every frame) ----
    def detect(self, frame_bgr, use_pose, use_phone):
        h, w = frame_bgr.shape[:2]
        n = self.frame_i
        self.frame_i += 1
        if n % config.DETECT_EVERY_N_FRAMES == 0:
            ts = max(int(time.monotonic() * 1000), self._ts + 1) # must keep increasing
            self._ts = ts
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.faces = []
            res = self.face.detect(mp_img)
            blends = res.face_blendshapes or []
            for i, lms in enumerate(res.face_landmarks or []):
                yaw, pitch, nose_px = face_metrics(lms, w, h)
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                emo, conf = None, 0.0
                if i < len(blends) and (bbox[2] - bbox[0]) >= config.EXPR_MIN_FACE_PX:
                    emo, conf = score_expression({c.category_name: c.score for c in blends[i]})
                self.faces.append({"yaw": yaw, "pitch": pitch, "bbox": bbox, "nose_px": nose_px, "emotion": emo, "emo_conf": conf})
            self.poses = []
            if use_pose:
                pose_res = self.pose.detect(mp_img)
                for lms in (pose_res.pose_landmarks or []):
                    self.poses.append({"nose": (lms[0].x * w, lms[0].y * h),
                                       "hand": hand_raised(lms)})
        if not use_pose:
            self.poses = []
        if not use_phone:
            self.phone_boxes = []
        elif n % config.PHONE_EVERY_N_FRAMES == 0:
            self._load_phone()
            if self.phone:
                boxes = self.phone.detect(frame_bgr)
                now = time.time()
                if boxes:
                    self.phone_boxes, self.phone_seen_at = boxes, now
                elif now - self.phone_seen_at > config.PHONE_HOLD_SEC:
                    self.phone_boxes = [] # a short miss keeps the old box

    # ---- 2. turn raw detections into students, levels and counts ----
    def evaluate(self, baseline, thr, question_active):
        yaw0, pitch0 = baseline
        raw = []
        for f in self.faces:
            raw.append({"bbox": f["bbox"], "nose": None,
                        "head": classify_head(f["yaw"], f["pitch"], yaw0, pitch0, thr),
                        "dyaw": abs(f["yaw"] - yaw0), "dpitch": abs(f["pitch"] - pitch0),
                        "hand": False, "phone": False})
        n_faces = len(raw)
        boxes = [s["bbox"] for s in raw]
        for p in self.poses:
            i = _match_face(p["nose"], boxes)
            if i is not None and i < n_faces:
                raw[i]["hand"] = raw[i]["hand"] or p["hand"]
            elif config.COUNT_HIDDEN_FACES:
                raw.append({"bbox": None, "nose": p["nose"], "head": "away",
                            "dyaw": 0, "dpitch": 0, "hand": p["hand"], "phone": False})
        assign_phones(raw, self.phone_boxes)
        students = self.tracker.update(raw) # steady, anonymous per-student state

        counts = {"engaged": 0, "partial": 0, "disengaged": 0, "hands": 0}
        for s in students:
            s["level"] = student_level(s["head"], s["phone"], s["hand"], question_active)
            counts[s["level"]] += 1
            counts["hands"] += int(s["hand"])
        counts["phones"] = len(self.phone_boxes)
        counts["total"] = len(students)
        counts["pct"] = class_percent(counts["engaged"], counts["partial"], counts["total"])
        return students, counts

    def calibrate(self):
        """Use the faces seen right now as the 'looking at the board' reference."""
        if not self.faces:
            return None
        n = len(self.faces)
        return (sum(f["yaw"] for f in self.faces) / n, sum(f["pitch"] for f in self.faces) / n)

    # ---- 3. draw boxes and labels ----
    def draw(self, frame, students, show_debug=False):
        for s in students:
            col = COLORS[s["level"]]
            label = f"S{s['id']} {s['level']}" + (" +hand" if s["hand"] else "") \
                + (" +phone" if s["phone"] else "")
            if s.get("emotion"):
                label += f" · {s['emotion']}"
            if s["bbox"] is not None:
                x0, y0, x1, y1 = map(int, s["bbox"])
                cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                if show_debug:
                    label += f" dy{s['dyaw']:.2f} dp{s['dpitch']:.2f}"
                cv2.putText(frame, label, (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            else:
                x, y = map(int, s["nose"])
                cv2.circle(frame, (x, y), 18, col, 2)
                cv2.putText(frame, f"S{s['id']} face hidden", (x - 45, max(15, y - 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        for b in self.phone_boxes:
            x0, y0, x1, y1 = map(int, b[:4])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 140, 60), 2)
            cv2.putText(frame, f"phone {b[4]:.2f}" if len(b) > 4 else "phone", (x0, max(15, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 60), 1, cv2.LINE_AA)
        return frame
