import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from expression import score_expression
from hand_raise import hand_raised
from head_pose import classify_head, face_metrics
from models import ensure_models
from phone_detect import assign_phones
from scoring import class_percent, student_level
from tracker import StudentTracker

_PHONE = {"det": None, "err": None}     # one YOLO model shared by all sessions
_PHONE_LOCK = threading.Lock()

COLORS = {"engaged": (80, 200, 60), "partial": (0, 165, 255), "disengaged": (60, 60, 240)}


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
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.IMAGE, num_faces=max_students,
            output_face_blendshapes=True))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths[config.POSE_MODEL]),
            running_mode=vision.RunningMode.IMAGE, num_poses=max_students))
        self.phone = None
        self.phone_error = None
        self.phone_t = 0.0
        self.frame_i = 0
        self.faces, self.poses, self.phone_boxes = [], [], []
        self.tracker = StudentTracker()

    def reset(self):
        self.tracker = StudentTracker()

    def _load_phone(self):
        if self.phone is None and self.phone_error is None:
            with _PHONE_LOCK:
                if _PHONE["det"] is None and _PHONE["err"] is None:
                    try:
                        from phone_detect import PhoneDetector
                        _PHONE["det"] = PhoneDetector()
                    except Exception as e:      # ultralytics missing, no internet, etc.
                        _PHONE["err"] = str(e)
            self.phone, self.phone_error = _PHONE["det"], _PHONE["err"]

    # ---- 1. run the heavy detectors (not on every frame) ----
    def detect(self, frame_bgr, use_pose, use_phone):
        h, w = frame_bgr.shape[:2]
        n = self.frame_i
        self.frame_i += 1
        if n % config.DETECT_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.faces = []
            res = self.face.detect(mp_img)
            blends = res.face_blendshapes or []
            for i, lms in enumerate(res.face_landmarks):
                yaw, pitch, bbox = face_metrics(lms, w, h)
                emo, conf = None, 0.0
                if i < len(blends) and bbox[2] - bbox[0] >= config.EXPR_MIN_FACE_PX:
                    emo, conf = score_expression({c.category_name: c.score for c in blends[i]})
                self.faces.append({"yaw": yaw, "pitch": pitch, "bbox": bbox, "emotion": emo, "emo_conf": conf})
            self.poses = []
            if use_pose:
                for lms in self.pose.detect(mp_img).pose_landmarks:
                    self.poses.append({"nose": (lms[0].x * w, lms[0].y * h),
                                       "hand": hand_raised(lms)})
        if not use_pose:
            self.poses = []
        if not use_phone:
            self.phone_boxes = []
        elif n % config.PHONE_EVERY_N_FRAMES == 0:
            self._load_phone()
            if self.phone:
                with _PHONE_LOCK:
                    found = self.phone.detect(frame_bgr)
                now = time.time()
                if found:
                    self.phone_boxes, self.phone_t = found, now
                elif now - self.phone_t > config.PHONE_HOLD_SEC:
                    self.phone_boxes = []     # keep the last box briefly so it does not blink

    # ---- 2. raw detections -> steady students -> levels and counts ----
    def evaluate(self, baseline, thr, question_active):
        yaw0, pitch0 = baseline
        raw = []
        for f in self.faces:
            raw.append({"bbox": f["bbox"], "nose": None, "hidden": False,
                        "head": classify_head(f["yaw"], f["pitch"], yaw0, pitch0, thr),
                        "dyaw": abs(f["yaw"] - yaw0), "dpitch": abs(f["pitch"] - pitch0),
                        "hand": False, "phone": False,
                        "emotion": f.get("emotion"), "emo_conf": f.get("emo_conf", 0.0)})
        n_faces = len(raw)
        boxes = [s["bbox"] for s in raw]
        for p in self.poses:
            i = _match_face(p["nose"], boxes)
            if i is not None and i < n_faces:
                raw[i]["hand"] = raw[i]["hand"] or p["hand"]
            elif config.COUNT_HIDDEN_FACES:
                raw.append({"bbox": None, "nose": p["nose"], "hidden": True,
                            "head": "away", "dyaw": 0, "dpitch": 0,
                            "hand": p["hand"], "phone": False})
        assign_phones(raw, self.phone_boxes)
        students = self.tracker.update(raw)       # majority vote over recent frames

        counts = {"engaged": 0, "partial": 0, "disengaged": 0, "hands": 0, "phones": 0}
        for s in students:
            s["level"] = student_level(s["head"], s["phone"], s["hand"], question_active)
            counts[s["level"]] += 1
            counts["hands"] += int(s["hand"])
            counts["phones"] += int(s["phone"])
        counts["total"] = len(students)
        counts["pct"] = class_percent(counts["engaged"], counts["partial"], counts["total"])
        emo = {}
        for s in students:
            if s.get("emotion"):
                emo[s["emotion"]] = emo.get(s["emotion"], 0) + 1
        confs = [s["emo_conf"] for s in students if s.get("emotion")]
        counts["emotions"], counts["emo_valid"] = emo, len(confs)
        counts["emo_conf"] = sum(confs) / len(confs) if confs else 0.0
        counts["facing"] = sum(1 for s in students if s["head"] == "facing")
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
                cv2.putText(frame, label + " (face hidden)", (x - 40, max(15, y - 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        for x0, y0, x1, y1, conf in self.phone_boxes:
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 200, 0), 2)
            cv2.putText(frame, f"phone {conf:.2f}", (int(x0), max(15, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
        return frame
