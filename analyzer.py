"""
analyzer.py – orchestrates the detection pipeline.

Analyzer
    .detect(frame, use_pose=True, use_phone=True)
        Runs FaceLandmarker, PoseLandmarker, YOLO on the frame.
        Results stored in self.faces / self.poses / self.phone_boxes.

    .evaluate(baseline, thr, q_active)
        Feeds stored results through StudentTracker.
        Returns (students, counts).

    .calibrate()
        Returns (yaw, pitch) baseline from the current frame, or None.

    .draw(frame, students, show_debug=False)
        Draws bounding boxes and labels on the frame in-place.

    .reset()
        Clears tracker state.

    .phone_error : str | None
        Set if YOLO could not be loaded or run.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from tracker import StudentTracker

# ── colour palette (BGR) ─────────────────────────────────────────────────
_CLR = {
    "engaged":    (63,  185, 80),    # green
    "partial":    (34,  153, 210),   # amber-ish (BGR: 34,153,210 ≈ orange-blue but clear)
    "disengaged": (73,  81,  248),   # red
    "phone":      (34,  153, 210),   # amber
    "hand":       (255, 210, 34),    # yellow
    "hud":        (255, 255, 255),   # white
}

# actually let's use clearer colours in BGR
_CLR = {
    "engaged":    ( 80, 185,  63),   # #3FB950 green
    "partial":    ( 34, 153, 210),   # #D29922 amber → BGR
    "disengaged": ( 73,  81, 248),   # #F85149 red → BGR
    "phone":      (  0, 165, 255),   # orange
    "hand":       (  0, 215, 255),   # gold
    "hud":        (255, 255, 255),
}


class Analyzer:
    """
    Full detection pipeline.
    All heavy models are loaded lazily on first use (initialised once and reused).
    """

    def __init__(self, max_students: int = config.MAX_STUDENTS):
        from head_pose import FaceAnalyzer
        from hand_raise import PoseAnalyzer

        self.tracker      = StudentTracker()
        self._face        = FaceAnalyzer(max_students)
        self._pose        = PoseAnalyzer(max_students)
        self._yolo        = None          # loaded on first call with use_phone=True
        self.phone_error: Optional[str] = None

        # detection results (set by detect, can be set directly in tests)
        self.faces:       list[dict] = []
        self.poses:       list[dict] = []
        self.phone_boxes: list       = []

        self._frame_n   = 0
        self._cached_phones: list = []

    # ──────────────────────────────────────────────────────────────────
    def detect(self, frame, use_pose: bool = True, use_phone: bool = True):
        """
        Run the detection pipeline on one BGR frame.
        Respects DETECT_EVERY_N_FRAMES and PHONE_EVERY_N_FRAMES throttling.
        """
        self._frame_n += 1

        if self._frame_n % config.DETECT_EVERY_N_FRAMES == 0:
            self.faces = self._face.process(frame)
            if use_pose:
                self.poses = self._pose.process(frame)
            else:
                self.poses = []

        if use_phone:
            if self._frame_n % config.PHONE_EVERY_N_FRAMES == 0:
                self._cached_phones = self._detect_phones(frame)
            self.phone_boxes = self._cached_phones
        else:
            self.phone_boxes = []

    # ──────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        baseline: tuple[float, float],
        thr:      dict,
        q_active: bool,
    ) -> tuple[list[dict], dict]:
        """
        Feed current detections through the tracker.
        Returns (students, counts).
        """
        return self.tracker.update(
            faces       = self.faces,
            poses       = self.poses,
            phone_boxes = self.phone_boxes,
            baseline    = baseline,
            thr         = thr,
            q_active    = q_active,
        )

    # ──────────────────────────────────────────────────────────────────
    def calibrate(self) -> Optional[tuple[float, float]]:
        """Return (yaw, pitch) baseline from the current faces, or None."""
        return self._face.calibrate(self.faces)

    # ──────────────────────────────────────────────────────────────────
    def reset(self):
        """Clear all tracker state (e.g. on session change)."""
        self.tracker.reset()
        self._frame_n = 0
        self.faces = []
        self.poses = []
        self.phone_boxes = []
        self._cached_phones = []

    # ──────────────────────────────────────────────────────────────────
    def draw(self, frame, students: list[dict], show_debug: bool = False):
        """Draw bounding boxes and labels on the frame in-place."""
        for s in students:
            lvl   = s["level"]
            colour = _CLR.get(lvl, _CLR["hud"])

            # face bounding box
            if s["bbox"] is not None:
                x0, y0, x1, y1 = s["bbox"]
                cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)

                # label above box
                tag = lvl[0].upper()
                if s["hand"]:
                    tag += " ✋"
                if s["phone"]:
                    tag += " 📱"
                cv2.putText(frame, tag, (x0, max(y0 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                            cv2.LINE_AA)

                if show_debug and s["nose"]:
                    nx, ny = s["nose"]
                    cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)

            elif s["nose"] is not None:
                # hidden body – draw a small marker at the nose point
                nx, ny = s["nose"]
                cv2.circle(frame, (nx, ny), 8, _CLR["disengaged"], 2)
                cv2.putText(frame, "hidden", (nx + 10, ny),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            _CLR["disengaged"], 1, cv2.LINE_AA)

        # phone boxes
        for pb in self.phone_boxes:
            x0, y0, x1, y1 = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
            cv2.rectangle(frame, (x0, y0), (x1, y1), _CLR["phone"], 2)
            cv2.putText(frame, "phone", (x0, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR["phone"], 1)

    # ──────────────────────────────────────────────────────────────────
    def _detect_phones(self, frame) -> list:
        """
        Run YOLOv8 phone detection. Returns list of (x0,y0,x1,y1,conf).
        Loads the model once; sets self.phone_error on failure.
        """
        if self.phone_error:          # already failed before, skip
            return []

        if self._yolo is None:
            try:
                from ultralytics import YOLO as _YOLO
                # prefer configured model; fall back to yolov8n.pt if not found
                model_path = config.MODEL_DIR / config.PHONE_MODEL
                if not model_path.exists():
                    model_path = Path(config.PHONE_MODEL)        # let ultralytics auto-download
                    if not model_path.exists():
                        # last fallback: yolov8n.pt in project root
                        model_path = Path("yolov8n.pt")
                self._yolo = _YOLO(str(model_path))
            except Exception as exc:
                self.phone_error = str(exc)
                return []

        try:
            results = self._yolo(
                frame,
                classes=[67],                 # COCO class 67 = "cell phone"
                conf=config.PHONE_CONF,
                imgsz=config.PHONE_IMGSZ,
                verbose=False,
            )
            boxes = []
            for r in results:
                for box in r.boxes:
                    x0, y0, x1, y1 = box.xyxy[0].cpu().numpy()
                    boxes.append((float(x0), float(y0), float(x1), float(y1),
                                  float(box.conf)))
            return boxes
        except Exception as exc:
            self.phone_error = str(exc)
            return []
