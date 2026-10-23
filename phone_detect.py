"""
phone_detect.py
===============
Phase 2 – Cell-phone detection using Ultralytics YOLOv8 nano.

The COCO class index for 'cell phone' is 67.
We run YOLO every 10th frame (controlled by caller) because it is
the heaviest step.  Results are cached between YOLO frames.

Phone-to-student matching uses simple bounding-box overlap:
  • extend each face box downward by 3× its own height
    (covers hands, desk, lap)
  • if the phone box centre falls inside that region → match
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

CELL_PHONE_CLASS = 67   # COCO index


@dataclass
class PhoneDetection:
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    matched_face_id: Optional[int] = None


class PhoneDetector:
    """Wraps YOLOv8n and matches phone boxes to face regions."""

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self._model = None
        self._last_detections: List[PhoneDetection] = []
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")   # downloaded automatically on first run
            self._available = True
        except Exception as e:
            print(f"[phone_detect] YOLO not available: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def process(
        self,
        bgr_frame: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[PhoneDetection], np.ndarray]:
        """
        Run YOLO on this frame, match phones to faces,
        return (detections, annotated_frame).
        """
        annotated = bgr_frame.copy()
        detections: List[PhoneDetection] = []

        if not self._available:
            return detections, annotated

        h, w = bgr_frame.shape[:2]

        try:
            results = self._model(
                bgr_frame,
                conf=self.conf_threshold,
                classes=[CELL_PHONE_CLASS],
                verbose=False,
            )
        except Exception:
            return detections, annotated

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                det  = PhoneDetection(box=(x1, y1, x2, y2), confidence=conf)

                # Match to a student face
                phone_cx = (x1 + x2) // 2
                phone_cy = (y1 + y2) // 2

                for fid, (fx1, fy1, fx2, fy2) in enumerate(face_bboxes):
                    face_h = max(fy2 - fy1, 1)
                    ext_y2 = fy2 + face_h * 3       # extend downward
                    ext_x1 = fx1 - face_h // 2
                    ext_x2 = fx2 + face_h // 2

                    if (ext_x1 <= phone_cx <= ext_x2 and
                            fy1 <= phone_cy <= ext_y2):
                        det.matched_face_id = fid
                        break

                detections.append(det)

                # Draw
                colour = (0, 0, 255) if det.matched_face_id is not None else (128, 128, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                label = f"Phone {conf:.0%}"
                if det.matched_face_id is not None:
                    label += f" → #{det.matched_face_id+1}"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        self._last_detections = detections
        return detections, annotated

    def get_cached(self) -> List[PhoneDetection]:
        """Return last YOLO result without re-running the model."""
        return self._last_detections
