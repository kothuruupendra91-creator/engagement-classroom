"""
phone_detect.py
===============
Phase 2 – Cell-phone detection using Ultralytics YOLOv8.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import cv2
import numpy as np

import config

CELL_PHONE_CLASS = 67   # COCO index


def assign_phones(students: list, phone_boxes: list):
    """Assign phone detections to students by bounding box proximity."""
    if not phone_boxes or not students:
        return

    for pb in phone_boxes:
        px0, py0, px1, py1 = pb[:4]
        pcx, pcy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
        best_s, best_d = None, 1e18

        for s in students:
            if s.get("bbox") is not None:
                x0, y0, x1, y1 = s["bbox"]
                w, h = x1 - x0, y1 - y0
                # torso/desk region
                if (x0 - 0.5 * w <= pcx <= x1 + 0.5 * w) and (y0 <= pcy <= y1 + 3.0 * h):
                    d = ((x0 + x1) / 2.0 - pcx) ** 2 + ((y0 + y1) / 2.0 - pcy) ** 2
                    if d < best_d:
                        best_s, best_d = s, d
            elif s.get("nose") is not None:
                nx, ny = s["nose"]
                if abs(nx - pcx) <= 80:
                    d = (nx - pcx) ** 2 + (ny - pcy) ** 2
                    if d < best_d:
                        best_s, best_d = s, d

        if best_s:
            best_s["phone"] = True
        elif config.PHONE_ASSIGN_ANY and students:
            # assign to horizontally nearest student
            nearest = min(
                students,
                key=lambda s: abs((s["bbox"][0] + s["bbox"][2]) / 2.0 - pcx) if s.get("bbox") else abs(s["nose"][0] - pcx)
            )
            nearest["phone"] = True


class PhoneDetector:
    """Wraps YOLO model to detect cell phones."""

    def __init__(self):
        from ultralytics import YOLO
        model_name = config.PHONE_MODEL
        # try yolov8s.pt or fallback to yolov8n.pt if not present
        if not (config.MODEL_DIR / model_name).exists() and not (config.BASE_DIR / model_name).exists():
            if (config.BASE_DIR / "yolov8n.pt").exists():
                model_name = "yolov8n.pt"
        self._model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Run YOLO and return list of (x0, y0, x1, y1, conf)."""
        res = self._model(
            frame_bgr,
            classes=[CELL_PHONE_CLASS],
            conf=config.PHONE_CONF,
            imgsz=config.PHONE_IMGSZ,
            verbose=False,
        )
        boxes = []
        for r in res:
            for b in r.boxes:
                x0, y0, x1, y1 = map(float, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                boxes.append((x0, y0, x1, y1, conf))
        return boxes
