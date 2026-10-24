"""
head_pose.py – face landmark processing.

Pure functions (importable without MediaPipe, used by tests):
    face_metrics(landmarks, width, height)  → (yaw, pitch, nose_px)
    classify_head(yaw, pitch, base_yaw, base_pitch, thresholds)  → str

FaceAnalyzer class (wraps MediaPipe Tasks API):
    process(bgr_frame)  → list[face_dict]
    calibrate(faces)    → (yaw, pitch) | None
    close()
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import config

# ── MediaPipe face landmark indices ──────────────────────────────────────
_LEFT_EYE  = 33
_RIGHT_EYE = 263
_NOSE_TIP  = 1
_CHIN      = 152


# ═══════════════════════════════════════════════════════════════════════════
# PURE FUNCTIONS  (no import of mediapipe required)
# ═══════════════════════════════════════════════════════════════════════════

def face_metrics(lms, width: int, height: int) -> tuple[float, float, tuple[int, int]]:
    """
    Compute yaw and pitch ratios from normalised face landmarks.

    Yaw   = |nose_x − eye_centre_x| / eye_separation
            → 0 when looking straight, grows as head turns sideways.

    Pitch = (nose_y − eye_centre_y) / (chin_y − eye_centre_y)
            → ~0.35 at rest; grows when looking up, shrinks when looking down.

    Returns (yaw_ratio, pitch_ratio, nose_pixel_xy).
    """
    le   = lms[_LEFT_EYE]
    re   = lms[_RIGHT_EYE]
    nose = lms[_NOSE_TIP]
    chin = lms[_CHIN]

    eye_cx = (le.x + re.x) / 2.0
    eye_cy = (le.y + re.y) / 2.0

    eye_dist = abs(re.x - le.x)
    if eye_dist < 1e-6:
        eye_dist = 1e-6

    yaw = abs(nose.x - eye_cx) / eye_dist

    chin_dist = abs(chin.y - eye_cy)
    if chin_dist < 1e-6:
        chin_dist = 1e-6
    pitch = (nose.y - eye_cy) / chin_dist

    nose_px = (int(nose.x * width), int(nose.y * height))
    return float(yaw), float(pitch), nose_px


def classify_head(
    yaw:         float,
    pitch:       float,
    baseline_yaw:   float,
    baseline_pitch: float,
    thr:         dict,
) -> str:
    """
    Compare yaw/pitch against calibrated baseline and classify gaze direction.

    Returns one of: "facing", "slight", "away".
    """
    yaw_delta   = abs(yaw   - baseline_yaw)
    pitch_delta = abs(pitch - baseline_pitch)

    if yaw_delta >= thr["yaw_away"] or pitch_delta >= thr["pitch_away"]:
        return "away"
    if yaw_delta >= thr["yaw_ok"] or pitch_delta >= thr["pitch_ok"]:
        return "slight"
    return "facing"


# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class FaceAnalyzer:
    """Wraps MediaPipe FaceLandmarker (Tasks API, mediapipe ≥ 0.10 / 1.x)."""

    def __init__(self, max_faces: int = config.MAX_STUDENTS):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        model_path = config.MODEL_DIR / "face_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}.\n"
                "Download it from:\n"
                "  https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    def process(self, bgr_frame) -> list[dict]:
        """
        Detect faces and compute head-pose metrics.

        Returns a list of dicts, one per face:
            { yaw, pitch, bbox:(x0,y0,x1,y1), nose_px:(x,y), landmarks }
        """
        import numpy as np

        h, w = bgr_frame.shape[:2]
        rgb = bgr_frame[:, :, ::-1]
        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        res   = self._lm.detect(img)
        faces = []
        if not res.face_landmarks:
            return faces

        for lm_list in res.face_landmarks:
            yaw, pitch, nose_px = face_metrics(lm_list, w, h)
            xs = [lm.x * w for lm in lm_list]
            ys = [lm.y * h for lm in lm_list]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            faces.append({
                "yaw":       yaw,
                "pitch":     pitch,
                "bbox":      bbox,
                "nose_px":   nose_px,
                "landmarks": lm_list,
            })
        return faces

    def calibrate(self, faces: list) -> Optional[tuple[float, float]]:
        """Return (yaw, pitch) baseline from the first visible face, or None."""
        if not faces:
            return None
        return faces[0]["yaw"], faces[0]["pitch"]

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass
