"""
hand_raise.py – pose landmark processing.

Pure function (importable without MediaPipe, used by tests):
    hand_raised(landmarks, margin) → bool

PoseAnalyzer class (wraps MediaPipe Tasks API):
    process(bgr_frame)  → list[pose_dict]
    close()
"""
from __future__ import annotations

from pathlib import Path

import config

# ── MediaPipe pose landmark indices ──────────────────────────────────────
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_NOSE           = 0


# ═══════════════════════════════════════════════════════════════════════════
# PURE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def hand_raised(lms, margin: float = config.HAND_MARGIN) -> bool:
    """
    True if either wrist is clearly above its corresponding shoulder.

    In image coordinates y increases downward, so "above" means smaller y.
    The margin prevents tiny residual elevation from counting.
    """
    left_up  = lms[_LEFT_WRIST].y  < lms[_LEFT_SHOULDER].y  - margin
    right_up = lms[_RIGHT_WRIST].y < lms[_RIGHT_SHOULDER].y - margin
    return left_up or right_up


# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class PoseAnalyzer:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API, mediapipe ≥ 0.10 / 1.x).

    Prefers pose_landmarker_full.task; falls back to pose_landmarker_lite.task
    if the full model has not been downloaded.
    """

    def __init__(self, max_poses: int = config.MAX_STUDENTS):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        # resolve model path: prefer configured name, fall back to lite
        full = config.MODEL_DIR / config.POSE_MODEL
        lite = config.MODEL_DIR / "pose_landmarker_lite.task"
        if full.exists():
            model_path = full
        elif lite.exists():
            model_path = lite
        else:
            raise FileNotFoundError(
                f"Pose model not found in {config.MODEL_DIR}.\n"
                "Download pose_landmarker_lite.task from:\n"
                "  https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )

        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=max_poses,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)
        self._mp = mp

    def process(self, bgr_frame) -> list[dict]:
        """
        Detect all poses in the frame.

        Returns a list of dicts, one per person:
            { nose:(x,y), hand:bool, bbox:(x0,y0,x1,y1), landmarks }
        """
        import numpy as np

        h, w = bgr_frame.shape[:2]
        rgb  = bgr_frame[:, :, ::-1]
        img  = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        res   = self._lm.detect(img)
        poses = []
        if not res.pose_landmarks:
            return poses

        for lm_list in res.pose_landmarks:
            nose_lm = lm_list[_NOSE]
            nose    = (int(nose_lm.x * w), int(nose_lm.y * h))
            raised  = hand_raised(lm_list)
            xs   = [lm.x * w for lm in lm_list]
            ys   = [lm.y * h for lm in lm_list]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            poses.append({
                "nose":      nose,
                "hand":      raised,
                "bbox":      bbox,
                "landmarks": lm_list,
            })
        return poses

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass
