"""
models.py – ensures task model assets exist locally in models/
"""
import urllib.request
from pathlib import Path

import config

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


def ensure_models() -> dict[str, str]:
    """Verify models exist on disk and return paths map."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for name, url in MODELS.items():
        p = config.MODEL_DIR / name
        if not p.exists():
            # if full pose doesn't exist, check if lite is present
            if name == "pose_landmarker_full.task" and (config.MODEL_DIR / "pose_landmarker_lite.task").exists():
                paths[name] = str(config.MODEL_DIR / "pose_landmarker_lite.task")
                continue
            try:
                urllib.request.urlretrieve(url, str(p))
            except Exception:
                if name == "pose_landmarker_full.task" and (config.MODEL_DIR / "pose_landmarker_lite.task").exists():
                    paths[name] = str(config.MODEL_DIR / "pose_landmarker_lite.task")
                    continue
        paths[name] = str(p if p.exists() else (config.MODEL_DIR / "pose_landmarker_lite.task"))

    return paths
