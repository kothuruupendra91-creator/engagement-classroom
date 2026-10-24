import os
import urllib.request

import config

BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "face_landmarker.task": BASE + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": BASE + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": BASE + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


def ensure_models():
    """Download the MediaPipe model files we need (once); return their paths."""
    config.MODEL_DIR.mkdir(exist_ok=True)
    paths = {}
    for name in ("face_landmarker.task", config.POSE_MODEL):
        path = config.MODEL_DIR / name
        if not path.exists():
            if name == "pose_landmarker_full.task" and (config.MODEL_DIR / "pose_landmarker_lite.task").exists():
                paths[name] = str(config.MODEL_DIR / "pose_landmarker_lite.task")
                continue
            print(f"Downloading {name} ...")
            tmp = str(path) + ".part"
            try:
                urllib.request.urlretrieve(MODELS[name], tmp)
                os.replace(tmp, path)
            except Exception:
                if (config.MODEL_DIR / "pose_landmarker_lite.task").exists():
                    paths[name] = str(config.MODEL_DIR / "pose_landmarker_lite.task")
                    continue
                raise
        paths[name] = str(path)
    return paths
