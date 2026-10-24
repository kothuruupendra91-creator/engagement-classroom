"""Rough facial-expression estimate from MediaPipe face blendshapes (heuristic, not a trained emotion model)."""
import config

EMOTIONS = ["neutral", "happy", "surprised", "sad", "angry", "fearful", "disgusted"]


def _avg(b, left, right):
    return (b.get(left, 0.0) + b.get(right, 0.0)) / 2


def _clip(x):
    return max(0.0, min(1.0, x))


def score_expression(b):
    """b = {blendshape name: 0..1}. Returns (label, confidence 0..1)."""
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    brow_up = max(b.get("browInnerUp", 0.0), _avg(b, "browOuterUpLeft", "browOuterUpRight"))
    wide = _avg(b, "eyeWideLeft", "eyeWideRight")
    scores = {
        "happy": smile + 0.2 * _avg(b, "cheekSquintLeft", "cheekSquintRight"),
        "surprised": 0.5 * brow_up + 0.3 * wide + 0.3 * b.get("jawOpen", 0.0),
        "sad": 0.7 * _avg(b, "mouthFrownLeft", "mouthFrownRight") + 0.3 * b.get("browInnerUp", 0.0),
        "angry": 0.6 * _avg(b, "browDownLeft", "browDownRight") + 0.25 * _avg(b, "eyeSquintLeft", "eyeSquintRight")
                 + 0.25 * _avg(b, "mouthPressLeft", "mouthPressRight"),
        "disgusted": 0.7 * _avg(b, "noseSneerLeft", "noseSneerRight") + 0.3 * _avg(b, "mouthUpperUpLeft", "mouthUpperUpRight"),
        "fearful": 0.3 * b.get("browInnerUp", 0.0) + 0.4 * wide + 0.3 * _avg(b, "mouthStretchLeft", "mouthStretchRight"),
    }
    scores = {k: _clip(v) for k, v in scores.items()}
    ranked = sorted(scores.values(), reverse=True)
    top = max(scores, key=scores.get)
    if scores[top] < config.EXPR_MIN:
        return "neutral", round(0.5 + 0.5 * (1 - scores[top] / config.EXPR_MIN), 2)
    return top, round(_clip(ranked[0] + 0.5 * (ranked[0] - ranked[1])), 2)
