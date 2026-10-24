import numpy as np

NOSE, L_EYE, R_EYE, CHIN = 1, 33, 263, 152


def face_metrics(lms, w, h):
    """Return (yaw, pitch, bbox) for one face. All in pixel space."""
    def pt(i):
        return np.array([lms[i].x * w, lms[i].y * h])

    nose, le, re, chin = pt(NOSE), pt(L_EYE), pt(R_EYE), pt(CHIN)
    eye_mid = (le + re) / 2
    eye_dist = np.linalg.norm(le - re) + 1e-6
    yaw = (nose[0] - eye_mid[0]) / eye_dist # sideways turn
    pitch = (nose[1] - eye_mid[1]) / (abs(chin[1] - eye_mid[1]) + 1e-6) # up/down tilt
    xs = [p.x * w for p in lms]
    ys = [p.y * h for p in lms]
    return float(yaw), float(pitch), (min(xs), min(ys), max(xs), max(ys))


def classify_head(yaw, pitch, yaw0, pitch0, thr):
    """'facing', 'slight' or 'away' compared with the calibrated baseline."""
    dy, dp = abs(yaw - yaw0), abs(pitch - pitch0)
    if dy <= thr["yaw_ok"] and dp <= thr["pitch_ok"]:
        return "facing"
    if dy <= thr["yaw_away"] and dp <= thr["pitch_away"]:
        return "slight"
    return "away"
