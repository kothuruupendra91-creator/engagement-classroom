"""
tracker.py – associates per-frame detections with persistent student identities
and applies anti-flicker smoothing to hand and phone signals.

Public API
──────────
    StudentTracker
        .update(faces, poses, phone_boxes, baseline, thr, q_active)
            → (students: list[dict], counts: dict)
        .reset()

Each student dict contains:
    id, direction, hand, phone, level, bbox, nose

counts keys:
    total, engaged, partial, disengaged, hands, phones, pct
"""
from __future__ import annotations

import math
from typing import Optional

import config
from scoring import class_percent, student_level
from head_pose import classify_head

# ── anti-flicker hold lengths (frames) ──────────────────────────────────
# Derived from config so they are easy to tune.
_HAND_HOLD  = math.ceil(config.STABLE_FRAMES * config.HAND_HOLD_RATIO)   # ≈ 5
_PHONE_HOLD = math.ceil(config.STABLE_FRAMES * config.PHONE_HOLD_RATIO)  # ≈ 4
_MIN_HITS   = config.TRACK_MIN_HITS     # 3
_FORGET     = config.TRACK_FORGET_FRAMES # 30


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL TRACK
# ═══════════════════════════════════════════════════════════════════════════

_next_id = 0

def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def _make_track(bbox, nose, immediate: bool = False) -> dict:
    return {
        "id":         _new_id(),
        "bbox":       bbox,       # (x0,y0,x1,y1) or None for hidden bodies
        "nose":       nose,       # (x,y) pixel coords
        "hits":       _MIN_HITS if immediate else 1,
        "miss":       0,
        "direction":  "away",
        "yaw":        0.0,
        "pitch":      0.0,
        "hand_hold":  0,          # frames of hand-hold remaining
        "phone_hold": 0,
        "_hand":      False,
        "_phone":     False,
    }


def _is_stable(tk: dict) -> bool:
    return tk["hits"] >= _MIN_HITS


def _update_hold(tk: dict, key_hold: str, key_val: str, seen: bool, max_hold: int):
    """Generic hold-counter updater for hand or phone signal."""
    if seen:
        tk[key_val]  = True
        tk[key_hold] = max_hold
    else:
        if tk[key_hold] > 0:
            tk[key_val]  = True
            tk[key_hold] -= 1
        else:
            tk[key_val] = False


def _centroid(bbox) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pose_matches_face(nose_xy: tuple, face_bbox: tuple) -> bool:
    """
    True if the pose nose lies within roughly one face-width of the face centre.
    Horizontal-only: the body extends below the face, so we only gate on x.
    """
    x0, y0, x1, y1 = face_bbox
    face_cx = (x0 + x1) / 2.0
    face_w  = x1 - x0
    return abs(nose_xy[0] - face_cx) < face_w       # within ± 1 face-width of centre


# ═══════════════════════════════════════════════════════════════════════════
# TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class StudentTracker:
    """
    Simple nearest-neighbour tracker with anti-flicker smoothing.

    Face tracks and hidden-body tracks are stored separately so matching logic
    stays clean.
    """

    def __init__(self):
        self._face_tracks: list[dict] = []   # tracks that have a visible face
        self._body_tracks: list[dict] = []   # tracks for bodies without a face

    def reset(self):
        self._face_tracks.clear()
        self._body_tracks.clear()

    # ------------------------------------------------------------------
    def update(
        self,
        faces:       list[dict],
        poses:       list[dict],
        phone_boxes: list,
        baseline:    tuple[float, float],
        thr:         dict,
        q_active:    bool,
    ) -> tuple[list[dict], dict]:
        """
        Parameters
        ----------
        faces       dicts with {yaw, pitch, bbox, ...}
        poses       dicts with {nose:(x,y), hand:bool, bbox, ...}
        phone_boxes list of (x0,y0,x1,y1[,conf,...])
        baseline    (yaw_base, pitch_base) from calibration
        thr         {yaw_ok, yaw_away, pitch_ok, pitch_away}
        q_active    question window open?

        Returns
        -------
        students    list of per-student dicts for drawing / recording
        counts      {total, engaged, partial, disengaged, hands, phones, pct}
        """
        byw, bpitch = baseline
        MAX_DIST = config.FRAME_WIDTH * 0.4   # max match radius (pixels)

        # ── Step 1: greedy face → face-track matching ─────────────────
        used_tracks: set[int] = set()
        used_faces:  set[int] = set()

        if self._face_tracks and faces:
            for fi, f in enumerate(faces):
                fc = _centroid(f["bbox"])
                best_d, best_ti = float("inf"), -1
                for ti, tk in enumerate(self._face_tracks):
                    if ti in used_tracks or tk["bbox"] is None:
                        continue
                    d = _dist(fc, _centroid(tk["bbox"]))
                    if d < best_d:
                        best_d, best_ti = d, ti
                if best_ti >= 0 and best_d < MAX_DIST:
                    used_faces.add(fi)
                    used_tracks.add(best_ti)
                    tk = self._face_tracks[best_ti]
                    tk["bbox"]      = f["bbox"]
                    tk["nose"]      = f.get("nose_px")
                    tk["hits"]      = min(tk["hits"] + 1, _MIN_HITS * 10)
                    tk["miss"]      = 0
                    tk["yaw"]       = f["yaw"]
                    tk["pitch"]     = f["pitch"]
                    tk["direction"] = classify_head(f["yaw"], f["pitch"], byw, bpitch, thr)

        # ── Step 2: new tracks for unmatched faces ─────────────────────
        for fi, f in enumerate(faces):
            if fi in used_faces:
                continue
            tk = _make_track(f["bbox"], f.get("nose_px"))
            tk["yaw"]       = f["yaw"]
            tk["pitch"]     = f["pitch"]
            tk["direction"] = classify_head(f["yaw"], f["pitch"], byw, bpitch, thr)
            self._face_tracks.append(tk)
            used_tracks.add(len(self._face_tracks) - 1)

        # ── Step 3: increment miss for unmatched face tracks ───────────
        for ti, tk in enumerate(self._face_tracks):
            if ti not in used_tracks:
                tk["miss"] += 1

        # ── Step 4: match poses → face tracks (ALL tracks, not just stable) ──
        assigned_poses: set[int] = set()
        for tk in self._face_tracks:
            if tk["bbox"] is None:
                continue
            hand_seen = False
            for pi, pose in enumerate(poses):
                if pi in assigned_poses or pose["nose"] is None:
                    continue
                if _pose_matches_face(pose["nose"], tk["bbox"]):
                    hand_seen = pose["hand"]
                    assigned_poses.add(pi)
                    break
            _update_hold(tk, "hand_hold", "_hand", hand_seen, _HAND_HOLD)

        # ── Step 5: hidden-body tracks for unmatched poses ────────────
        for pi, pose in enumerate(poses):
            if pi in assigned_poses or not config.COUNT_HIDDEN_FACES:
                continue
            if pose["nose"] is None:
                continue
            nx, ny = pose["nose"]
            # try to match to an existing body track by proximity
            matched = None
            for bt in self._body_tracks:
                if bt["nose"] and _dist(bt["nose"], (nx, ny)) < MAX_DIST:
                    matched = bt
                    break
            if matched is None:
                matched = _make_track(None, (nx, ny), immediate=True)
                self._body_tracks.append(matched)
            matched["nose"] = (nx, ny)
            matched["miss"] = 0
            matched["hits"] = min(matched["hits"] + 1, _MIN_HITS * 10)
            _update_hold(matched, "hand_hold", "_hand", pose["hand"], _HAND_HOLD)

        # ── Step 6: assign phones to stable face tracks ───────────────
        for tk in self._face_tracks:
            if not _is_stable(tk) or tk["bbox"] is None:
                continue
            x0, _, x1, _ = tk["bbox"]
            phone_seen = False
            for pb in phone_boxes:
                px0, py0, px1 = pb[0], pb[1], pb[2]
                # horizontal 1-D IoU
                inter = max(0, min(x1, px1) - max(x0, px0))
                union = max(x1, px1) - min(x0, px0)
                if union > 0 and inter / union > 0.1:
                    phone_seen = True
                    break
            _update_hold(tk, "phone_hold", "_phone", phone_seen, _PHONE_HOLD)

        # ── Step 7: forget stale tracks ───────────────────────────────
        self._face_tracks = [tk for tk in self._face_tracks if tk["miss"] < _FORGET]
        self._body_tracks = [bt for bt in self._body_tracks if bt["miss"] < _FORGET]

        # ── Step 8: build student list and counts ─────────────────────
        students: list[dict] = []
        counts = dict(total=0, engaged=0, partial=0, disengaged=0,
                      hands=0, phones=0, pct=0.0)

        for tk in self._face_tracks + self._body_tracks:
            if not _is_stable(tk):
                continue
            lvl = student_level(
                tk["direction"], tk["_phone"], tk["_hand"], q_active)
            students.append({
                "id":        tk["id"],
                "direction": tk["direction"],
                "hand":      tk["_hand"],
                "phone":     tk["_phone"],
                "level":     lvl,
                "bbox":      tk["bbox"],
                "nose":      tk["nose"],
            })
            counts["total"]  += 1
            counts[lvl]      += 1
            if tk["_hand"]:
                counts["hands"]  += 1
            if tk["_phone"]:
                counts["phones"] += 1

        counts["pct"] = class_percent(
            counts["engaged"], counts["partial"], counts["total"])
        return students, counts
