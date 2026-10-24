"""Checks the rules with fake data. No camera or model download needed."""
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config
import database as db
from analyzer import Analyzer
from events import EventDetector
from hand_raise import hand_raised
from head_pose import classify_head, face_metrics
from scoring import class_percent, student_level
from tracker import StudentTracker

# --- scoring rules ---
assert student_level("facing", False, False, False) == "engaged"
assert student_level("facing", True, False, False) == "partial"
assert student_level("slight", False, False, False) == "partial"
assert student_level("away", True, False, False) == "disengaged"
assert student_level("away", False, True, True) == "engaged"
assert student_level("away", False, True, False) == "disengaged"
assert class_percent(2, 2, 4) == 75.0 and class_percent(0, 0, 0) == 0.0

# --- head direction with fake face points ---
P = lambda x, y, v=1.0: types.SimpleNamespace(x=x, y=y, visibility=v)


def fake_face(nose_dx=0.0):
    lms = [P(0.5, 0.5) for _ in range(478)]
    lms[33], lms[263] = P(0.45, 0.45), P(0.55, 0.45)
    lms[1], lms[152] = P(0.5 + nose_dx, 0.52), P(0.5, 0.65)
    return lms


thr = dict(yaw_ok=.12, yaw_away=.28, pitch_ok=.1, pitch_away=.2)
y, p, _ = face_metrics(fake_face(0.0), 640, 480)
assert classify_head(y, p, 0, p, thr) == "facing"
y, p2, _ = face_metrics(fake_face(0.02), 640, 480)
assert classify_head(y, p2, 0, p, thr) == "slight"
y, p3, _ = face_metrics(fake_face(0.04), 640, 480)
assert classify_head(y, p3, 0, p, thr) == "away"

# --- hand raise ---
pose = list([P(.5, .1)] * 33)
pose[11], pose[12], pose[15], pose[16] = P(.4, .5), P(.6, .5), P(.3, .2), P(.7, .7)
assert hand_raised(pose)
pose[15] = P(.3, .6)
assert not hand_raised(pose)

# --- full evaluation (3 faces + 1 hidden body + 1 phone), models not loaded ---
a = Analyzer.__new__(Analyzer)
a.tracker = StudentTracker()
a.faces = [{"yaw": 0.0, "pitch": p, "bbox": (100, 100, 200, 220)},
           {"yaw": 0.2, "pitch": p, "bbox": (300, 100, 400, 220)},
           {"yaw": 0.6, "pitch": p, "bbox": (500, 100, 600, 220)}]
a.poses = [{"nose": (150, 150), "hand": True}, {"nose": (50, 400), "hand": False}]
a.phone_boxes = [(320, 230, 360, 300, 0.8)]
for _ in range(4): # a face needs a few frames before it is counted
    students, c = a.evaluate((0.0, p), thr, True)
assert c["total"] == 4 and c["engaged"] == 1 and c["partial"] == 1 and c["disengaged"] == 2, c
assert c["phones"] == 1 and c["hands"] == 1, c
import numpy as np
a.draw(np.zeros((480, 640, 3), dtype="uint8"), students, True)

# --- anti-flicker: one bad frame must not change a student's state ---
a2 = Analyzer.__new__(Analyzer)
a2.tracker = StudentTracker()
a2.poses, a2.phone_boxes = [], []
box = (100, 100, 200, 220)
for i in range(12):
    yaw = 0.6 if i == 8 else 0.0 # frame 8: a one-frame glitch
    a2.faces = [{"yaw": yaw, "pitch": p, "bbox": box}]
    students, c = a2.evaluate((0.0, p), thr, False)
    if i >= 3:
        assert c["engaged"] == 1 and c["total"] == 1, (i, c)
a2.faces = [] # face briefly lost: still counted for a few frames
students, c = a2.evaluate((0.0, p), thr, False)
assert c["total"] == 1

# --- events ---
e = EventDetector()
assert e.update(80, 0, 0, 100.0, False) == []
out = e.update(50, 1, 1, 110.0, True)
kinds = [k for k, _ in out]
assert kinds == ["engagement_drop", "hand_raised", "phone_detected"], kinds
assert [k for k, _ in e.update(50, 1, 0, 111.0, False)] == ["phone_cleared"]

# --- database ---
config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "test.db"
db.init_db()
db.log_reading("s1", c, 55.0)
db.log_reading("s1", c, 60.0)
db.log_event("s1", "hand_raised", "1 hand(s) raised", 55.0)
assert db.list_sessions() == ["s1"] and len(db.read_session("s1")) == 2
assert len(db.read_events("s1")) == 1 and db.session_summary().iloc[0]["n_events"] == 1
db.delete_session("s1")
assert db.list_sessions() == []

print("ALL TESTS PASSED")
