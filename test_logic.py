"""Checks the rules with fake data. No camera or model download needed."""
import pathlib
import sys
import tempfile
import time
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
a.phone_boxes = [(320, 230, 360, 300, 0.9)]
for _ in range(config.TRACK_MIN_HITS):          # a student is counted after a few frames
    students, c = a.evaluate((0.0, p), thr, True)
assert c["total"] == 4 and c["engaged"] == 1 and c["partial"] == 1 and c["disengaged"] == 2
assert c["phones"] == 1 and c["hands"] == 1

# --- flicker: one missed frame must not drop the raised hand ---
a.poses = [{"nose": (150, 150), "hand": False}, {"nose": (50, 400), "hand": False}]
students, c = a.evaluate((0.0, p), thr, True)
assert c["hands"] == 1
for _ in range(6):
    students, c = a.evaluate((0.0, p), thr, True)
assert c["hands"] == 0

# --- events ---
e = EventDetector()
assert e.update(80, 0, 0, 0.0, False) == []
kinds = [k for k, _ in e.update(50, 1, 1, 10.0, True)]
assert set(kinds) == {"engagement_drop", "hand_raised", "phone_detected"}
assert [k for k, _ in e.update(50, 1, 0, 11.0, True)] == ["phone_cleared"]

# --- database ---
config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "test.db"
db.init_db()
db.log_reading("s1", c, 55.0)
db.log_reading("s1", c, 60.0)
db.log_event("s1", "hand_raised", "1 hand(s) raised", 55.0)
assert db.list_sessions() == ["s1"] and len(db.read_session("s1")) == 2
assert len(db.read_events("s1")) == 1

# --- recorder: pixelated faces, video file, snapshot, database rows ---
import numpy as np
from recorder import Recorder, blur_faces

config.REC_DIR = pathlib.Path(tempfile.mkdtemp()) / "rec"
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
before = img[100:220, 100:200].copy()
blur_faces(img, [{"bbox": (100, 100, 200, 220), "nose": None}])
assert not np.array_equal(before, img[100:220, 100:200])
rec = Recorder()
assert rec.start("s1", img)
t = time.time()
for i in range(5):
    rec.write(img, t + i * 0.2)
snap = rec.snapshot(img, "hand_raised")
rec.close()
assert rec.path.exists() and rec.path.stat().st_size > 0
assert (config.REC_DIR / "snaps" / snap).exists()
r = db.read_recordings("s1")
assert len(r) == 1 and r.loc[0, "end_ts"]
vids, imgs = db.delete_recordings("s1")
assert len(vids) == 1 and db.read_recordings("s1").empty

# --- settings, analytics, report ---
import settings
from analytics import build_report, session_table
settings.save({**settings.DEFAULTS, "camera_index": 2})
assert settings.load()["camera_index"] == 2 and settings.load()["pixelate"] is True
db.log_reading("s2", c, 40.0)
rd, evs = db.read_all()
tb = session_table(rd, evs)
assert set(tb["Session"]) == {"s1", "s2"} and int(tb.loc[tb["Session"] == "s1", "Hand raises"].iloc[0]) == 1
page = build_report("s1", rd[rd["session_id"] == "s1"], evs[evs["session_id"] == "s1"])
assert "<polyline" in page and "AI-estimated" in page
db.clear_all()
assert db.read_all()[0].empty

# --- LiveProcessor: one frame in, annotated frame out, readings and events saved ---
from live_processor import LiveProcessor


class StubAnalyzer:                     # stands in for MediaPipe so no models are needed
    phone_error = None
    def reset(self): pass
    def detect(self, frame, use_pose, use_phone): pass
    def calibrate(self): return (0.0, 0.4)
    def draw(self, frame, students, debug=False): frame[0, 0] = 255; return frame
    def evaluate(self, baseline, thr, q):
        counts = {"engaged": 2, "partial": 1, "disengaged": 0, "hands": 1, "phones": 0, "total": 3, "pct": 83.0}
        return [], counts


config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "proc.db"
db.init_db()
slots = {}
proc = LiveProcessor(StubAnalyzer(), Recorder(), slots, "tok")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
assert proc.process(frame) is frame and proc.snapshot()["counts"] is None      # no session yet: untouched
proc.start_session("live1")
out = proc.process(frame)
assert out[0, 0].all() and "tok" in slots
snap = proc.snapshot()
assert snap["counts"]["total"] == 3 and round(snap["pct"]) == 83
proc.ask_question()
proc.request_calibration(); proc.process(frame)
assert proc.baseline == (0.0, 0.4)
proc.end_session()
assert "tok" not in slots
kinds = set(db.read_events("live1")["event_type"])
assert {"session_start", "hand_raised", "question_asked", "session_end"} <= kinds
assert len(db.read_session("live1")) == 1

# --- expression estimate, per-student smoothing, saved with each reading ---
from expression import score_expression
assert score_expression({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7})[0] == "happy"
assert score_expression({})[0] == "neutral" and score_expression({})[1] >= 0.5
assert score_expression({"browDownLeft": 0.9, "browDownRight": 0.9, "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5})[0] == "angry"
b = Analyzer.__new__(Analyzer)
b.tracker = StudentTracker()
b.faces = [{"yaw": 0.0, "pitch": p, "bbox": (100, 100, 200, 220), "emotion": "happy", "emo_conf": 0.8},
           {"yaw": 0.0, "pitch": p, "bbox": (300, 100, 400, 220), "emotion": None, "emo_conf": 0.0}]
b.poses, b.phone_boxes = [], []
for _ in range(config.TRACK_MIN_HITS):
    students, c2 = b.evaluate((0.0, p), thr, False)
assert c2["emotions"] == {"happy": 1} and c2["emo_valid"] == 1 and c2["facing"] == 2
db.log_reading("emo1", c2, 90.0)
from emotion_page import emotion_table
emo, reliable, conf = emotion_table(db.read_session("emo1"))
assert int(emo.loc[0, "happy"]) == 1 and bool(reliable.iloc[0])
from analytics import build_pdf
rd2, ev2 = db.read_all()
pdf = build_pdf("emo1", rd2[rd2["session_id"] == "emo1"], ev2[ev2["session_id"] == "emo1"])
assert pdf.startswith(b"%PDF")

print("ALL TESTS PASSED")
