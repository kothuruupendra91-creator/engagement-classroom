import threading
import time
from collections import deque

import cv2

import config
import database as db
from events import EventDetector
from recorder import blur_faces


class LiveProcessor:
    """Everything that happens for each camera frame.

    The local camera loop and the browser (WebRTC) callback both call process(frame),
    so the two camera sources behave identically.
    """

    def __init__(self, analyzer, recorder, slots=None, token="local"):
        self.analyzer, self.recorder = analyzer, recorder
        self.slots, self.token = (slots if slots is not None else {}), token
        self.thr = {"yaw_ok": config.YAW_OK, "yaw_away": config.YAW_AWAY,
                    "pitch_ok": config.PITCH_OK, "pitch_away": config.PITCH_AWAY}
        self.use_pose = self.use_phone = True
        self.show_debug = self.rec_on = False
        self.blur = True
        self.baseline = (0.0, config.DEFAULT_PITCH)
        self.calibrate_flag, self.q_until, self.last_log = False, 0.0, 0.0
        self.history, self.events = deque(maxlen=config.SMOOTH_WINDOW), EventDetector()
        self.session_id, self.active = None, False
        self.fps, self.t_last = 0.0, time.time()
        self.rec_error = False
        self.t_start, self.trend, self.t_trend = time.time(), deque(maxlen=120), 0.0
        self.lock = threading.Lock()
        self.stats = {"counts": None, "pct": 0.0, "fps": 0.0, "rec": False, "phone_error": None,
                      "elapsed": 0.0, "trend": []}

    # ---- controls (called from the Streamlit script) ----
    def configure(self, thr, use_pose, use_phone, show_debug, rec_on, blur):
        self.thr, self.use_pose, self.use_phone = thr, use_pose, use_phone
        self.show_debug, self.rec_on, self.blur = show_debug, rec_on, blur

    def start_session(self, session_id):
        self.session_id, self.active = session_id, True
        self.t_start, self.t_trend = time.time(), 0.0
        self.trend.clear()
        self.history.clear()
        self.events = EventDetector()
        self.analyzer.reset()
        self.last_log = 0.0
        self.slots[self.token] = time.time()
        db.log_event(session_id, "session_start", "Session started")

    def end_session(self):
        if not self.active:
            return
        self.active = False
        self._close_recording()
        db.log_event(self.session_id, "session_end", "Session ended")
        self.slots.pop(self.token, None)

    def ask_question(self):
        self.q_until = time.time() + config.QUESTION_WINDOW_SEC
        db.log_event(self.session_id, "question_asked", "Teacher opened a hand-raise window")

    def request_calibration(self):
        self.calibrate_flag = True

    def snapshot(self):
        with self.lock:
            return dict(self.stats)

    def _close_recording(self):
        if self.recorder.active and self.recorder.session_id == self.session_id:
            self.recorder.close()

    # ---- one frame in, annotated frame out ----
    def process(self, frame):
        if not self.active:
            return frame
        now = time.time()
        self.slots[self.token] = now
        a = self.analyzer
        clean = frame.copy() if self.rec_on else None
        a.detect(frame, self.use_pose, self.use_phone)
        if self.calibrate_flag:
            base = a.calibrate()
            if base:
                self.baseline, self.calibrate_flag = base, False

        q_left = self.q_until - now
        students, counts = a.evaluate(self.baseline, self.thr, q_left > 0)
        if counts["total"] > 0:
            self.history.append(counts["pct"])
        smooth = sum(self.history) / len(self.history) if self.history else 0.0

        new_events = []
        if counts["total"] > 0:
            new_events = self.events.update(smooth, counts["hands"], counts["phones"], now, q_left > 0)
            if now - self.last_log >= config.LOG_INTERVAL_SEC:
                db.log_reading(self.session_id, counts, round(smooth, 1))
                self.last_log = now

        a.draw(frame, students, self.show_debug)
        if q_left > 0:
            cv2.putText(frame, f"QUESTION: counting hands {q_left:.0f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        images = {}
        if self.rec_on:
            r = self.recorder
            if not r.active or r.session_id != self.session_id:
                if not r.start(self.session_id, clean):
                    self.rec_on, self.rec_error = False, True
            if self.rec_on:
                if self.blur:
                    blur_faces(clean, students)
                a.draw(clean, students)
                for kind, _ in new_events:
                    if kind in config.SNAPSHOT_EVENTS:
                        images[kind] = r.snapshot(clean, kind)
                r.write(clean, now)
        else:
            self._close_recording()
        for kind, detail in new_events:
            db.log_event(self.session_id, kind, detail, round(smooth, 1), images.get(kind))

        dt = max(now - self.t_last, 1e-3)
        self.t_last = now
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
        if counts["total"] > 0 and now - self.t_trend >= 1.0:
            self.trend.append(round(smooth, 1))
            self.t_trend = now
        with self.lock:
            self.stats = {"counts": counts, "pct": smooth, "fps": self.fps, "rec": self.rec_on,
                          "phone_error": a.phone_error, "elapsed": now - self.t_start, "trend": list(self.trend)}
        return frame
