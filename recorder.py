import os
import subprocess
import time

import cv2

import config
import database as db


def _pixelate(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    small = cv2.resize(frame[y0:y1, x0:x1], (max((x1 - x0) // 12, 2), max((y1 - y0) // 12, 2)))
    frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def blur_faces(frame, students):
    """Pixelate every face (and the head area of students whose face is hidden)."""
    for s in students:
        if s["bbox"] is not None:
            x0, y0, x1, y1 = s["bbox"]
            pw, ph = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            _pixelate(frame, x0 - pw, y0 - ph, x1 + pw, y1 + ph)
        elif s["nose"] is not None:
            x, y = s["nose"]
            _pixelate(frame, x - 60, y - 60, x + 60, y + 60)
    return frame


def _to_h264(path):
    """Browsers cannot play OpenCV's mp4v video, so convert it to H.264 if ffmpeg is available."""
    try:
        import imageio_ffmpeg
        out = path.with_name(path.stem + "_h264.mp4")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                            "-crf", "26", "-movflags", "+faststart", str(out)], capture_output=True)
        if r.returncode == 0 and out.exists():
            os.replace(out, path)
    except Exception:
        pass            # the mp4v file is still kept and can be downloaded


class Recorder:
    """Writes the annotated (optionally face-pixelated) video and key-moment snapshots."""

    def __init__(self):
        self.writer, self.session_id, self.rec_id = None, None, None
        self.path, self.t0, self.n = None, 0.0, 0

    @property
    def active(self):
        return self.writer is not None

    def _safe(self):
        return str(self.session_id).replace(":", "-").replace(" ", "_")

    def start(self, session_id, frame):
        self.close()
        h, w = frame.shape[:2]
        (config.REC_DIR / "snaps").mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        name = f"{self._safe()}_{int(time.time())}.mp4"
        self.path = config.REC_DIR / name
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      config.RECORD_FPS, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            return False
        self.t0, self.n = time.time(), 0
        self.rec_id = db.add_recording(session_id, name, config.RECORD_FPS)
        return True

    def write(self, frame, now):
        """Write frames so the video length matches real time (repeat the frame if we were slow)."""
        if not self.writer:
            return
        t = now - self.t0
        cv2.putText(frame, f"{int(t) // 60:02d}:{int(t) % 60:02d}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        due = int(t * config.RECORD_FPS) + 1
        for _ in range(min(max(due - self.n, 0), config.RECORD_FPS * 30)):
            self.writer.write(frame)
        self.n = max(self.n, due)

    def snapshot(self, frame, kind):
        name = f"{self._safe()}_{int(time.time() * 1000)}_{kind}.jpg"
        cv2.imwrite(str(config.REC_DIR / "snaps" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return name

    def close(self):
        if not self.writer:
            return
        self.writer.release()
        self.writer = None
        if self.rec_id:
            db.end_recording(self.rec_id)
        self.rec_id = None
        _to_h264(self.path)
