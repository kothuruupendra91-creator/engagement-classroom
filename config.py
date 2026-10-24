from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "engagement.db"
REC_DIR = BASE_DIR / "recordings"      # videos and snapshots (only if recording is switched on)

# ---- camera ----
CAMERA_INDEX = 0          # 0 = built-in webcam, 1 = USB webcam / phone camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_STUDENTS = 6

# ---- speed ----
DETECT_EVERY_N_FRAMES = 2   # face + pose
PHONE_EVERY_N_FRAMES = 5    # YOLO is the slowest part

# ---- anti-flicker (per student) ----
STABLE_FRAMES = 8           # a student's state = majority of the last N frames
HAND_HOLD_RATIO = 0.6       # hand counts as raised if up in 60% of the last N frames
PHONE_HOLD_RATIO = 0.4
TRACK_MIN_HITS = 3          # a face must be seen this many frames before it is counted
TRACK_HOLD_FRAMES = 8       # keep a student counted this long after detection drops out
TRACK_FORGET_FRAMES = 30    # forget a student after this many missed frames

# ---- behaviour ----
SMOOTH_WINDOW = 10
LOG_INTERVAL_SEC = 3
QUESTION_WINDOW_SEC = 15

# ---- events shown in the session history ----
EVENT_PCT_DELTA = 15        # log an event when class % moves this many points
EVENT_MIN_GAP_SEC = 5

# ---- head-direction limits (ratios, not degrees) ----
YAW_OK = 0.12
YAW_AWAY = 0.28
PITCH_OK = 0.10
PITCH_AWAY = 0.20
DEFAULT_PITCH = 0.45        # replaced when you press Calibrate

# ---- hand raise ----
HAND_MARGIN = 0.03
POSE_MODEL = "pose_landmarker_full.task"   # or "pose_landmarker_lite.task" (faster)

# ---- phone ----
PHONE_MODEL = "yolov8s.pt"   # use "yolov8n.pt" if too slow
PHONE_CONF = 0.25            # lower = detects more, more false alarms
PHONE_IMGSZ = 640
PHONE_HOLD_SEC = 1.5         # keep a phone box this long if one detection misses it
PHONE_ASSIGN_ANY = True      # a phone anywhere in view goes to the nearest student

COUNT_HIDDEN_FACES = True    # a body with no visible face counts as "disengaged"

# ---- recording (off by default, switched on from the Live sidebar) ----
RECORD_FPS = 10
SNAPSHOT_EVENTS = ("hand_raised", "phone_detected", "engagement_drop", "engagement_rise")

# ---- hosted (browser camera) use ----
MAX_LIVE_SESSIONS = 2        # most people who can run live detection at the same time

# ---- expression estimate ----
EXPR_MIN = 0.30             # below this score a face counts as neutral
EXPR_MIN_FACE_PX = 50       # smaller faces are "insufficient data" for expressions
