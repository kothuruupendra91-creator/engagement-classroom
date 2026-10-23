# Classroom Engagement Analyzer (Phase 1 & Phase 2)

An AI-powered real-time classroom student engagement detection dashboard built with OpenCV, MediaPipe, YOLOv8, and Streamlit.

## Features

- **Phase 1: Attention & Gaze Direction**
  - MediaPipe Face Landmarker tracks head orientation (yaw & pitch).
  - Classifies gaze into: *Engaged* (facing board), *Partial* (slightly turned), and *Disengaged* (head down or looking away).
  - Distance-invariant geometric ratio formulas.
  - Rolling smoothed class engagement percentage.

- **Phase 2: Hand Raise & Phone Detection**
  - MediaPipe Pose Landmarker detects raised hands (wrist above shoulder).
  - Question window mode ("Ask a question" 15s timer) to measure true response rate.
  - YOLOv8 cell phone detection flags phone usage.
  - Student identity tracking with anti-flicker state machine across frames.

- **Privacy & Safety First**
  - 100% anonymous — no facial recognition or persistent student identifiers.
  - Optional face pixelation/blurring on recorded video and key moment snapshots.
  - Raw camera frames are never saved to disk unless recording is explicitly enabled.

- **Session History & Analytics**
  - SQLite logging of engagement readings, events, and recordings.
  - Interactive Altair engagement timeline with clickable event markers.
  - Summary metrics: Average engagement, Peak minute, Lowest minute, Hand raises, and Phone detections.
  - Downloadable CSV data and MP4 video exports.

---

## Project Structure

```
engagement_analyzer/
├── app.py              # Streamlit dashboard (Live view + Session history)
├── analyzer.py         # Pipeline coordinator (Face, Pose, YOLO)
├── tracker.py          # Student tracking & anti-flicker state machine
├── head_pose.py        # Head pose geometry & MediaPipe Face Landmarker
├── hand_raise.py       # Hand raise detection & MediaPipe Pose Landmarker
├── scoring.py          # Engagement level rules & class percentage
├── events.py           # Real-time event detector (drops, rises, hands, phones)
├── recorder.py         # Video writer with face blurring & snapshot capture
├── database.py         # SQLite database management
├── config.py           # Configuration parameters and thresholds
├── test_logic.py       # Unit & logic verification tests
├── requirements.txt    # Python dependencies
├── START.bat           # One-click Windows launcher
└── models/             # MediaPipe task models
```

---

## Setup & Running

### 1. Requirements
- Python 3.10 – 3.14
- A standard webcam (index 0)

### 2. Installation
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Tests
```powershell
python test_logic.py
```

### 4. Launch Dashboard
```powershell
streamlit run app.py
```
Or double-click `START.bat`. Then open [http://localhost:8501](http://localhost:8501) in your browser.
