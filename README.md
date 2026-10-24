# Classroom Engagement Analyzer

## Install (Windows, Python 3.11 64-bit)
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## Check
python test_camera.py (webcam opens, press Q to close)
python test_logic.py (must print ALL TESTS PASSED)

## Run
run_app.bat (or: venv\Scripts\activate then streamlit run app.py)

## Files
- app.py pages: Home, Live Dashboard, Session History, Privacy & Settings
- ui.py colours, CSS and HTML helpers
- analyzer.py camera frame -> students, levels, counts
- tracker.py keeps each student's state steady (anti-flicker)
- head_pose.py which way a head is turned
- hand_raise.py wrist above shoulder
- phone_detect.py YOLO phone detection
- scoring.py engaged / partial / disengaged and class %
- events.py time-stamped events for the history
- database.py SQLite storage
- models.py downloads the MediaPipe models
- config.py every setting (app name, thresholds, speed)
- .streamlit/config.toml theme colours
