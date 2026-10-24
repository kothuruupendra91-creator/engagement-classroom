import shutil

import cv2
import streamlit as st

import config
import database as db
import settings
from ui import ESTIMATE_NOTE, get_camera_holder, get_recorder, hero, release_camera

ss = st.session_state


def _mb(path):
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6 if path.exists() else 0.0


def settings_page():
    hero("Privacy & settings", "Camera, detection sensitivity, recording and stored data", "⚙️")
    S = ss.settings
    release_camera(get_camera_holder())          # the camera is only on while the Live page is open

    with st.form("settings"):
        st.markdown('<div class="sec">Camera</div>', unsafe_allow_html=True)
        src = st.selectbox("Camera source", ["local", "browser"], index=0 if S["camera_source"] == "local" else 1,
                           format_func=lambda x: "This computer's camera (run locally)" if x == "local"
                           else "Visitor's browser camera (for the hosted website)")
        cam = st.number_input("Camera index (0 = built-in, 1 = USB or phone camera)", 0, 5, int(S["camera_index"]))
        st.markdown('<div class="sec">Detection sensitivity</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        yaw_ok = a.slider("Facing: max sideways turn", 0.02, 0.40, float(S["yaw_ok"]), 0.01)
        yaw_away = b.slider("Away: sideways turn beyond", 0.05, 0.60, float(S["yaw_away"]), 0.01)
        pitch_ok = a.slider("Facing: max up/down tilt", 0.02, 0.40, float(S["pitch_ok"]), 0.01)
        pitch_away = b.slider("Away: up/down tilt beyond", 0.05, 0.60, float(S["pitch_away"]), 0.01)
        phone = st.slider("Phone sensitivity (lower finds more phones, but more false alarms)",
                          0.10, 0.60, float(S["phone_conf"]), 0.01)
        st.markdown('<div class="sec">Recording</div>', unsafe_allow_html=True)
        rec = st.checkbox("Tick 'Record video + key moments' by default on the Live page", S["record_default"])
        pix = st.checkbox("Pixelate faces in recordings (recommended)", S["pixelate"])
        if st.form_submit_button("💾 Save settings"):
            ss.settings = {"camera_source": src, "camera_index": int(cam), "yaw_ok": yaw_ok, "yaw_away": max(yaw_away, yaw_ok + 0.01),
                           "pitch_ok": pitch_ok, "pitch_away": max(pitch_away, pitch_ok + 0.01),
                           "phone_conf": phone, "record_default": rec, "pixelate": pix}
            settings.save(ss.settings)
            st.success("Settings saved. They apply the next time the camera starts.")

    c1, c2 = st.columns(2)
    if c1.button("🔍 Find connected cameras"):
        with st.spinner("Checking camera 0-3..."):
            found = []
            for i in range(4):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    found.append(i)
                cap.release()
        st.info(f"Working camera index: {', '.join(map(str, found))}" if found else "No camera found.")
    if c2.button("↺ Reset all settings to defaults"):
        ss.settings = dict(settings.DEFAULTS)
        settings.save(ss.settings)
        st.rerun()

    st.markdown('<div class="sec">Storage</div>', unsafe_allow_html=True)
    m = st.columns(3)
    m[0].metric("Videos and snapshots", f"{_mb(config.REC_DIR):.1f} MB")
    m[1].metric("Session database", f"{_mb(config.DB_PATH):.2f} MB")
    m[2].metric("Location", "This computer only")
    ok = st.checkbox("I understand deleted data cannot be recovered")
    d1, d2 = st.columns(2)
    if d1.button("🗑 Delete all videos and snapshots", disabled=not ok):
        get_recorder().close()
        shutil.rmtree(config.REC_DIR, ignore_errors=True)
        db.clear_recordings()
        st.success("All videos and snapshots deleted.")
    if d2.button("🗑 Delete ALL session data", disabled=not ok):
        get_recorder().close()
        shutil.rmtree(config.REC_DIR, ignore_errors=True)
        db.clear_all()
        st.success("All sessions, events and recordings deleted.")

    st.markdown('<div class="sec">Privacy notices</div>', unsafe_allow_html=True)
    cards = [("What is stored", "Time-stamped counts, percentages and events. No names and no face recognition. "
                                "Students only get temporary labels like S1, S2."),
             ("Recording", "Optional and off unless you switch it on. While recording, a red REC badge is shown on the "
                           "Live page. Files stay on this computer and can be deleted here or per session."),
             ("Faces", "Recordings pixelate faces by default. If you turn pixelation off, faces are saved, so tell "
                       "the class first and get permission."),
             ("About the score", ESTIMATE_NOTE)]
    cols = st.columns(4)
    for col, (t, p) in zip(cols, cards):
        col.markdown(f'<div class="priv"><h4>{t}</h4><p>{p}</p></div>', unsafe_allow_html=True)
