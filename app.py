import os
import time
import uuid
from datetime import datetime

import altair as alt
import cv2
import pandas as pd
import streamlit as st

import config
import database as db
import settings
from analytics import reports_page
from emotion_page import analytics_view_page
from home_page import home_page
from settings_page import settings_page
from live_processor import LiveProcessor
from ui import (ESTIMATE_NOTE, EVENT_STYLE, acquire_slot, chip, get_camera_holder, get_recorder, get_slots,
                hero, inject, mmss, pct_colour, release_camera)

st.set_page_config(page_title="Classroom Engagement Analyzer", page_icon="🎓", layout="wide")
db.init_db()
inject()
ss = st.session_state


def new_session_id():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ss.setdefault("settings", settings.load())
ss.setdefault("session_id", new_session_id())
ss.setdefault("running", False)
ss.setdefault("token", uuid.uuid4().hex)


def open_camera(holder, index):
    cap = holder["cap"]
    if cap is not None and (holder.get("index") != index or not cap.isOpened()):
        cap.release()
        cap = None
    if cap is None:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # DSHOW = Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        holder["cap"], holder["index"] = cap, index
    return cap


def status_chips(cam_on, rec_on, blur, fps, elapsed=0, n=None):
    h = chip("CAMERA ON", "#22D3EE") if cam_on else chip("CAMERA OFF", "#64748B", "○")
    if cam_on:
        h += chip(f"⏱ {mmss(elapsed)}", "#94A3B8", "")
        h += chip(f"DETECTING · {n} student{'s' if n != 1 else ''}", "#4ADE80") if n else chip("NO FACES FOUND", "#FBBF24")
    if rec_on:
        h += chip("REC · faces pixelated" if blur else "REC · faces visible", "#F87171")
    if cam_on:
        h += chip(f"{fps:.0f} FPS", "#94A3B8", "◷")
    return h + chip("AI-estimated · anonymous", "#94A3B8", "ⓘ")


def feed_html(sid, n=6):
    ev = db.read_events(sid).tail(n).iloc[::-1]
    if ev.empty:
        return '<div class="note">No events yet.</div>'
    out = ""
    for _, e in ev.iterrows():
        label, icon, col = EVENT_STYLE.get(e["event_type"], (e["event_type"], "•", "#94A3B8"))
        out += (f'<div class="ev" style="--c:{col}"><div class="t">{str(e["ts"])[11:19]}</div>'
                f'<div><div class="k">{icon} {label}</div><div class="d">{e["detail"]}</div></div></div>')
    return out


def get_proc():
    """One LiveProcessor (and its own detector state) per visitor session."""
    if "proc" not in ss:
        from analyzer import Analyzer
        ss.proc = LiveProcessor(Analyzer(config.MAX_STUDENTS), get_recorder(), get_slots(), ss.token)
    return ss.proc


def rtc_config():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if os.environ.get("TURN_URL"):                  # optional relay server for strict networks
        servers.append({"urls": [os.environ["TURN_URL"]], "username": os.environ.get("TURN_USER", ""),
                        "credential": os.environ.get("TURN_PASS", "")})
    return {"iceServers": servers}


def draw_stats(w, snap, blur):
    c = snap["counts"]
    n = c["total"] if c else 0
    w["chips"].markdown(status_chips(True, snap["rec"], blur, snap["fps"], snap["elapsed"], n), unsafe_allow_html=True)
    if c is None:
        return
    w["pct"].markdown(
        f'<div class="card"><div class="lbl">Class engagement (AI estimate)</div>'
        f'<div class="val" style="color:{pct_colour(snap["pct"])}">{snap["pct"]:.0f}%</div></div>',
        unsafe_allow_html=True)
    w["engaged"].metric("🟢 Engaged", c["engaged"])
    w["partial"].metric("🟠 Partial", c["partial"])
    w["disengaged"].metric("🔴 Disengaged", c["disengaged"])
    w["hands"].metric("✋ Hands", c["hands"])
    w["phones"].metric("📱 Phones", c["phones"])
    w["total"].metric("👥 Students", c["total"])
    w["facing"].metric("👀 Facing board", f'{c["facing"]} of {c["total"]}')
    top = max(c["emotions"], key=c["emotions"].get) if c["emotions"] else None
    w["expr"].metric("🙂 Expression", top.capitalize() if top else "Not enough data")
    if len(snap["trend"]) > 2:
        w["trend"].line_chart(pd.DataFrame({"Engagement %": snap["trend"]}), height=150, color="#22D3EE")
    if snap["phone_error"]:
        w["note"].caption("phone detection unavailable: " + snap["phone_error"][:90])


# ------------------------------------------------------------------ LIVE PAGE
def live_page():
    S = ss.settings
    browser = S["camera_source"] == "browser"
    hero("Live dashboard", "Press Start session: the camera opens and detection begins automatically", "📡")
    sb = st.sidebar
    sb.markdown("### Session")
    if not ss.running:
        if sb.button("▶ Start session"):
            if not acquire_slot(ss.token):
                sb.error("The server is busy with other live sessions. Please try again in a minute.")
            else:
                ss.session_id = new_session_id()
                ss.running = True
                get_proc().start_session(ss.session_id)
    elif sb.button("⏹ End session"):
        ss.running = False
        get_proc().end_session()
    sb.caption(f"Session: {ss.session_id}" if ss.running else "No session running")

    rec_on, show_debug, use_pose, use_phone = False, False, True, True
    if ss.running:
        if sb.button(f"❓ Ask a question ({config.QUESTION_WINDOW_SEC}s)"):
            get_proc().ask_question()
        if sb.button("🎯 Calibrate (all look at board)"):
            get_proc().request_calibration()
        if browser:
            sb.caption("Recording is available only when running locally.")
        else:
            rec_on = sb.checkbox("⏺ Record video + key moments", value=S["record_default"], key="rec_on")
        with sb.expander("Features", expanded=False):
            use_pose = st.checkbox("Hand raise + hidden-face detection", True)
            use_phone = st.checkbox("Phone detection", True)
            show_debug = st.checkbox("Show debug numbers", False)
    sb.caption("Sensitivity, camera and privacy options are in ⚙️ Settings & privacy.")
    thr = {k: S[k] for k in ("yaw_ok", "yaw_away", "pitch_ok", "pitch_away")}
    config.PHONE_CONF = S["phone_conf"]

    chips_ph = st.empty()
    col_video, col_stats = st.columns([5, 2], gap="large")
    w = {"chips": chips_ph}
    with col_stats:
        w["pct"] = st.empty()
        for row in (("engaged", "partial"), ("disengaged", "hands"), ("phones", "total"), ("facing", "expr")):
            for k, c in zip(row, st.columns(2)):
                w[k] = c.empty()
        w["trend"] = st.empty()
        w["note"] = st.empty()
        st.caption(ESTIMATE_NOTE)

    if not ss.running:
        chips_ph.markdown(status_chips(False, False, True, 0), unsafe_allow_html=True)
        with col_video:
            st.info("Press **▶ Start session** in the sidebar. The camera starts automatically "
                    "and detection begins.")
            st.markdown('<div class="sec">Event timeline</div>', unsafe_allow_html=True)
            st.markdown(feed_html(ss.session_id), unsafe_allow_html=True)
        release_camera(get_camera_holder())
        return

    proc = get_proc()
    proc.configure(thr, use_pose, use_phone, show_debug, rec_on, S["pixelate"])
    ctx = video_ph = None
    with col_video:
        if browser:
            try:
                import av
                from streamlit_webrtc import WebRtcMode, webrtc_streamer
            except ImportError:
                st.error("Browser camera needs:  pip install streamlit-webrtc")
                return

            def on_frame(frame, _proc=proc):
                img = _proc.process(frame.to_ndarray(format="bgr24"))
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="engagement-cam", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config(),
                                  media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                                  video_frame_callback=on_frame, async_processing=True,
                                  desired_playing_state=True)
        else:
            video_ph = st.empty()
        st.markdown('<div class="sec">Event timeline</div>', unsafe_allow_html=True)
        feed_ph = st.empty()
    feed_ph.markdown(feed_html(ss.session_id), unsafe_allow_html=True)

    if browser:
        chips_ph.markdown(status_chips(False, False, True, 0), unsafe_allow_html=True)
        if not ctx.state.playing:
            col_video.caption("Allow camera access in your browser when asked. Detection starts as soon as it plays.")
        n = 0
        while ctx.state.playing:
            draw_stats(w, proc.snapshot(), S["pixelate"])
            n += 1
            if n % 4 == 0:
                feed_ph.markdown(feed_html(ss.session_id), unsafe_allow_html=True)
            time.sleep(0.25)
        return

    holder = get_camera_holder()
    cap = open_camera(holder, S["camera_index"])
    if not cap.isOpened():
        st.error("Cannot open the camera. Close other apps using it, or pick another camera in Settings.")
        return
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed.")
            break
        out = proc.process(frame)
        video_ph.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if proc.rec_error:
            st.warning("Could not start the video recorder.")
            proc.rec_error = False
        n += 1
        if n % 5 == 0:
            draw_stats(w, proc.snapshot(), S["pixelate"])
        if n % 10 == 0:
            feed_ph.markdown(feed_html(ss.session_id), unsafe_allow_html=True)


# --------------------------------------------------------------- RESULTS PAGE
def event_row(e, t0):
    label, icon, colour = EVENT_STYLE.get(e["event_type"], (e["event_type"], "•", "#94A3B8"))
    pct = "" if pd.isna(e["class_pct"]) else f'{e["class_pct"]:.0f}% engaged'
    return (f'<div class="ev" style="--c:{colour}"><div class="t">{mmss((e["ts"] - t0).total_seconds())}</div>'
            f'<div><div class="k">{icon} {label}</div><div class="d">{e["detail"]}</div></div>'
            f'<div class="p">{pct}</div></div>')


def set_seek(key, seconds):
    ss[key] = seconds


def recording_section(sid, ev, t0):
    st.markdown('<div class="sec">Recording and key moments</div>', unsafe_allow_html=True)
    recs = db.read_recordings(sid)
    if len(recs):
        recs = recs[recs["filename"].map(lambda f: (config.REC_DIR / f).exists())].reset_index(drop=True)
    if recs.empty:
        st.info("No video was saved for this session. Tick 'Record video + key moments' in the "
                "Live sidebar before you start.")
        return
    i = 0
    if len(recs) > 1:
        i = st.selectbox("Recording part", range(len(recs)),
                         format_func=lambda k: f"Part {k + 1} (started {recs.loc[k, 'start_ts'][11:19]})")
    r = recs.loc[i]
    path = config.REC_DIR / r["filename"]
    vstart = pd.to_datetime(r["start_ts"])
    vend = pd.to_datetime(r["end_ts"]) if pd.notna(r["end_ts"]) else pd.Timestamp.max
    key = f"seek_{sid}_{r['id']}"

    st.video(str(path), start_time=int(ss.get(key, 0)))
    b1, b2, b3 = st.columns(3)
    b1.button("⏮ From the start", key=f"start_{key}", on_click=set_seek, args=(key, 0))
    with open(path, "rb") as f:
        b2.download_button("⬇ Download video", f.read(), file_name=r["filename"], mime="video/mp4")
    if b3.button("🗑 Delete video and snapshots", key=f"del_{key}"):
        vids, imgs = db.delete_recordings(sid)
        for v in vids:
            (config.REC_DIR / v).unlink(missing_ok=True)
        for im in imgs:
            (config.REC_DIR / "snaps" / im).unlink(missing_ok=True)
        st.rerun()

    moments = ev[ev["image"].notna()] if len(ev) else ev
    cols = st.columns(4)
    shown = 0
    for _, e in moments.iterrows():
        img = config.REC_DIR / "snaps" / str(e["image"])
        off = (e["ts"] - vstart).total_seconds()
        if not img.exists() or off < 0 or e["ts"] > vend:
            continue
        with cols[shown % 4]:
            st.image(str(img), caption=f"{mmss(off)} · {EVENT_STYLE[e['event_type']][0]}",
                     use_container_width=True)
            st.button("▶ Play from here", key=f"play_{e['id']}", on_click=set_seek,
                      args=(key, max(int(off) - 3, 0)))
        shown += 1
    if shown == 0:
        st.caption("No key-moment snapshots in this recording yet. They are saved when a hand is raised, "
                   "a phone appears, or engagement changes while recording.")


def results_page():
    hero("Session history", "Pick a session, then click a key moment to watch it", "🎞")
    st.caption(ESTIMATE_NOTE)
    sessions = db.list_sessions()
    if not sessions:
        st.info("No data yet. Turn the camera on in the Live page for a few minutes first.")
        return
    recorder = get_recorder()
    if recorder.active:
        with st.spinner("Finishing the recording..."):
            recorder.close()
    sid = st.selectbox("Session", sessions)
    df = db.read_session(sid)
    ev = db.read_events(sid)
    df["ts"] = pd.to_datetime(df["ts"])
    ev["ts"] = pd.to_datetime(ev["ts"])
    t0 = min(df["ts"].min(), ev["ts"].min()) if len(ev) else df["ts"].min()
    df["elapsed"] = (df["ts"] - t0).dt.total_seconds()

    hi, lo = df.loc[df["class_pct"].idxmax()], df.loc[df["class_pct"].idxmin()]
    c = st.columns(5)
    c[0].metric("Duration", mmss((df["ts"].max() - t0).total_seconds()))
    c[1].metric("Average engagement", f"{df['class_pct'].mean():.0f}%")
    c[2].metric("Highest", f"{hi['class_pct']:.0f}%", f"at {mmss(hi['elapsed'])}", delta_color="off")
    c[3].metric("Lowest", f"{lo['class_pct']:.0f}%", f"at {mmss(lo['elapsed'])}", delta_color="off")
    n = ev["event_type"].value_counts() if len(ev) else {}
    c[4].metric("Hand raises / phones", f"{n.get('hand_raised', 0)} / {n.get('phone_detected', 0)}")

    # ---- engagement chart with event markers ----
    st.markdown('<div class="sec">Engagement over time, with events</div>', unsafe_allow_html=True)
    line = alt.Chart(df).mark_line(color="#22D3EE", strokeWidth=2.5, interpolate="monotone").encode(
        x=alt.X("ts:T", title="Time"),
        y=alt.Y("class_pct:Q", title="Engagement %", scale=alt.Scale(domain=[0, 100])))
    chart = line
    marks = ev[ev["event_type"].isin(["hand_raised", "phone_detected", "engagement_drop",
                                      "engagement_rise", "question_asked"])].copy()
    if len(marks):
        marks = pd.merge_asof(marks.sort_values("ts"),
                              df[["ts", "class_pct"]].rename(columns={"class_pct": "y"}).sort_values("ts"),
                              on="ts", direction="nearest")
        marks["Event"] = marks["event_type"].map(lambda k: EVENT_STYLE[k][0])
        marks["When"] = marks["ts"].map(lambda t: mmss((t - t0).total_seconds()))
        dom = [EVENT_STYLE[k][0] for k in EVENT_STYLE]
        rng = [EVENT_STYLE[k][2] for k in EVENT_STYLE]
        pts = alt.Chart(marks).mark_point(size=150, filled=True, opacity=0.95).encode(
            x="ts:T", y="y:Q", color=alt.Color("Event:N", scale=alt.Scale(domain=dom, range=rng)),
            tooltip=["When", "Event", "detail"])
        chart = line + pts
    st.altair_chart(chart.properties(height=340), use_container_width=True)

    recording_section(sid, ev, t0)

    # ---- time explorer ----
    st.markdown('<div class="sec">What was happening at a chosen moment</div>', unsafe_allow_html=True)
    stamps = [mmss(x) for x in df["elapsed"]]
    pick = st.select_slider("Move to a time (mm:ss)", options=list(range(len(df))),
                            format_func=lambda i: stamps[i])
    r = df.iloc[pick]
    m = st.columns(6)
    m[0].metric("Engagement", f"{r['class_pct']:.0f}%")
    m[1].metric("🟢 Engaged", int(r["engaged"]))
    m[2].metric("🟠 Partial", int(r["partial"]))
    m[3].metric("🔴 Disengaged", int(r["disengaged"]))
    m[4].metric("✋ Hands", int(r["hands"]))
    m[5].metric("📱 Phones", int(r["phones"]))
    if len(ev):
        near = ev[(ev["ts"] - r["ts"]).abs() <= pd.Timedelta(seconds=15)]
        if len(near):
            st.caption("Events within 15 seconds of this moment")
            st.markdown("".join(event_row(e, t0) for _, e in near.iterrows()), unsafe_allow_html=True)

    # ---- full event log ----
    st.markdown('<div class="sec">Event log</div>', unsafe_allow_html=True)
    if len(ev):
        kinds = [EVENT_STYLE[k][0] for k in EVENT_STYLE if k in set(ev["event_type"])]
        chosen = st.multiselect("Show", kinds, default=kinds)
        keep = ev[ev["event_type"].map(lambda k: EVENT_STYLE.get(k, (k,))[0]).isin(chosen)]
        st.markdown("".join(event_row(e, t0) for _, e in keep.iterrows()) or "No events selected.",
                    unsafe_allow_html=True)
    else:
        st.info("No events were recorded for this session.")

    # ---- other charts and downloads ----
    s = df.set_index("ts")
    rule = "1min" if (s.index.max() - s.index.min()).total_seconds() > 600 else "10s"
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="sec">Students per level</div>', unsafe_allow_html=True)
        st.line_chart(s[["engaged", "partial", "disengaged"]].resample(rule).mean().dropna(),
                      color=["#4ADE80", "#FBBF24", "#F87171"])
    with b:
        st.markdown('<div class="sec">Hands raised and phones</div>', unsafe_allow_html=True)
        st.line_chart(s[["hands", "phones"]].resample(rule).max().dropna(), color=["#22D3EE", "#F59E0B"])

    safe = sid.replace(":", "-").replace(" ", "_")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Readings CSV", df.to_csv(index=False), file_name=f"readings_{safe}.csv")
    d2.download_button("⬇ Events CSV", ev.to_csv(index=False), file_name=f"events_{safe}.csv")


PAGES = {"🏠 Home": home_page, "📡 Live dashboard": live_page, "📊 Analytics": analytics_view_page,
         "🎞 Session history": results_page, "📄 Reports": reports_page, "⚙️ Privacy & settings": settings_page}
st.sidebar.markdown("## 🎓 Engagement Analyzer")
choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed", key="nav")
if choice != "📡 Live dashboard":                 # camera light off, recording finished, when not on Live
    get_recorder().close()
    release_camera(get_camera_holder())
PAGES[choice]()
