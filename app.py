import time
from collections import deque
from datetime import datetime

import altair as alt
import cv2
import pandas as pd
import streamlit as st

import config
import database as db
from events import EventDetector
from recorder import blur_faces

st.set_page_config(page_title="Classroom Engagement Analyzer", page_icon="🎓", layout="wide")
db.init_db()
ss = st.session_state

CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1450px}
#MainMenu, footer{visibility:hidden}
.hero{background:linear-gradient(135deg,#0E7490 0%,#2563EB 55%,#7C3AED 100%);
  padding:18px 26px;border-radius:18px;margin-bottom:16px}
.hero h1{margin:0;font-size:1.65rem;color:#fff;letter-spacing:.2px}
.hero p{margin:3px 0 0;color:#E0F2FE;font-size:.92rem}
[data-testid="stMetric"]{background:#111A2E;border:1px solid #1F2A44;border-radius:14px;padding:12px 16px}
[data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stMetricLabel"]{color:#94A3B8}
section[data-testid="stSidebar"]{background:#0D1526;border-right:1px solid #1F2A44}
div.stButton>button, div.stDownloadButton>button{width:100%;border-radius:10px;font-weight:600;
  border:1px solid #22D3EE55;background:#0F1B33;color:#E6EDF7;transition:all .15s}
div.stButton>button:hover, div.stDownloadButton>button:hover{background:#22D3EE;color:#06202A;border-color:#22D3EE}
.card{background:#111A2E;border:1px solid #1F2A44;border-radius:14px;padding:12px 16px}
.card .lbl{color:#94A3B8;font-size:.85rem}
.card .val{font-size:2.3rem;font-weight:700;line-height:1.15}
.ev{display:flex;gap:12px;align-items:flex-start;padding:10px 14px;margin-bottom:8px;
  background:#111A2E;border:1px solid #1F2A44;border-left:4px solid var(--c);border-radius:10px}
.ev .t{font-variant-numeric:tabular-nums;font-weight:700;color:#E6EDF7;min-width:56px}
.ev .k{font-size:.78rem;color:var(--c);font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.ev .d{color:#CBD5E1;font-size:.92rem}
.ev .p{margin-left:auto;color:#94A3B8;font-size:.85rem;white-space:nowrap}
.sec{font-size:1.05rem;font-weight:700;margin:18px 0 8px;color:#E6EDF7}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EVENT_STYLE = {   # type: (label, icon, colour)
    "session_start": ("Session started", "▶️", "#94A3B8"),
    "question_asked": ("Question asked", "❓", "#A78BFA"),
    "hand_raised": ("Hand raised", "✋", "#22D3EE"),
    "phone_detected": ("Phone detected", "📱", "#F59E0B"),
    "phone_cleared": ("Phone cleared", "✅", "#34D399"),
    "engagement_drop": ("Engagement drop", "📉", "#F87171"),
    "engagement_rise": ("Engagement rise", "📈", "#4ADE80"),
}


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>🎓 {title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def pct_colour(p):
    return "#4ADE80" if p >= 70 else "#FBBF24" if p >= 40 else "#F87171"


def new_session_id():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mmss(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


ss.setdefault("session_id", new_session_id())
ss.setdefault("baseline", (0.0, config.DEFAULT_PITCH))
ss.setdefault("q_until", 0.0)
ss.setdefault("calibrate", False)
ss.setdefault("history", deque(maxlen=config.SMOOTH_WINDOW))
ss.setdefault("last_log", 0.0)
ss.setdefault("events", EventDetector())
ss.setdefault("logged_start", None)


@st.cache_resource
def get_analyzer():
    from analyzer import Analyzer
    return Analyzer(config.MAX_STUDENTS)


@st.cache_resource
def get_recorder():
    from recorder import Recorder
    return Recorder()


@st.cache_resource
def get_camera_holder():
    return {"cap": None}


def open_camera(holder):
    cap = holder["cap"]
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)   # DSHOW = Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        holder["cap"] = cap
    return cap


def release_camera(holder):
    if holder["cap"] is not None:
        holder["cap"].release()
        holder["cap"] = None


# ------------------------------------------------------------------ LIVE PAGE
def live_page():
    hero("Classroom Engagement Analyzer", "Live view · anonymous · no video or faces are stored")
    sb = st.sidebar
    sb.markdown("### Controls")
    run = sb.toggle("Camera on", key="run")
    sb.caption(f"Session: {ss.session_id}")
    if sb.button("➕ Start new session"):
        ss.session_id = new_session_id()
        ss.history.clear()
        ss.events = EventDetector()
        if "analyzer" in ss:
            ss.analyzer.reset()
    if sb.button(f"❓ Ask a question ({config.QUESTION_WINDOW_SEC}s)"):
        ss.q_until = time.time() + config.QUESTION_WINDOW_SEC
        db.log_event(ss.session_id, "question_asked", "Teacher opened a hand-raise window")
    if sb.button("🎯 Calibrate (all look at board)"):
        ss.calibrate = True

    with sb.expander("Features", expanded=False):
        use_pose = st.checkbox("Hand raise + hidden-face detection", True)
        use_phone = st.checkbox("Phone detection", True)
        show_debug = st.checkbox("Show debug numbers", False)
    with sb.expander("Recording (off by default)", expanded=False):
        rec_on = st.checkbox("Record video + key moments", False)
        blur = st.checkbox("Pixelate faces in the recording", True)
        st.caption("Saved only on this computer, in the 'recordings' folder. Tell the class before "
                   "recording. With pixelation off, faces are stored in the video.")
    with sb.expander("Head-turn limits", expanded=False):
        yaw_ok = st.slider("Facing: max sideways turn", 0.02, 0.40, config.YAW_OK, 0.01)
        yaw_away = st.slider("Away: sideways turn beyond", 0.05, 0.60, config.YAW_AWAY, 0.01)
        pitch_ok = st.slider("Facing: max up/down tilt", 0.02, 0.40, config.PITCH_OK, 0.01)
        pitch_away = st.slider("Away: up/down tilt beyond", 0.05, 0.60, config.PITCH_AWAY, 0.01)
    # keep each "away" limit above its "facing" limit even if the sliders are dragged badly
    thr = {"yaw_ok": yaw_ok, "yaw_away": max(yaw_away, yaw_ok + 0.01),
           "pitch_ok": pitch_ok, "pitch_away": max(pitch_away, pitch_ok + 0.01)}

    col_video, col_stats = st.columns([3, 2], gap="large")
    video_ph = col_video.empty()
    with col_stats:
        pct_ph = st.empty()
        r1 = st.columns(3)
        r2 = st.columns(3)
        ph = {"engaged": r1[0].empty(), "partial": r1[1].empty(), "disengaged": r1[2].empty(),
              "hands": r2[0].empty(), "phones": r2[1].empty(), "total": r2[2].empty()}
        note_ph = st.empty()

    holder = get_camera_holder()
    recorder = get_recorder()
    if not run:
        recorder.close()
        release_camera(holder)
        col_video.info("Switch on **Camera on** in the sidebar to start.")
        return

    analyzer = get_analyzer()
    ss.analyzer = analyzer
    cap = open_camera(holder)
    if not cap.isOpened():
        st.error("Cannot open the camera. Close other apps using it, or change CAMERA_INDEX in config.py.")
        return

    if ss.logged_start != ss.session_id:
        db.log_event(ss.session_id, "session_start", "Session started")
        ss.logged_start = ss.session_id

    frame_n, t_prev, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed.")
            break

        clean = frame.copy() if rec_on else None
        analyzer.detect(frame, use_pose, use_phone)
        if ss.calibrate:
            base = analyzer.calibrate()
            if base:
                ss.baseline, ss.calibrate = base, False

        now = time.time()
        q_left = ss.q_until - now
        students, counts = analyzer.evaluate(ss.baseline, thr, q_left > 0)

        if counts["total"] > 0:
            ss.history.append(counts["pct"])
        smooth = sum(ss.history) / len(ss.history) if ss.history else 0.0

        new_events = []
        if counts["total"] > 0:
            new_events = ss.events.update(smooth, counts["hands"], counts["phones"], now, q_left > 0)
            if now - ss.last_log >= config.LOG_INTERVAL_SEC:
                db.log_reading(ss.session_id, counts, round(smooth, 1))
                ss.last_log = now

        analyzer.draw(frame, students, show_debug)
        if q_left > 0:
            cv2.putText(frame, f"QUESTION: counting hands {q_left:.0f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                       use_container_width=True)

        # ---- optional recording + key-moment snapshots ----
        images = {}
        if rec_on:
            if not recorder.active or recorder.session_id != ss.session_id:
                if not recorder.start(ss.session_id, clean):
                    rec_on = False
                    st.warning("Could not start the video recorder.")
            if rec_on:
                if blur:
                    blur_faces(clean, students)
                analyzer.draw(clean, students)
                for kind, _ in new_events:
                    if kind in config.SNAPSHOT_EVENTS:
                        images[kind] = recorder.snapshot(clean, kind)
                recorder.write(clean, now)
        elif recorder.active:
            recorder.close()
        for kind, detail in new_events:
            db.log_event(ss.session_id, kind, detail, round(smooth, 1), images.get(kind))

        frame_n += 1
        if frame_n % 5 == 0:
            fps = 5 / max(now - t_prev, 1e-6)
            t_prev = now
            pct_ph.markdown(
                f'<div class="card"><div class="lbl">Class engagement</div>'
                f'<div class="val" style="color:{pct_colour(smooth)}">{smooth:.0f}%</div></div>',
                unsafe_allow_html=True)
            ph["engaged"].metric("🟢 Engaged", counts["engaged"])
            ph["partial"].metric("🟠 Partial", counts["partial"])
            ph["disengaged"].metric("🔴 Disengaged", counts["disengaged"])
            ph["hands"].metric("✋ Hands", counts["hands"])
            ph["phones"].metric("📱 Phones", counts["phones"])
            ph["total"].metric("👥 Students", counts["total"])
            msg = f"{fps:.1f} FPS" + ("  |  ● REC" if rec_on else "")
            if use_phone and analyzer.phone_error:
                msg += "  |  phone detection unavailable: " + analyzer.phone_error[:90]
            note_ph.caption(msg)


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
    hero("Session history", "Pick a session to see what happened and when")
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


st.sidebar.markdown("## 🎓 Engagement Analyzer")
mode = st.sidebar.radio("Page", ["Live", "Results"], label_visibility="collapsed",
                        format_func=lambda x: "📡 Live" if x == "Live" else "📊 Session history")
live_page() if mode == "Live" else results_page()
