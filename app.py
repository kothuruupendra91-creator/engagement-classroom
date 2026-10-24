import time
from collections import deque
from datetime import datetime

import altair as alt
import cv2
import pandas as pd
import streamlit as st

import config
import database as db
import ui
from events import EventDetector

st.set_page_config(page_title=config.APP_NAME, page_icon="\U0001F393", layout="wide")
ui.inject_css()
db.init_db()
ss = st.session_state


def new_session_id():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ss.setdefault("session_id", new_session_id())
ss.setdefault("baseline", (0.0, config.DEFAULT_PITCH))
ss.setdefault("q_until", 0.0)
ss.setdefault("calibrate", False)
ss.setdefault("history", deque(maxlen=config.SMOOTH_WINDOW))
ss.setdefault("trend", deque(maxlen=90))
ss.setdefault("last_log", 0.0)
ss.setdefault("last_pct", None)
ss.setdefault("events", EventDetector())


@st.cache_resource
def get_analyzer():
    from analyzer import Analyzer # loads the MediaPipe models once
    return Analyzer(config.MAX_STUDENTS)


@st.cache_resource
def get_camera_holder():
    return {"cap": None}


def open_camera(holder):
    cap = holder["cap"]
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW) # DSHOW = Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        holder["cap"] = cap
    return cap


def release_camera(holder=None):
    holder = holder or get_camera_holder()
    if holder["cap"] is not None:
        holder["cap"].release()
        holder["cap"] = None


def fmt_dur(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s"


# =================================================================== HOME
def home_page():
    release_camera()
    st.markdown(f'<div class="hero"><div class="hero-t">Understand classroom engagement<br>'
                f'with <span>responsible AI</span></div>'
                f'<div class="hero-s">{config.APP_NAME} estimates how attentive a class is, live, '
                f'without identifying anyone and without saving video.</div></div>',
                unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    if mid.button("Try demo", type="primary", icon=":material/play_arrow:", use_container_width=True):
        st.switch_page(LIVE_PAGE)
    st.markdown(
        '<div class="cards">'
        '<div class="card"><h4>Live Dashboard</h4><p>Camera view, engagement score, head direction, '
        'raised hands and phone detection in real time.</p></div>'
        '<div class="card"><h4>Session History</h4><p>Every session saved with a time-stamped timeline: '
        'when engagement changed, who raised a hand, when a phone appeared.</p></div>'
        '<div class="card"><h4>Privacy first</h4><p>No face recognition, no names, no video stored. '
        'Only counts and percentages are kept, and you can delete them anytime.</p></div></div>',
        unsafe_allow_html=True)


# =================================================================== LIVE
def live_page():
    st.markdown(ui.header_html("Live Dashboard", "Camera, detection status and current estimates",
                               live=bool(ss.get("run", False))), unsafe_allow_html=True)

    ctrl = st.columns([1.1, 1.7, 1.4, 1.4, 1.2])
    run = ctrl[0].toggle("Camera", key="run")
    if ctrl[1].button(f"Ask a question ({config.QUESTION_WINDOW_SEC}s)", type="primary",
                      icon=":material/quiz:", use_container_width=True):
        ss.q_until = time.time() + config.QUESTION_WINDOW_SEC
        db.log_event(ss.session_id, "question",
                     f"Question asked ({config.QUESTION_WINDOW_SEC}s window)", ss.last_pct)
    if ctrl[2].button("Calibrate", icon=":material/center_focus_strong:", use_container_width=True):
        ss.calibrate = True
    if ctrl[3].button("New session", icon=":material/add_circle:", use_container_width=True):
        ss.session_id = new_session_id()
        ss.history.clear()
        ss.trend.clear()
        ss.events = EventDetector()
        ss.last_pct = None
    with ctrl[4].popover("Settings", icon=":material/tune:", use_container_width=True):
        use_pose = st.checkbox("Hand raise + hidden-face detection", True)
        use_phone = st.checkbox("Phone detection", True)
        show_debug = st.checkbox("Show debug numbers", False)
        st.caption("Head-turn limits")
        thr = {
            "yaw_ok": st.slider("Facing: max sideways turn", 0.02, 0.40, config.YAW_OK, 0.01),
            "yaw_away": st.slider("Away: sideways turn beyond", 0.05, 0.60, config.YAW_AWAY, 0.01),
            "pitch_ok": st.slider("Facing: max up/down tilt", 0.02, 0.40, config.PITCH_OK, 0.01),
            "pitch_away": st.slider("Away: up/down tilt beyond", 0.05, 0.60, config.PITCH_AWAY, 0.01),
        }
    st.caption(f"Session started {ss.session_id}. Press Calibrate while everyone looks at the board.")

    left, right = st.columns([3, 2])
    video_ph = left.empty()
    gauge_ph = right.empty()
    kpi_ph = right.empty()
    note_ph = right.empty()
    warn_ph = right.empty()

    holder = get_camera_holder()
    if not run:
        release_camera(holder)
        video_ph.markdown(ui.idle_card("Camera is off. Turn on the Camera switch to start monitoring."),
                          unsafe_allow_html=True)
        gauge_ph.markdown(ui.gauge_html(0, list(ss.trend)), unsafe_allow_html=True)
        kpi_ph.markdown(ui.kpi_grid_html({}), unsafe_allow_html=True)
        return

    with st.spinner("Loading detection models (the first run downloads some files)..."):
        analyzer = get_analyzer()
        analyzer.reset()
        cap = open_camera(holder)
        if not cap.isOpened():
            st.error("Cannot open the camera. Close other apps using it, or change CAMERA_INDEX in config.py.")
            return

    frame_n, t_prev = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed.")
            break

        analyzer.detect(frame, use_pose, use_phone)
        if ss.calibrate:
            base = analyzer.calibrate()
            if base:
                ss.baseline, ss.calibrate = base, False

        q_left = ss.q_until - time.time()
        students, counts = analyzer.evaluate(ss.baseline, thr, q_left > 0)

        if counts["total"] > 0:
            ss.history.append(counts["pct"])
        smooth = sum(ss.history) / len(ss.history) if ss.history else 0.0

        now = time.time()
        if counts["total"] > 0:
            ss.last_pct = round(smooth, 1)
            for etype, detail in ss.events.update(smooth, counts["hands"], counts["phones"],
                                                   now, q_left > 0):
                db.log_event(ss.session_id, etype, detail, ss.last_pct)
            if now - ss.last_log >= config.LOG_INTERVAL_SEC:
                db.log_reading(ss.session_id, counts, ss.last_pct)
                ss.last_log = now

        analyzer.draw(frame, students, show_debug)
        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                       use_container_width=True)

        frame_n += 1
        if frame_n % 5 == 0:
            fps = 5 / max(now - t_prev, 1e-6)
            t_prev = now
            if counts["total"] > 0:
                ss.trend.append(smooth)
            gauge_ph.markdown(ui.gauge_html(smooth, list(ss.trend)), unsafe_allow_html=True)
            kpi_ph.markdown(ui.kpi_grid_html(counts), unsafe_allow_html=True)
            extra = f'<div class="qpill">Question window: {q_left:.0f}s left</div>' if q_left > 0 else ""
            note_ph.markdown(f'<div style="color:#8A97B8;font-size:.82rem;margin-top:8px">'
                             f'{counts["total"]} students seen · {fps:.1f} FPS</div>{extra}',
                             unsafe_allow_html=True)
            if use_phone and analyzer.phone_error:
                warn_ph.markdown(f'<div class="warn">Phone detection is unavailable: '
                                 f'{analyzer.phone_error[:140]}</div>', unsafe_allow_html=True)


# =================================================================== HISTORY
def history_page():
    release_camera()
    st.markdown(ui.header_html("Session History", "Every session as a time-stamped timeline of what happened"),
                unsafe_allow_html=True)
    summ = db.session_summary()
    if summ.empty:
        st.markdown(ui.idle_card("No sessions yet. Run the Live Dashboard for a few minutes first."),
                    unsafe_allow_html=True)
        return
    summ["start"], summ["end"] = pd.to_datetime(summ["start"]), pd.to_datetime(summ["end"])
    labels = {r.session_id: f"{r.start:%d %b %Y, %H:%M} · {fmt_dur((r.end - r.start).total_seconds())}"
                            f" · avg {r.avg_pct:.0f}% · {r.n_events} events"
              for r in summ.itertuples()}
    sid = st.selectbox("Choose a session", list(labels), format_func=labels.get)

    df = db.read_session(sid)
    df["ts"] = pd.to_datetime(df["ts"])
    ev = db.read_events(sid)
    ev["ts"] = pd.to_datetime(ev["ts"])
    ev["label"] = ev["etype"].map(lambda e: ui.EVENT_META.get(e, (e,))[0])

    if len(df) < 5:
        st.warning("Insufficient data: this session has fewer than 5 readings, so the numbers may not be reliable.")

    hi, lo = df.loc[df["class_pct"].idxmax()], df.loc[df["class_pct"].idxmin()]
    k = st.columns(6)
    k[0].metric("Duration", fmt_dur((df["ts"].max() - df["ts"].min()).total_seconds()))
    k[1].metric("Average engagement", f"{df['class_pct'].mean():.0f}%")
    k[2].metric("Highest", f"{hi['class_pct']:.0f}%", f"{hi['ts']:%H:%M:%S}", delta_color="off")
    k[3].metric("Lowest", f"{lo['class_pct']:.0f}%", f"{lo['ts']:%H:%M:%S}", delta_color="off")
    k[4].metric("Hand-raise events", int((ev["etype"] == "hand_raised").sum()))
    k[5].metric("Phone detections", int((ev["etype"] == "phone_detected").sum()))

    n_drop = int((ev["etype"] == "engagement_drop").sum())
    n_rise = int((ev["etype"] == "engagement_rise").sum())
    with st.container(border=True):
        st.markdown(
            f"**Session story:** engagement started at **{df['class_pct'].iloc[0]:.0f}%** "
            f"({df['ts'].iloc[0]:%H:%M:%S}), was lowest at **{lo['class_pct']:.0f}%** at {lo['ts']:%H:%M:%S} "
            f"and highest at **{hi['class_pct']:.0f}%** at {hi['ts']:%H:%M:%S}. "
            f"It dropped {n_drop} time(s) and recovered {n_rise} time(s).")

    # ---- annotated chart ----
    st.subheader("Engagement over time, with events")
    x = alt.X("ts:T", title="Time", axis=alt.Axis(format="%H:%M:%S"))
    base = alt.Chart(df).encode(x=x)
    area = base.mark_area(opacity=0.14, color="#5B7CFF").encode(y=alt.Y("class_pct:Q", scale=alt.Scale(domain=[0, 100])))
    line = base.mark_line(color="#8CA2FF", strokeWidth=2.5, interpolate="monotone").encode(
        y=alt.Y("class_pct:Q", title="Class engagement %", scale=alt.Scale(domain=[0, 100])))
    layers = [area, line]
    if not ev.empty:
        pts = pd.merge_asof(ev.sort_values("ts"),
                            df[["ts", "class_pct"]].sort_values("ts").rename(columns={"class_pct": "y"}),
                            on="ts", direction="nearest")
        names = [m[0] for m in ui.EVENT_META.values()]
        colors = [m[1] for m in ui.EVENT_META.values()]
        layers.append(alt.Chart(pts).mark_point(filled=True, size=120, opacity=1).encode(
            x="ts:T", y="y:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=names, range=colors),
                            legend=alt.Legend(title="Event", orient="bottom")),
            tooltip=[alt.Tooltip("ts:T", title="Time", format="%H:%M:%S"), alt.Tooltip("label:N", title="Event"),
                     alt.Tooltip("detail:N", title="Details"), alt.Tooltip("y:Q", title="Class %", format=".0f")]))
    st.altair_chart(alt.layer(*layers).properties(height=340).interactive(), use_container_width=True)

    with st.expander("How is engagement estimated?"):
        st.markdown(
            "- **Engaged**: facing the board and no phone, or hand raised during a question.\n"
            "- **Partial**: head slightly turned, or facing the board but a phone is near.\n"
            "- **Disengaged**: looking away or down, or face hidden.\n"
            "- **Class engagement %** = (engaged + 0.5 × partial) ÷ students seen × 100, "
            "averaged over the last few readings to avoid flicker.\n"
            "- This is an **estimate from visible behaviour**, not a measure of understanding.")

    # ---- timeline + inspect a moment ----
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Event timeline")
        present = [m[0] for m in ui.EVENT_META.values() if m[0] in set(ev["label"])]
        shown = ev
        if present:
            chosen = st.pills("Filter events", present, selection_mode="multi", default=present)
            shown = ev[ev["label"].isin(chosen)]
        with st.container(height=460):
            st.markdown(ui.timeline_html(shown), unsafe_allow_html=True)
    with col_b:
        st.subheader("Inspect a moment")
        times = df["ts"].dt.strftime("%H:%M:%S").tolist()
        pick = times[0] if len(times) < 2 else st.select_slider("Move through the session", options=times, value=times[0])
        row = df.iloc[times.index(pick)]
        st.markdown(ui.gauge_html(float(row["class_pct"]), label=f"at {pick}"), unsafe_allow_html=True)
        st.markdown(ui.kpi_grid_html({"engaged": int(row["engaged"]), "partial": int(row["partial"]),
                                      "disengaged": int(row["disengaged"]), "hands": int(row["hands"]),
                                      "phones": int(row["phones"]), "total": int(row["total"])}),
                    unsafe_allow_html=True)
        near = ev[(ev["ts"] - row["ts"]).abs() <= pd.Timedelta(seconds=15)]
        st.caption("Events within 15 seconds of this moment")
        st.markdown(ui.timeline_html(near), unsafe_allow_html=True)

    with st.expander("Minute-by-minute table"):
        s = df.set_index("ts")
        m = s.resample("1min").agg(avg_pct=("class_pct", "mean"), lowest=("class_pct", "min"),
                                   highest=("class_pct", "max"), students=("total", "max"),
                                   hands=("hands", "max"), phones=("phones", "max")).dropna()
        counts = ev.set_index("ts").resample("1min").size() if not ev.empty else pd.Series(dtype=int)
        m["events"] = counts.reindex(m.index).fillna(0).astype(int)
        m.index = m.index.strftime("%H:%M")
        m.index.name = "Minute"
        st.dataframe(m.round(0), use_container_width=True)

    safe = sid.replace(":", "-").replace(" ", "_")
    d = st.columns(4)
    d[0].download_button("Download readings (CSV)", df.to_csv(index=False), file_name=f"readings_{safe}.csv",
                         use_container_width=True)
    d[1].download_button("Download events (CSV)", ev.drop(columns=["label"]).to_csv(index=False),
                         file_name=f"events_{safe}.csv", use_container_width=True)
    with st.expander("Delete this session"):
        sure = st.checkbox("Yes, permanently delete this session's data")
        if st.button("Delete session", icon=":material/delete:", disabled=not sure):
            db.delete_session(sid)
            st.rerun()


# =================================================================== PRIVACY
def privacy_page():
    release_camera()
    st.markdown(ui.header_html("Privacy & Settings", "What is collected, and how to remove it"),
                unsafe_allow_html=True)
    st.markdown(
        '<div class="cards">'
        '<div class="card"><h4>No video is stored</h4><p>Camera frames are analysed in memory and discarded '
        'immediately. Nothing is recorded.</p></div>'
        '<div class="card"><h4>No identification</h4><p>There is no face recognition. Students appear only '
        'as temporary labels (S1, S2) while the camera runs.</p></div>'
        '<div class="card"><h4>Only numbers are saved</h4><p>Counts, percentages and event times are stored '
        'on this computer in one local file.</p></div></div>', unsafe_allow_html=True)
    st.info("Tell students and staff that the camera is running and why. The Live Dashboard shows a clear "
            "LIVE indicator while the camera is on.")
    st.subheader("Data retention")
    days = st.slider("Delete sessions older than (days)", 1, 365, 30)
    if st.button("Delete old sessions", icon=":material/auto_delete:"):
        st.success(f"Deleted {db.delete_older_than(days)} session(s).")
    with st.expander("Delete everything"):
        sure = st.checkbox("Yes, delete ALL saved sessions and events")
        if st.button("Delete all data", icon=":material/delete_forever:", disabled=not sure):
            db.delete_everything()
            st.success("All saved data was deleted.")
    st.caption(f"Data file: {config.DB_PATH}")


HOME_PAGE = st.Page(home_page, title="Home", icon=":material/home:", url_path="home", default=True)
LIVE_PAGE = st.Page(live_page, title="Live Dashboard", icon=":material/videocam:", url_path="live")
HISTORY_PAGE = st.Page(history_page, title="Session History", icon=":material/history:", url_path="history")
PRIVACY_PAGE = st.Page(privacy_page, title="Privacy & Settings", icon=":material/shield:", url_path="privacy")

st.navigation([HOME_PAGE, LIVE_PAGE, HISTORY_PAGE, PRIVACY_PAGE]).run()
