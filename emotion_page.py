import json

import altair as alt
import pandas as pd
import streamlit as st

import database as db
from expression import EMOTIONS
from ui import ESTIMATE_NOTE, hero, mmss

EMO_COLOURS = {"neutral": "#94A3B8", "happy": "#4ADE80", "surprised": "#22D3EE", "sad": "#60A5FA",
               "angry": "#F87171", "fearful": "#A78BFA", "disgusted": "#FBBF24"}
MIN_CONF = 0.5


def emotion_table(df):
    """One row per reading: expression counts, whether the reading is reliable, confidence."""
    rows = [json.loads(x) if isinstance(x, str) and x else {} for x in df["emotions"]]
    emo = pd.DataFrame(rows, index=df.index).reindex(columns=EMOTIONS).fillna(0)
    valid = df["emo_valid"].fillna(0)
    conf = df["emo_conf"].fillna(0)
    return emo, (valid > 0) & (conf >= MIN_CONF), conf


def analytics_view_page():
    hero("Analytics", "Expression estimates over time, and exactly how the score is calculated", "📊")
    st.caption(ESTIMATE_NOTE + " Expressions are rough estimates from facial landmarks and are not used "
                               "in the engagement score.")
    sessions = db.list_sessions()
    if not sessions:
        st.info("No data yet. Run a session on the Live dashboard first.")
        return
    sid = st.selectbox("Session", sessions)
    df = db.read_session(sid)
    df["ts"] = pd.to_datetime(df["ts"])
    t0 = df["ts"].min()
    df["Minutes"] = (df["ts"] - t0).dt.total_seconds() / 60
    emo, reliable, conf = emotion_table(df)

    c = st.columns(4)
    totals = emo[reliable].sum()
    c[0].metric("Most common expression", totals.idxmax().capitalize() if totals.sum() else "—")
    c[1].metric("Reliable readings", f"{reliable.mean() * 100:.0f}%")
    c[2].metric("Average confidence", f"{conf[reliable].mean() * 100:.0f}%" if reliable.any() else "—")
    c[3].metric("Readings", len(df))
    if reliable.mean() < 0.5:
        st.warning("Insufficient data: expressions could not be read reliably for most of this session "
                   "(faces too small, turned away, or poorly lit). The charts below may be misleading.")

    st.markdown('<div class="sec">Expression mix over time</div>', unsafe_allow_html=True)
    share = emo.div(emo.sum(axis=1).replace(0, float("nan")), axis=0).where(reliable)
    long = share.assign(Minutes=df["Minutes"]).melt("Minutes", var_name="Expression", value_name="Share").dropna()
    if long.empty:
        st.info("Insufficient data for an expression chart in this session.")
    else:
        st.altair_chart(alt.Chart(long).mark_area(opacity=0.9).encode(
            x=alt.X("Minutes:Q", title="Minutes into session"),
            y=alt.Y("Share:Q", stack="normalize", title="Share of students", axis=alt.Axis(format="%")),
            color=alt.Color("Expression:N", scale=alt.Scale(domain=EMOTIONS, range=[EMO_COLOURS[k] for k in EMOTIONS])),
            tooltip=["Expression", alt.Tooltip("Share:Q", format=".0%")]).properties(height=300),
            use_container_width=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="sec">Engagement and confidence</div>', unsafe_allow_html=True)
        d = pd.DataFrame({"Minutes": df["Minutes"], "Engagement %": df["class_pct"],
                          "Expression confidence %": conf * 100}).melt("Minutes", var_name="Series")
        st.altair_chart(alt.Chart(d).mark_line(strokeWidth=2.2).encode(
            x="Minutes:Q", y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None),
                            scale=alt.Scale(range=["#22D3EE", "#A78BFA"]))).properties(height=240),
            use_container_width=True)
    with b:
        st.markdown('<div class="sec">Expression totals</div>', unsafe_allow_html=True)
        tot = pd.DataFrame({"Expression": EMOTIONS, "Student-readings": [int(totals.get(k, 0)) for k in EMOTIONS]})
        tot["Share"] = (tot["Student-readings"] / max(tot["Student-readings"].sum(), 1) * 100).round(1).astype(str) + "%"
        st.dataframe(tot, hide_index=True, use_container_width=True)

    st.markdown('<div class="sec">How the engagement score is calculated</div>', unsafe_allow_html=True)
    st.markdown("""
| Situation (per student) | Level |
|---|---|
| Hand raised while a question is open | 🟢 Engaged |
| Facing the board, no phone | 🟢 Engaged |
| Facing the board but a phone is near, or head turned slightly | 🟠 Partial |
| Looking away or head down (or face hidden) | 🔴 Disengaged |

**Class engagement % = (engaged + 0.5 × partial) ÷ students × 100**, then averaged over the last 10 readings
to avoid jumps. Head direction is compared with the calibrated "looking at the board" position.""")
    pick = st.select_slider("Worked example: choose a time", options=list(range(len(df))),
                            format_func=lambda i: mmss((df['ts'].iloc[i] - t0).total_seconds()), value=len(df) - 1)
    r = df.iloc[pick]
    n = max(int(r["total"]), 1)
    st.code(f"({int(r['engaged'])} engaged + 0.5 × {int(r['partial'])} partial) ÷ {n} students × 100 "
            f"= {(r['engaged'] + 0.5 * r['partial']) / n * 100:.0f}%   (saved reading shows {r['class_pct']:.0f}%, smoothed)",
            language=None)
    st.caption("Expression labels come from face-landmark scores (smile, brow raise, frown and so on). "
               "They are approximate, vary between people and cultures, and need a clear, close, well-lit face.")
