import html

import altair as alt
import pandas as pd
import streamlit as st

import database as db
from ui import ESTIMATE_NOTE, EVENT_STYLE, hero, mmss

BANDS = ["Good (70%+)", "Mixed (40-69%)", "Low (under 40%)"]
BAND_COLOURS = ["#4ADE80", "#FBBF24", "#F87171"]


def band(p):
    return BANDS[0] if p >= 70 else BANDS[1] if p >= 40 else BANDS[2]


def session_table(rd, ev):
    rows = []
    for sid, g in rd.groupby("session_id", sort=False):
        e = ev[ev["session_id"] == sid]["event_type"]
        t0 = g["ts"].min()
        rows.append({"Session": sid, "Start": t0, "Minutes": round((g["ts"].max() - t0).total_seconds() / 60, 1),
                     "Average %": round(g["class_pct"].mean(), 1), "Lowest %": round(g["class_pct"].min(), 1),
                     "Highest %": round(g["class_pct"].max(), 1), "Hand raises": int((e == "hand_raised").sum()),
                     "Phone detections": int((e == "phone_detected").sum()),
                     "Engagement drops": int((e == "engagement_drop").sum())})
    return pd.DataFrame(rows).sort_values("Start").reset_index(drop=True)


def build_pdf(sid, df, ev):
    """A real PDF summary (reportlab): key numbers, engagement line chart, event table."""
    from io import BytesIO

    from reportlab.graphics.shapes import Drawing, PolyLine, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    t0 = df["ts"].min()
    xs = (df["ts"] - t0).dt.total_seconds()
    span, W, H = max(xs.max(), 1), 480, 150
    d = Drawing(W, H + 20)
    d.add(Rect(30, 20, W - 40, H - 20, fillColor=colors.HexColor("#F8FAFC"), strokeColor=colors.HexColor("#CBD5E1")))
    pts = [v for x, y in zip(xs, df["class_pct"]) for v in (30 + x / span * (W - 40), 20 + y / 100 * (H - 20))]
    if len(pts) >= 4:
        d.add(PolyLine(pts, strokeColor=colors.HexColor("#0E7490"), strokeWidth=2))
    d.add(String(2, H - 2, "100%", fontSize=7))
    d.add(String(8, 18, "0%", fontSize=7))
    d.add(String(30, 6, "00:00", fontSize=7))
    d.add(String(W - 36, 6, mmss(span), fontSize=7))
    st_ = getSampleStyleSheet()
    kpi = [["Average", "Lowest", "Highest", "Duration", "Hand raises", "Phones"],
           [f"{df['class_pct'].mean():.0f}%", f"{df['class_pct'].min():.0f}%", f"{df['class_pct'].max():.0f}%",
            mmss((df["ts"].max() - t0).total_seconds()), int((ev["event_type"] == "hand_raised").sum()),
            int((ev["event_type"] == "phone_detected").sum())]]
    rows = [["Time", "Event", "Detail"]] + [[mmss((e.ts - t0).total_seconds()),
            EVENT_STYLE.get(e.event_type, (e.event_type,))[0], str(e.detail)] for e in ev.itertuples()][:40]
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0F2FE")), ("FONTSIZE", (0, 0), (-1, -1), 8)])
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=f"Engagement report {sid}").build([
        Paragraph("Classroom Engagement Report", st_["Title"]), Paragraph(f"Session {sid}", st_["Normal"]), Spacer(1, 10),
        Table(kpi, style=grid), Spacer(1, 12), Paragraph("Engagement over time", st_["Heading3"]), d, Spacer(1, 8),
        Paragraph("Event timeline", st_["Heading3"]), Table(rows, style=grid, repeatRows=1, colWidths=[45, 110, 340]),
        Spacer(1, 12), Paragraph(ESTIMATE_NOTE + " No names or face recognition were used.", st_["Italic"])])
    return buf.getvalue()


def build_report(sid, df, ev):
    """A light, printable one-page summary (open it and press Ctrl+P to save as PDF)."""
    t0 = df["ts"].min()
    xs = (df["ts"] - t0).dt.total_seconds()
    span, W, H = max(xs.max(), 1), 640, 170
    pts = " ".join(f"{40 + x / span * (W - 60):.1f},{H - 20 - y / 100 * (H - 40):.1f}"
                   for x, y in zip(xs, df["class_pct"]))
    rows = "".join(f"<tr><td>{mmss((e.ts - t0).total_seconds())}</td>"
                   f"<td>{EVENT_STYLE.get(e.event_type, (e.event_type,))[0]}</td><td>{html.escape(str(e.detail))}</td></tr>"
                   for e in ev.itertuples())
    k = [("Average", f"{df['class_pct'].mean():.0f}%"), ("Lowest", f"{df['class_pct'].min():.0f}%"),
         ("Highest", f"{df['class_pct'].max():.0f}%"), ("Duration", mmss((df['ts'].max() - t0).total_seconds())),
         ("Hand raises", int((ev["event_type"] == "hand_raised").sum())),
         ("Phone detections", int((ev["event_type"] == "phone_detected").sum()))]
    kpi = "".join(f'<div class="k"><b>{v}</b><span>{n}</span></div>' for n, v in k)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Engagement report {html.escape(sid)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:30px auto;color:#0F172A}}
h1{{margin:0 0 4px}}.sub{{color:#475569}}.g{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}}
.k{{border:1px solid #CBD5E1;border-radius:10px;padding:10px 14px}}.k b{{display:block;font-size:1.6rem;color:#0E7490}}
.k span{{color:#64748B;font-size:.85rem}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}
td,th{{border-bottom:1px solid #E2E8F0;padding:6px 8px;text-align:left}}.n{{color:#64748B;font-size:.8rem;margin-top:18px}}</style></head><body>
<h1>Classroom Engagement Report</h1><div class="sub">Session {html.escape(sid)}</div>
<div class="g">{kpi}</div><h3>Engagement over time</h3>
<svg viewBox="0 0 {W} {H}" width="100%"><rect x="40" y="20" width="{W - 60}" height="{H - 40}" fill="#F8FAFC" stroke="#CBD5E1"/>
<polyline points="{pts}" fill="none" stroke="#0E7490" stroke-width="2.5"/>
<text x="4" y="26" font-size="11">100%</text><text x="14" y="{H - 18}" font-size="11">0%</text>
<text x="40" y="{H - 4}" font-size="11">00:00</text><text x="{W - 60}" y="{H - 4}" font-size="11">{mmss(span)}</text></svg>
<h3>Event timeline</h3><table><tr><th>Time</th><th>Event</th><th>Detail</th></tr>{rows}</table>
<p class="n">{ESTIMATE_NOTE} No names or face recognition were used.</p></body></html>"""


def reports_page():
    hero("Reports", "Session summaries, comparison, event breakdown and exports", "📄")
    rd, ev = db.read_all()
    if rd.empty:
        st.info("No data yet. Run a session on the Live dashboard first.")
        return
    st.caption(ESTIMATE_NOTE)
    tbl = session_table(rd, ev)
    tbl["Band"] = tbl["Average %"].map(band)
    c = st.columns(4)
    c[0].metric("Sessions", len(tbl))
    c[1].metric("Average engagement", f"{tbl['Average %'].mean():.0f}%")
    best = tbl.loc[tbl["Average %"].idxmax()]
    c[2].metric("Best session", f"{best['Average %']:.0f}%", best["Session"][5:16], delta_color="off")
    c[3].metric("Hand raises / phones", f"{tbl['Hand raises'].sum()} / {tbl['Phone detections'].sum()}")

    st.markdown('<div class="sec">Average engagement per session</div>', unsafe_allow_html=True)
    st.altair_chart(alt.Chart(tbl).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Session:N", sort=list(tbl["Session"]), title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Average %:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Band:N", scale=alt.Scale(domain=BANDS, range=BAND_COLOURS), legend=alt.Legend(title=None)),
        tooltip=["Session", "Average %", "Lowest %", "Highest %"]).properties(height=280),
        use_container_width=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="sec">Event breakdown</div>', unsafe_allow_html=True)
        scope = st.selectbox("Sessions", ["All sessions"] + list(tbl["Session"]))
        e = ev if scope == "All sessions" else ev[ev["session_id"] == scope]
        e = e[~e["event_type"].isin(["session_start", "session_end"])]
        cnt = e["event_type"].value_counts().rename_axis("type").reset_index(name="Count")
        if cnt.empty:
            st.caption("No events recorded.")
        else:
            cnt["Event"] = cnt["type"].map(lambda k: EVENT_STYLE.get(k, (k,))[0])
            known = [k for k in EVENT_STYLE if k not in ("session_start", "session_end")]
            st.altair_chart(alt.Chart(cnt).mark_bar(cornerRadiusEnd=6).encode(
                y=alt.Y("Event:N", sort="-x", title=None), x=alt.X("Count:Q", title=None),
                color=alt.Color("Event:N", legend=None, scale=alt.Scale(
                    domain=[EVENT_STYLE[k][0] for k in known], range=[EVENT_STYLE[k][2] for k in known])),
                tooltip=["Event", "Count"]).properties(height=240), use_container_width=True)
    with b:
        st.markdown('<div class="sec">Compare sessions</div>', unsafe_allow_html=True)
        pick = st.multiselect("Sessions to overlay", list(tbl["Session"]), default=list(tbl["Session"])[-3:])
        if pick:
            d = rd[rd["session_id"].isin(pick)].copy()
            d["Minutes"] = d.groupby("session_id")["ts"].transform(lambda s: (s - s.min()).dt.total_seconds() / 60)
            st.altair_chart(alt.Chart(d).mark_line(strokeWidth=2.5, interpolate="monotone").encode(
                x=alt.X("Minutes:Q"), y=alt.Y("class_pct:Q", title="Engagement %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("session_id:N", title=None, legend=alt.Legend(orient="bottom"))).properties(height=240),
                use_container_width=True)

    st.markdown('<div class="sec">All sessions</div>', unsafe_allow_html=True)
    st.dataframe(tbl.drop(columns="Band"), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Exports</div>', unsafe_allow_html=True)
    x = st.columns(3)
    x[0].download_button("⬇ Session summary CSV", tbl.drop(columns="Band").to_csv(index=False), file_name="sessions_summary.csv")
    x[1].download_button("⬇ All readings CSV", rd.to_csv(index=False), file_name="all_readings.csv")
    x[2].download_button("⬇ All events CSV", ev.drop(columns=["image"], errors="ignore").to_csv(index=False), file_name="all_events.csv")
    sid = st.selectbox("Printable summary for", list(tbl["Session"])[::-1])
    one_rd, one_ev = rd[rd["session_id"] == sid], ev[ev["session_id"] == sid]
    safe = sid.replace(":", "-").replace(" ", "_")
    p1, p2 = st.columns(2)
    p1.download_button("📄 Download PDF report", build_pdf(sid, one_rd, one_ev),
                       file_name=f"report_{safe}.pdf", mime="application/pdf")
    p2.download_button("🖨 Download printable HTML", build_report(sid, one_rd, one_ev),
                       file_name=f"report_{safe}.html", mime="text/html")
