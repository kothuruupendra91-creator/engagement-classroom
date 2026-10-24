"""Look and feel: colours, CSS and small HTML helpers."""
import html

import pandas as pd
import streamlit as st

import config

C = {"engaged": "#22C55E", "partial": "#F59E0B", "disengaged": "#EF4444",
     "hands": "#A78BFA", "phones": "#F472B6", "students": "#38BDF8", "accent": "#5B7CFF"}

# event type -> (label, colour, icon)
EVENT_META = {
    "engagement_drop": ("Engagement dropped", "#EF4444", "↓"),
    "engagement_rise": ("Engagement improved", "#22C55E", "↑"),
    "hand_raised": ("Hand raised", "#A78BFA", "\U0000270B"),
    "phone_detected": ("Phone detected", "#F472B6", "\U0001F4F1"),
    "phone_cleared": ("Phone put away", "#94A3B8", "✓"),
    "question": ("Question asked", "#38BDF8", "?"),
}

_CSS = """
.block-container{padding-top:1.4rem;max-width:1400px}
footer,#MainMenu{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.stApp{background:radial-gradient(1100px 520px at 8% -8%,#17235A 0%,#0A0F1E 58%)}
section[data-testid="stSidebar"]{background:#0D1328;border-right:1px solid #1B2547}
[data-testid="stSidebarNav"]::before{content:"__BRAND__";display:block;font-weight:700;font-size:1.2rem;
color:#fff;padding:20px 18px 6px;letter-spacing:-.01em}
[data-testid="stSidebarNav"] a{border-radius:10px;margin:2px 8px}
[data-testid="stSidebarNav"]
a[aria-current="page"]{background:linear-gradient(135deg,rgba(91,124,255,.28),rgba(124,92,255,.22));
border:1px solid rgba(91,124,255,.45)}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:20px
24px;border-radius:18px;background:linear-gradient(135deg,#3949E8 0%,#6D5CF6
100%);margin-bottom:16px;box-shadow:0 10px 34px rgba(79,70,229,.30)}
.hdr-t{font-size:1.55rem;font-weight:700;color:#fff;letter-spacing:-.01em}
.hdr-s{font-size:.92rem;color:#D9DEFF;margin-top:2px}
.pill{padding:6px 14px;border-radius:999px;font-size:.78rem;font-weight:700;letter-spacing:.06em}
.pill.live{background:rgba(34,197,94,.18);color:#4ADE80;border:1px solid
rgba(34,197,94,.55);animation:pulse 1.6s infinite}
.pill.idle{background:rgba(255,255,255,.14);color:#E6EBF7;border:1px solid rgba(255,255,255,.3)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
.card{background:#111833;border:1px solid #1F2A4D;border-radius:16px;padding:18px 20px;height:100%}
.card h4{margin:0 0 6px 0;color:#fff;font-size:1.02rem}
.card p{margin:0;color:#9AA7C7;font-size:.9rem;line-height:1.5}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0}
.hero{text-align:center;padding:46px 20px 26px}
.hero-t{font-size:2.6rem;font-weight:800;color:#fff;letter-spacing:-.02em;line-height:1.15}
.hero-t span{background:linear-gradient(135deg,#6C8CFF,#A78BFA);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-s{font-size:1.08rem;color:#9AA7C7;max-width:640px;margin:14px auto 6px}
.idle-card{display:flex;align-items:center;justify-content:center;min-height:340px;border-radius:16px;border:1px dashed
#2B3A66;color:#8A97B8;background:#0E1530;font-size:1.02rem;text-align:center;padding:20px}
.gauge-card{background:#111833;border:1px solid
#1F2A4D;border-radius:16px;padding:18px;text-align:center}
.gauge{width:170px;height:170px;border-radius:50%;margin:0
auto;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),#1D2748
0)}
.gauge-in{width:132px;height:132px;border-radius:50%;background:#111833;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge-v{font-size:2.2rem;font-weight:700;color:#fff;line-height:1}
.gauge-l{font-size:.74rem;color:#8A97B8;margin-top:4px}
.spark{width:100%;height:56px;margin-top:12px}
.spark-empty{color:#64708F;font-size:.8rem;margin-top:12px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}
.kpi{background:#111833;border:1px solid #1F2A4D;border-left:4px solid
var(--c);border-radius:12px;padding:12px 14px}
.kpi-l{font-size:.72rem;color:#8A97B8;text-transform:uppercase;letter-spacing:.06em}
.kpi-v{font-size:1.8rem;font-weight:700;color:#F1F5FF;line-height:1.2}
.qpill{display:inline-block;margin-top:10px;padding:5px
12px;border-radius:999px;background:rgba(56,189,248,.16);color:#7DD3FC;border:1px solid
rgba(56,189,248,.45);font-size:.8rem;font-weight:600}
.warn{margin-top:10px;padding:10px 14px;border-radius:12px;background:rgba(245,158,11,.12);border:1px
solid rgba(245,158,11,.4);color:#FCD34D;font-size:.85rem}
.tl{position:relative;padding-left:6px}
.tl::before{content:"";position:absolute;left:21px;top:6px;bottom:6px;width:2px;background:#1F2A4D}
.tl-item{display:flex;gap:12px;margin-bottom:12px;position:relative}
.tl-dot{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
font-size:.9rem;color:#0A0F1E;font-weight:700;flex:0 0 32px;z-index:1}
.tl-body{background:#111833;border:1px solid #1F2A4D;border-radius:12px;padding:8px 12px;flex:1}
.tl-top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.tl-time{font-family:ui-monospace,Consolas,monospace;color:#8CA2FF;font-size:.82rem}
.tl-title{font-weight:600;color:#F1F5FF}
.tl-pct{margin-left:auto;font-size:.78rem;color:#9AA7C7;background:#18214A;padding:2px
8px;border-radius:999px}
.tl-detail{color:#9AA7C7;font-size:.85rem;margin-top:2px}
.tl-empty{color:#64708F;padding:12px}
div[data-testid="stImage"] img{border-radius:14px;border:1px solid #1F2A4D}
div[data-testid="stMetric"]{background:#111833;border:1px solid #1F2A4D;border-radius:14px;padding:12px
16px}
button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#4F6BFF,#7C5CFF);border:0;color:#fff;border-radius:10px;font-weight:600}
button[data-testid="stBaseButton-secondary"]{background:#141C3D;border:1px solid
#2A3769;color:#E6EBF7;border-radius:10px;font-weight:600}
button[data-testid="stBaseButton-primary"]:hover,button[data-testid="stBaseButton-secondary"]:hover{filter:brightness(1.15);border-color:#5B7CFF}
@media (max-width:900px){.cards{grid-template-columns:1fr}.hero-t{font-size:2rem}}
"""


def inject_css():
    css = "".join(line.strip() for line in _CSS.splitlines()).replace("__BRAND__", "◆ " + config.APP_NAME)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header_html(title, subtitle, live=None):
    pill = ""
    if live is not None:
        pill = '<span class="pill live">● LIVE</span>' if live else '<span class="pill idle">● CAMERA OFF</span>'
    return (f'<div class="hdr"><div><div class="hdr-t">{html.escape(title)}</div>'
            f'<div class="hdr-s">{html.escape(subtitle)}</div></div>{pill}</div>')


def gauge_html(pct, trend=None, label="Class engagement"):
    c = C["engaged"] if pct >= 70 else C["partial"] if pct >= 40 else C["disengaged"]
    spark = sparkline_html(trend) if trend is not None else ""
    return (f'<div class="gauge-card"><div class="gauge" style="--p:{pct:.0f};--c:{c}">'
            f'<div class="gauge-in"><div class="gauge-v">{pct:.0f}%</div>'
            f'<div class="gauge-l">{html.escape(label)}</div></div></div>{spark}</div>')


def sparkline_html(values, w=300, h=56):
    if len(values) < 2:
        return '<div class="spark-empty">Trend appears after a few seconds</div>'
    n, pts = len(values), []
    for i, v in enumerate(values):
        x = i * (w - 4) / (n - 1) + 2
        y = h - 2 - (max(0, min(100, v)) / 100) * (h - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="#6C8CFF" stroke-width="2.5" stroke-linejoin="round" '
            f'points="{" ".join(pts)}"/></svg>')


def kpi_grid_html(counts):
    items = [("Engaged", counts.get("engaged", 0), C["engaged"]),
             ("Partial", counts.get("partial", 0), C["partial"]),
             ("Disengaged", counts.get("disengaged", 0), C["disengaged"]),
             ("Hands raised", counts.get("hands", 0), C["hands"]),
             ("Phones", counts.get("phones", 0), C["phones"]),
             ("Students seen", counts.get("total", 0), C["students"])]
    cells = "".join(f'<div class="kpi" style="--c:{c}"><div class="kpi-l">{l}</div>'
                    f'<div class="kpi-v">{v}</div></div>' for l, v, c in items)
    return f'<div class="kpis">{cells}</div>'


def timeline_html(ev):
    """ev: DataFrame with ts (datetime), etype, detail, pct."""
    if ev.empty:
        return '<div class="tl-empty">No events in this selection.</div>'
    items = []
    for r in ev.itertuples():
        label, color, icon = EVENT_META.get(r.etype, (r.etype, "#94A3B8", "•"))
        pct = "" if pd.isna(r.pct) else f'<span class="tl-pct">{r.pct:.0f}%</span>'
        items.append(f'<div class="tl-item"><div class="tl-dot" style="background:{color}">{icon}</div>'
                     f'<div class="tl-body"><div class="tl-top"><span class="tl-time">{r.ts:%H:%M:%S}</span>'
                     f'<span class="tl-title">{html.escape(label)}</span>{pct}</div>'
                     f'<div class="tl-detail">{html.escape(str(r.detail))}</div></div></div>')
    return '<div class="tl">' + "".join(items) + "</div>"


def idle_card(text):
    return f'<div class="idle-card">{html.escape(text)}</div>'
