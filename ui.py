import streamlit as st

CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
#MainMenu, footer{visibility:hidden}
.hero{background:linear-gradient(135deg,#0E7490 0%,#2563EB 55%,#7C3AED 100%);
  padding:18px 26px;border-radius:18px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.65rem;color:#fff}
.hero p{margin:3px 0 0;color:#E0F2FE;font-size:.92rem}
[data-testid="stMetric"]{background:#111A2E;border:1px solid #1F2A44;border-radius:14px;padding:12px 16px}
[data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stMetricLabel"]{color:#94A3B8}
section[data-testid="stSidebar"]{background:#0D1526;border-right:1px solid #1F2A44}
div.stButton>button, div.stDownloadButton>button, div.stFormSubmitButton>button{width:100%;border-radius:10px;
  font-weight:600;border:1px solid #22D3EE55;background:#0F1B33;color:#E6EDF7;transition:all .15s}
div.stButton>button:hover, div.stDownloadButton>button:hover, div.stFormSubmitButton>button:hover{
  background:#22D3EE;color:#06202A;border-color:#22D3EE}
.card{background:#111A2E;border:1px solid #1F2A44;border-radius:14px;padding:12px 16px;margin-bottom:10px}
.card .lbl{color:#94A3B8;font-size:.85rem}
.card .val{font-size:2.6rem;font-weight:700;line-height:1.1}
.chip{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:700;
  border:1px solid var(--c);color:var(--c);background:#0F1B33;margin:0 8px 8px 0;letter-spacing:.3px}
.note{color:#94A3B8;font-size:.82rem}
.priv{background:#111A2E;border:1px solid #1F2A44;border-radius:14px;padding:14px 18px;min-height:150px}
.priv h4{margin:0 0 6px;color:#22D3EE;font-size:1rem}
.priv p{margin:0;color:#CBD5E1;font-size:.9rem}
.ev{display:flex;gap:12px;align-items:flex-start;padding:10px 14px;margin-bottom:8px;
  background:#111A2E;border:1px solid #1F2A44;border-left:4px solid var(--c);border-radius:10px}
.ev .t{font-variant-numeric:tabular-nums;font-weight:700;color:#E6EDF7;min-width:56px}
.ev .k{font-size:.78rem;color:var(--c);font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.ev .d{color:#CBD5E1;font-size:.92rem}
.ev .p{margin-left:auto;color:#94A3B8;font-size:.85rem;white-space:nowrap}
.sec{font-size:1.05rem;font-weight:700;margin:18px 0 8px;color:#E6EDF7}
</style>
"""

EVENT_STYLE = {   # type: (label, icon, colour)
    "session_start": ("Session started", "▶️", "#94A3B8"),
    "question_asked": ("Question asked", "❓", "#A78BFA"),
    "hand_raised": ("Hand raised", "✋", "#22D3EE"),
    "phone_detected": ("Phone detected", "📱", "#F59E0B"),
    "phone_cleared": ("Phone cleared", "✅", "#34D399"),
    "engagement_drop": ("Engagement drop", "📉", "#F87171"),
    "engagement_rise": ("Engagement rise", "📈", "#4ADE80"),
}

ESTIMATE_NOTE = ("Engagement is an AI-estimated indicator from head direction, raised hands and phones. "
                 "It is not a measurement of attention or learning.")


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, sub, icon="🎓"):
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def chip(text, colour="#22D3EE", dot="●"):
    return f'<span class="chip" style="--c:{colour}">{dot} {text}</span>'


def pct_colour(p):
    return "#4ADE80" if p >= 70 else "#FBBF24" if p >= 40 else "#F87171"


def mmss(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@st.cache_resource
def get_analyzer():
    import config
    from analyzer import Analyzer
    return Analyzer(config.MAX_STUDENTS)


@st.cache_resource
def get_recorder():
    from recorder import Recorder
    return Recorder()


@st.cache_resource
def get_camera_holder():
    return {"cap": None, "index": None}


def release_camera(holder):
    if holder["cap"] is not None:
        holder["cap"].release()
        holder["cap"] = None
