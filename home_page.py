import streamlit as st

ss = st.session_state


def go(page):
    ss["nav"] = page


def home_page():
    st.markdown("""
<div class="home">
  <div class="home-logo">🎓 ClassInsight AI</div>
  <h1 class="home-title">Understand classroom engagement,<br>responsibly.</h1>
  <p class="home-sub">An anonymous, AI-estimated view of how a class is responding, live and afterwards.
  No names. No face recognition. Recording is optional and off by default.</p>
</div>""", unsafe_allow_html=True)
    _, a, b, c, _ = st.columns([2, 2, 2, 2, 2])
    a.button("▶ Try demo", on_click=go, args=("📡 Live dashboard",), type="primary")
    b.button("Explore features", on_click=go, args=("📊 Analytics",))
    c.button("Sign in (coming soon)", disabled=True)

    st.markdown('<div class="sec" style="text-align:center;margin-top:34px">What you get</div>', unsafe_allow_html=True)
    feats = [("📡 Live dashboard", "Camera, detection status, session timer, engagement estimate, head direction, hand raises and phones."),
             ("📊 Analytics", "Expression mix over time, confidence levels, an honest 'insufficient data' state and a plain explanation of the score."),
             ("🎞 History & 📄 Reports", "Past sessions, comparisons, event timeline, key-moment playback, CSV and PDF exports."),
             ("⚙️ Privacy & settings", "Camera and sensitivity controls, recording off by default, face pixelation and one-click deletion.")]
    for col, (t, p) in zip(st.columns(4), feats):
        col.markdown(f'<div class="priv"><h4>{t}</h4><p>{p}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:26px">Our privacy statement</div>', unsafe_allow_html=True)
    st.markdown("Students are never named or recognised. Only counts, percentages and events are saved. "
                "The engagement score is an estimate based on head direction, raised hands and phones. It is not a "
                "measurement of attention or learning. Video is saved only if you switch recording on, with faces "
                "pixelated by default.")
    st.caption("ClassInsight AI is a placeholder name, not a confirmed available brand or domain.")
