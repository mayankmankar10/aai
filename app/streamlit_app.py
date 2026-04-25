"""
app/streamlit_app.py
====================
Interface Layer — Streamlit Web UI for the CapsNet Traffic Sign Agent
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import streamlit as st
from PIL import Image

from agent.agent_pipeline import agent_pipeline, AgentResponse

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CapsNet Traffic Sign Agent",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Main header ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        color: rgba(255,255,255,0.55);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    }

    /* ── Section headings ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Prediction badge ── */
    .prediction-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 0.3rem;
    }
    .prediction-class {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.45);
        margin-bottom: 1.2rem;
    }

    /* ── Confidence bar ── */
    .conf-bar-container {
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
        margin-bottom: 0.4rem;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #6366f1, #a78bfa);
        transition: width 0.8s ease;
    }
    .conf-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.55);
    }

    /* ── Info pills ── */
    .pill {
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
    }
    .pill-purple { background: rgba(167,139,250,0.18); color: #a78bfa; border: 1px solid rgba(167,139,250,0.35); }
    .pill-blue   { background: rgba(96,165,250,0.18);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.35); }
    .pill-green  { background: rgba(52,211,153,0.18);  color: #34d399; border: 1px solid rgba(52,211,153,0.35); }
    .pill-red    { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid rgba(248,113,113,0.35); }

    /* ── Explanation block ── */
    .explanation-box {
        background: rgba(167,139,250,0.08);
        border-left: 3px solid #a78bfa;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        color: rgba(255,255,255,0.82);
        font-size: 0.97rem;
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    /* ── Safety tip ── */
    .safety-box {
        background: rgba(52,211,153,0.07);
        border-left: 3px solid #34d399;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        color: rgba(255,255,255,0.82);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── Trace expander content ── */
    .trace-line {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        font-family: 'Courier New', monospace;
        line-height: 1.7;
    }

    /* ── Error card ── */
    .error-card {
        background: rgba(248,113,113,0.1);
        border: 1px solid rgba(248,113,113,0.3);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        color: #f87171;
        font-size: 0.95rem;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.35) !important;
        border-radius: 16px !important;
        background: rgba(167,139,250,0.05) !important;
        padding: 1rem !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Streamlit element overrides ── */
    .stMarkdown p { color: rgba(255,255,255,0.75); }
    label { color: rgba(255,255,255,0.65) !important; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s ease, transform 0.15s ease;
    }
    .stButton>button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _confidence_color_class(confidence: float) -> str:
    if confidence >= 0.80:
        return "pill-green"
    elif confidence >= 0.50:
        return "pill-blue"
    else:
        return "pill-red"


def _render_confidence_bar(confidence: float):
    pct = int(confidence * 100)
    st.markdown(f"""
    <div class="conf-bar-container">
        <div class="conf-bar-fill" style="width:{pct}%"></div>
    </div>
    <div class="conf-label">{pct}% confidence</div>
    """, unsafe_allow_html=True)


def _render_result(response: AgentResponse):
    """Render the agent response with rich visual components."""

    if not response.success:
        st.markdown(f"""
        <div class="error-card">
            ⚠️ <strong>Classification Failed</strong><br><br>
            {response.error_message}
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Prediction card ──
    conf_pill = _confidence_color_class(response.confidence)
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">🎯 Prediction</div>
        <div class="prediction-label">{response.label}</div>
        <div class="prediction-class">Class Index: {response.class_index}</div>
        <span class="pill pill-purple">🏷️ {response.category}</span>
        <span class="pill {conf_pill}">📊 {response.confidence:.1%} confidence</span>
    </div>
    """, unsafe_allow_html=True)

    _render_confidence_bar(response.confidence)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Explanation card ──
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">💡 Agent Explanation</div>
        <div class="explanation-box">{response.explanation}</div>
        <div class="section-title" style="margin-top:1rem;">🛡️ Safety Advice</div>
        <div class="safety-box">{response.safety_tip}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence assessment ──
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">📈 Confidence Assessment</div>
        <div style="color:rgba(255,255,255,0.75);font-size:0.95rem;line-height:1.65;">
            {response.confidence_assessment}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reasoning trace (collapsible) ──
    with st.expander("🔍 View Agent Reasoning Trace", expanded=False):
        trace_html = "<br>".join(
            f'<span class="trace-line">{line}</span>'
            for line in response.reasoning_trace
        )
        st.markdown(trace_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.8rem;">
        <div style="font-size:3rem;">🚦</div>
        <div style="font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.9);">CapsNet Agent</div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:0.3rem;">Traffic Sign Intelligence</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**ℹ️ About**")
    st.markdown(
        "<div style='color:rgba(255,255,255,0.6);font-size:0.85rem;line-height:1.6;'>"
        "This system uses a <b>Capsule Network (CapsNet)</b> trained on the "
        "<b>GTSRB dataset</b> (43 traffic sign classes) to classify uploaded "
        "traffic sign images and provide contextual explanations."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🗂️ Supported Classes**")
    from model.capsnet_model import GTSRB_LABELS
    selected_class = st.selectbox(
        "Browse class labels",
        options=list(GTSRB_LABELS.keys()),
        format_func=lambda k: f"[{k:02d}] {GTSRB_LABELS[k]}",
        label_visibility="collapsed",
    )
    st.markdown(
        f"<div class='pill pill-purple' style='margin-top:0.5rem;font-size:0.75rem;'>"
        f"Class {selected_class}: {GTSRB_LABELS[selected_class]}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:rgba(255,255,255,0.3);font-size:0.72rem;text-align:center;'>"
        "Model: CapsNet · Dataset: GTSRB<br>Inference only — no retraining"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**⚙️ Model Path**")
    custom_model_path = st.text_input(
        "Custom .h5 path (optional)",
        placeholder="Leave blank for default path",
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-title">🚦 CapsNet Traffic Sign Agent</div>
<div class="hero-sub">
    Upload a traffic sign image — the agentic AI will classify it,<br>
    explain its meaning, and provide contextual safety advice.
</div>
""", unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop or select a traffic sign image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.4);font-size:0.78rem;margin-top:0.4rem;'>"
            f"📄 {uploaded_file.name} &nbsp;|&nbsp; {img.size[0]}×{img.size[1]} px</div>",
            unsafe_allow_html=True,
        )

        run_btn = st.button("🚀 Analyze Sign", use_container_width=True)
    else:
        st.markdown("""
        <div style="color:rgba(255,255,255,0.35);font-size:0.88rem;text-align:center;padding:2rem 1rem;">
            No image uploaded yet.<br>
            Supported formats: PNG · JPG · BMP · WebP
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col_result:
    st.markdown('<div class="section-title">🤖 Agent Analysis</div>', unsafe_allow_html=True)

    if uploaded_file and run_btn:
        with st.spinner("Agent is reasoning…"):
            # Save upload to a temp file so OpenCV can read it
            suffix = Path(uploaded_file.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            model_path_arg = custom_model_path.strip() if custom_model_path.strip() else None
            response = agent_pipeline(tmp_path, model_path=model_path_arg)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        _render_result(response)

    elif not uploaded_file:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🔍</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.95rem;line-height:1.7;">
                Upload a traffic sign image on the left<br>
                and click <b>Analyze Sign</b> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and not run_btn:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2.5rem 2rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">⬅️</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.95rem;">
                Click <b>Analyze Sign</b> to run the agent pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.25);font-size:0.78rem;padding-bottom:1rem;">
    CapsNet Traffic Sign Agent &nbsp;·&nbsp; GTSRB 43-Class Classifier &nbsp;·&nbsp; Inference Only
</div>
""", unsafe_allow_html=True)
