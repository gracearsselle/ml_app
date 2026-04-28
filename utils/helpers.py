"""
utils/helpers.py — Shared utilities for the ML Cloud App
"""
import streamlit as st
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Theme & CSS ───────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"

def load_css():
    """Inject global CSS into Streamlit app."""
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def page_banner(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled page header banner."""
    st.markdown(f"""
    <div class="page-banner animate-in">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label: str, value, delta=None, unit: str = ""):
    """Render a styled metric card via HTML."""
    delta_html = ""
    if delta is not None:
        color = "#10B981" if delta >= 0 else "#EF4444"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color:{color};font-size:0.8rem;margin-top:4px">{arrow} {abs(delta):.4f}</div>'
    st.markdown(f"""
    <div class="ml-card" style="text-align:center;">
        <div class="ml-card-title">{label}</div>
        <div class="ml-card-value">{value}{unit}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def badge(text: str, kind: str = "info"):
    """Render inline status badge."""
    return f'<span class="badge badge-{kind}">{text}</span>'

def animated_progress(label: str, steps: list):
    """Run a multi-step progress bar with labels."""
    prog = st.progress(0, text=label)
    for i, step in enumerate(steps):
        time.sleep(0.3)
        prog.progress((i + 1) / len(steps), text=step)
    time.sleep(0.2)
    prog.empty()

def sidebar_info(model_name: str, status: str = "Loaded", color: str = "#10B981"):
    """Show model status in sidebar."""
    st.sidebar.markdown(f"""
    <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
         border-radius:10px;padding:10px 14px;margin-bottom:10px;">
        <div style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:.06em;">Modèle actif</div>
        <div style="font-weight:600;color:#E2E8F0;margin:2px 0;">{model_name}</div>
        <div style="color:{color};font-size:0.78rem;">● {status}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Plotly Theme ──────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    plot_bgcolor  = "rgba(15,15,26,0)",
    paper_bgcolor = "rgba(15,15,26,0)",
    font_color    = "#E2E8F0",
    font_family   = "Inter",
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    colorway=["#7C3AED","#10B981","#F59E0B","#EF4444","#3B82F6","#EC4899","#06B6D4"],
)

def apply_dark_theme(fig):
    """Apply the dark theme layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_DARK)
    return fig

def plotly_line(df, x, y, title="", color="#7C3AED", name=""):
    """Quick dark-themed line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(124,58,237,0.07)",
        name=name or y,
    ))
    fig.update_layout(title=title, **PLOTLY_DARK)
    return fig

def plotly_gauge(value: float, title: str = "", max_val: float = 1.0,
                 thresholds=(0.5, 0.75)):
    """Gauge chart for probability display."""
    low, high = thresholds
    color = "#EF4444" if value < low else ("#F59E0B" if value < high else "#10B981")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#E2E8F0"}},
        title={"text": title, "font": {"color": "#A78BFA", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
            "bar":  {"color": color},
            "bgcolor": "#1A1A2E",
            "bordercolor": "rgba(124,58,237,0.3)",
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.08)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.08)"},
                {"range": [75,100], "color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": value * 100},
        }
    ))
    fig.update_layout(height=280, **PLOTLY_DARK)
    return fig

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pytorch_model(path: str, model_class, **kwargs):
    """Load a PyTorch .pth model with caching."""
    import torch
    model = model_class(**kwargs)
    
    # MODIFIEZ CETTE LIGNE : remplacez weights_only=True par False
    state = torch.load(path, map_location="cpu", weights_only=False) 
    
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    """Load a Keras/TF .h5 model with caching."""
    import tensorflow as tf
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    """Load a sklearn scaler from pickle."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Misc ──────────────────────────────────────────────────────────────────────
def show_error(msg: str):
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.35);
         border-radius:10px;padding:12px 16px;color:#FCA5A5;">
        ⚠️ {msg}
    </div>
    """, unsafe_allow_html=True)

def show_success(msg: str):
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.35);
         border-radius:10px;padding:12px 16px;color:#6EE7B7;">
        ✅ {msg}
    </div>
    """, unsafe_allow_html=True)

def show_info(msg: str):
    st.markdown(f"""
    <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.3);
         border-radius:10px;padding:12px 16px;color:#C4B5FD;">
        ℹ️ {msg}
    </div>
    """, unsafe_allow_html=True)

def model_path_exists(path: str) -> bool:
    """Check if model file exists and warn if not."""
    if not os.path.exists(path):
        show_error(f"Modèle introuvable : `{path}` — veuillez d'abord entraîner ou déposer le modèle.")
        return False
    return True

def format_currency(value: float, symbol: str = "$") -> str:
    return f"{symbol}{value:,.2f}"
