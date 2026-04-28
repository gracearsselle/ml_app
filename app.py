"""
app.py — Page d'accueil / Dashboard principal
Application Streamlit ML dans le Cloud — Groupe 3
"""
import streamlit as st
import time
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import load_css, apply_dark_theme, PLOTLY_DARK

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Cloud Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div class="shimmer-text" style="font-size:1.1rem;font-weight:700;letter-spacing:.02em;">
            ML Cloud App
        </div>
        <div style="color:#64748B;font-size:0.75rem;margin-top:2px;">Groupe 3 — Deep Learning Suite</div>
    </div>
    <hr style="border-color:rgba(124,58,237,0.2);margin:0.75rem 0;">
    """, unsafe_allow_html=True)
    st.markdown("### 🗂️ Navigation")
    st.page_link("app.py",                      label="🏠 Tableau de bord",       use_container_width=True)
    st.page_link("pages/1_maisons.py",          label="🏘️ Prix des Maisons",       use_container_width=True)
    st.page_link("pages/2_diabete.py",          label="🩺 Diabète — Classification",use_container_width=True)
    st.page_link("pages/3_lstm_actions.py",     label="📈 LSTM — Prédiction Actions",use_container_width=True)
    st.page_link("pages/4_rag_pdf.py",          label="📄 RAG — Chatbot PDF",      use_container_width=True)
    st.page_link("pages/5_audio.py",            label="🎙️ Audio STT / TTS",        use_container_width=True)
    st.page_link("pages/6_monitoring.py",       label="📊 Monitoring Langsmith",   use_container_width=True)
    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#475569;text-align:center;">
        KEMAYOU · NZOGNI · TANA · MBOUDA · TSAGNING<br>
        <span style="color:rgba(124,58,237,0.6);">Deep Learning & Cloud ML</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.5rem;">
    <div style="font-size:3rem;margin-bottom:0.5rem;">🧠</div>
    <h1 class="shimmer-text" style="font-size:2.4rem;font-weight:800;margin:0;">
        ML Cloud Dashboard
    </h1>
    <p style="color:#94A3B8;font-size:1.05rem;margin-top:0.5rem;">
        Suite complète d'intelligence artificielle — Modèles ANN, LSTM, RAG &amp; Audio
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Cards ──────────────────────────────────────────────────────────
st.markdown("### 🚀 Modules disponibles")

cards = [
    ("🏘️", "Prix des Maisons", "Régression ANN", "Prédiction du prix à partir des caractéristiques d'un bien immobilier. Modèle ANN entraîné sur Kaggle House Data.", "#7C3AED", "pages/1_maisons.py"),
    ("🩺", "Diabète", "Classification ANN", "Diagnostic de diabète basé sur des indicateurs de santé. Sortie probabiliste avec jauge de risque.", "#10B981", "pages/2_diabete.py"),
    ("📈", "Actions Bourse", "LSTM Prédictif", "Prédiction LSTM des cours d'AMZN, INTC, AMD, GE. Chatbot IA intégré & TimeForecasting HuggingFace.", "#F59E0B", "pages/3_lstm_actions.py"),
    ("📄", "RAG PDF", "LangChain + Chroma", "Chargez un PDF, posez des questions en langage naturel. Réponses sourcées avec ChromaDB et LangChain.", "#3B82F6", "pages/4_rag_pdf.py"),
    ("🎙️", "Audio STT/TTS", "Gemini Live + Groq", "Parlez ou uploadez un audio. Transcription Groq Whisper + réponse vocale Gemini Live.", "#EC4899", "pages/5_audio.py"),
    ("📊", "Monitoring", "Langsmith Analytics", "Surveillance des appels LLM, latences, coûts et traces complètes via Langsmith.", "#06B6D4", "pages/6_monitoring.py"),
]

cols = st.columns(3)
for i, (icon, title, subtitle, desc, color, page) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="ml-card" style="border-color:{color}33;min-height:180px;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700;color:#E2E8F0;font-size:1rem;">{title}</div>
            <div style="color:{color};font-size:0.78rem;font-weight:600;text-transform:uppercase;
                 letter-spacing:.05em;margin:2px 0 8px;">{subtitle}</div>
            <div style="color:#94A3B8;font-size:0.82rem;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Ouvrir {title}", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Stats Overview ────────────────────────────────────────────────────────────
st.markdown("### 📊 Vue d'ensemble")

col1, col2, col3, col4, col5 = st.columns(5)
stats = [
    ("Modèles déployés", "6", "#7C3AED"),
    ("Architectures", "ANN + LSTM", "#10B981"),
    ("Entreprises LSTM", "4", "#F59E0B"),
    ("Fenêtre prédiction", "3–12 mois", "#3B82F6"),
    ("Stack RAG", "LangChain", "#EC4899"),
]
for col, (label, val, color) in zip([col1,col2,col3,col4,col5], stats):
    with col:
        st.markdown(f"""
        <div class="ml-card" style="text-align:center;border-color:{color}33;padding:1rem;">
            <div style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:.06em;">{label}</div>
            <div style="font-size:1.2rem;font-weight:700;color:{color};margin-top:4px;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown("### 🕸️ Profil des capacités")
col_r, col_t = st.columns([1, 1])

with col_r:
    categories = ["Régression", "Classification", "Séries temp.", "NLP / RAG", "Audio", "Monitoring"]
    values     = [92, 88, 85, 80, 78, 75]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="#7C3AED", width=2),
        marker=dict(color="#A78BFA", size=6),
        name="Capacités ML"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(26,26,46,0.8)",
            radialaxis=dict(visible=True, range=[0,100], tickcolor="#475569", gridcolor="rgba(124,58,237,0.15)"),
            angularaxis=dict(tickcolor="#94A3B8", gridcolor="rgba(124,58,237,0.1)"),
        ),
        showlegend=False, height=320,
        **PLOTLY_DARK
    )
    st.plotly_chart(fig, use_container_width=True)

with col_t:
    st.markdown("""
    <div style="padding:1rem 0;">
        <div style="color:#94A3B8;font-size:0.9rem;line-height:1.8;">
            <p>Cette application intègre <strong style="color:#A78BFA;">6 modules</strong> de machine learning 
            avancés, couvrant les principales tâches de l'IA moderne :</p>
            <ul style="color:#94A3B8;padding-left:1.2rem;">
                <li><strong style="color:#7C3AED;">Régression</strong> — ANN pour prédiction immobilière</li>
                <li><strong style="color:#10B981;">Classification</strong> — Diagnostic médical diabète</li>
                <li><strong style="color:#F59E0B;">Séries temporelles</strong> — LSTM multi-actions en bourse</li>
                <li><strong style="color:#3B82F6;">NLP / RAG</strong> — Q&amp;A sur documents PDF</li>
                <li><strong style="color:#EC4899;">Audio</strong> — STT/TTS via Gemini &amp; Groq</li>
                <li><strong style="color:#06B6D4;">Monitoring</strong> — Observabilité LLM Langsmith</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("### 🛠️ Stack technologique")
techs = [
    ("PyTorch", "🔥", "Modèles ANN"),
    ("TensorFlow", "🤖", "LSTM .h5"),
    ("LangChain", "🔗", "RAG Pipeline"),
    ("ChromaDB", "🗄️", "Vectorstore"),
    ("HuggingFace", "🤗", "TimeForecasting"),
    ("Gemini API", "✨", "Audio TTS"),
    ("Groq", "⚡", "Whisper STT"),
    ("Langsmith", "🔭", "Monitoring"),
    ("Plotly", "📊", "Visualisation"),
    ("yfinance", "💹", "Données bourse"),
]
t_cols = st.columns(5)
for i, (name, icon, role) in enumerate(techs):
    with t_cols[i % 5]:
        st.markdown(f"""
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.15);
             border-radius:10px;padding:10px;text-align:center;margin-bottom:8px;">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-weight:600;color:#E2E8F0;font-size:0.82rem;">{name}</div>
            <div style="color:#64748B;font-size:0.72rem;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""<br>""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem;border-top:1px solid rgba(124,58,237,0.15);color:#475569;font-size:0.78rem;">
    <strong style="color:#7C3AED;">ML Cloud App</strong> — Groupe 3 &nbsp;|&nbsp;
    KEMAYOU · NZOGNI · TANA · MBOUDA · TSAGNING &nbsp;|&nbsp;
    Deep Learning &amp; Machine Learning dans le Cloud
</div>
""", unsafe_allow_html=True)
