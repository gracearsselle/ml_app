import streamlit as st
import pandas as pd
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoring LLM",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Monitoring LLM Dashboard")
st.markdown("Suivi basique des performances et activités du système IA")

# ─────────────────────────────────────────────
# FAKE DATA (tu pourras remplacer après)
# ─────────────────────────────────────────────
data = {
    "date": pd.date_range("2026-01-01", periods=10),
    "latence_ms": [120, 200, 150, 300, 180, 220, 170, 260, 190, 210],
    "cout_usd": [0.01, 0.02, 0.015, 0.03, 0.018, 0.025, 0.02, 0.028, 0.017, 0.022],
    "requetes": [5, 8, 6, 10, 7, 9, 6, 11, 7, 8]
}

df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# DASHBOARD STATS
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Latence moyenne (ms)", f"{df['latence_ms'].mean():.2f}")

with col2:
    st.metric("Coût total ($)", f"{df['cout_usd'].sum():.3f}")

with col3:
    st.metric("Requêtes totales", df["requetes"].sum())

st.divider()

# ─────────────────────────────────────────────
# GRAPHIQUES SIMPLES
# ─────────────────────────────────────────────
st.subheader("📈 Évolution des performances")

st.line_chart(df.set_index("date")[["latence_ms"]])
st.line_chart(df.set_index("date")[["cout_usd"]])

# ─────────────────────────────────────────────
# TABLE HISTORIQUE
# ─────────────────────────────────────────────
st.subheader("📜 Historique des requêtes")
st.dataframe(df)

# ─────────────────────────────────────────────
# SECTION FUTURE LANGSMITH
# ─────────────────────────────────────────────
st.subheader("🔮 LangSmith (future intégration)")
st.info("Ici tu pourras connecter LangSmith API plus tard pour logs LLM réels.")

st.markdown("""
- Traces LLM  
- Latence API  
- Coûts OpenAI / Gemini  
- Historique des prompts  
""")

# ─────────────────────────────────────────────
# SIMULATION LIVE
# ─────────────────────────────────────────────
if st.button("🔄 Simuler activité"):
    with st.spinner("Analyse en cours..."):
        time.sleep(2)
    st.success("Simulation terminée ✔️")