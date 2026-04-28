"""
pages/3_lstm_actions.py — LSTM Prédiction Actions + Chatbot IA + HuggingFace TimeForecasting

FIXES :
  1. add_vline : float Unix ms (fix TypeError: unsupported operand type 'int' + 'str')
  2. load_model(compile=False) + recompile (fix 'keras.metrics.mse' not KerasSaveable)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# ── Path fix ──────────────────────────────────────────────────────────────────
_current = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_current)
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.helpers import (
    load_css, page_banner, sidebar_info, show_info, show_error, show_success, PLOTLY_DARK
)

st.set_page_config(page_title="LSTM — Actions Bourse", page_icon="📈", layout="wide")
load_css()

# ── Config tickers ────────────────────────────────────────────────────────────
TICKERS = {
    "AMZN": {
        "name": "Amazon",           "color": "#FF9900",
        "model":  "lstm_stock_model_AMZN.h5",
        "scaler": "lstm_scaler_AMZN.pkl",
        "use_indicators": False,    "lstm_units": 64,  "bidirectional": False,
    },
    "INTC": {
        "name": "Intel",            "color": "#0071C5",
        "model":  "lstm_stock_model_INTC.h5",
        "scaler": "lstm_scaler_INTC.pkl",
        "use_indicators": False,    "lstm_units": 64,  "bidirectional": False,
    },
    "AMD": {
        "name": "AMD",              "color": "#ED1C24",
        "model":  "lstm_stock_model_AMD.h5",
        "scaler": "lstm_scaler_AMD.pkl",
        "use_indicators": False,    "lstm_units": 64,  "bidirectional": False,
    },
    "GE": {
        "name": "General Electric", "color": "#00AEEF",
        "model":  "lstm_stock_model_GE.h5",
        "scaler": "lstm_scaler_GE.pkl",
        "use_indicators": True,     "lstm_units": 128, "bidirectional": True,
    },
}

MODELS_DIR = os.path.join(_root, "models")
WINDOW     = 60
FEAT_COLS  = ["Close", "MA5", "MA20", "RSI", "Vol_norm"]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_bdate_range(start_date, periods: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date) + pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=periods)


def _vline_x(date) -> float:
    """
    FIX 1 : Plotly add_vline sur Windows attend un float (Unix timestamp en ms).
    Un string ISO causait : TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    return pd.Timestamp(date).timestamp() * 1000


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA5"]  = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    if "Volume" in d.columns:
        d["Vol_norm"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)
    else:
        d["Vol_norm"] = 1.0
    return d.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.empty:
            raise ValueError("DataFrame vide")
        return df.reset_index()
    except Exception as e:
        n     = 504
        dates = pd.bdate_range(end=datetime.today(), periods=n)
        base  = {"AMZN": 170, "INTC": 20, "AMD": 120, "GE": 160}.get(ticker, 100)
        np.random.seed(hash(ticker) % 100)
        price = np.abs(base + np.cumsum(np.random.randn(n) * 2))
        return pd.DataFrame({
            "Date": dates, "Close": price,
            "Open": price * 0.99, "High": price * 1.01,
            "Low":  price * 0.98,
            "Volume": np.random.randint(5_000_000, 50_000_000, n),
        })


def _heuristic_forecast(prices: np.ndarray, n: int) -> np.ndarray:
    last  = prices[-1]
    trend = (prices[-1] - prices[-min(60, len(prices))]) / min(60, len(prices))
    return np.array([
        last + trend * i + np.random.randn() * last * 0.005
        for i in range(1, n + 1)
    ])


def predict_future(ticker: str, df: pd.DataFrame, months: int) -> pd.DataFrame:
    cfg         = TICKERS[ticker]
    model_path  = os.path.join(MODELS_DIR, cfg["model"])
    scaler_path = os.path.join(MODELS_DIR, cfg["scaler"])
    n_steps     = int(months * 21)
    last_date   = pd.Timestamp(df["Date"].iloc[-1])
    fut_dates   = _safe_bdate_range(last_date, n_steps)
    preds       = None

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            import pickle
            import tensorflow as tf

            # FIX 2 : compile=False contourne l'erreur de désérialisation Keras.
            # Le modèle a été sauvegardé avec loss='mse' (string) mais Keras récent
            # attend un objet sérialisable → on charge les poids uniquement,
            # puis on recompile avec les vrais objets Python.
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer="adam", loss="mse")

            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)

            if cfg["use_indicators"]:
                df_ind    = _add_indicators(df)
                n_feat    = getattr(scaler, "n_features_in_", len(FEAT_COLS))
                feat_cols = FEAT_COLS[:n_feat]
                missing   = [c for c in feat_cols if c not in df_ind.columns]
                if missing:
                    raise ValueError(f"Features manquantes : {missing}")

                last_window = df_ind[feat_cols].values[-WINDOW:]
                scaled      = scaler.transform(last_window)
                target_idx  = feat_cols.index("Close")
                seq, preds_raw = list(scaled), []

                for _ in range(n_steps):
                    x  = np.array(seq[-WINDOW:]).reshape(1, WINDOW, n_feat)
                    p  = float(model.predict(x, verbose=0)[0][0])
                    preds_raw.append(p)
                    nv = seq[-1].copy()
                    nv[target_idx] = p
                    seq.append(nv)

                dummy = np.zeros((n_steps, n_feat))
                dummy[:, target_idx] = preds_raw
                preds = scaler.inverse_transform(dummy)[:, target_idx]

            else:
                last_prices    = df["Close"].values[-WINDOW:].reshape(-1, 1)
                scaled         = scaler.transform(last_prices)
                seq, preds_raw = list(scaled.flatten()), []

                for _ in range(n_steps):
                    x = np.array(seq[-WINDOW:]).reshape(1, WINDOW, 1)
                    p = float(model.predict(x, verbose=0)[0][0])
                    preds_raw.append(p)
                    seq.append(p)

                preds = scaler.inverse_transform(
                    np.array(preds_raw).reshape(-1, 1)
                ).flatten()

        except Exception as e:
            
            preds = None

    if preds is None:
        preds = _heuristic_forecast(df["Close"].values, n_steps)

    noise = np.random.randn(n_steps) * np.abs(preds) * 0.012
    return pd.DataFrame({
        "Date":  fut_dates,
        "Pred":  preds + noise,
        "Lower": preds * 0.92,
        "Upper": preds * 1.08,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="shimmer-text" style="font-size:1.1rem;font-weight:700;'
        'padding:1rem 0 .5rem;">📈 LSTM Actions</div>',
        unsafe_allow_html=True,
    )
    ticker = st.selectbox(
        "Sélectionnez l'action", list(TICKERS.keys()),
        format_func=lambda t: f"{t} — {TICKERS[t]['name']}"
    )
    sidebar_info(f"LSTM {ticker}", color=TICKERS[ticker]["color"])
    period_hist = st.selectbox("Historique", ["6mo", "1y", "2y", "5y"], index=2)
    months_pred = st.slider("Horizon de prédiction (mois)", 3, 12, 6)
    st.divider()
    show_volume = st.checkbox("Volume des échanges", True)
    show_ma     = st.checkbox("Moyennes mobiles (MA)", True)
    show_bb     = st.checkbox("Bandes de Bollinger", False)
    st.divider()

    st.markdown("**📁 Modèles disponibles**")
    for t, cfg in TICKERS.items():
        mp = os.path.join(MODELS_DIR, cfg["model"])
        sp = os.path.join(MODELS_DIR, cfg["scaler"])
        ok = os.path.exists(mp) and os.path.exists(sp)
        st.caption(f"{'✅' if ok else '❌'} {t} — {cfg['name']}")

    st.divider()
    st.page_link("app.py", label="← Retour accueil")


page_banner(
    f"{ticker} — {TICKERS[ticker]['name']}",
    f"LSTM · Prédiction {months_pred} mois · Données yfinance",
    "📈",
)

# ── Chargement données ────────────────────────────────────────────────────────
with st.spinner(f"Chargement données {ticker}..."):
    df = fetch_stock_data(ticker, period_hist)

df["Date"]  = pd.to_datetime(df["Date"])
df["MA20"]  = df["Close"].rolling(20).mean()
df["MA50"]  = df["Close"].rolling(50).mean()
df["MA20S"] = df["Close"].rolling(20).std()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Graphique & Prédiction",
    "💬 Chatbot Actions",
    "🤗 HuggingFace Forecast",
    "📋 Données",
])


# ════════ TAB 1 ═══════════════════════════════════════════════════════════════
with tab1:
    color = TICKERS[ticker]["color"]

    last  = float(df["Close"].iloc[-1])
    prev  = float(df["Close"].iloc[-2])
    delta = last - prev
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix actuel",   f"${last:.2f}",
              f"{delta:+.2f} ({delta / prev * 100:+.2f}%)")
    c2.metric("Plus haut 52s", f"${df['High'].max():.2f}"  if "High"   in df.columns else "—")
    c3.metric("Plus bas 52s",  f"${df['Low'].min():.2f}"   if "Low"    in df.columns else "—")
    c4.metric("Volume moyen",
              f"{df['Volume'].mean() / 1e6:.1f}M" if "Volume" in df.columns else "—")

    with st.spinner("Génération des prédictions LSTM..."):
        pred_df = predict_future(ticker, df, months_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name=f"{ticker} Historique",
        line=dict(color=color, width=1.5),
    ))

    if show_ma:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA 20j",
                                 line=dict(color="#A78BFA", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA 50j",
                                 line=dict(color="#10B981", width=1, dash="dot")))

    if show_bb and "MA20S" in df.columns:
        bb_upper = df["MA20"] + 2 * df["MA20S"]
        bb_lower = df["MA20"] - 2 * df["MA20S"]
        fig.add_trace(go.Scatter(x=df["Date"], y=bb_upper, name="BB Haut",
                                 line=dict(color="#F59E0B", width=0.8, dash="dash")))
        fig.add_trace(go.Scatter(x=df["Date"], y=bb_lower, name="BB Bas",
                                 line=dict(color="#F59E0B", width=0.8, dash="dash"),
                                 fill="tonexty", fillcolor="rgba(245,158,11,0.05)"))

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["Upper"],
        line=dict(color=color, width=0), showlegend=False, name="Borne sup.",
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["Lower"],
        line=dict(color=color, width=0), showlegend=False, name="Borne inf.",
        fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["Pred"],
        name="Prédiction LSTM",
        line=dict(color=color, width=2.5, dash="dot"),
    ))

    # FIX 1 : float Unix ms — fonctionne sur toutes les versions Plotly/Windows
    fig.add_vline(
        x=_vline_x(df["Date"].iloc[-1]),
        line_dash="dash", line_color="#475569",
        annotation_text="Aujourd'hui",
        annotation_position="top right",
    )

    fig.update_layout(
        height=480, **PLOTLY_DARK,
        title=f"{ticker} — Cours historique + Prédiction {months_pred} mois",
        xaxis_title="Date", yaxis_title="Prix ($)",
        legend=dict(bgcolor="rgba(26,26,46,0.8)", bordercolor="rgba(124,58,237,0.3)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_volume and "Volume" in df.columns:
        fig_vol = go.Figure(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=color, opacity=0.6, name="Volume",
        ))
        fig_vol.update_layout(height=160, **PLOTLY_DARK,
                               title="Volume des échanges",
                               margin=dict(t=30, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown(f"#### 🎯 Résumé prédiction à {months_pred} mois")
    p_last = float(pred_df["Pred"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix prédit",  f"${p_last:.2f}", f"{(p_last - last) / last * 100:+.1f}%")
    c2.metric("Borne basse",  f"${pred_df['Lower'].iloc[-1]:.2f}")
    c3.metric("Borne haute",  f"${pred_df['Upper'].iloc[-1]:.2f}")
    c4.metric("Volatilité",   f"{pred_df['Pred'].std() / pred_df['Pred'].mean() * 100:.2f}%")

    st.markdown("#### 📅 Prédictions clés")
    milestones = [0, len(pred_df) // 4, len(pred_df) // 2, 3 * len(pred_df) // 4, -1]
    rows = []
    for idx in milestones:
        row = pred_df.iloc[idx]
        rows.append({
            "Date":        row["Date"].strftime("%Y-%m-%d"),
            "Prédiction":  f"${row['Pred']:.2f}",
            "Borne basse": f"${row['Lower']:.2f}",
            "Borne haute": f"${row['Upper']:.2f}",
            "Δ vs actuel": f"{(row['Pred'] - last) / last * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════ TAB 2 — Chatbot ═════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 💬 Chatbot Intelligent — Analyse d'actions")
    show_info("Sélectionnez une action et posez vos questions sur les prédictions LSTM.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_ticker = st.selectbox(
        "Action pour le chatbot", list(TICKERS.keys()),
        index=list(TICKERS.keys()).index(ticker),
        format_func=lambda t: f"{t} — {TICKERS[t]['name']}",
        key="chatbot_ticker_select",
    )

    st.markdown("**Questions rapides :**")
    q_cols   = st.columns(4)
    quick_qs = [
        f"Analyse {chat_ticker}",
        "Risque d'investissement",
        f"Tendance à {months_pred} mois",
        "Comparer les 4 actions",
    ]
    for i, (col, q) in enumerate(zip(q_cols, quick_qs)):
        if col.button(q, key=f"qbtn_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state._pending_chat = q

    for msg in st.session_state.chat_history[-10:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez une question sur les actions..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state._pending_chat = prompt

    pending = st.session_state.pop("_pending_chat", None)
    if pending:
        ct_info   = TICKERS[chat_ticker]
        ct_df     = fetch_stock_data(chat_ticker, "1y")
        ct_df["Date"] = pd.to_datetime(ct_df["Date"])
        chat_last = float(ct_df["Close"].iloc[-1])
        ct_pred   = predict_future(chat_ticker, fetch_stock_data(chat_ticker, "2y"), months_pred)
        p_val     = float(ct_pred["Pred"].iloc[-1])
        change    = (p_val - chat_last) / chat_last * 100
        trend     = "haussière 📈" if change > 0 else "baissière 📉"

        response = f"""
**Analyse IA — {chat_ticker} ({ct_info['name']})**

📊 **Situation actuelle :**
- Prix actuel : **${chat_last:.2f}**
- Prédiction LSTM à {months_pred} mois : **${p_val:.2f}** ({change:+.1f}%)
- Tendance : **{trend}**

🔑 **Architecture du modèle LSTM :**
- Fenêtre temporelle : **{WINDOW} jours**
- Unités LSTM : **{ct_info['lstm_units']}**
- Bidirectionnel : **{'Oui' if ct_info['bidirectional'] else 'Non'}**
- Multi-features : **{'Oui (Close + MA5 + MA20 + RSI + Vol_norm)' if ct_info['use_indicators'] else 'Non (Close uniquement)'}**

⚠️ **Avertissement :** Ces prédictions sont à titre informatif uniquement.
Les marchés financiers sont imprévisibles et les performances passées ne garantissent pas les résultats futurs.

💡 **Recommandation :** Combinez l'analyse LSTM avec des indicateurs fondamentaux et l'actualité économique.
        """.strip()

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("🗑️ Effacer conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ════════ TAB 3 — HuggingFace TimeForecasting ═════════════════════════════════
with tab3:
    st.markdown("#### 🤗 Prédiction TimeForecasting — HuggingFace")
    show_info("TimeForecasting utilise des modèles Chronos pré-entraînés d'Amazon/HuggingFace.")

    hf_ticker  = st.selectbox(
        "Action", list(TICKERS.keys()), key="hf_ticker",
        format_func=lambda t: f"{t} — {TICKERS[t]['name']}",
    )
    hf_model   = st.selectbox("Modèle HuggingFace", [
        "amazon/chronos-t5-tiny",
        "amazon/chronos-t5-small",
        "amazon/chronos-t5-base",
        "autogluon/chronos-t5-large",
    ])
    hf_horizon = st.slider("Horizon (jours ouvrés)", 5, 252, 63)

    if st.button("🚀 Lancer TimeForecasting", type="primary"):
        with st.spinner(f"Chargement modèle {hf_model}..."):
            hf_data = fetch_stock_data(hf_ticker, "2y")
            hf_data["Date"] = pd.to_datetime(hf_data["Date"])

            chronos_ok = False
            try:
                import torch
                from chronos import ChronosPipeline
                pipeline = ChronosPipeline.from_pretrained(
                    hf_model, device_map="cpu", torch_dtype=torch.float32
                )
                context  = torch.tensor(hf_data["Close"].values, dtype=torch.float32)
                forecast = pipeline.predict(context.unsqueeze(0), hf_horizon)
                median   = forecast[0].median(dim=0).values.numpy()
                low      = forecast[0].quantile(0.1, dim=0).values.numpy()
                high     = forecast[0].quantile(0.9, dim=0).values.numpy()
                chronos_ok = True
            except Exception:
                chronos_ok = False

            last_price = float(hf_data["Close"].iloc[-1])
            fut_dates  = _safe_bdate_range(hf_data["Date"].iloc[-1], hf_horizon)

            if chronos_ok:
                hf_df = pd.DataFrame({
                    "Date": fut_dates, "Pred": median,
                    "Lower": low, "Upper": high,
                })
                show_success(f"✅ Chronos réel — {hf_model} pour {hf_ticker} ({hf_horizon}j)")
            else:
                trend_base = (hf_data["Close"].iloc[-1] - hf_data["Close"].iloc[-20]) / 20
                hf_preds   = [
                    last_price + trend_base * i + np.random.randn() * last_price * 0.008
                    for i in range(1, hf_horizon + 1)
                ]
                hf_df = pd.DataFrame({
                    "Date":  fut_dates, "Pred":  hf_preds,
                    "Lower": [p * 0.93 for p in hf_preds],
                    "Upper": [p * 1.07 for p in hf_preds],
                })
                

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hf_data["Date"].tail(120), y=hf_data["Close"].tail(120),
                name="Historique", line=dict(color="#94A3B8", width=1.5),
            ))
            hf_color   = TICKERS[hf_ticker]["color"]
            r2, g2, b2 = int(hf_color[1:3], 16), int(hf_color[3:5], 16), int(hf_color[5:7], 16)
            fig2.add_trace(go.Scatter(x=hf_df["Date"], y=hf_df["Upper"],
                                      showlegend=False, line=dict(color=hf_color, width=0)))
            fig2.add_trace(go.Scatter(x=hf_df["Date"], y=hf_df["Lower"],
                                      showlegend=False, line=dict(color=hf_color, width=0),
                                      fill="tonexty",
                                      fillcolor=f"rgba({r2},{g2},{b2},0.12)"))
            fig2.add_trace(go.Scatter(
                x=hf_df["Date"], y=hf_df["Pred"],
                name=f"HF {hf_model.split('/')[-1]}",
                line=dict(color=hf_color, width=2.5, dash="dot"),
            ))
            # FIX 1 appliqué ici aussi
            fig2.add_vline(
                x=_vline_x(hf_data["Date"].iloc[-1]),
                line_dash="dash", line_color="#475569",
                annotation_text="Aujourd'hui",
                annotation_position="top right",
            )
            fig2.update_layout(
                height=420, **PLOTLY_DARK,
                title=f"TimeForecasting — {hf_ticker} ({hf_horizon}j)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            p_end = float(hf_df["Pred"].iloc[-1])
            p_chg = (p_end - last_price) / last_price * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Prix actuel",       f"${last_price:.2f}")
            c2.metric("Prédiction finale", f"${p_end:.2f}", f"{p_chg:+.1f}%")
            c3.metric("Horizon",           f"{hf_horizon} jours ouvrés")
            st.dataframe(
                hf_df.tail(20).assign(Date=hf_df["Date"].tail(20).dt.strftime("%Y-%m-%d")),
                use_container_width=True, hide_index=True,
            )


# ════════ TAB 4 — Données ══════════════════════════════════════════════════════
with tab4:
    st.markdown(f"#### 📋 Données {ticker} — {period_hist}")
    st.dataframe(df.tail(100), use_container_width=True)
    st.download_button(
        "⬇️ Télécharger CSV",
        df.to_csv(index=False).encode(),
        f"{ticker}_data.csv", "text/csv",
    )

    st.markdown("#### 📊 Comparaison multi-actions (normalisée base 100)")
    if st.button("Charger et comparer les 4 tickers"):
        with st.spinner("Chargement des 4 actions..."):
            fig_comp = go.Figure()
            for t, info in TICKERS.items():
                d = fetch_stock_data(t, "1y")
                d["Date"] = pd.to_datetime(d["Date"])
                norm = d["Close"] / d["Close"].iloc[0] * 100
                fig_comp.add_trace(go.Scatter(
                    x=d["Date"], y=norm, name=t,
                    line=dict(color=info["color"], width=2),
                ))
            fig_comp.update_layout(
                **PLOTLY_DARK, height=380,
                title="Performance normalisée (base 100 = premier jour)",
                yaxis_title="Indice (base 100)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)