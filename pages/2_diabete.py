"""
pages/2_diabete.py — Classification Diabète (ANN)
CORRECTION FINALE :
  - 23 features exactes (21 Kaggle + BMI_category + risk_score)
  - StandardScaler chargé depuis processed/scaler.pkl
  - DiabetesNet reconstruit depuis le config dans best_model.pth
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import sys, os

# ── Path fix robuste pour app multipage ───────────────────────────────────────
_current = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_current)          # pages/ → racine du projet
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.helpers import (load_css, page_banner, plotly_gauge,
                            show_error, show_success, show_info,
                            sidebar_info, PLOTLY_DARK)

st.set_page_config(page_title="Diabète — Classification", page_icon="🩺", layout="wide")
load_css()

# ══════════════════════════════════════════════════════════════════════════════
# DiabetesNet — copie exacte de model_search.py
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn

    class DiabetesNet(nn.Module):
        def __init__(self, input_size, hidden_layers, dropout_rate=0.0,
                     use_batchnorm=False, activation='relu'):
            super(DiabetesNet, self).__init__()
            self.layers      = nn.ModuleList()
            self.batch_norms = nn.ModuleList() if use_batchnorm else None
            self.dropouts    = nn.ModuleList() if dropout_rate > 0 else None
            acts = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.1), 'elu': nn.ELU()}
            self.activation  = acts.get(activation, nn.ReLU())
            prev = input_size
            for h in hidden_layers:
                self.layers.append(nn.Linear(prev, h))
                if use_batchnorm:
                    self.batch_norms.append(nn.BatchNorm1d(h))
                if dropout_rate > 0:
                    self.dropouts.append(nn.Dropout(dropout_rate))
                prev = h
            self.output  = nn.Linear(prev, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                if self.dropouts is not None:
                    x = self.dropouts[i](x)
            return self.sigmoid(self.output(x))

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Chemins des fichiers ──────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(_root, "models",    "best_model.pth")
SCALER_PATH = os.path.join(_root, "models", "scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# Chargement modèle + scaler (une seule fois grâce au cache)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Chargement du modèle ANN…")
def load_model_and_scaler(model_path, scaler_path):
    # ── 1. Checkpoint ─────────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Format .pth invalide. Clés trouvées : {list(checkpoint.keys())}"
        )
    cfg        = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # ── 2. input_size réel depuis les poids de la 1ère couche ─────────────────
    first_w = next((k for k in state_dict if k == "layers.0.weight"), None)
    if first_w is None:
        raise ValueError(f"Clé 'layers.0.weight' introuvable. Clés : {list(state_dict.keys())[:8]}")
    real_input_size = state_dict[first_w].shape[1]   # (out, in) → in

    # ── 3. Reconstruction du modèle ───────────────────────────────────────────
    model = DiabetesNet(
        input_size    = real_input_size,
        hidden_layers = cfg["hidden_layers"],
        dropout_rate  = cfg["dropout"],
        use_batchnorm = cfg["batchnorm"],
        activation    = cfg["activation"],
    )
    model.load_state_dict(state_dict)
    model.eval()

    # ── 4. Scaler ─────────────────────────────────────────────────────────────
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, cfg, checkpoint.get("metrics", {}), real_input_size, scaler


_model_obj     = None
_model_cfg     = {}
_model_metrics = {}
_input_size    = None
_scaler        = None
_load_error    = None

if TORCH_OK and os.path.exists(MODEL_PATH):
    try:
        _model_obj, _model_cfg, _model_metrics, _input_size, _scaler = \
            load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    except Exception as e:
        _load_error = str(e)

# ══════════════════════════════════════════════════════════════════════════════
# Construction du vecteur de 23 features
# Ordre EXACT du preprocessing.py :
#   [21 colonnes Kaggle brutes] + [BMI_category] + [risk_score]
# ══════════════════════════════════════════════════════════════════════════════
BASE_21 = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income",
]

def build_feature_vector(inputs: dict) -> np.ndarray:
    """
    Reproduit exactement feature_engineering() de preprocessing.py :
      1. 21 features brutes Kaggle
      2. BMI_category  : 0 si BMI<18.5 | 1 si <25 | 2 si <30 | 3 si >=30
      3. risk_score    : HighBP + HighChol + (BMI>30)
    Puis applique le StandardScaler si disponible.
    """
    bmi = inputs["BMI"]

    # BMI_category (pd.cut bins=[0,18.5,25,30,50] labels=[0,1,2,3])
    if bmi <= 18.5:
        bmi_cat = 0.0
    elif bmi <= 25:
        bmi_cat = 1.0
    elif bmi <= 30:
        bmi_cat = 2.0
    else:
        bmi_cat = 3.0

    # risk_score
    risk_score = float(inputs["HighBP"] + inputs["HighChol"] + (bmi > 30))

    # Vecteur ordonné : 21 + BMI_category + risk_score = 23
    vec = np.array(
        [inputs[f] for f in BASE_21] + [bmi_cat, risk_score],
        dtype=np.float32
    ).reshape(1, -1)

    # Appliquer le scaler si dispo (INDISPENSABLE car le modèle a été
    # entraîné sur des données StandardScaler-ées)
    if _scaler is not None:
        vec = _scaler.transform(vec).astype(np.float32)

    return vec


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="shimmer-text" style="font-size:1.1rem;font-weight:700;'
        'padding:1rem 0 .5rem;">🩺 Diabète ANN</div>',
        unsafe_allow_html=True,
    )
    sidebar_info("Diabetes ANN (best_model.pth)", color="#10B981")
    st.divider()
    threshold = st.slider("Seuil de décision", 0.1, 0.9, 0.5, 0.05)
    st.caption("Probabilité au-dessus du seuil → Diabétique")
    st.divider()
    st.page_link("app.py", label="← Retour accueil")
    st.divider()

    # ── Statut modèle ─────────────────────────────────────────────────────────
    if not TORCH_OK:
        st.error("⚠️ PyTorch non installé")
    elif not os.path.exists(MODEL_PATH):
        st.warning("Modèle introuvable :\n`models/best_model.pth`")
    elif _load_error:
        st.error(f"Erreur chargement :\n{_load_error}")
    else:
        arch = _model_cfg.get("hidden_layers", "?")
        st.success(f"✅ Modèle chargé\n`{arch}`\n{_input_size} features")
        scaler_ok = "✅ Scaler OK" if _scaler else "⚠️ Scaler absent"
        st.caption(scaler_ok)
        if _model_metrics:
            st.caption(
                f"F1 : **{_model_metrics.get('f1',0):.3f}** | "
                f"AUC : **{_model_metrics.get('auc_roc',0):.3f}** | "
                f"Acc : **{_model_metrics.get('accuracy',0):.3f}**"
            )

page_banner("Diagnostic Diabète",
            "Classification ANN — Health Indicators Dataset (Kaggle)", "🩺")

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Diagnostic", "📊 Analyse Dataset", "📂 Batch", "📈 Performance"]
)

# ════════ TAB 1 — Diagnostic ══════════════════════════════════════════════════
with tab1:
    st.markdown("#### Saisie des indicateurs de santé")
    inputs = {}

    with st.expander("🫀 Antécédents médicaux", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["HighBP"]               = int(st.toggle("Hypertension", False))
            inputs["HighChol"]             = int(st.toggle("Cholestérol élevé", False))
            inputs["CholCheck"]            = int(st.toggle("Bilan cholestérol récent", True))
        with c2:
            inputs["Stroke"]               = int(st.toggle("Antécédent d'AVC", False))
            inputs["HeartDiseaseorAttack"] = int(st.toggle("Maladie cardiaque", False))
        with c3:
            inputs["Smoker"]               = int(st.toggle("Fumeur", False))
            inputs["HvyAlcoholConsump"]    = int(st.toggle("Alcool excessif", False))

    with st.expander("📏 Mesures biologiques", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            inputs["BMI"] = st.slider("IMC (Body Mass Index)", 10.0, 60.0, 26.0, 0.5)
            bmi_val = inputs["BMI"]
            bmi_cat = ("Insuffisance pondérale" if bmi_val < 18.5 else
                       "Poids normal"           if bmi_val < 25   else
                       "Surpoids"               if bmi_val < 30   else "Obésité")
            bmi_col = "#3B82F6" if bmi_val < 25 else "#F59E0B" if bmi_val < 30 else "#EF4444"
            st.markdown(
                f'<span style="color:{bmi_col};font-size:.85rem;">IMC : {bmi_cat}</span>',
                unsafe_allow_html=True,
            )
        with c2:
            inputs["MentHlth"] = st.slider("Jours santé mentale affectée (30j)", 0, 30, 0)
            inputs["PhysHlth"] = st.slider("Jours santé physique affectée (30j)", 0, 30, 0)

    with st.expander("🧬 Mode de vie & Démographie"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["PhysActivity"] = int(st.toggle("Activité physique régulière", True))
            inputs["Fruits"]       = int(st.toggle("Consomme fruits quotidien", True))
            inputs["Veggies"]      = int(st.toggle("Consomme légumes quotidien", True))
            inputs["DiffWalk"]     = int(st.toggle("Difficulté à marcher", False))
        with c2:
            inputs["AnyHealthcare"] = int(st.toggle("Couverture santé", True))
            inputs["NoDocbcCost"]   = int(st.toggle("Renoncé médecin (coût)", False))
            inputs["Sex"]           = int(
                st.selectbox("Sexe", ["Femme (0)", "Homme (1)"]) == "Homme (1)"
            )
        with c3:
            inputs["GenHlth"]   = st.slider("Santé générale (1=Excellente, 5=Mauvaise)", 1, 5, 3)
            inputs["Age"]       = st.slider("Classe d'âge (1=18-24 … 13=80+)", 1, 13, 7)
            inputs["Education"] = st.slider("Niveau éducation (1–6)", 1, 6, 4)
            inputs["Income"]    = st.slider("Tranche de revenus (1–8)", 1, 8, 5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Analyser le risque", type="primary", use_container_width=True):

        prob      = None
        used_demo = False

        # ── Prédiction RÉELLE ─────────────────────────────────────────────────
        if _model_obj is not None:
            try:
                feat_vec = build_feature_vector(inputs)   # (1, 23) scalé
                with torch.no_grad():
                    prob = float(
                        _model_obj(torch.tensor(feat_vec, dtype=torch.float32)).item()
                    )
            except Exception as e:
                show_error(f"Erreur prédiction : {e}")

        # ── Fallback heuristique (seulement si le modèle a échoué) ───────────
        if prob is None:
            used_demo = True
            risk = 0.05
            if inputs["HighBP"]:               risk += 0.18
            if inputs["HighChol"]:             risk += 0.12
            if inputs["Stroke"]:               risk += 0.20
            if inputs["HeartDiseaseorAttack"]: risk += 0.18
            if inputs["BMI"] > 30:             risk += 0.15
            if inputs["BMI"] > 35:             risk += 0.10
            if inputs["Smoker"]:               risk += 0.05
            if inputs["PhysActivity"]:         risk -= 0.07
            if inputs["Fruits"]:               risk -= 0.03
            risk += inputs["Age"] * 0.02
            prob = float(min(max(risk, 0.02), 0.98))

        # ── Affichage ─────────────────────────────────────────────────────────
        if not used_demo:
            show_success("✅ Prédiction réelle — modèle ANN `best_model.pth`")
        else:
            show_info("⚠️ Mode démo — vérifiez que `best_model.pth` est dans `models/`.")

        decision   = prob >= threshold
        risk_label = "🔴 DIABÉTIQUE"  if decision else "🟢 NON DIABÉTIQUE"
        risk_color = "#EF4444"        if decision else "#10B981"
        rgb        = "239,68,68"      if decision else "16,185,129"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(
                plotly_gauge(prob, "Probabilité Diabète", thresholds=(0.4, 0.65)),
                use_container_width=True,
            )
        with col2:
            st.markdown(f"""
            <div style="background:rgba({rgb},0.12);border:2px solid {risk_color};
                 border-radius:14px;padding:2rem;text-align:center;margin-top:1rem;">
                <div style="font-size:2rem;font-weight:800;color:{risk_color};">{risk_label}</div>
                <div style="font-size:1.2rem;color:#E2E8F0;margin-top:.5rem;">
                    Probabilité : <strong>{prob*100:.1f}%</strong>
                </div>
                <div style="color:#94A3B8;font-size:.85rem;margin-top:.5rem;">
                    Seuil : {threshold*100:.0f}% | IMC : {inputs['BMI']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilité", f"{prob*100:.1f}%")
            c2.metric("Seuil",       f"{threshold*100:.0f}%")
            c3.metric("IMC",         f"{inputs['BMI']:.1f}")

        # ── Facteurs de risque ────────────────────────────────────────────────
        st.markdown("#### 🔑 Facteurs de risque identifiés")
        risk_factors = {
            "Hypertension":   inputs["HighBP"]               * 0.18,
            "Cholestérol":    inputs["HighChol"]             * 0.12,
            "AVC":            inputs["Stroke"]               * 0.20,
            "IMC élevé":      max(0, (inputs["BMI"] - 25) / 35 * 0.25),
            "Tabac":          inputs["Smoker"]               * 0.05,
            "Activité phys.": -inputs["PhysActivity"]        * 0.07,
            "Cardio":         inputs["HeartDiseaseorAttack"] * 0.18,
            "Âge":            inputs["Age"]                  * 0.018,
        }
        rf_df = (
            pd.DataFrame({"Facteur": list(risk_factors.keys()),
                          "Impact":  list(risk_factors.values())})
            .sort_values("Impact")
        )
        fig = go.Figure(go.Bar(
            x=rf_df["Impact"], y=rf_df["Facteur"], orientation="h",
            marker_color=["#EF4444" if v > 0 else "#10B981" for v in rf_df["Impact"]],
        ))
        fig.update_layout(height=320, **PLOTLY_DARK, xaxis_title="Impact sur le risque")
        st.plotly_chart(fig, use_container_width=True)

# ════════ TAB 2 — Analyse Dataset ═════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Exploration du dataset Diabète")
    up = st.file_uploader("Upload diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
                          type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.success(f"{len(df):,} lignes chargées — {df.shape[1]} colonnes")
    else:
        np.random.seed(0); n = 1000
        df = pd.DataFrame({
            "Diabetes_binary": np.random.binomial(1, 0.5,  n),
            "BMI":             np.random.normal(27, 5, n).clip(15, 50),
            "Age":             np.random.randint(1, 14, n),
            "GenHlth":         np.random.randint(1, 6, n),
            "HighBP":          np.random.binomial(1, 0.32, n),
            "HighChol":        np.random.binomial(1, 0.42, n),
            "PhysActivity":    np.random.binomial(1, 0.72, n),
        })
        show_info("Données de démonstration (50/50 split) — uploadez le vrai CSV Kaggle.")

    c1, c2 = st.columns(2)
    with c1:
        if "Diabetes_binary" in df.columns:
            counts = df["Diabetes_binary"].value_counts()
            fig = go.Figure(go.Pie(
                labels=["Non diabétique", "Diabétique"],
                values=[counts.get(0, 0), counts.get(1, 0)],
                hole=0.45, marker_colors=["#10B981", "#EF4444"],
            ))
            fig.update_layout(title="Distribution classes", height=280, **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df,
                     x="Diabetes_binary" if "Diabetes_binary" in df.columns else None,
                     y="BMI", title="IMC par classe",
                     color_discrete_sequence=["#7C3AED","#EF4444"])
        fig.update_layout(**PLOTLY_DARK, height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.describe(), use_container_width=True)

# ════════ TAB 3 — Batch ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📂 Classification en lot")
    with st.expander("ℹ️ Format CSV attendu"):
        st.markdown("Le CSV doit contenir les **21 colonnes Kaggle** suivantes :")
        st.code(", ".join(BASE_21))
        st.info(
            "Les features `BMI_category` et `risk_score` sont calculées automatiquement. "
            "Le scaler est appliqué automatiquement."
        )

    b = st.file_uploader("CSV patients", type=["csv"], key="diab_batch")
    if b:
        bdf = pd.read_csv(b)
        st.write(f"**{len(bdf)} patients — {bdf.shape[1]} colonnes**")

        if st.button("🚀 Analyser le lot", type="primary"):
            missing_cols = set(BASE_21) - set(bdf.columns)
            if missing_cols:
                show_error(f"Colonnes manquantes : {missing_cols}")
            elif _model_obj is not None:
                try:
                    rows = []
                    for _, row in bdf.iterrows():
                        inp = {f: float(row[f]) for f in BASE_21}
                        rows.append(build_feature_vector(inp).flatten())
                    X_batch = np.array(rows, dtype=np.float32)
                    _model_obj.eval()
                    with torch.no_grad():
                        probs_b = _model_obj(
                            torch.tensor(X_batch)
                        ).numpy().flatten()
                    bdf["prob_diabete"]  = probs_b.round(3)
                    bdf["diagnostic"]    = ["Diabétique" if p >= threshold else "Sain"
                                             for p in probs_b]
                    bdf["confiance_pct"] = [
                        round((p if p >= threshold else 1-p)*100, 1) for p in probs_b
                    ]
                    show_success("✅ Prédictions réelles effectuées")
                except Exception as e:
                    show_error(f"Erreur batch : {e}")
            else:
                # démo
                bdf["prob_diabete"] = np.random.beta(2, 8, len(bdf)).round(3)
                bdf["diagnostic"]   = ["Diabétique" if p >= threshold else "Sain"
                                         for p in bdf["prob_diabete"]]
                show_info("Mode démo — modèle non disponible.")

            if "diagnostic" in bdf.columns:
                n_diab = (bdf["diagnostic"] == "Diabétique").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total",            len(bdf))
                c2.metric("Diabétiques",      n_diab)
                c3.metric("Prévalence",        f"{n_diab/len(bdf)*100:.1f}%")
                st.dataframe(bdf, use_container_width=True)
                st.download_button("⬇️ Télécharger",
                                   bdf.to_csv(index=False).encode(),
                                   "diagnostics_diabete.csv", "text/csv")

# ════════ TAB 4 — Performance ═════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📈 Performances du modèle entraîné")

    if _model_metrics:
        acc_val = f"{_model_metrics.get('accuracy', 0)*100:.1f}%"
        auc_val = f"{_model_metrics.get('auc_roc',  0):.3f}"
        f1_val  = f"{_model_metrics.get('f1',       0):.3f}"
        note    = "✅ Métriques réelles issues de l'entraînement"
    else:
        acc_val, auc_val, f1_val = "—", "—", "—"
        note = "⚠️ Modèle non chargé"

    st.caption(note)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", acc_val)
    c2.metric("AUC-ROC",  auc_val)
    c3.metric("F1-Score", f1_val)

    if _model_cfg:
        st.info(
            f"**Architecture :** `{_model_cfg.get('name','?')}` — "
            f"couches `{_model_cfg.get('hidden_layers','?')}` | "
            f"Dropout `{_model_cfg.get('dropout',0)}` | "
            f"BatchNorm `{_model_cfg.get('batchnorm',False)}` | "
            f"Activation `{_model_cfg.get('activation','relu')}` | "
            f"**{_input_size} features en entrée**"
        )

    # Courbe ROC illustrative
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) * 0.95 + fpr * 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ANN",
                             line=dict(color="#7C3AED", width=2),
                             fill="tozeroy", fillcolor="rgba(124,58,237,0.1)"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(color="#475569", dash="dash")))
    fig.update_layout(title="Courbe ROC (forme illustrative)",
                      xaxis_title="FPR", yaxis_title="TPR",
                      **PLOTLY_DARK, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Matrice de confusion (illustrative)**")
    cm = np.array([[8420, 820], [640, 2120]])
    fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                     labels=dict(x="Prédit", y="Réel"),
                     x=["Sain","Diabétique"], y=["Sain","Diabétique"])
    fig2.update_layout(**PLOTLY_DARK, height=280)
    st.plotly_chart(fig2, use_container_width=True)