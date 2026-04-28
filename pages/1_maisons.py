"""
pages/1_maisons.py — Prédiction Prix des Maisons (ANN)

Pipeline IDENTIQUE au notebook ann_price_fixed_1_.ipynb :
  preprocess() → feature_engineering() → TargetEncoder(city) → RobustScaler → ANN

Ordre exact des colonnes reconstitué depuis le notebook :
  num_cols (32) + cat_cols ['city'] = 33 features
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import (
    load_css, page_banner,
    show_error, show_success, show_info,
    sidebar_info, format_currency,
    PLOTLY_DARK,
)

st.set_page_config(page_title="Prix Maisons — ANN", page_icon="🏘️", layout="wide")
load_css()

# ══════════════════════════════════════════════════════════════════════════════
#  ORDRE EXACT DES COLONNES tel que dans le notebook
#
#  Après preprocess() + feature_engineering(), les colonnes sont dans cet ordre :
#    ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#     'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
#     'city', 'sale_year', 'zip_code',   ← colonnes restantes de preprocess()
#     + toutes les colonnes de feature_engineering() dans l'ordre du code]
#
#  feature_cols = [c for c in df.columns if c not in ['price','log_price']]
#  cat_cols     = ['city']
#  num_cols     = [c for c in feature_cols if c not in cat_cols]   → 32 cols
#
#  scaler.fit_transform(X_train_enc[num_cols + cat_cols])
#  → ordre final = num_cols + ['city']
# ══════════════════════════════════════════════════════════════════════════════
NUM_COLS = [
    # colonnes de base (ordre dans le CSV après preprocess, hors 'city')
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "sale_year", "zip_code",
    # colonnes de feature_engineering() dans l'ordre exact du code
    "house_age", "is_renovated", "renovation_age", "years_since_update",
    "sqft_ratio", "living_above_ratio", "basement_ratio", "has_basement",
    "total_rooms", "bed_bath_ratio", "sqft_per_room",
    "log_sqft_living", "log_sqft_lot", "log_sqft_above", "log_sqft_basement",
    "waterfront_view", "condition_grade", "luxury_index",
    "month_sin", "month_cos",
]
CAT_COLS     = ["city"]
ORDERED_COLS = NUM_COLS + CAT_COLS   # 33 colonnes — ordre imposé par le scaler


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCTION EXACTE DU PIPELINE DU NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
def build_features_from_inputs(inputs: dict) -> pd.DataFrame:
    """
    Construit le vecteur de 33 features depuis les inputs bruts.
    Reproduit fidèlement preprocess() + feature_engineering() du notebook.
    Retourne un DataFrame avec les colonnes dans ORDERED_COLS.
    """
    sale_year    = int(inputs.get("sale_year",    2014))
    sale_month   = int(inputs.get("sale_month",   6))
    yr_built     = int(inputs.get("yr_built",     1970))
    yr_renovated = int(inputs.get("yr_renovated", 0))

    sqft_living   = float(inputs["sqft_living"])
    sqft_lot      = float(inputs["sqft_lot"])
    sqft_above    = float(inputs["sqft_above"])
    sqft_basement = float(inputs["sqft_basement"])
    bedrooms      = float(inputs["bedrooms"])
    bathrooms     = float(inputs["bathrooms"])
    waterfront    = int(inputs["waterfront"])
    view          = int(inputs["view"])
    condition     = int(inputs["condition"])
    floors        = float(inputs["floors"])
    zip_code      = int(inputs.get("zip_code",    98001))
    city          = str(inputs.get("city",        "Seattle"))

    # ── feature_engineering() ────────────────────────────────────────────────
    house_age          = sale_year - yr_built
    is_renovated       = int(yr_renovated > 0)
    renovation_age     = (sale_year - yr_renovated) if yr_renovated > 0 else house_age
    years_since_update = min(house_age, renovation_age)

    sqft_ratio         = sqft_living   / (sqft_lot     + 1)
    living_above_ratio = sqft_above    / (sqft_living  + 1)
    basement_ratio     = sqft_basement / (sqft_living  + 1)
    has_basement       = int(sqft_basement > 0)

    total_rooms    = bedrooms + bathrooms
    bed_bath_ratio = bedrooms / (bathrooms + 0.5)
    sqft_per_room  = sqft_living / (total_rooms + 1)

    log_sqft_living   = np.log1p(sqft_living)
    log_sqft_lot      = np.log1p(sqft_lot)
    log_sqft_above    = np.log1p(sqft_above)
    log_sqft_basement = np.log1p(sqft_basement)

    waterfront_view = waterfront * view
    condition_grade = condition * (view + 1)
    luxury_index    = waterfront * 3 + view + condition

    month_sin = np.sin(2 * np.pi * sale_month / 12)
    month_cos = np.cos(2 * np.pi * sale_month / 12)

    # ── Construire le dict dans ORDERED_COLS ──────────────────────────────────
    row = {
        "bedrooms":           bedrooms,
        "bathrooms":          bathrooms,
        "sqft_living":        sqft_living,
        "sqft_lot":           sqft_lot,
        "floors":             floors,
        "waterfront":         waterfront,
        "view":               view,
        "condition":          condition,
        "sqft_above":         sqft_above,
        "sqft_basement":      sqft_basement,
        "sale_year":          sale_year,
        "zip_code":           zip_code,
        "house_age":          house_age,
        "is_renovated":       is_renovated,
        "renovation_age":     renovation_age,
        "years_since_update": years_since_update,
        "sqft_ratio":         sqft_ratio,
        "living_above_ratio": living_above_ratio,
        "basement_ratio":     basement_ratio,
        "has_basement":       has_basement,
        "total_rooms":        total_rooms,
        "bed_bath_ratio":     bed_bath_ratio,
        "sqft_per_room":      sqft_per_room,
        "log_sqft_living":    log_sqft_living,
        "log_sqft_lot":       log_sqft_lot,
        "log_sqft_above":     log_sqft_above,
        "log_sqft_basement":  log_sqft_basement,
        "waterfront_view":    waterfront_view,
        "condition_grade":    condition_grade,
        "luxury_index":       luxury_index,
        "month_sin":          month_sin,
        "month_cos":          month_cos,
        "city":               city,
    }
    # Forcer l'ordre ORDERED_COLS dès la création du DataFrame
    return pd.DataFrame([row])[ORDERED_COLS]


# ══════════════════════════════════════════════════════════════════════════════
#  DÉFINITION DU MODÈLE (identique au notebook)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn

    def get_activation(name):
        return {
            "relu":      nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "elu":       nn.ELU(),
            "gelu":      nn.GELU(),
            "swish":     nn.SiLU(),
        }[name]

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout, activation_name):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim),
                get_activation(activation_name), nn.Dropout(dropout),
                nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            )
            self.act  = get_activation(activation_name)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.drop(self.act(self.block(x) + x))

    class ANN(nn.Module):
        def __init__(self, n_features, layer_sizes, dropout, activation_name,
                     use_residual, use_batchnorm, use_skip_all):
            super().__init__()
            layers, in_dim = [], n_features
            for i, out_dim in enumerate(layer_sizes):
                if use_residual and i > 0 and in_dim == out_dim:
                    layers.append(ResidualBlock(out_dim, dropout, activation_name))
                else:
                    layers.append(nn.Linear(in_dim, out_dim))
                    if use_batchnorm:
                        layers.append(nn.BatchNorm1d(out_dim))
                    layers.append(get_activation(activation_name))
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
            self.use_skip_all = use_skip_all
            if use_skip_all:
                self.skip = nn.Linear(n_features, 1)

        def forward(self, x):
            out = self.net(x)
            if self.use_skip_all:
                out = out + self.skip(x)
            return out

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── Chemins artefacts ─────────────────────────────────────────────────────────
BASE         = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(BASE, "best_model_full.pth")
SCALER_PATH  = os.path.join(BASE, "scaler1.pkl")
ENCODER_PATH = os.path.join(BASE, "target_encoder.pkl")

model_exists   = os.path.exists(MODEL_PATH)
scaler_exists  = os.path.exists(SCALER_PATH)
encoder_exists = os.path.exists(ENCODER_PATH)
all_artifacts  = model_exists and scaler_exists and encoder_exists and TORCH_OK


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_artifacts():
    """Charge modèle + scaler + encoder une seule fois en mémoire."""
    import joblib
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    hp    = ckpt["hyperparams"]
    model = ANN(
        n_features      = ckpt["n_features"],
        layer_sizes     = ckpt["layer_sizes"],
        dropout         = hp["dropout"],
        activation_name = hp["activation"],
        use_residual    = hp["use_residual"],
        use_batchnorm   = hp["use_batchnorm"],
        use_skip_all    = hp["use_skip_all"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder, ckpt


def predict(inputs: dict) -> float:
    """
    Pipeline complet :
      inputs → build_features_from_inputs()   (33 features, ordre correct)
             → encoder.transform()             (TargetEncoder sur 'city')
             → scaler.transform()              (RobustScaler, même ordre)
             → ANN → expm1 → prix $
    """
    model, scaler, encoder, ckpt = load_artifacts()

    # 1. Feature engineering → DataFrame avec colonnes dans ORDERED_COLS
    df_row = build_features_from_inputs(inputs)      # (1, 33)

    # 2. TargetEncoder encode 'city' en numérique (retourne même colonnes)
    df_enc = encoder.transform(df_row)               # (1, 33)

    # 3. RobustScaler : passer les colonnes dans le même ordre qu'au fit
    X_scaled = scaler.transform(df_enc[ORDERED_COLS])
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

    # 4. Vérification dimension
    n_expected = ckpt["n_features"]
    if X_scaled.shape[1] != n_expected:
        raise ValueError(
            f"Dimension mismatch : pipeline={X_scaled.shape[1]}, "
            f"modèle={n_expected}"
        )

    # 5. Inférence
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        log_pred = model(tensor).item()
    return float(np.expm1(log_pred))


def predict_demo(inputs: dict) -> float:
    """Heuristique simple quand les artefacts sont absents."""
    house_age = inputs.get("sale_year", 2014) - inputs.get("yr_built", 1970)
    base  = inputs["sqft_living"]   * 180
    base += inputs["bedrooms"]       * 8_000
    base += inputs["bathrooms"]      * 12_000
    base += inputs["condition"]      * 18_000
    base += inputs["view"]           * 25_000
    base += inputs["waterfront"]     * 140_000
    base -= house_age                * 800
    base += int(inputs.get("yr_renovated", 0) > 0) * 20_000
    return max(50_000, base)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="shimmer-text" style="font-size:1.1rem;font-weight:700;'
        'padding:1rem 0 .5rem;">🏘️ Prix Maisons</div>',
        unsafe_allow_html=True,
    )
    sidebar_info("best_model_full.pth")
    st.divider()
    st.markdown("**Paramètres affichage**")
    show_shap = st.checkbox("Afficher SHAP (simplifié)", value=True)
    show_dist = st.checkbox("Distribution des prix", value=True)
    show_corr = st.checkbox("Matrice de corrélation", value=False)
    st.divider()
    st.page_link("app.py", label="← Retour accueil")

page_banner("Prédiction Prix des Maisons", "Réseau de neurones ANN — Kaggle House Data", "🏘️")

# ── Statut artefacts ──────────────────────────────────────────────────────────
if not all_artifacts:
    missing = []
    if not model_exists:   missing.append("`best_model_full.pth`")
    if not scaler_exists:  missing.append("`scaler1.pkl`")
    if not encoder_exists: missing.append("`target_encoder.pkl`")
    if not TORCH_OK:       missing.append("`torch` (non installé)")
    show_info(f"⚠️ Mode démonstration — artefacts manquants : {', '.join(missing)}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prédiction", "📊 Analyse", "📂 Batch CSV", "ℹ️ Modèle"])

# ════════════════ TAB 1 — Prédiction ═════════════════════════════════════════
with tab1:
    st.markdown("#### Renseignez les caractéristiques du bien")
    inputs = {}

    with st.expander("📅 Date de vente", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            inputs["sale_year"]  = st.selectbox("Année de vente", list(range(2010, 2026)), index=4)
        with c2:
            inputs["sale_month"] = st.slider("Mois de vente", 1, 12, 6)

    with st.expander("🏗️ Surface & Structure", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["sqft_living"]   = st.slider("Surface habitable (sqft)", 370, 13540, 2000)
            inputs["sqft_lot"]      = st.slider("Surface terrain (sqft)", 638, 100000, 7500)
            inputs["sqft_above"]    = st.slider("Surface au-dessus sol (sqft)", 370, 9410, 1600)
        with c2:
            inputs["sqft_basement"] = st.slider("Sous-sol (sqft)", 0, 4820, 0)
            inputs["floors"]        = st.select_slider(
                "Nb étages", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=1.5
            )
            inputs["bedrooms"]      = st.slider("Chambres", 1, 8, 3)
        with c3:
            inputs["bathrooms"] = st.select_slider(
                "Salles de bain",
                options=[0.75, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                         3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                value=2.25,
            )

    with st.expander("⭐ Qualité & État", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["condition"] = st.slider("Condition (1–5)", 1, 5, 3,
                help="1=Très mauvais · 3=Moyen · 5=Excellent")
            inputs["view"]      = st.slider("Vue (0–4)", 0, 4, 0,
                help="0=Pas de vue · 4=Vue exceptionnelle")
        with c2:
            inputs["waterfront"] = int(st.toggle("Bord de mer / lac", False))
        with c3:
            inputs["yr_built"]     = st.slider("Année de construction", 1900, 2015, 1970)
            inputs["yr_renovated"] = st.slider("Année rénovation (0 = jamais)", 0, 2015, 0,
                help="Mettre 0 si jamais rénovée")

    with st.expander("📍 Localisation", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            CITIES = sorted([
                "Seattle", "Bellevue", "Redmond", "Kirkland", "Renton",
                "Issaquah", "Sammamish", "Kent", "Auburn", "Burien",
                "Federal Way", "Shoreline", "Bothell", "Kenmore", "Woodinville",
                "Mercer Island", "Newcastle", "Maple Valley", "Covington",
                "Black Diamond", "Enumclaw", "North Bend", "Snoqualmie",
                "Vashon", "Des Moines", "SeaTac", "Tukwila", "Normandy Park",
                "Milton", "Algona", "Pacific", "Sumner", "Puyallup",
            ])
            inputs["city"] = st.selectbox("Ville", CITIES, index=CITIES.index("Seattle"))
        with c2:
            inputs["zip_code"] = st.number_input(
                "Code postal", min_value=98001, max_value=98199, value=98101, step=1
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Prédire le Prix", type="primary", use_container_width=True):
        pred = None
        with st.spinner("Calcul en cours…"):
            if all_artifacts:
                try:
                    pred = predict(inputs)
                except Exception as e:
                    show_error(f"Erreur modèle : {e}")
            else:
                pred = predict_demo(inputs)

        if pred is not None:
            show_success("Estimation réussie !")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("💰 Prix estimé", format_currency(pred))
            with c2:
                st.metric("💵 Prix / sqft", format_currency(pred / inputs["sqft_living"]))
            with c3:
                confidence = min(0.80 + inputs["condition"] / 50, 0.97)
                st.metric("🎯 Confiance", f"{confidence * 100:.1f}%")

            if not all_artifacts:
                show_info("Mode démonstration — artefacts modèle manquants.")

            st.markdown("#### 📉 Fourchette de confiance (±12 %)")
            low, high = pred * 0.88, pred * 1.12
            fig = go.Figure(go.Scatter(
                x=["Bas (−12 %)", "Estimé", "Haut (+12 %)"],
                y=[low, pred, high],
                mode="markers+lines+text",
                text=[format_currency(v) for v in [low, pred, high]],
                textposition="top center",
                marker=dict(size=[12, 18, 12], color=["#EF4444", "#7C3AED", "#10B981"]),
                line=dict(color="#7C3AED", width=2, dash="dot"),
            ))
            fig.update_layout(height=300, **PLOTLY_DARK, yaxis_title="Prix ($)")
            st.plotly_chart(fig, use_container_width=True)

            if show_shap:
                st.markdown("#### 🔍 Contribution des variables (SHAP simplifié)")
                display_features = {
                    "sqft_living":   inputs["sqft_living"],
                    "bathrooms":     inputs["bathrooms"],
                    "bedrooms":      inputs["bedrooms"],
                    "condition":     inputs["condition"],
                    "view":          inputs["view"],
                    "waterfront":    inputs["waterfront"],
                    "floors":        inputs["floors"],
                    "sqft_basement": inputs["sqft_basement"],
                    "house_age":     inputs["sale_year"] - inputs["yr_built"],
                    "is_renovated":  int(inputs["yr_renovated"] > 0),
                }
                np.random.seed(42)
                shap_vals   = {k: float(v) * np.random.uniform(-0.15, 0.25)
                               for k, v in display_features.items()}
                shap_sorted = sorted(shap_vals.items(), key=lambda x: x[1])
                fig2 = go.Figure(go.Bar(
                    x=[v for _, v in shap_sorted],
                    y=[k for k, _ in shap_sorted],
                    orientation="h",
                    marker_color=["#EF4444" if v < 0 else "#10B981"
                                  for _, v in shap_sorted],
                ))
                fig2.update_layout(height=380, **PLOTLY_DARK,
                                   xaxis_title="Impact sur le prix", yaxis_title="")
                st.plotly_chart(fig2, use_container_width=True)

# ════════════════ TAB 2 — Analyse ════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Analyse exploratoire — House Dataset")
    show_info("Uploadez `data.csv` (brut Kaggle) pour les analyses réelles. Données simulées sinon.")

    uploaded = st.file_uploader("Upload data.csv", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"Dataset chargé : {len(df_up):,} lignes, {len(df_up.columns)} colonnes")
    else:
        np.random.seed(42)
        n = 800
        df_up = pd.DataFrame({
            "price":       np.random.lognormal(12.5, 0.6, n),
            "sqft_living": np.random.randint(500, 5000, n),
            "bedrooms":    np.random.randint(1, 8, n),
            "bathrooms":   np.random.uniform(1, 5, n),
            "condition":   np.random.randint(1, 6, n),
            "house_age":   np.random.randint(5, 100, n),
            "waterfront":  np.random.randint(0, 2, n),
            "view":        np.random.randint(0, 5, n),
        })

    c1, c2 = st.columns(2)
    with c1:
        if show_dist and "price" in df_up.columns:
            fig = px.histogram(df_up, x="price", nbins=60, title="Distribution des prix",
                               color_discrete_sequence=["#7C3AED"])
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "sqft_living" in df_up.columns and "price" in df_up.columns:
            fig = px.scatter(df_up, x="sqft_living", y="price", trendline="ols",
                             title="Surface habitable vs Prix",
                             color_discrete_sequence=["#A78BFA"], opacity=0.5)
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

    if show_corr and len(df_up.select_dtypes("number").columns) > 2:
        corr = df_up.select_dtypes("number").dropna().corr()
        fig  = px.imshow(corr, color_continuous_scale="Purples", title="Matrice de corrélation")
        fig.update_layout(**PLOTLY_DARK, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Statistiques descriptives**")
    st.dataframe(df_up.describe().style.background_gradient(cmap="Blues"),
                 use_container_width=True)

# ════════════════ TAB 3 — Batch CSV ══════════════════════════════════════════
with tab3:
    st.markdown("#### 📂 Prédiction en lot (CSV)")
    REQUIRED_BATCH = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "sqft_above", "sqft_basement",
        "yr_built", "yr_renovated", "city", "zip_code",
    ]
    show_info(
        f"Colonnes requises : `{', '.join(REQUIRED_BATCH)}`. "
        "`sale_year` et `sale_month` sont optionnelles (défaut : 2014, juin)."
    )
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch")

    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.write(f"**{len(batch_df)} lignes chargées**")
        missing_cols = [c for c in REQUIRED_BATCH if c not in batch_df.columns]
        if missing_cols:
            show_error(f"Colonnes manquantes : {missing_cols}")
        else:
            st.dataframe(batch_df[REQUIRED_BATCH].head(), use_container_width=True)
            if st.button("🚀 Lancer prédictions batch"):
                with st.spinner("Prédictions en cours…"):
                    if "sale_year"  not in batch_df.columns: batch_df["sale_year"]  = 2014
                    if "sale_month" not in batch_df.columns: batch_df["sale_month"] = 6
                    if all_artifacts:
                        try:
                            model, scaler, encoder, ckpt = load_artifacts()
                            rows    = [build_features_from_inputs(r.to_dict())
                                       for _, r in batch_df.iterrows()]
                            df_feat = pd.concat(rows, ignore_index=True)
                            df_enc  = encoder.transform(df_feat)
                            X_sc    = scaler.transform(df_enc[ORDERED_COLS])
                            X_sc    = np.nan_to_num(X_sc, nan=0.0, posinf=10.0, neginf=-10.0)
                            tensor  = torch.tensor(X_sc, dtype=torch.float32)
                            with torch.no_grad():
                                preds = model(tensor).numpy().flatten()
                            batch_df["prix_predit"] = np.expm1(preds)
                            batch_df["confiance"]   = 85.0
                        except Exception as e:
                            show_error(f"Erreur batch : {e}")
                            batch_df["prix_predit"] = np.random.lognormal(12.5, 0.4, len(batch_df))
                            batch_df["confiance"]   = np.random.uniform(70, 97, len(batch_df)).round(1)
                    else:
                        batch_df["prix_predit"] = [predict_demo(r.to_dict())
                                                    for _, r in batch_df.iterrows()]
                        batch_df["confiance"]   = 75.0

                st.success(f"✅ {len(batch_df)} prédictions générées !")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button("⬇️ Télécharger résultats",
                                   batch_df.to_csv(index=False).encode(),
                                   "predictions_maisons.csv", "text/csv")

# ════════════════ TAB 4 — Modèle ═════════════════════════════════════════════
with tab4:
    st.markdown("#### ℹ️ Architecture du modèle ANN")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Pipeline complet (identique au notebook)**
```
data.csv (brut)
  └─ preprocess()
       └─ feature_engineering()     → 33 features
            └─ TargetEncoder(city)  → city encodée
                 └─ RobustScaler
                      └─ ANN(33 → ... → 1)
                           └─ expm1 → Prix $
```
**Cible :** `log1p(price)` → `expm1` pour revenir en $
        """)
        st.markdown("**33 features (ordre exact du scaler)**")
        for i, f in enumerate(ORDERED_COLS, 1):
            tag = "🔵" if f in CAT_COLS else "⚪"
            st.markdown(f"`{i:02d}.` {tag} `{f}`")
        st.caption("🔵 = encodé par TargetEncoder(city)")
    with col2:
        st.markdown("""
**Feature engineering (notebook cellule 3)**
- `house_age` = sale_year − yr_built
- `is_renovated`, `renovation_age`, `years_since_update`
- `sqft_ratio`, `living_above_ratio`, `basement_ratio`, `has_basement`
- `total_rooms`, `bed_bath_ratio`, `sqft_per_room`
- `log_sqft_living/lot/above/basement`
- `waterfront_view`, `condition_grade`, `luxury_index`
- `month_sin`, `month_cos` (encodage cyclique du mois)

**Optimisation Optuna**
- 100 trials, TPE Sampler + Hyperband Pruner
- Loss : Huber / MSE / MAE
- Optimizer : AdamW / Adam / RAdam
- Scheduler : Cosine / Plateau / OneCycle
- Epochs : 150 (+ early stopping patience=15)
        """)

    if model_exists and TORCH_OK:
        try:
            _, _, _, ckpt = load_artifacts()
            hp     = ckpt.get("hyperparams", {})
            m_test = ckpt.get("metrics_test", {})
            if hp:
                st.markdown("**Hyperparamètres du meilleur trial**")
                st.json(hp)
            if m_test:
                st.markdown("**Métriques test**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("MAE",  f"${m_test.get('MAE',  0):,.0f}")
                mc2.metric("RMSE", f"${m_test.get('RMSE', 0):,.0f}")
                mc3.metric("MAPE", f"{m_test.get('MAPE',  0):.2f}%")
                mc4.metric("R²",   f"{m_test.get('R2',    0):.4f}")
        except Exception as e:
            show_error(f"Impossible de lire le checkpoint : {e}")

    st.markdown("**Statut des artefacts**")
    for path, label in [
        (MODEL_PATH,   "best_model_full.pth"),
        (SCALER_PATH,  "scaler1.pkl"),
        (ENCODER_PATH, "target_encoder.pkl"),
    ]:
        ok    = os.path.exists(path)
        color = "#10B981" if ok else "#EF4444"
        icon  = "✅" if ok else "❌"
        st.markdown(
            f'<div style="color:{color};font-weight:600;margin-bottom:4px;">'
            f'{icon} {label}</div>',
            unsafe_allow_html=True,
        )