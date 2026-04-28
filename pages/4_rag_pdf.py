"""
pages/4_rag_pdf.py — RAG PDF · LangChain + ChromaDB + Gemini
Structure du projet :
  mon_projet/
  ├── app.py
  ├── .env
  ├── pages/
  │   └── 4_rag_pdf.py   ← ce fichier
  └── utils/              ← minuscules !
      ├── helpers.py
      ├── Pre_process.py
      └── Chain.py
"""
import streamlit as st
import os, sys, time, tempfile

# ═══════════════════════════════════════════════════════════════════════
# CHEMINS — racine du projet + utils/ ajoutés à sys.path
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UTILS_DIR    = os.path.join(PROJECT_ROOT, "utils")   # minuscules

for _p in [PROJECT_ROOT, UTILS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ═══════════════════════════════════════════════════════════════════════
# VARIABLES D'ENVIRONNEMENT
# ═══════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Clés injectées directement (fallback si .env absent)
os.environ.setdefault("GOOGLE_API_KEY",       "")
os.environ.setdefault("LANGCHAIN_API_KEY",    "")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT",    "RAG")

# ═══════════════════════════════════════════════════════════════════════
# IMPORT helpers UI  (utils/helpers.py)
# ═══════════════════════════════════════════════════════════════════════
try:
    from helpers import load_css, page_banner, sidebar_info, show_info, show_error, show_success, PLOTLY_DARK
except ImportError:
    def load_css(): pass
    def page_banner(t, s, i): st.title(f"{i} {t}"); st.caption(s)
    def sidebar_info(t, color="#3B82F6"): st.sidebar.info(t)
    def show_info(m): st.info(m)
    def show_error(m): st.error(m)
    def show_success(m): st.success(m)
    PLOTLY_DARK = {}

# ═══════════════════════════════════════════════════════════════════════
# IMPORT pipeline RAG  (utils/Pre_process.py + utils/Chain.py)
# ═══════════════════════════════════════════════════════════════════════
_rag_error = None
try:
    from Pre_process import load_pdf, chunking, get_or_create_vector_store
    from Chain import get_rag_chain
    RAG_OK = True
except ImportError as e:
    RAG_OK     = False
    _rag_error = str(e)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="RAG PDF", page_icon="📄", layout="wide")
load_css()

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="shimmer-text" style="font-size:1.1rem;font-weight:700;padding:1rem 0 .5rem;">📄 RAG PDF</div>',
        unsafe_allow_html=True,
    )
    sidebar_info("LangChain + ChromaDB + Gemini", color="#3B82F6")
    st.divider()

    st.markdown("**Configuration RAG**")
    top_k         = st.slider("Chunks retournés (top-k)", 1, 10, 3)
    chunk_size    = st.slider("Taille des chunks",         200, 2000, 1000, 50)
    chunk_overlap = st.slider("Chevauchement",             0,   300,  100,  10)
    st.divider()

    show_sources_cb = st.checkbox("Afficher les sources", True)
    show_chunks_cb  = st.checkbox("Afficher les chunks bruts", False)
    st.divider()

    if st.button("🗑️ Réinitialiser la base vectorielle", use_container_width=True):
        import shutil
        for _db in ["chroma_db", ".chroma_db"]:
            _p = os.path.join(PROJECT_ROOT, _db)
            if os.path.exists(_p):
                shutil.rmtree(_p)
        for k in ["vectorstore", "rag_chain", "pdf_processed", "pdf_name", "doc_stats", "rag_messages"]:
            st.session_state.pop(k, None)
        st.success("✅ Base vectorielle supprimée !")
        st.rerun()

    st.page_link("app.py", label="← Retour accueil")

# ═══════════════════════════════════════════════════════════════════════
# BANNIÈRE
# ═══════════════════════════════════════════════════════════════════════
page_banner("RAG — Chatbot PDF", "LangChain · ChromaDB · Gemini · Q&A sur vos documents", "📄")

# ═══════════════════════════════════════════════════════════════════════
# GARDE-FOU : diagnostic si import échoué
# ═══════════════════════════════════════════════════════════════════════
if not RAG_OK:
    show_error(f"**Impossible d'importer le pipeline RAG.** Erreur : `{_rag_error}`")
    st.markdown("#### 🔎 Diagnostic automatique")
    st.code(f"""
Racine projet    : {PROJECT_ROOT}
Dossier utils/   : {UTILS_DIR}
  Pre_process.py : {'OK' if os.path.exists(os.path.join(UTILS_DIR,'Pre_process.py')) else 'ABSENT'}
  Chain.py       : {'OK' if os.path.exists(os.path.join(UTILS_DIR,'Chain.py'))       else 'ABSENT'}
  helpers.py     : {'OK' if os.path.exists(os.path.join(UTILS_DIR,'helpers.py'))     else 'ABSENT'}

Structure attendue :
  mon_projet/
  ├── app.py
  ├── .env
  ├── pages/
  │   └── 4_rag_pdf.py
  └── utils/
      ├── helpers.py
      ├── Pre_process.py
      └── Chain.py

Dépendances :
  pip install langchain langchain-google-genai langchain-chroma chromadb pypdf python-dotenv langchain-community
    """, language="text")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════
for _k, _v in [
    ("rag_messages",  []),
    ("vectorstore",   None),
    ("rag_chain",     None),
    ("pdf_processed", False),
    ("pdf_name",      ""),
    ("doc_stats",     {}),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📄 Chargement PDF", "💬 Chat Q&A", "📊 Statistiques"])

# ──────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Vectorisation
# ──────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### 📤 Chargement et vectorisation du PDF")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_pdf = st.file_uploader(
            "Glissez-déposez votre PDF",
            type=["pdf"],
            help="Vectorisé avec gemini-embedding-001 → ChromaDB",
        )
    with col2:
        st.markdown("""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
             border-radius:10px;padding:12px 16px;">
            <div style="font-weight:600;color:#E2E8F0;margin-bottom:8px;">💡 Pipeline</div>
            <ol style="color:#94A3B8;font-size:.82rem;padding-left:1.2rem;line-height:1.9;">
                <li>PyPDFLoader → pages</li>
                <li>RecursiveCharacterTextSplitter → chunks</li>
                <li>gemini-embedding-001 → vecteurs</li>
                <li>ChromaDB (cosine) → stockage</li>
                <li>gemini-2.0-flash-lite → réponse</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_pdf:
        file_size = len(uploaded_pdf.getvalue()) / 1024 / 1024
        st.markdown(f"""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
             border-radius:10px;padding:10px 16px;margin:8px 0;">
            📄 <strong style="color:#E2E8F0;">{uploaded_pdf.name}</strong>
            &nbsp;|&nbsp;<span style="color:#94A3B8;">{file_size:.2f} MB</span>
        </div>""", unsafe_allow_html=True)

        if st.button("⚡ Vectoriser le document", type="primary", use_container_width=True):
            prog   = st.progress(0, "Initialisation…")
            status = st.empty()
            try:
                prog.progress(10, "💾 Sauvegarde temporaire…")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name

                prog.progress(25, "📖 Lecture avec PyPDFLoader…")
                pages = load_pdf(tmp_path)
                status.info(f"✅ {len(pages)} page(s) chargée(s)")

                prog.progress(45, "✂️ Découpage en chunks…")
                chunks = chunking(pages)
                status.info(f"✅ {len(chunks)} chunk(s) créé(s)")

                prog.progress(60, "🔢 Génération embeddings Gemini (1-2 min)…")
                st.info("⏳ Indexation par lots avec pauses — quota Gemini respecté…")
                vector_store = get_or_create_vector_store(chunks)

                prog.progress(90, "🔗 Création de la chaîne LangChain…")
                rag_chain = get_rag_chain(vector_store)

                st.session_state.vectorstore   = vector_store
                st.session_state.rag_chain     = rag_chain
                st.session_state.pdf_processed = True
                st.session_state.pdf_name      = uploaded_pdf.name
                st.session_state.doc_stats     = {
                    "pages":       len(pages),
                    "chunks":      len(chunks),
                    "chunk_size":  chunk_size,
                    "model_embed": "gemini-embedding-001",
                    "model_llm":   "gemini-2.0-flash-lite",
                }

                os.unlink(tmp_path)
                prog.progress(100, "✅ Terminé !")
                time.sleep(0.4)
                prog.empty()
                status.empty()

            except Exception as e:
                prog.empty()
                status.empty()
                show_error(f"**Erreur lors de la vectorisation :** {e}")
                st.exception(e)

    if st.session_state.pdf_processed:
        stats = st.session_state.doc_stats
        show_success(f"✅ **{st.session_state.pdf_name}** indexé avec succès !")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Pages",     stats.get("pages", "—"))
        c2.metric("✂️ Chunks",    stats.get("chunks", "—"))
        c3.metric("🔢 Embedding", stats.get("model_embed", "—"))
        c4.metric("🤖 LLM",       stats.get("model_llm", "—"))

    st.divider()
    st.markdown("#### 📚 Types de PDFs compatibles")
    ref_docs = [
        ("Rapport Annuel",         "Rapports d'entreprises", "📊"),
        ("Article Scientifique",   "Publications ML/IA",     "🔬"),
        ("Document Légal",         "Contrats, règlements",   "⚖️"),
        ("Documentation Technique","Manuels, API docs",      "📖"),
    ]
    cols = st.columns(4)
    for col, (name, desc, icon) in zip(cols, ref_docs):
        with col:
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.2);
                 border-radius:10px;padding:10px;text-align:center;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-weight:600;color:#E2E8F0;font-size:.85rem;">{name}</div>
                <div style="color:#64748B;font-size:.72rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# TAB 2 — Chat Q&A
# ──────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.pdf_processed:
        show_info("📤 Veuillez d'abord charger un PDF dans l'onglet **Chargement PDF**.")
    else:
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);
             border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:.83rem;color:#94A3B8;">
            📄 <strong style="color:#60A5FA;">{st.session_state.pdf_name}</strong>
            &nbsp;|&nbsp;{stats.get('chunks','?')} chunks
            &nbsp;|&nbsp;Top-{top_k}
            &nbsp;|&nbsp;🤖 {stats.get('model_llm','Gemini')}
        </div>""", unsafe_allow_html=True)

        # Questions suggérées
        st.markdown("**Suggestions :**")
        suggestions = [
            "Résume ce document",
            "Quels sont les points clés ?",
            "Liste les conclusions principales",
            "Y a-t-il des données numériques ?",
        ]
        s_cols = st.columns(4)
        for i, (col, sug) in enumerate(zip(s_cols, suggestions)):
            if col.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.rag_messages.append({"role": "user", "content": sug})
                st.session_state._pending_rag = sug

        # Historique
        for msg in st.session_state.rag_messages[-14:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources") and show_sources_cb:
                    with st.expander("📚 Passages sources"):
                        for src in msg["sources"]:
                            st.markdown(f"```\n{src}\n```")

        # Input
        if prompt := st.chat_input("Posez une question sur le document…"):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            st.session_state._pending_rag = prompt

        # Traitement
        pending = st.session_state.pop("_pending_rag", None)
        if pending:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Recherche via Gemini…"):
                    sources = []
                    try:
                        answer = st.session_state.rag_chain.invoke(pending)

                        if show_sources_cb or show_chunks_cb:
                            try:
                                retriever   = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                                source_docs = retriever.invoke(pending)
                                sources = [
                                    f"[Chunk {i+1} | Page {d.metadata.get('page', 0)+1}]\n{d.page_content[:300]}…"
                                    for i, d in enumerate(source_docs)
                                ]
                            except Exception:
                                sources = []

                    except Exception as e:
                        answer = f"⚠️ Erreur : {e}"
                        st.exception(e)

                st.markdown(answer)

                if show_sources_cb and sources:
                    with st.expander("📚 Passages sources utilisés"):
                        for src in sources:
                            st.markdown(f"```\n{src}\n```")
                if show_chunks_cb and sources:
                    with st.expander("🔍 Chunks bruts"):
                        for src in sources:
                            st.code(src, language="text")

            st.session_state.rag_messages.append({
                "role": "assistant", "content": answer, "sources": sources,
            })

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Effacer conversation", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()
        with col2:
            if st.session_state.rag_messages:
                chat_text = "\n\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.rag_messages]
                )
                st.download_button(
                    "⬇️ Exporter conversation", chat_text.encode(),
                    "conversation_rag.txt", "text/plain", use_container_width=True,
                )

# ──────────────────────────────────────────────────────────────────────
# TAB 3 — Statistiques
# ──────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📊 Statistiques & Architecture RAG")
    if not st.session_state.pdf_processed:
        show_info("Chargez un document pour voir les statistiques.")
    else:
        import plotly.graph_objects as go
        import numpy as np

        stats = st.session_state.doc_stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents indexés", "1")
        c2.metric("Chunks vectorisés",  stats.get("chunks", "—"))
        c3.metric("Questions posées",   len([m for m in st.session_state.rag_messages if m["role"] == "user"]))
        c4.metric("Top-k configuré",    top_k)

        scores = np.random.beta(5, 2, 200) * 0.4 + 0.5
        fig = go.Figure(go.Histogram(x=scores, nbinsx=30, marker_color="#3B82F6", opacity=0.8))
        fig.update_layout(
            **PLOTLY_DARK, height=260,
            title="Distribution des scores de similarité cosinus (simulée)",
            xaxis_title="Score cosinus", yaxis_title="Fréquence",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**🏗️ Architecture du pipeline**")
        st.code(f"""
PDF : {st.session_state.pdf_name}
  └─► PyPDFLoader
        └─► RecursiveCharacterTextSplitter
              chunk_size={stats.get('chunk_size',1000)}, overlap=100
              └─► {stats.get('chunks','?')} chunks
                    └─► GoogleGenerativeAIEmbeddings
                          model : gemini-embedding-001
                          cache : ./cache_embeddings/
                          └─► ChromaDB (cosine) → ./chroma_db/
                                └─► Retriever top-k={top_k}
                                      └─► ChatGoogleGenerativeAI
                                            model : gemini-2.0-flash-lite
                                            temp  : 0.1
                                            └─► Réponse
        """, language="text")