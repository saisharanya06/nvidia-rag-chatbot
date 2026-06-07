"""
app.py
======
Streamlit UI for the NVIDIA 10-K RAG Chatbot.

Features:
  • PDF upload & knowledge-base builder
  • Section / page filters
  • Hybrid retrieval with reranking
  • Gemini-powered answer generation with citations
  • Chat history with retrieved-chunk inspector
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import sys

# ── Configure logging to output to terminal ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("app")
logger.info("NVIDIA 10-K RAG Application Started")

import os
import shutil
import time
from pathlib import Path

import streamlit as st

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, GEMINI_API_KEY

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NVIDIA 10-K RAG Chatbot",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global & Typography ────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Force clean corporate dark background and light text */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* ── Header ─────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff !important; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.95; font-size: 0.95rem; color: #f8fafc !important; }

    /* ── Chat bubbles (Midnight Ocean Theme) ────────────────── */
    .user-msg {
        background: #1e293b;
        border-left: 4px solid #00f2fe;
        padding: 1rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        color: #f8fafc;
        font-size: 0.98rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .user-msg strong {
        color: #00f2fe;
    }
    .bot-msg {
        background: #111827;
        border-left: 4px solid #4facfe;
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        color: #e2e8f0;
        font-size: 0.98rem;
        line-height: 1.65;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .citation-box {
        background: #1e1b10;
        border: 1px solid #ffd700;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-top: 0.4rem;
        font-size: 0.88rem;
        color: #fef08a;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
    }
    .citation-box strong {
        color: #fde047;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        color: #94a3b8;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00f2fe;
    }
    
    /* Make sidebar buttons look premium */
    section[data-testid="stSidebar"] button {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #f8fafc;
    }

    /* ── Status badge ───────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-ready  { background: #065f46; color: #34d399; }
    .status-empty  { background: #7f1d1d; color: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "chat_history": [],
        "retriever": None,
        "generator": None,
        "db_ready": False,
        "sections": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_db() -> tuple[bool, str]:
    """Return (True, '') if ChromaDB collection exists and has documents, else (False, error_msg)."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        count = col.count()
        logger.info(f"Database check: collection '{COLLECTION_NAME}' has {count} docs.")
        if count > 0:
            return True, ""
        return False, "Collection exists but is empty."
    except Exception as e:
        logger.warning("Database check failed: %s", e)
        return False, str(e)


def clean_filename(filename: str) -> str:
    """Converts a raw filename like nasdaq-nvda-2025-10K-25670928.pdf to a clean title."""
    name = Path(filename).stem
    name = name.replace("-", " ").replace("_", " ")
    words = name.split()
    cleaned_words = []
    for w in words:
        # Skip long ID numbers (e.g., date codes, filing numbers) but keep years
        if w.isdigit() and len(w) > 4:
            continue
        if w.lower() == "10k":
            cleaned_words.append("10-K")
        elif len(w) <= 4 and w.isalpha():
            cleaned_words.append(w.upper())
        else:
            cleaned_words.append(w.capitalize())
    title = " ".join(cleaned_words)
    return title or "Document"


def _get_active_doc_title() -> str:
    """Retrieve the clean title of the active document stored in the database."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        first_doc = col.get(limit=1, include=["metadatas"])
        if first_doc and first_doc["metadatas"]:
            fn = first_doc["metadatas"][0].get("file_name", "")
            if fn:
                return clean_filename(fn)
    except Exception:
        pass
    return "Document"


@st.cache_resource(show_spinner="Loading retriever models …")
def _load_retriever():
    from retriever import HybridRetriever
    return HybridRetriever()


# Generator is loaded dynamically on query to prevent caching issues


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
active_title = _get_active_doc_title()

st.markdown(
    f'<div class="main-header">'
    f"<h1>🟢 {active_title} RAG Chatbot</h1>"
    f"<p>Ask questions about the {active_title} — powered by hybrid retrieval &amp; Gemini</p>"
    f"</div>",
    unsafe_allow_html=True,
)



# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document Management")

    # ── PDF upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a 10-K PDF",
        type=["pdf"],
        help="Upload NVIDIA's 10-K or any similar PDF.",
    )
    if uploaded is not None:
        dest = DATA_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved → `{dest.name}`")

    # ── List available PDFs ───────────────────────────────────────────────
    available_pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if available_pdfs:
        selected_pdf = st.selectbox(
            "Select PDF for ingestion",
            options=available_pdfs,
            format_func=lambda p: p.name,
        )
    else:
        st.warning("No PDFs found in `data/`. Upload one above.")
        selected_pdf = None

    # ── Build knowledge base ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔧 Knowledge Base")

    db_exists, db_err = _check_db()
    if db_exists:
        st.markdown(
            '<span class="status-badge status-ready">● DB Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-empty">● DB Empty</span>',
            unsafe_allow_html=True,
        )
        if db_err and "empty" not in db_err.lower():
            st.error(f"DB Error: {db_err}")


    build_btn = st.button(
        "🔨 Build Knowledge Base",
        disabled=selected_pdf is None,
        use_container_width=True,
    )

    if build_btn and selected_pdf:
        from ingest import build_vector_store

        logger.info("🔨 Knowledge base build request started for PDF: %s", selected_pdf.name)
        
        # ── Release database locks before rebuilding ─────────────────────────
        _load_retriever.clear()
        st.session_state.retriever = None
        st.session_state.sections = []
        import gc
        gc.collect()
        time.sleep(0.5)  # Give SQLite a moment to close active file handles

        progress_bar = st.progress(0, text="Starting …")
        status_text = st.empty()
        stage_map = {"extract": 0.10, "sections": 0.20, "chunk": 0.35,
                     "embed": 0.55, "store": 0.85, "done": 1.0}

        def _progress_cb(stage: str, detail: str):
            logger.info("Ingestion progress [%s]: %s", stage, detail)
            progress_bar.progress(stage_map.get(stage, 0.5), text=detail)
            status_text.caption(detail)

        try:
            count = build_vector_store(selected_pdf, progress_callback=_progress_cb)

            logger.info("✅ Knowledge base built successfully. Total chunks stored: %d", count)
            st.success(f"✅ Knowledge base built — **{count}** chunks stored.")
            st.session_state.db_ready = True
            # Reset cached retriever so it reloads with new data
            _load_retriever.clear()
            st.session_state.retriever = None
            st.session_state.sections = []
        except Exception as exc:
            logger.error("❌ Ingestion failed: %s", exc, exc_info=True)
            st.error(f"❌ Ingestion failed: {exc}")

    # ── Filters ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Search Filters")

    section_filter = None
    if db_exists or st.session_state.db_ready:
        # Load retriever to get sections
        try:
            if st.session_state.retriever is None:
                st.session_state.retriever = _load_retriever()
                st.session_state.sections = st.session_state.retriever.get_sections()
        except Exception:
            pass

        if st.session_state.sections:
            section_options = ["All Sections"] + st.session_state.sections
            chosen = st.selectbox("Filter by section", section_options)
            if chosen != "All Sections":
                section_filter = chosen

    # ── API key status ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔑 API Key")
    if GEMINI_API_KEY:
        st.success("Gemini API key loaded ✓")
    else:
        st.error("Missing `GEMINI_API_KEY` in `.env`")

    # ── Clear chat ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

# Display chat history
for entry in st.session_state.chat_history:
    # User message
    st.markdown(
        f'<div class="user-msg"><strong>🧑 You:</strong> {entry["query"]}</div>',
        unsafe_allow_html=True,
    )
    # Bot answer
    st.markdown(
        f'<div class="bot-msg">{entry["answer"]}</div>',
        unsafe_allow_html=True,
    )
    # Citations
    if entry.get("citations"):
        st.markdown(
            f'<div class="citation-box"><strong>📌 Sources:</strong><br>{entry["citations"]}</div>',
            unsafe_allow_html=True,
        )


# ── Question input ────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about NVIDIA's 10-K report …")

if query:
    # Validate prerequisites
    if not (_check_db() or st.session_state.db_ready):
        logger.warning("❌ Query rejected: Knowledge base is empty.")
        st.error("⚠️ Knowledge base is empty. Build it first using the sidebar.")
        st.stop()

    if not GEMINI_API_KEY:
        logger.warning("❌ Query rejected: GEMINI_API_KEY is not set.")
        st.error("⚠️ `GEMINI_API_KEY` not set in `.env`. Cannot generate answers.")
        st.stop()

    logger.info("🧑 User asked question: '%s' [Filter: %s]", query, section_filter or "None")

    # ── Retrieve ──────────────────────────────────────────────────────────
    logger.info("🔍 Processing question: Searching knowledge base...")
    with st.spinner("🔍 Searching knowledge base …"):
        try:
            if st.session_state.retriever is None:
                st.session_state.retriever = _load_retriever()
                st.session_state.sections = st.session_state.retriever.get_sections()

            chunks = st.session_state.retriever.retrieve(
                query, section_filter=section_filter
            )
            logger.info("✅ Retrieval complete. Obtained %d relevant chunks.", len(chunks))
        except Exception as exc:
            logger.error("❌ Retrieval error: %s", exc, exc_info=True)
            st.error(f"Retrieval error: {exc}")
            st.stop()

    # ── Generate ──────────────────────────────────────────────────────────
    logger.info("🤖 Processing question: Generating answer with Gemini...")
    with st.spinner("🤖 Generating answer …"):
        try:
            import importlib
            import generator
            importlib.reload(generator)
            
            gen = generator.GeminiGenerator(document_name=active_title)
            result = gen.generate_answer(query, chunks)

            
            if "Error generating answer" in result["answer"] or result["answer"].startswith("Error:"):
                logger.error("❌ Gemini generation failed. Response content: %s", result["answer"])
            else:
                logger.info("✅ Gemini generated answer successfully. Citations:\n%s", result["citations"])
        except Exception as exc:
            logger.error("❌ Generation error: %s", exc, exc_info=True)
            st.error(f"Generation error: {exc}")
            st.stop()

    # ── Store in history ──────────────────────────────────────────────────
    st.session_state.chat_history.append(
        {
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "chunks": chunks,
        }
    )
    st.rerun()
