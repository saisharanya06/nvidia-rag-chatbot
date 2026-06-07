"""
ingest.py
=========
Document ingestion pipeline for the NVIDIA 10-K RAG application.

Responsibilities:
  1. Extract text page-by-page from the PDF via pdfplumber.
  2. Detect 10-K section boundaries using regex heading patterns.
  3. Chunk the text with RecursiveCharacterTextSplitter (800 tok / 150 overlap).
  4. Embed chunks with BAAI/bge-m3.
  5. Persist chunks + embeddings + metadata into ChromaDB.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pdfplumber
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_PDF,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    SECTION_PATTERNS,
)

logger = logging.getLogger(__name__)

# ── Tokenizer for accurate chunk sizing ──────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    """Return the token count of *text* using the cl100k_base tokenizer."""
    return len(_TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_with_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from every page of a PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[dict]
        Each element: ``{"page_number": int, "text": str}``

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    ValueError
        If the PDF yields zero extractable text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i, "text": text})

    if not pages:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    logger.info("Extracted text from %d pages.", len(pages))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 2. Section Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Walk through extracted pages and assign a *section_name* to each page
    based on 10-K heading patterns defined in ``config.SECTION_PATTERNS``.

    Pages appearing before the first detected heading are labelled
    ``"Front Matter"``.

    Parameters
    ----------
    pages : list[dict]
        Output of :func:`extract_text_with_pages`.

    Returns
    -------
    list[dict]
        Same dicts with an added ``"section_name"`` key.
    """
    current_section = "Front Matter"

    for page in pages:
        # Check the first few lines of the page for a section heading
        lines = page["text"].split("\n")
        for line in lines[:15]:  # headings usually near top of page
            for pattern, section_name in SECTION_PATTERNS:
                if pattern.search(line):
                    current_section = section_name
                    break
        page["section_name"] = current_section

    section_counts: dict[str, int] = {}
    for p in pages:
        section_counts[p["section_name"]] = section_counts.get(p["section_name"], 0) + 1
    logger.info("Detected %d sections across %d pages.", len(section_counts), len(pages))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split section-annotated pages into smaller chunks using
    ``RecursiveCharacterTextSplitter`` with token-based sizing.

    Parameters
    ----------
    pages : list[dict]
        Output of :func:`detect_sections`.

    Returns
    -------
    list[dict]
        Each element::

            {
                "chunk_id": str,       # deterministic hash
                "text": str,
                "page_number": int,
                "section_name": str,
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for fragment in page_chunks:
            chunk_id = hashlib.md5(
                f"{page['page_number']}:{fragment[:80]}".encode()
            ).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": fragment,
                    "page_number": page["page_number"],
                    "section_name": page["section_name"],
                }
            )

    logger.info("Created %d chunks from %d pages.", len(chunks), len(pages))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embedding Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
    progress_callback=None,
) -> tuple[list[list[float]], SentenceTransformer]:
    """
    Generate dense embeddings for every chunk using BAAI/bge-m3.

    Parameters
    ----------
    chunks : list[dict]
        Output of :func:`chunk_documents`.
    model : SentenceTransformer | None
        Optional pre-loaded model (avoids reloading).
    progress_callback : callable | None
        Called with (current_batch_index, total_batches).

    Returns
    -------
    (embeddings, model)
        *embeddings* is a list of float lists aligned with *chunks*.
    """
    if model is None:
        logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    all_embeddings: list[list[float]] = []
    for batch_idx in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[batch_idx : batch_idx + EMBEDDING_BATCH_SIZE]
        batch_emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(batch_emb.tolist())
        if progress_callback:
            progress_callback(batch_idx // EMBEDDING_BATCH_SIZE + 1, total_batches)

    logger.info("Generated %d embeddings.", len(all_embeddings))
    return all_embeddings, model


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChromaDB Persistence
# ─────────────────────────────────────────────────────────────────────────────

def store_in_chromadb(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Persist chunks, embeddings, and metadata into a ChromaDB collection.

    Parameters
    ----------
    chunks : list[dict]
    embeddings : list[list[float]]

    Returns
    -------
    int
        Number of documents stored.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to avoid duplicates on re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB add in batches of 5000 (API limit safeguard)
    batch_size = 5000
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            documents=[c["text"] for c in chunks[start:end]],
            embeddings=embeddings[start:end],
            metadatas=[
                {
                    "page_number": c["page_number"],
                    "section_name": c["section_name"],
                }
                for c in chunks[start:end]
            ],
        )

    count = collection.count()
    logger.info("Stored %d documents in ChromaDB.", count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(
    pdf_path: str | Path | None = None,
    progress_callback=None,
) -> int:
    """
    End-to-end pipeline: PDF → pages → sections → chunks → embeddings → ChromaDB.

    Parameters
    ----------
    pdf_path : str | Path | None
        Path to PDF.  Defaults to ``config.DEFAULT_PDF``.
    progress_callback : callable | None
        ``(stage: str, detail: str)`` called at each major stage.

    Returns
    -------
    int
        Total number of chunks persisted.

    Raises
    ------
    FileNotFoundError, ValueError
        Propagated from extraction or embedding stages.
    """
    pdf_path = Path(pdf_path) if pdf_path else DEFAULT_PDF

    def _cb(stage: str, detail: str = ""):
        if progress_callback:
            progress_callback(stage, detail)

    _cb("extract", f"Extracting text from {pdf_path.name}…")
    pages = extract_text_with_pages(pdf_path)

    _cb("sections", "Detecting document sections…")
    pages = detect_sections(pages)

    _cb("chunk", "Chunking documents…")
    chunks = chunk_documents(pages)

    _cb("embed", f"Generating embeddings for {len(chunks)} chunks…")
    embeddings, _ = generate_embeddings(chunks)

    _cb("store", "Storing in ChromaDB…")
    count = store_in_chromadb(chunks, embeddings)

    _cb("done", f"✅ Stored {count} chunks successfully.")
    return count


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    total = build_vector_store()
    print(f"\n[OK] Ingestion complete - {total} chunks in ChromaDB.")
"""
ingest.py
=========
Document ingestion pipeline for the NVIDIA 10-K RAG application.

Responsibilities:
  1. Extract text page-by-page from the PDF via pdfplumber.
  2. Detect 10-K section boundaries using regex heading patterns.
  3. Chunk the text with RecursiveCharacterTextSplitter (800 tok / 150 overlap).
  4. Embed chunks with BAAI/bge-m3.
  5. Persist chunks + embeddings + metadata into ChromaDB.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pdfplumber
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_PDF,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    SECTION_PATTERNS,
)

logger = logging.getLogger(__name__)

# ── Tokenizer for accurate chunk sizing ──────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    """Return the token count of *text* using the cl100k_base tokenizer."""
    return len(_TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_with_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from every page of a PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[dict]
        Each element: ``{"page_number": int, "text": str}``

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    ValueError
        If the PDF yields zero extractable text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i, "text": text})

    if not pages:
        raise ValueError(f"No extractable text found in: {pdf_path}")

    logger.info("Extracted text from %d pages.", len(pages))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 2. Section Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Walk through extracted pages and assign a *section_name* to each page
    based on 10-K heading patterns defined in ``config.SECTION_PATTERNS``.

    Pages appearing before the first detected heading are labelled
    ``"Front Matter"``.

    Parameters
    ----------
    pages : list[dict]
        Output of :func:`extract_text_with_pages`.

    Returns
    -------
    list[dict]
        Same dicts with an added ``"section_name"`` key.
    """
    current_section = "Front Matter"

    for page in pages:
        # Check the first few lines of the page for a section heading
        lines = page["text"].split("\n")
        for line in lines[:15]:  # headings usually near top of page
            for pattern, section_name in SECTION_PATTERNS:
                if pattern.search(line):
                    current_section = section_name
                    break
        page["section_name"] = current_section

    section_counts: dict[str, int] = {}
    for p in pages:
        section_counts[p["section_name"]] = section_counts.get(p["section_name"], 0) + 1
    logger.info("Detected %d sections across %d pages.", len(section_counts), len(pages))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split section-annotated pages into smaller chunks using
    ``RecursiveCharacterTextSplitter`` with token-based sizing.

    Parameters
    ----------
    pages : list[dict]
        Output of :func:`detect_sections`.

    Returns
    -------
    list[dict]
        Each element::

            {
                "chunk_id": str,       # deterministic hash
                "text": str,
                "page_number": int,
                "section_name": str,
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for fragment in page_chunks:
            chunk_id = hashlib.md5(
                f"{page['page_number']}:{fragment[:80]}".encode()
            ).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": fragment,
                    "page_number": page["page_number"],
                    "section_name": page["section_name"],
                }
            )

    logger.info("Created %d chunks from %d pages.", len(chunks), len(pages))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embedding Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
    progress_callback=None,
) -> tuple[list[list[float]], SentenceTransformer]:
    """
    Generate dense embeddings for every chunk using BAAI/bge-m3.

    Parameters
    ----------
    chunks : list[dict]
        Output of :func:`chunk_documents`.
    model : SentenceTransformer | None
        Optional pre-loaded model (avoids reloading).
    progress_callback : callable | None
        Called with (current_batch_index, total_batches).

    Returns
    -------
    (embeddings, model)
        *embeddings* is a list of float lists aligned with *chunks*.
    """
    if model is None:
        logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    all_embeddings: list[list[float]] = []
    for batch_idx in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[batch_idx : batch_idx + EMBEDDING_BATCH_SIZE]
        batch_emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(batch_emb.tolist())
        if progress_callback:
            progress_callback(batch_idx // EMBEDDING_BATCH_SIZE + 1, total_batches)

    logger.info("Generated %d embeddings.", len(all_embeddings))
    return all_embeddings, model


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChromaDB Persistence
# ─────────────────────────────────────────────────────────────────────────────

def store_in_chromadb(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Persist chunks, embeddings, and metadata into a ChromaDB collection.

    Parameters
    ----------
    chunks : list[dict]
    embeddings : list[list[float]]

    Returns
    -------
    int
        Number of documents stored.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to avoid duplicates on re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'.", COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB add in batches of 5000 (API limit safeguard)
    batch_size = 5000
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            documents=[c["text"] for c in chunks[start:end]],
            embeddings=embeddings[start:end],
            metadatas=[
                {
                    "page_number": c["page_number"],
                    "section_name": c["section_name"],
                }
                for c in chunks[start:end]
            ],
        )

    count = collection.count()
    logger.info("Stored %d documents in ChromaDB.", count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(
    pdf_path: str | Path | None = None,
    progress_callback=None,
) -> int:
    """
    End-to-end pipeline: PDF → pages → sections → chunks → embeddings → ChromaDB.

    Parameters
    ----------
    pdf_path : str | Path | None
        Path to PDF.  Defaults to ``config.DEFAULT_PDF``.
    progress_callback : callable | None
        ``(stage: str, detail: str)`` called at each major stage.

    Returns
    -------
    int
        Total number of chunks persisted.

    Raises
    ------
    FileNotFoundError, ValueError
        Propagated from extraction or embedding stages.
    """
    pdf_path = Path(pdf_path) if pdf_path else DEFAULT_PDF

    def _cb(stage: str, detail: str = ""):
        if progress_callback:
            progress_callback(stage, detail)

    _cb("extract", f"Extracting text from {pdf_path.name}…")
    pages = extract_text_with_pages(pdf_path)

    _cb("sections", "Detecting document sections…")
    pages = detect_sections(pages)

    _cb("chunk", "Chunking documents…")
    chunks = chunk_documents(pages)

    _cb("embed", f"Generating embeddings for {len(chunks)} chunks…")
    embeddings, _ = generate_embeddings(chunks)

    _cb("store", "Storing in ChromaDB…")
    count = store_in_chromadb(chunks, embeddings)

    _cb("done", f"✅ Stored {count} chunks successfully.")
    return count


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    total = build_vector_store()
    print(f"\n[OK] Ingestion complete - {total} chunks in ChromaDB.")
