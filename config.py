"""
config.py
=========
Central configuration for the NVIDIA 10-K RAG application.
All tuneable parameters, paths, and model identifiers live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DEFAULT_PDF = DATA_DIR / "nasdaq-nvda-2025-10K-25670928.pdf"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32

# ── Reranker model ───────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking parameters ─────────────────────────────────────────────────────
CHUNK_SIZE = 800       # tokens
CHUNK_OVERLAP = 150    # tokens

# ── Retrieval parameters ────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # candidates from each retriever before reranking
TOP_K_RERANK = 7       # final chunks after reranking
RRF_K = 60             # reciprocal rank fusion constant

# ── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "nvidia_10k"

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "gemini-flash-lite-latest"

# ── 10-K Section heading patterns ────────────────────────────────────────────
# Ordered list of (regex_pattern, canonical_section_name).
# Patterns are matched case-insensitively against each line of extracted text.
import re

SECTION_PATTERNS = [
    (re.compile(r"(?i)^\s*item\s+1a[\.\s:\-—]+risk\s+factors"),           "Item 1A – Risk Factors"),
    (re.compile(r"(?i)^\s*item\s+1b[\.\s:\-—]+unresolved\s+staff"),       "Item 1B – Unresolved Staff Comments"),
    (re.compile(r"(?i)^\s*item\s+1c[\.\s:\-—]+cybersecurity"),            "Item 1C – Cybersecurity"),
    (re.compile(r"(?i)^\s*item\s+1[\.\s:\-—]+business"),                  "Item 1 – Business"),
    (re.compile(r"(?i)^\s*item\s+2[\.\s:\-—]+properties"),                "Item 2 – Properties"),
    (re.compile(r"(?i)^\s*item\s+3[\.\s:\-—]+legal\s+proceedings"),       "Item 3 – Legal Proceedings"),
    (re.compile(r"(?i)^\s*item\s+4[\.\s:\-—]+mine\s+safety"),             "Item 4 – Mine Safety Disclosures"),
    (re.compile(r"(?i)^\s*item\s+5[\.\s:\-—]+market"),                    "Item 5 – Market Information"),
    (re.compile(r"(?i)^\s*item\s+6[\.\s:\-—]+"),                          "Item 6 – Reserved"),
    (re.compile(r"(?i)^\s*item\s+7a[\.\s:\-—]+quantitative"),             "Item 7A – Quantitative Disclosures"),
    (re.compile(r"(?i)^\s*item\s+7[\.\s:\-—]+management"),                "Item 7 – MD&A"),
    (re.compile(r"(?i)^\s*item\s+8[\.\s:\-—]+financial\s+statements"),    "Item 8 – Financial Statements"),
    (re.compile(r"(?i)^\s*item\s+9a[\.\s:\-—]+controls"),                 "Item 9A – Controls and Procedures"),
    (re.compile(r"(?i)^\s*item\s+9b[\.\s:\-—]+other\s+information"),      "Item 9B – Other Information"),
    (re.compile(r"(?i)^\s*item\s+9[\.\s:\-—]+changes"),                   "Item 9 – Disagreements on Accounting"),
    (re.compile(r"(?i)^\s*item\s+10[\.\s:\-—]+directors"),                "Item 10 – Directors and Officers"),
    (re.compile(r"(?i)^\s*item\s+11[\.\s:\-—]+executive\s+compensation"), "Item 11 – Executive Compensation"),
    (re.compile(r"(?i)^\s*item\s+12[\.\s:\-—]+security\s+ownership"),     "Item 12 – Security Ownership"),
    (re.compile(r"(?i)^\s*item\s+13[\.\s:\-—]+certain\s+relationships"),  "Item 13 – Certain Relationships"),
    (re.compile(r"(?i)^\s*item\s+14[\.\s:\-—]+principal\s+accountant"),   "Item 14 – Principal Accountant Fees"),
    (re.compile(r"(?i)^\s*item\s+15[\.\s:\-—]+exhibits"),                 "Item 15 – Exhibits and Schedules"),
    (re.compile(r"(?i)^\s*item\s+16[\.\s:\-—]+form\s+10-k\s+summary"),   "Item 16 – Form 10-K Summary"),
]
