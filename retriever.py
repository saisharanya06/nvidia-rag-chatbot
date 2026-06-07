"""
retriever.py
============
Hybrid retrieval engine for the NVIDIA 10-K RAG application.

Combines:
  • Dense vector search  (ChromaDB + BAAI/bge-m3)
  • Sparse keyword search (BM25)
  • Reciprocal Rank Fusion to merge results
  • Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
import time

# Optimize PyTorch CPU execution on constrained containers (Streamlit Cloud)
torch.set_num_threads(1)


from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RRF_K,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)


# ── Utility ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Retriever that fuses dense (vector) and sparse (BM25) search results,
    then reranks the union with a cross-encoder.

    Usage
    -----
    >>> retriever = HybridRetriever()
    >>> results = retriever.retrieve("What are NVIDIA's risk factors?")
    """

    def __init__(self) -> None:
        """
        Load models and build the BM25 index from the ChromaDB collection.

        Raises
        ------
        RuntimeError
            If the ChromaDB collection is empty or missing.
        """
        # ── ChromaDB ──────────────────────────────────────────────────────
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found. "
                "Please build the knowledge base first."
            ) from exc

        doc_count = self._collection.count()
        if doc_count == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Please ingest the PDF first."
            )
        logger.info("Loaded ChromaDB collection with %d documents.", doc_count)

        # ── Fetch all documents for BM25 index ────────────────────────────
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._all_ids: list[str] = all_data["ids"]
        self._all_docs: list[str] = all_data["documents"]
        self._all_metas: list[dict] = all_data["metadatas"]

        # Get active file name from metadata
        self.active_file_name = "Document"
        if self._all_metas:
            self.active_file_name = self._all_metas[0].get("file_name", "Document")


        # ── BM25 index ────────────────────────────────────────────────────
        tokenized_corpus = [_tokenize(doc) for doc in self._all_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents.", len(self._all_docs))

        # ── Embedding model ───────────────────────────────────────────────
        logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        # ── Reranker ──────────────────────────────────────────────────────
        logger.info("Loading reranker: %s …", RERANKER_MODEL)
        self._reranker = CrossEncoder(RERANKER_MODEL, device="cpu")


    # ── Vector search ────────────────────────────────────────────────────────

    def vector_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense semantic search using ChromaDB.

        Parameters
        ----------
        query : str
        k : int
        where : dict | None
            ChromaDB ``where`` filter (e.g. ``{"section_name": "Item 1A …"}``).

        Returns
        -------
        list[dict]
            Each dict: ``{id, text, metadata, score}``
        """
        # bge-m3 recommends a search instruction prefix
        query_embedding = self._embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        hits: list[dict[str, Any]] = []
        for idx in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "score": 1 - results["distances"][0][idx],  # cosine dist → similarity
                }
            )
        return hits

    # ── BM25 search ──────────────────────────────────────────────────────────

    def bm25_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        section_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Sparse keyword search using BM25Okapi.

        Parameters
        ----------
        query : str
        k : int
        section_filter : str | None
            If set, only consider docs from this section.

        Returns
        -------
        list[dict]
        """
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Build (index, score) pairs, optionally filtered
        scored: list[tuple[int, float]] = []
        for i, s in enumerate(scores):
            if section_filter and self._all_metas[i].get("section_name") != section_filter:
                continue
            scored.append((i, float(s)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        return [
            {
                "id": self._all_ids[i],
                "text": self._all_docs[i],
                "metadata": self._all_metas[i],
                "score": s,
            }
            for i, s in top
        ]

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────────

    @staticmethod
    def reciprocal_rank_fusion(
        *result_lists: list[dict[str, Any]],
        k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.

        RRF score for a document = Σ  1 / (k + rank_i)

        Parameters
        ----------
        *result_lists : list[dict]
            One or more ranked result lists.
        k : int
            Smoothing constant (default 60).

        Returns
        -------
        list[dict]
            Merged results sorted by fused score.
        """
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc["id"]
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        merged = []
        for doc_id in sorted_ids:
            entry = dict(doc_map[doc_id])
            entry["rrf_score"] = fused_scores[doc_id]
            merged.append(entry)

        return merged

    # ── Cross-encoder reranking ──────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = TOP_K_RERANK,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate chunks with the cross-encoder model.

        Parameters
        ----------
        query : str
        candidates : list[dict]
        top_n : int

        Returns
        -------
        list[dict]
            Top-n candidates with ``rerank_score`` added.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    # ── Full retrieval pipeline ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        section_filter: str | None = None,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_k_rerank: int = TOP_K_RERANK,
    ) -> list[dict[str, Any]]:
        """
        Full hybrid retrieval pipeline:

        Query → (Vector Search ∪ BM25) → RRF Merge → Rerank → Top-k

        Parameters
        ----------
        query : str
        section_filter : str | None
            Restrict retrieval to a specific section.
        top_k_retrieval : int
            Candidates per retriever.
        top_k_rerank : int
            Final results after reranking.

        Returns
        -------
        list[dict]
            Final ranked chunks with metadata and scores.
        """
        # Build optional ChromaDB where clause
        where = None
        if section_filter:
            where = {"section_name": section_filter}

        # 1. Dense retrieval
        t_start = time.time()
        vec_results = self.vector_search(query, k=top_k_retrieval, where=where)
        t_vector = time.time() - t_start
        logger.info("Vector search returned %d results in %.2f seconds.", len(vec_results), t_vector)

        # 2. Sparse retrieval
        t_start = time.time()
        bm25_results = self.bm25_search(
            query, k=top_k_retrieval, section_filter=section_filter
        )
        t_bm25 = time.time() - t_start
        logger.info("BM25 search returned %d results in %.2f seconds.", len(bm25_results), t_bm25)

        # 3. Reciprocal Rank Fusion
        merged = self.reciprocal_rank_fusion(vec_results, bm25_results)
        logger.info("RRF merged to %d unique candidates.", len(merged))

        # Take top candidates for reranking (limit to save compute)
        candidates = merged[: top_k_retrieval * 2]

        # 4. Cross-encoder reranking
        t_start = time.time()
        final = self.rerank(query, candidates, top_n=top_k_rerank)
        t_rerank = time.time() - t_start
        logger.info("Reranked to top %d results in %.2f seconds.", len(final), t_rerank)
        logger.info("Total retrieval + reranking time: %.2f seconds.", (t_vector + t_bm25 + t_rerank))

        return final


    # ── Available sections ───────────────────────────────────────────────────

    def get_sections(self) -> list[str]:
        """Return a sorted list of unique section names in the collection."""
        sections = sorted({m.get("section_name", "") for m in self._all_metas})
        return [s for s in sections if s]
