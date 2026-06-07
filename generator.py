"""
generator.py
============
LLM answer generation for the NVIDIA 10-K RAG application.

Uses Google Gemini 2.5 Flash to synthesise answers strictly from
retrieved context chunks, with source citations.
"""

from __future__ import annotations

import logging
from typing import Any

import google.generativeai as genai

from config import GEMINI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """You are an AI financial assistant specializing in analyzing the NVIDIA 2025 Annual Report (10-K).

RULES:
1. Answer the user's question ONLY using the provided context passages.
2. If asked about your identity (e.g., "who are you", "what is your purpose", "what do you do"), explain that you are an AI assistant designed to help analyze and answer questions about the NVIDIA 2025 Annual Report (10-K).
3. Do NOT use robotic introduction phrases such as "Based on the provided context", "According to the document", or "Based on the above information". Answer questions naturally, directly, and professionally as an expert assistant.
4. If the context does not contain enough information to answer a factual question, say:
   "I could not find sufficient information in the retrieved context to answer this question."
5. When referencing information, naturally mention the source section and page number in-text.
6. Do NOT append a "Sources" list or bibliography section at the end of your answer.
7. Use markdown formatting for readability (bullets, bold, tables when appropriate).
"""




class GeminiGenerator:
    """
    Wrapper around the Google Generative AI SDK for grounded answer generation.

    Parameters
    ----------
    api_key : str | None
        Gemini API key.  Falls back to ``config.GEMINI_API_KEY``.

    Raises
    ------
    ValueError
        If no API key is available.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass it explicitly."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        logger.info("Gemini generator initialised (model=%s).", LLM_MODEL)

    # ── Prompt construction ──────────────────────────────────────────────────

    @staticmethod
    def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
        """
        Build a user prompt embedding retrieved context chunks.

        Each chunk is clearly delimited and tagged with its source metadata
        so the LLM can cite accurately.

        Parameters
        ----------
        query : str
        chunks : list[dict]
            Must contain ``text`` and ``metadata`` (with ``section_name``,
            ``page_number``).

        Returns
        -------
        str
        """
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            section = meta.get("section_name", "Unknown")
            page = meta.get("page_number", "?")
            context_parts.append(
                f"--- Context Passage {i} ---\n"
                f"[Source: {section} | Page {page}]\n"
                f"{chunk['text']}\n"
            )

        context_block = "\n".join(context_parts)
        return (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Using ONLY the context above, provide a detailed answer with citations."
        )

    # ── Citation extraction ──────────────────────────────────────────────────

    @staticmethod
    def format_citations(chunks: list[dict[str, Any]]) -> str:
        """
        Build a deduplicated citation block from chunk metadata.

        Returns
        -------
        str
            Formatted citation lines.
        """
        seen: set[tuple[str, int]] = set()
        lines: list[str] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            section = meta.get("section_name", "Unknown")
            page = meta.get("page_number", "?")
            key = (section, page)
            if key not in seen:
                seen.add(key)
                lines.append(f"- {section} (Page {page})")
        return "\n".join(lines)

    # ── Answer generation ────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Generate a grounded answer with citations.

        Parameters
        ----------
        query : str
        chunks : list[dict]

        Returns
        -------
        dict
            ``{"answer": str, "citations": str, "prompt": str}``
        """
        if not chunks:
            return {
                "answer": "No relevant chunks were retrieved. "
                          "Please try rephrasing your question.",
                "citations": "",
                "prompt": "",
            }

        prompt = self.build_prompt(query, chunks)
        logger.info("Sending prompt to Gemini (%d chars).", len(prompt))

        try:
            from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

            def is_retryable_error(exception: Exception) -> bool:
                exc_str = str(exception).upper()
                return any(k in exc_str for k in ["503", "502", "429", "UNAVAILABLE", "EXHAUSTED", "LIMIT"])

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=2, min=3, max=10),
                retry=retry_if_exception(is_retryable_error),
                reraise=True
            )
            def call_api():
                return self._model.generate_content(prompt, request_options={"timeout": 15.0})

            logger.info("Calling Gemini API (timeout=15s, 3 attempts max)...")
            response = call_api()
            answer_text = response.text
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            answer_text = f"⚠️ Error generating answer: {exc}"


        citations = self.format_citations(chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "prompt": prompt,
        }
