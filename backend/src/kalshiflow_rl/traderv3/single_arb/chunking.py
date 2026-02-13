"""Chunking pipeline for splitting news articles into embeddable chunks.

Key Responsibilities:
    - Split news articles into appropriately sized chunks for embedding
    - Preserve paragraph boundaries for medium-length articles
    - Use fixed-size overlapping windows for large articles
    - Propagate parent metadata to all child chunks

Architecture Position:
    Sits between news fetching (event_understanding / news enrichment) and the
    vector memory layer (SessionMemoryStore / pgvector). Raw article text goes in,
    a list of ArticleChunk dataclasses comes out, ready for embedding.

Design Principles:
    - No external dependencies (stdlib only, no tiktoken)
    - Fast token estimation via len(text) // 4 heuristic
    - Three-tier strategy: passthrough (<500 tok), paragraph (500-1500 tok),
      fixed-window (>1500 tok)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.chunking")


@dataclass
class ArticleChunk:
    """A single embeddable chunk derived from a news article."""

    content: str
    chunk_index: int
    total_chunks: int
    parent_url: str
    token_count: int
    metadata: Dict = field(default_factory=dict)


class ChunkingPipeline:
    """Splits news articles into embeddable chunks using a three-tier strategy.

    - Small articles  (<500 tokens):  returned as a single chunk
    - Medium articles (500-1500 tokens): split at paragraph boundaries
    - Large articles  (>1500 tokens): 512-token chunks with 64-token overlap
    """

    # ------------------------------------------------------------------ #
    #  Tier thresholds (in estimated tokens)
    # ------------------------------------------------------------------ #
    _SMALL_THRESHOLD = 500
    _MEDIUM_THRESHOLD = 1500
    _FIXED_CHUNK_SIZE = 512
    _FIXED_OVERLAP = 64

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast token estimate: ~4 characters per token."""
        return len(text) // 4

    @staticmethod
    def chunk_article(content: str, metadata: Dict) -> List[ArticleChunk]:
        """Split *content* into embeddable chunks, inheriting *metadata*.

        Returns a list of :class:`ArticleChunk` instances.  The ``parent_url``
        field is pulled from ``metadata.get("news_url", "")``.
        """
        if not content or not content.strip():
            return []

        parent_url = metadata.get("news_url", "")
        token_est = ChunkingPipeline.estimate_tokens(content)

        # -- small: single chunk ----------------------------------------
        if token_est < ChunkingPipeline._SMALL_THRESHOLD:
            logger.debug(
                "Small article (%d tokens), returning single chunk", token_est
            )
            return [
                ArticleChunk(
                    content=content,
                    chunk_index=0,
                    total_chunks=1,
                    parent_url=parent_url,
                    token_count=token_est,
                    metadata=dict(metadata),
                )
            ]

        # -- medium: paragraph splitting --------------------------------
        if token_est <= ChunkingPipeline._MEDIUM_THRESHOLD:
            logger.debug(
                "Medium article (%d tokens), splitting at paragraph boundaries",
                token_est,
            )
            parts = ChunkingPipeline._split_paragraphs(
                content, max_tokens=ChunkingPipeline._FIXED_CHUNK_SIZE
            )
        else:
            # -- large: fixed-size overlapping windows ------------------
            logger.debug(
                "Large article (%d tokens), using fixed-size chunking", token_est
            )
            parts = ChunkingPipeline._split_fixed(
                content,
                chunk_size=ChunkingPipeline._FIXED_CHUNK_SIZE,
                overlap=ChunkingPipeline._FIXED_OVERLAP,
            )

        # Build ArticleChunk list
        total = len(parts)
        chunks: List[ArticleChunk] = []
        for idx, part in enumerate(parts):
            chunks.append(
                ArticleChunk(
                    content=part,
                    chunk_index=idx,
                    total_chunks=total,
                    parent_url=parent_url,
                    token_count=ChunkingPipeline.estimate_tokens(part),
                    metadata=dict(metadata),
                )
            )
        return chunks

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_paragraphs(text: str, max_tokens: int) -> List[str]:
        """Split *text* at ``\\n\\n`` boundaries, merging small paragraphs.

        Adjacent paragraphs are merged together until the next paragraph would
        push the accumulated chunk over *max_tokens*.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return [text]

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = ChunkingPipeline.estimate_tokens(para)

            # If a single paragraph exceeds max_tokens, flush current then
            # add the oversized paragraph as its own chunk.
            if para_tokens > max_tokens:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                chunks.append(para)
                continue

            # Would adding this paragraph exceed the budget?
            # Account for the "\n\n" separator between parts.
            separator_tokens = (
                ChunkingPipeline.estimate_tokens("\n\n") if current_parts else 0
            )
            if current_tokens + separator_tokens + para_tokens > max_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens + (
                ChunkingPipeline.estimate_tokens("\n\n") if len(current_parts) > 1 else 0
            )

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    @staticmethod
    def _split_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fixed-size character-level splits with overlap.

        *chunk_size* and *overlap* are expressed in tokens.  Since we estimate
        4 characters per token, the actual character windows are
        ``chunk_size * 4`` and ``overlap * 4``.
        """
        char_chunk = chunk_size * 4
        char_overlap = overlap * 4

        if char_chunk <= 0:
            return [text]

        stride = max(char_chunk - char_overlap, 1)
        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + char_chunk
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += stride

        return chunks if chunks else [text]
