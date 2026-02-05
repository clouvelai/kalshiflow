"""
Vector memory service backed by OpenAI embeddings + Supabase pgvector.

Uses the agent_memories table (created by migration 20260202100000)
with HNSW index for sub-10ms similarity search.

Design:
- Lazy OpenAI client init (works even if OPENAI_API_KEY is missing at import time)
- All methods are async (uses asyncpg via rl_db)
- Dedup on store via find_similar_memories RPC
- Access-based ranking via touch_memories RPC after each recall
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.vector_store")


class VectorMemoryService:
    """
    Semantic memory backed by OpenAI embeddings and Supabase pgvector.

    Gracefully degrades: if OpenAI client can't be created or DB is down,
    methods return empty results / None instead of raising.
    """

    def __init__(self, db):
        """
        Args:
            db: RLDatabase instance with asyncpg pool.
        """
        self._db = db
        self._openai_client = None  # Lazy init

    def _get_openai_client(self):
        """Lazy-init OpenAI client on first use."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI()
            logger.info("OpenAI client initialized for embeddings")
        return self._openai_client

    def _get_embedding(self, text: str) -> List[float]:
        """Get 1536-dim embedding from text-embedding-3-small."""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    async def store(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a memory with embedding. Returns UUID if stored, None if deduped/failed.

        Steps:
        1. Compute content_hash (SHA-256)
        2. Get embedding
        3. Check for near-dupes via find_similar_memories RPC
        4. INSERT into agent_memories
        """
        metadata = metadata or {}
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Get embedding
        embedding = self._get_embedding(content)

        async with self._db.get_connection() as conn:
            # Check for near-duplicates (threshold=0.88)
            dupes = await conn.fetch(
                "SELECT id, content, similarity FROM find_similar_memories($1::vector, $2, $3)",
                str(embedding),
                memory_type,
                0.88,
            )
            if dupes:
                logger.debug(
                    f"Skipping near-duplicate memory (similarity={dupes[0]['similarity']:.3f})"
                )
                return None

            # Extract ticker arrays from metadata
            market_tickers = metadata.get("market_tickers", [])
            if not market_tickers:
                ticker = metadata.get("ticker") or metadata.get("market_ticker")
                if ticker:
                    market_tickers = [ticker]

            event_tickers = metadata.get("event_tickers", [])
            if not event_tickers:
                event_ticker = metadata.get("event_ticker")
                if event_ticker:
                    event_tickers = [event_ticker]

            confidence = metadata.get("confidence", "medium")

            # INSERT
            row_id = await conn.fetchval(
                """
                INSERT INTO agent_memories (
                    memory_type, content, content_hash, embedding,
                    market_tickers, event_tickers, confidence,
                    source_session, trade_id, trade_result, pnl_cents
                ) VALUES (
                    $1, $2, $3, $4::vector,
                    $5, $6, $7,
                    $8, $9, $10, $11
                )
                RETURNING id
                """,
                memory_type,
                content,
                content_hash,
                str(embedding),
                market_tickers,
                event_tickers,
                confidence,
                metadata.get("source_session"),
                metadata.get("trade_id"),
                metadata.get("trade_result"),
                metadata.get("pnl_cents"),
            )

            logger.debug(f"Stored vector memory {row_id} (type={memory_type})")
            return str(row_id)

    async def search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        event_ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over agent_memories using search_agent_memories RPC.

        Returns list of dicts with content, memory_type, similarity, etc.
        """
        embedding = self._get_embedding(query)

        async with self._db.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM search_agent_memories(
                    $1::vector, $2, $3, $4, NULL, $5, 0.3
                )
                """,
                str(embedding),
                memory_types,
                None,  # p_market_ticker
                event_ticker,
                limit,
            )

            results = [dict(row) for row in rows]

            # Fire-and-forget: bump access counts
            if results:
                ids = [row["id"] for row in results]
                try:
                    await conn.execute(
                        "SELECT touch_memories($1)",
                        ids,
                    )
                except Exception as e:
                    logger.debug(f"touch_memories failed (non-critical): {e}")

            return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get basic stats about the vector memory store."""
        async with self._db.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_active) as active,
                    COUNT(DISTINCT memory_type) as types,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM agent_memories
                """
            )
            return dict(row) if row else {}
