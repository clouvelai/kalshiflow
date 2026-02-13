"""
Vector memory service backed by OpenAI embeddings + Supabase pgvector.

Uses the agent_memories table (created by migration 20260202100000)
with HNSW index for sub-10ms similarity search.

Design:
- Lazy OpenAI client init (works even if OPENAI_API_KEY is missing at import time)
- All methods are async (uses asyncpg via rl_db)
- Embedding calls run via asyncio.to_thread() to avoid blocking the event loop
- Dedup on store via find_similar_memories RPC
- Access-based ranking via touch_memories RPC after each recall
- Batch embedding support for chunked article storage
"""

import asyncio
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

    # text-embedding-3-small has an 8192 token limit.
    # ~4 chars per token on average; cap at 24000 chars (~6000 tokens) for safety headroom.
    _MAX_EMBEDDING_CHARS = 24000

    def _truncate_for_embedding(self, text: str) -> str:
        """Truncate text to fit within the embedding model's token limit.

        text-embedding-3-small supports 8192 tokens. We cap at ~6000 tokens
        (~24000 characters) to leave headroom for tokenization variance.
        """
        if len(text) <= self._MAX_EMBEDDING_CHARS:
            return text
        logger.debug(
            f"Truncating text for embedding: {len(text)} chars -> {self._MAX_EMBEDDING_CHARS} chars"
        )
        return text[: self._MAX_EMBEDDING_CHARS]

    def _get_embedding_sync(self, text: str) -> List[float]:
        """Get 1536-dim embedding using configured embedding model (synchronous).

        Must be called via asyncio.to_thread() from async contexts to avoid
        blocking the event loop.
        """
        from ..mentions_models import get_embedding_model

        client = self._get_openai_client()
        response = client.embeddings.create(
            model=get_embedding_model(),
            input=self._truncate_for_embedding(text),
        )
        return response.data[0].embedding

    def _get_embeddings_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a single API call (synchronous).

        Much more efficient than N individual calls for chunked articles.
        Must be called via asyncio.to_thread().
        """
        from ..mentions_models import get_embedding_model

        client = self._get_openai_client()
        truncated = [self._truncate_for_embedding(t) for t in texts]
        response = client.embeddings.create(
            model=get_embedding_model(),
            input=truncated,
        )
        # Sort by index to ensure correct ordering
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding without blocking the event loop."""
        return await asyncio.to_thread(self._get_embedding_sync, text)

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings without blocking the event loop."""
        return await asyncio.to_thread(self._get_embeddings_batch_sync, texts)

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

        # Get embedding (non-blocking)
        embedding = await self._get_embedding(content)

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

            return await self._insert_memory(
                conn, content, content_hash, embedding, memory_type, metadata
            )

    async def store_chunked(
        self,
        content: str,
        memory_type: str = "news",
        metadata: Optional[Dict[str, Any]] = None,
        chunks: Optional[List] = None,
    ) -> Optional[str]:
        """Store a parent article + N chunks with batch embedding.

        Uses a single embedding API call for all chunks + parent.
        Each chunk links to parent via parent_memory_id.

        Returns parent UUID if stored, None if deduped/failed.
        """
        metadata = metadata or {}
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Collect all texts for batch embedding: [parent, chunk0, chunk1, ...]
        chunk_texts = []
        if chunks:
            chunk_texts = [
                c.content if hasattr(c, "content") else str(c) for c in chunks
            ]

        all_texts = [content] + chunk_texts

        # Batch embed (single API call)
        all_embeddings = await self._get_embeddings_batch(all_texts)
        parent_embedding = all_embeddings[0]
        chunk_embeddings = all_embeddings[1:]

        async with self._db.get_connection() as conn:
            # Check for near-duplicates on the parent
            dupes = await conn.fetch(
                "SELECT id, content, similarity FROM find_similar_memories($1::vector, $2, $3)",
                str(parent_embedding),
                memory_type,
                0.88,
            )
            if dupes:
                logger.debug(
                    f"Skipping near-duplicate parent (similarity={dupes[0]['similarity']:.3f})"
                )
                return None

            # Insert parent
            parent_id = await self._insert_memory(
                conn, content, content_hash, parent_embedding, memory_type, metadata
            )
            if not parent_id:
                return None

            # Insert chunks with parent_memory_id link
            if chunks and chunk_embeddings:
                for i, (chunk, embedding) in enumerate(
                    zip(chunks, chunk_embeddings)
                ):
                    chunk_content = (
                        chunk.content if hasattr(chunk, "content") else str(chunk)
                    )
                    chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
                    chunk_metadata = dict(metadata)
                    chunk_metadata["chunk_index"] = getattr(chunk, "chunk_index", i)
                    chunk_metadata["total_chunks"] = getattr(
                        chunk, "total_chunks", len(chunks)
                    )
                    chunk_metadata["parent_memory_id"] = parent_id

                    await self._insert_memory(
                        conn,
                        chunk_content,
                        chunk_hash,
                        embedding,
                        "news_chunk",
                        chunk_metadata,
                        parent_memory_id=parent_id,
                        skip_dedup=True,
                    )

                logger.debug(
                    f"Stored {len(chunks)} chunks for parent {parent_id}"
                )

            return parent_id

    async def _insert_memory(
        self,
        conn,
        content: str,
        content_hash: str,
        embedding: List[float],
        memory_type: str,
        metadata: Dict[str, Any],
        parent_memory_id: Optional[str] = None,
        skip_dedup: bool = False,
    ) -> Optional[str]:
        """Insert a single memory row. Shared by store() and store_chunked()."""
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

        # Extract news-specific fields (NULL for non-news types)
        news_url = metadata.get("news_url")
        news_title = metadata.get("news_title")
        news_published_at = metadata.get("news_published_at") or None
        news_source = metadata.get("news_source")
        price_snapshot = (
            json.dumps(metadata["price_snapshot"])
            if metadata.get("price_snapshot")
            else None
        )

        # Article analysis (from ArticleAnalyzer)
        article_analysis = (
            json.dumps(metadata["article_analysis"])
            if metadata.get("article_analysis")
            else None
        )

        # Signal quality from analysis (seeds the 0.5 default)
        signal_quality = metadata.get("signal_quality")

        # INSERT — use ON CONFLICT for news articles to handle duplicate URLs on restart.
        # idx_memories_news_url is a partial unique index: ON agent_memories(news_url) WHERE news_url IS NOT NULL
        if news_url is not None:
            row_id = await conn.fetchval(
                """
                INSERT INTO agent_memories (
                    memory_type, content, content_hash, embedding,
                    market_tickers, event_tickers, confidence,
                    source_session, trade_id, trade_result, pnl_cents,
                    news_url, news_title, news_published_at, news_source, price_snapshot,
                    parent_memory_id, article_analysis
                ) VALUES (
                    $1, $2, $3, $4::vector,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, $14::timestamptz, $15, $16::jsonb,
                    $17, $18::jsonb
                )
                ON CONFLICT (news_url) WHERE news_url IS NOT NULL DO NOTHING
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
                news_url,
                news_title,
                news_published_at,
                news_source,
                price_snapshot,
                parent_memory_id,
                article_analysis,
            )
            if row_id is None:
                logger.debug(f"Skipped duplicate news_url: {news_url[:80]}")
                return None
        else:
            row_id = await conn.fetchval(
                """
                INSERT INTO agent_memories (
                    memory_type, content, content_hash, embedding,
                    market_tickers, event_tickers, confidence,
                    source_session, trade_id, trade_result, pnl_cents,
                    news_url, news_title, news_published_at, news_source, price_snapshot,
                    parent_memory_id, article_analysis
                ) VALUES (
                    $1, $2, $3, $4::vector,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, $14::timestamptz, $15, $16::jsonb,
                    $17, $18::jsonb
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
                news_url,
                news_title,
                news_published_at,
                news_source,
                price_snapshot,
                parent_memory_id,
                article_analysis,
            )

        logger.debug(f"Stored vector memory {row_id} (type={memory_type})")
        return str(row_id)

    async def search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        event_ticker: Optional[str] = None,
        min_recency_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over agent_memories using search_agent_memories RPC.

        Returns list of dicts with content, memory_type, similarity, etc.
        Sets ef_search=64 for better recall on HNSW index.
        """
        embedding = await self._get_embedding(query)

        async with self._db.get_connection() as conn:
            # Set ef_search for better HNSW recall
            try:
                await conn.execute("SET LOCAL hnsw.ef_search = 64")
            except Exception:
                pass  # Non-critical if SET fails

            rows = await conn.fetch(
                """
                SELECT * FROM search_agent_memories(
                    $1::vector, $2, $3, $4, $5, $6, 0.3
                )
                """,
                str(embedding),
                memory_types,
                None,  # p_market_ticker
                event_ticker,
                min_recency_hours,
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

    async def get_index_health(self) -> Dict[str, Any]:
        """Get HNSW index health metrics for /v3/health endpoint."""
        try:
            async with self._db.get_connection() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(*) FILTER (WHERE is_active) as active_memories,
                        COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embedded,
                        COUNT(*) FILTER (WHERE parent_memory_id IS NOT NULL) as chunks,
                        COUNT(DISTINCT memory_type) as memory_types,
                        pg_size_pretty(pg_total_relation_size('agent_memories')) as table_size
                    FROM agent_memories
                    """
                )
                result = dict(stats) if stats else {}

                # Check index exists and is valid
                idx = await conn.fetchrow(
                    """
                    SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
                    FROM pg_indexes
                    WHERE tablename = 'agent_memories' AND indexname = 'idx_memories_embedding'
                    """
                )
                if idx:
                    result["hnsw_index_size"] = idx["index_size"]
                    result["hnsw_index_exists"] = True
                else:
                    result["hnsw_index_exists"] = False

                return result
        except Exception as e:
            logger.debug(f"get_index_health failed: {e}")
            return {"error": str(e)}
