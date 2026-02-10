"""Two-tier memory: fast in-memory FAISS (session) + pgvector (persistent cross-session).

Session store uses FAISS for <1ms local recall. Persistent store uses existing
VectorMemoryService for cross-session semantic search. pgvector writes are
fire-and-forget via asyncio.create_task().
"""

import asyncio
import collections
import logging
import time
from typing import Any, Dict, List, Optional

from ..models import MemoryEntry, RecallResult

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.session_store")


class SessionMemoryStore:
    """Two-tier memory: FAISS (session) + pgvector (persistent).

    - store(): writes to both FAISS and pgvector (pgvector is fire-and-forget)
    - recall(): hybrid search - FAISS first, pgvector second, merged by similarity
    - journal: in-memory list of trade/insight entries for this session
    """

    # Maximum journal entries to prevent unbounded memory growth in long sessions
    MAX_JOURNAL_ENTRIES = 500
    # Maximum FAISS entries before LRU eviction triggers
    MAX_FAISS_ENTRIES = 2000
    FAISS_EVICTION_KEEP = 1500

    def __init__(self, vector_store=None, embedding_model: str = "text-embedding-3-small"):
        self._vector_store = vector_store  # VectorMemoryService (pgvector)
        self._embedding_model = embedding_model
        self._openai_client = None  # Lazy init

        # FAISS session store
        self._faiss_store = None  # langchain_community FAISS instance
        self._faiss_docs: List[Dict] = []  # Parallel metadata list
        self._faiss_ready = False
        self._faiss_unavailable = False  # Set True if import fails (don't retry)

        # Session journal (bounded deque to prevent unbounded memory growth)
        self._journal: collections.deque = collections.deque(maxlen=self.MAX_JOURNAL_ENTRIES)

        # Pending pgvector write tasks (for flush on shutdown)
        self._pending_writes: List[asyncio.Task] = []
        self._pgvector_failure_count: int = 0

    def _get_embeddings(self):
        """Lazy-init OpenAI embeddings for FAISS."""
        if self._openai_client is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._openai_client = OpenAIEmbeddings(model=self._embedding_model)
            except Exception as e:
                logger.warning(f"OpenAI embeddings init failed: {e}")
        return self._openai_client

    async def store(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store to FAISS (session) + pgvector (persistent, fire-and-forget)."""
        metadata = metadata or {}
        now = time.time()

        # Add to journal
        self._journal.append({
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata,
            "timestamp": now,
        })

        # Store in FAISS (session-local) - skip if import previously failed
        if not self._faiss_unavailable:
            try:
                embeddings = self._get_embeddings()
                if embeddings:
                    if not self._faiss_ready:
                        from langchain_community.vectorstores import FAISS
                        self._faiss_store = await asyncio.to_thread(
                            FAISS.from_texts,
                            [content],
                            embeddings,
                            metadatas=[{"memory_type": memory_type, "timestamp": now, **metadata}],
                        )
                        self._faiss_ready = True
                        self._faiss_docs.append({"content": content, "memory_type": memory_type, "timestamp": now})
                    else:
                        await asyncio.to_thread(
                            self._faiss_store.add_texts,
                            [content],
                            metadatas=[{"memory_type": memory_type, "timestamp": now, **metadata}],
                        )
                        self._faiss_docs.append({"content": content, "memory_type": memory_type, "timestamp": now})

                        # LRU eviction: rebuild FAISS from most recent entries
                        if len(self._faiss_docs) > self.MAX_FAISS_ENTRIES:
                            await self._evict_faiss(embeddings)
            except ImportError:
                self._faiss_unavailable = True
                logger.info("FAISS unavailable (langchain_community not installed), using pgvector only")
            except Exception as e:
                logger.debug(f"FAISS store failed (non-critical): {e}")

        # Fire-and-forget to pgvector (tracked for flush on shutdown)
        if self._vector_store:
            task = asyncio.create_task(self._store_pgvector(content, memory_type, metadata))
            self._pending_writes.append(task)
            # Prune completed tasks to prevent list growth
            self._pending_writes = [t for t in self._pending_writes if not t.done()]

    async def _store_pgvector(self, content: str, memory_type: str, metadata: Dict) -> None:
        """Background pgvector store. Errors are logged, not raised."""
        try:
            await self._vector_store.store(
                content=content,
                memory_type=memory_type,
                metadata=metadata,
            )
            self._pgvector_failure_count = 0
        except Exception as e:
            self._pgvector_failure_count += 1
            if self._pgvector_failure_count <= 3:
                logger.warning(f"pgvector store failed ({self._pgvector_failure_count} consecutive): {e}")
            else:
                logger.debug(f"pgvector store failed ({self._pgvector_failure_count} consecutive): {e}")

    async def recall(
        self, query: str, limit: int = 5, memory_types: Optional[List[str]] = None,
    ) -> RecallResult:
        """Hybrid search: FAISS session + pgvector persistent, merged by similarity.

        Args:
            query: Search query
            limit: Max results
            memory_types: Optional list of memory types to filter by (e.g., ["trade_outcome"])
        """
        results: List[MemoryEntry] = []
        now = time.time()

        # 1. FAISS session search (fast, <1ms)
        if self._faiss_ready and self._faiss_store:
            try:
                embeddings = self._get_embeddings()
                if embeddings:
                    faiss_results = await asyncio.to_thread(
                        self._faiss_store.similarity_search_with_score,
                        query,
                        k=min(limit, len(self._faiss_docs)),
                    )
                    for doc, score in faiss_results:
                        doc_type = doc.metadata.get("memory_type", "")
                        # Apply memory_types filter for FAISS results
                        if memory_types and doc_type not in memory_types:
                            continue
                        # FAISS returns L2 distance; convert to similarity (lower = more similar)
                        similarity = max(0, 1 - score / 2)
                        age_hours = (now - doc.metadata.get("timestamp", now)) / 3600
                        results.append(MemoryEntry(
                            content=doc.page_content,
                            memory_type=doc_type,
                            similarity=round(similarity, 3),
                            age_hours=round(age_hours, 1),
                        ))
            except Exception as e:
                logger.debug(f"FAISS recall failed: {e}")

        # 2. pgvector persistent search (~10ms)
        if self._vector_store:
            try:
                pg_results = await self._vector_store.search(
                    query=query,
                    limit=limit,
                    memory_types=memory_types,
                )
                for r in pg_results:
                    age_hours = None
                    created_at = r.get("created_at")
                    if created_at:
                        try:
                            age_hours = round((now - created_at.timestamp()) / 3600, 1)
                        except (AttributeError, TypeError):
                            pass
                    results.append(MemoryEntry(
                        content=r.get("content", ""),
                        memory_type=r.get("memory_type", ""),
                        similarity=round(r.get("similarity", 0), 3),
                        age_hours=age_hours,
                    ))
            except Exception as e:
                logger.debug(f"pgvector recall failed: {e}")

        # Deduplicate by content prefix (first 100 chars) and sort by similarity
        seen = set()
        unique = []
        for entry in sorted(results, key=lambda e: e.similarity, reverse=True):
            key = entry.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        return RecallResult(
            query=query,
            results=unique[:limit],
            count=len(unique[:limit]),
        )

    async def _evict_faiss(self, embeddings) -> None:
        """Rebuild FAISS index keeping only the most recent entries."""
        keep = self._faiss_docs[-self.FAISS_EVICTION_KEEP:]
        texts = [d["content"] for d in keep]
        metadatas = [{"memory_type": d["memory_type"], "timestamp": d["timestamp"]} for d in keep]

        try:
            from langchain_community.vectorstores import FAISS
            self._faiss_store = await asyncio.to_thread(
                FAISS.from_texts, texts, embeddings, metadatas=metadatas,
            )
            evicted = len(self._faiss_docs) - self.FAISS_EVICTION_KEEP
            self._faiss_docs = keep
            logger.info(f"[MEMORY:FAISS] Evicted {evicted} entries, kept {len(keep)}")
        except Exception as e:
            logger.warning(f"[MEMORY:FAISS] Eviction rebuild failed: {e}")

    async def flush(self, timeout: float = 5.0) -> int:
        """Await all pending pgvector writes. Called on shutdown.

        Returns number of writes that completed successfully.
        """
        pending = [t for t in self._pending_writes if not t.done()]
        if not pending:
            return 0

        logger.info(f"[MEMORY] Flushing {len(pending)} pending pgvector writes...")
        done, not_done = await asyncio.wait(pending, timeout=timeout)

        for task in not_done:
            task.cancel()

        completed = len(done)
        if not_done:
            logger.warning(f"[MEMORY] {len(not_done)} pgvector writes timed out during flush")

        self._pending_writes.clear()
        return completed

    def get_journal(self) -> List[Dict]:
        """Get session journal entries (trades, insights, learnings)."""
        return list(self._journal)

    def journal_summary(self, max_entries: int = 10) -> str:
        """Compact journal summary for prompt injection."""
        if not self._journal:
            return ""
        lines = []
        # deque doesn't support slicing, so convert tail to list
        recent = list(self._journal)[-max_entries:]
        for entry in recent:
            mtype = entry.get("memory_type", "note")
            content = entry.get("content", "")[:100]
            lines.append(f"  [{mtype}] {content}")
        return "SESSION LOG:\n" + "\n".join(lines)
