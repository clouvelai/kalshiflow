"""
DualMemoryStore - combines FileMemoryStore (always available) with
VectorMemoryService (best-effort semantic search).

Hot-path writes go to FileMemoryStore synchronously.
Vector store writes are fire-and-forget async tasks.
Searches merge keyword (file) + semantic (vector) results.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.dual_store")


class DualMemoryStore:
    """
    Unified memory interface: file store (fast, local) + vector store (semantic).

    File store is always the source of truth. Vector store enriches search
    with semantic similarity but never blocks the hot path.
    """

    def __init__(self, file_store, vector_store=None):
        """
        Args:
            file_store: FileMemoryStore instance (required, always available)
            vector_store: VectorMemoryService instance (optional, best-effort)
        """
        self._file_store = file_store
        self._vector_store = vector_store

    @property
    def file_store(self):
        """Direct access to the FileMemoryStore."""
        return self._file_store

    @property
    def vector_store(self):
        """Direct access to the VectorMemoryService (may be None)."""
        return self._vector_store

    def append(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory. File store is synchronous (always succeeds).
        Vector store is fire-and-forget async.

        Returns the file store entry dict.
        """
        entry = self._file_store.append(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )

        if self._vector_store:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._vector_store_bg(content, memory_type, metadata)
                )
            except RuntimeError:
                # No running event loop (shouldn't happen in async context)
                pass

        return entry

    async def _vector_store_bg(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Background vector store write. Never propagates exceptions."""
        try:
            await self._vector_store.store(
                content=content,
                memory_type=memory_type,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Vector store write failed (non-critical): {e}")

    def search_journal(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Keyword search on file store only (synchronous)."""
        return self._file_store.search_journal(query=query, limit=limit)

    async def search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        event_ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: semantic (vector) + keyword (file), merged and deduped.

        Semantic results come first, then keyword-only results.
        """
        # Keyword results from file store
        keyword_results = self._file_store.search_journal(query=query, limit=limit)

        # Semantic results from vector store
        semantic_results = []
        if self._vector_store:
            try:
                semantic_results = await self._vector_store.search(
                    query=query,
                    limit=limit,
                    memory_types=memory_types,
                    event_ticker=event_ticker,
                )
            except Exception as e:
                logger.warning(f"Vector search failed, using keyword-only: {e}")

        if not semantic_results:
            return keyword_results[:limit]

        # Merge: semantic first, then keyword-only (deduped by content)
        seen_content = set()
        merged = []

        for r in semantic_results:
            content = r.get("content", "")
            if content not in seen_content:
                seen_content.add(content)
                r["_source"] = "vector"
                merged.append(r)

        for r in keyword_results:
            content = r.get("content", "")
            if content not in seen_content:
                seen_content.add(content)
                r["_source"] = "keyword"
                merged.append(r)

        return merged[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        stats = self._file_store.get_stats()
        stats["vector_available"] = self._vector_store is not None
        return stats

    # --- Passthrough methods to file store ---

    def get_journal(self, limit: int = 50, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._file_store.get_journal(limit=limit, memory_type=memory_type)

    def get_validation(self, pair_id: str) -> Optional[Dict[str, Any]]:
        return self._file_store.get_validation(pair_id)

    def save_validation(self, pair_id: str, data: Dict[str, Any]) -> None:
        self._file_store.save_validation(pair_id, data)

    def get_all_validations(self) -> Dict[str, Dict[str, Any]]:
        return self._file_store.get_all_validations()

    def clear_validations(self) -> int:
        return self._file_store.clear_validations()

    def get_session(self) -> Dict[str, Any]:
        return self._file_store.get_session()

    def save_session(self, data: Dict[str, Any]) -> None:
        self._file_store.save_session(data)
