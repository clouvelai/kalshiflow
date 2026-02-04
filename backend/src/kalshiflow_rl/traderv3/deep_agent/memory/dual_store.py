"""
Dual memory store combining local file storage with pgvector semantic search.

File store (always works, fast) + VectorMemoryService (semantic search, embeddings).
Writes go to both. Searches prefer pgvector, fall back to file keyword search.
"""

import logging
from typing import Any, Dict, List, Optional

from .file_store import FileMemoryStore

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.memory.dual_store")


class DualMemoryStore:
    """
    Unified memory interface: file (local, fast, always works) + pgvector (semantic).

    - store() writes to both (file always succeeds, pgvector best-effort)
    - search() queries pgvector primary, file keyword fallback
    - Validation cache delegates to file store (O(1) lookup)
    """

    def __init__(
        self,
        file_store: FileMemoryStore,
        vector_service: Optional[Any] = None,
    ):
        self._file = file_store
        self._vector = vector_service

    @property
    def file_store(self) -> FileMemoryStore:
        """Direct access to file store (for validation cache, stats, etc.)."""
        return self._file

    async def store(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store to both file and pgvector. File always succeeds."""
        # File store (synchronous, always works)
        file_entry = self._file.append(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )

        # Vector store (async, best-effort)
        vector_id = None
        if self._vector:
            try:
                vector_id = await self._vector.store(
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Vector store failed (file store succeeded): {e}")

        return {
            "status": "stored",
            "type": memory_type,
            "vector_id": vector_id,
            "file_stored": True,
        }

    async def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search pgvector first, fall back to file keyword search."""
        # Try pgvector semantic search
        if self._vector:
            try:
                results = await self._vector.search(
                    query=query,
                    memory_type=memory_type,
                    limit=limit,
                )
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to file: {e}")

        # Fallback: file keyword search
        return self._file.search_journal(query=query, limit=limit)

    def get_validation(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Get cached validation (delegates to file store)."""
        return self._file.get_validation(pair_id)

    def save_validation(self, pair_id: str, data: Dict[str, Any]) -> None:
        """Save validation (delegates to file store)."""
        self._file.save_validation(pair_id, data)

    def get_all_validations(self) -> Dict[str, Dict[str, Any]]:
        """Get all validations (delegates to file store)."""
        return self._file.get_all_validations()

    def clear_validations(self) -> int:
        """Clear all cached validations (delegates to file store)."""
        return self._file.clear_validations()

    def get_stats(self) -> Dict[str, Any]:
        """Combined stats from both stores."""
        stats = self._file.get_stats()
        if self._vector:
            stats["vector_status"] = self._vector.get_status()
        else:
            stats["vector_status"] = {"available": False}
        return stats
