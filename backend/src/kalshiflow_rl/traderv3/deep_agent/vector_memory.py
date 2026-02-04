"""
Vector memory service using pgvector for semantic search.

Stores agent learnings, mistakes, and insights in the agent_memories table
with embedding vectors for semantic retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.vector_memory")


class VectorMemoryService:
    """
    Semantic memory using Supabase pgvector.

    Stores memories with embeddings for semantic retrieval.
    Uses the agent_memories table created by migration 20260202100000.

    Gracefully degrades when schema is missing (table/function not created yet).
    After first failure, disables DB operations to avoid log spam.
    """

    def __init__(self, supabase_client: Any = None, embedding_model: Any = None, agent_id: str = "arb_agent"):
        self._supabase = supabase_client
        self._embedding_model = embedding_model
        self._agent_id = agent_id
        self._store_count = 0
        self._search_count = 0
        self._db_available = True  # flipped to False after first schema error

    async def store(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a memory with optional embedding vector. Returns memory ID or None."""
        if not self._supabase or not self._db_available:
            return None

        try:
            embedding = None
            if self._embedding_model:
                embedding = await self._embedding_model.aembed_query(content)

            row = {
                "agent_id": self._agent_id,
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or {},
            }

            if embedding:
                row["embedding"] = embedding

            result = self._supabase.table("agent_memories").insert(row).execute()
            self._store_count += 1

            memory_id = result.data[0]["id"] if result.data else None
            logger.debug(f"Stored memory [{memory_type}]: {content[:80]}...")
            return memory_id

        except Exception as e:
            err_str = str(e)
            if "could not find" in err_str.lower() or "column" in err_str.lower() or "relation" in err_str.lower():
                logger.info("Vector memory schema not available - disabling DB writes (file store still active)")
                self._db_available = False
            else:
                logger.warning(f"Failed to store memory: {e}")
            return None

    async def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity. Falls back to recency if no embeddings."""
        if not self._supabase or not self._db_available:
            return []

        self._search_count += 1

        try:
            if self._embedding_model:
                query_embedding = await self._embedding_model.aembed_query(query)

                params = {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "p_agent_id": self._agent_id,
                }
                if memory_type:
                    params["p_memory_type"] = memory_type

                result = self._supabase.rpc("search_agent_memories", params).execute()
                return result.data or []

            query_builder = self._supabase.table("agent_memories").select("*").eq(
                "agent_id", self._agent_id
            ).order("created_at", desc=True).limit(limit)

            if memory_type:
                query_builder = query_builder.eq("memory_type", memory_type)

            result = query_builder.execute()
            return result.data or []

        except Exception as e:
            err_str = str(e)
            if "could not find" in err_str.lower() or "column" in err_str.lower() or "relation" in err_str.lower():
                logger.info("Vector memory schema not available - disabling DB reads (file store still active)")
                self._db_available = False
            else:
                logger.warning(f"Memory search failed: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get memory service status."""
        return {
            "agent_id": self._agent_id,
            "store_count": self._store_count,
            "search_count": self._search_count,
            "has_embeddings": self._embedding_model is not None,
            "has_db": self._supabase is not None,
            "db_available": self._db_available,
        }
