"""
Vector Memory Service - Semantic vector memory backed by Supabase pgvector + OpenAI embeddings.

Provides:
- Dual-write: every memory stored with embedding for semantic recall
- Access tracking: frequently-recalled memories get ranking boost
- Deduplication: SHA-256 exact + cosine 0.88 semantic
- Consolidation: periodic LLM-driven merge of similar memories
- Retention: stale low-value memories soft-deleted over time

Architecture:
- WRITE PATH: embed via OpenAI -> dedup check -> insert into agent_memories
- READ PATH: embed query -> search_agent_memories RPC -> touch accessed IDs
- CONSOLIDATION: cluster similar memories -> LLM merge -> supersede originals
"""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.vector_memory")


class VectorMemoryService:
    """Semantic vector memory backed by Supabase pgvector + OpenAI embeddings."""

    FILENAME_TO_TYPE = {
        "learnings.md": "learning",
        "mistakes.md": "mistake",
        "patterns.md": "pattern",
        "cycle_journal.md": "journal",
        "market_knowledge.md": "market_knowledge",
    }
    # strategy.md and golden_rules.md are NOT stored in vector memory

    # Additional memory types not tied to files (used by trade_executor/tools)
    VALID_TYPES = {
        "learning", "mistake", "pattern", "journal",
        "market_knowledge", "consolidation",
        "signal", "research", "thesis",
    }

    def __init__(self, supabase_getter: Callable, openai_api_key: str):
        self._get_supabase = supabase_getter
        self._openai_client = None  # Lazy init AsyncOpenAI
        self._openai_api_key = openai_api_key
        self._embed_cache: Dict[str, List[float]] = {}  # LRU-ish cache (max 100)

    # ──────────────────────────────────────────────
    # WRITE PATH
    # ──────────────────────────────────────────────

    async def store(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Embed, dedup-check, insert. Returns UUID or None if deduped."""
        if not content or len(content.strip()) < 30:
            return None

        try:
            embedding = await self._embed(content)
        except Exception as e:
            logger.warning(f"[vector_memory] Embedding failed, skipping store: {e}")
            return None

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        try:
            supabase = self._get_supabase()
            if not supabase:
                return None

            # Fast dedup: exact hash match
            existing = (
                supabase.table("agent_memories")
                .select("id")
                .eq("content_hash", content_hash)
                .limit(1)
                .execute()
            )
            if existing.data:
                return None

            # Semantic dedup: cosine > 0.88
            dupes = supabase.rpc(
                "find_similar_memories",
                {
                    "query_embedding": embedding,
                    "p_memory_type": memory_type,
                    "p_threshold": 0.88,
                },
            ).execute()
            if dupes.data:
                logger.info(
                    f"[vector_memory] Deduped: sim={dupes.data[0]['similarity']:.3f}"
                )
                return None

            market_tickers, event_tickers = self._extract_tickers(content)
            confidence = self._extract_confidence(content)

            row = {
                "content": content,
                "content_hash": content_hash,
                "embedding": embedding,
                "memory_type": memory_type,
                "market_tickers": market_tickers,
                "event_tickers": event_tickers,
                "confidence": confidence,
            }
            if metadata:
                # Only include known DB columns from metadata
                allowed_keys = {
                    "source_cycle", "source_session", "source_file",
                    "trade_id", "trade_result", "pnl_cents",
                }
                for k, v in metadata.items():
                    if k in allowed_keys and v is not None:
                        row[k] = v

            result = supabase.table("agent_memories").insert(row).execute()
            mem_id = result.data[0]["id"] if result.data else None
            if mem_id:
                logger.debug(
                    f"[vector_memory] Stored {memory_type}: {content[:60]}... -> {mem_id}"
                )
            return mem_id

        except Exception as e:
            logger.warning(f"[vector_memory] Store failed (non-fatal): {e}")
            return None

    async def store_signal_summary(
        self,
        event_ticker: str,
        summary: str,
        signal_type: str = "signal",
    ) -> Optional[str]:
        """Store a signal/research/thesis summary for semantic recall.

        Lightweight wrapper around store() for non-file-based memory types.
        Used by tools.py to embed extraction signal summaries and GDELT results.
        """
        if signal_type not in self.VALID_TYPES:
            logger.warning(
                "[vector_memory] Invalid signal_type '%s', falling back to 'signal'",
                signal_type,
            )
            signal_type = "signal"
        return await self.store(
            content=summary,
            memory_type=signal_type,
            metadata={"source_session": "signal_store"},
        )

    # ──────────────────────────────────────────────
    # READ PATH
    # ──────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        types: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Semantic search with access-boosted ranking. Returns list of memory dicts."""
        if not query or len(query.strip()) < 5:
            return []

        try:
            embedding = await self._embed(query)
        except Exception as e:
            logger.warning(f"[vector_memory] Embed failed for recall: {e}")
            return []

        try:
            supabase = self._get_supabase()
            if not supabase:
                return []

            results = supabase.rpc(
                "search_agent_memories",
                {
                    "query_embedding": embedding,
                    "p_memory_types": types,
                    "p_market_ticker": ticker,
                    "p_event_ticker": event,
                    "p_limit": limit,
                },
            ).execute()
            memories = results.data or []

            # Fire-and-forget: update access tracking for recalled memories
            if memories:
                recalled_ids = [m["id"] for m in memories]
                asyncio.create_task(self._touch_memories(recalled_ids))

            return memories

        except Exception as e:
            logger.warning(f"[vector_memory] Recall failed (non-fatal): {e}")
            return []

    async def recall_for_context(
        self,
        query: str,
        types: Optional[List[str]] = None,
        char_limit: int = 800,
    ) -> str:
        """Formatted string of recalled memories within char_limit."""
        results = await self.recall(query, types=types, limit=12)
        if not results:
            return ""

        output = []
        remaining = char_limit
        for r in results:
            if remaining <= 0:
                break
            # Show access_count as strength indicator for high-access memories
            strength = (
                f"[x{r['access_count']}] "
                if r.get("access_count", 0) >= 5
                else ""
            )
            tag = (
                f"[{r['confidence']}] "
                if r.get("confidence") not in (None, "medium")
                else ""
            )
            line = f"- {strength}{tag}{r['content'][:300]}"
            output.append(line)
            remaining -= len(line) + 1

        return "\n".join(output)

    async def _touch_memories(self, ids: List[str]) -> None:
        """Batch update access_count and last_accessed_at."""
        try:
            supabase = self._get_supabase()
            if supabase:
                supabase.rpc("touch_memories", {"p_ids": ids}).execute()
        except Exception as e:
            logger.debug(f"[vector_memory] Touch failed (non-fatal): {e}")

    # ──────────────────────────────────────────────
    # CONSOLIDATION
    # ──────────────────────────────────────────────

    async def consolidate(
        self,
        memory_type: str,
        llm_callable: Callable,
    ) -> int:
        """Find clusters of similar memories and LLM-merge them.

        Args:
            memory_type: Type to consolidate (learning, mistake, pattern)
            llm_callable: async fn(prompt) -> str, uses the agent's LLM

        Returns:
            Number of memories consolidated (superseded)
        """
        try:
            supabase = self._get_supabase()
            if not supabase:
                return 0

            # Fetch candidates
            candidates = supabase.rpc(
                "find_consolidation_candidates",
                {
                    "p_memory_type": memory_type,
                    "p_cluster_threshold": 0.80,
                    "p_limit": 50,
                },
            ).execute()

            if not candidates.data or len(candidates.data) < 3:
                return 0

            # Build similarity clusters using greedy approach
            clusters = self._build_clusters(candidates.data, threshold=0.80)

            total_superseded = 0
            for cluster in clusters:
                if len(cluster) < 3:
                    continue

                # LLM-synthesize the cluster into one consolidated memory
                cluster_contents = [c["content"] for c in cluster]
                prompt = self._build_consolidation_prompt(memory_type, cluster_contents)

                try:
                    consolidated_text = await llm_callable(prompt)
                    if not consolidated_text or len(consolidated_text) < 50:
                        continue

                    # Store the consolidated memory with ORIGINAL type so it
                    # appears in type-filtered searches (not "consolidation" type
                    # which would be invisible to recall queries)
                    new_id = await self.store(
                        consolidated_text,
                        memory_type,
                        {
                            "source_session": "consolidation",
                        },
                    )

                    if new_id:
                        # Supersede the originals
                        old_ids = [c["id"] for c in cluster]
                        result = supabase.rpc(
                            "supersede_memories",
                            {"p_old_ids": old_ids, "p_new_id": new_id},
                        ).execute()
                        superseded = (
                            result.data if isinstance(result.data, int) else 0
                        )
                        total_superseded += superseded
                        logger.info(
                            f"[vector_memory] Consolidated {superseded} {memory_type}s "
                            f"into 1 entry ({len(consolidated_text)} chars)"
                        )
                except Exception as e:
                    logger.warning(
                        f"[vector_memory] Consolidation failed for cluster: {e}"
                    )
                    continue

            return total_superseded

        except Exception as e:
            logger.warning(f"[vector_memory] Consolidation error: {e}")
            return 0

    def _build_clusters(
        self,
        candidates: List[Dict],
        threshold: float = 0.80,
    ) -> List[List[Dict]]:
        """Greedy clustering: group memories by pairwise cosine similarity."""
        try:
            import numpy as np
        except ImportError:
            logger.warning("[vector_memory] numpy not available for clustering")
            return []

        n = len(candidates)
        if n == 0:
            return []

        embeddings = []
        for c in candidates:
            emb = c.get("embedding")
            if emb and isinstance(emb, list):
                embeddings.append(np.array(emb, dtype=np.float32))
            else:
                embeddings.append(None)

        assigned = set()
        clusters = []

        for i in range(n):
            if i in assigned or embeddings[i] is None:
                continue
            cluster = [candidates[i]]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned or embeddings[j] is None:
                    continue
                # Cosine similarity
                dot = np.dot(embeddings[i], embeddings[j])
                norm = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                sim = dot / (norm + 1e-8)
                if sim >= threshold:
                    cluster.append(candidates[j])
                    assigned.add(j)

            if len(cluster) >= 3:
                clusters.append(cluster)

        return clusters

    def _build_consolidation_prompt(
        self,
        memory_type: str,
        contents: List[str],
    ) -> str:
        """Build LLM prompt to merge a cluster of similar memories."""
        type_labels = {
            "learning": "learnings/observations",
            "mistake": "mistakes/failures",
            "pattern": "patterns/strategies",
            "journal": "journal entries",
        }
        label = type_labels.get(memory_type, "memories")

        entries_text = "\n---\n".join(
            f"Entry {i + 1}:\n{c}" for i, c in enumerate(contents)
        )

        return (
            f"You are consolidating {len(contents)} related {label} into ONE concise, "
            f"high-quality entry.\n\n"
            f"These memories overlap significantly. Merge them into a single entry that:\n"
            f"1. Preserves ALL unique facts, tickers, and specific observations\n"
            f"2. Removes redundancy and repetition\n"
            f"3. Keeps the most specific/actionable version when entries conflict\n"
            f"4. Includes confidence markers [high]/[low] where appropriate\n"
            f"5. Maintains any trade results (win/loss/pnl) data\n\n"
            f"Related {label}:\n{entries_text}\n\n"
            f"Write the consolidated entry (markdown format, ## header, concise):"
        )

    # ──────────────────────────────────────────────
    # RETENTION POLICY
    # ──────────────────────────────────────────────

    async def enforce_retention(self) -> Dict[str, int]:
        """Soft-delete stale, low-value memories per retention policy."""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return {}

            result = supabase.rpc("enforce_retention_policy").execute()
            summary = {}
            for row in result.data or []:
                summary[row["memory_type"]] = row["deleted_count"]
            if summary:
                logger.info(f"[vector_memory] Retention cleanup: {summary}")
            return summary
        except Exception as e:
            logger.warning(f"[vector_memory] Retention enforcement failed: {e}")
            return {}

    # ──────────────────────────────────────────────
    # EMBEDDING
    # ──────────────────────────────────────────────

    async def _embed(self, text: str) -> List[float]:
        """Get embedding for text, with simple cache."""
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        if not self._openai_client:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=self._openai_api_key)

        resp = await self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        embedding = resp.data[0].embedding

        # Simple bounded cache
        if len(self._embed_cache) > 100:
            oldest = next(iter(self._embed_cache))
            del self._embed_cache[oldest]
        self._embed_cache[cache_key] = embedding

        return embedding

    # ──────────────────────────────────────────────
    # BACKFILL
    # ──────────────────────────────────────────────

    async def backfill_from_files(
        self,
        memory_dir: Path,
        session_id: str,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        """One-time migration of existing flat-file memories into vector store.

        For journal entries, if llm_callable is provided, extracts key facts
        before embedding for higher-quality embeddings.
        """
        try:
            supabase = self._get_supabase()
            if not supabase:
                return

            count = (
                supabase.table("agent_memories")
                .select("id", count="exact")
                .limit(1)
                .execute()
            )
            if count.count and count.count > 0:
                logger.info(
                    f"[vector_memory] Already have {count.count} memories, skipping backfill"
                )
                return
        except Exception as e:
            logger.warning(f"[vector_memory] Backfill check failed: {e}")
            return

        logger.info("[vector_memory] Starting backfill from flat files...")
        stored = 0

        for filename, memory_type in self.FILENAME_TO_TYPE.items():
            filepath = memory_dir / filename
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8")
            entries = re.split(r"\n(?=## )", content)
            for entry in entries:
                entry = entry.strip()
                if len(entry) < 50:
                    continue

                store_content = entry
                # For journal entries, LLM-extract key observations for better embeddings
                if (
                    memory_type == "journal"
                    and llm_callable
                    and len(entry) > 200
                ):
                    try:
                        extracted = await llm_callable(
                            "Extract the 2-3 key observations, decisions, and outcomes "
                            f"from this trading journal entry. Be concise:\n\n{entry[:2000]}"
                        )
                        if extracted and len(extracted) > 30:
                            store_content = (
                                f"{extracted}\n\n[source: {entry[:200]}...]"
                            )
                    except Exception:
                        pass  # Use raw entry

                result = await self.store(
                    store_content,
                    memory_type,
                    {"source_file": filename, "source_session": session_id},
                )
                if result:
                    stored += 1

        # Also backfill archives
        archive_dir = memory_dir / "memory_archive"
        if archive_dir.exists():
            for session_dir in sorted(archive_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                for filepath in session_dir.iterdir():
                    if filepath.suffix != ".md":
                        continue
                    filename_with_ext = filepath.name
                    if filename_with_ext not in self.FILENAME_TO_TYPE:
                        continue
                    memory_type = self.FILENAME_TO_TYPE[filename_with_ext]
                    content = filepath.read_text(encoding="utf-8")
                    entries = re.split(r"\n(?=## )", content)
                    for entry in entries:
                        entry = entry.strip()
                        if len(entry) > 50:
                            result = await self.store(
                                entry,
                                memory_type,
                                {
                                    "source_file": filename_with_ext,
                                    "source_session": session_dir.name,
                                },
                            )
                            if result:
                                stored += 1

        logger.info(f"[vector_memory] Backfill complete: {stored} memories stored")

    # ──────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return memory store statistics."""
        try:
            supabase = self._get_supabase()
            if not supabase:
                return {"active_memories": "unavailable"}

            result = (
                supabase.table("agent_memories")
                .select("id", count="exact")
                .eq("is_active", True)
                .execute()
            )
            return {"active_memories": result.count or 0}
        except Exception:
            return {"active_memories": "unavailable"}

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _extract_tickers(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract market and event tickers from content."""
        # Market tickers: full form with date/strike suffixes
        # e.g. KXBONDIOUT-25FEB07-T42.5, KXGOVTFUND-26FEB02
        market_tickers = list(
            set(re.findall(r"\b(KX[A-Z]{2,}-[A-Z0-9][A-Z0-9._-]*)\b", content))
        )
        # Event tickers: base form without suffixes e.g. KXBONDIOUT, KXGOVTFUND
        event_tickers = list(
            set(re.findall(r"\b(KX[A-Z]{4,})\b", content))
        )
        return market_tickers[:10], event_tickers[:5]

    def _extract_confidence(self, content: str) -> str:
        """Extract confidence level from content markers."""
        lower = content.lower()
        if "[high]" in lower:
            return "high"
        if "[low]" in lower:
            return "low"
        return "medium"
