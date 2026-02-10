"""Unit tests for SessionMemoryStore.

T2 tests — async, mocked embeddings and pgvector. No network calls.
Tests store/recall round-trip, journal, and two-tier deduplication.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.memory.session_store import SessionMemoryStore
from kalshiflow_rl.traderv3.single_arb.models import MemoryEntry, RecallResult


# ===========================================================================
# TestJournal
# ===========================================================================


class TestJournal:
    @pytest.mark.asyncio
    async def test_journal_empty(self):
        store = SessionMemoryStore()
        assert store.get_journal() == []

    @pytest.mark.asyncio
    async def test_journal_appends(self):
        store = SessionMemoryStore()
        # store() without embeddings will still append to journal
        await store.store("test entry 1", memory_type="learning")
        await store.store("test entry 2", memory_type="trade")
        journal = store.get_journal()
        assert len(journal) == 2
        assert journal[0]["content"] == "test entry 1"
        assert journal[1]["memory_type"] == "trade"

    @pytest.mark.asyncio
    async def test_journal_has_timestamp(self):
        store = SessionMemoryStore()
        await store.store("test", memory_type="note")
        journal = store.get_journal()
        assert "timestamp" in journal[0]
        assert journal[0]["timestamp"] > 0

    def test_journal_summary_empty(self):
        store = SessionMemoryStore()
        assert store.journal_summary() == ""

    @pytest.mark.asyncio
    async def test_journal_summary_format(self):
        store = SessionMemoryStore()
        await store.store("learned about VPIN", memory_type="learning")
        await store.store("placed order MKT-A", memory_type="trade")
        summary = store.journal_summary()
        assert "SESSION LOG:" in summary
        assert "[learning]" in summary
        assert "[trade]" in summary

    @pytest.mark.asyncio
    async def test_journal_summary_max_entries(self):
        store = SessionMemoryStore()
        for i in range(20):
            await store.store(f"entry {i}", memory_type="note")
        summary = store.journal_summary(max_entries=5)
        # Should only have 5 entries (the last 5)
        lines = [l for l in summary.split("\n") if l.strip().startswith("[")]
        assert len(lines) == 5


# ===========================================================================
# TestPgvectorFireAndForget
# ===========================================================================


class TestPgvectorIntegration:
    @pytest.mark.asyncio
    async def test_pgvector_called_on_store(self):
        mock_vs = MagicMock()
        mock_vs.store = AsyncMock()
        store = SessionMemoryStore(vector_store=mock_vs)
        await store.store("test content", memory_type="learning")
        # Give fire-and-forget task time to execute
        await asyncio.sleep(0.1)
        mock_vs.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_pgvector_error_non_fatal(self):
        mock_vs = MagicMock()
        mock_vs.store = AsyncMock(side_effect=Exception("DB error"))
        store = SessionMemoryStore(vector_store=mock_vs)
        # Should not raise
        await store.store("test content", memory_type="learning")
        await asyncio.sleep(0.1)
        # Journal should still have the entry
        assert len(store.get_journal()) == 1


# ===========================================================================
# TestRecall
# ===========================================================================


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_empty_store(self):
        store = SessionMemoryStore()
        result = await store.recall("anything")
        assert isinstance(result, RecallResult)
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_recall_with_pgvector(self):
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {"content": "prior learning about VPIN", "memory_type": "learning",
             "similarity": 0.85, "created_at": None},
        ])
        store = SessionMemoryStore(vector_store=mock_vs)
        result = await store.recall("VPIN toxicity")
        assert result.count == 1
        assert result.results[0].content == "prior learning about VPIN"
        assert result.results[0].similarity == 0.85

    @pytest.mark.asyncio
    async def test_recall_deduplicates(self):
        mock_vs = MagicMock()
        # Return duplicate content from pgvector
        mock_vs.search = AsyncMock(return_value=[
            {"content": "same content here", "memory_type": "learning", "similarity": 0.9, "created_at": None},
            {"content": "same content here", "memory_type": "learning", "similarity": 0.85, "created_at": None},
        ])
        store = SessionMemoryStore(vector_store=mock_vs)
        result = await store.recall("test")
        # Dedup by first 100 chars
        assert result.count == 1

    @pytest.mark.asyncio
    async def test_recall_pgvector_error_non_fatal(self):
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(side_effect=Exception("DB error"))
        store = SessionMemoryStore(vector_store=mock_vs)
        # Should not raise, returns empty
        result = await store.recall("test")
        assert result.count == 0


# ===========================================================================
# TestMetadata
# ===========================================================================


class TestMetadata:
    @pytest.mark.asyncio
    async def test_metadata_passed_to_journal(self):
        store = SessionMemoryStore()
        await store.store("test", memory_type="trade", metadata={"ticker": "MKT-A"})
        journal = store.get_journal()
        assert journal[0]["metadata"]["ticker"] == "MKT-A"

    @pytest.mark.asyncio
    async def test_metadata_passed_to_pgvector(self):
        mock_vs = MagicMock()
        mock_vs.store = AsyncMock()
        store = SessionMemoryStore(vector_store=mock_vs)
        await store.store("test", memory_type="trade", metadata={"ticker": "MKT-A"})
        await asyncio.sleep(0.1)
        call_args = mock_vs.store.call_args
        assert call_args.kwargs.get("metadata", {}).get("ticker") == "MKT-A"


# ===========================================================================
# TestJournalBound
# ===========================================================================


class TestJournalBound:
    @pytest.mark.asyncio
    async def test_journal_respects_max_size(self):
        """Journal should evict oldest entries when max is exceeded."""
        store = SessionMemoryStore()
        # Store more than MAX_JOURNAL_ENTRIES
        for i in range(store.MAX_JOURNAL_ENTRIES + 50):
            await store.store(f"entry {i}", memory_type="note")
        journal = store.get_journal()
        assert len(journal) == store.MAX_JOURNAL_ENTRIES
        # Oldest should be evicted, newest should be present
        assert journal[-1]["content"] == f"entry {store.MAX_JOURNAL_ENTRIES + 49}"
        assert journal[0]["content"] == f"entry 50"

    @pytest.mark.asyncio
    async def test_journal_returns_list_not_deque(self):
        """get_journal() should return a plain list for JSON serialization."""
        store = SessionMemoryStore()
        await store.store("test", memory_type="note")
        journal = store.get_journal()
        assert isinstance(journal, list)


# ===========================================================================
# TestRecallLimit
# ===========================================================================


class TestRecallLimit:
    @pytest.mark.asyncio
    async def test_recall_respects_limit(self):
        """Recall should return at most `limit` results."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {"content": f"result {i}", "memory_type": "learning",
             "similarity": 0.9 - i * 0.05, "created_at": None}
            for i in range(10)
        ])
        store = SessionMemoryStore(vector_store=mock_vs)
        result = await store.recall("test", limit=3)
        assert result.count <= 3

    @pytest.mark.asyncio
    async def test_recall_results_sorted_by_similarity(self):
        """Results should be sorted highest similarity first."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {"content": "low sim", "memory_type": "learning", "similarity": 0.5, "created_at": None},
            {"content": "high sim", "memory_type": "learning", "similarity": 0.95, "created_at": None},
            {"content": "mid sim", "memory_type": "learning", "similarity": 0.7, "created_at": None},
        ])
        store = SessionMemoryStore(vector_store=mock_vs)
        result = await store.recall("test", limit=10)
        similarities = [r.similarity for r in result.results]
        assert similarities == sorted(similarities, reverse=True)


# ===========================================================================
# TestHybridSearch
# ===========================================================================


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_pgvector_only_no_faiss(self):
        """When FAISS is empty, results come only from pgvector."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {"content": "from pgvector", "memory_type": "learning",
             "similarity": 0.8, "created_at": None},
        ])
        store = SessionMemoryStore(vector_store=mock_vs)
        # No FAISS data stored, so only pgvector results
        result = await store.recall("test")
        assert result.count == 1
        assert result.results[0].content == "from pgvector"

    @pytest.mark.asyncio
    async def test_pgvector_down_returns_empty(self):
        """When both FAISS and pgvector fail, returns empty cleanly."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(side_effect=Exception("connection refused"))
        store = SessionMemoryStore(vector_store=mock_vs)
        result = await store.recall("test")
        assert isinstance(result, RecallResult)
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_no_vector_store_recall(self):
        """When no vector store at all, recall returns empty."""
        store = SessionMemoryStore(vector_store=None)
        result = await store.recall("anything")
        assert result.count == 0


# ===========================================================================
# TestConcurrency
# ===========================================================================


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_stores(self):
        """Multiple concurrent stores should all land in journal."""
        store = SessionMemoryStore()
        tasks = [store.store(f"concurrent-{i}", memory_type="note") for i in range(20)]
        await asyncio.gather(*tasks)
        journal = store.get_journal()
        assert len(journal) == 20
        contents = {e["content"] for e in journal}
        assert all(f"concurrent-{i}" in contents for i in range(20))

    @pytest.mark.asyncio
    async def test_concurrent_store_and_recall(self):
        """Store and recall can run concurrently without error."""
        mock_vs = MagicMock()
        mock_vs.store = AsyncMock()
        mock_vs.search = AsyncMock(return_value=[])
        store = SessionMemoryStore(vector_store=mock_vs)

        async def store_loop():
            for i in range(10):
                await store.store(f"entry-{i}", memory_type="note")

        async def recall_loop():
            for _ in range(10):
                await store.recall("test")

        # Should not raise
        await asyncio.gather(store_loop(), recall_loop())
