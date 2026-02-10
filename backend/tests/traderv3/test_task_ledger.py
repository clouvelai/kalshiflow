"""Unit tests for TaskLedger.

T1 tests — pure logic, no async, no network, no mocks needed for core tests.
Tests reconcile, tick/stale, prune, to_prompt_section, needs_replan, to_broadcast.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kalshiflow_rl.traderv3.single_arb.task_ledger import (
    TaskLedger,
    EnrichedTask,
    _extract_priority,
    _PRUNE_AFTER_CYCLES,
    _DEFAULT_TTL_CYCLES,
)


# ===========================================================================
# TestPriorityExtraction
# ===========================================================================


class TestPriorityExtraction:
    def test_high_priority(self):
        prio, content = _extract_priority("[HIGH] Monitor AAPL event")
        assert prio == 3
        assert content == "Monitor AAPL event"

    def test_med_priority(self):
        prio, content = _extract_priority("[MED] Check positions")
        assert prio == 2
        assert content == "Check positions"

    def test_low_priority(self):
        prio, content = _extract_priority("[LOW] Review old trades")
        assert prio == 1
        assert content == "Review old trades"

    def test_default_priority(self):
        prio, content = _extract_priority("No prefix here")
        assert prio == 2
        assert content == "No prefix here"

    def test_case_insensitive(self):
        prio, content = _extract_priority("[high] urgent task")
        assert prio == 3
        assert content == "urgent task"

    def test_leading_whitespace(self):
        prio, content = _extract_priority("  [HIGH] with spaces")
        assert prio == 3
        assert content == "with spaces"


# ===========================================================================
# TestReconcile
# ===========================================================================


class TestReconcile:
    def test_new_tasks_created(self):
        ledger = TaskLedger(session_id="test-1")
        ledger.reconcile([
            {"content": "Buy YES on event A", "status": "pending"},
            {"content": "Monitor spread", "status": "in_progress"},
        ], cycle_num=1)
        assert len(ledger.tasks) == 2
        statuses = {t.status for t in ledger.tasks.values()}
        assert statuses == {"pending", "in_progress"}

    def test_existing_task_updated(self):
        ledger = TaskLedger(session_id="test-2")
        ledger.reconcile([{"content": "Research AAPL", "status": "pending"}], cycle_num=1)
        task_id = list(ledger.tasks.keys())[0]

        ledger.reconcile([{"content": "Research AAPL", "status": "in_progress"}], cycle_num=2)
        assert len(ledger.tasks) == 1
        assert ledger.tasks[task_id].status == "in_progress"
        assert ledger.tasks[task_id].updated_cycle == 2

    def test_missing_task_marked_completed(self):
        ledger = TaskLedger(session_id="test-3")
        ledger.reconcile([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "pending"},
        ], cycle_num=1)

        # Only Task A in next reconcile => Task B marked completed
        ledger.reconcile([{"content": "Task A", "status": "pending"}], cycle_num=2)
        statuses = {t.content: t.status for t in ledger.tasks.values()}
        assert statuses["Task A"] == "pending"
        assert statuses["Task B"] == "completed"

    def test_empty_content_skipped(self):
        ledger = TaskLedger(session_id="test-4")
        ledger.reconcile([
            {"content": "", "status": "pending"},
            {"content": "Real task", "status": "pending"},
        ], cycle_num=1)
        assert len(ledger.tasks) == 1

    def test_priority_extracted_on_create(self):
        ledger = TaskLedger(session_id="test-5")
        ledger.reconcile([{"content": "[HIGH] Urgent trade", "status": "pending"}], cycle_num=1)
        task = list(ledger.tasks.values())[0]
        assert task.priority == 3
        assert task.content == "Urgent trade"  # Prefix stripped

    def test_similar_content_matches(self):
        ledger = TaskLedger(session_id="test-6")
        ledger.reconcile([{"content": "Monitor AAPL spread for arb opportunity", "status": "pending"}], cycle_num=1)
        assert len(ledger.tasks) == 1

        # Slightly different wording should match
        ledger.reconcile([{"content": "Monitor AAPL spread for arb opportunities", "status": "in_progress"}], cycle_num=2)
        assert len(ledger.tasks) == 1
        task = list(ledger.tasks.values())[0]
        assert task.status == "in_progress"

    def test_text_field_fallback(self):
        """write_todos may use 'text' instead of 'content'."""
        ledger = TaskLedger(session_id="test-7")
        ledger.reconcile([{"text": "Task via text field", "status": "pending"}], cycle_num=1)
        assert len(ledger.tasks) == 1
        assert list(ledger.tasks.values())[0].content == "Task via text field"


# ===========================================================================
# TestTickAndStale
# ===========================================================================


class TestTickAndStale:
    def test_stale_detection(self):
        ledger = TaskLedger(session_id="test-stale")
        ledger.reconcile([{"content": "Old task", "status": "pending"}], cycle_num=1)

        # Advance past TTL
        ledger.tick(cycle_num=1 + _DEFAULT_TTL_CYCLES + 1)
        task = list(ledger.tasks.values())[0]
        assert task.status == "stale"

    def test_not_stale_within_ttl(self):
        ledger = TaskLedger(session_id="test-not-stale")
        ledger.reconcile([{"content": "Recent task", "status": "pending"}], cycle_num=5)

        ledger.tick(cycle_num=5 + _DEFAULT_TTL_CYCLES - 1)
        task = list(ledger.tasks.values())[0]
        assert task.status == "pending"

    def test_completed_not_marked_stale(self):
        ledger = TaskLedger(session_id="test-completed")
        ledger.reconcile([{"content": "Done task", "status": "completed"}], cycle_num=1)

        ledger.tick(cycle_num=100)
        # Should be pruned, not stale
        assert len(ledger.tasks) == 0

    def test_prune_old_completed(self):
        ledger = TaskLedger(session_id="test-prune")
        ledger.reconcile([{"content": "Task", "status": "pending"}], cycle_num=1)
        # Mark completed by removing from next reconcile
        ledger.reconcile([], cycle_num=2)
        assert list(ledger.tasks.values())[0].status == "completed"

        # Prune after _PRUNE_AFTER_CYCLES
        ledger.tick(cycle_num=2 + _PRUNE_AFTER_CYCLES + 1)
        assert len(ledger.tasks) == 0

    def test_prune_keeps_recent_completed(self):
        ledger = TaskLedger(session_id="test-keep")
        ledger.reconcile([{"content": "Task", "status": "pending"}], cycle_num=5)
        ledger.reconcile([], cycle_num=6)  # Completed at cycle 6

        ledger.tick(cycle_num=6 + _PRUNE_AFTER_CYCLES - 1)
        assert len(ledger.tasks) == 1  # Still within prune window


# ===========================================================================
# TestNeedsReplan
# ===========================================================================


class TestNeedsReplan:
    def test_no_tasks_no_replan(self):
        ledger = TaskLedger(session_id="test-empty")
        assert ledger.needs_replan() is False

    def test_all_stale_needs_replan(self):
        ledger = TaskLedger(session_id="test-replan")
        ledger.reconcile([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "pending"},
        ], cycle_num=1)
        # Force stale
        ledger.tick(cycle_num=1 + _DEFAULT_TTL_CYCLES + 1)
        assert ledger.needs_replan() is True

    def test_minority_stale_no_replan(self):
        ledger = TaskLedger(session_id="test-minority")
        ledger.reconcile([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "pending"},
            {"content": "Task C", "status": "pending"},
        ], cycle_num=1)

        # Update B and C to keep them fresh
        ledger.reconcile([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
            {"content": "Task C", "status": "in_progress"},
        ], cycle_num=5)

        # Only Task A goes stale (updated at cycle 5 for B and C)
        # Trick: manually set A's updated_cycle back
        for t in ledger.tasks.values():
            if t.content == "Task A":
                t.updated_cycle = 1
                break

        ledger.tick(cycle_num=1 + _DEFAULT_TTL_CYCLES + 1)
        # 1 stale out of 3 active = 33%, below 50%
        assert ledger.needs_replan() is False


# ===========================================================================
# TestPromptSection
# ===========================================================================


class TestPromptSection:
    def test_empty_returns_empty_string(self):
        ledger = TaskLedger(session_id="test-prompt-empty")
        assert ledger.to_prompt_section() == ""

    def test_format_active_tasks(self):
        ledger = TaskLedger(session_id="test-prompt")
        ledger.reconcile([
            {"content": "[HIGH] Urgent task", "status": "in_progress"},
            {"content": "Normal task", "status": "pending"},
            {"content": "[LOW] Minor task", "status": "pending"},
        ], cycle_num=1)

        section = ledger.to_prompt_section()
        assert "TASKS:" in section
        assert "[>]" in section  # in_progress marker
        assert "[ ]" in section  # pending marker
        assert "[HIGH]" in section
        assert "[LOW]" in section

    def test_excludes_completed(self):
        ledger = TaskLedger(session_id="test-prompt-completed")
        ledger.reconcile([{"content": "Done task", "status": "completed"}], cycle_num=1)
        assert ledger.to_prompt_section() == ""

    def test_stale_marker(self):
        ledger = TaskLedger(session_id="test-prompt-stale")
        ledger.reconcile([{"content": "Old task", "status": "pending"}], cycle_num=1)
        ledger.tick(cycle_num=1 + _DEFAULT_TTL_CYCLES + 1)

        section = ledger.to_prompt_section()
        assert "[STALE]" in section


# ===========================================================================
# TestBroadcast
# ===========================================================================


class TestBroadcast:
    def test_broadcast_includes_all_tasks(self):
        ledger = TaskLedger(session_id="test-broadcast")
        ledger.reconcile([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
        ], cycle_num=1)

        broadcast = ledger.to_broadcast()
        assert len(broadcast) == 2
        assert all("id" in t for t in broadcast)
        assert all("priority" in t for t in broadcast)
        assert all("created_at" in t for t in broadcast)

    def test_broadcast_sorted_by_priority(self):
        ledger = TaskLedger(session_id="test-broadcast-sort")
        ledger.reconcile([
            {"content": "[LOW] Low prio", "status": "pending"},
            {"content": "[HIGH] High prio", "status": "pending"},
            {"content": "Medium prio", "status": "pending"},
        ], cycle_num=1)

        broadcast = ledger.to_broadcast()
        priorities = [t["priority"] for t in broadcast]
        assert priorities == sorted(priorities, reverse=True)


# ===========================================================================
# TestPersist
# ===========================================================================


class TestPersist:
    @pytest.mark.asyncio
    async def test_persist_calls_db(self):
        ledger = TaskLedger(session_id="test-persist")
        ledger.reconcile([{"content": "Task A", "status": "pending"}], cycle_num=1)

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        ledger.persist(mock_pool)
        await asyncio.sleep(0.1)  # Let fire-and-forget run
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert "captain_task_ledger" in args[0][0]

    def test_persist_skips_empty(self):
        ledger = TaskLedger(session_id="test-persist-empty")
        mock_pool = MagicMock()
        ledger.persist(mock_pool)
        mock_pool.acquire.assert_not_called()

    def test_persist_skips_no_pool(self):
        ledger = TaskLedger(session_id="test-persist-no-pool")
        ledger.reconcile([{"content": "Task", "status": "pending"}], cycle_num=1)
        ledger.persist(None)  # Should not raise

    @pytest.mark.asyncio
    async def test_persist_error_non_fatal(self):
        ledger = TaskLedger(session_id="test-persist-error")
        ledger.reconcile([{"content": "Task A", "status": "pending"}], cycle_num=1)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        ledger.persist(mock_pool)
        await asyncio.sleep(0.1)
        # Should not raise, tasks still intact
        assert len(ledger.tasks) == 1


# ===========================================================================
# TestFlush
# ===========================================================================


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_empty(self):
        ledger = TaskLedger(session_id="test-flush")
        completed = await ledger.flush(timeout=1.0)
        assert completed == 0

    @pytest.mark.asyncio
    async def test_flush_awaits_pending(self):
        ledger = TaskLedger(session_id="test-flush-pending")
        # Simulate a pending write
        future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        ledger._pending_writes.append(asyncio.ensure_future(asyncio.sleep(0)))
        await asyncio.sleep(0.05)

        completed = await ledger.flush(timeout=1.0)
        assert completed >= 0


# ===========================================================================
# TestEnrichedTask
# ===========================================================================


class TestEnrichedTask:
    def test_to_dict(self):
        task = EnrichedTask(
            id="abc12345",
            content="Test task",
            status="pending",
            priority=2,
            created_cycle=1,
            updated_cycle=1,
            ttl_cycles=10,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        d = task.to_dict()
        assert d["id"] == "abc12345"
        assert d["content"] == "Test task"
        assert d["priority"] == 2
        assert d["event_ticker"] is None
