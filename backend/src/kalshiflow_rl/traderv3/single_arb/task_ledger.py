"""Task Ledger - Enriched TODO tracking with TTL, priority, and persistence.

Intercepts write_todos from deepagents TodoListMiddleware, enriches with
timestamps/priority/TTL, detects stale tasks, and persists to Supabase.

Key Responsibilities:
- Reconcile raw write_todos output against existing tasks (content matching)
- Enrich with id, timestamps, cycle tracking, TTL, priority
- Detect stale tasks (unchanged > ttl_cycles)
- Prune completed tasks older than 5 cycles
- Format for cycle prompt injection and frontend broadcast
- Fire-and-forget persistence to Supabase (append-only audit trail)

Architecture Position:
- captain.py intercepts write_todos on_tool_start -> TaskLedger.reconcile()
- Each cycle: TaskLedger.tick() -> to_prompt_section() injected into prompt
- TaskLedger.to_broadcast() -> WebSocket -> Frontend TodoListSection
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.task_ledger")

# Priority prefixes the agent can use in task content
_PRIORITY_MAP = {"[HIGH]": 3, "[MED]": 2, "[LOW]": 1}
_PRUNE_AFTER_CYCLES = 5
_DEFAULT_TTL_CYCLES = 10
_CONTENT_SIMILARITY_THRESHOLD = 0.7


@dataclass
class EnrichedTask:
    """A single enriched task with metadata beyond raw write_todos output."""
    id: str
    content: str
    status: str  # pending, in_progress, completed, stale
    priority: int  # 1=low, 2=medium, 3=high
    created_cycle: int
    updated_cycle: int
    ttl_cycles: int
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp
    event_ticker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
            "created_cycle": self.created_cycle,
            "updated_cycle": self.updated_cycle,
            "ttl_cycles": self.ttl_cycles,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_ticker": self.event_ticker,
        }


def _extract_priority(content: str) -> tuple:
    """Extract priority prefix from content. Returns (priority, cleaned_content)."""
    upper = content.lstrip().upper()
    for prefix, prio in _PRIORITY_MAP.items():
        if upper.startswith(prefix):
            cleaned = content.lstrip()[len(prefix):].lstrip()
            return prio, cleaned
    return 2, content  # Default medium


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TaskLedger:
    """Enriched task tracking with TTL, priority, stale detection, and Supabase persistence."""

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._tasks: Dict[str, EnrichedTask] = {}  # id -> EnrichedTask
        self._pending_writes: List[asyncio.Task] = []

    @property
    def tasks(self) -> Dict[str, EnrichedTask]:
        return self._tasks

    def reconcile(self, raw_todos: List[Dict], cycle_num: int) -> None:
        """Match new write_todos output against existing tasks.

        New items get fresh metadata. Existing items get status updated.
        Items in existing set but missing from raw_todos are marked completed.
        """
        now = _iso_now()
        matched_ids = set()

        for raw in raw_todos:
            content = raw.get("content", "") or raw.get("text", "") or ""
            status = raw.get("status", "pending") or "pending"
            if not content.strip():
                continue

            priority, cleaned_content = _extract_priority(content)

            # Try to match against existing task by content similarity
            best_match_id = None
            best_score = 0.0
            for tid, task in self._tasks.items():
                if tid in matched_ids:
                    continue
                score = SequenceMatcher(None, task.content.lower(), cleaned_content.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match_id = tid

            if best_match_id and best_score >= _CONTENT_SIMILARITY_THRESHOLD:
                # Update existing task
                task = self._tasks[best_match_id]
                task.status = status
                task.updated_cycle = cycle_num
                task.updated_at = now
                task.priority = priority
                # Update content if it changed (agent may refine wording)
                if best_score < 1.0:
                    task.content = cleaned_content
                matched_ids.add(best_match_id)
            else:
                # New task
                tid = _short_id()
                self._tasks[tid] = EnrichedTask(
                    id=tid,
                    content=cleaned_content,
                    status=status,
                    priority=priority,
                    created_cycle=cycle_num,
                    updated_cycle=cycle_num,
                    ttl_cycles=_DEFAULT_TTL_CYCLES,
                    created_at=now,
                    updated_at=now,
                )
                matched_ids.add(tid)

        # Mark tasks not in the new list as completed (agent dropped them)
        for tid, task in self._tasks.items():
            if tid not in matched_ids and task.status not in ("completed", "stale"):
                task.status = "completed"
                task.updated_cycle = cycle_num
                task.updated_at = now

    def tick(self, cycle_num: int) -> None:
        """Called each cycle start. Mark stale tasks and prune old completed ones."""
        prune_ids = []
        for tid, task in self._tasks.items():
            # Stale detection: active tasks not updated for > ttl_cycles
            if task.status in ("pending", "in_progress"):
                if cycle_num - task.updated_cycle > task.ttl_cycles:
                    task.status = "stale"
                    task.updated_cycle = cycle_num
                    task.updated_at = _iso_now()

            # Prune completed/stale tasks older than _PRUNE_AFTER_CYCLES
            if task.status in ("completed", "stale"):
                if cycle_num - task.updated_cycle > _PRUNE_AFTER_CYCLES:
                    prune_ids.append(tid)

        for tid in prune_ids:
            del self._tasks[tid]

    def needs_replan(self) -> bool:
        """True if >50% of active tasks are stale."""
        active = [t for t in self._tasks.values() if t.status in ("pending", "in_progress", "stale")]
        if not active:
            return False
        stale_count = sum(1 for t in active if t.status == "stale")
        return stale_count > len(active) / 2

    def to_prompt_section(self) -> str:
        """Format active tasks for cycle prompt injection.

        Sorted by priority (high first) then status (in_progress > pending > stale > completed).
        """
        active = [t for t in self._tasks.values() if t.status != "completed"]
        if not active:
            return ""

        status_order = {"in_progress": 0, "pending": 1, "stale": 2}
        active.sort(key=lambda t: (-(t.priority), status_order.get(t.status, 9)))

        lines = ["TASKS:"]
        for t in active:
            marker = "[>]" if t.status == "in_progress" else (
                "[STALE]" if t.status == "stale" else "[ ]"
            )
            prio_label = "[HIGH] " if t.priority == 3 else ("[LOW] " if t.priority == 1 else "")
            lines.append(f"  {marker} {prio_label}{t.content}")

        return "\n".join(lines)

    def to_broadcast(self) -> List[Dict]:
        """Format all tasks for frontend broadcast with full metadata."""
        tasks = list(self._tasks.values())
        status_order = {"in_progress": 0, "pending": 1, "stale": 2, "completed": 3}
        tasks.sort(key=lambda t: (-(t.priority), status_order.get(t.status, 9)))
        return [t.to_dict() for t in tasks]

    def persist(self, pool) -> None:
        """Fire-and-forget async write to Supabase (append-only audit trail).

        Args:
            pool: asyncpg connection pool (or None to skip)
        """
        if not pool or not self._tasks:
            return

        import json
        snapshot = json.dumps(self.to_broadcast())
        cycle_num = max((t.updated_cycle for t in self._tasks.values()), default=0)

        task = asyncio.create_task(
            self._persist_to_db(pool, snapshot, cycle_num)
        )
        self._pending_writes.append(task)
        # Prune completed tasks
        self._pending_writes = [t for t in self._pending_writes if not t.done()]

    async def _persist_to_db(self, pool, snapshot: str, cycle_num: int) -> None:
        """Background Supabase write. Errors logged, not raised."""
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO captain_task_ledger (session_id, cycle, tasks)
                       VALUES ($1, $2, $3::jsonb)""",
                    self._session_id, cycle_num, snapshot,
                )
        except Exception as e:
            logger.debug(f"[TASK_LEDGER] Persist failed (non-critical): {e}")

    async def cleanup_old_entries(self, pool, days: int = 30) -> None:
        """Delete old entries from captain_task_ledger table (fire-and-forget).

        Args:
            pool: asyncpg connection pool (or None to skip)
            days: Delete entries older than this many days
        """
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"DELETE FROM captain_task_ledger WHERE created_at < NOW() - INTERVAL '{days} days'"
                )
                # result is e.g. "DELETE 42"
                count = int(result.split()[-1]) if result else 0
                if count > 0:
                    logger.info(f"[TASK_LEDGER] Cleaned up {count} entries older than {days} days")
        except Exception as e:
            logger.debug(f"[TASK_LEDGER] Cleanup failed (non-critical): {e}")

    async def flush(self, timeout: float = 3.0) -> int:
        """Await pending Supabase writes. Called on shutdown."""
        pending = [t for t in self._pending_writes if not t.done()]
        if not pending:
            return 0

        logger.info(f"[TASK_LEDGER] Flushing {len(pending)} pending writes...")
        done, not_done = await asyncio.wait(pending, timeout=timeout)

        for task in not_done:
            task.cancel()

        self._pending_writes.clear()
        return len(done)
