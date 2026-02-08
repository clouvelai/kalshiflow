"""
File-based memory store for the single-arb agent.

Provides fast, local, always-available storage:
- journal.jsonl: append-only log of all memory events
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.file_store")

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)


class FileMemoryStore:
    """
    Local file-based memory store. Always succeeds (no network dependency).

    Files:
    - journal.jsonl: append-only log of all memory events
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._journal_path = self._data_dir / "journal.jsonl"

    def append(
        self,
        content: str,
        memory_type: str = "learning",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a memory event to the journal. Always succeeds."""
        entry = {
            "content": content,
            "type": memory_type,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"Failed to append to journal: {e}")

        return entry

    def get_journal(self, limit: int = 50, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read recent journal entries (newest first)."""
        entries: List[Dict[str, Any]] = []
        if not self._journal_path.exists():
            return entries

        try:
            with open(self._journal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if memory_type and entry.get("type") != memory_type:
                            continue
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning(f"Failed to read journal: {e}")

        # Return newest first, limited
        entries.reverse()
        return entries[:limit]

    def search_journal(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple keyword search across journal entries."""
        query_lower = query.lower()
        matches: List[Dict[str, Any]] = []

        entries = self.get_journal(limit=500)  # Search recent entries
        for entry in entries:
            content = entry.get("content", "").lower()
            if query_lower in content:
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        journal_count = 0
        type_counts: Dict[str, int] = {}
        oldest_ts = None
        newest_ts = None

        if self._journal_path.exists():
            try:
                with open(self._journal_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            journal_count += 1
                            mt = entry.get("type", "unknown")
                            type_counts[mt] = type_counts.get(mt, 0) + 1
                            ts = entry.get("timestamp")
                            if ts:
                                if oldest_ts is None or ts < oldest_ts:
                                    oldest_ts = ts
                                if newest_ts is None or ts > newest_ts:
                                    newest_ts = ts
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass

        return {
            "journal_entries": journal_count,
            "type_counts": type_counts,
            "oldest_entry": oldest_ts,
            "newest_entry": newest_ts,
            "data_dir": str(self._data_dir),
        }
