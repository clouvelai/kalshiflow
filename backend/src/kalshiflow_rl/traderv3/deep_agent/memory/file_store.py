"""
File-based memory store for the arb deep agent.

Provides fast, local, always-available storage:
- journal.jsonl: append-only log of all memory events
- validations.json: EventAnalyst validation cache
- session.json: current session state (overwritten each cycle)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.memory.file_store")

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)


class FileMemoryStore:
    """
    Local file-based memory store. Always succeeds (no network dependency).

    Files:
    - journal.jsonl: append-only log of all memory events
    - validations.json: {pair_id: {status, reasoning, validated_at, spread_assessment}}
    - session.json: current session state
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._journal_path = self._data_dir / "journal.jsonl"
        self._validations_path = self._data_dir / "validations.json"
        self._session_path = self._data_dir / "session.json"

        # In-memory validation cache (loaded from disk on init)
        self._validations: Dict[str, Dict[str, Any]] = {}
        self._load_validations()

    def _load_validations(self) -> None:
        """Load validations cache from disk."""
        if self._validations_path.exists():
            try:
                with open(self._validations_path, "r") as f:
                    self._validations = json.load(f)
                logger.info(f"Loaded {len(self._validations)} validations from cache")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load validations cache: {e}")
                self._validations = {}

    def _save_validations(self) -> None:
        """Persist validations cache to disk."""
        try:
            with open(self._validations_path, "w") as f:
                json.dump(self._validations, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save validations cache: {e}")

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

    def get_validation(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Get cached validation for a pair. O(1) lookup."""
        return self._validations.get(pair_id)

    def save_validation(self, pair_id: str, data: Dict[str, Any]) -> None:
        """Save or update a validation entry."""
        data["validated_at"] = time.time()
        data["validated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._validations[pair_id] = data
        self._save_validations()

        # Also journal it
        self.append(
            content=f"Validation for {pair_id}: {data.get('status', 'unknown')} - {data.get('reasoning', '')[:200]}",
            memory_type="validation",
            metadata={"pair_id": pair_id, **data},
        )

    def get_all_validations(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached validations."""
        return dict(self._validations)

    def clear_validations(self) -> int:
        """Clear all cached validations. Returns the number cleared."""
        count = len(self._validations)
        self._validations.clear()
        self._save_validations()
        if count:
            logger.info(f"Cleared {count} cached validations")
        return count

    def get_session(self) -> Dict[str, Any]:
        """Get current session state."""
        if self._session_path.exists():
            try:
                with open(self._session_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save_session(self, data: Dict[str, Any]) -> None:
        """Save session state (overwrites previous)."""
        data["updated_at"] = time.time()
        try:
            with open(self._session_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save session state: {e}")

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
            "validations_cached": len(self._validations),
            "oldest_entry": oldest_ts,
            "newest_entry": newest_ts,
            "data_dir": str(self._data_dir),
        }
