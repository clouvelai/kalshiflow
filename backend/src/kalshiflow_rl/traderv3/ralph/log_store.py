"""Persistent JSON log storage for RALPH actions.

All RALPH actions are logged to backend/ralph_logs/{date}/:
- issues.jsonl -- every detected issue
- fixes.jsonl -- every fix attempt
- escalations.jsonl -- issues that exceeded retry budget
"""

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import RALPHConfig
from .models import DetectedIssue, FixAttempt, FixResultStatus

logger = logging.getLogger("ralph.log_store")


class RALPHLogStore:
    """Persistent JSONL logging for all RALPH actions."""

    def __init__(self, config: RALPHConfig):
        self._config = config
        self._log_dir = config.log_dir

    def _get_day_dir(self) -> Path:
        """Get today's log directory, creating if needed."""
        day_dir = self._log_dir / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir

    def log_issue(self, issue: DetectedIssue) -> None:
        """Log a detected issue."""
        filepath = self._get_day_dir() / "issues.jsonl"
        record = {
            "issue_id": issue.issue_id,
            "severity": issue.severity.value,
            "category": issue.category.value,
            "summary": issue.summary,
            "source": issue.source,
            "detected_at": issue.detected_at,
            "evidence_preview": str(issue.evidence)[:500],
            "logged_at": time.time(),
        }
        self._append_jsonl(filepath, record)

    def log_fix_attempt(self, attempt: FixAttempt) -> None:
        """Log a fix attempt."""
        filepath = self._get_day_dir() / "fixes.jsonl"
        record = {
            "issue_id": attempt.issue_id,
            "attempt_number": attempt.attempt_number,
            "result": attempt.result.value,
            "validation_passed": attempt.validation_passed,
            "trader_restarted": attempt.trader_restarted,
            "git_commit": attempt.git_commit,
            "files_changed": attempt.files_changed,
            "duration_seconds": attempt.duration_seconds,
            "timestamp": attempt.timestamp,
            "prompt_preview": attempt.claude_prompt[:300],
            "output_preview": attempt.claude_output[:500],
        }
        self._append_jsonl(filepath, record)

    def log_escalation(self, issue: DetectedIssue) -> None:
        """Log an escalated issue."""
        filepath = self._get_day_dir() / "escalations.jsonl"
        record = {
            "issue_id": issue.issue_id,
            "severity": issue.severity.value,
            "category": issue.category.value,
            "summary": issue.summary,
            "attempt_count": issue.attempt_count,
            "evidence_preview": str(issue.evidence)[:500],
            "escalated_at": time.time(),
        }
        self._append_jsonl(filepath, record)
        logger.warning(
            "[ralph.log_store] ESCALATED: %s (after %d attempts)",
            issue.summary[:100], issue.attempt_count,
        )

    @staticmethod
    def _append_jsonl(filepath: Path, record: dict) -> None:
        """Append a JSON record to a JSONL file."""
        try:
            with filepath.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error("[ralph.log_store] Failed to write to %s: %s", filepath, e)
