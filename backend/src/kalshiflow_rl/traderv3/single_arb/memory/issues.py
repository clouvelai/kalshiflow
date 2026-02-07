"""
Issues tracking for the Captain self-improvement system.

Writes to issues.jsonl in the memory data directory.
Captain reports issues it discovers; scripts/self-fix.sh reads and resolves them.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.issues")

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)

VALID_SEVERITIES = ("critical", "high", "medium", "low")
VALID_CATEGORIES = (
    "memory_corruption",
    "bad_trade_logic",
    "tool_failure",
    "prompt_gap",
    "pattern_detection_error",
    "config_issue",
)
VALID_STATUSES = ("open", "in_progress", "fixed", "wont_fix")


def _issues_path(data_dir: str = DEFAULT_DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "issues.jsonl")


def _read_all_issues(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, Any]]:
    """Read all issues from the JSONL file."""
    path = _issues_path(data_dir)
    issues = []
    if not os.path.exists(path):
        return issues
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                issues.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return issues


def _write_all_issues(issues: List[Dict[str, Any]], data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Rewrite the entire issues file (used for updates)."""
    path = _issues_path(data_dir)
    with open(path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue) + "\n")


def report_issue(
    title: str,
    description: str,
    severity: str = "medium",
    category: str = "tool_failure",
    evidence: Optional[Dict[str, Any]] = None,
    proposed_fix: str = "",
    source_agent: str = "captain",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    """Report a new issue to issues.jsonl.

    Returns the created issue dict.
    """
    if severity not in VALID_SEVERITIES:
        severity = "medium"
    if category not in VALID_CATEGORIES:
        category = "tool_failure"

    issue = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "severity": severity,
        "category": category,
        "title": title,
        "description": description,
        "evidence": evidence or {},
        "proposed_fix": proposed_fix,
        "source_agent": source_agent,
        "status": "open",
        "resolution": None,
    }

    path = _issues_path(data_dir)
    with open(path, "a") as f:
        f.write(json.dumps(issue) + "\n")

    logger.info(f"[ISSUES] Reported: [{severity}] {title} (id={issue['id']})")
    return issue


def get_open_issues(
    severity_filter: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> List[Dict[str, Any]]:
    """Get all open issues, optionally filtered by severity."""
    issues = _read_all_issues(data_dir)
    open_issues = [i for i in issues if i.get("status") == "open"]
    if severity_filter and severity_filter in VALID_SEVERITIES:
        open_issues = [i for i in open_issues if i.get("severity") == severity_filter]

    # Sort: critical > high > medium > low, then oldest first
    severity_order = {s: idx for idx, s in enumerate(VALID_SEVERITIES)}
    open_issues.sort(key=lambda i: (
        severity_order.get(i.get("severity", "low"), 99),
        i.get("timestamp", ""),
    ))
    return open_issues


def resolve_issue(
    issue_id: str,
    resolution: str,
    status: str = "fixed",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[Dict[str, Any]]:
    """Mark an issue as resolved. Returns the updated issue or None if not found."""
    if status not in ("fixed", "wont_fix"):
        status = "fixed"

    issues = _read_all_issues(data_dir)
    updated = None
    for issue in issues:
        if issue.get("id") == issue_id:
            issue["status"] = status
            issue["resolution"] = resolution
            issue["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            updated = issue
            break

    if updated:
        _write_all_issues(issues, data_dir)
        logger.info(f"[ISSUES] Resolved: {issue_id} ({status})")

    return updated


def get_issues_summary(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Get counts by status and severity."""
    issues = _read_all_issues(data_dir)
    by_status: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for issue in issues:
        status = issue.get("status", "unknown")
        severity = issue.get("severity", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1

    return {
        "total": len(issues),
        "by_status": by_status,
        "by_severity": by_severity,
        "open_count": by_status.get("open", 0),
    }
