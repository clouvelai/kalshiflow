"""Unit tests for the Captain self-improvement issues system."""

import json
import os

import pytest

from kalshiflow_rl.traderv3.single_arb.memory.issues import (
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    get_issues_summary,
    get_open_issues,
    report_issue,
    resolve_issue,
)


# ---------------------------------------------------------------------------
# TestReportIssue
# ---------------------------------------------------------------------------


class TestReportIssue:
    def test_creates_with_uuid(self, tmp_path):
        issue = report_issue(
            title="Test bug",
            description="Something broke",
            severity="high",
            category="tool_failure",
            data_dir=str(tmp_path),
        )
        assert "id" in issue
        assert len(issue["id"]) == 8

    def test_defaults_to_open_status(self, tmp_path):
        issue = report_issue(
            title="Test bug",
            description="Something broke",
            data_dir=str(tmp_path),
        )
        assert issue["status"] == "open"

    def test_invalid_severity_defaults(self, tmp_path):
        issue = report_issue(
            title="Bad sev",
            description="desc",
            severity="bogus",
            data_dir=str(tmp_path),
        )
        assert issue["severity"] == "medium"

    def test_invalid_category_defaults(self, tmp_path):
        issue = report_issue(
            title="Bad cat",
            description="desc",
            category="bogus",
            data_dir=str(tmp_path),
        )
        assert issue["category"] == "tool_failure"

    def test_appends_to_file(self, tmp_path):
        report_issue(title="First", description="a", data_dir=str(tmp_path))
        report_issue(title="Second", description="b", data_dir=str(tmp_path))

        path = os.path.join(tmp_path, "issues.jsonl")
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# TestGetOpenIssues
# ---------------------------------------------------------------------------


class TestGetOpenIssues:
    def test_returns_open_only(self, tmp_path):
        i1 = report_issue(title="A", description="a", data_dir=str(tmp_path))
        report_issue(title="B", description="b", data_dir=str(tmp_path))
        resolve_issue(i1["id"], resolution="done", data_dir=str(tmp_path))

        open_issues = get_open_issues(data_dir=str(tmp_path))
        assert len(open_issues) == 1
        assert open_issues[0]["title"] == "B"

    def test_severity_filter(self, tmp_path):
        report_issue(title="High", description="h", severity="high", data_dir=str(tmp_path))
        report_issue(title="Low", description="l", severity="low", data_dir=str(tmp_path))

        filtered = get_open_issues(severity_filter="high", data_dir=str(tmp_path))
        assert len(filtered) == 1
        assert filtered[0]["severity"] == "high"

    def test_sort_order(self, tmp_path):
        report_issue(title="Med", description="m", severity="medium", data_dir=str(tmp_path))
        report_issue(title="Crit", description="c", severity="critical", data_dir=str(tmp_path))
        report_issue(title="High", description="h", severity="high", data_dir=str(tmp_path))

        issues = get_open_issues(data_dir=str(tmp_path))
        severities = [i["severity"] for i in issues]
        assert severities == ["critical", "high", "medium"]

    def test_empty_file(self, tmp_path):
        issues = get_open_issues(data_dir=str(tmp_path))
        assert issues == []


# ---------------------------------------------------------------------------
# TestResolveIssue
# ---------------------------------------------------------------------------


class TestResolveIssue:
    def test_status_change(self, tmp_path):
        issue = report_issue(title="Bug", description="d", data_dir=str(tmp_path))
        resolved = resolve_issue(issue["id"], resolution="patched", data_dir=str(tmp_path))
        assert resolved["status"] == "fixed"

    def test_resolved_at_timestamp(self, tmp_path):
        issue = report_issue(title="Bug", description="d", data_dir=str(tmp_path))
        resolved = resolve_issue(issue["id"], resolution="patched", data_dir=str(tmp_path))
        assert "resolved_at" in resolved

    def test_not_found(self, tmp_path):
        result = resolve_issue("nonexistent", resolution="n/a", data_dir=str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# TestGetIssuesSummary
# ---------------------------------------------------------------------------


class TestGetIssuesSummary:
    def test_counts_by_status(self, tmp_path):
        report_issue(title="A", description="a", data_dir=str(tmp_path))
        report_issue(title="B", description="b", data_dir=str(tmp_path))
        i3 = report_issue(title="C", description="c", data_dir=str(tmp_path))
        resolve_issue(i3["id"], resolution="done", data_dir=str(tmp_path))

        summary = get_issues_summary(data_dir=str(tmp_path))
        assert summary["by_status"] == {"open": 2, "fixed": 1}

    def test_empty_file(self, tmp_path):
        summary = get_issues_summary(data_dir=str(tmp_path))
        assert summary["total"] == 0
        assert summary["open_count"] == 0
