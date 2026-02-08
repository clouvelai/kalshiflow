#!/usr/bin/env python3
"""Read open issues from issues.jsonl for the self-fix script.

Usage:
    python scripts/read_issues.py              # JSON array of open issues
    python scripts/read_issues.py --summary    # Summary counts
    python scripts/read_issues.py --resolve ID "resolution text"
"""

import json
import os
import sys

ISSUES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "backend", "src", "kalshiflow_rl", "traderv3",
    "single_arb", "memory", "data", "issues.jsonl",
)


def read_all():
    if not os.path.exists(ISSUES_PATH):
        return []
    issues = []
    with open(ISSUES_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    issues.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return issues


def write_all(issues):
    with open(ISSUES_PATH, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue) + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        issues = read_all()
        by_status = {}
        for i in issues:
            s = i.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1
        print(json.dumps({"total": len(issues), "by_status": by_status}))
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--resolve":
        issue_id = sys.argv[2]
        resolution = sys.argv[3] if len(sys.argv) > 3 else "Auto-fixed by self-fix.sh"
        issues = read_all()
        found = False
        for issue in issues:
            if issue.get("id") == issue_id:
                issue["status"] = "fixed"
                issue["resolution"] = resolution
                found = True
                break
        if found:
            write_all(issues)
            print(json.dumps({"status": "resolved", "id": issue_id}))
        else:
            print(json.dumps({"status": "not_found", "id": issue_id}))
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--mark-attempted":
        issue_id = sys.argv[2]
        issues = read_all()
        import time
        for issue in issues:
            if issue.get("id") == issue_id:
                issue["last_attempt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                break
        write_all(issues)
        return

    # Default: print open issues sorted by severity
    issues = read_all()
    open_issues = [i for i in issues if i.get("status") == "open"]
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    open_issues.sort(key=lambda i: (
        severity_order.get(i.get("severity", "low"), 99),
        i.get("timestamp", ""),
    ))
    print(json.dumps(open_issues))


if __name__ == "__main__":
    main()
