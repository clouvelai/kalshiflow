"""IssueReportingAgent - Structures and files issues for RALPH.

The deep agent delegates here with a natural language description of
what's broken. This subagent distills it into a structured issue report
and writes it to ralph_issues.jsonl for RALPH to pick up.

Also handles checking for RALPH patches so the deep agent can validate.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SubAgent, SubAgentResult

logger = logging.getLogger("deep_agent.subagents.issue_reporter")

RALPH_ISSUES_FILE = "ralph_issues.jsonl"
RALPH_PATCHES_FILE = "ralph_patches.jsonl"


class IssueReportingAgent(SubAgent):
    """Subagent that structures bug reports for RALPH."""

    name = "issue_reporter"
    description = (
        "Report system bugs, impossible data, or code defects for automated fixing. "
        "Delegate here when you encounter broken tools, impossible prices, tracking "
        "failures, or any system malfunction. Describe what you observed and this "
        "agent will structure it into a detailed issue report for the RALPH coding "
        "agent to diagnose and fix. Also checks for patches RALPH has applied."
    )
    max_turns = 3
    model = "claude-haiku-4-20250414"

    def get_system_prompt(self) -> str:
        return """You are the Issue Reporting Agent. Your job is to take a bug description from the trading agent and structure it into a precise issue report.

You have two tools:
1. file_issue - Write a structured issue for RALPH (the coding agent) to fix
2. check_patches - Check if RALPH has applied any fixes recently

When the trading agent reports a problem:
1. Identify the category (pricing_bug, position_tracking, api_error, websocket_failure, agent_crash, state_corruption, memory_corruption)
2. Assess severity (low, medium, high, critical)
3. Extract the affected component/tool from the description
4. Identify any specific error data (numbers, ticker names, expected vs actual values)
5. Call file_issue with all of this structured

When asked to check patches, call check_patches and summarize what was fixed.

Be concise in your final response -- the trading agent just needs confirmation the issue was filed or a summary of patches."""

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "file_issue",
                "description": "File a structured issue report for RALPH to pick up and fix.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Clear description of the bug: what happened, what was expected, what was observed"
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "pricing_bug", "position_tracking", "api_error",
                                "websocket_failure", "agent_crash", "state_corruption",
                                "memory_corruption", "unknown",
                            ],
                            "description": "Issue category"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Impact severity"
                        },
                        "affected_component": {
                            "type": "string",
                            "description": "Which tool, function, or module is affected (e.g., 'trade() tool', 'position tracking', 'orderbook client')"
                        },
                        "error_data": {
                            "type": "string",
                            "description": "Raw error output, impossible values, or data that demonstrates the bug"
                        },
                        "reproduction_context": {
                            "type": "string",
                            "description": "What was happening when the bug occurred (which ticker, what action, what cycle)"
                        },
                    },
                    "required": ["description", "category", "severity"],
                },
            },
            {
                "name": "check_patches",
                "description": "Check for fixes RALPH has applied. Returns unacknowledged patches.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def handle_tool_call(self, tool_name: str, tool_input: Dict) -> Any:
        if tool_name == "file_issue":
            return await self._file_issue(tool_input)
        elif tool_name == "check_patches":
            return await self._check_patches()
        return {"error": f"Unknown tool: {tool_name}"}

    async def _file_issue(self, inp: Dict) -> Dict[str, Any]:
        """Write a structured issue to ralph_issues.jsonl."""
        issues_path = self._memory_dir / RALPH_ISSUES_FILE
        now = time.time()
        issue_id = f"da-{int(now * 1000) % 100000000}"

        record = {
            "issue_id": issue_id,
            "reported_by": "deep_agent",
            "description": inp.get("description", ""),
            "category": inp.get("category", "unknown"),
            "severity": inp.get("severity", "medium"),
            "affected_component": inp.get("affected_component", ""),
            "error_data": (inp.get("error_data") or "")[:2000],
            "reproduction_context": inp.get("reproduction_context", ""),
            "timestamp": now,
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "new",
        }

        try:
            with issues_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

            logger.info(
                "[issue_reporter] Issue %s filed: [%s/%s] %s",
                issue_id, record["severity"], record["category"],
                record["description"][:100],
            )

            # Broadcast to frontend
            if self._ws_manager:
                await self._ws_manager.broadcast_message("ralph_issue_reported", {
                    "issue_id": issue_id,
                    "severity": record["severity"],
                    "category": record["category"],
                    "description": record["description"][:200],
                    "timestamp": time.strftime("%H:%M:%S"),
                })

            return {
                "success": True,
                "issue_id": issue_id,
                "message": f"Issue {issue_id} filed for RALPH.",
            }
        except Exception as e:
            logger.error("[issue_reporter] Failed to write issue: %s", e)
            return {"success": False, "error": str(e)}

    async def _check_patches(self) -> Dict[str, Any]:
        """Read ralph_patches.jsonl and return unacknowledged patches."""
        patches_path = self._memory_dir / RALPH_PATCHES_FILE
        if not patches_path.exists():
            return {"patches": [], "message": "No patches file found. RALPH hasn't applied any fixes yet."}

        try:
            lines = patches_path.read_text(encoding="utf-8").strip().split("\n")
            patches = []
            to_ack = []

            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not record.get("acked_by_agent"):
                        patches.append(record)
                        to_ack.append(record.get("issue_id"))
                except json.JSONDecodeError:
                    continue

            # Mark as acknowledged
            if to_ack:
                all_lines = []
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("issue_id") in to_ack:
                            rec["acked_by_agent"] = True
                            rec["acked_at"] = time.time()
                        all_lines.append(json.dumps(rec, default=str))
                    except json.JSONDecodeError:
                        all_lines.append(line)
                patches_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

            return {
                "patches": patches,
                "count": len(patches),
                "message": f"{len(patches)} new patch(es) from RALPH" if patches else "No new patches.",
            }
        except Exception as e:
            logger.error("[issue_reporter] Failed to read patches: %s", e)
            return {"patches": [], "error": str(e)}
