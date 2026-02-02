"""Claude Code CLI session spawner for autonomous bug fixing.

The core innovation: RALPH delegates bug fixing to Claude Code CLI,
which has full codebase access (file reading, editing, grep, bash).
"""

import asyncio
import logging
import re
import time
from typing import List, Optional

from .config import RALPHConfig
from .models import (
    CATEGORY_FILE_MAP,
    DetectedIssue,
    FixResult,
    FixResultStatus,
)

logger = logging.getLogger("ralph.claude_code")


class ClaudeCodeSession:
    """Spawn and manage Claude Code CLI sessions for bug fixing."""

    def __init__(self, config: RALPHConfig):
        self._config = config

    async def run_fix_session(self, issue: DetectedIssue) -> FixResult:
        """Spawn a Claude Code CLI session to diagnose and fix an issue.

        Returns a FixResult indicating whether the fix succeeded.
        """
        prompt = self._build_prompt(issue)
        start_time = time.time()

        logger.info(
            "[ralph.claude_code] Spawning Claude Code session for issue %s (%s)",
            issue.issue_id, issue.category.value,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p", prompt,
                "--allowedTools", "Edit,Read,Bash,Grep,Glob",
                "--max-turns", str(self._config.max_claude_turns),
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.claude_session_timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            duration = time.time() - start_time

            logger.info(
                "[ralph.claude_code] Session completed in %.1fs (exit=%s, output=%d chars)",
                duration, proc.returncode, len(stdout),
            )

            return self._parse_output(stdout, stderr, issue, duration)

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(
                "[ralph.claude_code] Session timed out after %.1fs", duration,
            )
            return FixResult(
                status=FixResultStatus.TIMEOUT,
                output="Claude Code session timed out",
                duration_seconds=duration,
            )
        except FileNotFoundError:
            return FixResult(
                status=FixResultStatus.FAILURE,
                output="'claude' CLI not found on PATH",
                error="claude CLI not installed",
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error("[ralph.claude_code] Session error: %s", e, exc_info=True)
            return FixResult(
                status=FixResultStatus.FAILURE,
                output=str(e),
                error=str(e),
                duration_seconds=duration,
            )

    def _build_prompt(self, issue: DetectedIssue) -> str:
        """Build a structured prompt for Claude Code to diagnose and fix the issue."""
        evidence_text = self._format_evidence(issue.evidence)
        category_files = self._get_category_files(issue.category)

        return f"""You are RALPH, an autonomous bug-fixing agent for the Kalshi V3 trader system.

## Detected Issue
- **Issue ID**: {issue.issue_id}
- **Severity**: {issue.severity.value}
- **Category**: {issue.category.value}
- **Summary**: {issue.summary}
- **Source**: {issue.source}
- **Attempt**: {issue.attempt_count + 1} of {self._config.max_fix_attempts}

## Error Evidence
{evidence_text}

## Your Task
1. Read the relevant source files to understand the bug
2. Diagnose the root cause with high confidence
3. Apply a minimal, targeted fix (maximum {self._config.max_lines_per_fix} lines changed)
4. Run `cd backend && uv run pytest tests/ -x -q --ignore=tests/test_odmr_features.py --ignore=tests/test_rl --ignore=tests/traderv3/test_trading_integration.py --ignore=tests/test_backend_e2e_regression.py` to validate
5. Print your result as the LAST line of output, exactly one of:
   - RALPH_RESULT: SUCCESS - <brief description of fix>
   - RALPH_RESULT: FAILURE - <what went wrong>
   - RALPH_RESULT: ESCALATE - <why this needs human intervention>

## Safety Rules
- Only modify files in backend/src/kalshiflow_rl/traderv3/
- NEVER modify .env files, app.py, or any file outside traderv3/
- Keep fixes minimal and targeted -- do not refactor surrounding code
- Always run tests after fixing
- If you cannot diagnose the issue with high confidence, print RALPH_RESULT: ESCALATE
- Binary contract prices must always be in the 1-99 cent range

## Key Files (start your investigation here)
{category_files}

## Codebase Context
- Backend root: backend/src/kalshiflow_rl/traderv3/
- Deep agent: backend/src/kalshiflow_rl/traderv3/deep_agent/
- Core state: backend/src/kalshiflow_rl/traderv3/core/
- Test command: cd backend && uv run pytest tests/ -x -q --ignore=tests/test_odmr_features.py --ignore=tests/test_rl --ignore=tests/traderv3/test_trading_integration.py --ignore=tests/test_backend_e2e_regression.py
"""

    @staticmethod
    def _format_evidence(evidence: list) -> str:
        """Format evidence list into readable text for the prompt."""
        parts = []
        for i, item in enumerate(evidence, 1):
            if isinstance(item, dict):
                lines = []
                for k, v in item.items():
                    val_str = str(v)
                    if len(val_str) > 300:
                        val_str = val_str[:300] + "..."
                    lines.append(f"  {k}: {val_str}")
                parts.append(f"Evidence {i}:\n" + "\n".join(lines))
            else:
                parts.append(f"Evidence {i}: {str(item)[:500]}")
        return "\n\n".join(parts) if parts else "No detailed evidence available."

    def _get_category_files(self, category) -> str:
        """Get relevant file paths for a given issue category."""
        files = CATEGORY_FILE_MAP.get(category, [])
        base = "backend/src/kalshiflow_rl/traderv3/"
        return "\n".join(f"- {base}{f}" for f in files) if files else "- backend/src/kalshiflow_rl/traderv3/ (explore as needed)"

    @staticmethod
    def _parse_output(stdout: str, stderr: str, issue: DetectedIssue, duration: float) -> FixResult:
        """Parse Claude Code output to determine fix result."""
        combined = stdout + "\n" + stderr

        # Look for RALPH_RESULT marker in the last part of output
        result_match = re.search(
            r"RALPH_RESULT:\s*(SUCCESS|FAILURE|ESCALATE)\s*[-–]?\s*(.*)",
            combined,
            re.IGNORECASE,
        )

        if result_match:
            status_str = result_match.group(1).upper()
            description = result_match.group(2).strip()

            status_map = {
                "SUCCESS": FixResultStatus.SUCCESS,
                "FAILURE": FixResultStatus.FAILURE,
                "ESCALATE": FixResultStatus.ESCALATE,
            }
            status = status_map.get(status_str, FixResultStatus.FAILURE)

            # Extract changed files from output (look for Edit tool usage patterns)
            files_changed = _extract_changed_files(combined)

            return FixResult(
                status=status,
                output=combined[-2000:],  # Keep tail of output
                files_changed=files_changed,
                duration_seconds=duration,
            )

        # No explicit result marker -- assume failure
        logger.warning(
            "[ralph.claude_code] No RALPH_RESULT marker found in output (issue %s)",
            issue.issue_id,
        )
        return FixResult(
            status=FixResultStatus.FAILURE,
            output=combined[-2000:],
            error="No RALPH_RESULT marker in Claude Code output",
            duration_seconds=duration,
        )


def _extract_changed_files(output: str) -> List[str]:
    """Extract file paths that were edited from Claude Code output."""
    # Match patterns like "Edit: /path/to/file.py" or "Edited /path/to/file.py"
    patterns = [
        re.compile(r"(?:Edit|Edited|Updated|Modified).*?(/\S+\.py)", re.IGNORECASE),
        re.compile(r"file_path.*?:\s*[\"']?(/\S+\.py)", re.IGNORECASE),
    ]
    files = set()
    for pattern in patterns:
        for match in pattern.finditer(output):
            files.add(match.group(1))
    return list(files)
