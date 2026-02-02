"""RALPH Agent - Main loop: monitor -> triage -> heal -> validate -> restart.

Usage:
    python -m kalshiflow_rl.traderv3.ralph.agent
"""

import asyncio
import json
import logging
import subprocess
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from .claude_code import ClaudeCodeSession
from .config import RALPHConfig
from .log_store import RALPHLogStore
from .models import (
    DetectedIssue,
    FixAttempt,
    FixResultStatus,
    RALPHPhase,
)
from .monitor import RALPHMonitor
from .trader_control import TraderControl
from .validator import RALPHValidator

logger = logging.getLogger("ralph.agent")


class RALPHAgent:
    """Self-healing agent that monitors the V3 trader and fixes bugs autonomously."""

    def __init__(self, config: Optional[RALPHConfig] = None):
        self._config = config or RALPHConfig()
        self._phase = RALPHPhase.MONITORING
        self._running = False

        # Issue queue (priority sorted)
        self._issue_queue: Deque[DetectedIssue] = deque()
        self._handled_issues: Dict[str, DetectedIssue] = {}  # issue_id -> issue

        # Rate limiting
        self._issues_this_hour: List[float] = []

        # Components
        self._monitor = RALPHMonitor(self._config, on_issue=self._on_issue_detected)
        self._claude = ClaudeCodeSession(self._config)
        self._validator = RALPHValidator(self._config)
        self._trader = TraderControl(self._config)
        self._log_store = RALPHLogStore(self._config)

        # Stats
        self._total_issues_detected = 0
        self._total_fixes_applied = 0
        self._total_escalations = 0

    async def run(self) -> None:
        """Main RALPH loop."""
        self._running = True
        logger.info("=" * 60)
        logger.info("  RALPH Agent Starting")
        logger.info("  Trader: %s", self._config.trader_base_url)
        logger.info("  Repo: %s", self._config.repo_root)
        logger.info("=" * 60)

        # Start monitor in background
        monitor_task = asyncio.create_task(self._monitor.start())

        try:
            while self._running:
                self._phase = RALPHPhase.MONITORING

                # Process any queued issues
                if self._issue_queue:
                    await self._process_next_issue()

                await asyncio.sleep(self._config.main_loop_interval)

        except asyncio.CancelledError:
            logger.info("[ralph.agent] Shutting down...")
        except KeyboardInterrupt:
            logger.info("[ralph.agent] Interrupted by user")
        finally:
            self._running = False
            await self._monitor.stop()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "[ralph.agent] RALPH stopped. Stats: issues=%d, fixes=%d, escalations=%d",
            self._total_issues_detected, self._total_fixes_applied, self._total_escalations,
        )

    async def stop(self) -> None:
        self._running = False

    # ─── Issue Handling ───────────────────────────────────────

    def _on_issue_detected(self, issue: DetectedIssue) -> None:
        """Callback from monitor when an issue is detected."""
        self._total_issues_detected += 1
        self._log_store.log_issue(issue)

        # Rate limiting
        now = time.time()
        self._issues_this_hour = [t for t in self._issues_this_hour if now - t < 3600]
        if len(self._issues_this_hour) >= self._config.max_issues_per_hour:
            logger.warning(
                "[ralph.agent] Rate limit: %d issues this hour, dropping %s",
                len(self._issues_this_hour), issue.issue_id,
            )
            return
        self._issues_this_hour.append(now)

        # Skip if already handled
        if issue.issue_id in self._handled_issues:
            return

        # Add to queue sorted by severity
        self._issue_queue.append(issue)
        # Re-sort: CRITICAL first
        sorted_issues = sorted(self._issue_queue, key=lambda i: i.severity_rank)
        self._issue_queue = deque(sorted_issues)

        logger.info(
            "[ralph.agent] Issue queued: %s [%s/%s] (queue size: %d)",
            issue.issue_id, issue.severity.value, issue.category.value, len(self._issue_queue),
        )

    async def _process_next_issue(self) -> None:
        """Process the highest-priority issue in the queue."""
        if not self._issue_queue:
            return

        issue = self._issue_queue.popleft()

        # Check if should escalate
        if issue.should_escalate:
            self._escalate(issue)
            return

        logger.info(
            "[ralph.agent] Processing issue %s: %s (attempt %d/%d)",
            issue.issue_id, issue.summary[:80],
            issue.attempt_count + 1, self._config.max_fix_attempts,
        )

        # Phase: DIAGNOSING + FIXING
        self._phase = RALPHPhase.FIXING
        issue.attempt_count += 1
        issue.last_attempt_at = time.time()
        issue.status = "handling"

        # Spawn Claude Code to fix
        fix_result = await self._claude.run_fix_session(issue)

        # Build fix attempt record
        attempt = FixAttempt(
            issue_id=issue.issue_id,
            attempt_number=issue.attempt_count,
            claude_prompt="(see logs)",
            claude_output=fix_result.output[:2000],
            result=fix_result.status,
            git_diff="",
            files_changed=fix_result.files_changed,
            duration_seconds=fix_result.duration_seconds,
            timestamp=time.time(),
        )

        if fix_result.status == FixResultStatus.ESCALATE:
            self._escalate(issue)
            self._log_store.log_fix_attempt(attempt)
            return

        if fix_result.status == FixResultStatus.SUCCESS:
            # Step 2: SIMPLIFY -- run code-simplifier on changed files
            logger.info("[ralph.agent] Running code-simplifier on fix...")
            await self._run_code_simplifier(fix_result.files_changed)

            # Step 3: SELF-VALIDATE -- pytest + static checks
            self._phase = RALPHPhase.VALIDATING
            trader_running = await self._trader.is_running()
            validation_passed = await self._validator.validate_fix(trader_is_running=trader_running)
            attempt.validation_passed = validation_passed

            if validation_passed:
                # Commit the simplified, validated fix
                if self._config.auto_commit_fixes:
                    commit_hash = await self._git_commit(issue)
                    attempt.git_commit = commit_hash
                    attempt.git_diff = await self._git_diff()

                # Step 4: TRADER-VALIDATE -- restart if needed, then observe
                needs_restart = issue.category.value in (
                    "pricing_bug", "state_corruption", "position_tracking",
                    "process_death",
                )
                if trader_running and needs_restart:
                    self._phase = RALPHPhase.RESTARTING
                    restarted = await self._trader.restart()
                    attempt.trader_restarted = restarted

                    if restarted:
                        # Observe trader for 2 health checks to confirm fix holds
                        fix_holds = await self._trader_validate(issue)
                        if not fix_holds:
                            logger.warning(
                                "[ralph.agent] Trader validation failed for %s, issue may recur",
                                issue.issue_id,
                            )

                # Write patch record for the deep agent to see
                await self._write_patch_record(issue, attempt)

                issue.status = "fixed"
                self._handled_issues[issue.issue_id] = issue
                self._total_fixes_applied += 1
                self._phase = RALPHPhase.COMPLETE

                logger.info(
                    "[ralph.agent] FIX APPLIED for %s (commit=%s)",
                    issue.issue_id, attempt.git_commit or "none",
                )
            else:
                # Validation failed -- rollback
                if self._config.auto_rollback_on_failure:
                    await self._git_rollback()
                    logger.warning("[ralph.agent] Fix rolled back for %s", issue.issue_id)

                # Re-queue if attempts remain
                if not issue.should_escalate:
                    self._issue_queue.append(issue)
                else:
                    self._escalate(issue)
        else:
            # Fix failed -- rollback any partial changes
            if self._config.auto_rollback_on_failure:
                await self._git_rollback()

            if not issue.should_escalate:
                self._issue_queue.append(issue)
            else:
                self._escalate(issue)

        self._log_store.log_fix_attempt(attempt)

    def _escalate(self, issue: DetectedIssue) -> None:
        """Mark an issue as escalated (needs human intervention)."""
        issue.status = "escalated"
        self._handled_issues[issue.issue_id] = issue
        self._total_escalations += 1
        self._phase = RALPHPhase.ESCALATED
        self._log_store.log_escalation(issue)
        logger.error(
            "[ralph.agent] ESCALATED: %s (%s) after %d attempts -- needs human intervention",
            issue.issue_id, issue.summary[:80], issue.attempt_count,
        )

    # ─── Code Simplifier ────────────────────────────────────

    async def _run_code_simplifier(self, files_changed: List[str]) -> None:
        """Run code-simplifier on recently changed files via Claude Code CLI.

        This ensures fixes are clean and maintainable before validation.
        Runs as a separate, short Claude Code session focused only on simplification.
        """
        if not files_changed:
            return

        files_list = "\n".join(f"- {f}" for f in files_changed[:5])
        prompt = f"""You are the code-simplifier. Your ONLY job is to simplify and clean up the recently modified code for clarity, consistency, and maintainability while preserving ALL functionality.

Recently modified files:
{files_list}

Rules:
- Read each file listed above
- Simplify variable names, reduce nesting, remove dead code
- Preserve ALL existing functionality -- do not change behavior
- Do not add features, comments, or docstrings
- Keep changes minimal and targeted
- Print RALPH_RESULT: SUCCESS when done"""

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", prompt,
                "--allowedTools", "Edit,Read,Grep,Glob",
                "--max-turns", "8",
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            logger.info("[ralph.agent] Code simplifier completed (exit=%s)", proc.returncode)
        except asyncio.TimeoutError:
            logger.warning("[ralph.agent] Code simplifier timed out")
        except FileNotFoundError:
            logger.debug("[ralph.agent] claude CLI not available for simplifier")
        except Exception as e:
            logger.warning("[ralph.agent] Code simplifier error: %s", e)

    # ─── Trader Validation ────────────────────────────────────

    async def _trader_validate(self, issue: DetectedIssue) -> bool:
        """Observe the trader for 2 health check cycles to confirm the fix holds.

        Returns True if the trader stays healthy and the specific issue
        doesn't recur during observation.
        """
        logger.info("[ralph.agent] Observing trader post-fix (2 checks)...")
        for i in range(2):
            await asyncio.sleep(self._config.health_poll_interval)
            if not await self._trader.is_running():
                logger.error("[ralph.agent] Trader died during post-fix observation")
                return False

            # Check pricing sanity if the issue was pricing-related
            if issue.category.value == "pricing_bug":
                if not await self._validator._check_pricing_sanity():
                    logger.error("[ralph.agent] Pricing issue recurred post-fix")
                    return False

            logger.info("[ralph.agent] Post-fix check %d/2 passed", i + 1)

        return True

    # ─── Patch Records (Deep Agent Feedback) ──────────────────

    async def _write_patch_record(self, issue: DetectedIssue, attempt: 'FixAttempt') -> None:
        """Write a patch record to ralph_patches.jsonl for the deep agent to see.

        The deep agent checks this at cycle start and sees:
        'RALPH patched issue X -> validate it's working'
        """
        patches_path = self._config.memory_dir / "ralph_patches.jsonl"
        record = {
            "issue_id": issue.issue_id,
            "fix_status": "applied",
            "description": issue.summary[:200],
            "category": issue.category.value,
            "severity": issue.severity.value,
            "commit": attempt.git_commit,
            "files_changed": attempt.files_changed,
            "trader_restarted": attempt.trader_restarted,
            "validate_instructions": f"Confirm the {issue.category.value} issue is resolved. "
                                     f"If you still see the problem, re-report via task(agent='issue_reporter').",
            "patched_at": time.time(),
            "patched_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "acked_by_agent": False,
        }
        try:
            with patches_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info("[ralph.agent] Patch record written for %s", issue.issue_id)
        except Exception as e:
            logger.error("[ralph.agent] Failed to write patch record: %s", e)

    # ─── Git Operations ───────────────────────────────────────

    async def _git_commit(self, issue: DetectedIssue) -> Optional[str]:
        """Commit current changes with a RALPH-tagged message."""
        try:
            msg = (
                f"fix(ralph): auto-fix {issue.category.value} - {issue.summary[:60]}\n\n"
                f"Issue: {issue.issue_id}\n"
                f"Severity: {issue.severity.value}\n"
                f"Source: {issue.source}\n\n"
                f"Co-Authored-By: RALPH Agent <ralph@kalshiflow.dev>"
            )

            # Stage changes in traderv3 only
            proc = await asyncio.create_subprocess_exec(
                "git", "add", "backend/src/kalshiflow_rl/traderv3/",
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Commit
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", msg,
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode()

            if proc.returncode == 0:
                # Extract commit hash
                proc2 = await asyncio.create_subprocess_exec(
                    "git", "rev-parse", "--short", "HEAD",
                    cwd=str(self._config.repo_root),
                    stdout=asyncio.subprocess.PIPE,
                )
                hash_out, _ = await proc2.communicate()
                return hash_out.decode().strip()
            else:
                logger.warning("[ralph.agent] Git commit failed: %s", output)
                return None

        except Exception as e:
            logger.error("[ralph.agent] Git commit error: %s", e)
            return None

    async def _git_rollback(self) -> None:
        """Rollback all uncommitted changes."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", "--", ".",
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            logger.info("[ralph.agent] Git rollback completed")
        except Exception as e:
            logger.error("[ralph.agent] Git rollback error: %s", e)

    async def _git_diff(self) -> str:
        """Get the git diff of staged changes."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "HEAD~1", "--stat",
                cwd=str(self._config.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode()[:1000]
        except Exception:
            return ""


# ─── Entry Point ──────────────────────────────────────────

async def main() -> None:
    """Run RALPH as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser(description="RALPH - Self-Healing Agent for V3 Trader")
    parser.add_argument("--port", type=int, default=8005, help="Trader port")
    parser.add_argument("--env", default="paper", help="Environment (paper/production)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = RALPHConfig(
        trader_port=args.port,
        environment=args.env,
    )

    agent = RALPHAgent(config)

    try:
        await agent.run()
    except KeyboardInterrupt:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
