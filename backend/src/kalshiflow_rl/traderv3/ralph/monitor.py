"""RALPH Monitor - Detects issues from multiple sources.

Sources:
1. Health endpoint (GET /v3/health) -- polled every 30s
2. WebSocket (ws://v3/ws) -- real-time error messages
3. Memory files (mistakes.md, strategy.md) -- scanned every 60s
4. Process check (HTTP connect) -- every 15s
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from .config import RALPHConfig
from .models import DetectedIssue, IssueCategory, IssueSeverity

logger = logging.getLogger("ralph.monitor")

# Error patterns: (regex, category, severity)
ERROR_PATTERNS = [
    (re.compile(r"impossible.*price.*?(\d+)c", re.IGNORECASE), IssueCategory.PRICING_BUG, IssueSeverity.CRITICAL),
    (re.compile(r"price.*?(\d{3,})c", re.IGNORECASE), IssueCategory.PRICING_BUG, IssueSeverity.CRITICAL),
    (re.compile(r"avg_price.*?[>]\s*99|avg_price.*?[>]\s*100", re.IGNORECASE), IssueCategory.PRICING_BUG, IssueSeverity.HIGH),
    (re.compile(r"E[4-9].*?Trigger|EMERGENCY.*?PROTOCOL", re.IGNORECASE), IssueCategory.STATE_CORRUPTION, IssueSeverity.HIGH),
    (re.compile(r"position\s+tracking.*?fail|complete.*?disconnect", re.IGNORECASE), IssueCategory.POSITION_TRACKING, IssueSeverity.CRITICAL),
    (re.compile(r"Watchdog.*?permanently\s+stopped", re.IGNORECASE), IssueCategory.AGENT_CRASH, IssueSeverity.HIGH),
    (re.compile(r"CATASTROPHIC", re.IGNORECASE), IssueCategory.STATE_CORRUPTION, IssueSeverity.CRITICAL),
    (re.compile(r"trading.*?halted|HALT.*?ALL.*?TRADING", re.IGNORECASE), IssueCategory.STATE_CORRUPTION, IssueSeverity.HIGH),
    (re.compile(r"WebSocket.*?disconnect|ws.*?connection.*?lost", re.IGNORECASE), IssueCategory.WEBSOCKET_FAILURE, IssueSeverity.MEDIUM),
    (re.compile(r"deep_agent.*?crash|agent.*?exception.*?fatal", re.IGNORECASE), IssueCategory.AGENT_CRASH, IssueSeverity.HIGH),
]


class RALPHMonitor:
    """Monitors the V3 trader for issues from multiple sources."""

    def __init__(self, config: RALPHConfig, on_issue: Callable[[DetectedIssue], None]):
        self._config = config
        self._on_issue = on_issue
        self._running = False

        # Deduplication: issue_hash -> last_seen_at
        self._seen_issues: Dict[str, float] = {}

        # Track memory file hashes to detect changes
        self._memory_hashes: Dict[str, str] = {}

        # Track how many lines of ralph_issues.jsonl we've already processed
        self._issues_file_offset: int = 0

        # WebSocket connection
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_connection = None

    async def start(self) -> None:
        """Start all monitoring tasks."""
        self._running = True
        logger.info("[ralph.monitor] Starting monitors...")
        tasks = [
            asyncio.create_task(self._issues_file_loop()),  # PRIMARY: deep agent issue reports
            asyncio.create_task(self._health_loop()),
            asyncio.create_task(self._process_check_loop()),
            asyncio.create_task(self._memory_scan_loop()),   # FALLBACK: regex scanning
            asyncio.create_task(self._websocket_loop()),
        ]
        # Run until stopped; if any task crashes, log and continue
        for coro in asyncio.as_completed(tasks):
            try:
                await coro
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[ralph.monitor] Monitor task crashed: %s", e, exc_info=True)

    async def stop(self) -> None:
        """Stop all monitoring."""
        self._running = False
        if self._ws_connection:
            await self._ws_connection.close()
        if self._ws_session:
            await self._ws_session.close()
        logger.info("[ralph.monitor] Monitors stopped.")

    # ─── Primary: Deep Agent Issue Reports ───────────────────

    async def _issues_file_loop(self) -> None:
        """Watch ralph_issues.jsonl for structured issue reports from the deep agent.

        This is the PRIMARY issue source. The deep agent calls
        task(agent='issue_reporter') which writes structured JSONL records.
        RALPH reads new lines and converts them to DetectedIssue objects.
        """
        issues_path = self._config.memory_dir / "ralph_issues.jsonl"
        while self._running:
            try:
                if issues_path.exists():
                    lines = issues_path.read_text(encoding="utf-8").strip().split("\n")
                    new_lines = lines[self._issues_file_offset:]
                    self._issues_file_offset = len(lines)

                    for line in new_lines:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            if record.get("status") != "new":
                                continue  # Already processed

                            # Map the deep agent's report to a DetectedIssue
                            category_str = record.get("category", "unknown")
                            severity_str = record.get("severity", "medium")

                            try:
                                category = IssueCategory(category_str)
                            except ValueError:
                                category = IssueCategory.STATE_CORRUPTION

                            try:
                                severity = IssueSeverity(severity_str)
                            except ValueError:
                                severity = IssueSeverity.MEDIUM

                            self._emit_issue(
                                category=category,
                                severity=severity,
                                summary=record.get("description", "Unknown issue")[:200],
                                evidence=[{
                                    "reported_by": "deep_agent",
                                    "affected_component": record.get("affected_component", ""),
                                    "error_data": record.get("error_data", ""),
                                    "reproduction_context": record.get("reproduction_context", ""),
                                    "original_issue_id": record.get("issue_id", ""),
                                }],
                                source="deep_agent_report",
                            )
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.debug("[ralph.monitor] Issues file scan error: %s", e)
            await asyncio.sleep(10)  # Check every 10s

    # ─── Health Endpoint Monitor ──────────────────────────────

    async def _health_loop(self) -> None:
        """Poll /v3/health periodically."""
        while self._running:
            try:
                await self._check_health()
            except Exception as e:
                logger.debug("[ralph.monitor] Health check error: %s", e)
            await asyncio.sleep(self._config.health_poll_interval)

    async def _check_health(self) -> None:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self._config.trader_health_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        self._emit_issue(
                            category=IssueCategory.API_ERROR,
                            severity=IssueSeverity.HIGH,
                            summary=f"Health endpoint returned HTTP {resp.status}",
                            evidence=[{"status": resp.status, "url": self._config.trader_health_url}],
                            source="health",
                        )
                        return

                    data = await resp.json()
                    status = data.get("status", "unknown")
                    if status != "healthy":
                        self._emit_issue(
                            category=IssueCategory.STATE_CORRUPTION,
                            severity=IssueSeverity.HIGH,
                            summary=f"Trader health status: {status}",
                            evidence=[data],
                            source="health",
                        )

                    # Check for impossible prices in status data
                    await self._check_pricing_sanity()

            except aiohttp.ClientError:
                # Connection refused = process might be dead
                self._emit_issue(
                    category=IssueCategory.PROCESS_DEATH,
                    severity=IssueSeverity.CRITICAL,
                    summary="Cannot connect to trader health endpoint",
                    evidence=[{"url": self._config.trader_health_url}],
                    source="health",
                )

    async def _check_pricing_sanity(self) -> None:
        """Check /v3/status for impossible prices in positions."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self._config.trader_status_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
                    positions = data.get("positions", [])
                    for pos in positions:
                        avg_price = pos.get("avg_price", 0)
                        if avg_price and avg_price > 99:
                            self._emit_issue(
                                category=IssueCategory.PRICING_BUG,
                                severity=IssueSeverity.CRITICAL,
                                summary=f"Impossible avg_price {avg_price}c for {pos.get('ticker', '?')}",
                                evidence=[pos],
                                source="health",
                            )
            except Exception:
                pass  # Non-critical; health check already covers connectivity

    # ─── Process Check ────────────────────────────────────────

    async def _process_check_loop(self) -> None:
        """Check if trader process is alive via HTTP connect."""
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self._config.trader_health_url,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        pass  # Connection succeeded, process is alive
            except aiohttp.ClientError:
                self._emit_issue(
                    category=IssueCategory.PROCESS_DEATH,
                    severity=IssueSeverity.CRITICAL,
                    summary="Trader process unreachable",
                    evidence=[{"port": self._config.trader_port}],
                    source="process",
                )
            except Exception as e:
                logger.debug("[ralph.monitor] Process check error: %s", e)
            await asyncio.sleep(self._config.process_check_interval)

    # ─── WebSocket Monitor ────────────────────────────────────

    async def _websocket_loop(self) -> None:
        """Listen to trader WebSocket for real-time error messages."""
        import websockets

        while self._running:
            try:
                async with websockets.connect(
                    self._config.trader_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("[ralph.monitor] WebSocket connected to %s", self._config.trader_ws_url)
                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            self._process_ws_message(msg)
                        except json.JSONDecodeError:
                            pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("[ralph.monitor] WebSocket error: %s, reconnecting in 10s", e)
                await asyncio.sleep(10)

    def _process_ws_message(self, msg: Dict[str, Any]) -> None:
        """Process a WebSocket message for error indicators."""
        msg_type = msg.get("type", "")
        data = msg.get("data", {})

        # Check for deep_agent_error messages
        if msg_type == "deep_agent_error":
            error_msg = data.get("error", "") or data.get("message", "")
            self._match_error_patterns(error_msg, source="websocket", extra_evidence=data)

        # Check for trade messages with impossible prices
        if msg_type == "deep_agent_trade":
            price = data.get("price_cents", 0)
            if price and price > 99:
                self._emit_issue(
                    category=IssueCategory.PRICING_BUG,
                    severity=IssueSeverity.CRITICAL,
                    summary=f"Trade broadcast with impossible price {price}c",
                    evidence=[data],
                    source="websocket",
                )

        # Check for console/log messages relayed via WS
        if msg_type in ("console_message", "deep_agent_log", "deep_agent_status"):
            text = data.get("message", "") or data.get("text", "") or str(data)
            self._match_error_patterns(text, source="websocket", extra_evidence=data)

    # ─── Memory File Scanner ──────────────────────────────────

    async def _memory_scan_loop(self) -> None:
        """Periodically scan memory files for critical patterns."""
        while self._running:
            try:
                self._scan_memory_files()
            except Exception as e:
                logger.debug("[ralph.monitor] Memory scan error: %s", e)
            await asyncio.sleep(self._config.memory_scan_interval)

    def _scan_memory_files(self) -> None:
        """Read mistakes.md and strategy.md for emergency triggers."""
        memory_dir = self._config.memory_dir
        files_to_scan = ["mistakes.md", "strategy.md", "golden_rules.md"]

        for filename in files_to_scan:
            filepath = memory_dir / filename
            if not filepath.exists():
                continue

            content = filepath.read_text(errors="replace")
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Only process if content changed since last scan
            if self._memory_hashes.get(filename) == content_hash:
                continue
            self._memory_hashes[filename] = content_hash

            # Scan for error patterns
            self._match_error_patterns(content, source="memory", extra_evidence={"file": filename})

    # ─── Pattern Matching ─────────────────────────────────────

    def _match_error_patterns(
        self,
        text: str,
        source: str,
        extra_evidence: Optional[Dict] = None,
    ) -> None:
        """Match text against known error patterns and emit issues."""
        for pattern, category, severity in ERROR_PATTERNS:
            match = pattern.search(text)
            if match:
                snippet = text[max(0, match.start() - 50):match.end() + 100].strip()
                evidence = [{"matched_text": snippet, "pattern": pattern.pattern}]
                if extra_evidence:
                    evidence.append(extra_evidence)

                self._emit_issue(
                    category=category,
                    severity=severity,
                    summary=f"Pattern match: {snippet[:120]}",
                    evidence=evidence,
                    source=source,
                )

    # ─── Issue Emission ───────────────────────────────────────

    def _emit_issue(
        self,
        category: IssueCategory,
        severity: IssueSeverity,
        summary: str,
        evidence: List[Dict],
        source: str,
    ) -> None:
        """Create and emit a DetectedIssue, with deduplication."""
        # Generate a dedup key from category + summary prefix
        dedup_key = f"{category.value}:{summary[:60]}"
        now = time.time()

        # Skip if we've seen this exact issue recently
        last_seen = self._seen_issues.get(dedup_key, 0)
        if now - last_seen < self._config.dedup_window:
            return
        self._seen_issues[dedup_key] = now

        issue_id = hashlib.sha256(
            f"{dedup_key}:{now}".encode()
        ).hexdigest()[:12]

        issue = DetectedIssue(
            issue_id=issue_id,
            severity=severity,
            category=category,
            summary=summary,
            evidence=evidence,
            source=source,
            detected_at=now,
        )

        logger.warning(
            "[ralph.monitor] ISSUE DETECTED [%s/%s] %s (source=%s)",
            severity.value, category.value, summary[:100], source,
        )

        self._on_issue(issue)
