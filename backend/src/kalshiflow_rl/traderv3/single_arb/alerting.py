"""Alerting module for Captain operational events.

Sends notifications via Slack webhook for critical events.
Rate-limited to prevent spam. No-op if webhook URL not configured.

Key Responsibilities:
- Send Slack webhook alerts for critical/warning/info events
- Rate-limit alerts by title to prevent spam
- Prune stale rate-limit entries to prevent unbounded growth
- Gracefully degrade (no-op) when webhook URL is not configured

Architecture Position:
- Instantiated by SingleArbCoordinator with config.slack_webhook_url
- Called by AccountHealthService, Captain, and other components
- Fire-and-forget: alert failures are logged but never raised
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.alerting")


class AlertManager:
    """Rate-limited Slack webhook alerter."""

    def __init__(self, webhook_url: Optional[str] = None, rate_limit_seconds: int = 300):
        self._webhook_url = webhook_url
        self._rate_limit = rate_limit_seconds
        self._last_sent: dict[str, float] = {}  # title -> last_sent_time

    async def alert(self, level: str, title: str, details: str = ""):
        """Send alert if webhook configured and not rate-limited.

        Args:
            level: "critical", "warning", or "info"
            title: Alert title (also used as rate-limit key)
            details: Additional context
        """
        if not self._webhook_url:
            logger.debug(f"Alert suppressed (no webhook): [{level}] {title}")
            return

        now = time.time()
        if title in self._last_sent and (now - self._last_sent[title]) < self._rate_limit:
            logger.debug(f"Alert rate-limited: [{level}] {title}")
            return

        self._last_sent[title] = now

        # Prune old entries to prevent unbounded growth
        cutoff = now - self._rate_limit * 2
        self._last_sent = {k: v for k, v in self._last_sent.items() if v > cutoff}

        emoji = {"critical": "\u26a0\ufe0f", "warning": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}.get(level, "")
        payload = {
            "text": f"{emoji} *[{level.upper()}] {title}*\n{details}"
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Slack alert failed: HTTP {resp.status}")
                    else:
                        logger.info(f"Slack alert sent: [{level}] {title}")
        except Exception as e:
            logger.warning(f"Alert send failed: {e}")
