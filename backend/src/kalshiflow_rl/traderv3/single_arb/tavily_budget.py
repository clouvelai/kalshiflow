"""
TavilyBudgetManager - Credit usage tracker for Tavily web search.

Soft tracking: logs warnings at thresholds, switches to DDG fallback
when credits exhausted. Does NOT hard-block searches.

1 basic search = 1 credit, 1 advanced search = 2 credits, 1 extract = 1 credit.
Free tier = 1,000 credits/month.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Set

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tavily_budget")

DEFAULT_MONTHLY_LIMIT = 10000
WARNING_THRESHOLDS = [5000, 7500, 9000]


@dataclass
class TavilyBudget:
    """Tracks Tavily credit and token usage."""

    total_credits_used: int = 0
    monthly_limit: int = DEFAULT_MONTHLY_LIMIT
    searches_by_event: Dict[str, int] = field(default_factory=dict)
    searches_by_type: Dict[str, int] = field(default_factory=dict)
    extract_credits_used: int = 0
    total_tokens_used: int = 0
    last_search_ts: float = 0.0
    warnings_emitted: Set[int] = field(default_factory=set)

    @property
    def credits_remaining(self) -> int:
        return max(0, self.monthly_limit - self.total_credits_used)

    @property
    def usage_pct(self) -> float:
        if self.monthly_limit <= 0:
            return 100.0
        return min(100.0, (self.total_credits_used / self.monthly_limit) * 100)


class TavilyBudgetManager:
    """Manages Tavily search credit budget.

    Soft tracking only - warns at thresholds, falls back to DDG
    when credits exhausted.
    """

    def __init__(self, monthly_limit: int = DEFAULT_MONTHLY_LIMIT):
        self._budget = TavilyBudget(monthly_limit=monthly_limit)

    def record_search(
        self,
        credits: int,
        event_ticker: str = "",
        search_type: str = "general",
    ) -> None:
        """Record a completed search against the budget."""
        self._budget.total_credits_used += credits
        self._budget.last_search_ts = time.time()

        if event_ticker:
            self._budget.searches_by_event[event_ticker] = (
                self._budget.searches_by_event.get(event_ticker, 0) + 1
            )
        self._budget.searches_by_type[search_type] = (
            self._budget.searches_by_type.get(search_type, 0) + 1
        )

        # Emit warnings at thresholds
        for threshold in WARNING_THRESHOLDS:
            if (
                self._budget.total_credits_used >= threshold
                and threshold not in self._budget.warnings_emitted
            ):
                self._budget.warnings_emitted.add(threshold)
                remaining = self._budget.credits_remaining
                logger.warning(
                    f"[TAVILY] Budget warning: {self._budget.total_credits_used}/"
                    f"{self._budget.monthly_limit} credits used "
                    f"({remaining} remaining)"
                )

    def record_extract(
        self,
        credits: int = 1,
        event_ticker: str = "",
    ) -> None:
        """Record a completed extract call against the budget.

        Each Tavily Extract call costs 1 credit per 5 URLs (minimum 1).
        """
        self._budget.total_credits_used += credits
        self._budget.extract_credits_used += credits
        self._budget.last_search_ts = time.time()

        if event_ticker:
            self._budget.searches_by_event[event_ticker] = (
                self._budget.searches_by_event.get(event_ticker, 0) + 1
            )
        self._budget.searches_by_type["extract"] = (
            self._budget.searches_by_type.get("extract", 0) + 1
        )

        # Emit warnings at thresholds
        for threshold in WARNING_THRESHOLDS:
            if (
                self._budget.total_credits_used >= threshold
                and threshold not in self._budget.warnings_emitted
            ):
                self._budget.warnings_emitted.add(threshold)
                remaining = self._budget.credits_remaining
                logger.warning(
                    f"[TAVILY] Budget warning: {self._budget.total_credits_used}/"
                    f"{self._budget.monthly_limit} credits used "
                    f"({remaining} remaining)"
                )

    def record_usage(self, tokens: int) -> None:
        """Record token-level usage from Tavily include_usage responses.

        Called by TavilySearchService when the API returns a usage object.
        Tracks cumulative token consumption for monitoring and debugging.
        """
        if tokens > 0:
            self._budget.total_tokens_used += tokens

    def can_afford(self, credits: int = 1) -> bool:
        """Pre-flight check: can we afford this many credits?"""
        return self._budget.credits_remaining >= credits

    def should_fallback(self) -> bool:
        """True when credits exhausted - service should use DDG."""
        return self._budget.credits_remaining <= 0

    def credits_remaining(self) -> int:
        return self._budget.credits_remaining

    def get_budget_status(self) -> Dict[str, Any]:
        """Get budget status for health endpoint."""
        return {
            "total_credits_used": self._budget.total_credits_used,
            "monthly_limit": self._budget.monthly_limit,
            "credits_remaining": self._budget.credits_remaining,
            "usage_pct": round(self._budget.usage_pct, 1),
            "extract_credits_used": self._budget.extract_credits_used,
            "total_tokens_used": self._budget.total_tokens_used,
            "searches_by_event": dict(self._budget.searches_by_event),
            "searches_by_type": dict(self._budget.searches_by_type),
            "last_search_ts": self._budget.last_search_ts,
            "should_fallback": self.should_fallback(),
        }
