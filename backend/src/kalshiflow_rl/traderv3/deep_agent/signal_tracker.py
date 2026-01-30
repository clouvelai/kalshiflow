"""
Signal Evaluation Tracker - Lifecycle management for price impact signals.

Tracks how many times each signal has been evaluated by the agent,
what decisions were made, and whether the signal is still actionable.

Key behaviors:
- Signals loaded from Supabase on restart are marked "historical" (never re-evaluated)
- New signals get up to `max_eval_cycles` evaluations before auto-expiring
- Terminal states: traded, passed, expired, historical
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.signal_tracker")

# Terminal statuses - no further evaluation
TERMINAL_STATUSES = frozenset({"traded", "passed", "expired", "historical"})


@dataclass
class SignalEvaluation:
    """A single evaluation of a signal within one agent cycle."""
    cycle_number: int
    decision: str          # "TRADE", "WAIT", "PASS"
    reason_summary: str    # 1-line from agent's think() call
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrackedSignal:
    """Lifecycle state for a single price impact signal."""
    signal_id: str
    market_ticker: str
    entity_name: str
    status: str = "new"            # new | evaluating | traded | passed | expired | historical
    evaluations: List[SignalEvaluation] = field(default_factory=list)
    max_evaluations: int = 3
    is_historical: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluations)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def latest_decision(self) -> Optional[str]:
        if self.evaluations:
            return self.evaluations[-1].decision
        return None

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "market_ticker": self.market_ticker,
            "entity_name": self.entity_name,
            "status": self.status,
            "evaluation_count": self.evaluation_count,
            "max_evaluations": self.max_evaluations,
            "latest_decision": self.latest_decision,
            "evaluations": [
                {
                    "cycle": e.cycle_number,
                    "decision": e.decision,
                    "reason": e.reason_summary,
                    "timestamp": e.timestamp,
                }
                for e in self.evaluations
            ],
            "is_historical": self.is_historical,
        }


class SignalEvaluationTracker:
    """
    In-memory tracker mapping signal_id -> evaluation lifecycle.

    Provides:
    - Idempotent registration of signals
    - Filtering out terminal/exhausted signals
    - Recording evaluation decisions per cycle
    - Snapshot data for WebSocket broadcast
    """

    def __init__(self, max_eval_cycles: int = 3):
        self._max_eval_cycles = max_eval_cycles
        self._signals: Dict[str, TrackedSignal] = {}

    def register_signal(
        self,
        signal_id: str,
        market_ticker: str,
        entity_name: str,
        is_historical: bool = False,
    ) -> None:
        """
        Register a signal for tracking. Idempotent - skips if already registered.

        Args:
            signal_id: Unique signal identifier
            market_ticker: Market ticker for the signal
            entity_name: Entity name from the signal
            is_historical: If True, signal is pre-restart and marked historical immediately
        """
        if signal_id in self._signals:
            return

        status = "historical" if is_historical else "new"
        self._signals[signal_id] = TrackedSignal(
            signal_id=signal_id,
            market_ticker=market_ticker,
            entity_name=entity_name,
            status=status,
            max_evaluations=self._max_eval_cycles,
            is_historical=is_historical,
        )

        logger.debug(
            f"[signal_tracker] Registered signal {signal_id} "
            f"({market_ticker}, {entity_name}) as {status}"
        )

    def get_actionable_signals(self, signal_ids: List[str]) -> List[str]:
        """
        Filter a list of signal_ids to only those that are still actionable.

        Filters out:
        - Terminal signals (traded, passed, expired, historical)
        - Unknown signals (not registered - these are passed through)

        Args:
            signal_ids: List of signal IDs to filter

        Returns:
            List of signal IDs that are still actionable
        """
        actionable = []
        filtered_count = 0

        for sid in signal_ids:
            tracked = self._signals.get(sid)
            if tracked is None:
                # Unknown signal - pass through (will be registered later)
                actionable.append(sid)
            elif not tracked.is_terminal:
                actionable.append(sid)
            else:
                filtered_count += 1

        if filtered_count > 0:
            logger.info(
                f"[signal_tracker] Filtered {filtered_count} terminal signals, "
                f"{len(actionable)} actionable remaining"
            )

        return actionable

    def record_evaluation(
        self,
        signal_id: str,
        cycle: int,
        decision: str,
        reason: str = "",
    ) -> None:
        """
        Record an evaluation decision for a signal.

        Auto-expires the signal if max evaluations reached without a TRADE.

        Args:
            signal_id: Signal being evaluated
            cycle: Current agent cycle number
            decision: "TRADE", "WAIT", or "PASS"
            reason: 1-line summary from the agent's think() call
        """
        tracked = self._signals.get(signal_id)
        if tracked is None:
            logger.warning(f"[signal_tracker] Cannot record eval for unknown signal: {signal_id}")
            return

        if tracked.is_terminal:
            logger.debug(f"[signal_tracker] Skipping eval for terminal signal: {signal_id}")
            return

        # Record the evaluation
        tracked.evaluations.append(SignalEvaluation(
            cycle_number=cycle,
            decision=decision,
            reason_summary=reason[:200],
        ))

        # Update status based on decision
        if decision == "TRADE":
            tracked.status = "traded"
        elif decision == "PASS":
            tracked.status = "passed"
        elif decision == "WAIT":
            tracked.status = "evaluating"
            # Check if max evaluations exceeded
            if tracked.evaluation_count >= tracked.max_evaluations:
                tracked.status = "expired"
                logger.info(
                    f"[signal_tracker] Signal {signal_id} expired after "
                    f"{tracked.evaluation_count}/{tracked.max_evaluations} evaluations"
                )

        logger.info(
            f"[signal_tracker] Signal {signal_id}: eval {tracked.evaluation_count}/{tracked.max_evaluations} "
            f"decision={decision} -> status={tracked.status}"
        )

    def mark_traded(self, signal_id: str) -> None:
        """Mark a signal as traded (called when trade() succeeds)."""
        tracked = self._signals.get(signal_id)
        if tracked:
            tracked.status = "traded"
            logger.info(f"[signal_tracker] Signal {signal_id} marked as traded")

    def mark_passed(self, signal_id: str, reason: str = "") -> None:
        """Mark a signal as passed (agent decided not to trade)."""
        tracked = self._signals.get(signal_id)
        if tracked and not tracked.is_terminal:
            tracked.status = "passed"
            logger.info(f"[signal_tracker] Signal {signal_id} marked as passed: {reason[:100]}")

    def get_signal_state(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get the current state of a tracked signal."""
        return self._signals.get(signal_id)

    def get_all_states(self) -> List[Dict]:
        """Get all signal states for WebSocket snapshot."""
        return [s.to_dict() for s in self._signals.values()]

    def get_summary_counts(self) -> Dict[str, int]:
        """
        Get lifecycle summary counts for section header display.

        Returns:
            Dict with counts per status: {new, evaluating, traded, passed, expired, historical}
        """
        counts: Dict[str, int] = {
            "new": 0,
            "evaluating": 0,
            "traded": 0,
            "passed": 0,
            "expired": 0,
            "historical": 0,
        }
        for s in self._signals.values():
            if s.status in counts:
                counts[s.status] += 1
        return counts

    def get_signal_count(self) -> int:
        """Get total number of tracked signals."""
        return len(self._signals)
