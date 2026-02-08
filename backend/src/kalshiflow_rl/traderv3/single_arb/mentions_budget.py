"""
SimulationBudgetManager - Intelligent simulation scheduling for mentions markets.

Manages per-event simulation budget (~$1/event) with progressive refinement:
1. Startup baseline (Phase 1): 5 blind sims at startup
2. Pre-event refinement (Phase 2): 5 more blind sims every 2h if >6h to close
3. News-triggered informed (Phase 3): 5 informed sims when understanding updates
4. Live-phase burst (Phase 4): 10 informed sims when <2h to close
5. Edge-triggered deep dive (Phase 5): 10 more sims when potential edge detected

Cost model: ~$0.003 per LLM simulation call. 5 segment calls per simulation.
Budget of $1 = ~330 calls = ~66 full simulations per event.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_budget")

# Cost constants
COST_PER_CALL = 0.003  # ~$0.003 per LLM call
CALLS_PER_SIMULATION = 5  # 5 segment calls per simulation
COST_PER_SIMULATION = COST_PER_CALL * CALLS_PER_SIMULATION  # ~$0.015
DEFAULT_BUDGET_DOLLARS = 1.0
MAX_SIMULATIONS = int(DEFAULT_BUDGET_DOLLARS / COST_PER_SIMULATION)  # ~66

# Phase thresholds (hours to close)
PHASE_PRE_EVENT_HOURS = 6.0
PHASE_LIVE_HOURS = 2.0

# Scheduling intervals
PRE_EVENT_INTERVAL_SECONDS = 2 * 60 * 60  # 2 hours
LIVE_PHASE_INTERVAL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class EventBudget:
    """Per-event simulation budget tracker."""
    event_ticker: str
    total_calls_used: int = 0
    total_simulations: int = 0
    total_estimated_cost: float = 0.0
    budget_dollars: float = DEFAULT_BUDGET_DOLLARS
    phase: str = "startup"  # startup, pre_event, live, post_event
    last_sim_ts: float = 0.0
    last_sim_mode: str = ""
    next_scheduled_ts: float = 0.0
    phase_sims: Dict[str, int] = field(default_factory=dict)  # phase -> sim count

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_dollars - self.total_estimated_cost)

    @property
    def simulations_remaining(self) -> int:
        return max(0, int(self.budget_remaining / COST_PER_SIMULATION))

    @property
    def budget_pct_used(self) -> float:
        if self.budget_dollars <= 0:
            return 100.0
        return min(100.0, (self.total_estimated_cost / self.budget_dollars) * 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_ticker": self.event_ticker,
            "total_calls_used": self.total_calls_used,
            "total_simulations": self.total_simulations,
            "total_estimated_cost": round(self.total_estimated_cost, 4),
            "budget_dollars": self.budget_dollars,
            "budget_remaining": round(self.budget_remaining, 4),
            "budget_pct_used": round(self.budget_pct_used, 1),
            "simulations_remaining": self.simulations_remaining,
            "phase": self.phase,
            "last_sim_ts": self.last_sim_ts,
            "last_sim_mode": self.last_sim_mode,
            "next_scheduled_ts": self.next_scheduled_ts,
            "phase_sims": self.phase_sims,
        }


class SimulationBudgetManager:
    """
    Manages simulation budgets across all mentions events.

    Decides when and how many simulations to run based on:
    - Time to market close (phase detection)
    - Budget remaining
    - Time since last simulation
    - Edge detection signals
    """

    def __init__(self, budget_per_event: float = DEFAULT_BUDGET_DOLLARS):
        self._budgets: Dict[str, EventBudget] = {}
        self._default_budget = budget_per_event

    def register_event(self, event_ticker: str, budget: Optional[float] = None) -> None:
        """Register a mentions event for budget tracking."""
        if event_ticker not in self._budgets:
            self._budgets[event_ticker] = EventBudget(
                event_ticker=event_ticker,
                budget_dollars=budget or self._default_budget,
            )
            logger.info(
                f"[BUDGET] Registered {event_ticker}: "
                f"budget=${budget or self._default_budget:.2f} "
                f"(~{int((budget or self._default_budget) / COST_PER_SIMULATION)} sims)"
            )

    def should_simulate(
        self,
        event_ticker: str,
        time_to_close_hours: Optional[float] = None,
        has_baseline: bool = False,
        has_edge_signal: bool = False,
        understanding_updated: bool = False,
    ) -> Tuple[bool, str, int, str]:
        """Decide whether to run a simulation now.

        Args:
            event_ticker: The event to check
            time_to_close_hours: Hours until market close (None = unknown)
            has_baseline: Whether baseline probability exists
            has_edge_signal: Whether compute_edge detected potential edge
            understanding_updated: Whether update_understanding was just called

        Returns:
            Tuple of (should_run, reason, n_simulations, mode)
            - should_run: True if simulation should run now
            - reason: Human-readable explanation
            - n_simulations: Recommended simulation count
            - mode: "blind" or "informed"
        """
        budget = self._budgets.get(event_ticker)
        if not budget:
            return False, "Event not registered", 0, ""

        now = time.time()

        # Budget exhausted
        if budget.simulations_remaining <= 0:
            return False, "Budget exhausted", 0, ""

        # Detect phase
        old_phase = budget.phase
        if time_to_close_hours is not None:
            if time_to_close_hours <= 0:
                budget.phase = "post_event"
            elif time_to_close_hours <= PHASE_LIVE_HOURS:
                budget.phase = "live"
            elif time_to_close_hours <= PHASE_PRE_EVENT_HOURS:
                budget.phase = "pre_event"
            else:
                budget.phase = "pre_event"

        # Phase 0: No baseline yet → establish immediately
        if not has_baseline:
            return True, "No baseline - establish baseline", 10, "blind"

        # Phase transition → always simulate on transition
        if old_phase != budget.phase and old_phase != "startup":
            n = 10 if budget.phase == "live" else 5
            mode = "informed" if budget.phase == "live" else "blind"
            return True, f"Phase transition: {old_phase} -> {budget.phase}", n, mode

        # Phase 5: Edge signal → deep dive
        if has_edge_signal and budget.simulations_remaining >= 10:
            time_since_last = now - budget.last_sim_ts
            if time_since_last > 120:  # Don't re-trigger within 2 minutes
                return True, "Edge signal detected - deep dive", 10, "informed"

        # Phase 3: Understanding updated → informed refresh
        if understanding_updated and budget.simulations_remaining >= 5:
            return True, "Understanding updated - informed refresh", 5, "informed"

        # Phase 4: Live phase → frequent informed sims
        if budget.phase == "live":
            time_since_last = now - budget.last_sim_ts
            if time_since_last >= LIVE_PHASE_INTERVAL_SECONDS:
                return True, "Live phase - periodic refresh", 10, "informed"
            return False, f"Live phase - next in {int(LIVE_PHASE_INTERVAL_SECONDS - time_since_last)}s", 0, ""

        # Phase 2: Pre-event → periodic blind refinement
        if budget.phase == "pre_event":
            time_since_last = now - budget.last_sim_ts
            if time_since_last >= PRE_EVENT_INTERVAL_SECONDS:
                return True, "Pre-event - periodic refinement", 5, "blind"
            return False, f"Pre-event - next in {int(PRE_EVENT_INTERVAL_SECONDS - time_since_last)}s", 0, ""

        # Post-event → no more simulations
        if budget.phase == "post_event":
            return False, "Post-event - no more simulations", 0, ""

        return False, "No trigger condition met", 0, ""

    def record_simulation(
        self,
        event_ticker: str,
        n_simulations: int,
        mode: str,
    ) -> None:
        """Record a completed simulation against the budget.

        Args:
            event_ticker: The event
            n_simulations: Number of simulations run
            mode: "blind" or "informed"
        """
        budget = self._budgets.get(event_ticker)
        if not budget:
            return

        n_calls = n_simulations * CALLS_PER_SIMULATION
        cost = n_simulations * COST_PER_SIMULATION

        budget.total_calls_used += n_calls
        budget.total_simulations += n_simulations
        budget.total_estimated_cost += cost
        budget.last_sim_ts = time.time()
        budget.last_sim_mode = mode

        # Track per-phase
        phase = budget.phase
        budget.phase_sims[phase] = budget.phase_sims.get(phase, 0) + n_simulations

        # Schedule next
        if budget.phase == "live":
            budget.next_scheduled_ts = time.time() + LIVE_PHASE_INTERVAL_SECONDS
        elif budget.phase == "pre_event":
            budget.next_scheduled_ts = time.time() + PRE_EVENT_INTERVAL_SECONDS
        else:
            budget.next_scheduled_ts = 0

        logger.info(
            f"[BUDGET] {event_ticker}: +{n_simulations} sims ({mode}), "
            f"total={budget.total_simulations}/{MAX_SIMULATIONS}, "
            f"cost=${budget.total_estimated_cost:.3f}/${budget.budget_dollars:.2f}, "
            f"phase={budget.phase}"
        )

    def get_budget_status(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        """Get budget status for agent visibility."""
        budget = self._budgets.get(event_ticker)
        if not budget:
            return None
        return budget.to_dict()

    def get_all_budgets(self) -> Dict[str, Dict[str, Any]]:
        """Get all budget statuses."""
        return {
            ticker: budget.to_dict()
            for ticker, budget in self._budgets.items()
        }

    def schedule_next(self, event_ticker: str) -> Optional[float]:
        """Get the next scheduled simulation timestamp for an event.

        Returns None if no next simulation is scheduled.
        """
        budget = self._budgets.get(event_ticker)
        if not budget or budget.next_scheduled_ts <= 0:
            return None
        return budget.next_scheduled_ts
