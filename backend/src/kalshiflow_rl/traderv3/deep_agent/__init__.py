"""
Arbitrage Deep Agent v2 - Captain + EventAnalyst + MemoryCurator.

Captain (create_react_agent) delegates to subagents via tool calls.
Trade execution via self-contained buy_arb_position / sell_arb_position tools.
Shared data (PairRegistry, SpreadMonitor, EventCodex) accessed via read-only snapshot tools.
"""

from .orchestrator import ArbOrchestrator, ArbOrchestratorConfig

__all__ = ["ArbOrchestrator", "ArbOrchestratorConfig"]
