"""
Trading Flow Orchestrator for V3 Trader.

Manages the complete trading flow with clean cycle tracking and 
Kalshi-as-truth state management. Extracted from coordinator to
provide clear separation of concerns.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..services.trading_decision_service import TradingDecisionService, TradingDecision
    from ..services.whale_tracker import WhaleTracker
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus
    from ..core.state_machine import TraderStateMachine as V3StateMachine
    from ..config.environment import V3Config

from ..services.trading_decision_service import TradingStrategy

from ..core.state_machine import TraderState as V3State

logger = logging.getLogger("kalshiflow_rl.traderv3.core.trading_flow")


class CyclePhase(Enum):
    """Trading cycle phases."""
    SYNC = "sync"
    EVALUATE = "evaluate" 
    DECIDE = "decide"
    EXECUTE = "execute"
    COMPLETE = "complete"


@dataclass
class TradingCycle:
    """Represents a complete trading cycle."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)
    phase: CyclePhase = CyclePhase.SYNC
    markets_evaluated: List[str] = field(default_factory=list)
    decisions_made: List['TradingDecision'] = field(default_factory=list)
    trades_executed: int = 0
    sync_performed: bool = False
    completed_at: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get cycle duration in seconds."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/broadcasting."""
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "phase": self.phase.value,
            "markets_evaluated": len(self.markets_evaluated),
            "decisions_made": len(self.decisions_made),
            "trades_executed": self.trades_executed,
            "sync_performed": self.sync_performed,
            "duration": self.duration,
            "completed": self.completed_at is not None,
            "error": self.error
        }


class TradingFlowOrchestrator:
    """
    Orchestrates the trading flow with clean cycle management.
    
    Responsibilities:
    - Manage trading cycles with unique IDs
    - Enforce Kalshi-as-truth state synchronization
    - Coordinate market evaluation and decision execution
    - Track cycle metrics and performance
    - Handle errors gracefully within cycles
    """
    
    def __init__(
        self,
        config: 'V3Config',
        trading_client: 'V3TradingClientIntegration',
        orderbook_integration: 'V3OrderbookIntegration',
        trading_service: 'TradingDecisionService',
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        state_machine: 'V3StateMachine',
        whale_tracker: Optional['WhaleTracker'] = None
    ):
        """
        Initialize the trading flow orchestrator.

        Args:
            config: V3 configuration
            trading_client: Trading client for API operations
            orderbook_integration: Orderbook data source
            trading_service: Trading decision service
            state_container: Shared state container
            event_bus: Event bus for system events
            state_machine: State machine for transitions
            whale_tracker: Optional whale tracker for WHALE_FOLLOWER strategy
        """
        self._config = config
        self._trading_client = trading_client
        self._orderbook_integration = orderbook_integration
        self._trading_service = trading_service
        self._state_container = state_container
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._whale_tracker = whale_tracker
        
        # Cycle tracking
        self._current_cycle: Optional[TradingCycle] = None
        self._cycle_history: List[TradingCycle] = []
        self._max_history_size = 100
        
        # Timing configuration
        self._cycle_interval = 30.0  # Seconds between cycles
        self._last_cycle_time = 0.0
        self._sync_before_each_cycle = True  # Always sync before trading
        
        # Metrics
        self._total_cycles = 0
        self._successful_cycles = 0
        self._failed_cycles = 0
        
        logger.info("Trading Flow Orchestrator initialized")
    
    async def check_and_run_cycle(self) -> bool:
        """
        Check if it's time to run a trading cycle and execute if so.
        
        Returns:
            True if cycle was run, False if skipped
        """
        # Check if enough time has passed
        current_time = time.time()
        if current_time - self._last_cycle_time < self._cycle_interval:
            return False
        
        # Allow trading in READY or ERROR state (for degraded mode)
        # ERROR state can occur when orderbook is down but trading client is still functional
        if self._state_machine.current_state.value not in ["ready", "error"]:
            logger.debug(f"Skipping cycle - state is {self._state_machine.current_state.value}")
            return False
        
        # Run the trading cycle
        await self.run_trading_cycle()
        self._last_cycle_time = current_time
        return True
    
    async def run_trading_cycle(self) -> TradingCycle:
        """
        Run a complete trading cycle with all phases.
        
        Phases:
        1. SYNC - Synchronize with Kalshi (forced)
        2. EVALUATE - Evaluate markets for opportunities
        3. DECIDE - Make trading decisions
        4. EXECUTE - Execute trades
        5. COMPLETE - Finalize and report
        
        Returns:
            Completed trading cycle
        """
        # Create new cycle
        cycle = TradingCycle()
        self._current_cycle = cycle
        self._total_cycles += 1
        
        logger.info(f"🔄 Starting trading cycle {cycle.cycle_id}")
        
        try:
            # Emit cycle start event
            await self._event_bus.emit_system_activity(
                activity_type="trading_cycle",
                message=f"Starting trading cycle {cycle.cycle_id}",
                metadata={"cycle_id": cycle.cycle_id, "phase": "start"}
            )
            
            # Phase 1: SYNC - Always sync with Kalshi first
            cycle.phase = CyclePhase.SYNC
            await self._sync_phase(cycle)
            
            # Phase 2: EVALUATE - Evaluate markets
            cycle.phase = CyclePhase.EVALUATE
            await self._evaluate_phase(cycle)
            
            # Phase 3: DECIDE - Make decisions
            cycle.phase = CyclePhase.DECIDE
            await self._decide_phase(cycle)
            
            # Phase 4: EXECUTE - Execute trades if any
            if cycle.decisions_made:
                cycle.phase = CyclePhase.EXECUTE
                await self._execute_phase(cycle)
            
            # Phase 5: COMPLETE - Mark complete
            cycle.phase = CyclePhase.COMPLETE
            cycle.completed_at = time.time()
            self._successful_cycles += 1
            
            logger.info(
                f"✅ Completed trading cycle {cycle.cycle_id} "
                f"({cycle.duration:.1f}s, {len(cycle.decisions_made)} decisions, "
                f"{cycle.trades_executed} trades)"
            )
            
            # Emit completion event
            await self._event_bus.emit_system_activity(
                activity_type="trading_cycle",
                message=f"Completed cycle {cycle.cycle_id}: {cycle.trades_executed} trades",
                metadata=cycle.to_dict()
            )
            
        except Exception as e:
            # Handle cycle failure
            cycle.error = str(e)
            cycle.completed_at = time.time()
            self._failed_cycles += 1
            
            logger.error(f"❌ Trading cycle {cycle.cycle_id} failed: {e}")
            
            # Emit failure event
            await self._event_bus.emit_system_activity(
                activity_type="trading_cycle",
                message=f"Cycle {cycle.cycle_id} failed: {str(e)}",
                metadata=cycle.to_dict()
            )
        
        finally:
            # Store in history
            self._cycle_history.append(cycle)
            if len(self._cycle_history) > self._max_history_size:
                self._cycle_history.pop(0)
            
            self._current_cycle = None
        
        return cycle
    
    async def _sync_phase(self, cycle: TradingCycle) -> None:
        """
        Phase 1: Synchronize with Kalshi to get truth state.
        
        This is critical - we ALWAYS sync before making decisions
        to ensure we have the latest state from Kalshi.
        """
        logger.debug(f"Cycle {cycle.cycle_id}: Starting SYNC phase")
        
        try:
            # Perform sync through trading client
            state, changes = await self._trading_client.sync_with_kalshi()
            
            # Update state container
            state_changed = self._state_container.update_trading_state(state, changes)
            
            # Mark sync performed
            cycle.sync_performed = True
            
            # Log sync results
            if changes and (abs(changes.balance_change) > 0 or 
                          changes.position_count_change != 0 or 
                          changes.order_count_change != 0):
                logger.info(
                    f"Cycle {cycle.cycle_id}: State synced - "
                    f"Balance: ${state.balance/100:.2f} ({changes.balance_change:+d} cents), "
                    f"Positions: {state.position_count} ({changes.position_count_change:+d}), "
                    f"Orders: {state.order_count} ({changes.order_count_change:+d})"
                )
            else:
                logger.debug(
                    f"Cycle {cycle.cycle_id}: State synced - no changes"
                )
            
        except Exception as e:
            logger.error(f"Cycle {cycle.cycle_id}: Sync failed: {e}")
            # Don't fail the cycle on sync error - continue with stale state
            # but mark that sync failed
            cycle.sync_performed = False
    
    async def _evaluate_phase(self, cycle: TradingCycle) -> None:
        """
        Phase 2: Evaluate markets for trading opportunities.
        """
        logger.debug(f"Cycle {cycle.cycle_id}: Starting EVALUATE phase")
        
        # Get markets to evaluate
        markets = self._select_markets_to_evaluate()
        
        for market in markets:
            try:
                # Get orderbook for market
                orderbook = self._orderbook_integration.get_orderbook(market)
                
                if not orderbook:
                    logger.debug(f"Cycle {cycle.cycle_id}: No orderbook for {market}")
                    continue
                
                # Mark as evaluated
                cycle.markets_evaluated.append(market)
                
                logger.debug(
                    f"Cycle {cycle.cycle_id}: Evaluated {market} "
                    f"(bid: {orderbook.yes[0][0] if orderbook.yes else 'N/A'}, "
                    f"ask: {orderbook.no[0][0] if orderbook.no else 'N/A'})"
                )
                
            except Exception as e:
                logger.error(f"Cycle {cycle.cycle_id}: Error evaluating {market}: {e}")
    
    async def _decide_phase(self, cycle: TradingCycle) -> None:
        """
        Phase 3: Make trading decisions based on evaluated markets.

        For WHALE_FOLLOWER strategy, this evaluates the whale queue instead
        of individual markets.
        """
        logger.debug(f"Cycle {cycle.cycle_id}: Starting DECIDE phase")

        # Check if using whale follower strategy
        if self._trading_service._strategy == TradingStrategy.WHALE_FOLLOWER:
            await self._decide_phase_whale_follower(cycle)
            return

        # Standard market-by-market decision making
        for market in cycle.markets_evaluated:
            try:
                # Get orderbook again for decision
                orderbook = self._orderbook_integration.get_orderbook(market)

                # Use trading service to make decision
                decision = await self._trading_service.evaluate_market(market, orderbook)

                # Store non-hold decisions
                if decision and decision.action != "hold":
                    cycle.decisions_made.append(decision)
                    logger.info(
                        f"Cycle {cycle.cycle_id}: Decision for {market}: "
                        f"{decision.action} {decision.quantity} {decision.side}"
                    )

            except Exception as e:
                logger.error(f"Cycle {cycle.cycle_id}: Error deciding on {market}: {e}")

    async def _decide_phase_whale_follower(self, cycle: TradingCycle) -> None:
        """
        Whale Follower decision phase.

        Gets the whale queue from whale_tracker and evaluates it
        through the trading service.
        """
        if not self._whale_tracker:
            logger.warning(
                f"Cycle {cycle.cycle_id}: WHALE_FOLLOWER strategy but no whale tracker"
            )
            return

        try:
            # Get whale queue from tracker
            queue_state = self._whale_tracker.get_queue_state()
            whale_queue = queue_state.get("queue", [])

            if not whale_queue:
                logger.debug(f"Cycle {cycle.cycle_id}: No whales in queue to follow")
                return

            logger.debug(
                f"Cycle {cycle.cycle_id}: Evaluating {len(whale_queue)} whales"
            )

            # Evaluate whale queue through trading service
            decision = await self._trading_service.evaluate_whale_queue(whale_queue)

            if decision and decision.action != "hold":
                cycle.decisions_made.append(decision)
                logger.info(
                    f"Cycle {cycle.cycle_id}: Whale follow decision: "
                    f"{decision.action} {decision.quantity} {decision.side} on {decision.market}"
                )
            else:
                logger.debug(f"Cycle {cycle.cycle_id}: No valid whale to follow")

        except Exception as e:
            logger.error(f"Cycle {cycle.cycle_id}: Error in whale follower decide: {e}")
    
    async def _execute_phase(self, cycle: TradingCycle) -> None:
        """
        Phase 4: Execute trading decisions.
        """
        logger.debug(f"Cycle {cycle.cycle_id}: Starting EXECUTE phase")
        
        # Transition to ACTING state
        await self._state_machine.transition_to(
            V3State.ACTING,
            context=f"Executing {len(cycle.decisions_made)} trades",
            metadata={
                "cycle_id": cycle.cycle_id,
                "decisions": len(cycle.decisions_made)
            }
        )
        
        try:
            # Execute each decision
            for decision in cycle.decisions_made:
                try:
                    success = await self._trading_service.execute_decision(decision)
                    
                    if success:
                        cycle.trades_executed += 1
                        logger.info(
                            f"Cycle {cycle.cycle_id}: Executed trade on {decision.market}"
                        )
                    else:
                        logger.warning(
                            f"Cycle {cycle.cycle_id}: Failed to execute trade on {decision.market}"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"Cycle {cycle.cycle_id}: Error executing trade on {decision.market}: {e}"
                    )
            
            # Perform post-execution sync to get updated state
            if cycle.trades_executed > 0:
                logger.debug(f"Cycle {cycle.cycle_id}: Post-execution sync")
                await self._sync_phase(cycle)
            
        finally:
            # Always transition back to READY
            await self._state_machine.transition_to(
                V3State.READY,
                context=f"Completed {cycle.trades_executed} trades",
                metadata={
                    "cycle_id": cycle.cycle_id,
                    "trades_executed": cycle.trades_executed
                }
            )
    
    def _select_markets_to_evaluate(self) -> List[str]:
        """
        Select which markets to evaluate in this cycle.
        
        Returns:
            List of market tickers to evaluate
        """
        # For MVP, evaluate first few markets
        # In production, implement smart selection based on:
        # - Volume/activity
        # - Time since last evaluation
        # - Position exposure
        # - Market conditions
        
        markets = self._config.market_tickers[:3]  # Evaluate first 3 markets
        return markets
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_cycles": self._total_cycles,
            "successful_cycles": self._successful_cycles,
            "failed_cycles": self._failed_cycles,
            "cycle_interval": self._cycle_interval,
            "current_cycle": self._current_cycle.to_dict() if self._current_cycle else None,
            "recent_cycles": [c.to_dict() for c in self._cycle_history[-5:]],
            "sync_before_each_cycle": self._sync_before_each_cycle
        }
    
    def set_cycle_interval(self, interval: float) -> None:
        """Set the interval between trading cycles."""
        self._cycle_interval = max(5.0, interval)  # Minimum 5 seconds
        logger.info(f"Trading cycle interval set to {self._cycle_interval}s")
    
    def is_healthy(self) -> bool:
        """Check if orchestrator is healthy."""
        # Consider unhealthy if too many recent failures
        if self._total_cycles > 10:
            failure_rate = self._failed_cycles / self._total_cycles
            return failure_rate < 0.5
        return True