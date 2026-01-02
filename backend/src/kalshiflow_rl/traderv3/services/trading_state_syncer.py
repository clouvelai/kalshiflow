"""
Trading State Syncer Service for TRADER V3.

Dedicated service for periodic synchronization of trading state
(balance, positions, orders, settlements) with Kalshi API.
Runs in its own asyncio task for reliability.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus
    from ..core.status_reporter import StatusReporter

logger = logging.getLogger("kalshiflow_rl.traderv3.services.trading_state_syncer")


class TradingStateSyncer:
    """
    Dedicated service for trading state synchronization.

    - Initial sync on startup
    - Periodic refresh every 20s
    - Syncs balance, positions, orders, settlements
    - Emits console messages for visibility
    - Updates state_container with latest data
    - Triggers state broadcast after updating

    This service ensures trading state is kept fresh with Kalshi API
    and provides reliable periodic sync via a dedicated asyncio task.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        status_reporter: 'StatusReporter',
        sync_interval: float = 20.0
    ):
        """
        Initialize trading state syncer.

        Args:
            trading_client: Trading client for API calls
            state_container: State container for storing trading state
            event_bus: Event bus for system activity events
            status_reporter: Status reporter for broadcasting state
            sync_interval: Seconds between periodic syncs (default 20)
        """
        self._client = trading_client
        self._state = state_container
        self._event_bus = event_bus
        self._status_reporter = status_reporter
        self._sync_interval = sync_interval

        # Syncer state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Health metrics
        self._last_sync_time: Optional[float] = None
        self._sync_count: int = 0
        self._sync_errors: int = 0
        self._last_error: Optional[str] = None

        logger.info(f"TradingStateSyncer initialized (sync_interval={sync_interval}s)")

    async def start(self) -> None:
        """
        Start the trading state syncer.

        Performs initial sync immediately, then starts periodic sync loop.
        """
        if self._running:
            logger.warning("TradingStateSyncer already running")
            return

        self._running = True
        self._started_at = time.time()

        # Initial sync
        logger.info("Starting TradingStateSyncer - performing initial sync...")
        await self._sync_trading_state()

        # Start periodic sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("TradingStateSyncer started")

    async def stop(self) -> None:
        """Stop the trading state syncer."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info(f"TradingStateSyncer stopped (syncs: {self._sync_count})")

    async def _sync_loop(self) -> None:
        """Periodic sync every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_trading_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading state sync loop: {e}")
                self._sync_errors += 1
                self._last_error = str(e)

    async def _sync_trading_state(self) -> None:
        """
        Sync trading state (balance, positions, orders, settlements) from Kalshi API.

        Updates state_container and emits console message.
        """
        try:
            # Call the sync method
            state, changes = await self._client.sync_with_kalshi()

            # Always update sync timestamp
            state.sync_timestamp = time.time()

            # Update state container
            state_changed = await self._state.update_trading_state(state, changes)

            # Update health metrics
            self._last_sync_time = time.time()
            self._sync_count += 1

            # Build console message
            balance_str = f"${state.balance/100:.2f}"
            positions_count = state.position_count
            orders_count = state.order_count
            settlements_count = self._state._total_settlements_count if hasattr(self._state, '_total_settlements_count') else 0

            # Log sync completion
            logger.info(
                f"Trading state sync complete: {positions_count} positions, "
                f"{balance_str} balance, {settlements_count} settlements (sync #{self._sync_count})"
            )

            # Build message with change info if any
            if changes and (abs(changes.balance_change) > 0 or
                          changes.position_count_change != 0 or
                          changes.order_count_change != 0):
                message = (
                    f"Trading session synced: {positions_count} positions, "
                    f"{balance_str} balance (changed: {changes.balance_change:+d}c)"
                )
            else:
                message = (
                    f"Trading session synced: {positions_count} positions, "
                    f"{balance_str} balance, {settlements_count} settlements"
                )

            # Emit system activity for console visibility
            await self._event_bus.emit_system_activity(
                activity_type="sync_trading",
                message=message,
                metadata={
                    "sync_type": "trading_state",
                    "balance": state.balance,
                    "position_count": positions_count,
                    "order_count": orders_count,
                    "settlements_count": settlements_count,
                    "sync_count": self._sync_count,
                    "severity": "info"
                }
            )

            # Broadcast state update
            await self._status_reporter.emit_trading_state()

        except Exception as e:
            logger.error(f"Trading state sync failed: {e}")
            self._sync_errors += 1
            self._last_error = str(e)

            # Emit error activity
            await self._event_bus.emit_system_activity(
                activity_type="sync_trading",
                message=f"Trading session sync failed: {str(e)}",
                metadata={
                    "sync_type": "trading_state",
                    "error": str(e),
                    "severity": "error"
                }
            )

    def get_health_details(self) -> Dict[str, Any]:
        """Get health details for status reporting."""
        now = time.time()
        uptime = now - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "last_sync_time": self._last_sync_time,
            "last_sync_age_seconds": now - self._last_sync_time if self._last_sync_time else None,
            "sync_count": self._sync_count,
            "sync_errors": self._sync_errors,
            "last_error": self._last_error,
            "sync_interval": self._sync_interval,
            "healthy": self._running and (
                self._last_sync_time is None or
                (now - self._last_sync_time) < self._sync_interval * 3
            )
        }

    def is_healthy(self) -> bool:
        """Check if syncer is healthy."""
        if not self._running:
            return False

        # Healthy if running and last sync wasn't too long ago
        if self._last_sync_time:
            age = time.time() - self._last_sync_time
            if age > self._sync_interval * 3:  # 3x interval = unhealthy
                return False

        return True

    @property
    def sync_count(self) -> int:
        """Get number of syncs completed."""
        return self._sync_count

    @property
    def last_sync_time(self) -> Optional[float]:
        """Get timestamp of last successful sync."""
        return self._last_sync_time
