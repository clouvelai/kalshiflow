"""
Order Cleanup Service for TRADER V3.

Handles cleanup of orphaned order groups on startup to prevent capital
lock-up from previous sessions. This ensures a clean slate for each
new trading session.

Extracted from coordinator.py to reduce complexity while maintaining
identical behavior and error handling.
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.services.order_cleanup_service")


class OrderCleanupService:
    """
    Cleans up orphaned order groups from previous trading sessions.

    On startup, iterates through all order groups and:
    1. Skips the current session's order group (preserves it)
    2. Resets each old order group (cancels resting orders)
    3. Deletes the empty order group

    This prevents capital from being locked up in old resting orders
    that may have been left behind from previous sessions.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        event_bus: 'EventBus',
    ):
        """
        Initialize order cleanup service.

        Args:
            trading_client: Trading client integration for order group operations
            event_bus: Event bus for emitting system activity notifications
        """
        self._trading_client = trading_client
        self._event_bus = event_bus

        logger.debug("OrderCleanupService initialized")

    async def cleanup_orphaned_orders(self) -> None:
        """
        Cleanup previous order groups on startup.

        This is called during startup to reset any existing order groups
        from previous sessions, ensuring a clean slate for the new session.
        The current session's order group (created in _connect_trading_client)
        is preserved.
        """
        logger.info("Cleaning up previous order groups...")

        try:
            # Get the current order group ID (created earlier in _connect_trading_client)
            current_order_group_id = self._trading_client.get_order_group_id()

            # List all order groups (API doesn't support status filter)
            order_groups = await self._trading_client.list_order_groups()

            if not order_groups:
                logger.info("No previous order groups to clean up")
                return

            deleted_count = 0
            skip_count = 0
            error_count = 0

            for group in order_groups:
                # API returns "id" field, not "order_group_id"
                group_id = group.get("id", "")
                if not group_id:
                    continue

                # Skip the current session's order group
                if group_id == current_order_group_id:
                    logger.debug(f"Skipping current order group: {group_id[:8]}...")
                    skip_count += 1
                    continue

                # Reset order group first (cancels all resting orders), then delete
                try:
                    # Reset cancels all resting orders in the group
                    await self._trading_client.reset_order_group_by_id(group_id)
                    logger.info(f"Reset old order group (cancelled orders): {group_id[:8]}...")

                    # Now delete the empty group
                    success = await self._trading_client.delete_order_group_by_id(group_id)
                    if success:
                        deleted_count += 1
                        logger.info(f"Deleted previous order group: {group_id[:8]}...")
                    else:
                        error_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete order group {group_id[:8]}...: {e}")
                    error_count += 1

            # Always emit cleanup status for visibility
            cleanup_message = (
                f"Startup cleanup: {deleted_count} order groups deleted, {skip_count} preserved"
                if deleted_count > 0 or error_count > 0
                else "Startup cleanup: No old order groups to clean"
            )
            await self._event_bus.emit_system_activity(
                activity_type="cleanup",
                message=cleanup_message,
                metadata={
                    "deleted_count": deleted_count,
                    "skip_count": skip_count,
                    "error_count": error_count,
                    "total_groups": len(order_groups),
                    "severity": "info" if error_count == 0 else "warning"
                }
            )
            logger.info(
                f"Startup cleanup complete: {deleted_count} order groups deleted, "
                f"{skip_count} preserved, {error_count} errors"
            )

        except Exception as e:
            # Don't fail startup on cleanup errors - just log and continue
            logger.warning(f"Order group cleanup failed (non-critical): {e}")
            await self._event_bus.emit_system_activity(
                activity_type="cleanup",
                message=f"Startup cleanup failed: {str(e)}",
                metadata={"error": str(e), "severity": "warning"}
            )
