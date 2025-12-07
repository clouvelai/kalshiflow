"""
Trade processor service that handles incoming trades from Kalshi client.

Coordinates between Kalshi WebSocket client, database storage, 
in-memory aggregation, and WebSocket broadcasting to frontend.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Any, Dict

from .models import Trade, TickerState, TradeUpdateMessage
from .database import get_database
from .aggregator import get_aggregator
from .time_analytics_service import get_analytics_service


logger = logging.getLogger(__name__)


class TradeProcessor:
    """Central service for processing trades from Kalshi client."""
    
    def __init__(self):
        """Initialize the trade processor with database and aggregator."""
        self.database = get_database()
        self.aggregator = get_aggregator()
        self.analytics_service = get_analytics_service()
        self.websocket_broadcaster = None
        self._running = False
        self._trade_callbacks = []
        self._hot_markets_task = None
        self._analytics_task = None
        
        # Analytics broadcast frequency (every 1 second for smooth updates)
        self._analytics_interval = 1.0
        
        # Hot markets broadcast frequency (every 5 seconds)
        self._hot_markets_interval = 5
        
        # Statistics for monitoring
        self.stats = {
            "trades_processed": 0,
            "trades_stored": 0,
            "processing_errors": 0,
            "analytics_broadcasts_sent": 0,
            "analytics_broadcast_errors": 0,
            "last_trade_time": None,
            "started_at": None
        }
        
    
    async def start(self):
        """Start the trade processor service."""
        if self._running:
            return
            
        logger.info("Starting trade processor service...")
        
        # Initialize database
        await self.database.initialize()
        
        # Start aggregator
        await self.aggregator.start()
        
        # Start analytics service
        await self.analytics_service.start()
        
        self._running = True
        self.stats["started_at"] = datetime.now()
        
        # Start hot markets broadcast task
        self._hot_markets_task = asyncio.create_task(self._hot_markets_broadcast_loop())
        
        # Start analytics broadcast task  
        self._analytics_task = asyncio.create_task(self._analytics_broadcast_loop())
        
        logger.info("Trade processor service started successfully")
    
    async def stop(self):
        """Stop the trade processor service."""
        if not self._running:
            return
            
        logger.info("Stopping trade processor service...")
        
        self._running = False
        
        # Stop hot markets broadcast task
        if self._hot_markets_task:
            self._hot_markets_task.cancel()
            try:
                await self._hot_markets_task
            except asyncio.CancelledError:
                pass
            self._hot_markets_task = None
        
        # Stop analytics broadcast task
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
            self._analytics_task = None
        
        # Stop aggregator
        await self.aggregator.stop()
        
        # Stop analytics service
        await self.analytics_service.stop()
        
        logger.info("Trade processor service stopped")
    
    def set_websocket_broadcaster(self, broadcaster):
        """Set the WebSocket broadcaster for real-time updates."""
        self.websocket_broadcaster = broadcaster
    
    def add_trade_callback(self, callback: Callable[[Trade, TickerState], None]):
        """Add callback function to be called when trade is processed."""
        self._trade_callbacks.append(callback)
    
    async def process_trade(self, trade: Trade) -> bool:
        """
        Process a single trade through the complete pipeline.
        
        Returns True if processing succeeded, False otherwise.
        """
        if not self._running:
            logger.warning("Trade processor not running, ignoring trade")
            return False
        
        try:
            logger.debug(f"Processing trade: {trade.market_ticker} {trade.taker_side} {trade.count}@{trade.price_display}")
            
            # Store trade in database
            await self._store_trade(trade)
            
            # Update in-memory aggregations
            ticker_state = None
            try:
                ticker_state = self._update_aggregations(trade)
            except Exception as e:
                logger.error(f"Aggregation failed for trade {trade.market_ticker}, continuing with limited functionality: {e}")
                # Create a minimal ticker state for the broadcast
                ticker_state = TickerState(
                    ticker=trade.market_ticker,
                    last_yes_price=trade.yes_price,
                    last_no_price=trade.no_price,
                    last_trade_time=trade.ts,
                    volume_window=trade.count,
                    trade_count_window=1,
                    yes_flow=trade.count if trade.taker_side == "yes" else 0,
                    no_flow=trade.count if trade.taker_side == "no" else 0,
                    price_points=[trade.yes_price_dollars]
                )
            
            # Update analytics service (broadcasting handled by periodic task)
            try:
                # Process the trade in the analytics service
                self.analytics_service.process_trade(trade)
                logger.debug(f"Processed trade {trade.market_ticker} in analytics service")
                
            except Exception as e:
                logger.error(f"Analytics processing failed for trade {trade.market_ticker}: {e}")
                # Continue processing - analytics failures shouldn't stop trade flow
            
            # Broadcast to WebSocket clients
            await self._broadcast_trade_update(trade, ticker_state)
            
            # Call registered callbacks
            await self._call_trade_callbacks(trade, ticker_state)
            
            # Update statistics
            self.stats["trades_processed"] += 1
            self.stats["last_trade_time"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing trade {trade.market_ticker}: {e}")
            self.stats["processing_errors"] += 1
            return False
    
    async def _store_trade(self, trade: Trade):
        """Store trade in PostgreSQL database."""
        try:
            trade_id = await self.database.insert_trade(trade)
            self.stats["trades_stored"] += 1
            logger.debug(f"Stored trade {trade_id} in database")
        except Exception as e:
            logger.error(f"Failed to store trade in database: {e}")
            raise
    
    def _update_aggregations(self, trade: Trade) -> TickerState:
        """Update in-memory aggregations with new trade."""
        try:
            ticker_state = self.aggregator.process_trade(trade)
            logger.debug(f"Updated aggregations for {trade.market_ticker}")
            return ticker_state
        except Exception as e:
            logger.error(f"Failed to update aggregations for ticker {trade.market_ticker}: {e}")
            logger.debug(f"Trade details: {trade.model_dump()}")
            raise
    
    async def _broadcast_trade_update(self, trade: Trade, ticker_state: TickerState):
        """Broadcast trade update and analytics data to WebSocket clients."""
        if not self.websocket_broadcaster:
            return
        
        try:
            # Get current global stats
            global_stats = self.aggregator.get_global_stats()
            
            # Create lightweight trade update message (no hot markets - too big!)
            # Use dict() instead of model_dump() to include computed properties like net_flow
            ticker_state_dump = ticker_state.dict()
            
            update_message = TradeUpdateMessage(
                type="trade",
                data={
                    "trade": trade.model_dump(),
                    "ticker_state": ticker_state_dump,
                    "global_stats": global_stats
                    # REMOVED: hot_markets - this was causing 88KB messages!
                    # Hot markets updates will be sent separately via periodic broadcasts
                }
            )
            
            # Broadcast trade update to all connected clients
            # The WebSocket broadcaster handles JSON serialization with custom encoder
            message_dict = update_message.model_dump()
            await self.websocket_broadcaster.broadcast(message_dict)
            
            # Note: Analytics data is now broadcast on a 1-second timer via _analytics_broadcast_loop()
            
            logger.debug(f"Broadcast trade update for {trade.market_ticker}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast trade update: {e}")
            # Don't re-raise - broadcasting errors shouldn't stop trade processing
    
    async def _call_trade_callbacks(self, trade: Trade, ticker_state: TickerState):
        """Call all registered trade callbacks."""
        for callback in self._trade_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trade, ticker_state)
                else:
                    callback(trade, ticker_state)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
                # Continue with other callbacks even if one fails
    
    async def get_snapshot_data(self) -> Dict[str, Any]:
        """Get current snapshot data for new WebSocket connections."""
        try:
            recent_trades = self.aggregator.get_recent_trades()
            # Use metadata-enriched hot markets for snapshot
            hot_markets = await self.aggregator.get_hot_markets_with_metadata()
            global_stats = self.aggregator.get_global_stats()
            # Include dual-mode analytics data in snapshot using new service
            hour_mode_data = self.analytics_service.get_mode_data("hour", limit=60)
            day_mode_data = self.analytics_service.get_mode_data("day", limit=24)
            
            # Structure analytics data to match frontend expectations
            analytics_data = {
                "hour_mode": hour_mode_data,
                "day_mode": day_mode_data
            }
            
            return {
                "recent_trades": recent_trades,
                "hot_markets": hot_markets,
                "global_stats": global_stats,
                "analytics_data": analytics_data
            }
        except Exception as e:
            logger.error(f"Error getting snapshot data: {e}")
            return {
                "recent_trades": [],
                "hot_markets": [],
                "global_stats": {
                    "daily_trades_count": 0,
                    "session_start_time": None,
                    "active_markets_count": 0,
                    "total_window_volume": 0
                },
                "analytics_data": {
                    "hour_mode": {
                        "current_period": {
                            "timestamp": 0,
                            "volume_usd": 0.0,
                            "trade_count": 0
                        },
                        "summary_stats": {
                            "total_volume_usd": 0.0,
                            "total_trades": 0,
                            "peak_volume_usd": 0.0,
                            "peak_trades": 0
                        },
                        "time_series": [],
                        "mode": "hour"
                    },
                    "day_mode": {
                        "current_period": {
                            "timestamp": 0,
                            "volume_usd": 0.0,
                            "trade_count": 0
                        },
                        "summary_stats": {
                            "total_volume_usd": 0.0,
                            "total_trades": 0,
                            "peak_volume_usd": 0.0,
                            "peak_trades": 0
                        },
                        "time_series": [],
                        "mode": "day"
                    }
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trade processor statistics."""
        runtime_seconds = None
        if self.stats["started_at"]:
            runtime_seconds = (datetime.now() - self.stats["started_at"]).total_seconds()
        
        processor_stats = {
            **self.stats,
            "runtime_seconds": runtime_seconds,
            "is_running": self._running,
            "callbacks_count": len(self._trade_callbacks),
            "has_websocket_broadcaster": self.websocket_broadcaster is not None
        }
        
        # Include aggregator and analytics stats
        processor_stats["aggregator"] = self.aggregator.get_stats()
        processor_stats["analytics"] = self.analytics_service.get_stats()
        
        return processor_stats
    
    async def handle_kalshi_trade_message(self, trade_message: Dict[str, Any]) -> bool:
        """
        Handle a trade message from Kalshi WebSocket client.
        
        Converts raw message to Trade object and processes it.
        """
        try:
            # Convert raw message to Trade object  
            # Note: Timestamp conversion from seconds to milliseconds is handled in TradeMessage.to_trade()
            trade = Trade(
                market_ticker=trade_message["market_ticker"],
                yes_price=trade_message["yes_price"],
                no_price=trade_message["no_price"],
                yes_price_dollars=trade_message["yes_price"] / 100.0,
                no_price_dollars=trade_message["no_price"] / 100.0,
                count=trade_message["count"],
                taker_side=trade_message["taker_side"],
                ts=trade_message["ts"]
            )
            
            # Process the trade
            return await self.process_trade(trade)
            
        except Exception as e:
            logger.error(f"Error handling Kalshi trade message: {e}")
            logger.error(f"Raw message: {trade_message}")
            return False
    
    async def _hot_markets_broadcast_loop(self):
        """Periodically broadcast hot markets updates to all connected clients."""
        try:
            # Wait a bit before starting to ensure aggregator has data
            await asyncio.sleep(5)
            
            while self._running:
                try:
                    if self.websocket_broadcaster:
                        # Get fresh hot markets with metadata
                        hot_markets = await self.aggregator.get_hot_markets_with_metadata()
                        
                        # Create hot markets update message
                        message = {
                            "type": "hot_markets_update",
                            "data": {
                                "hot_markets": hot_markets
                            }
                        }
                        
                        # Broadcast to all connected clients
                        await self.websocket_broadcaster.broadcast(message)
                        
                        logger.debug(f"Broadcast hot markets update: {len(hot_markets)} markets")
                    
                    # Wait for next update
                    await asyncio.sleep(self._hot_markets_interval)
                    
                except Exception as e:
                    logger.error(f"Error in hot markets broadcast loop: {e}")
                    # Don't break the loop for individual errors
                    await asyncio.sleep(self._hot_markets_interval)
                    
        except asyncio.CancelledError:
            logger.debug("Hot markets broadcast loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in hot markets broadcast loop: {e}")
    
    async def _analytics_broadcast_loop(self):
        """Periodically broadcast analytics updates for smooth real-time updates."""
        try:
            # Wait a bit before starting to ensure analytics service has data
            logger.info("Analytics broadcast loop starting (waiting 2 seconds)")
            await asyncio.sleep(2)
            
            logger.info(f"Analytics broadcast loop running, websocket_broadcaster: {self.websocket_broadcaster is not None}")
            
            while self._running:
                try:
                    if self.websocket_broadcaster:
                        # Send analytics updates for both modes every second for smooth updates
                        
                        # Hour mode analytics update (current minute data + 60-minute window)
                        hour_analytics = self.analytics_service.get_mode_data("hour", limit=60)
                        await self.websocket_broadcaster.broadcast_analytics_update(hour_analytics)
                        
                        # Day mode analytics update (current hour data + 24-hour window)
                        day_analytics = self.analytics_service.get_mode_data("day", limit=24)
                        await self.websocket_broadcaster.broadcast_analytics_update(day_analytics)
                        
                        # Update stats
                        self.stats["analytics_broadcasts_sent"] += 2  # Two modes sent
                        
                        logger.info("Sent periodic analytics updates for both hour and day modes")
                    else:
                        logger.debug("Analytics broadcast loop waiting for websocket broadcaster to be set")
                    
                    # Wait for next update (1 second for smooth incremental updates)
                    await asyncio.sleep(self._analytics_interval)
                    
                except Exception as e:
                    logger.error(f"Error in analytics broadcast loop: {e}")
                    self.stats["analytics_broadcast_errors"] += 1
                    # Don't break the loop for individual errors
                    await asyncio.sleep(self._analytics_interval)
                    
        except asyncio.CancelledError:
            logger.debug("Analytics broadcast loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in analytics broadcast loop: {e}")


# Global trade processor instance
_processor_instance = None

def get_trade_processor() -> TradeProcessor:
    """Get the global trade processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TradeProcessor()
    return _processor_instance