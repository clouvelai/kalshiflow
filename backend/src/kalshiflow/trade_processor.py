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
        
        # Statistics for monitoring
        self.stats = {
            "trades_processed": 0,
            "trades_stored": 0,
            "processing_errors": 0,
            "last_trade_time": None,
            "started_at": None
        }
        
        # Analytics broadcasting task
        self._analytics_task = None
    
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
        
        # Start analytics broadcast task (every 1 second for real-time feel)
        self._analytics_task = asyncio.create_task(self._analytics_broadcast_loop())
        
        logger.info("Trade processor service started successfully")
    
    async def stop(self):
        """Stop the trade processor service."""
        if not self._running:
            return
            
        logger.info("Stopping trade processor service...")
        
        self._running = False
        
        # Stop analytics broadcast task
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
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
            
            # Update analytics (non-blocking)
            try:
                self.analytics_service.process_trade(trade)
                
                # REALTIME UPDATE: Broadcast current period data everywhere it appears
                # This provides instant updates for current stats, chart current bar, totals, and peaks
                if self.websocket_broadcaster:
                    realtime_data = self.analytics_service.get_realtime_update_data(trade.ts)
                    await self.websocket_broadcaster.broadcast_realtime_update(realtime_data)
                    
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
            
            # Get hot markets with metadata to ensure metadata is preserved in real-time updates
            hot_markets = await self.aggregator.get_hot_markets_with_metadata()
            
            # Create trade update message
            # Use dict() instead of model_dump() to include computed properties like net_flow
            ticker_state_dump = ticker_state.dict()
            
            update_message = TradeUpdateMessage(
                type="trade",
                data={
                    "trade": trade.model_dump(),
                    "ticker_state": ticker_state_dump,
                    "global_stats": global_stats,
                    "hot_markets": hot_markets  # Include metadata-enriched hot markets
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
            # Include dual-mode analytics data in snapshot
            analytics_data = self.analytics_service.get_analytics_data()
            
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
                    "hour_minute_mode": {
                        "time_series": [],
                        "summary_stats": {
                            "peak_volume_usd": 0.0,
                            "total_volume_usd": 0.0,
                            "peak_trades": 0,
                            "total_trades": 0,
                            "current_minute_volume_usd": 0.0,
                            "current_minute_trades": 0
                        }
                    },
                    "day_hour_mode": {
                        "time_series": [],
                        "summary_stats": {
                            "peak_volume_usd": 0.0,
                            "total_volume_usd": 0.0,
                            "peak_trades": 0,
                            "total_trades": 0,
                            "current_hour_volume_usd": 0.0,
                            "current_hour_trades": 0
                        }
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
    
    async def _analytics_broadcast_loop(self):
        """Background task to broadcast chart data every 60 seconds.
        
        Optimized frequency: Historical chart data doesn't need sub-second updates.
        Current period stats are handled by realtime_update messages on every trade.
        This clean separation prevents data conflicts and optimizes bandwidth usage.
        """
        try:
            logger.info("Started chart data broadcast task (60-second interval for historical chart data)")
            
            while self._running:
                try:
                    # Only broadcast if we have a websocket broadcaster
                    if self.websocket_broadcaster:
                        # Use chart data for historical chart bars only (excluding current period)
                        chart_data = self.analytics_service.get_chart_data()
                        if chart_data:
                            await self.websocket_broadcaster.broadcast_chart_data(chart_data)
                            logger.debug("Broadcast chart data (60-second timer) - historical bars only")
                    
                    # Wait 60 seconds - historical chart data doesn't need frequent updates
                    await asyncio.sleep(60.0)
                    
                except Exception as e:
                    logger.error(f"Error in analytics broadcast loop: {e}")
                    # Continue - don't let broadcast errors stop the loop
                    await asyncio.sleep(60.0)
                    
        except asyncio.CancelledError:
            logger.info("Analytics broadcast task cancelled")
        except Exception as e:
            logger.error(f"Analytics broadcast task failed: {e}")


# Global trade processor instance
_processor_instance = None

def get_trade_processor() -> TradeProcessor:
    """Get the global trade processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TradeProcessor()
    return _processor_instance