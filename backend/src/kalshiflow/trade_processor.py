"""
Trade processor service that handles incoming trades from Kalshi client.

Coordinates between Kalshi WebSocket client, in-memory aggregation,
and WebSocket broadcasting to frontend. No persistent storage.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Any, Dict, List

from .models import Trade, TickerState, TradeUpdateMessage
from .aggregator import get_aggregator
from .time_analytics_service import get_analytics_service


logger = logging.getLogger(__name__)


class TradeProcessor:
    """Central service for processing trades from Kalshi client."""
    
    def __init__(self):
        """Initialize the trade processor with in-memory aggregator."""
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
        
        # Top trades broadcast task
        self._top_trades_task = None
        self._top_trades_interval = 60  # Update every minute
        
        # Trade broadcast batching for WebSocket efficiency
        self._trade_broadcast_batch = []
        self._trade_batch_lock = asyncio.Lock()
        self._trade_batch_task = None
        self._trade_batch_interval = 0.75  # Flush trades every 750ms
        self._trade_batch_max_size = 10  # Or when batch reaches 10 trades
        
        # Statistics for monitoring
        self.stats = {
            "trades_processed": 0,
            "processing_errors": 0,
            "analytics_broadcasts_sent": 0,
            "analytics_broadcast_errors": 0,
            "last_trade_time": None,
            "started_at": None,
        }
        
    
    async def start(self):
        """Start the trade processor service."""
        if self._running:
            return
            
        logger.info("Starting trade processor service...")
        
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
        
        # Start top trades broadcast task
        self._top_trades_task = asyncio.create_task(self._top_trades_broadcast_loop())
        
        # Start trade batch flush task
        self._trade_batch_task = asyncio.create_task(self._trade_batch_flush_loop())
        
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
        
        # Stop top trades broadcast task
        if self._top_trades_task:
            self._top_trades_task.cancel()
            try:
                await self._top_trades_task
            except asyncio.CancelledError:
                pass
            self._top_trades_task = None
        
        # Stop trade batch task and flush remaining trades
        if self._trade_batch_task:
            self._trade_batch_task.cancel()
            try:
                await self._trade_batch_task
            except asyncio.CancelledError:
                pass
            self._trade_batch_task = None
            
            # Flush any remaining trades in the batch
            async with self._trade_batch_lock:
                if self._trade_broadcast_batch:
                    await self._flush_trade_batch()
        
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
        Process a single trade through the complete pipeline with immediate analytics updates.
        
        This method prioritizes real-time analytics by:
        1. Immediately updating in-memory aggregations and analytics
        2. Broadcasting updates to WebSocket clients
        
        Returns True if processing succeeded, False otherwise.
        """
        if not self._running:
            logger.warning("Trade processor not running, ignoring trade")
            return False
        
        try:
            logger.debug(f"Processing trade: {trade.market_ticker} {trade.taker_side} {trade.count}@{trade.price_display}")
            
            # IMMEDIATE ANALYTICS: Update in-memory aggregations first (no blocking)
            ticker_state = None
            try:
                ticker_state = self._update_aggregations(trade)
                if ticker_state is None:
                    # Duplicate detected - stop processing this trade
                    logger.info(f"Skipping duplicate trade for {trade.market_ticker}")
                    # Don't count duplicates in statistics
                    return True  # Return True since this isn't an error
                logger.debug(f"Updated aggregations for {trade.market_ticker} immediately")
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
            
            # IMMEDIATE ANALYTICS: Update analytics service (no blocking)
            try:
                self.analytics_service.process_trade(trade)
                logger.debug(f"Processed trade {trade.market_ticker} in analytics service immediately")
                
            except Exception as e:
                logger.error(f"Analytics processing failed for trade {trade.market_ticker}: {e}")
                # Continue processing - analytics failures shouldn't stop trade flow
            
            # IMMEDIATE BROADCAST: Send updates to WebSocket clients immediately
            await self._broadcast_trade_update(trade, ticker_state)
            
            # Check if this trade might affect top trades list
            if self.aggregator.should_broadcast_top_trades(trade):
                # Trigger immediate top trades update
                asyncio.create_task(self._broadcast_top_trades_immediate())
            
            # IMMEDIATE CALLBACKS: Call registered callbacks immediately
            await self._call_trade_callbacks(trade, ticker_state)
            
            # Update statistics
            self.stats["trades_processed"] += 1
            self.stats["last_trade_time"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing trade {trade.market_ticker}: {e}")
            self.stats["processing_errors"] += 1
            return False
    
    def _update_aggregations(self, trade: Trade) -> Optional[TickerState]:
        """Update in-memory aggregations with new trade.
        
        Returns:
            TickerState if trade was processed (not duplicate), None if duplicate detected
        """
        try:
            ticker_state = self.aggregator.process_trade(trade)
            if ticker_state is None:
                # Duplicate detected - this is handled by the aggregator
                logger.debug(f"Duplicate trade detected for {trade.market_ticker}, skipping aggregation")
                return None
            logger.debug(f"Updated aggregations for {trade.market_ticker}")
            return ticker_state
        except Exception as e:
            logger.error(f"Failed to update aggregations for ticker {trade.market_ticker}: {e}")
            logger.debug(f"Trade details: {trade.model_dump()}")
            raise
    
    async def _broadcast_trade_update(self, trade: Trade, ticker_state: TickerState):
        """Add trade to batch for efficient WebSocket broadcasting."""
        if not self.websocket_broadcaster:
            return
        
        try:
            # Create minimal trade dict without market_ticker (saves ~30-50 bytes per message)
            minimal_trade = {
                "yes_price": trade.yes_price,
                "no_price": trade.no_price,
                "count": trade.count,
                "taker_side": trade.taker_side,
                "ts": trade.ts
            }
            
            # Add to batch instead of broadcasting immediately
            async with self._trade_batch_lock:
                self._trade_broadcast_batch.append(minimal_trade)
                
                # Flush if batch is full
                if len(self._trade_broadcast_batch) >= self._trade_batch_max_size:
                    await self._flush_trade_batch()
            
            logger.debug(f"Added trade to batch for {trade.market_ticker}")
            
        except Exception as e:
            logger.error(f"Failed to add trade to batch: {e}")
            # Don't re-raise - broadcasting errors shouldn't stop trade processing
    
    async def _flush_trade_batch(self):
        """Flush the current batch of trades to WebSocket clients."""
        if not self._trade_broadcast_batch:
            return
        
        try:
            # Create batched message with all trades
            batch_message = {
                "type": "trades",  # New message type for batched trades
                "data": {
                    "trades": self._trade_broadcast_batch.copy()
                }
            }
            
            # Broadcast the batch
            await self.websocket_broadcaster.broadcast(batch_message)
            logger.debug(f"Flushed batch of {len(self._trade_broadcast_batch)} trades")
            
            # Clear the batch
            self._trade_broadcast_batch.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush trade batch: {e}")
    
    async def _trade_batch_flush_loop(self):
        """Periodically flush trade batches to ensure timely delivery."""
        while self._running:
            try:
                await asyncio.sleep(self._trade_batch_interval)
                
                async with self._trade_batch_lock:
                    if self._trade_broadcast_batch:
                        await self._flush_trade_batch()
                        
            except Exception as e:
                logger.error(f"Error in trade batch flush loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
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
            global_stats = self.aggregator.get_global_stats()
            top_trades = await self.aggregator.get_top_trades()
            hour_mode_data = self.analytics_service.get_mode_data("hour", limit=60)
            
            analytics_data = {
                "hour_mode": hour_mode_data,
            }
            
            return {
                "recent_trades": recent_trades,
                "hot_markets": [],
                "global_stats": global_stats,
                "analytics_data": analytics_data,
                "top_trades": top_trades
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
            "has_websocket_broadcaster": self.websocket_broadcaster is not None,
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
    
    async def _broadcast_top_trades_immediate(self):
        """Broadcast top trades immediately when a significant trade is detected."""
        try:
            if self.websocket_broadcaster:
                # Force update and get fresh top trades
                top_trades = await self.aggregator.update_top_trades(force=True)
                
                # Import TopTradesMessage
                from .models import TopTradesMessage
                
                # Create top trades message
                top_trades_message = TopTradesMessage(
                    trades=top_trades,
                    window_minutes=self.aggregator._top_trades_window_minutes
                )
                
                # Broadcast to all connected clients
                await self.websocket_broadcaster.broadcast(top_trades_message.model_dump())
                
                logger.debug(f"Immediate broadcast of top trades: {len(top_trades)} trades")
                
        except Exception as e:
            logger.error(f"Error in immediate top trades broadcast: {e}")
    
    async def _top_trades_broadcast_loop(self):
        """Periodically broadcast top trades by volume to all connected clients."""
        try:
            # Wait a bit before starting to ensure aggregator has data
            await asyncio.sleep(10)
            
            while self._running:
                try:
                    if self.websocket_broadcaster:
                        # Get top trades from aggregator
                        top_trades = await self.aggregator.get_top_trades()
                        
                        # Import TopTradesMessage
                        from .models import TopTradesMessage
                        
                        # Create top trades message
                        top_trades_message = TopTradesMessage(
                            trades=top_trades, 
                            window_minutes=self.aggregator._top_trades_window_minutes
                        )
                        
                        # Broadcast to all connected clients
                        await self.websocket_broadcaster.broadcast(top_trades_message.model_dump())
                        
                        logger.debug(f"Broadcast top trades update: {len(top_trades)} trades")
                    
                    # Wait for next update
                    await asyncio.sleep(self._top_trades_interval)
                    
                except Exception as e:
                    logger.error(f"Error in top trades broadcast loop: {e}")
                    # Don't break the loop for individual errors
                    await asyncio.sleep(self._top_trades_interval)
                    
        except asyncio.CancelledError:
            logger.debug("Top trades broadcast loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in top trades broadcast loop: {e}")
    
    async def _analytics_broadcast_loop(self):
        """
        Periodically broadcast analytics updates for smooth real-time frontend updates.
        
        This method runs continuously as a background task, sending analytics updates
        every 1 second to provide smooth, incremental data changes to connected WebSocket
        clients. The loop broadcasts both hour-mode and day-mode analytics data.
        
        Features:
        - 1-second broadcast frequency for smooth UI animations
        - Dual-mode broadcasting (hour and day analytics)
        - Robust error handling without breaking the loop
        - Graceful cancellation support
        - Statistics tracking for monitoring
        
        Broadcast format:
        - Each mode is sent as a separate 'analytics_update' message
        - Messages contain: mode, current_period, summary_stats, time_series
        - Hour mode: Current minute data + 60-minute historical window
        - Day mode: Current hour data + 24-hour historical window
        
        Error handling:
        - Individual broadcast errors don't break the loop
        - Errors are logged and statistics are tracked
        - Loop continues with next scheduled broadcast
        
        Performance considerations:
        - Lightweight data structures to minimize bandwidth
        - Single-pass data processing in analytics service
        - Efficient JSON serialization by WebSocket broadcaster
        
        Lifecycle:
        - Started automatically when TradeProcessor.start() is called
        - Cancelled and cleaned up when TradeProcessor.stop() is called
        - Waits 2 seconds before starting to allow analytics service initialization
        """
        try:
            # Wait a bit before starting to ensure analytics service has data
            logger.debug("Analytics broadcast loop starting (waiting 2 seconds)")
            await asyncio.sleep(2)
            
            logger.debug(f"Analytics broadcast loop running, websocket_broadcaster: {self.websocket_broadcaster is not None}")
            
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
                        
                        logger.debug("Sent periodic analytics updates for both hour and day modes")
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