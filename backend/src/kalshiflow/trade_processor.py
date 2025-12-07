"""
Trade processor service that handles incoming trades from Kalshi client.

Coordinates between Kalshi WebSocket client, database storage, 
in-memory aggregation, and WebSocket broadcasting to frontend.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Any, Dict, List

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
        # This high frequency enables smooth counter animations and real-time chart updates
        # on the frontend without overwhelming network bandwidth
        self._analytics_interval = 1.0
        
        # Hot markets broadcast frequency (every 5 seconds)
        self._hot_markets_interval = 5
        
        # Write queue system for non-blocking database writes
        self._write_queue = asyncio.Queue(maxsize=10000)  # Large queue to handle bursts
        self._write_worker_task = None
        self._write_batch_size = 25  # Process in batches for efficiency
        self._write_timeout_ms = 100  # Max wait time for batching (100ms)
        self._write_retry_attempts = 3  # Number of retry attempts for failed writes
        
        # Statistics for monitoring
        self.stats = {
            "trades_processed": 0,
            "trades_stored": 0,
            "processing_errors": 0,
            "analytics_broadcasts_sent": 0,
            "analytics_broadcast_errors": 0,
            "last_trade_time": None,
            "started_at": None,
            "queued_writes": 0,
            "batch_writes_executed": 0,
            "write_errors": 0,
            "write_queue_size": 0,
            "max_queue_size": 0
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
        
        # Start write worker task for background database writes
        self._write_worker_task = asyncio.create_task(self._write_worker_loop())
        
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
        
        # Stop write worker and flush remaining trades
        await self._shutdown_write_worker()
        
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
        3. Queueing database writes for background processing
        
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
            
            # IMMEDIATE CALLBACKS: Call registered callbacks immediately
            await self._call_trade_callbacks(trade, ticker_state)
            
            # BACKGROUND WRITE: Queue trade for background database write (non-blocking)
            await self._queue_trade_for_write(trade)
            
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
    
    async def _queue_trade_for_write(self, trade: Trade):
        """Queue a trade for background database write (non-blocking)."""
        try:
            # Add trade to the write queue (non-blocking if space available)
            self._write_queue.put_nowait(trade)
            self.stats["queued_writes"] += 1
            
            # Update queue size statistics
            current_queue_size = self._write_queue.qsize()
            self.stats["write_queue_size"] = current_queue_size
            if current_queue_size > self.stats["max_queue_size"]:
                self.stats["max_queue_size"] = current_queue_size
            
            logger.debug(f"Queued trade for background write: {trade.market_ticker} (queue size: {current_queue_size})")
            
        except asyncio.QueueFull:
            # Queue is full - this is a critical situation, try to write immediately
            logger.warning(f"Write queue full ({self._write_queue.maxsize}), writing trade {trade.market_ticker} immediately")
            await self._store_trade(trade)
        except Exception as e:
            logger.error(f"Failed to queue trade for write: {e}")
            # Fallback to immediate write
            await self._store_trade(trade)
    
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
            "has_websocket_broadcaster": self.websocket_broadcaster is not None,
            "write_queue_stats": {
                "current_queue_size": self._write_queue.qsize(),
                "max_queue_size": self.stats["max_queue_size"],
                "queued_writes": self.stats["queued_writes"],
                "batch_writes_executed": self.stats["batch_writes_executed"],
                "write_errors": self.stats["write_errors"],
                "batch_size": self._write_batch_size,
                "timeout_ms": self._write_timeout_ms
            }
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
    
    async def _write_worker_loop(self):
        """
        Background worker loop for batch database writes.
        
        This worker continuously processes queued trades in batches for efficiency.
        Features:
        - Batch processing (up to _write_batch_size trades or _write_timeout_ms timeout)
        - Automatic retry with exponential backoff for failed writes
        - Graceful handling of individual trade write failures
        - Performance statistics tracking
        
        The worker optimizes database writes by:
        1. Collecting trades into batches to reduce database round trips
        2. Using PostgreSQL batch insert for maximum efficiency
        3. Handling failures gracefully without blocking the queue
        """
        logger.debug("Write worker loop starting")
        
        try:
            while self._running:
                try:
                    # Collect trades for batch processing
                    batch_trades = await self._collect_write_batch()
                    
                    if batch_trades:
                        # Process the batch with retry logic
                        await self._process_write_batch(batch_trades)
                    
                except Exception as e:
                    logger.error(f"Error in write worker loop: {e}")
                    # Don't break the loop for individual errors
                    await asyncio.sleep(0.1)  # Brief pause before retrying
                    
        except asyncio.CancelledError:
            logger.debug("Write worker loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in write worker loop: {e}")
    
    async def _collect_write_batch(self) -> List[Trade]:
        """Collect trades from the queue for batch processing."""
        batch = []
        batch_start_time = asyncio.get_event_loop().time()
        timeout_seconds = self._write_timeout_ms / 1000.0
        
        try:
            # Get the first trade (wait up to timeout)
            trade = await asyncio.wait_for(self._write_queue.get(), timeout=timeout_seconds)
            batch.append(trade)
            
            # Collect additional trades up to batch size (non-blocking)
            while len(batch) < self._write_batch_size:
                try:
                    # Check if we've exceeded the timeout
                    elapsed = asyncio.get_event_loop().time() - batch_start_time
                    if elapsed >= timeout_seconds:
                        break
                    
                    # Try to get another trade immediately (non-blocking)
                    trade = self._write_queue.get_nowait()
                    batch.append(trade)
                    
                except asyncio.QueueEmpty:
                    # No more trades available immediately
                    break
            
            logger.debug(f"Collected batch of {len(batch)} trades for writing")
            return batch
            
        except asyncio.TimeoutError:
            # No trades available within timeout
            return []
        except Exception as e:
            logger.error(f"Error collecting write batch: {e}")
            return []
    
    async def _process_write_batch(self, batch_trades: List[Trade]):
        """Process a batch of trades with retry logic."""
        if not batch_trades:
            return
        
        retry_count = 0
        while retry_count < self._write_retry_attempts:
            try:
                # Perform batch insert
                trade_ids = await self.database.insert_trades_batch(batch_trades)
                
                # Update statistics
                self.stats["trades_stored"] += len(batch_trades)
                self.stats["batch_writes_executed"] += 1
                
                # Update queue size stat
                self.stats["write_queue_size"] = self._write_queue.qsize()
                
                logger.debug(f"Successfully wrote batch of {len(batch_trades)} trades (IDs: {trade_ids[0]}-{trade_ids[-1]})")
                return  # Success - exit retry loop
                
            except Exception as e:
                retry_count += 1
                
                # Check if this is a duplicate key error (expected during startup/recovery)
                if "duplicate key value violates unique constraint" in str(e):
                    logger.debug(f"Batch write detected {len(batch_trades)} duplicate trades (expected during recovery), skipping batch")
                    # Don't count duplicates as errors - they're expected during startup
                    return
                
                self.stats["write_errors"] += 1
                
                if retry_count < self._write_retry_attempts:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    retry_delay = 0.1 * (2 ** (retry_count - 1))
                    logger.warning(f"Batch write failed (attempt {retry_count}/{self._write_retry_attempts}), retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    # Final attempt failed - log critical error
                    logger.error(f"Batch write failed after {self._write_retry_attempts} attempts, dropping {len(batch_trades)} trades: {e}")
                    # Note: In production, you might want to write these to a dead letter queue
                    break
    
    async def _shutdown_write_worker(self):
        """Gracefully shutdown the write worker and flush remaining trades."""
        logger.info("Shutting down write worker and flushing remaining trades...")
        
        # Cancel the worker task
        if self._write_worker_task:
            self._write_worker_task.cancel()
            try:
                await self._write_worker_task
            except asyncio.CancelledError:
                pass
            self._write_worker_task = None
        
        # Flush any remaining trades in the queue
        remaining_trades = []
        try:
            while True:
                trade = self._write_queue.get_nowait()
                remaining_trades.append(trade)
        except asyncio.QueueEmpty:
            pass
        
        if remaining_trades:
            logger.info(f"Flushing {len(remaining_trades)} remaining trades from write queue...")
            try:
                # Process remaining trades in batches
                for i in range(0, len(remaining_trades), self._write_batch_size):
                    batch = remaining_trades[i:i + self._write_batch_size]
                    await self._process_write_batch(batch)
                
                logger.info(f"Successfully flushed {len(remaining_trades)} trades")
            except Exception as e:
                logger.error(f"Error flushing remaining trades: {e}")
        else:
            logger.info("No remaining trades to flush")


# Global trade processor instance
_processor_instance = None

def get_trade_processor() -> TradeProcessor:
    """Get the global trade processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TradeProcessor()
    return _processor_instance