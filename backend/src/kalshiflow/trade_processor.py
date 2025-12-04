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


logger = logging.getLogger(__name__)


class TradeProcessor:
    """Central service for processing trades from Kalshi client."""
    
    def __init__(self):
        """Initialize the trade processor with database and aggregator."""
        self.database = get_database()
        self.aggregator = get_aggregator()
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
    
    async def start(self):
        """Start the trade processor service."""
        if self._running:
            return
            
        logger.info("Starting trade processor service...")
        
        # Initialize database
        await self.database.initialize()
        
        # Start aggregator
        await self.aggregator.start()
        
        self._running = True
        self.stats["started_at"] = datetime.now()
        
        logger.info("Trade processor service started successfully")
    
    async def stop(self):
        """Stop the trade processor service."""
        if not self._running:
            return
            
        logger.info("Stopping trade processor service...")
        
        self._running = False
        
        # Stop aggregator
        await self.aggregator.stop()
        
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
        """Store trade in SQLite database."""
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
            logger.debug(f"Trade details: {trade.dict()}")
            raise
    
    async def _broadcast_trade_update(self, trade: Trade, ticker_state: TickerState):
        """Broadcast trade update to WebSocket clients."""
        if not self.websocket_broadcaster:
            return
        
        try:
            # Create trade update message
            update_message = TradeUpdateMessage(
                type="trade",
                data={
                    "trade": trade.dict(),
                    "ticker_state": ticker_state.dict()
                }
            )
            
            # Broadcast to all connected clients
            await self.websocket_broadcaster.broadcast(update_message.dict())
            
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
            hot_markets = self.aggregator.get_hot_markets()
            
            return {
                "recent_trades": recent_trades,
                "hot_markets": hot_markets
            }
        except Exception as e:
            logger.error(f"Error getting snapshot data: {e}")
            return {
                "recent_trades": [],
                "hot_markets": []
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
        
        # Include aggregator and database stats
        processor_stats["aggregator"] = self.aggregator.get_stats()
        
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


# Global trade processor instance
_processor_instance = None

def get_trade_processor() -> TradeProcessor:
    """Get the global trade processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TradeProcessor()
    return _processor_instance