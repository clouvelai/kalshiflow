"""
In-memory trade aggregation and ticker state management for real-time market data.
"""

import os
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import heapq

from .models import Trade, TickerState


class TradeAggregator:
    """Manages real-time trade aggregation with sliding window calculations."""
    
    def __init__(self, window_minutes: int = None, max_price_points: int = 50):
        """Initialize aggregator with time window settings."""
        self.window_minutes = window_minutes or int(os.getenv("WINDOW_MINUTES", "10"))
        self.max_price_points = max_price_points
        
        # Data structures for efficient aggregation
        self.ticker_states: Dict[str, TickerState] = {}
        self.trades_by_ticker: Dict[str, deque] = defaultdict(deque)
        self.trades_by_time: List[Tuple[int, int, str, Trade]] = []  # heap of (timestamp, seq, ticker, trade)
        self.recent_trades: deque = deque(maxlen=int(os.getenv("RECENT_TRADES_LIMIT", "200")))
        
        # Global statistics tracking
        self._daily_trades_count = 0
        self._total_volume = 0
        self._total_net_flow = 0
        self._session_start_time = None
        
        # Sequence number for heap ordering
        self._trade_sequence = 0
        
        # Cleanup task
        self._cleanup_task = None
        self._running = False
        
    async def start(self):
        """Start the aggregator and cleanup task."""
        if self._running:
            return
            
        self._running = True
        self._session_start_time = datetime.now()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def stop(self):
        """Stop the aggregator and cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def process_trade(self, trade: Trade) -> TickerState:
        """Process a new trade and update aggregations."""
        ticker = trade.market_ticker
        
        # Increment global daily trades count
        self._daily_trades_count += 1
        
        # Update global volume (trade volume in shares)
        self._total_volume += trade.count
        
        # Update global net flow (YES trades are positive, NO trades are negative)
        if trade.taker_side == "yes":
            self._total_net_flow += trade.count
        else:
            self._total_net_flow -= trade.count
        
        # Add to recent trades (newest first)
        self.recent_trades.appendleft(trade.dict())
        
        # Add to ticker-specific trade queue
        ticker_trades = self.trades_by_ticker[ticker]
        ticker_trades.append(trade)
        
        # Add to global time-ordered heap for cleanup (with sequence number to avoid comparison issues)
        self._trade_sequence += 1
        heapq.heappush(self.trades_by_time, (trade.ts, self._trade_sequence, ticker, trade))
        
        # Update or create ticker state
        ticker_state = self._update_ticker_state(ticker, trade)
        
        return ticker_state
    
    def _update_ticker_state(self, ticker: str, new_trade: Trade) -> TickerState:
        """Update aggregated state for a ticker with new trade."""
        # Get current window cutoff
        window_start = int((datetime.now().timestamp() - self.window_minutes * 60) * 1000)
        
        # Get existing state or create new one
        if ticker in self.ticker_states:
            state = self.ticker_states[ticker]
        else:
            state = TickerState(
                ticker=ticker,
                last_yes_price=new_trade.yes_price,
                last_no_price=new_trade.no_price,
                last_trade_time=new_trade.ts,
                volume_window=0,
                trade_count_window=0,
                yes_flow=0,
                no_flow=0,
                price_points=[]
            )
        
        # Update latest prices and time
        state.last_yes_price = new_trade.yes_price
        state.last_no_price = new_trade.no_price
        state.last_trade_time = new_trade.ts
        
        # Add new enhanced price point with volume and timestamp for advanced visualizations
        price_point = {
            "price": new_trade.yes_price_dollars,
            "volume": new_trade.count,
            "timestamp": new_trade.ts,
            "side": new_trade.taker_side  # yes/no for momentum calculation
        }
        state.price_points.append(price_point)
        if len(state.price_points) > self.max_price_points:
            state.price_points.pop(0)
        
        # Store updated state first
        self.ticker_states[ticker] = state
        
        # Recalculate window aggregates
        self._recalculate_window_stats(ticker, window_start)
        
        return state
    
    def _recalculate_window_stats(self, ticker: str, window_start: int):
        """Recalculate aggregated statistics for a ticker within the time window."""
        # Check if ticker state exists (defensive programming)
        if ticker not in self.ticker_states:
            return
        
        ticker_trades = self.trades_by_ticker[ticker]
        state = self.ticker_states[ticker]
        
        # Reset window stats
        volume_window = 0
        trade_count_window = 0
        yes_flow = 0
        no_flow = 0
        
        # Aggregate trades within window
        for trade in ticker_trades:
            if trade.ts >= window_start:
                volume_window += trade.count
                trade_count_window += 1
                
                if trade.taker_side == "yes":
                    yes_flow += trade.count
                else:
                    no_flow += trade.count
        
        # Update state with new aggregates
        state.volume_window = volume_window
        state.trade_count_window = trade_count_window
        state.yes_flow = yes_flow
        state.no_flow = no_flow
    
    def get_hot_markets(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get markets sorted by volume in current window."""
        limit = limit or int(os.getenv("HOT_MARKETS_LIMIT", "12"))
        
        # Sort tickers by volume (descending)
        sorted_tickers = sorted(
            self.ticker_states.values(),
            key=lambda state: state.volume_window,
            reverse=True
        )
        
        return [state.dict() for state in sorted_tickers[:limit]]
    
    def get_recent_trades(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get recent trades across all markets."""
        limit = limit or int(os.getenv("RECENT_TRADES_LIMIT", "200"))
        
        # Recent trades are already stored newest first
        return list(self.recent_trades)[:limit]
    
    def get_ticker_state(self, ticker: str) -> Optional[TickerState]:
        """Get current state for a specific ticker."""
        return self.ticker_states.get(ticker)
    
    def get_all_ticker_states(self) -> Dict[str, TickerState]:
        """Get all current ticker states."""
        return self.ticker_states.copy()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global market statistics."""
        return {
            "daily_trades_count": self._daily_trades_count,
            "total_volume": self._total_volume,
            "total_net_flow": self._total_net_flow,
            "session_start_time": self._session_start_time.isoformat() if self._session_start_time else None,
            "active_markets_count": len(self.ticker_states),
            "total_window_volume": sum(state.volume_window for state in self.ticker_states.values())
        }
    
    async def _periodic_cleanup(self):
        """Periodically remove old trades from memory."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._prune_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error during periodic cleanup: {e}")
    
    async def _prune_old_data(self):
        """Remove trades older than the window from memory."""
        window_start = int((datetime.now().timestamp() - self.window_minutes * 60) * 1000)
        
        # Remove old trades from time-ordered heap
        while self.trades_by_time and self.trades_by_time[0][0] < window_start:
            _, _, ticker, old_trade = heapq.heappop(self.trades_by_time)
            
            # Remove from ticker-specific queue
            ticker_trades = self.trades_by_ticker[ticker]
            if ticker_trades and ticker_trades[0] == old_trade:
                ticker_trades.popleft()
        
        # Recalculate window stats for all active tickers
        for ticker in list(self.ticker_states.keys()):
            # Double-check ticker still exists (could be removed by concurrent operations)
            if ticker not in self.ticker_states:
                continue
                
            if ticker in self.trades_by_ticker and self.trades_by_ticker[ticker]:
                self._recalculate_window_stats(ticker, window_start)
            else:
                # Remove ticker state if no recent trades
                if ticker in self.ticker_states:
                    try:
                        last_trade_time = self.ticker_states[ticker].last_trade_time
                        if last_trade_time < window_start:
                            del self.ticker_states[ticker]
                            if ticker in self.trades_by_ticker:
                                del self.trades_by_ticker[ticker]
                    except KeyError:
                        # Ticker was already removed by another operation, continue
                        continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics for debugging."""
        return {
            "active_tickers": len(self.ticker_states),
            "recent_trades_count": len(self.recent_trades),
            "total_trades_in_memory": sum(len(trades) for trades in self.trades_by_ticker.values()),
            "window_minutes": self.window_minutes,
            "max_price_points": self.max_price_points,
            "oldest_trade_in_heap": (
                datetime.fromtimestamp(self.trades_by_time[0][0] / 1000).isoformat()
                if self.trades_by_time else None
            )
        }


class TickerStateManager:
    """Manages individual ticker state and provides convenience methods."""
    
    def __init__(self, aggregator: TradeAggregator):
        """Initialize with reference to main aggregator."""
        self.aggregator = aggregator
    
    def update_ticker_state(self, ticker: str, trade: Trade) -> TickerState:
        """Update state for a specific ticker."""
        return self.aggregator.process_trade(trade)
    
    def get_ticker_state(self, ticker: str) -> Optional[TickerState]:
        """Get current state for a ticker."""
        return self.aggregator.get_ticker_state(ticker)
    
    def get_ticker_summary(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a ticker."""
        state = self.get_ticker_state(ticker)
        if not state:
            return None
        
        return {
            "ticker": state.ticker,
            "last_price_yes": state.last_yes_price_dollars,
            "last_price_no": state.last_no_price_dollars,
            "last_trade_time": datetime.fromtimestamp(state.last_trade_time / 1000).isoformat(),
            "volume_window": state.volume_window,
            "trade_count_window": state.trade_count_window,
            "net_flow": state.net_flow,
            "flow_direction": state.flow_direction,
            "price_points_count": len(state.price_points)
        }


# Global aggregator instance
_aggregator_instance = None

def get_aggregator() -> TradeAggregator:
    """Get the global aggregator instance."""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = TradeAggregator()
    return _aggregator_instance