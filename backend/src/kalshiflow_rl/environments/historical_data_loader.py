"""
Historical data loader for Kalshi RL Trading Subsystem.

Loads historical orderbook data from PostgreSQL for training environments.
Ensures NO database queries during training episodes by preloading all data.

CRITICAL ARCHITECTURAL REQUIREMENT:
- ALL data must be preloaded before training begins
- NO database access allowed during env.step() or env.reset()
- Data format must match live OrderbookState for consistency
- Efficient memory management for large datasets
"""

import asyncio
import asyncpg
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
import gc

from ..config import config
from ..data.orderbook_state import OrderbookState

logger = logging.getLogger("kalshiflow_rl.historical_data_loader")


@dataclass
class HistoricalDataPoint:
    """Single historical data point for training."""
    timestamp_ms: int
    market_ticker: str
    orderbook_state: Dict[str, Any]
    sequence_number: int
    is_snapshot: bool  # True for snapshots, False for deltas


@dataclass
class DataLoadConfig:
    """Configuration for historical data loading."""
    
    # Time range settings
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window_hours: int = 24  # Default 24-hour window
    
    # Memory management
    max_data_points: int = 100000  # Maximum data points to load
    batch_size: int = 1000  # Batch size for database queries
    preload_strategy: str = "reconstruct"  # "reconstruct", "snapshot_only", "time_ordered"
    
    # Data filtering
    min_activity_threshold: int = 100  # Minimum total volume to include
    include_inactive_periods: bool = False
    sample_rate: int = 1  # Keep 1 out of N data points
    
    # Market filtering
    market_tickers: Optional[List[str]] = None  # None = load all from config
    
    # Data quality
    validate_sequences: bool = True
    fill_gaps: bool = True  # Fill sequence gaps with interpolated data
    remove_outliers: bool = True


class HistoricalDataLoader:
    """
    Loads historical orderbook data from PostgreSQL for RL training.
    
    Ensures all data is preloaded and formatted consistently with live data.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize historical data loader.
        
        Args:
            db_url: Database URL (uses config if None)
        """
        self.db_url = db_url or config.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
        self._cache: Dict[str, List[HistoricalDataPoint]] = {}
        self._cache_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self) -> None:
        """Initialize database connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=5,
                command_timeout=60
            )
            logger.info("Connected to database for historical data loading")
    
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Disconnected from database")
    
    async def load_historical_data(
        self,
        load_config: DataLoadConfig
    ) -> Dict[str, List[HistoricalDataPoint]]:
        """
        Load historical data for training environment.
        
        Args:
            load_config: Data loading configuration
            
        Returns:
            Dict mapping market_ticker -> list of historical data points (time-ordered)
        """
        if not self.pool:
            await self.connect()
        
        # Determine market tickers to load
        market_tickers = load_config.market_tickers or config.RL_MARKET_TICKERS
        
        # Determine time range
        if load_config.start_time and load_config.end_time:
            start_time = load_config.start_time
            end_time = load_config.end_time
        else:
            # Use recent data window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=load_config.window_hours)
        
        logger.info(f"Loading historical data for {len(market_tickers)} markets "
                   f"from {start_time} to {end_time}")
        
        # Load data for each market
        all_market_data = {}
        
        for market_ticker in market_tickers:
            try:
                market_data = await self._load_market_data(
                    market_ticker, start_time, end_time, load_config
                )
                
                if market_data:
                    all_market_data[market_ticker] = market_data
                    logger.info(f"Loaded {len(market_data)} data points for {market_ticker}")
                else:
                    logger.warning(f"No data found for {market_ticker}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {market_ticker}: {e}")
                continue
        
        # Cache data and metadata
        self._cache = all_market_data
        self._cache_metadata = {
            'loaded_at': datetime.utcnow(),
            'start_time': start_time,
            'end_time': end_time,
            'total_data_points': sum(len(data) for data in all_market_data.values()),
            'markets': list(all_market_data.keys())
        }
        
        logger.info(f"Successfully loaded {self._cache_metadata['total_data_points']} total data points")
        
        return all_market_data
    
    async def _load_market_data(
        self,
        market_ticker: str,
        start_time: datetime,
        end_time: datetime,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """Load data for a single market."""
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        if load_config.preload_strategy == "snapshot_only":
            # Load only snapshots (simpler, less accurate)
            return await self._load_snapshots(market_ticker, start_ms, end_ms, load_config)
        
        elif load_config.preload_strategy == "reconstruct":
            # Reconstruct orderbook states from snapshots and deltas
            return await self._reconstruct_orderbook_states(market_ticker, start_ms, end_ms, load_config)
        
        else:
            # Default: Load snapshots and interpolate
            snapshots = await self._load_snapshots(market_ticker, start_ms, end_ms, load_config)
            
            # Apply sampling if requested
            if load_config.sample_rate > 1:
                snapshots = snapshots[::load_config.sample_rate]
            
            return snapshots
    
    async def _load_snapshots(
        self,
        market_ticker: str,
        start_ms: int,
        end_ms: int,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """Load orderbook snapshots."""
        query = """
        SELECT timestamp_ms, sequence_number, yes_bids, yes_asks, no_bids, no_asks
        FROM rl_orderbook_snapshots
        WHERE market_ticker = $1 
          AND timestamp_ms >= $2 
          AND timestamp_ms <= $3
        ORDER BY timestamp_ms, sequence_number
        LIMIT $4
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, market_ticker, start_ms, end_ms, load_config.max_data_points)
        
        if not rows:
            logger.debug(f"No snapshots found for {market_ticker} in time range")
            return []
        
        snapshots = []
        for row in rows:
            # Convert JSONB to Python dict (handle both dict and JSON string)
            def parse_orderbook_field(value):
                if value is None:
                    return {}
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                return {}
            
            orderbook_state = {
                'market_ticker': market_ticker,
                'timestamp_ms': row['timestamp_ms'],
                'sequence_number': row['sequence_number'],
                'yes_bids': parse_orderbook_field(row['yes_bids']),
                'yes_asks': parse_orderbook_field(row['yes_asks']),
                'no_bids': parse_orderbook_field(row['no_bids']),
                'no_asks': parse_orderbook_field(row['no_asks'])
            }
            
            # Calculate derived fields
            orderbook_state.update(self._calculate_derived_fields(orderbook_state))
            
            # Filter by activity threshold
            if (not load_config.include_inactive_periods and 
                orderbook_state.get('total_volume', 0) < load_config.min_activity_threshold):
                continue
            
            snapshot = HistoricalDataPoint(
                timestamp_ms=row['timestamp_ms'],
                market_ticker=market_ticker,
                orderbook_state=orderbook_state,
                sequence_number=row['sequence_number'],
                is_snapshot=True
            )
            snapshots.append(snapshot)
        
        logger.debug(f"Loaded {len(snapshots)} snapshots for {market_ticker}")
        return snapshots
    
    async def _reconstruct_orderbook_states(
        self,
        market_ticker: str,
        start_ms: int,
        end_ms: int,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """
        Reconstruct orderbook states by applying deltas to snapshots.
        This provides the most accurate historical replay.
        """
        
        # Get the latest snapshot before or at start time
        baseline_query = """
        SELECT timestamp_ms, sequence_number, yes_bids, yes_asks, no_bids, no_asks
        FROM rl_orderbook_snapshots
        WHERE market_ticker = $1 AND timestamp_ms <= $2
        ORDER BY timestamp_ms DESC, sequence_number DESC
        LIMIT 1
        """
        
        async with self.pool.acquire() as conn:
            baseline_row = await conn.fetchrow(baseline_query, market_ticker, start_ms)
        
        if not baseline_row:
            logger.warning(f"No baseline snapshot found for {market_ticker} before {start_ms}")
            # Try loading snapshots only
            return await self._load_snapshots(market_ticker, start_ms, end_ms, load_config)
        
        # Initialize OrderbookState with baseline
        current_state = OrderbookState(market_ticker)
        
        # Helper to parse JSONB fields
        def parse_orderbook_field(value):
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return json.loads(value)
            return {}
        
        baseline_data = {
            'market_ticker': market_ticker,
            'timestamp_ms': baseline_row['timestamp_ms'],
            'sequence_number': baseline_row['sequence_number'],
            'yes_bids': parse_orderbook_field(baseline_row['yes_bids']),
            'yes_asks': parse_orderbook_field(baseline_row['yes_asks']),
            'no_bids': parse_orderbook_field(baseline_row['no_bids']),
            'no_asks': parse_orderbook_field(baseline_row['no_asks'])
        }
        current_state.apply_snapshot(baseline_data)
        
        # Store reconstructed states
        reconstructed_states = []
        
        # Add baseline if it's within our time range
        if baseline_row['timestamp_ms'] >= start_ms:
            reconstructed_states.append(HistoricalDataPoint(
                timestamp_ms=baseline_row['timestamp_ms'],
                market_ticker=market_ticker,
                orderbook_state=current_state.to_dict(),
                sequence_number=baseline_row['sequence_number'],
                is_snapshot=True
            ))
        
        # Load all snapshots and deltas in the time range
        combined_query = """
        WITH combined AS (
            SELECT 
                timestamp_ms,
                sequence_number,
                'snapshot' as event_type,
                yes_bids,
                yes_asks,
                no_bids,
                no_asks,
                NULL as side,
                NULL as action,
                NULL as price,
                NULL as old_size,
                NULL as new_size
            FROM rl_orderbook_snapshots
            WHERE market_ticker = $1 
              AND timestamp_ms > $2 
              AND timestamp_ms <= $3
              AND sequence_number > $4
            
            UNION ALL
            
            SELECT 
                timestamp_ms,
                sequence_number,
                'delta' as event_type,
                NULL as yes_bids,
                NULL as yes_asks,
                NULL as no_bids,
                NULL as no_asks,
                side,
                action,
                price,
                old_size,
                new_size
            FROM rl_orderbook_deltas
            WHERE market_ticker = $1 
              AND timestamp_ms > $2 
              AND timestamp_ms <= $3
              AND sequence_number > $4
        )
        SELECT * FROM combined
        ORDER BY sequence_number, timestamp_ms
        LIMIT $5
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                combined_query, 
                market_ticker, 
                baseline_row['timestamp_ms'],
                end_ms,
                baseline_row['sequence_number'],
                load_config.max_data_points
            )
        
        logger.info(f"Found {len(rows)} events for {market_ticker} to reconstruct")
        
        for row in rows:
            if row['event_type'] == 'snapshot':
                # Apply snapshot
                snapshot_data = {
                    'market_ticker': market_ticker,
                    'timestamp_ms': row['timestamp_ms'],
                    'sequence_number': row['sequence_number'],
                    'yes_bids': parse_orderbook_field(row['yes_bids']),
                    'yes_asks': parse_orderbook_field(row['yes_asks']),
                    'no_bids': parse_orderbook_field(row['no_bids']),
                    'no_asks': parse_orderbook_field(row['no_asks'])
                }
                current_state.apply_snapshot(snapshot_data)
                
                # Only include if in our time range
                if row['timestamp_ms'] >= start_ms:
                    reconstructed_states.append(HistoricalDataPoint(
                        timestamp_ms=row['timestamp_ms'],
                        market_ticker=market_ticker,
                        orderbook_state=current_state.to_dict(),
                        sequence_number=row['sequence_number'],
                        is_snapshot=True
                    ))
                    
            elif row['event_type'] == 'delta':
                # Apply delta
                delta_data = {
                    'sequence_number': row['sequence_number'],
                    'timestamp_ms': row['timestamp_ms'],
                    'side': row['side'],
                    'action': row['action'],
                    'price': row['price'],
                    'old_size': row['old_size'],
                    'new_size': row['new_size']
                }
                
                success = current_state.apply_delta(delta_data)
                
                if not success and load_config.validate_sequences:
                    logger.warning(f"Failed to apply delta seq={row['sequence_number']} for {market_ticker}")
                    continue
                
                # Only include if in our time range
                if row['timestamp_ms'] >= start_ms:
                    reconstructed_states.append(HistoricalDataPoint(
                        timestamp_ms=row['timestamp_ms'],
                        market_ticker=market_ticker,
                        orderbook_state=current_state.to_dict(),
                        sequence_number=row['sequence_number'],
                        is_snapshot=False
                    ))
        
        logger.info(f"Reconstructed {len(reconstructed_states)} states for {market_ticker}")
        return reconstructed_states
    
    def _calculate_derived_fields(self, orderbook_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived fields like spreads, mid prices, total volume."""
        yes_bids = orderbook_state.get('yes_bids', {})
        yes_asks = orderbook_state.get('yes_asks', {})
        no_bids = orderbook_state.get('no_bids', {})
        no_asks = orderbook_state.get('no_asks', {})
        
        # Calculate spreads
        yes_spread = None
        no_spread = None
        yes_mid_price = None
        no_mid_price = None
        
        if yes_bids and yes_asks:
            best_yes_bid = max(int(p) for p in yes_bids.keys())
            best_yes_ask = min(int(p) for p in yes_asks.keys())
            yes_spread = best_yes_ask - best_yes_bid
            yes_mid_price = (best_yes_bid + best_yes_ask) / 2.0
        
        if no_bids and no_asks:
            best_no_bid = max(int(p) for p in no_bids.keys())
            best_no_ask = min(int(p) for p in no_asks.keys())
            no_spread = best_no_ask - best_no_bid
            no_mid_price = (best_no_bid + best_no_ask) / 2.0
        
        # Calculate total volume
        total_volume = (
            sum(yes_bids.values()) + sum(yes_asks.values()) +
            sum(no_bids.values()) + sum(no_asks.values())
        )
        
        return {
            'yes_spread': yes_spread,
            'no_spread': no_spread,
            'yes_mid_price': yes_mid_price,
            'no_mid_price': no_mid_price,
            'total_volume': total_volume
        }
    
    async def get_episode_iterator(
        self,
        market_ticker: str,
        episode_length: int = 1000
    ) -> Iterator[List[HistoricalDataPoint]]:
        """
        Iterator for training episodes from cached historical data.
        
        Args:
            market_ticker: Market to iterate over
            episode_length: Number of data points per episode
            
        Yields:
            List of HistoricalDataPoint for one episode
        """
        if market_ticker not in self._cache:
            raise ValueError(f"No cached data for {market_ticker}. Call load_historical_data first.")
        
        market_data = self._cache[market_ticker]
        
        # Sliding window over historical data
        for i in range(0, len(market_data) - episode_length + 1):
            yield market_data[i:i + episode_length]
    
    def get_cache_metadata(self) -> Dict[str, Any]:
        """Get metadata about cached historical data."""
        return self._cache_metadata.copy()
    
    def clear_cache(self) -> None:
        """Clear cached historical data to free memory."""
        self._cache.clear()
        self._cache_metadata.clear()
        gc.collect()
        logger.info("Cleared historical data cache")