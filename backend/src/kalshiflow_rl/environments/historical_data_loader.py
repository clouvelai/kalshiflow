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
    preload_strategy: str = "time_ordered"  # "time_ordered", "snapshot_only", "sample_uniform"
    
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
                    logger.warning(f"No historical data found for {market_ticker}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {market_ticker}: {e}")
                continue
        
        # Post-process data
        processed_data = await self._post_process_data(all_market_data, load_config)
        
        # Cache the data for potential reuse
        cache_key = self._generate_cache_key(market_tickers, start_time, end_time, load_config)
        self._cache[cache_key] = processed_data
        self._cache_metadata[cache_key] = {
            "market_tickers": market_tickers,
            "start_time": start_time,
            "end_time": end_time,
            "total_points": sum(len(points) for points in processed_data.values()),
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"Historical data loading complete: "
                   f"{len(processed_data)} markets, "
                   f"{sum(len(points) for points in processed_data.values())} total points")
        
        return processed_data
    
    async def _load_market_data(
        self,
        market_ticker: str,
        start_time: datetime,
        end_time: datetime,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """Load data for a single market."""
        
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Load snapshots first (they establish baseline state)
        snapshots = await self._load_snapshots(market_ticker, start_ms, end_ms, load_config)
        
        # Load deltas if needed
        if load_config.preload_strategy != "snapshot_only":
            deltas = await self._load_deltas(market_ticker, start_ms, end_ms, load_config)
        else:
            deltas = []
        
        # Combine and sort by timestamp
        all_data = snapshots + deltas
        all_data.sort(key=lambda x: (x.timestamp_ms, x.sequence_number))
        
        # Apply sampling if configured
        if load_config.sample_rate > 1:
            all_data = all_data[::load_config.sample_rate]
        
        # Apply data point limit
        if len(all_data) > load_config.max_data_points:
            # Keep evenly spaced data points
            indices = np.linspace(0, len(all_data) - 1, load_config.max_data_points, dtype=int)
            all_data = [all_data[i] for i in indices]
            logger.warning(f"Truncated {market_ticker} data to {load_config.max_data_points} points")
        
        return all_data
    
    async def _load_snapshots(
        self,
        market_ticker: str,
        start_ms: int,
        end_ms: int,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """Load orderbook snapshots."""
        query = """
        SELECT timestamp, sequence_number, yes_bids, yes_asks, no_bids, no_asks, received_at
        FROM orderbook_snapshots
        WHERE market_ticker = $1 
          AND timestamp >= $2 
          AND timestamp <= $3
        ORDER BY timestamp, sequence_number
        LIMIT $4
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, market_ticker, start_ms, end_ms, load_config.max_data_points)
        
        snapshots = []
        for row in rows:
            # Convert to OrderbookState-compatible format
            orderbook_state = {
                'market_ticker': market_ticker,
                'timestamp_ms': row['timestamp'],
                'sequence_number': row['sequence_number'],
                'yes_bids': dict(row['yes_bids']) if row['yes_bids'] else {},
                'yes_asks': dict(row['yes_asks']) if row['yes_asks'] else {},
                'no_bids': dict(row['no_bids']) if row['no_bids'] else {},
                'no_asks': dict(row['no_asks']) if row['no_asks'] else {},
                'last_update_time': row['timestamp'],
                'last_sequence': row['sequence_number']
            }
            
            # Calculate derived fields
            orderbook_state.update(self._calculate_derived_fields(orderbook_state))
            
            # Filter by activity threshold
            if (not load_config.include_inactive_periods and 
                orderbook_state.get('total_volume', 0) < load_config.min_activity_threshold):
                continue
            
            snapshot = HistoricalDataPoint(
                timestamp_ms=row['timestamp'],
                market_ticker=market_ticker,
                orderbook_state=orderbook_state,
                sequence_number=row['sequence_number'],
                is_snapshot=True
            )
            snapshots.append(snapshot)
        
        logger.debug(f"Loaded {len(snapshots)} snapshots for {market_ticker}")
        return snapshots
    
    async def _load_deltas(
        self,
        market_ticker: str,
        start_ms: int,
        end_ms: int,
        load_config: DataLoadConfig
    ) -> List[HistoricalDataPoint]:
        """Load orderbook deltas."""
        
        # For MVP, we'll reconstruct orderbook states by applying deltas to snapshots
        # This is more complex but provides complete historical replay
        
        # First, get a baseline snapshot just before our window
        baseline_snapshot = await self._get_baseline_snapshot(market_ticker, start_ms)
        if not baseline_snapshot:
            logger.warning(f"No baseline snapshot found for {market_ticker}")
            return []
        
        # Load deltas in the time window
        query = """
        SELECT timestamp, sequence_number, delta_type, side, price, quantity
        FROM orderbook_deltas
        WHERE market_ticker = $1 
          AND timestamp >= $2 
          AND timestamp <= $3
        ORDER BY timestamp, sequence_number
        LIMIT $4
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, market_ticker, start_ms, end_ms, load_config.max_data_points)
        
        # Reconstruct orderbook states by applying deltas
        current_state = OrderbookState(market_ticker)
        current_state.apply_snapshot(baseline_snapshot)
        
        deltas = []
        for row in rows:
            # Apply delta to current state
            delta_data = {
                'sequence_number': row['sequence_number'],
                'timestamp_ms': row['timestamp'],
                'side': row['side'],
                'action': row['delta_type'],
                'price': row['price'],
                'new_size': row['quantity']
            }
            
            success = current_state.apply_delta(delta_data)
            if not success and load_config.validate_sequences:
                logger.warning(f"Failed to apply delta {row['sequence_number']} for {market_ticker}")
                continue
            
            # Create historical data point
            orderbook_dict = current_state.to_dict()
            
            # Filter by activity threshold
            if (not load_config.include_inactive_periods and 
                orderbook_dict.get('total_volume', 0) < load_config.min_activity_threshold):
                continue
            
            delta_point = HistoricalDataPoint(
                timestamp_ms=row['timestamp'],
                market_ticker=market_ticker,
                orderbook_state=orderbook_dict,
                sequence_number=row['sequence_number'],
                is_snapshot=False
            )
            deltas.append(delta_point)
        
        logger.debug(f"Loaded {len(deltas)} deltas for {market_ticker}")
        return deltas
    
    async def _get_baseline_snapshot(
        self,
        market_ticker: str,
        start_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Get baseline snapshot before the time window."""
        query = """
        SELECT timestamp, sequence_number, yes_bids, yes_asks, no_bids, no_asks
        FROM orderbook_snapshots
        WHERE market_ticker = $1 AND timestamp < $2
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, market_ticker, start_ms)
        
        if not row:
            return None
        
        return {
            'market_ticker': market_ticker,
            'timestamp_ms': row['timestamp'],
            'sequence_number': row['sequence_number'],
            'yes_bids': dict(row['yes_bids']) if row['yes_bids'] else {},
            'yes_asks': dict(row['yes_asks']) if row['yes_asks'] else {},
            'no_bids': dict(row['no_bids']) if row['no_bids'] else {},
            'no_asks': dict(row['no_asks']) if row['no_asks'] else {}
        }
    
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
    
    async def _post_process_data(
        self,
        market_data: Dict[str, List[HistoricalDataPoint]],
        load_config: DataLoadConfig
    ) -> Dict[str, List[HistoricalDataPoint]]:
        """Post-process loaded data for quality and consistency."""
        
        processed_data = {}
        
        for market_ticker, data_points in market_data.items():
            if not data_points:
                continue
            
            processed_points = data_points.copy()
            
            # Remove outliers if requested
            if load_config.remove_outliers:
                processed_points = self._remove_outliers(processed_points)
            
            # Fill gaps if requested
            if load_config.fill_gaps:
                processed_points = self._fill_sequence_gaps(processed_points)
            
            # Validate sequences if requested
            if load_config.validate_sequences:
                processed_points = self._validate_sequences(processed_points, market_ticker)
            
            processed_data[market_ticker] = processed_points
            
            logger.debug(f"Post-processed {market_ticker}: "
                        f"{len(data_points)} -> {len(processed_points)} points")
        
        # Synchronize timestamps across markets for multi-market training
        if len(processed_data) > 1:
            processed_data = self._synchronize_timestamps(processed_data)
        
        # Memory cleanup
        gc.collect()
        
        return processed_data
    
    def _remove_outliers(self, data_points: List[HistoricalDataPoint]) -> List[HistoricalDataPoint]:
        """Remove outlier data points based on volume and price statistics."""
        if len(data_points) < 10:
            return data_points
        
        # Extract total volumes for outlier detection
        volumes = [point.orderbook_state.get('total_volume', 0) for point in data_points]
        
        # Use IQR method for outlier detection
        q1 = np.percentile(volumes, 25)
        q3 = np.percentile(volumes, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_points = []
        for point in data_points:
            volume = point.orderbook_state.get('total_volume', 0)
            if lower_bound <= volume <= upper_bound:
                filtered_points.append(point)
        
        outliers_removed = len(data_points) - len(filtered_points)
        if outliers_removed > 0:
            logger.debug(f"Removed {outliers_removed} outlier data points")
        
        return filtered_points
    
    def _fill_sequence_gaps(self, data_points: List[HistoricalDataPoint]) -> List[HistoricalDataPoint]:
        """Fill sequence number gaps with interpolated data."""
        if len(data_points) < 2:
            return data_points
        
        filled_points = []
        for i in range(len(data_points) - 1):
            current_point = data_points[i]
            next_point = data_points[i + 1]
            
            filled_points.append(current_point)
            
            # Check for sequence gap
            seq_gap = next_point.sequence_number - current_point.sequence_number
            if seq_gap > 1 and seq_gap < 10:  # Only fill small gaps
                # Create interpolated points
                for j in range(1, seq_gap):
                    interp_factor = j / seq_gap
                    interp_timestamp = int(
                        current_point.timestamp_ms + 
                        interp_factor * (next_point.timestamp_ms - current_point.timestamp_ms)
                    )
                    interp_sequence = current_point.sequence_number + j
                    
                    # Use current point's orderbook state (simple interpolation)
                    interp_point = HistoricalDataPoint(
                        timestamp_ms=interp_timestamp,
                        market_ticker=current_point.market_ticker,
                        orderbook_state=current_point.orderbook_state.copy(),
                        sequence_number=interp_sequence,
                        is_snapshot=False
                    )
                    filled_points.append(interp_point)
        
        # Add the last point
        if data_points:
            filled_points.append(data_points[-1])
        
        gaps_filled = len(filled_points) - len(data_points)
        if gaps_filled > 0:
            logger.debug(f"Filled {gaps_filled} sequence gaps")
        
        return filled_points
    
    def _validate_sequences(
        self, 
        data_points: List[HistoricalDataPoint], 
        market_ticker: str
    ) -> List[HistoricalDataPoint]:
        """Validate and filter data points with sequence issues."""
        if len(data_points) < 2:
            return data_points
        
        valid_points = []
        last_sequence = 0
        
        for point in data_points:
            # Check sequence ordering
            if point.sequence_number >= last_sequence:
                valid_points.append(point)
                last_sequence = point.sequence_number
            else:
                logger.debug(f"Dropped out-of-order point {point.sequence_number} for {market_ticker}")
        
        invalid_count = len(data_points) - len(valid_points)
        if invalid_count > 0:
            logger.debug(f"Removed {invalid_count} invalid sequence points for {market_ticker}")
        
        return valid_points
    
    def _synchronize_timestamps(
        self,
        market_data: Dict[str, List[HistoricalDataPoint]]
    ) -> Dict[str, List[HistoricalDataPoint]]:
        """Synchronize timestamps across markets for consistent multi-market training."""
        
        # Find common time range
        all_timestamps = []
        for data_points in market_data.values():
            timestamps = [point.timestamp_ms for point in data_points]
            all_timestamps.extend(timestamps)
        
        if not all_timestamps:
            return market_data
        
        # Use common time grid
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Create synchronized data
        synchronized_data = {}
        
        for market_ticker, data_points in market_data.items():
            # Filter to common time range and ensure consistent spacing
            filtered_points = [
                point for point in data_points
                if min_time <= point.timestamp_ms <= max_time
            ]
            
            synchronized_data[market_ticker] = filtered_points
        
        return synchronized_data
    
    def _generate_cache_key(
        self,
        market_tickers: List[str],
        start_time: datetime,
        end_time: datetime,
        load_config: DataLoadConfig
    ) -> str:
        """Generate cache key for loaded data."""
        tickers_str = "_".join(sorted(market_tickers))
        start_str = start_time.strftime("%Y%m%d_%H%M%S")
        end_str = end_time.strftime("%Y%m%d_%H%M%S")
        config_hash = hash(str(load_config))
        
        return f"{tickers_str}_{start_str}_{end_str}_{config_hash}"
    
    def create_data_iterator(
        self,
        market_data: Dict[str, List[HistoricalDataPoint]],
        batch_size: int = 1,
        shuffle: bool = False
    ) -> Iterator[Dict[str, HistoricalDataPoint]]:
        """
        Create iterator over historical data for training.
        
        Args:
            market_data: Loaded historical data
            batch_size: Batch size for iteration (currently only supports 1)
            shuffle: Whether to shuffle the data order
            
        Yields:
            Dict mapping market_ticker -> HistoricalDataPoint for each timestamp
        """
        if not market_data:
            return
        
        # Get all unique timestamps across markets
        all_timestamps = set()
        for data_points in market_data.values():
            for point in data_points:
                all_timestamps.add(point.timestamp_ms)
        
        timestamps = sorted(all_timestamps)
        
        if shuffle:
            np.random.shuffle(timestamps)
        
        # Create timestamp -> market_data mapping
        timestamp_map = {}
        for market_ticker, data_points in market_data.items():
            for point in data_points:
                if point.timestamp_ms not in timestamp_map:
                    timestamp_map[point.timestamp_ms] = {}
                timestamp_map[point.timestamp_ms][market_ticker] = point
        
        # Yield data for each timestamp
        for timestamp in timestamps:
            if timestamp in timestamp_map:
                yield timestamp_map[timestamp]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_points = sum(
            metadata.get("total_points", 0) 
            for metadata in self._cache_metadata.values()
        )
        
        return {
            "cached_datasets": len(self._cache),
            "total_cached_points": total_points,
            "cache_keys": list(self._cache.keys()),
            "memory_usage_mb": sum(
                len(str(data).encode('utf-8')) for data in self._cache.values()
            ) / (1024 * 1024)
        }