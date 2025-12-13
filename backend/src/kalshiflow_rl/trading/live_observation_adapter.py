"""
Live Observation Adapter for Kalshi Trading Actor MVP.

Converts real-time SharedOrderbookState data to training-consistent 52-feature observations
using the same feature extraction functions as training. Maintains sliding window history
for temporal feature computation.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal

from ..data.orderbook_state import SharedOrderbookState
from ..environments.feature_extractors import (
    build_observation_from_session_data,
    extract_temporal_features,
    extract_market_agnostic_features,
    _get_default_market_features
)
from ..environments.session_data_loader import SessionDataPoint

logger = logging.getLogger("kalshiflow_rl.live_observation_adapter")


@dataclass 
class LiveOrderbookSnapshot:
    """Live orderbook state converted to SessionDataPoint format."""
    timestamp: datetime
    timestamp_ms: int
    market_ticker: str
    orderbook_data: Dict[str, Any]  # SharedOrderbookState format
    total_volume: int = 0


class LiveObservationAdapter:
    """
    Adapter to convert live SharedOrderbookState to training-consistent observations.
    
    Features:
    - Maintains sliding window history for temporal features
    - Converts SharedOrderbookState to SessionDataPoint format
    - Uses same feature extraction as training (52 features)
    - Per-market sliding windows for temporal computations
    - Performance optimized for <1ms observation building
    - Dependency injection support for orderbook state registry
    """
    
    def __init__(
        self,
        window_size: int = 10,
        max_markets: int = 1,
        temporal_context_minutes: int = 30,
        orderbook_state_registry: Optional[Any] = None
    ):
        """
        Initialize live observation adapter.
        
        Args:
            window_size: Number of snapshots to keep for temporal features
            max_markets: Maximum markets to include in observation
            temporal_context_minutes: Minutes of historical context to maintain
        """
        self.window_size = window_size
        self.max_markets = max_markets
        self.temporal_context_minutes = temporal_context_minutes
        
        # Dependency injection for orderbook state registry
        self._orderbook_state_registry = orderbook_state_registry
        
        # Per-market sliding windows for temporal features
        self._market_windows: Dict[str, Deque[LiveOrderbookSnapshot]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        
        # Cache for performance optimization
        self._last_observation_time: Dict[str, float] = {}
        self._cached_observations: Dict[str, np.ndarray] = {}
        self._cache_ttl_seconds = 0.1  # Cache observations for 100ms
        
        # Performance monitoring
        self._observations_built = 0
        self._cache_hits = 0
        self._total_build_time = 0.0
        
        logger.info(
            f"LiveObservationAdapter initialized: "
            f"window_size={window_size}, max_markets={max_markets}, "
            f"temporal_context={temporal_context_minutes}min, "
            f"registry_injected={orderbook_state_registry is not None}"
        )
    
    async def build_observation(
        self,
        market_ticker: str,
        position_data: Optional[Dict[str, Any]] = None,
        portfolio_value: float = 10000.0,  # Default starting value
        cash_balance: float = 10000.0,     # Default starting cash
        order_features: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Build observation for a specific market using live orderbook state.
        
        This creates a 52-feature observation that exactly matches the training format
        by converting SharedOrderbookState to SessionDataPoint and using the same
        feature extraction pipeline.
        
        Args:
            market_ticker: Market to build observation for
            position_data: Current portfolio positions (for portfolio features)
            portfolio_value: Total portfolio value in dollars
            cash_balance: Available cash in dollars  
            order_features: Optional order state features (5 elements)
            
        Returns:
            52-feature numpy array matching training format, or None if no data
        """
        start_time = time.time()
        
        try:
            # Check cache first (performance optimization)
            cache_key = f"{market_ticker}_{portfolio_value}_{cash_balance}"
            current_time = time.time()
            
            if (cache_key in self._cached_observations and
                current_time - self._last_observation_time.get(cache_key, 0) < self._cache_ttl_seconds):
                self._cache_hits += 1
                return self._cached_observations[cache_key]
            
            # Get live orderbook state via injected registry
            if self._orderbook_state_registry:
                shared_state = await self._orderbook_state_registry.get_shared_orderbook_state(market_ticker)
            else:
                # Fallback for backward compatibility during migration
                from ..data.orderbook_state import get_shared_orderbook_state
                shared_state = await get_shared_orderbook_state(market_ticker)
            
            # Get current orderbook snapshot
            current_snapshot = await self._get_live_snapshot(shared_state, market_ticker)
            if not current_snapshot:
                logger.debug(f"No live snapshot available for {market_ticker}")
                return None
            
            # Update sliding window with current snapshot
            self._market_windows[market_ticker].append(current_snapshot)
            
            # Convert to SessionDataPoint format for feature extraction
            session_data_point = self._convert_to_session_data_point(current_snapshot)
            
            # Build historical context for temporal features
            historical_data = self._build_historical_context(market_ticker)
            
            # Ensure position_data has proper format
            if position_data is None:
                position_data = {}
            
            # Build observation using the same function as training
            observation = build_observation_from_session_data(
                session_data=session_data_point,
                historical_data=historical_data,
                position_data=position_data,
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                max_markets=self.max_markets,
                order_features=order_features
            )
            
            # Cache the observation
            self._cached_observations[cache_key] = observation
            self._last_observation_time[cache_key] = current_time
            
            # Update performance metrics
            build_time = time.time() - start_time
            self._observations_built += 1
            self._total_build_time += build_time
            
            if build_time > 0.001:  # Log if over 1ms target
                logger.warning(f"Slow observation build: {build_time*1000:.2f}ms for {market_ticker}")
            
            logger.debug(
                f"Built live observation for {market_ticker}: {len(observation)} features, "
                f"build_time={build_time*1000:.2f}ms"
            )
            
            return observation
            
        except Exception as e:
            logger.error(f"Error building observation for {market_ticker}: {e}")
            return None
    
    async def _get_live_snapshot(
        self,
        shared_state: SharedOrderbookState,
        market_ticker: str
    ) -> Optional[LiveOrderbookSnapshot]:
        """
        Get current orderbook snapshot from SharedOrderbookState.
        
        Converts SharedOrderbookState to our internal format for processing.
        """
        try:
            # Get current snapshot from shared state
            snapshot = await shared_state.get_snapshot()
            
            if not snapshot:
                logger.debug(f"No snapshot available from SharedOrderbookState for {market_ticker}")
                return None
            
            # Calculate total volume for activity sorting
            total_volume = 0
            for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                if side in snapshot:
                    total_volume += sum(snapshot[side].values())
            
            # Create live snapshot in our format
            now = datetime.utcnow()
            live_snapshot = LiveOrderbookSnapshot(
                timestamp=now,
                timestamp_ms=int(now.timestamp() * 1000),
                market_ticker=market_ticker,
                orderbook_data=snapshot,
                total_volume=total_volume
            )
            
            return live_snapshot
            
        except Exception as e:
            logger.error(f"Error getting live snapshot for {market_ticker}: {e}")
            return None
    
    def _convert_to_session_data_point(
        self,
        live_snapshot: LiveOrderbookSnapshot
    ) -> SessionDataPoint:
        """
        Convert LiveOrderbookSnapshot to SessionDataPoint format.
        
        This enables us to use the same feature extraction pipeline as training.
        """
        # Calculate time gap from previous snapshot in this market
        market_ticker = live_snapshot.market_ticker
        time_gap = 0.0
        
        windows = self._market_windows[market_ticker]
        if len(windows) >= 2:
            prev_snapshot = windows[-2]  # Second to last (before current)
            time_gap = (live_snapshot.timestamp - prev_snapshot.timestamp).total_seconds()
        
        # Convert to SessionDataPoint format
        markets_data = {
            live_snapshot.market_ticker: {
                **live_snapshot.orderbook_data,
                'total_volume': live_snapshot.total_volume
            }
        }
        
        session_data_point = SessionDataPoint(
            timestamp=live_snapshot.timestamp,
            timestamp_ms=live_snapshot.timestamp_ms,
            markets_data=markets_data,
            time_gap=time_gap
        )
        
        # Calculate temporal features using existing logic
        historical_data = self._build_historical_context(market_ticker)
        session_data_point = self._compute_temporal_features(session_data_point, historical_data)
        
        return session_data_point
    
    def _build_historical_context(
        self,
        market_ticker: str
    ) -> List[SessionDataPoint]:
        """
        Build historical context from sliding window for temporal features.
        
        Converts our sliding window to SessionDataPoint list for feature extraction.
        """
        historical_data = []
        
        windows = self._market_windows[market_ticker]
        if len(windows) <= 1:
            return historical_data
        
        # Convert all but the latest snapshot to historical data
        for i, snapshot in enumerate(list(windows)[:-1]):  # Exclude current (last) snapshot
            # Calculate time gap from previous snapshot
            time_gap = 0.0
            if i > 0:
                prev_snapshot = list(windows)[i-1]
                time_gap = (snapshot.timestamp - prev_snapshot.timestamp).total_seconds()
            
            # Create session data point
            markets_data = {
                snapshot.market_ticker: {
                    **snapshot.orderbook_data,
                    'total_volume': snapshot.total_volume
                }
            }
            
            historical_point = SessionDataPoint(
                timestamp=snapshot.timestamp,
                timestamp_ms=snapshot.timestamp_ms,
                markets_data=markets_data,
                time_gap=time_gap
            )
            
            historical_data.append(historical_point)
        
        return historical_data
    
    def _compute_temporal_features(
        self,
        session_data_point: SessionDataPoint,
        historical_data: List[SessionDataPoint]
    ) -> SessionDataPoint:
        """
        Compute temporal features for a SessionDataPoint using historical context.
        
        This uses the same temporal feature extraction as training.
        """
        try:
            # Use the existing temporal feature extraction
            temporal_features = extract_temporal_features(session_data_point, historical_data)
            
            # Update the session data point with computed features
            session_data_point.activity_score = temporal_features.get('activity_intensity', 0.0)
            session_data_point.momentum = temporal_features.get('price_momentum', 0.0)
            
            return session_data_point
            
        except Exception as e:
            logger.error(f"Error computing temporal features: {e}")
            # Return with default values
            session_data_point.activity_score = 0.0
            session_data_point.momentum = 0.0
            return session_data_point
    
    def cleanup_old_data(self, max_age_minutes: int = None) -> None:
        """
        Clean up old data from sliding windows.
        
        Args:
            max_age_minutes: Maximum age of data to keep (defaults to temporal_context_minutes)
        """
        if max_age_minutes is None:
            max_age_minutes = self.temporal_context_minutes
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        for market_ticker, window in self._market_windows.items():
            # Remove old snapshots
            while window and window[0].timestamp < cutoff_time:
                window.popleft()
        
        # Clean up cache entries for markets with no recent data
        current_time = time.time()
        expired_cache_keys = [
            key for key, timestamp in self._last_observation_time.items()
            if current_time - timestamp > max_age_minutes * 60
        ]
        
        for key in expired_cache_keys:
            self._cached_observations.pop(key, None)
            self._last_observation_time.pop(key, None)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        avg_build_time = (
            self._total_build_time / max(self._observations_built, 1)
            if self._observations_built > 0 else 0.0
        )
        
        cache_hit_rate = (
            self._cache_hits / max(self._observations_built, 1)
            if self._observations_built > 0 else 0.0
        )
        
        return {
            "observations_built": self._observations_built,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_build_time_ms": avg_build_time * 1000,
            "active_markets": len(self._market_windows),
            "cached_observations": len(self._cached_observations),
            "total_snapshots": sum(len(window) for window in self._market_windows.values())
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring."""
        return {
            "adapter": "LiveObservationAdapter",
            "window_size": self.window_size,
            "max_markets": self.max_markets,
            "temporal_context_minutes": self.temporal_context_minutes,
            "metrics": self.get_metrics(),
            "active_markets": list(self._market_windows.keys()),
            "performance": {
                "avg_build_time_ms": (
                    self._total_build_time / max(self._observations_built, 1) * 1000
                    if self._observations_built > 0 else 0.0
                ),
                "cache_efficiency": (
                    self._cache_hits / max(self._observations_built, 1)
                    if self._observations_built > 0 else 0.0
                )
            }
        }


# ===============================================================================
# Dependency Injection Support
# ===============================================================================

# Global adapter instance (maintained for backward compatibility during migration)
_live_observation_adapter: Optional[LiveObservationAdapter] = None
_adapter_lock = asyncio.Lock()


async def get_live_observation_adapter() -> Optional[LiveObservationAdapter]:
    """
    Get the global live observation adapter instance.
    
    DEPRECATED: Use dependency injection via ServiceContainer instead.
    This function is maintained for backward compatibility during migration.
    
    Returns:
        LiveObservationAdapter instance or None if not initialized
    """
    # Try to get from service container first
    try:
        from .service_container import get_default_container
        container = await get_default_container()
        if container.is_registered("live_observation_adapter"):
            return await container.get_service("live_observation_adapter")
    except Exception:
        pass  # Fall back to old singleton pattern
    
    async with _adapter_lock:
        return _live_observation_adapter


async def shutdown_live_observation_adapter() -> None:
    """Shutdown the global live observation adapter."""
    global _live_observation_adapter
    
    async with _adapter_lock:
        if _live_observation_adapter:
            _live_observation_adapter = None
            logger.info("✅ Global LiveObservationAdapter shutdown complete")


async def initialize_live_observation_adapter(
    window_size: int = 10,
    max_markets: int = 1,
    **kwargs
) -> LiveObservationAdapter:
    """
    Initialize the global live observation adapter.
    
    DEPRECATED: Use dependency injection via ServiceContainer instead.
    This function is maintained for backward compatibility during migration.
    
    Args:
        window_size: Sliding window size for temporal features  
        max_markets: Maximum markets in observation
        **kwargs: Additional adapter arguments
        
    Returns:
        Initialized LiveObservationAdapter instance
    """
    global _live_observation_adapter
    
    async with _adapter_lock:
        if _live_observation_adapter is not None:
            logger.warning("LiveObservationAdapter already initialized")
            return _live_observation_adapter
        
        logger.info("Initializing global LiveObservationAdapter...")
        
        _live_observation_adapter = LiveObservationAdapter(
            window_size=window_size,
            max_markets=max_markets,
            **kwargs
        )
        
        logger.info("✅ Global LiveObservationAdapter initialized")
        return _live_observation_adapter


async def build_live_observation(
    market_ticker: str,
    position_data: Optional[Dict[str, Any]] = None,
    portfolio_value: float = 10000.0,
    cash_balance: float = 10000.0,
    order_features: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Build live observation using the global adapter.
    
    DEPRECATED: Use dependency injection via ServiceContainer instead.
    Convenience function for ActorService integration during migration.
    
    Args:
        market_ticker: Market to build observation for
        position_data: Portfolio positions
        portfolio_value: Total portfolio value
        cash_balance: Available cash
        order_features: Order state features
        
    Returns:
        52-feature observation array or None if no data
    """
    adapter = await get_live_observation_adapter()
    if not adapter:
        logger.error("LiveObservationAdapter not initialized")
        return None
    
    return await adapter.build_observation(
        market_ticker=market_ticker,
        position_data=position_data,
        portfolio_value=portfolio_value,
        cash_balance=cash_balance,
        order_features=order_features
    )