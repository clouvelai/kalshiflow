"""
Cached Data Layer for Strategy Validation.

This module provides precomputed, cached data for fast validation.
On first run, it loads the raw trade data and computes market-level
aggregations. Subsequent runs use the cached parquet file, providing
10-20x speedup.

Key Responsibilities:
    1. Load and precompute market-level aggregations from trade data
    2. Cache to parquet with hash-based invalidation
    3. Build baseline win rates by price bucket
    4. Provide vectorized filtering for signal conditions

Cache Invalidation:
    The cache is invalidated when:
    - Source CSV file is modified (hash check)
    - Cache version changes (for schema updates)

Architecture Position:
    The CachedDataLayer is used by:
    - StrategyValidator: Primary consumer for market data
    - CLI tools: For listing available data

Design Principles:
    - Lazy loading: Only load data when needed
    - Hash-based invalidation: Automatic cache refresh
    - Memory efficient: Use parquet for on-disk caching
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SignalCondition

logger = logging.getLogger("kalshiflow.validation.cache")

# Cache version - increment when schema changes
# v1.1.0: Added temporal fields (avg_hour, weekend_trade_ratio, is_mostly_weekend, etc.)
#         Added trade size extremes (max_trade_size, min_trade_size, trade_size_range)
CACHE_VERSION = "1.1.0"


@dataclass
class BaselineStats:
    """Baseline statistics for a price bucket."""
    win_rate: float
    n_markets: int


@dataclass
class CachedDataLayer:
    """
    Precomputed data for fast validation.

    This class loads trade data, computes market-level aggregations,
    and caches results for fast subsequent access.

    Attributes:
        data_path: Path to the source CSV file
        cache_dir: Directory for cached parquet files
        markets_df: Precomputed market-level DataFrame
        baseline_cache: Win rates by price bucket
    """
    data_path: Path
    cache_dir: Path
    bucket_size: int = 5  # Price bucket size in cents
    _markets_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _trades_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _baseline_cache: Dict[int, BaselineStats] = field(default_factory=dict, repr=False)
    _source_hash: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize paths as Path objects."""
        self.data_path = Path(self.data_path)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def markets_df(self) -> pd.DataFrame:
        """Lazy-load the markets DataFrame."""
        if self._markets_df is None:
            self._load_or_compute_markets()
        return self._markets_df

    @property
    def trades_df(self) -> pd.DataFrame:
        """Lazy-load the trades DataFrame (for trade-level analysis)."""
        if self._trades_df is None:
            self._load_trades()
        return self._trades_df

    @property
    def baseline_cache(self) -> Dict[int, BaselineStats]:
        """Lazy-load the baseline cache."""
        if not self._baseline_cache:
            self._build_baseline()
        return self._baseline_cache

    def _compute_source_hash(self) -> str:
        """Compute a hash of the source file for cache invalidation."""
        # Use file size + mtime as a fast proxy for content hash
        stat = self.data_path.stat()
        hash_input = f"{stat.st_size}:{stat.st_mtime}:{CACHE_VERSION}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _get_cache_path(self) -> Path:
        """Get the cache file path based on source hash."""
        source_hash = self._compute_source_hash()
        return self.cache_dir / f"markets_cache_{source_hash}.parquet"

    def _is_cache_valid(self) -> bool:
        """Check if the cache is valid and up-to-date."""
        cache_path = self._get_cache_path()
        return cache_path.exists()

    def _load_or_compute_markets(self) -> None:
        """Load cached markets or compute from scratch."""
        cache_path = self._get_cache_path()

        if self._is_cache_valid():
            logger.info(f"Loading cached markets from {cache_path}")
            self._markets_df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(self._markets_df):,} markets from cache")
        else:
            logger.info("Cache miss - computing market aggregations...")
            self._compute_markets()
            # Save to cache
            self._markets_df.to_parquet(cache_path)
            logger.info(f"Saved cache to {cache_path}")

    def _load_trades(self) -> None:
        """Load raw trade data."""
        logger.info(f"Loading trades from {self.data_path}")
        self._trades_df = pd.read_csv(self.data_path)
        self._trades_df["datetime"] = pd.to_datetime(self._trades_df["datetime"])
        logger.info(f"Loaded {len(self._trades_df):,} trades")

    def _compute_markets(self) -> None:
        """
        Compute market-level aggregations from trade data.

        This is the expensive operation that we cache to avoid repeating.
        Computes all features needed for strategy validation.
        """
        # Load trades if needed
        if self._trades_df is None:
            self._load_trades()

        df = self._trades_df.copy()

        # Add derived columns for aggregation
        df["trade_value_cents"] = df["count"] * df["trade_price"]
        df["is_whale"] = df["trade_value_cents"] >= 10000  # $100+
        df["is_yes_trade"] = df["taker_side"] == "yes"
        df["is_no_trade"] = df["taker_side"] == "no"

        # Round sizes for bot detection
        round_sizes = [10, 25, 50, 100, 250, 500, 1000]
        df["is_round_size"] = df["count"].isin(round_sizes)

        # Extract temporal features
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Sort for first/last calculations
        df = df.sort_values(["market_ticker", "datetime"])

        # Aggregate to market level
        logger.info("Computing market-level aggregations...")

        market_agg = df.groupby("market_ticker").agg(
            # Result and basic info (column name is 'result' in the data)
            market_result=("result", "first"),

            # Trade counts
            n_trades=("count", "size"),
            total_contracts=("count", "sum"),

            # Trade direction ratios
            yes_trade_count=("is_yes_trade", "sum"),
            no_trade_count=("is_no_trade", "sum"),

            # Prices
            first_yes_price=("yes_price", "first"),
            last_yes_price=("yes_price", "last"),
            avg_yes_price=("yes_price", "mean"),
            yes_price_std=("yes_price", "std"),

            first_no_price=("no_price", "first"),
            last_no_price=("no_price", "last"),
            avg_no_price=("no_price", "mean"),

            # Trade values
            total_value_cents=("trade_value_cents", "sum"),
            avg_trade_value=("trade_value_cents", "mean"),
            avg_trade_size=("count", "mean"),

            # Whale activity
            whale_trade_count=("is_whale", "sum"),
            has_whale=("is_whale", "any"),

            # Timing
            first_trade_time=("datetime", "first"),
            last_trade_time=("datetime", "last"),

            # Leverage
            avg_leverage=("leverage_ratio", "mean"),
            leverage_std=("leverage_ratio", "std"),

            # Bot detection
            round_size_count=("is_round_size", "sum"),

            # Trade size extremes
            max_trade_size=("count", "max"),
            min_trade_size=("count", "min"),

            # Temporal features (mode/dominant values)
            avg_hour=("hour", "mean"),
            weekend_trade_ratio=("is_weekend", "mean"),

            # Category from event_ticker or market_ticker
            category=("market_ticker", lambda x: x.iloc[0].split("-")[0] if "-" in x.iloc[0] else x.iloc[0][:8])
        ).reset_index()

        # Compute derived fields
        market_agg["yes_trade_ratio"] = market_agg["yes_trade_count"] / market_agg["n_trades"]
        market_agg["no_trade_ratio"] = market_agg["no_trade_count"] / market_agg["n_trades"]

        # Price movement
        market_agg["yes_price_dropped"] = market_agg["last_yes_price"] < market_agg["first_yes_price"]
        market_agg["yes_price_drop"] = market_agg["first_yes_price"] - market_agg["last_yes_price"]
        market_agg["no_price_dropped"] = market_agg["last_no_price"] < market_agg["first_no_price"]
        market_agg["no_price_drop"] = market_agg["first_no_price"] - market_agg["last_no_price"]

        # Price movement toward NO (YES price dropped means NO is stronger)
        market_agg["price_moved_toward_no"] = market_agg["yes_price_dropped"]
        market_agg["price_move_toward_no"] = market_agg["yes_price_drop"]

        # Market duration
        market_agg["market_duration_hours"] = (
            (market_agg["last_trade_time"] - market_agg["first_trade_time"])
            .dt.total_seconds() / 3600
        )

        # Fill NaN values
        market_agg["yes_price_std"] = market_agg["yes_price_std"].fillna(0)
        market_agg["leverage_std"] = market_agg["leverage_std"].fillna(0)

        # Price buckets
        market_agg["no_price_bucket"] = (market_agg["avg_no_price"] // self.bucket_size) * self.bucket_size
        market_agg["yes_price_bucket"] = (market_agg["avg_yes_price"] // self.bucket_size) * self.bucket_size

        # Boolean flags for common conditions
        market_agg["is_resolved"] = market_agg["market_result"].isin(["yes", "no"])
        market_agg["no_won"] = market_agg["market_result"] == "no"
        market_agg["yes_won"] = market_agg["market_result"] == "yes"

        # Extract week number for temporal analysis
        market_agg["week"] = pd.to_datetime(market_agg["first_trade_time"]).dt.isocalendar().week

        # Derived temporal fields
        market_agg["is_mostly_weekend"] = market_agg["weekend_trade_ratio"] > 0.5
        market_agg["trade_size_range"] = market_agg["max_trade_size"] - market_agg["min_trade_size"]
        market_agg["is_late_night"] = (market_agg["avg_hour"] >= 22) | (market_agg["avg_hour"] <= 6)
        market_agg["is_morning"] = (market_agg["avg_hour"] >= 6) & (market_agg["avg_hour"] < 12)
        market_agg["is_afternoon"] = (market_agg["avg_hour"] >= 12) & (market_agg["avg_hour"] < 18)
        market_agg["is_evening"] = (market_agg["avg_hour"] >= 18) & (market_agg["avg_hour"] < 22)

        # Store result
        self._markets_df = market_agg

        logger.info(f"Computed aggregations for {len(market_agg):,} markets")

    def _build_baseline(self) -> None:
        """Build baseline win rates by price bucket."""
        markets = self.markets_df

        # Only use resolved markets for baseline
        resolved = markets[markets["is_resolved"]]

        logger.info(f"Building baseline from {len(resolved):,} resolved markets")

        for bucket in sorted(resolved["no_price_bucket"].unique()):
            bucket_markets = resolved[resolved["no_price_bucket"] == bucket]
            n = len(bucket_markets)

            if n >= 20:  # Minimum markets for reliable baseline
                win_rate = bucket_markets["no_won"].mean()
                self._baseline_cache[int(bucket)] = BaselineStats(
                    win_rate=float(win_rate),
                    n_markets=int(n)
                )

        logger.info(f"Built baseline across {len(self._baseline_cache)} price buckets")

    def get_baseline(self, bucket: int) -> Optional[BaselineStats]:
        """
        Get cached baseline for a price bucket.

        Args:
            bucket: Price bucket (e.g., 65 for 65-69c range)

        Returns:
            BaselineStats or None if bucket not found
        """
        return self.baseline_cache.get(int(bucket))

    def filter_markets(
        self,
        conditions: List[SignalCondition],
        resolved_only: bool = True
    ) -> pd.DataFrame:
        """
        Filter markets by signal conditions using vectorized operations.

        Args:
            conditions: List of SignalCondition objects
            resolved_only: If True, only return resolved markets

        Returns:
            Filtered DataFrame of markets
        """
        markets = self.markets_df.copy()

        if resolved_only:
            markets = markets[markets["is_resolved"]]

        # Apply each condition
        for cond in conditions:
            field = cond.field
            op = cond.operator
            value = cond.value

            if field not in markets.columns:
                # Check for field mapping (handle aliases)
                field_mapping = {
                    "trade_count": "n_trades",
                    "price_dropped": "yes_price_dropped",
                    "price_move_toward_no": "yes_price_drop",
                }
                if field in field_mapping:
                    field = field_mapping[field]
                else:
                    logger.warning(f"Unknown field '{cond.field}', skipping condition")
                    continue

            # Apply operator
            if op == ">":
                markets = markets[markets[field] > value]
            elif op == ">=":
                markets = markets[markets[field] >= value]
            elif op == "<":
                markets = markets[markets[field] < value]
            elif op == "<=":
                markets = markets[markets[field] <= value]
            elif op == "==":
                markets = markets[markets[field] == value]
            elif op == "!=":
                markets = markets[markets[field] != value]
            elif op == "in":
                markets = markets[markets[field].isin(value)]
            elif op == "not_in":
                markets = markets[~markets[field].isin(value)]

        return markets

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the cached data."""
        markets = self.markets_df

        return {
            "data_path": str(self.data_path),
            "cache_path": str(self._get_cache_path()),
            "total_markets": len(markets),
            "resolved_markets": int(markets["is_resolved"].sum()),
            "total_trades": int(markets["n_trades"].sum()),
            "date_range": {
                "start": str(markets["first_trade_time"].min()),
                "end": str(markets["last_trade_time"].max())
            },
            "baseline_buckets": len(self.baseline_cache)
        }

    def clear_cache(self) -> None:
        """Clear all cached parquet files."""
        for cache_file in self.cache_dir.glob("markets_cache_*.parquet"):
            cache_file.unlink()
            logger.info(f"Deleted cache file: {cache_file}")


def create_cache(
    data_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    bucket_size: int = 5
) -> CachedDataLayer:
    """
    Factory function to create a CachedDataLayer with default paths.

    Args:
        data_path: Path to source CSV (defaults to enriched_trades_resolved_ALL.csv)
        cache_dir: Cache directory (defaults to research/data/cache/)
        bucket_size: Price bucket size in cents

    Returns:
        Configured CachedDataLayer instance
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent

    if data_path is None:
        data_path = project_root / "research" / "data" / "trades" / "enriched_trades_resolved_ALL.csv"

    if cache_dir is None:
        cache_dir = project_root / "research" / "data" / "cache"

    return CachedDataLayer(
        data_path=Path(data_path),
        cache_dir=Path(cache_dir),
        bucket_size=bucket_size
    )
