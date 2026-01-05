#!/usr/bin/env python3
"""
Market Maker Strategy Analysis for Kalshi Prediction Markets
=============================================================

This script performs comprehensive analysis of market making hypotheses:
1. Spread dynamics by market category
2. MM-002: Sustained NO Imbalance (consecutive NO-biased trades)
3. MM-003: Large Order + Direction Match (whale following validation)
4. MM-004: Thin BBO Depth = Adverse Selection (trade size as depth proxy)
5. Market microstructure profiling (time-of-day, category liquidity)

Uses rigorous bucket-matched validation methodology:
- Minimum 50 unique markets
- No single market > 30% of profit (concentration check)
- Statistical significance (p < 0.05)
- Proper breakeven calculations

Author: Quant Research Agent
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "trades"
ENRICHED_TRADES = DATA_DIR / "enriched_trades_resolved_ALL.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def extract_market_category(ticker: str) -> str:
    """Extract the market category prefix from a ticker."""
    match = re.match(r'^(KX[A-Z]+)', ticker)
    return match.group(1) if match else 'UNKNOWN'


def extract_base_market(ticker: str) -> str:
    """Extract the base market (game/event) from a ticker, removing team suffix."""
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker


def get_price_bucket(price: float) -> str:
    """Get the 10-cent price bucket."""
    if pd.isna(price):
        return 'unknown'
    bucket = int(price // 10) * 10
    return f"{bucket}-{bucket+10}"


def get_5c_price_bucket(price: float) -> str:
    """Get 5-cent price bucket for finer analysis."""
    if pd.isna(price):
        return 'unknown'
    bucket = int(price // 5) * 5
    return f"{bucket}-{bucket+5}"


def calculate_breakeven(price: float, side: str) -> float:
    """Calculate breakeven win rate for a given price and side."""
    if side == 'yes':
        return price / 100.0
    else:
        return (100 - price) / 100.0


def binomial_test(wins: int, total: int, null_rate: float) -> float:
    """Calculate p-value for win rate being different from null rate."""
    if total == 0 or null_rate <= 0 or null_rate >= 1:
        return 1.0
    try:
        result = stats.binomtest(wins, total, null_rate, alternative='greater')
        return result.pvalue
    except Exception:
        return 1.0


def calculate_max_market_share(profits_by_market: pd.Series) -> float:
    """Calculate the maximum share of profit from any single market."""
    if len(profits_by_market) == 0:
        return 1.0
    positive_profits = profits_by_market[profits_by_market > 0]
    if len(positive_profits) == 0 or positive_profits.sum() == 0:
        return 1.0
    return positive_profits.max() / positive_profits.sum()


class MarketMakerAnalyzer:
    """
    Comprehensive market maker strategy analyzer for Kalshi markets.
    """

    def __init__(self, data_path: Path = ENRICHED_TRADES):
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df):,} trades across {self.df['market_ticker'].nunique():,} unique markets")

        # Parse datetime
        self.df['dt'] = pd.to_datetime(self.df['datetime'])
        self.df['hour'] = self.df['dt'].dt.hour
        self.df['minute'] = self.df['dt'].dt.minute
        self.df['day_of_week'] = self.df['dt'].dt.dayofweek
        self.df['date'] = self.df['dt'].dt.date

        # Parse category from ticker
        self.df['category'] = self.df['market_ticker'].apply(extract_market_category)
        self.df['base_market'] = self.df['market_ticker'].apply(extract_base_market)

        # Create price buckets
        self.df['price_bucket'] = self.df['trade_price'].apply(get_price_bucket)
        self.df['price_bucket_5c'] = self.df['trade_price'].apply(get_5c_price_bucket)

        # Calculate is_winner for trades (did taker win?)
        self.df['is_winner'] = (
            ((self.df['taker_side'] == 'yes') & (self.df['result'] == 'yes')) |
            ((self.df['taker_side'] == 'no') & (self.df['result'] == 'no'))
        )

        # Calculate actual profit
        self.df['actual_profit_dollars'] = np.where(
            self.df['is_winner'],
            self.df['potential_profit_dollars'],
            -self.df['cost_dollars']
        )

        # Filter to resolved markets only
        self.resolved_df = self.df[self.df['status'].isin(['finalized', 'determined'])].copy()
        print(f"Resolved trades: {len(self.resolved_df):,}")

        # Results storage
        self.results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_trades': len(self.df),
                'resolved_trades': len(self.resolved_df),
                'unique_markets': self.df['market_ticker'].nunique(),
                'unique_base_markets': self.df['base_market'].nunique(),
            },
            'spread_dynamics': {},
            'mm_hypotheses': {},
            'microstructure': {},
            'validated_strategies': [],
            'rejected_strategies': [],
        }

    def calculate_market_level_stats(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        min_markets: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate statistics at the MARKET level with proper validation.
        """
        if len(df) == 0:
            return None

        # Build aggregation dict based on available columns
        agg_dict = {
            'is_winner': 'first',
            'actual_profit_dollars': 'sum',
            'cost_dollars': 'sum',
            'trade_price': 'mean',
            'taker_side': 'first',
        }

        # Add optional columns if they exist
        if 'result' in df.columns:
            agg_dict['result'] = 'first'

        # Group by base market to avoid counting same game multiple times
        market_stats = df.groupby('base_market').agg(agg_dict)
        market_stats['trade_count'] = df.groupby('base_market').size()

        n_markets = len(market_stats)
        if n_markets < 10:
            return None

        # Market-level win rate
        n_wins = int(market_stats['is_winner'].sum())
        win_rate = n_wins / n_markets

        # Total P/L
        total_profit = float(market_stats['actual_profit_dollars'].sum())
        total_cost = float(market_stats['cost_dollars'].sum())
        roi = total_profit / total_cost if total_cost > 0 else 0

        # Concentration check
        max_share = calculate_max_market_share(market_stats['actual_profit_dollars'])

        # Calculate breakeven
        avg_price = df['trade_price'].mean()
        mode_side = df['taker_side'].mode().iloc[0] if len(df['taker_side'].mode()) > 0 else 'yes'
        breakeven = calculate_breakeven(avg_price, mode_side)

        # Edge
        edge = win_rate - breakeven

        # P-value
        p_value = binomial_test(n_wins, n_markets, breakeven)

        # Validation
        passes_market_count = n_markets >= min_markets
        passes_concentration = max_share < 0.30
        passes_significance = p_value < 0.05
        is_valid = passes_market_count and passes_concentration and passes_significance and edge > 0

        return {
            'strategy': strategy_name,
            'n_trades': len(df),
            'n_markets': n_markets,
            'n_wins': n_wins,
            'win_rate': round(win_rate, 4),
            'breakeven': round(breakeven, 4),
            'edge': round(edge, 4),
            'edge_pct': round(edge * 100, 2),
            'total_profit': round(total_profit, 2),
            'total_cost': round(total_cost, 2),
            'roi': round(roi, 4),
            'roi_pct': round(roi * 100, 2),
            'max_market_share': round(max_share, 4),
            'p_value': round(p_value, 6),
            'avg_price': round(avg_price, 2),
            'side': mode_side,
            'passes_market_count': passes_market_count,
            'passes_concentration': passes_concentration,
            'passes_significance': passes_significance,
            'is_valid': is_valid
        }

    def analyze_spread_dynamics(self) -> Dict[str, Any]:
        """
        Analyze spread dynamics by market category.

        Since we don't have real-time orderbook spread data, we use
        bid-ask spread proxies:
        - Effective spread = |trade_price - mid| where mid = 50 for binary
        - Trade size as liquidity proxy
        - Category-level spread characteristics
        """
        print("\n" + "="*80)
        print("SPREAD DYNAMICS ANALYSIS")
        print("="*80)

        results = {
            'by_category': {},
            'around_price_moves': {},
            'theoretical_spread_capture': {}
        }

        # 1. Effective spread proxy by category
        # For binary markets, "spread" from 50c represents distance from uncertainty
        self.resolved_df['price_distance_from_50'] = abs(self.resolved_df['trade_price'] - 50)

        category_spread = self.resolved_df.groupby('category').agg({
            'price_distance_from_50': ['mean', 'std', 'median'],
            'count': ['mean', 'std', 'median', 'sum'],  # Trade size as liquidity proxy
            'trade_price': ['mean', 'std'],
            'market_ticker': 'nunique',
            'base_market': 'nunique',
        })
        category_spread.columns = ['_'.join(col).strip() for col in category_spread.columns]

        # Only keep categories with significant volume
        significant_cats = category_spread[category_spread['market_ticker_nunique'] >= 100].copy()

        print(f"\nCategory-level spread characteristics ({len(significant_cats)} categories):")
        print("-" * 90)
        print(f"{'Category':<20} {'Markets':>8} {'Avg Price':>10} {'Avg Size':>10} {'Dist 50':>10} {'Volatility':>10}")
        print("-" * 90)

        for cat in significant_cats.index:
            row = significant_cats.loc[cat]
            results['by_category'][cat] = {
                'unique_markets': int(row['market_ticker_nunique']),
                'unique_base_markets': int(row['base_market_nunique']),
                'avg_price': round(row['trade_price_mean'], 2),
                'price_volatility': round(row['trade_price_std'], 2),
                'avg_trade_size': round(row['count_mean'], 1),
                'median_trade_size': round(row['count_median'], 1),
                'total_volume': int(row['count_sum']),
                'avg_distance_from_50': round(row['price_distance_from_50_mean'], 2),
            }
            print(f"{cat:<20} {int(row['market_ticker_nunique']):>8} {row['trade_price_mean']:>10.1f} "
                  f"{row['count_mean']:>10.1f} {row['price_distance_from_50_mean']:>10.1f} "
                  f"{row['trade_price_std']:>10.1f}")

        # 2. Spread dynamics around significant price moves
        print("\n\nSpread behavior around significant price moves (>10c):")
        print("-" * 70)

        # Sort by market and time to find price moves
        sorted_df = self.resolved_df.sort_values(['base_market', 'timestamp']).copy()
        sorted_df['prev_price'] = sorted_df.groupby('base_market')['trade_price'].shift(1)
        sorted_df['price_move'] = sorted_df['trade_price'] - sorted_df['prev_price']
        sorted_df['abs_price_move'] = abs(sorted_df['price_move'])

        # Find significant moves
        sig_moves = sorted_df[sorted_df['abs_price_move'] > 10].copy()
        print(f"Found {len(sig_moves):,} trades with >10c price moves")

        if len(sig_moves) > 0:
            # Analyze trade sizes around big moves
            big_move_stats = sig_moves.groupby(
                pd.cut(sig_moves['abs_price_move'], bins=[10, 15, 20, 30, 50, 100])
            ).agg({
                'count': ['mean', 'median', 'count'],
                'is_winner': 'mean'
            })
            big_move_stats.columns = ['avg_size', 'median_size', 'n_trades', 'win_rate']

            print("\nMove Size     Avg Trade Size  Median Size   N Trades   Win Rate")
            print("-" * 70)
            for idx, row in big_move_stats.iterrows():
                print(f"{str(idx):<12} {row['avg_size']:>14.1f} {row['median_size']:>12.1f} "
                      f"{int(row['n_trades']):>10} {row['win_rate']:>10.1%}")

            results['around_price_moves'] = {
                'total_significant_moves': len(sig_moves),
                'by_move_size': {str(k): v for k, v in big_move_stats.to_dict('index').items()}
            }

        # 3. Theoretical spread capture opportunities
        print("\n\nTheoretical spread capture by price bucket:")
        print("-" * 70)

        # Calculate what a market maker would earn crossing the spread
        # Spread capture = 100 - yes_price - no_price (in cents, should be ~1-3c on liquid markets)
        # We estimate this using trade price clustering

        for bucket in ['0-10', '10-20', '20-30', '30-40', '40-50',
                       '50-60', '60-70', '70-80', '80-90', '90-100']:
            bucket_df = self.resolved_df[self.resolved_df['price_bucket'] == bucket]
            if len(bucket_df) > 0:
                # Use yes_price and no_price to estimate spread
                if 'yes_price' in bucket_df.columns and 'no_price' in bucket_df.columns:
                    bucket_df = bucket_df[bucket_df['yes_price'].notna() & bucket_df['no_price'].notna()]
                    if len(bucket_df) > 0:
                        implied_spread = 100 - bucket_df['yes_price'] - bucket_df['no_price']
                        results['theoretical_spread_capture'][bucket] = {
                            'n_trades': len(bucket_df),
                            'avg_implied_spread': round(implied_spread.mean(), 2),
                            'median_implied_spread': round(implied_spread.median(), 2),
                            'pct_positive_spread': round((implied_spread > 0).mean() * 100, 1),
                        }
                        print(f"{bucket}c: Avg spread={implied_spread.mean():.2f}c, "
                              f"Median={implied_spread.median():.2f}c, "
                              f"Positive={((implied_spread > 0).mean()*100):.1f}%")

        self.results['spread_dynamics'] = results
        return results

    def analyze_mm002_sustained_no_imbalance(self) -> Dict[str, Any]:
        """
        MM-002: Sustained NO Imbalance Hypothesis

        Definition: When there are 3+ consecutive NO trades in a market,
        bet NO on the next opportunity.

        Rationale: Sustained NO flow may indicate informed selling.
        """
        print("\n" + "="*80)
        print("MM-002: SUSTAINED NO IMBALANCE ANALYSIS")
        print("="*80)

        results = {'hypothesis': 'MM-002: Sustained NO Imbalance', 'tests': []}

        # Sort by market and time
        sorted_df = self.resolved_df.sort_values(['base_market', 'timestamp']).copy()
        print(f"Processing {len(sorted_df):,} trades for consecutive NO analysis...")

        # Calculate consecutive NO trades using vectorized approach
        sorted_df['is_no'] = (sorted_df['taker_side'] == 'no').astype(int)

        # Identify where consecutive runs break (market changes or side changes from NO)
        sorted_df['market_changed'] = sorted_df['base_market'] != sorted_df['base_market'].shift(1)
        sorted_df['run_break'] = sorted_df['market_changed'] | (sorted_df['is_no'] == 0)

        # Create run group IDs
        sorted_df['run_group'] = sorted_df['run_break'].cumsum()

        # Count consecutive NOs within each run group (only for NO trades)
        sorted_df['consecutive_no'] = sorted_df.groupby('run_group')['is_no'].cumsum()
        # For non-NO trades, consecutive_no should be 0
        sorted_df.loc[sorted_df['is_no'] == 0, 'consecutive_no'] = 0

        consec_df = sorted_df[['base_market', 'taker_side', 'trade_price', 'is_winner',
                               'result', 'actual_profit_dollars', 'cost_dollars', 'consecutive_no']].copy()

        # Test different thresholds of consecutive NO trades
        for min_consecutive in [2, 3, 4, 5]:
            print(f"\n--- Consecutive NO >= {min_consecutive} ---")

            # Trades that follow sustained NO imbalance
            sustained_no = consec_df[consec_df['consecutive_no'] >= min_consecutive]

            if len(sustained_no) < 100:
                print(f"  Insufficient data: {len(sustained_no)} trades")
                continue

            # Filter to NO trades only (we're testing whether continued NO is profitable)
            no_after_sustained = sustained_no[sustained_no['taker_side'] == 'no']

            # Group by market for proper statistics
            for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]:
                price_mask = (no_after_sustained['trade_price'] >= price_low) & \
                             (no_after_sustained['trade_price'] < price_high)
                bucket_df = no_after_sustained[price_mask]

                if len(bucket_df) < 50:
                    continue

                strategy_name = f"NO after {min_consecutive}+ consecutive NO @ {price_low}-{price_high}c"
                stats = self.calculate_market_level_stats(bucket_df, strategy_name)

                if stats:
                    results['tests'].append(stats)
                    status = "VALID" if stats['is_valid'] else ""
                    print(f"  {status:5} {price_low}-{price_high}c: {stats['n_markets']} mkts, "
                          f"WR={stats['win_rate']:.1%}, BE={stats['breakeven']:.1%}, "
                          f"Edge={stats['edge']:+.1%}, Profit=${stats['total_profit']:,.0f}")

        # Also test YES trades after NO runs (contrarian)
        print("\n--- Contrarian: YES after sustained NO ---")
        for min_consecutive in [3, 4, 5]:
            sustained_no = consec_df[consec_df['consecutive_no'] >= min_consecutive]
            yes_after_no = sustained_no[sustained_no['taker_side'] == 'yes']

            if len(yes_after_no) >= 50:
                stats = self.calculate_market_level_stats(
                    yes_after_no,
                    f"YES after {min_consecutive}+ consecutive NO"
                )
                if stats:
                    results['tests'].append(stats)
                    status = "VALID" if stats['is_valid'] else ""
                    print(f"  {status:5} {min_consecutive}+ NO: {stats['n_markets']} mkts, "
                          f"Edge={stats['edge']:+.1%}, Profit=${stats['total_profit']:,.0f}")

        self.results['mm_hypotheses']['MM-002'] = results
        return results

    def analyze_mm003_large_order_direction(self) -> Dict[str, Any]:
        """
        MM-003: Large Order + Direction Match Hypothesis

        Definition: Follow large orders (top 10% by size) when they match
        the subsequent price direction.

        Rationale: Large orders may represent informed trader activity.
        """
        print("\n" + "="*80)
        print("MM-003: LARGE ORDER + DIRECTION MATCH ANALYSIS")
        print("="*80)

        results = {'hypothesis': 'MM-003: Large Order + Direction Match', 'tests': []}

        # Define whale thresholds
        size_percentiles = self.resolved_df['count'].quantile([0.90, 0.95, 0.99])
        print(f"Trade size percentiles: 90th={size_percentiles[0.90]:.0f}, "
              f"95th={size_percentiles[0.95]:.0f}, 99th={size_percentiles[0.99]:.0f}")

        # Sort by market and time
        sorted_df = self.resolved_df.sort_values(['base_market', 'timestamp']).copy()

        # Calculate price movement after each trade
        sorted_df['next_price'] = sorted_df.groupby('base_market')['trade_price'].shift(-1)
        sorted_df['price_change'] = sorted_df['next_price'] - sorted_df['trade_price']
        sorted_df['price_direction'] = np.sign(sorted_df['price_change'])

        # Test different size thresholds
        for threshold_name, min_size in [
            ('Top 10% (Whale)', size_percentiles[0.90]),
            ('Top 5% (Large Whale)', size_percentiles[0.95]),
            ('Top 1% (Mega Whale)', size_percentiles[0.99]),
            ('100+ contracts', 100),
            ('500+ contracts', 500),
            ('1000+ contracts', 1000),
        ]:
            print(f"\n--- {threshold_name} (size >= {min_size:.0f}) ---")

            large_orders = sorted_df[sorted_df['count'] >= min_size].copy()

            if len(large_orders) < 100:
                print(f"  Insufficient data: {len(large_orders)} trades")
                continue

            # YES trades: Following large YES orders
            large_yes = large_orders[large_orders['taker_side'] == 'yes']
            # Direction match: price went up after (price_direction > 0)
            yes_direction_match = large_yes[large_yes['price_direction'] > 0]

            # NO trades: Following large NO orders
            large_no = large_orders[large_orders['taker_side'] == 'no']
            # Direction match: price went down after (price_direction < 0)
            no_direction_match = large_no[large_no['price_direction'] < 0]

            # Test at different price levels
            for side, direction_df, side_name in [
                ('yes', yes_direction_match, 'YES'),
                ('no', no_direction_match, 'NO'),
            ]:
                for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]:
                    price_mask = (direction_df['trade_price'] >= price_low) & \
                                 (direction_df['trade_price'] < price_high)
                    bucket_df = direction_df[price_mask]

                    if len(bucket_df) < 30:
                        continue

                    strategy_name = f"{threshold_name} {side_name} + Direction Match @ {price_low}-{price_high}c"
                    stats = self.calculate_market_level_stats(bucket_df, strategy_name, min_markets=30)

                    if stats:
                        results['tests'].append(stats)
                        if stats['is_valid'] or stats['edge'] > 0.03:
                            status = "VALID" if stats['is_valid'] else "MAYBE"
                            print(f"  {status:5} {side_name} @ {price_low}-{price_high}c: "
                                  f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

            # Also test just following large orders (without direction match)
            print(f"\n  Simple following (no direction match):")
            for side in ['yes', 'no']:
                side_df = large_orders[large_orders['taker_side'] == side]
                for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 100)]:
                    price_mask = (side_df['trade_price'] >= price_low) & \
                                 (side_df['trade_price'] < price_high)
                    bucket_df = side_df[price_mask]

                    if len(bucket_df) >= 50:
                        stats = self.calculate_market_level_stats(
                            bucket_df,
                            f"{threshold_name} {side.upper()} @ {price_low}-{price_high}c (simple)"
                        )
                        if stats and (stats['is_valid'] or stats['edge'] > 0.05):
                            status = "VALID" if stats['is_valid'] else "MAYBE"
                            print(f"    {status:5} {side.upper()} @ {price_low}-{price_high}c: "
                                  f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

        self.results['mm_hypotheses']['MM-003'] = results
        return results

    def analyze_mm004_thin_depth_adverse_selection(self) -> Dict[str, Any]:
        """
        MM-004: Thin BBO Depth = Adverse Selection Hypothesis

        Definition: Use trade size as a proxy for depth. Small trades
        may indicate thin markets where adverse selection is higher.

        Rationale: In thin markets, trades are more likely to be informed.
        """
        print("\n" + "="*80)
        print("MM-004: THIN BBO DEPTH = ADVERSE SELECTION ANALYSIS")
        print("="*80)

        results = {'hypothesis': 'MM-004: Thin Depth = Adverse Selection', 'tests': []}

        # Use trade size as depth proxy (small trades = thin depth)
        size_percentiles = self.resolved_df['count'].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"Trade size percentiles:")
        print(f"  10th={size_percentiles[0.10]:.0f}, 25th={size_percentiles[0.25]:.0f}, "
              f"50th={size_percentiles[0.50]:.0f}")
        print(f"  75th={size_percentiles[0.75]:.0f}, 90th={size_percentiles[0.90]:.0f}")

        # Also look at market-level trade count as liquidity proxy
        market_trade_counts = self.resolved_df.groupby('base_market').size()
        market_percentiles = market_trade_counts.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"\nMarket trade count percentiles:")
        print(f"  10th={market_percentiles[0.10]:.0f}, 25th={market_percentiles[0.25]:.0f}, "
              f"50th={market_percentiles[0.50]:.0f}")

        # Test 1: Small trades vs large trades by outcome
        print("\n--- Trade Size Impact on Outcome ---")

        size_buckets = [
            ('Micro (<10)', (0, 10)),
            ('Small (10-50)', (10, 50)),
            ('Medium (50-100)', (50, 100)),
            ('Large (100-500)', (100, 500)),
            ('Whale (500+)', (500, float('inf')))
        ]

        for bucket_name, (size_low, size_high) in size_buckets:
            size_mask = (self.resolved_df['count'] >= size_low) & \
                        (self.resolved_df['count'] < size_high)
            bucket_df = self.resolved_df[size_mask]

            if len(bucket_df) < 100:
                continue

            for side in ['yes', 'no']:
                side_df = bucket_df[bucket_df['taker_side'] == side]
                for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]:
                    price_mask = (side_df['trade_price'] >= price_low) & \
                                 (side_df['trade_price'] < price_high)
                    test_df = side_df[price_mask]

                    if len(test_df) >= 50:
                        strategy_name = f"{bucket_name} {side.upper()} @ {price_low}-{price_high}c"
                        stats = self.calculate_market_level_stats(test_df, strategy_name)

                        if stats and (stats['is_valid'] or abs(stats['edge']) > 0.03):
                            results['tests'].append(stats)
                            status = "VALID" if stats['is_valid'] else ""
                            print(f"  {status:5} {bucket_name} {side.upper()} @ {price_low}-{price_high}c: "
                                  f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

        # Test 2: Illiquid markets (low trade count) vs liquid markets
        print("\n--- Market Liquidity Impact ---")

        self.resolved_df['market_trade_count'] = self.resolved_df.groupby('base_market')['base_market'].transform('count')

        liquidity_buckets = [
            ('Illiquid (<10 trades)', (0, 10)),
            ('Low (10-50 trades)', (10, 50)),
            ('Medium (50-200 trades)', (50, 200)),
            ('High (200+ trades)', (200, float('inf')))
        ]

        for liq_name, (count_low, count_high) in liquidity_buckets:
            liq_mask = (self.resolved_df['market_trade_count'] >= count_low) & \
                       (self.resolved_df['market_trade_count'] < count_high)
            liq_df = self.resolved_df[liq_mask]

            if len(liq_df) < 100:
                continue

            for side in ['yes', 'no']:
                side_df = liq_df[liq_df['taker_side'] == side]
                stats = self.calculate_market_level_stats(side_df, f"{liq_name} {side.upper()} (all prices)")

                if stats:
                    results['tests'].append(stats)
                    if stats['is_valid'] or abs(stats['edge']) > 0.02:
                        status = "VALID" if stats['is_valid'] else ""
                        print(f"  {status:5} {liq_name} {side.upper()}: "
                              f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

        # Test 3: Small trades in illiquid markets (double adverse selection)
        print("\n--- Small Trades in Illiquid Markets (Double Adverse Selection) ---")

        small_in_illiquid = self.resolved_df[
            (self.resolved_df['count'] < 50) &
            (self.resolved_df['market_trade_count'] < 50)
        ]

        for side in ['yes', 'no']:
            side_df = small_in_illiquid[small_in_illiquid['taker_side'] == side]
            for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 100)]:
                price_mask = (side_df['trade_price'] >= price_low) & \
                             (side_df['trade_price'] < price_high)
                test_df = side_df[price_mask]

                if len(test_df) >= 50:
                    strategy_name = f"Small+Illiquid {side.upper()} @ {price_low}-{price_high}c"
                    stats = self.calculate_market_level_stats(test_df, strategy_name)

                    if stats:
                        results['tests'].append(stats)
                        status = "VALID" if stats['is_valid'] else ""
                        print(f"  {status:5} {side.upper()} @ {price_low}-{price_high}c: "
                              f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

        self.results['mm_hypotheses']['MM-004'] = results
        return results

    def analyze_microstructure(self) -> Dict[str, Any]:
        """
        Market Microstructure Profiling:
        - Trade flow toxicity by time of day
        - Category-specific liquidity characteristics
        - Spread behavior patterns
        """
        print("\n" + "="*80)
        print("MARKET MICROSTRUCTURE PROFILING")
        print("="*80)

        results = {
            'time_of_day': {},
            'category_liquidity': {},
            'market_close_effects': {}
        }

        # 1. Time of Day Analysis
        print("\n--- Trade Flow by Time of Day ---")

        hourly_stats = self.resolved_df.groupby('hour').agg({
            'market_ticker': 'count',
            'base_market': 'nunique',
            'count': ['mean', 'median', 'sum'],
            'is_winner': 'mean',
            'trade_price': 'mean',
            'cost_dollars': 'sum'
        })
        hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns]

        print(f"{'Hour':>6} {'Trades':>10} {'Markets':>10} {'Avg Size':>10} {'Win Rate':>10} {'Avg Price':>10}")
        print("-" * 66)

        for hour in range(24):
            if hour in hourly_stats.index:
                row = hourly_stats.loc[hour]
                results['time_of_day'][hour] = {
                    'trades': int(row['market_ticker_count']),
                    'unique_markets': int(row['base_market_nunique']),
                    'avg_trade_size': round(row['count_mean'], 1),
                    'taker_win_rate': round(row['is_winner_mean'], 4),
                    'avg_price': round(row['trade_price_mean'], 2),
                }
                print(f"{hour:>6} {int(row['market_ticker_count']):>10} "
                      f"{int(row['base_market_nunique']):>10} "
                      f"{row['count_mean']:>10.1f} {row['is_winner_mean']:>10.1%} "
                      f"{row['trade_price_mean']:>10.1f}")

        # Find interesting time windows
        print("\n--- Time-Based Edge Analysis ---")

        time_windows = [
            ('Late Night (0-4)', (0, 4)),
            ('Early Morning (4-8)', (4, 8)),
            ('Morning (8-12)', (8, 12)),
            ('Afternoon (12-16)', (12, 16)),
            ('Evening (16-20)', (16, 20)),
            ('Night (20-24)', (20, 24)),
            ('Weekend Nights (Fri-Sun 22-02)', None),  # Special handling
        ]

        for window_name, hours in time_windows:
            if hours:
                time_mask = (self.resolved_df['hour'] >= hours[0]) & \
                            (self.resolved_df['hour'] < hours[1])
            else:
                # Weekend nights special case
                weekend_mask = self.resolved_df['day_of_week'].isin([4, 5, 6])  # Fri, Sat, Sun
                night_mask = (self.resolved_df['hour'] >= 22) | (self.resolved_df['hour'] < 2)
                time_mask = weekend_mask & night_mask

            window_df = self.resolved_df[time_mask]

            for side in ['yes', 'no']:
                for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 100)]:
                    side_mask = (window_df['taker_side'] == side) & \
                                (window_df['trade_price'] >= price_low) & \
                                (window_df['trade_price'] < price_high)
                    test_df = window_df[side_mask]

                    if len(test_df) >= 100:
                        strategy_name = f"{window_name} {side.upper()} @ {price_low}-{price_high}c"
                        stats = self.calculate_market_level_stats(test_df, strategy_name)

                        if stats and (stats['is_valid'] or stats['edge'] > 0.03):
                            status = "VALID" if stats['is_valid'] else ""
                            print(f"  {status:5} {window_name} {side.upper()} @ {price_low}-{price_high}c: "
                                  f"{stats['n_markets']} mkts, Edge={stats['edge']:+.1%}")

                            if stats['is_valid']:
                                self.results['validated_strategies'].append(stats)

        # 2. Category-Specific Liquidity Analysis
        print("\n--- Category Liquidity Characteristics ---")

        category_liq = self.resolved_df.groupby('category').agg({
            'market_ticker': 'count',
            'base_market': 'nunique',
            'count': ['mean', 'median', 'std', 'sum'],
            'is_winner': 'mean',
            'leverage_ratio': 'mean',
        })
        category_liq.columns = ['_'.join(col).strip() for col in category_liq.columns]
        category_liq = category_liq.sort_values('count_sum', ascending=False)

        print(f"{'Category':<18} {'Markets':>8} {'Trades':>10} {'Avg Size':>10} {'Size Std':>10} {'Lev Ratio':>10}")
        print("-" * 76)

        for cat in category_liq.head(20).index:
            row = category_liq.loc[cat]
            results['category_liquidity'][cat] = {
                'unique_markets': int(row['base_market_nunique']),
                'total_trades': int(row['market_ticker_count']),
                'total_volume': int(row['count_sum']),
                'avg_trade_size': round(row['count_mean'], 1),
                'trade_size_std': round(row['count_std'], 1),
                'avg_leverage': round(row['leverage_ratio_mean'], 3),
                'taker_win_rate': round(row['is_winner_mean'], 4),
            }
            print(f"{cat:<18} {int(row['base_market_nunique']):>8} "
                  f"{int(row['market_ticker_count']):>10} "
                  f"{row['count_mean']:>10.1f} {row['count_std']:>10.1f} "
                  f"{row['leverage_ratio_mean']:>10.3f}")

        self.results['microstructure'] = results
        return results

    def run_bucket_matched_validation(self) -> Dict[str, Any]:
        """
        Run proper bucket-matched validation for promising signals.

        For each strategy, compare against baseline (same price bucket + side).
        Only validate if improvement over baseline is significant.
        """
        print("\n" + "="*80)
        print("BUCKET-MATCHED VALIDATION")
        print("="*80)

        validation_results = []

        # First, calculate baseline for each price bucket + side
        baselines = {}
        for price_bucket in ['0-10', '10-20', '20-30', '30-40', '40-50',
                             '50-60', '60-70', '70-80', '80-90', '90-100']:
            for side in ['yes', 'no']:
                mask = (self.resolved_df['price_bucket'] == price_bucket) & \
                       (self.resolved_df['taker_side'] == side)
                baseline_df = self.resolved_df[mask]

                if len(baseline_df) >= 50:
                    stats = self.calculate_market_level_stats(
                        baseline_df,
                        f"Baseline {side.upper()} @ {price_bucket}c"
                    )
                    if stats:
                        baselines[(price_bucket, side)] = stats

        print(f"\nCalculated {len(baselines)} baseline buckets")
        print("\n--- Baseline Win Rates ---")
        for (bucket, side), stats in sorted(baselines.items()):
            print(f"  {side.upper():>3} @ {bucket}c: WR={stats['win_rate']:.1%}, "
                  f"BE={stats['breakeven']:.1%}, Edge={stats['edge']:+.1%}")

        # Now validate any promising strategies against their baselines
        print("\n--- Improvement Over Baseline ---")

        all_hypothesis_tests = []
        for hyp_key in ['MM-002', 'MM-003', 'MM-004']:
            if hyp_key in self.results['mm_hypotheses']:
                all_hypothesis_tests.extend(self.results['mm_hypotheses'][hyp_key].get('tests', []))

        for test in all_hypothesis_tests:
            if not test.get('is_valid'):
                continue

            # Find matching baseline
            bucket = None
            side = test.get('side', 'yes')

            # Parse price from strategy name
            strategy = test.get('strategy', '')
            for pb in ['0-10', '10-20', '20-30', '30-40', '40-50',
                       '50-60', '60-70', '70-80', '80-90', '90-100']:
                if pb in strategy or f"{pb.replace('-', '-')}c" in strategy:
                    bucket = pb
                    break

            if not bucket or (bucket, side) not in baselines:
                continue

            baseline = baselines[(bucket, side)]
            improvement = test['win_rate'] - baseline['win_rate']

            validation_results.append({
                'strategy': test['strategy'],
                'strategy_win_rate': test['win_rate'],
                'baseline_win_rate': baseline['win_rate'],
                'improvement': round(improvement, 4),
                'improvement_pct': round(improvement * 100, 2),
                'strategy_edge': test['edge'],
                'baseline_edge': baseline['edge'],
                'is_genuine_improvement': improvement > 0.02,  # >2% improvement
            })

            status = "GENUINE" if improvement > 0.02 else "PRICE PROXY"
            print(f"  {status:12} {test['strategy'][:50]}")
            print(f"              Strategy WR={test['win_rate']:.1%}, Baseline WR={baseline['win_rate']:.1%}, "
                  f"Improvement={improvement:+.1%}")

        return validation_results

    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        # Run all analyses
        print("\n1. Spread Dynamics Analysis...")
        self.analyze_spread_dynamics()

        print("\n2. MM-002: Sustained NO Imbalance Analysis...")
        self.analyze_mm002_sustained_no_imbalance()

        print("\n3. MM-003: Large Order + Direction Match Analysis...")
        self.analyze_mm003_large_order_direction()

        print("\n4. MM-004: Thin Depth Adverse Selection Analysis...")
        self.analyze_mm004_thin_depth_adverse_selection()

        print("\n5. Market Microstructure Profiling...")
        self.analyze_microstructure()

        print("\n6. Bucket-Matched Validation...")
        validation_results = self.run_bucket_matched_validation()
        self.results['bucket_matched_validation'] = validation_results

        # Collect validated strategies
        validated = []
        rejected = []

        for hyp_key in ['MM-002', 'MM-003', 'MM-004']:
            if hyp_key in self.results['mm_hypotheses']:
                for test in self.results['mm_hypotheses'][hyp_key].get('tests', []):
                    if test.get('is_valid'):
                        validated.append(test)
                    elif test.get('n_markets', 0) >= 30:
                        rejected.append(test)

        self.results['validated_strategies'] = validated
        self.results['rejected_strategies'] = rejected

        # Summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        print(f"\nTotal hypotheses tested: {len(validated) + len(rejected)}")
        print(f"Validated strategies: {len(validated)}")
        print(f"Rejected strategies: {len(rejected)}")

        if validated:
            print("\n--- VALIDATED STRATEGIES ---")
            for v in sorted(validated, key=lambda x: x['edge'], reverse=True):
                print(f"\n  {v['strategy']}")
                print(f"    Markets: {v['n_markets']} | Win Rate: {v['win_rate']:.1%} | Edge: {v['edge']:+.1%}")
                print(f"    Profit: ${v['total_profit']:,.0f} | ROI: {v['roi_pct']:.1%}")
                print(f"    P-value: {v['p_value']:.4f} | Max Market Share: {v['max_market_share']:.1%}")

        # Save results
        output_path = OUTPUT_DIR / "market_maker_strategy_research.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

        return self.results


def main():
    """Run the market maker strategy analysis."""
    analyzer = MarketMakerAnalyzer()
    results = analyzer.generate_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Print executive summary
    print("\n### EXECUTIVE SUMMARY ###\n")
    print("Market Making Context:")
    print("  - Pure market making is NOT viable on Kalshi (latency disadvantage, no rebates)")
    print("  - Analysis focused on MM-inspired signal detection:")
    print("    1. Spread dynamics and liquidity patterns")
    print("    2. Order flow imbalance signals")
    print("    3. Large order information content")
    print("    4. Adverse selection in thin markets")

    print("\nKey Findings:")
    print(f"  - Validated strategies: {len(results.get('validated_strategies', []))}")
    print(f"  - Rejected strategies: {len(results.get('rejected_strategies', []))}")

    validated = results.get('validated_strategies', [])
    if validated:
        print("\n  Top validated strategies by edge:")
        for v in sorted(validated, key=lambda x: x['edge'], reverse=True)[:5]:
            print(f"    - {v['strategy']}: Edge={v['edge']:+.1%}, Markets={v['n_markets']}")
    else:
        print("\n  No strategies passed full validation criteria.")
        print("  This confirms pure market making edge is difficult to find.")


if __name__ == "__main__":
    main()
