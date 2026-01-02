#!/usr/bin/env python3
"""
Session 002 Deep Pattern Analysis - December 29, 2025

This script performs exhaustive hypothesis testing on 1.6M+ trades to find
profitable trading patterns beyond the validated YES/NO at 80-90c strategies.

Hypotheses tested:
- H007: Fade whale consensus (contrarian)
- H005: Time-of-day patterns
- H006: Category-specific efficiency
- H008: New market mispricing
- H009: Price velocity/momentum
- H010: Trade sequencing patterns
- H011: Volume-weighted signals
- H012: Round number effects

Author: Quant Agent (Opus 4.5)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets" / "market_outcomes_ALL.csv"


class DeepPatternAnalyzer:
    """
    Rigorous pattern analysis with proper market-level validation.
    """

    def __init__(self):
        print("=" * 80)
        print("SESSION 002 DEEP PATTERN ANALYSIS")
        print("=" * 80)
        print(f"\nLoading trades from {TRADES_FILE}...")

        self.df = pd.read_csv(TRADES_FILE)
        print(f"Loaded {len(self.df):,} trades across {self.df['market_ticker'].nunique():,} unique markets")

        # Parse datetime
        self.df['dt'] = pd.to_datetime(self.df['datetime'])
        self.df['hour'] = self.df['dt'].dt.hour
        self.df['minute'] = self.df['dt'].dt.minute
        self.df['day_of_week'] = self.df['dt'].dt.dayofweek
        self.df['date'] = self.df['dt'].dt.date

        # Parse category from ticker
        self.df['category'] = self.df['market_ticker'].str.extract(r'^(KX[A-Z]+)')[0]

        # Load market outcomes for additional context
        print(f"\nLoading market outcomes from {MARKETS_FILE}...")
        self.markets = pd.read_csv(MARKETS_FILE)
        print(f"Loaded {len(self.markets):,} market records")

        # Merge category info
        self.markets['category'] = self.markets['ticker'].str.extract(r'^(KX[A-Z]+)')[0]

        # Summary statistics
        self._print_data_summary()

    def _print_data_summary(self):
        """Print data summary."""
        print("\n" + "-" * 60)
        print("DATA SUMMARY")
        print("-" * 60)

        date_range = f"{self.df['dt'].min().strftime('%Y-%m-%d')} to {self.df['dt'].max().strftime('%Y-%m-%d')}"
        print(f"Date Range: {date_range}")
        print(f"Total Trades: {len(self.df):,}")
        print(f"Unique Markets: {self.df['market_ticker'].nunique():,}")
        print(f"YES Trades: {(self.df['taker_side'] == 'yes').sum():,}")
        print(f"NO Trades: {(self.df['taker_side'] == 'no').sum():,}")

        # Top categories
        print("\nTop 10 Categories by Trade Count:")
        cat_counts = self.df['category'].value_counts().head(10)
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count:,}")

    def validate_strategy(self, mask, strategy_name: str, min_markets: int = 50, max_concentration: float = 0.30):
        """
        Validate a strategy with proper market-level statistics.

        Returns dict with all validation metrics or None if insufficient data.
        """
        filtered = self.df[mask].copy()

        if len(filtered) == 0:
            return None

        # Group by market - each market is ONE observation
        market_stats = filtered.groupby('market_ticker').agg({
            'is_winner': 'first',  # All trades same outcome
            'actual_profit_dollars': 'sum',
            'cost_dollars': 'sum',
            'count': 'sum',
            'trade_price': 'mean',
            'taker_side': 'first',
            'market_result': 'first'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets < 10:
            return None

        # Win rate at market level
        n_wins = market_stats['is_winner'].sum()
        win_rate = n_wins / n_markets

        # P/L
        total_profit = market_stats['actual_profit_dollars'].sum()
        total_cost = market_stats['cost_dollars'].sum()
        roi = total_profit / total_cost if total_cost > 0 else 0

        # Concentration risk
        market_profits = market_stats.sort_values('actual_profit_dollars', ascending=False)
        top_market = market_profits.iloc[0]
        top_market_pct = abs(top_market['actual_profit_dollars']) / max(abs(total_profit), 1)

        # Breakeven and edge
        avg_price = filtered['trade_price'].mean()
        side = filtered['taker_side'].mode().iloc[0]
        if side == 'yes':
            breakeven = avg_price / 100
        else:
            breakeven = (100 - avg_price) / 100

        edge = win_rate - breakeven

        # Statistical significance (binomial test)
        p_value = 1.0
        if n_markets >= 20:
            try:
                direction = 'greater' if edge > 0 else 'less'
                result = stats.binomtest(n_wins, n_markets, breakeven, alternative=direction)
                p_value = result.pvalue
            except:
                pass

        # Validation checks
        is_valid = (
            n_markets >= min_markets and
            top_market_pct < max_concentration and
            p_value < 0.05
        )

        return {
            'strategy': strategy_name,
            'n_trades': len(filtered),
            'n_markets': n_markets,
            'win_rate': win_rate,
            'breakeven': breakeven,
            'edge': edge,
            'total_profit': total_profit,
            'total_cost': total_cost,
            'roi': roi,
            'top_market': top_market['market_ticker'],
            'top_market_profit': top_market['actual_profit_dollars'],
            'top_market_pct': top_market_pct,
            'p_value': p_value,
            'is_valid': is_valid
        }

    def test_h007_fade_whale_consensus(self):
        """
        H007: Fade whale consensus (contrarian strategy).

        When multiple whales bet the same direction, bet AGAINST them.
        Previous finding: Following 100% consensus = 27% win rate.
        Therefore: Fading should = 73% win rate?
        """
        print("\n" + "=" * 80)
        print("H007: FADE WHALE CONSENSUS (CONTRARIAN)")
        print("=" * 80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()
        df_sorted['is_whale'] = df_sorted['count'] >= 100

        # Calculate whale consensus per market
        market_whale_consensus = []

        for ticker, group in df_sorted.groupby('market_ticker'):
            whales = group[group['is_whale']]
            if len(whales) < 2:
                continue

            yes_count = (whales['taker_side'] == 'yes').sum()
            no_count = (whales['taker_side'] == 'no').sum()
            total = yes_count + no_count

            majority_side = 'yes' if yes_count > no_count else 'no'
            consensus_pct = max(yes_count, no_count) / total

            # Get market result
            market_result = group['market_result'].iloc[0]
            is_winner = group['is_winner'].iloc[0]

            # For fading: the OPPOSITE side
            fade_side = 'no' if majority_side == 'yes' else 'yes'
            fade_wins = (fade_side == market_result)

            # Get price info for the fade side
            fade_trades = group[group['taker_side'] == fade_side]
            if len(fade_trades) > 0:
                avg_fade_price = fade_trades['trade_price'].mean()
            else:
                avg_fade_price = group['trade_price'].mean()

            market_whale_consensus.append({
                'market_ticker': ticker,
                'whale_count': total,
                'yes_whales': yes_count,
                'no_whales': no_count,
                'majority_side': majority_side,
                'consensus_pct': consensus_pct,
                'market_result': market_result,
                'fade_wins': fade_wins,
                'avg_fade_price': avg_fade_price,
                'total_cost': group['cost_dollars'].sum(),
                'total_profit': group['actual_profit_dollars'].sum()
            })

        consensus_df = pd.DataFrame(market_whale_consensus)

        if len(consensus_df) == 0:
            print("No markets with whale consensus data")
            return None

        print(f"\nMarkets with 2+ whale trades: {len(consensus_df):,}")

        results = []

        # Test fading at different consensus levels
        for min_consensus in [0.6, 0.7, 0.8, 0.9, 1.0]:
            subset = consensus_df[consensus_df['consensus_pct'] >= min_consensus]

            if len(subset) < 20:
                continue

            n_markets = len(subset)
            fade_wins = subset['fade_wins'].sum()
            win_rate = fade_wins / n_markets

            # Average fade price to calculate breakeven
            avg_price = subset['avg_fade_price'].mean()
            # Assuming we're fading by betting the opposite side
            # If we fade YES consensus by betting NO at avg_price
            breakeven = 0.5  # Simplified - would need actual trade data

            edge = win_rate - 0.5  # Against coin flip baseline

            # Binomial test against 50%
            try:
                result = stats.binomtest(fade_wins, n_markets, 0.5, alternative='greater')
                p_value = result.pvalue
            except:
                p_value = 1.0

            is_valid = n_markets >= 50 and p_value < 0.05

            results.append({
                'strategy': f'Fade whale consensus >= {min_consensus*100:.0f}%',
                'n_markets': n_markets,
                'win_rate': win_rate,
                'edge': edge,
                'p_value': p_value,
                'is_valid': is_valid
            })

            status = "VALID" if is_valid else "NOT VALID"
            print(f"\nFade {min_consensus*100:.0f}%+ consensus:")
            print(f"  Markets: {n_markets} | Win Rate: {win_rate:.1%} | Edge vs 50%: {edge:+.1%} | p-value: {p_value:.4f} | {status}")

        return pd.DataFrame(results)

    def test_h005_time_patterns(self):
        """
        H005: Time-of-day patterns.

        Test if certain hours or time periods have better edge.
        """
        print("\n" + "=" * 80)
        print("H005: TIME-OF-DAY PATTERNS")
        print("=" * 80)

        results = []

        # Hour of day analysis
        print("\n--- HOUR OF DAY ANALYSIS ---")

        for hour in range(24):
            for side in ['yes', 'no']:
                mask = (self.df['hour'] == hour) & (self.df['taker_side'] == side)
                strategy_name = f"{side.upper()} at hour {hour:02d}"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)

        # Time blocks
        print("\n--- TIME BLOCK ANALYSIS ---")

        time_blocks = [
            ((0, 6), "Night (0-6am)"),
            ((6, 9), "Early Morning (6-9am)"),
            ((9, 12), "Late Morning (9am-12pm)"),
            ((12, 15), "Early Afternoon (12-3pm)"),
            ((15, 18), "Late Afternoon (3-6pm)"),
            ((18, 21), "Evening (6-9pm)"),
            ((21, 24), "Late Night (9pm-12am)")
        ]

        for (start, end), block_name in time_blocks:
            for side in ['yes', 'no']:
                mask = (
                    (self.df['hour'] >= start) &
                    (self.df['hour'] < end) &
                    (self.df['taker_side'] == side)
                )
                strategy_name = f"{side.upper()} during {block_name}"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)

        # Day of week
        print("\n--- DAY OF WEEK ANALYSIS ---")

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for dow, day_name in enumerate(days):
            for side in ['yes', 'no']:
                mask = (self.df['day_of_week'] == dow) & (self.df['taker_side'] == side)
                strategy_name = f"{side.upper()} on {day_name}"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Show validated time patterns
            validated = results_df[results_df['is_valid']].sort_values('edge', ascending=False)

            print(f"\n{'='*60}")
            print("VALIDATED TIME PATTERNS")
            print(f"{'='*60}")

            if len(validated) > 0:
                print(f"\n{'Strategy':<45} {'Markets':>8} {'WinRate':>8} {'Edge':>8} {'Profit':>12}")
                print("-" * 90)
                for _, row in validated.iterrows():
                    print(f"{row['strategy']:<45} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f}")
            else:
                print("\nNo time patterns passed validation criteria.")

        return results_df

    def test_h006_category_efficiency(self):
        """
        H006: Category-specific efficiency variance.

        Some market categories may be less efficient than others.
        """
        print("\n" + "=" * 80)
        print("H006: CATEGORY-SPECIFIC EFFICIENCY")
        print("=" * 80)

        results = []

        # Get categories with enough data
        category_counts = self.df.groupby('category')['market_ticker'].nunique()
        significant_categories = category_counts[category_counts >= 50].index.tolist()

        print(f"\nAnalyzing {len(significant_categories)} categories with 50+ markets...")

        for category in significant_categories:
            cat_mask = self.df['category'] == category

            # Test different price buckets within category
            for price_low, price_high in [(70, 80), (80, 90), (90, 100)]:
                for side in ['yes', 'no']:
                    mask = (
                        cat_mask &
                        (self.df['trade_price'] >= price_low) &
                        (self.df['trade_price'] < price_high) &
                        (self.df['taker_side'] == side)
                    )

                    strategy_name = f"{category} {side.upper()} at {price_low}-{price_high}c"
                    result = self.validate_strategy(mask, strategy_name, min_markets=30)

                    if result and result['n_markets'] >= 30:
                        results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Sort by edge and show top performers
            results_df = results_df.sort_values('edge', ascending=False)

            print(f"\n{'='*60}")
            print("TOP CATEGORY PATTERNS (by Edge)")
            print(f"{'='*60}")

            print(f"\n{'Strategy':<55} {'Markets':>8} {'Edge':>8} {'Profit':>12} {'Valid':>6}")
            print("-" * 100)

            for _, row in results_df.head(20).iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<55} {row['n_markets']:>8} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {valid:>6}")

            # Show validated only
            validated = results_df[results_df['is_valid']]
            if len(validated) > 0:
                print(f"\n{'='*60}")
                print(f"VALIDATED CATEGORY STRATEGIES: {len(validated)}")
                print(f"{'='*60}")

                for _, row in validated.iterrows():
                    print(f"\n{row['strategy']}")
                    print(f"  Markets: {row['n_markets']} | Win Rate: {row['win_rate']:.1%} | Edge: {row['edge']:+.1%}")
                    print(f"  Profit: ${row['total_profit']:,.0f} | p-value: {row['p_value']:.4f}")

        return results_df

    def test_h008_new_market_mispricing(self):
        """
        H008: New market mispricing.

        Are the FIRST trades in a market mispriced? Do early entrants have edge?
        """
        print("\n" + "=" * 80)
        print("H008: NEW MARKET MISPRICING (EARLY TRADES)")
        print("=" * 80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()

        # Add trade sequence number within each market
        df_sorted['trade_seq'] = df_sorted.groupby('market_ticker').cumcount() + 1

        # Calculate time since market open for each trade
        df_sorted['market_open_time'] = df_sorted.groupby('market_ticker')['timestamp'].transform('min')
        df_sorted['time_since_open_ms'] = df_sorted['timestamp'] - df_sorted['market_open_time']
        df_sorted['time_since_open_min'] = df_sorted['time_since_open_ms'] / 60000

        results = []

        # Test first N trades
        print("\n--- FIRST N TRADES ANALYSIS ---")

        for first_n in [1, 3, 5, 10, 20]:
            for side in ['yes', 'no']:
                mask = (df_sorted['trade_seq'] <= first_n) & (df_sorted['taker_side'] == side)
                strategy_name = f"First {first_n} trades - {side.upper()}"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)
                    print(f"{strategy_name}: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        # Test first N minutes
        print("\n--- FIRST N MINUTES ANALYSIS ---")

        for first_min in [1, 5, 10, 30, 60]:
            for side in ['yes', 'no']:
                mask = (df_sorted['time_since_open_min'] <= first_min) & (df_sorted['taker_side'] == side)
                strategy_name = f"First {first_min} min - {side.upper()}"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)
                    print(f"{strategy_name}: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            validated = results_df[results_df['is_valid']]

            if len(validated) > 0:
                print(f"\n{'='*60}")
                print("VALIDATED EARLY TRADE PATTERNS")
                print(f"{'='*60}")

                for _, row in validated.iterrows():
                    print(f"\n{row['strategy']}")
                    print(f"  Markets: {row['n_markets']} | Edge: {row['edge']:+.1%} | Profit: ${row['total_profit']:,.0f}")
            else:
                print("\nNo early trade patterns passed validation.")

        return results_df

    def test_h009_price_velocity(self):
        """
        H009: Price velocity/momentum patterns.

        Do rapid price moves create mean reversion opportunities?
        """
        print("\n" + "=" * 80)
        print("H009: PRICE VELOCITY / MOMENTUM")
        print("=" * 80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()

        # Calculate price change from previous trade
        df_sorted['prev_price'] = df_sorted.groupby('market_ticker')['trade_price'].shift(1)
        df_sorted['price_change'] = df_sorted['trade_price'] - df_sorted['prev_price']
        df_sorted['time_diff_ms'] = df_sorted.groupby('market_ticker')['timestamp'].diff()

        # Price velocity = price change per second
        df_sorted['price_velocity'] = df_sorted['price_change'] / (df_sorted['time_diff_ms'] / 1000)

        results = []

        # Test momentum (follow the move)
        print("\n--- MOMENTUM (FOLLOW THE MOVE) ---")

        for threshold in [5, 10, 15, 20]:
            # After price up, bet YES (momentum)
            mask_up = (df_sorted['price_change'] >= threshold) & (df_sorted['taker_side'] == 'yes')
            result = self.validate_strategy(mask_up, f"After +{threshold}c move, bet YES")
            if result:
                results.append(result)
                print(f"After +{threshold}c: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

            # After price down, bet NO (momentum)
            mask_down = (df_sorted['price_change'] <= -threshold) & (df_sorted['taker_side'] == 'no')
            result = self.validate_strategy(mask_down, f"After -{threshold}c move, bet NO")
            if result:
                results.append(result)
                print(f"After -{threshold}c: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        # Test mean reversion (fade the move)
        print("\n--- MEAN REVERSION (FADE THE MOVE) ---")

        for threshold in [5, 10, 15, 20]:
            # After price up, bet NO (mean reversion)
            mask_up_fade = (df_sorted['price_change'] >= threshold) & (df_sorted['taker_side'] == 'no')
            result = self.validate_strategy(mask_up_fade, f"After +{threshold}c move, bet NO (fade)")
            if result:
                results.append(result)
                print(f"Fade +{threshold}c: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

            # After price down, bet YES (mean reversion)
            mask_down_fade = (df_sorted['price_change'] <= -threshold) & (df_sorted['taker_side'] == 'yes')
            result = self.validate_strategy(mask_down_fade, f"After -{threshold}c move, bet YES (fade)")
            if result:
                results.append(result)
                print(f"Fade -{threshold}c: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            validated = results_df[results_df['is_valid']]

            if len(validated) > 0:
                print(f"\n{'='*60}")
                print("VALIDATED PRICE VELOCITY PATTERNS")
                print(f"{'='*60}")

                for _, row in validated.iterrows():
                    print(f"\n{row['strategy']}")
                    print(f"  Markets: {row['n_markets']} | Edge: {row['edge']:+.1%} | Profit: ${row['total_profit']:,.0f}")
            else:
                print("\nNo price velocity patterns passed validation.")

        return results_df

    def test_h011_volume_patterns(self):
        """
        H011: Volume-weighted signals.

        Do high-volume markets behave differently?
        """
        print("\n" + "=" * 80)
        print("H011: VOLUME PATTERNS")
        print("=" * 80)

        # Calculate total volume per market
        market_volume = self.df.groupby('market_ticker').agg({
            'count': 'sum',
            'cost_dollars': 'sum'
        }).reset_index()
        market_volume.columns = ['market_ticker', 'total_contracts', 'total_cost']

        # Merge back
        df_vol = self.df.merge(market_volume, on='market_ticker', suffixes=('', '_market'))

        # Volume percentiles
        vol_percentiles = market_volume['total_contracts'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()

        print(f"\nVolume percentiles:")
        print(f"  25th: {vol_percentiles[0.25]:,.0f} contracts")
        print(f"  50th: {vol_percentiles[0.5]:,.0f} contracts")
        print(f"  75th: {vol_percentiles[0.75]:,.0f} contracts")
        print(f"  90th: {vol_percentiles[0.9]:,.0f} contracts")

        results = []

        # Test different volume segments
        volume_segments = [
            (0, vol_percentiles[0.25], "Low volume (bottom 25%)"),
            (vol_percentiles[0.25], vol_percentiles[0.5], "Below median volume"),
            (vol_percentiles[0.5], vol_percentiles[0.75], "Above median volume"),
            (vol_percentiles[0.75], float('inf'), "High volume (top 25%)")
        ]

        print("\n--- VOLUME SEGMENT ANALYSIS ---")

        for vol_min, vol_max, segment_name in volume_segments:
            vol_mask = (df_vol['total_contracts'] >= vol_min) & (df_vol['total_contracts'] < vol_max)

            for side in ['yes', 'no']:
                for price_low, price_high in [(70, 80), (80, 90), (90, 100)]:
                    mask = (
                        vol_mask &
                        (df_vol['taker_side'] == side) &
                        (df_vol['trade_price'] >= price_low) &
                        (df_vol['trade_price'] < price_high)
                    )

                    strategy_name = f"{segment_name} {side.upper()} at {price_low}-{price_high}c"
                    result = self.validate_strategy(mask, strategy_name)

                    if result and result['n_markets'] >= 30:
                        results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values('edge', ascending=False)

            print(f"\n{'Strategy':<60} {'Markets':>8} {'Edge':>8} {'Valid':>6}")
            print("-" * 90)

            for _, row in results_df.head(15).iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<60} {row['n_markets']:>8} {row['edge']:>+7.1%} {valid:>6}")

            validated = results_df[results_df['is_valid']]
            if len(validated) > 0:
                print(f"\n{'='*60}")
                print(f"VALIDATED VOLUME PATTERNS: {len(validated)}")
                print(f"{'='*60}")

        return results_df

    def test_h012_round_numbers(self):
        """
        H012: Round number effects.

        Do prices cluster at psychological levels? Is there edge at round numbers?
        """
        print("\n" + "=" * 80)
        print("H012: ROUND NUMBER EFFECTS")
        print("=" * 80)

        # Check price distribution
        price_counts = self.df['trade_price'].value_counts().sort_index()

        # Round numbers
        round_5 = [p for p in range(5, 100, 5)]
        round_10 = [p for p in range(10, 100, 10)]
        round_25 = [25, 50, 75]

        print("\nTrade counts at round prices:")
        for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
            count = price_counts.get(p, 0)
            print(f"  {p}c: {count:,}")

        results = []

        # Test trades AT round numbers
        print("\n--- AT ROUND NUMBERS ---")

        for price in [25, 50, 75]:
            for side in ['yes', 'no']:
                mask = (self.df['trade_price'] == price) & (self.df['taker_side'] == side)
                strategy_name = f"{side.upper()} at exactly {price}c"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)
                    print(f"{strategy_name}: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        # Test trades NEAR round numbers (within 2c)
        print("\n--- NEAR ROUND NUMBERS (+/- 2c) ---")

        for price in [25, 50, 75]:
            for side in ['yes', 'no']:
                mask = (
                    (self.df['trade_price'] >= price - 2) &
                    (self.df['trade_price'] <= price + 2) &
                    (self.df['taker_side'] == side)
                )
                strategy_name = f"{side.upper()} near {price}c (+/-2)"

                result = self.validate_strategy(mask, strategy_name)
                if result:
                    results.append(result)
                    print(f"{strategy_name}: Markets={result['n_markets']} Edge={result['edge']:+.1%} Valid={result['is_valid']}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            validated = results_df[results_df['is_valid']]

            if len(validated) > 0:
                print(f"\n{'='*60}")
                print("VALIDATED ROUND NUMBER PATTERNS")
                print(f"{'='*60}")

                for _, row in validated.iterrows():
                    print(f"\n{row['strategy']}")
                    print(f"  Markets: {row['n_markets']} | Edge: {row['edge']:+.1%} | Profit: ${row['total_profit']:,.0f}")
            else:
                print("\nNo round number patterns passed validation.")

        return results_df

    def test_h010_trade_sequencing(self):
        """
        H010: Trade sequencing patterns.

        Does the order/pattern of trades predict outcome?
        """
        print("\n" + "=" * 80)
        print("H010: TRADE SEQUENCING PATTERNS")
        print("=" * 80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()

        # Create sequence features
        df_sorted['prev_side'] = df_sorted.groupby('market_ticker')['taker_side'].shift(1)
        df_sorted['side_changed'] = df_sorted['taker_side'] != df_sorted['prev_side']

        # Count consecutive same-side trades
        df_sorted['same_side_streak'] = df_sorted.groupby('market_ticker')['side_changed'].transform(
            lambda x: (~x.fillna(True)).cumsum()
        )

        results = []

        # Test after N consecutive same-side trades
        print("\n--- AFTER CONSECUTIVE SAME-SIDE TRADES ---")

        for n_consec in [2, 3, 4, 5]:
            for side in ['yes', 'no']:
                # After N consecutive YES trades, what happens to next YES bet?
                mask_same = (
                    (df_sorted['same_side_streak'] >= n_consec) &
                    (df_sorted['prev_side'] == side) &
                    (df_sorted['taker_side'] == side)
                )
                result = self.validate_strategy(mask_same, f"After {n_consec}+ {side.upper()} trades, bet {side.upper()}")
                if result:
                    results.append(result)

                # After N consecutive opposite trades, bet the opposite
                mask_fade = (
                    (df_sorted['same_side_streak'] >= n_consec) &
                    (df_sorted['prev_side'] != side) &
                    (df_sorted['taker_side'] == side)
                )
                opposite = 'no' if side == 'yes' else 'yes'
                result = self.validate_strategy(mask_fade, f"After {n_consec}+ {opposite.upper()} trades, bet {side.upper()}")
                if result:
                    results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values('edge', ascending=False)

            print(f"\n{'Strategy':<60} {'Markets':>8} {'Edge':>8} {'Valid':>6}")
            print("-" * 90)

            for _, row in results_df.head(15).iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<60} {row['n_markets']:>8} {row['edge']:>+7.1%} {valid:>6}")

            validated = results_df[results_df['is_valid']]
            if len(validated) > 0:
                print(f"\n{'='*60}")
                print(f"VALIDATED SEQUENCING PATTERNS: {len(validated)}")
                print(f"{'='*60}")

        return results_df

    def run_all_tests(self):
        """Run all hypothesis tests and summarize findings."""
        print("\n" + "#" * 80)
        print("# RUNNING ALL HYPOTHESIS TESTS")
        print("#" * 80)

        all_results = {}

        # Run each test
        all_results['H007_whale_fade'] = self.test_h007_fade_whale_consensus()
        all_results['H005_time'] = self.test_h005_time_patterns()
        all_results['H006_category'] = self.test_h006_category_efficiency()
        all_results['H008_early'] = self.test_h008_new_market_mispricing()
        all_results['H009_velocity'] = self.test_h009_price_velocity()
        all_results['H010_sequence'] = self.test_h010_trade_sequencing()
        all_results['H011_volume'] = self.test_h011_volume_patterns()
        all_results['H012_round'] = self.test_h012_round_numbers()

        # Summarize all validated strategies
        print("\n" + "#" * 80)
        print("# FINAL SUMMARY: ALL VALIDATED STRATEGIES")
        print("#" * 80)

        all_validated = []

        for test_name, results_df in all_results.items():
            if results_df is not None and len(results_df) > 0 and 'is_valid' in results_df.columns:
                validated = results_df[results_df['is_valid']]
                for _, row in validated.iterrows():
                    row_dict = row.to_dict()
                    row_dict['test'] = test_name
                    all_validated.append(row_dict)

        if all_validated:
            final_df = pd.DataFrame(all_validated)
            final_df = final_df.sort_values('edge', ascending=False)

            print(f"\n{'Strategy':<55} {'Test':<15} {'Markets':>8} {'Edge':>8} {'Profit':>12}")
            print("=" * 110)

            for _, row in final_df.iterrows():
                profit_str = f"${row.get('total_profit', 0):,.0f}" if 'total_profit' in row else "N/A"
                print(f"{row['strategy']:<55} {row['test']:<15} {row['n_markets']:>8} {row['edge']:>+7.1%} {profit_str:>12}")

            print(f"\nTotal validated strategies: {len(final_df)}")
        else:
            print("\nNo strategies passed all validation criteria.")

        return all_results, all_validated


def main():
    analyzer = DeepPatternAnalyzer()
    all_results, validated = analyzer.run_all_tests()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return all_results, validated


if __name__ == "__main__":
    main()
