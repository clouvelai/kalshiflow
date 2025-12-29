#!/usr/bin/env python3
"""
Public Trade Feed Strategy Analysis - Proper Market-Level Validation

This script analyzes the public trade feed to find profitable patterns,
with CORRECT statistical methodology:
1. Uses UNIQUE MARKET counts (not trade counts)
2. Checks concentration risk (no single market > 30% of profit)
3. Requires statistical significance (p < 0.05)
4. Focuses on PUBLIC TRADE FEED signals for MVP integration

Usage:
    python public_trade_feed_analysis.py --analyze
    python public_trade_feed_analysis.py --whale-following
    python public_trade_feed_analysis.py --timing
    python public_trade_feed_analysis.py --all
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import json
import argparse
from datetime import datetime

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent.parent / "training" / "reports"
ENRICHED_ALL = DATA_DIR / "enriched_trades_resolved_ALL.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "traderv3" / "planning"


class PublicTradeFeedAnalyzer:
    """
    Analyzes public trade feed for profitable patterns with proper validation.
    """

    def __init__(self, data_path: Path = ENRICHED_ALL):
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
        self.df['category'] = self.df['market_ticker'].str.extract(r'^(KX[A-Z]+)')

        # Create price buckets
        self.df['price_bucket'] = pd.cut(
            self.df['trade_price'],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        )

    def calculate_market_level_stats(self, mask, strategy_name: str = ""):
        """
        Calculate statistics at the MARKET level (not trade level).
        This is the CORRECT way to analyze binary outcome markets.
        """
        filtered = self.df[mask].copy()

        if len(filtered) == 0:
            return None

        # Group by market and aggregate
        market_stats = filtered.groupby('market_ticker').agg({
            'is_winner': 'first',  # All trades in market have same outcome
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

        # Market-level win rate
        n_wins = market_stats['is_winner'].sum()
        win_rate = n_wins / n_markets

        # Total P/L
        total_profit = market_stats['actual_profit_dollars'].sum()
        total_cost = market_stats['cost_dollars'].sum()
        roi = total_profit / total_cost if total_cost > 0 else 0

        # Concentration check
        market_profits = market_stats.sort_values('actual_profit_dollars', ascending=False)
        top_market = market_profits.iloc[0]
        top_market_pct = abs(top_market['actual_profit_dollars']) / max(abs(total_profit), 1) * 100

        top_10_profit = market_profits.head(10)['actual_profit_dollars'].sum()
        top_10_pct = abs(top_10_profit) / max(abs(total_profit), 1) * 100

        # Calculate breakeven and edge
        avg_price = filtered['trade_price'].mean()
        if filtered['taker_side'].mode().iloc[0] == 'yes':
            breakeven = avg_price / 100
        else:
            breakeven = (100 - avg_price) / 100

        edge = win_rate - breakeven

        # Statistical significance (binomial test)
        if n_markets >= 20:
            try:
                if edge > 0:
                    result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
                else:
                    result = stats.binomtest(n_wins, n_markets, breakeven, alternative='less')
                p_value = result.pvalue
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0

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
            'top_10_pct': top_10_pct,
            'p_value': p_value,
            'is_valid': n_markets >= 50 and top_market_pct < 30 and p_value < 0.05
        }

    def analyze_whale_following_basic(self):
        """
        Analyze basic whale-following: what happens when we follow large trades?

        Key question: Do whale trades (100+ contracts) predict market outcomes?
        """
        print("\n" + "="*80)
        print("WHALE FOLLOWING ANALYSIS (Proper Market-Level)")
        print("="*80)

        results = []

        # Different whale thresholds
        for min_contracts in [50, 100, 200, 500, 1000]:
            whale_mask = self.df['count'] >= min_contracts

            for side in ['yes', 'no']:
                for price_low, price_high in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]:
                    mask = (
                        whale_mask &
                        (self.df['taker_side'] == side) &
                        (self.df['trade_price'] >= price_low) &
                        (self.df['trade_price'] < price_high)
                    )

                    strategy_name = f"Whale {side.upper()} >= {min_contracts} @ {price_low}-{price_high}c"
                    stats = self.calculate_market_level_stats(mask, strategy_name)

                    if stats:
                        results.append(stats)

        # Sort by edge
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('edge', ascending=False)

        print("\n### TOP 20 WHALE STRATEGIES (by Edge) ###\n")
        print(f"{'Strategy':<50} {'Markets':>8} {'WinRate':>8} {'Breakev':>8} {'Edge':>8} {'Profit':>12} {'TopMkt%':>8} {'Valid':>6}")
        print("-" * 120)

        for _, row in results_df.head(20).iterrows():
            valid = "YES" if row['is_valid'] else "NO"
            print(f"{row['strategy']:<50} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['breakeven']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {row['top_market_pct']:>7.1f}% {valid:>6}")

        print("\n### VALIDATED STRATEGIES ONLY ###\n")
        validated = results_df[results_df['is_valid']]
        if len(validated) > 0:
            for _, row in validated.iterrows():
                print(f"\n{row['strategy']}")
                print(f"  Markets: {row['n_markets']} | Win Rate: {row['win_rate']:.1%} | Edge: {row['edge']:+.1%}")
                print(f"  Profit: ${row['total_profit']:,.0f} | ROI: {row['roi']:.1%}")
                print(f"  Top Market: {row['top_market'][:50]} ({row['top_market_pct']:.1f}% of profit)")
                print(f"  P-value: {row['p_value']:.4f}")
        else:
            print("  No whale-following strategies passed validation criteria.")

        return results_df

    def analyze_whale_momentum(self):
        """
        Analyze whale momentum: after a whale trade, do subsequent trades in the
        same direction predict the outcome better?

        This is a PUBLIC TRADE FEED signal - we see the whale trade come through
        and can react to it.
        """
        print("\n" + "="*80)
        print("WHALE MOMENTUM ANALYSIS")
        print("="*80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()

        # Identify whale trades
        df_sorted['is_whale'] = df_sorted['count'] >= 100

        # For each trade, check if previous trade in same market was a whale
        df_sorted['prev_whale'] = df_sorted.groupby('market_ticker')['is_whale'].shift(1)
        df_sorted['prev_side'] = df_sorted.groupby('market_ticker')['taker_side'].shift(1)
        df_sorted['prev_count'] = df_sorted.groupby('market_ticker')['count'].shift(1)
        df_sorted['time_since_prev'] = df_sorted.groupby('market_ticker')['timestamp'].diff()

        # Trades that follow a whale trade
        follows_whale = df_sorted['prev_whale'] == True

        # Same direction vs opposite
        same_direction = df_sorted['taker_side'] == df_sorted['prev_side']
        opposite_direction = df_sorted['taker_side'] != df_sorted['prev_side']

        results = []

        # Follow whale same direction
        for max_time_ms in [5000, 30000, 60000, 300000]:  # 5s, 30s, 1min, 5min
            for side in ['yes', 'no']:
                mask = (
                    follows_whale &
                    same_direction &
                    (df_sorted['time_since_prev'] <= max_time_ms) &
                    (df_sorted['taker_side'] == side)
                )

                strategy_name = f"Follow whale {side.upper()} within {max_time_ms//1000}s"
                stats = self.calculate_market_level_stats(mask, strategy_name)
                if stats:
                    results.append(stats)

        # Fade whale (opposite direction)
        for max_time_ms in [30000, 60000]:
            for side in ['yes', 'no']:
                mask = (
                    follows_whale &
                    opposite_direction &
                    (df_sorted['time_since_prev'] <= max_time_ms) &
                    (df_sorted['taker_side'] == side)
                )

                strategy_name = f"Fade whale {side.upper()} within {max_time_ms//1000}s"
                stats = self.calculate_market_level_stats(mask, strategy_name)
                if stats:
                    results.append(stats)

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('edge', ascending=False)

            print("\n### WHALE MOMENTUM PATTERNS ###\n")
            print(f"{'Strategy':<45} {'Markets':>8} {'WinRate':>8} {'Edge':>8} {'Profit':>12} {'TopMkt%':>8} {'Valid':>6}")
            print("-" * 110)

            for _, row in results_df.iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<45} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {row['top_market_pct']:>7.1f}% {valid:>6}")

        return results_df

    def analyze_whale_size_signal(self):
        """
        Analyze if LARGER whale trades have more predictive power.

        Hypothesis: Very large trades (1000+ contracts) represent more informed money.
        """
        print("\n" + "="*80)
        print("WHALE SIZE SIGNAL ANALYSIS")
        print("="*80)

        results = []

        # Size buckets
        size_buckets = [
            (50, 100, "Small whale (50-100)"),
            (100, 250, "Medium whale (100-250)"),
            (250, 500, "Large whale (250-500)"),
            (500, 1000, "Very large whale (500-1000)"),
            (1000, 5000, "Mega whale (1000-5000)"),
            (5000, float('inf'), "Ultra whale (5000+)")
        ]

        for min_size, max_size, bucket_name in size_buckets:
            size_mask = (self.df['count'] >= min_size) & (self.df['count'] < max_size)

            for side in ['yes', 'no']:
                mask = size_mask & (self.df['taker_side'] == side)

                strategy_name = f"{bucket_name} {side.upper()}"
                stats = self.calculate_market_level_stats(mask, strategy_name)
                if stats:
                    results.append(stats)

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('edge', ascending=False)

            print("\n### WHALE SIZE ANALYSIS ###\n")
            print(f"{'Strategy':<40} {'Markets':>8} {'WinRate':>8} {'Edge':>8} {'Profit':>12} {'Valid':>6}")
            print("-" * 90)

            for _, row in results_df.iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<40} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {valid:>6}")

        return results_df

    def analyze_time_patterns(self):
        """
        Analyze time-of-day and timing patterns in the public trade feed.
        """
        print("\n" + "="*80)
        print("TIME-BASED PATTERN ANALYSIS")
        print("="*80)

        results = []

        # Hour of day
        for hour in range(24):
            mask = self.df['hour'] == hour

            for side in ['yes', 'no']:
                side_mask = mask & (self.df['taker_side'] == side)
                strategy_name = f"Hour {hour:02d} {side.upper()}"
                stats = self.calculate_market_level_stats(side_mask, strategy_name)
                if stats:
                    results.append(stats)

        # Day of week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for dow in range(7):
            mask = self.df['day_of_week'] == dow

            for side in ['yes', 'no']:
                side_mask = mask & (self.df['taker_side'] == side)
                strategy_name = f"{days[dow]} {side.upper()}"
                stats = self.calculate_market_level_stats(side_mask, strategy_name)
                if stats:
                    results.append(stats)

        # Time blocks
        time_blocks = [
            ((0, 6), "Night (0-6am)"),
            ((6, 12), "Morning (6am-12pm)"),
            ((12, 18), "Afternoon (12-6pm)"),
            ((18, 24), "Evening (6pm-12am)")
        ]

        for (start, end), block_name in time_blocks:
            mask = (self.df['hour'] >= start) & (self.df['hour'] < end)

            for side in ['yes', 'no']:
                side_mask = mask & (self.df['taker_side'] == side)
                strategy_name = f"{block_name} {side.upper()}"
                stats = self.calculate_market_level_stats(side_mask, strategy_name)
                if stats:
                    results.append(stats)

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            validated = results_df[results_df['is_valid']]

            print("\n### VALIDATED TIME PATTERNS ONLY ###\n")
            if len(validated) > 0:
                validated = validated.sort_values('edge', ascending=False)
                print(f"{'Strategy':<30} {'Markets':>8} {'WinRate':>8} {'Edge':>8} {'Profit':>12}")
                print("-" * 80)
                for _, row in validated.iterrows():
                    print(f"{row['strategy']:<30} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f}")
            else:
                print("  No time-based patterns passed validation criteria.")

        return results_df

    def analyze_category_whale_patterns(self):
        """
        Analyze whale patterns by market category.
        """
        print("\n" + "="*80)
        print("CATEGORY + WHALE ANALYSIS")
        print("="*80)

        results = []

        # Get top categories
        top_categories = self.df['category'].value_counts().head(20).index.tolist()

        for category in top_categories:
            cat_mask = self.df['category'] == category

            # Whale trades in category
            for min_contracts in [100, 500]:
                whale_mask = cat_mask & (self.df['count'] >= min_contracts)

                for side in ['yes', 'no']:
                    side_mask = whale_mask & (self.df['taker_side'] == side)
                    strategy_name = f"{category} whale>={min_contracts} {side.upper()}"
                    stats = self.calculate_market_level_stats(side_mask, strategy_name)
                    if stats and stats['n_markets'] >= 20:
                        results.append(stats)

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('edge', ascending=False)

            print("\n### TOP CATEGORY + WHALE PATTERNS ###\n")
            print(f"{'Strategy':<45} {'Markets':>8} {'WinRate':>8} {'Edge':>8} {'Profit':>12} {'Valid':>6}")
            print("-" * 100)

            for _, row in results_df.head(30).iterrows():
                valid = "YES" if row['is_valid'] else "NO"
                print(f"{row['strategy']:<45} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {valid:>6}")

            # Show validated only
            print("\n### VALIDATED ONLY ###\n")
            validated = results_df[results_df['is_valid']]
            if len(validated) > 0:
                for _, row in validated.iterrows():
                    print(f"{row['strategy']}: Edge={row['edge']:+.1%}, Markets={row['n_markets']}, Profit=${row['total_profit']:,.0f}")
            else:
                print("  No category-whale patterns passed validation.")

        return results_df

    def analyze_price_side_comprehensive(self):
        """
        Comprehensive price x side analysis with proper validation.
        This is the foundation that already found YES at 80-90c.
        """
        print("\n" + "="*80)
        print("PRICE x SIDE COMPREHENSIVE ANALYSIS")
        print("="*80)

        results = []

        for price_low, price_high in [
            (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)
        ]:
            for side in ['yes', 'no']:
                mask = (
                    (self.df['trade_price'] >= price_low) &
                    (self.df['trade_price'] < price_high) &
                    (self.df['taker_side'] == side)
                )

                strategy_name = f"{side.upper()} at {price_low}-{price_high}c"
                stats = self.calculate_market_level_stats(mask, strategy_name)
                if stats:
                    results.append(stats)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('edge', ascending=False)

        print("\n### ALL PRICE x SIDE PATTERNS ###\n")
        print(f"{'Strategy':<25} {'Markets':>8} {'WinRate':>8} {'Breakev':>8} {'Edge':>8} {'Profit':>12} {'p-value':>10} {'Valid':>6}")
        print("-" * 100)

        for _, row in results_df.iterrows():
            valid = "YES" if row['is_valid'] else "NO"
            print(f"{row['strategy']:<25} {row['n_markets']:>8} {row['win_rate']:>7.1%} {row['breakeven']:>7.1%} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f} {row['p_value']:>10.4f} {valid:>6}")

        return results_df

    def analyze_consecutive_whale_direction(self):
        """
        Analyze consecutive whale trades in the same direction.

        Hypothesis: Multiple whales betting the same way indicates stronger signal.
        """
        print("\n" + "="*80)
        print("CONSECUTIVE WHALE DIRECTION ANALYSIS")
        print("="*80)

        # Sort by market and time
        df_sorted = self.df.sort_values(['market_ticker', 'timestamp']).copy()

        # Mark whale trades
        df_sorted['is_whale'] = df_sorted['count'] >= 100

        # Count consecutive same-direction whales per market
        market_whale_stats = []

        for ticker, group in df_sorted.groupby('market_ticker'):
            whales = group[group['is_whale']]

            if len(whales) >= 2:
                # Check if majority of whales are same direction
                yes_whales = (whales['taker_side'] == 'yes').sum()
                no_whales = (whales['taker_side'] == 'no').sum()

                total_whales = len(whales)
                majority_side = 'yes' if yes_whales > no_whales else 'no'
                majority_pct = max(yes_whales, no_whales) / total_whales

                market_whale_stats.append({
                    'market_ticker': ticker,
                    'total_whales': total_whales,
                    'yes_whales': yes_whales,
                    'no_whales': no_whales,
                    'majority_side': majority_side,
                    'majority_pct': majority_pct,
                    'market_result': group['market_result'].iloc[0],
                    'total_profit': group['actual_profit_dollars'].sum(),
                    'is_winner': group['is_winner'].iloc[0]
                })

        whale_df = pd.DataFrame(market_whale_stats)

        if len(whale_df) > 0:
            # Analyze by whale consensus levels
            for min_pct in [0.6, 0.7, 0.8, 0.9, 1.0]:
                consensus = whale_df[whale_df['majority_pct'] >= min_pct]

                if len(consensus) >= 20:
                    # Check if majority side = market result
                    correct = ((consensus['majority_side'] == 'yes') & (consensus['market_result'] == 'yes')) | \
                              ((consensus['majority_side'] == 'no') & (consensus['market_result'] == 'no'))

                    win_rate = correct.sum() / len(consensus)
                    profit = consensus[correct]['total_profit'].sum()

                    print(f"\nWhale consensus >= {min_pct*100:.0f}%:")
                    print(f"  Markets: {len(consensus)}")
                    print(f"  Win rate following majority: {win_rate:.1%}")
                    print(f"  Profit if following majority: ${profit:,.0f}")

        return whale_df

    def generate_report(self, output_path: Path = None):
        """
        Generate comprehensive analysis report.
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        all_results = []

        # Run all analyses
        print("\n1. Price x Side Analysis...")
        price_side = self.analyze_price_side_comprehensive()
        all_results.append(('Price x Side', price_side))

        print("\n2. Whale Following Analysis...")
        whale = self.analyze_whale_following_basic()
        all_results.append(('Whale Following', whale))

        print("\n3. Whale Size Signal Analysis...")
        whale_size = self.analyze_whale_size_signal()
        all_results.append(('Whale Size', whale_size))

        print("\n4. Whale Momentum Analysis...")
        whale_momentum = self.analyze_whale_momentum()
        all_results.append(('Whale Momentum', whale_momentum))

        print("\n5. Category + Whale Analysis...")
        category = self.analyze_category_whale_patterns()
        all_results.append(('Category Whale', category))

        print("\n6. Time Pattern Analysis...")
        time_patterns = self.analyze_time_patterns()
        all_results.append(('Time Patterns', time_patterns))

        print("\n7. Consecutive Whale Direction Analysis...")
        whale_consensus = self.analyze_consecutive_whale_direction()

        # Collect all validated strategies
        print("\n" + "="*80)
        print("FINAL VALIDATED STRATEGIES")
        print("="*80)

        validated = []
        for name, df in all_results:
            if df is not None and len(df) > 0:
                valid_df = df[df['is_valid']]
                if len(valid_df) > 0:
                    for _, row in valid_df.iterrows():
                        row_dict = row.to_dict()
                        row_dict['analysis_type'] = name
                        validated.append(row_dict)

        if validated:
            validated_df = pd.DataFrame(validated)
            validated_df = validated_df.sort_values('edge', ascending=False)

            print(f"\n{'Strategy':<55} {'Type':<15} {'Markets':>8} {'Edge':>8} {'Profit':>12}")
            print("-" * 110)

            for _, row in validated_df.iterrows():
                print(f"{row['strategy']:<55} {row['analysis_type']:<15} {row['n_markets']:>8} {row['edge']:>+7.1%} ${row['total_profit']:>11,.0f}")
        else:
            print("\nNo strategies passed all validation criteria.")

        return all_results, validated


def main():
    parser = argparse.ArgumentParser(description='Public Trade Feed Strategy Analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--whale', action='store_true', help='Whale following analysis')
    parser.add_argument('--timing', action='store_true', help='Time pattern analysis')
    parser.add_argument('--price', action='store_true', help='Price/side analysis')
    parser.add_argument('--momentum', action='store_true', help='Whale momentum analysis')
    parser.add_argument('--category', action='store_true', help='Category analysis')
    parser.add_argument('--consensus', action='store_true', help='Whale consensus analysis')

    args = parser.parse_args()

    # Default to all if no specific analysis requested
    if not any([args.all, args.whale, args.timing, args.price, args.momentum, args.category, args.consensus]):
        args.all = True

    analyzer = PublicTradeFeedAnalyzer()

    if args.all:
        analyzer.generate_report()
    else:
        if args.price:
            analyzer.analyze_price_side_comprehensive()
        if args.whale:
            analyzer.analyze_whale_following_basic()
        if args.momentum:
            analyzer.analyze_whale_momentum()
        if args.timing:
            analyzer.analyze_time_patterns()
        if args.category:
            analyzer.analyze_category_whale_patterns()
        if args.consensus:
            analyzer.analyze_consecutive_whale_direction()


if __name__ == "__main__":
    main()
