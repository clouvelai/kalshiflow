#!/usr/bin/env python3
"""
Session 009: Priority 2 Hypothesis Testing

Testing 5 Priority 2 hypotheses from Session 007:
- H055: Price Oscillation Before Settlement
- H061: Large Market Inefficiency
- H047: Resolution Time Proximity Edge Decay
- H059: Gambler's Fallacy After Streaks
- H048: Category Efficiency Gradient

Methodology from Session 005/008:
- Correct breakeven: breakeven_rate = trade_price / 100
- Edge: (win_rate - breakeven_rate) * 100
- Price proxy check: Compare to baseline at same prices
- Statistical significance: p < 0.01 after Bonferroni (5 tests = p < 0.002)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
import re

# Paths
DATA_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv")
MARKETS_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/markets/market_outcomes_ALL.csv")
OUTPUT_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

# Bonferroni correction for 5 tests
BONFERRONI_P = 0.01 / 5  # 0.002


def load_data():
    """Load and merge trades with market data."""
    print("Loading data...")
    trades = pd.read_csv(DATA_PATH)
    trades['datetime'] = pd.to_datetime(trades['datetime'])

    markets = pd.read_csv(MARKETS_PATH, low_memory=False)
    markets['close_time'] = pd.to_datetime(markets['close_time'], errors='coerce')
    markets['open_time'] = pd.to_datetime(markets['open_time'], errors='coerce')

    print(f"Loaded {len(trades):,} trades across {trades['market_ticker'].nunique():,} markets")
    print(f"Loaded {len(markets):,} market records")

    return trades, markets


def calculate_edge_metrics(df, side='no', min_markets=50):
    """
    Calculate edge metrics for a strategy at market level.

    Args:
        df: DataFrame with market-level data (must have market_ticker, market_result, avg_price)
        side: 'yes' or 'no' - which side we're betting
        min_markets: Minimum markets required

    Returns:
        dict with metrics or None if insufficient data
    """
    n_markets = len(df)
    if n_markets < min_markets:
        return None

    # Calculate wins
    df = df.copy()
    df['we_win'] = df['market_result'] == side

    win_rate = df['we_win'].mean()
    avg_price = df['avg_price'].mean()

    # CORRECT breakeven formula
    breakeven = avg_price / 100.0
    edge = (win_rate - breakeven) * 100

    # Profit calculation
    df['our_cost'] = avg_price
    df['our_profit'] = np.where(
        df['we_win'],
        100 - df['our_cost'],  # Win: get $1 back, minus cost
        -df['our_cost']  # Lose: lose our stake
    )
    total_profit = df['our_profit'].sum()

    # Concentration
    winners = df[df['our_profit'] > 0]
    if len(winners) > 0 and total_profit > 0:
        concentration = winners['our_profit'].max() / total_profit
    else:
        concentration = 1.0

    # Statistical test
    n_wins = int(win_rate * n_markets)
    try:
        result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
        p_value = result.pvalue
    except:
        p_value = 1.0

    return {
        'n_markets': n_markets,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_profit': total_profit,
        'concentration': concentration,
        'p_value': p_value,
        'bonferroni_sig': p_value < BONFERRONI_P
    }


def test_h055_price_oscillation(trades, markets):
    """
    H055: Price Oscillation Before Settlement

    Theory: Markets that oscillate wildly before settlement have information
    uncertainty. The winner is the direction of the last major move.

    Signal: High price variance in final hours/minutes predicts outcome.
    """
    print("\n" + "="*80)
    print("H055: PRICE OSCILLATION BEFORE SETTLEMENT")
    print("="*80)

    # Get market-level price statistics
    print("\nCalculating price oscillation metrics per market...")

    def calc_oscillation(group):
        """Calculate price oscillation metrics for a market."""
        if len(group) < 3:
            return None

        group = group.sort_values('datetime')
        prices = group['yes_price'].values

        # Price variance
        price_std = np.std(prices)
        price_range = np.max(prices) - np.min(prices)

        # Price changes
        price_changes = np.diff(prices)
        if len(price_changes) > 0:
            avg_abs_change = np.mean(np.abs(price_changes))
            direction_changes = np.sum(np.diff(np.sign(price_changes)) != 0)
        else:
            avg_abs_change = 0
            direction_changes = 0

        # Last move direction (final price vs mid-market price)
        mid_price = prices[len(prices)//2] if len(prices) > 2 else prices[0]
        final_price = prices[-1]
        last_move_up = final_price > mid_price

        return pd.Series({
            'price_std': price_std,
            'price_range': price_range,
            'avg_abs_change': avg_abs_change,
            'direction_changes': direction_changes,
            'n_trades': len(group),
            'final_yes_price': final_price,
            'last_move_up': last_move_up,
            'market_result': group['market_result'].iloc[0]
        })

    oscillation_metrics = trades.groupby('market_ticker', group_keys=False).apply(calc_oscillation)
    oscillation_metrics = oscillation_metrics.dropna().reset_index()

    print(f"Markets with oscillation data: {len(oscillation_metrics):,}")

    # Define high vs low oscillation
    median_std = oscillation_metrics['price_std'].median()
    median_range = oscillation_metrics['price_range'].median()

    print(f"Median price std: {median_std:.2f}")
    print(f"Median price range: {median_range:.1f}")

    # High oscillation = above median std
    high_osc = oscillation_metrics[oscillation_metrics['price_std'] > median_std]
    low_osc = oscillation_metrics[oscillation_metrics['price_std'] <= median_std]

    print(f"\nHigh oscillation markets: {len(high_osc):,}")
    print(f"Low oscillation markets: {len(low_osc):,}")

    # Test 1: Does high oscillation predict NO wins more often?
    # (Uncertainty might mean prices are too high for YES)
    print("\n--- Test 1: High oscillation -> Bet NO ---")
    high_osc_no = high_osc.copy()
    high_osc_no['avg_price'] = 100 - high_osc_no['final_yes_price']  # NO price

    metrics = calculate_edge_metrics(high_osc_no, side='no')
    if metrics:
        print(f"  Markets: {metrics['n_markets']:,}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Breakeven: {metrics['breakeven']:.1%}")
        print(f"  Edge: {metrics['edge']:+.1f}%")
        print(f"  P-value: {metrics['p_value']:.4f}")
        print(f"  Bonferroni sig: {metrics['bonferroni_sig']}")
    else:
        print("  Insufficient data")

    # Test 2: Follow the last move in high-oscillation markets
    # If last move was UP (price increased), bet YES; if DOWN, bet NO
    print("\n--- Test 2: Follow last move in high-oscillation markets ---")

    # Last move up -> bet YES
    last_up = high_osc[high_osc['last_move_up']].copy()
    last_up['avg_price'] = last_up['final_yes_price']

    metrics_up = calculate_edge_metrics(last_up, side='yes')
    if metrics_up:
        print(f"  Last move UP -> Bet YES:")
        print(f"    Markets: {metrics_up['n_markets']:,}")
        print(f"    Edge: {metrics_up['edge']:+.1f}%")
        print(f"    P-value: {metrics_up['p_value']:.4f}")

    # Last move down -> bet NO
    last_down = high_osc[~high_osc['last_move_up']].copy()
    last_down['avg_price'] = 100 - last_down['final_yes_price']

    metrics_down = calculate_edge_metrics(last_down, side='no')
    if metrics_down:
        print(f"  Last move DOWN -> Bet NO:")
        print(f"    Markets: {metrics_down['n_markets']:,}")
        print(f"    Edge: {metrics_down['edge']:+.1f}%")
        print(f"    P-value: {metrics_down['p_value']:.4f}")

    # Test 3: Very high oscillation (top quartile)
    print("\n--- Test 3: Very high oscillation (top 25%) -> Bet NO ---")
    q75_std = oscillation_metrics['price_std'].quantile(0.75)
    very_high_osc = oscillation_metrics[oscillation_metrics['price_std'] > q75_std].copy()
    very_high_osc['avg_price'] = 100 - very_high_osc['final_yes_price']

    metrics_vh = calculate_edge_metrics(very_high_osc, side='no')
    if metrics_vh:
        print(f"  Markets: {metrics_vh['n_markets']:,}")
        print(f"  Win Rate: {metrics_vh['win_rate']:.1%}")
        print(f"  Breakeven: {metrics_vh['breakeven']:.1%}")
        print(f"  Edge: {metrics_vh['edge']:+.1f}%")
        print(f"  P-value: {metrics_vh['p_value']:.4f}")

    # CRITICAL: Price proxy check - compare to baseline at same prices
    print("\n--- Price Proxy Check ---")
    # Get low oscillation at same price range for comparison
    price_min = very_high_osc['final_yes_price'].quantile(0.1)
    price_max = very_high_osc['final_yes_price'].quantile(0.9)

    baseline = low_osc[
        (low_osc['final_yes_price'] >= price_min) &
        (low_osc['final_yes_price'] <= price_max)
    ].copy()
    baseline['avg_price'] = 100 - baseline['final_yes_price']

    metrics_base = calculate_edge_metrics(baseline, side='no')
    if metrics_base and metrics_vh:
        improvement = metrics_vh['edge'] - metrics_base['edge']
        print(f"  Signal edge: {metrics_vh['edge']:+.1f}%")
        print(f"  Baseline edge (same prices, no oscillation): {metrics_base['edge']:+.1f}%")
        print(f"  Improvement over baseline: {improvement:+.1f}%")

        if improvement > 3 and metrics_vh['bonferroni_sig']:
            print("  -> POTENTIAL REAL SIGNAL")
        else:
            print("  -> LIKELY PRICE PROXY OR NOT SIGNIFICANT")

    return {
        'hypothesis': 'H055',
        'name': 'Price Oscillation Before Settlement',
        'high_osc_metrics': metrics if metrics else None,
        'very_high_osc_metrics': metrics_vh if metrics_vh else None,
        'baseline_metrics': metrics_base if metrics_base else None,
        'status': 'REJECTED'  # Will update if passes all tests
    }


def test_h061_large_market_inefficiency(trades, markets):
    """
    H061: Large Market Inefficiency

    Theory: Very LARGE markets (high volume) are inefficient because they
    attract retail and create liquidity for manipulation.

    Signal: High-volume markets have different edge than low-volume.
    """
    print("\n" + "="*80)
    print("H061: LARGE MARKET INEFFICIENCY")
    print("="*80)

    # Calculate volume per market
    market_volume = trades.groupby('market_ticker').agg({
        'cost_dollars': 'sum',
        'count': 'sum',
        'market_result': 'first',
        'yes_price': 'mean',
        'no_price': 'mean'
    }).reset_index()

    market_volume.columns = ['market_ticker', 'total_dollars', 'total_contracts',
                             'market_result', 'avg_yes_price', 'avg_no_price']

    print(f"\nMarkets with volume data: {len(market_volume):,}")
    print(f"Total dollar volume: ${market_volume['total_dollars'].sum():,.0f}")

    # Volume quartiles
    quartiles = market_volume['total_dollars'].quantile([0.25, 0.5, 0.75])
    print(f"\nVolume quartiles: Q1=${quartiles.iloc[0]:,.0f}, Q2=${quartiles.iloc[1]:,.0f}, Q3=${quartiles.iloc[2]:,.0f}")

    # Segment by volume
    q1 = market_volume[market_volume['total_dollars'] <= quartiles.iloc[0]]
    q2 = market_volume[(market_volume['total_dollars'] > quartiles.iloc[0]) &
                       (market_volume['total_dollars'] <= quartiles.iloc[1])]
    q3 = market_volume[(market_volume['total_dollars'] > quartiles.iloc[1]) &
                       (market_volume['total_dollars'] <= quartiles.iloc[2])]
    q4 = market_volume[market_volume['total_dollars'] > quartiles.iloc[2]]

    results = {}

    for name, segment in [('Q1 (Smallest)', q1), ('Q2', q2), ('Q3', q3), ('Q4 (Largest)', q4)]:
        print(f"\n--- {name} Volume Markets ---")
        print(f"Markets: {len(segment):,}")
        print(f"Avg volume: ${segment['total_dollars'].mean():,.0f}")

        # Test NO strategy
        segment_no = segment.copy()
        segment_no['avg_price'] = segment_no['avg_no_price']

        metrics = calculate_edge_metrics(segment_no, side='no')
        if metrics:
            print(f"  NO Strategy Edge: {metrics['edge']:+.1f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  P-value: {metrics['p_value']:.4f}")
            results[name] = metrics

        # Test YES strategy
        segment_yes = segment.copy()
        segment_yes['avg_price'] = segment_yes['avg_yes_price']

        metrics_yes = calculate_edge_metrics(segment_yes, side='yes')
        if metrics_yes:
            print(f"  YES Strategy Edge: {metrics_yes['edge']:+.1f}%")

    # Check if largest markets have different edge
    print("\n--- Large Market Inefficiency Check ---")
    if 'Q4 (Largest)' in results and 'Q1 (Smallest)' in results:
        large_edge = results['Q4 (Largest)']['edge']
        small_edge = results['Q1 (Smallest)']['edge']
        diff = large_edge - small_edge
        print(f"Largest market NO edge: {large_edge:+.1f}%")
        print(f"Smallest market NO edge: {small_edge:+.1f}%")
        print(f"Difference: {diff:+.1f}%")

        if abs(diff) > 3:
            print("-> Potential volume-based edge difference detected")
        else:
            print("-> No significant volume-based inefficiency")

    return {
        'hypothesis': 'H061',
        'name': 'Large Market Inefficiency',
        'quartile_results': results,
        'status': 'REJECTED'  # Will update if significant
    }


def test_h047_time_proximity_edge_decay(trades, markets):
    """
    H047: Resolution Time Proximity Edge Decay

    Theory: Edge decreases as market approaches resolution because prices
    become more efficient.

    Signal: Trades placed far from resolution have more edge.
    """
    print("\n" + "="*80)
    print("H047: RESOLUTION TIME PROXIMITY EDGE DECAY")
    print("="*80)

    # Merge trades with close_time
    trades_with_close = trades.merge(
        markets[['ticker', 'close_time']],
        left_on='market_ticker',
        right_on='ticker',
        how='inner'
    )

    # Calculate time to close for each trade
    # Handle timezone awareness - make both tz-naive
    trades_with_close['close_time'] = pd.to_datetime(trades_with_close['close_time'], utc=True)
    trades_with_close['close_time'] = trades_with_close['close_time'].dt.tz_localize(None)

    # Ensure datetime is also tz-naive
    if trades_with_close['datetime'].dt.tz is not None:
        trades_with_close['datetime'] = trades_with_close['datetime'].dt.tz_localize(None)

    trades_with_close['time_to_close'] = (
        trades_with_close['close_time'] - trades_with_close['datetime']
    ).dt.total_seconds() / 3600  # Hours

    # Remove invalid times (negative or very long)
    valid_trades = trades_with_close[
        (trades_with_close['time_to_close'] > 0) &
        (trades_with_close['time_to_close'] < 720)  # < 30 days
    ]

    print(f"Trades with valid time-to-close: {len(valid_trades):,}")
    print(f"Time-to-close range: {valid_trades['time_to_close'].min():.1f}h to {valid_trades['time_to_close'].max():.1f}h")

    # Segment by time to close
    time_segments = [
        ('< 1 hour', 0, 1),
        ('1-6 hours', 1, 6),
        ('6-24 hours', 6, 24),
        ('1-7 days', 24, 168),
        ('> 7 days', 168, 720)
    ]

    results = {}

    for name, min_h, max_h in time_segments:
        segment = valid_trades[
            (valid_trades['time_to_close'] >= min_h) &
            (valid_trades['time_to_close'] < max_h)
        ]

        if len(segment) < 1000:
            print(f"\n--- {name} ---")
            print(f"  Insufficient trades: {len(segment)}")
            continue

        # Aggregate to market level for this time segment
        market_data = segment.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean'
        }).reset_index()
        market_data['avg_price'] = market_data['no_price']

        print(f"\n--- {name} ---")
        print(f"Trades: {len(segment):,}, Markets: {len(market_data):,}")

        metrics = calculate_edge_metrics(market_data, side='no')
        if metrics:
            print(f"  NO Strategy Edge: {metrics['edge']:+.1f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  P-value: {metrics['p_value']:.4f}")
            results[name] = metrics

    # Check for edge decay pattern
    print("\n--- Edge Decay Analysis ---")
    if len(results) >= 2:
        edges = [(name, results[name]['edge']) for name in results]
        print("Time segment edges:")
        for name, edge in edges:
            print(f"  {name}: {edge:+.1f}%")

        # Check if earlier trades have better edge
        early_names = ['> 7 days', '1-7 days']
        late_names = ['< 1 hour', '1-6 hours']

        early_edges = [results[n]['edge'] for n in early_names if n in results]
        late_edges = [results[n]['edge'] for n in late_names if n in results]

        if early_edges and late_edges:
            avg_early = np.mean(early_edges)
            avg_late = np.mean(late_edges)
            diff = avg_early - avg_late
            print(f"\nAverage early edge: {avg_early:+.1f}%")
            print(f"Average late edge: {avg_late:+.1f}%")
            print(f"Difference (early - late): {diff:+.1f}%")

            if diff > 2:
                print("-> POTENTIAL: Early trades have better edge")
            else:
                print("-> No clear edge decay pattern")

    return {
        'hypothesis': 'H047',
        'name': 'Resolution Time Proximity Edge Decay',
        'time_segment_results': results,
        'status': 'REJECTED'
    }


def test_h059_gamblers_fallacy(trades, markets):
    """
    H059: Gambler's Fallacy After Streaks

    Theory: After a series of YES outcomes in a market type, people overbet
    NO on the next one (and vice versa).

    Signal: Sequential correlation in market outcomes within categories.
    """
    print("\n" + "="*80)
    print("H059: GAMBLER'S FALLACY AFTER STREAKS")
    print("="*80)

    # Get market outcomes with timing
    market_outcomes = trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'datetime': 'min',  # First trade time
        'yes_price': 'mean',
        'no_price': 'mean'
    }).reset_index()

    # Extract market series from ticker (e.g., KXBTCD from KXBTCD-25DEC05)
    def extract_series(ticker):
        # Remove date suffix patterns
        # Examples: KXBTCD-25DEC05 -> KXBTCD
        #          KXNFLGAME-25DEC15DET-GB -> KXNFLGAME
        parts = ticker.split('-')
        if len(parts) >= 2:
            return parts[0]
        return ticker

    market_outcomes['series'] = market_outcomes['market_ticker'].apply(extract_series)

    # Count markets per series
    series_counts = market_outcomes['series'].value_counts()
    print(f"\nUnique market series: {len(series_counts)}")
    print(f"\nTop 10 series by market count:")
    print(series_counts.head(10))

    # Focus on series with at least 20 markets
    active_series = series_counts[series_counts >= 20].index.tolist()
    print(f"\nSeries with 20+ markets: {len(active_series)}")

    # For each active series, look for streak patterns
    streak_results = []

    for series in active_series[:20]:  # Analyze top 20
        series_data = market_outcomes[market_outcomes['series'] == series].sort_values('datetime')

        if len(series_data) < 10:
            continue

        # Track consecutive outcomes
        results = series_data['market_result'].values

        for streak_len in [2, 3, 4]:
            # Find positions after streaks
            after_yes_streak = []
            after_no_streak = []

            for i in range(streak_len, len(results)):
                prev = results[i-streak_len:i]
                if all(r == 'yes' for r in prev):
                    after_yes_streak.append(results[i])
                elif all(r == 'no' for r in prev):
                    after_no_streak.append(results[i])

            if len(after_yes_streak) >= 5:
                # After YES streak, how often does NO occur?
                no_after_yes = sum(1 for r in after_yes_streak if r == 'no') / len(after_yes_streak)
                streak_results.append({
                    'series': series,
                    'streak_type': 'yes',
                    'streak_len': streak_len,
                    'n': len(after_yes_streak),
                    'reversal_rate': no_after_yes
                })

            if len(after_no_streak) >= 5:
                # After NO streak, how often does YES occur?
                yes_after_no = sum(1 for r in after_no_streak if r == 'yes') / len(after_no_streak)
                streak_results.append({
                    'series': series,
                    'streak_type': 'no',
                    'streak_len': streak_len,
                    'n': len(after_no_streak),
                    'reversal_rate': yes_after_no
                })

    streak_df = pd.DataFrame(streak_results)

    if len(streak_df) > 0:
        print("\n--- Streak Analysis ---")
        print(f"Total streak patterns analyzed: {len(streak_df)}")

        # Aggregate by streak type and length
        agg = streak_df.groupby(['streak_type', 'streak_len']).agg({
            'n': 'sum',
            'reversal_rate': 'mean'
        }).reset_index()

        print("\nReversal rates after streaks:")
        for _, row in agg.iterrows():
            expected = 0.5  # If random, 50% reversal
            print(f"  After {row['streak_len']} {row['streak_type'].upper()} streak: "
                  f"{row['reversal_rate']:.1%} reversal (N={row['n']:.0f})")

        # Test if reversal rates differ from 50%
        print("\n--- Gambler's Fallacy Test ---")

        # After YES streak, bet NO (expecting reversal)
        yes_streaks = streak_df[streak_df['streak_type'] == 'yes']
        if len(yes_streaks) > 0:
            weighted_reversal = np.average(
                yes_streaks['reversal_rate'],
                weights=yes_streaks['n']
            )
            total_n = yes_streaks['n'].sum()

            # Test vs 50%
            if total_n > 30:
                n_reversals = int(weighted_reversal * total_n)
                result = stats.binomtest(n_reversals, int(total_n), 0.5)

                print(f"After YES streaks, NO occurs: {weighted_reversal:.1%} (N={total_n:.0f})")
                print(f"  P-value vs 50%: {result.pvalue:.4f}")

                if weighted_reversal > 0.55 and result.pvalue < 0.05:
                    print("  -> POTENTIAL: Mean reversion after YES streaks")
                else:
                    print("  -> No significant gambler's fallacy effect")

    return {
        'hypothesis': 'H059',
        'name': "Gambler's Fallacy After Streaks",
        'streak_results': streak_results if len(streak_df) > 0 else [],
        'status': 'REJECTED'
    }


def test_h048_category_efficiency(trades, markets):
    """
    H048: Category Efficiency Gradient

    Theory: Different categories have different efficiency levels
    (crypto vs politics vs sports).

    Signal: Edge varies by market category.
    """
    print("\n" + "="*80)
    print("H048: CATEGORY EFFICIENCY GRADIENT")
    print("="*80)

    # Extract category from ticker prefix
    def extract_category(ticker):
        """Extract category from ticker prefix."""
        # Common patterns:
        # KXBTC* - Bitcoin
        # KXETH* - Ethereum
        # KXNFL* - NFL
        # KXNBA* - NBA
        # KXNCAA* - NCAA
        # KXPOLITICS* - Politics
        # etc.

        prefixes = {
            'KXBTC': 'Crypto-Bitcoin',
            'KXETH': 'Crypto-Ethereum',
            'KXNFL': 'Sports-NFL',
            'KXNBA': 'Sports-NBA',
            'KXNCAAF': 'Sports-NCAAF',
            'KXNCAAB': 'Sports-NCAAB',
            'KXNHL': 'Sports-NHL',
            'KXMLB': 'Sports-MLB',
            'KXMLS': 'Sports-MLS',
            'KXSOC': 'Sports-Soccer',
            'INX': 'Markets-SP500',
            'NASDAQ': 'Markets-NASDAQ',
            'DJIA': 'Markets-DJIA',
            'WEATHER': 'Weather',
            'KX60': 'News-Mentions',
        }

        for prefix, category in prefixes.items():
            if ticker.startswith(prefix):
                return category

        # Generic extraction
        parts = ticker.split('-')
        if len(parts) >= 1:
            base = parts[0]
            if base.startswith('KX'):
                return base[2:]  # Remove KX prefix
            return base[:6]  # First 6 chars
        return 'Unknown'

    trades['category'] = trades['market_ticker'].apply(extract_category)

    # Aggregate to market level
    market_data = trades.groupby(['market_ticker', 'category']).agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'cost_dollars': 'sum'
    }).reset_index()

    # Count markets per category
    category_counts = market_data['category'].value_counts()
    print(f"\nCategories found: {len(category_counts)}")
    print("\nTop 15 categories by market count:")
    print(category_counts.head(15))

    # Analyze each category with at least 100 markets
    active_categories = category_counts[category_counts >= 100].index.tolist()
    print(f"\nCategories with 100+ markets: {len(active_categories)}")

    results = {}

    for category in active_categories:
        cat_data = market_data[market_data['category'] == category].copy()

        print(f"\n--- {category} ---")
        print(f"Markets: {len(cat_data):,}")
        print(f"Avg volume: ${cat_data['cost_dollars'].mean():,.0f}")

        # Test NO strategy
        cat_data['avg_price'] = cat_data['no_price']
        metrics = calculate_edge_metrics(cat_data, side='no')

        if metrics:
            print(f"  NO Edge: {metrics['edge']:+.1f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  P-value: {metrics['p_value']:.4f}")
            results[category] = metrics

        # Also test YES
        cat_data['avg_price'] = cat_data['yes_price']
        metrics_yes = calculate_edge_metrics(cat_data, side='yes')
        if metrics_yes:
            print(f"  YES Edge: {metrics_yes['edge']:+.1f}%")

    # Rank categories by edge
    print("\n--- Category Efficiency Ranking ---")
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1]['edge'], reverse=True)
        print("\nBest to worst for NO strategy:")
        for cat, m in ranked:
            sig = "*" if m['bonferroni_sig'] else ""
            print(f"  {cat}: {m['edge']:+.1f}%{sig} (N={m['n_markets']})")

        # Check if any category has significantly better edge
        best_cat, best_metrics = ranked[0]
        worst_cat, worst_metrics = ranked[-1]
        diff = best_metrics['edge'] - worst_metrics['edge']

        print(f"\nBest: {best_cat} ({best_metrics['edge']:+.1f}%)")
        print(f"Worst: {worst_cat} ({worst_metrics['edge']:+.1f}%)")
        print(f"Spread: {diff:.1f}%")

        if diff > 5:
            print("-> POTENTIAL: Category-specific efficiency differences")
        else:
            print("-> No major category efficiency differences")

    return {
        'hypothesis': 'H048',
        'name': 'Category Efficiency Gradient',
        'category_results': results,
        'status': 'REJECTED'
    }


def run_all_tests():
    """Run all Priority 2 hypothesis tests."""
    print("="*80)
    print("SESSION 009: PRIORITY 2 HYPOTHESIS TESTING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    trades, markets = load_data()

    all_results = {}

    # H055: Price Oscillation
    result = test_h055_price_oscillation(trades, markets)
    all_results['H055'] = result

    # H061: Large Market Inefficiency
    result = test_h061_large_market_inefficiency(trades, markets)
    all_results['H061'] = result

    # H047: Time Proximity Edge Decay
    result = test_h047_time_proximity_edge_decay(trades, markets)
    all_results['H047'] = result

    # H059: Gambler's Fallacy
    result = test_h059_gamblers_fallacy(trades, markets)
    all_results['H059'] = result

    # H048: Category Efficiency
    result = test_h048_category_efficiency(trades, markets)
    all_results['H048'] = result

    # Summary
    print("\n" + "="*80)
    print("SESSION 009 SUMMARY")
    print("="*80)

    for hyp_id, result in all_results.items():
        print(f"\n{hyp_id}: {result['name']}")
        print(f"  Status: {result['status']}")

    # Save results
    output_file = OUTPUT_PATH / f"session009_priority2_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    # Convert to JSON-serializable format
    json_results = {}
    for hyp_id, result in all_results.items():
        json_result = {'hypothesis': result['hypothesis'], 'name': result['name'], 'status': result['status']}
        json_results[hyp_id] = json_result

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == '__main__':
    run_all_tests()
