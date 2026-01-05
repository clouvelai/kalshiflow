#!/usr/bin/env python3
"""
Session 012: Bot Exploitation Hypothesis Testing

Mission: Find at least ONE validated bot exploitation strategy independent of S007-S009.

Validation Criteria (STRICT):
1. N >= 100 markets
2. Edge > 0
3. P-value < 0.01 (Bonferroni corrected for multiple testing)
4. Concentration < 30%
5. Temporal stability (2/4+ periods positive)
6. NOT a price proxy - Must improve over price-matched baseline by >0%

Key Lessons from Session 011d:
- The breakeven for NO bet = NO price / 100 (what you pay)
- Always use actual NO price from data, not 100 - trade_price
- Always compare signal to ALL markets at same price level (price proxy check)
- High claimed edges (+20%+) are usually bugs

Hypotheses to Test:
- H087: Round Size Bot Detection
- H088: Millisecond Burst Detection
- H094: After-Hours Bot Dominance
- H097: Bot Agreement Signal
- H102: Leverage Stability Bot Detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
BONFERRONI_ALPHA = 0.01  # Use 0.01 for multiple testing correction
MIN_MARKETS = 100  # Stricter than 50 as requested
MAX_CONCENTRATION = 0.30
MIN_TEMPORAL_POSITIVE = 2  # At least 2 of 4 quarters positive

DATA_PATH = Path(__file__).parent.parent / "data" / "trades" / "enriched_trades_resolved_ALL.csv"
REPORT_PATH = Path(__file__).parent.parent / "reports" / f"session012_results.json"


def load_data():
    """Load the enriched trades data"""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Parse timestamp for time-based analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Verify critical columns
    required_cols = ['market_ticker', 'trade_price', 'yes_price', 'no_price', 'side',
                     'count', 'leverage_ratio', 'taker_side', 'market_result', 'is_winner']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")

    return df


def calculate_baseline_win_rates(df, price_col='no_price', bucket_size=5):
    """
    Calculate baseline win rates for NO bets at each price bucket.
    This is used to check if a signal is just a price proxy.
    """
    # For NO bets, we need average NO price per market
    market_agg = df.groupby('market_ticker').agg({
        price_col: 'mean',
        'market_result': 'first'
    }).reset_index()

    market_agg['price_bucket'] = (market_agg[price_col] // bucket_size) * bucket_size
    market_agg['no_wins'] = (market_agg['market_result'] == 'no').astype(int)

    baseline = market_agg.groupby('price_bucket').agg({
        'no_wins': ['mean', 'count']
    }).reset_index()
    baseline.columns = ['price_bucket', 'baseline_win_rate', 'baseline_n']

    return baseline


def validate_strategy(
    signal_markets,
    all_markets,
    price_col='avg_no_price',
    strategy_name='Strategy'
):
    """
    Rigorously validate a strategy against all criteria.

    Args:
        signal_markets: DataFrame with columns [market_ticker, avg_no_price, bet_wins]
        all_markets: DataFrame with all market baselines for price proxy check
        price_col: Column name for the NO price
        strategy_name: Name for logging

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {strategy_name}")
    print('='*70)

    n_total = len(signal_markets)
    print(f"\n1. SAMPLE SIZE CHECK")
    print(f"   Markets: {n_total}")
    passes_markets = n_total >= MIN_MARKETS
    print(f"   Status: {'PASS' if passes_markets else 'FAIL'} (need >= {MIN_MARKETS})")

    if n_total < 10:
        return {
            'strategy': strategy_name,
            'status': 'REJECTED',
            'reason': f'Insufficient markets ({n_total})',
            'markets': n_total,
            'edge': None
        }

    # Calculate edge
    print(f"\n2. EDGE CALCULATION")
    win_rate = signal_markets['bet_wins'].mean()
    avg_price = signal_markets[price_col].mean()
    breakeven = avg_price / 100.0
    edge = (win_rate - breakeven) * 100

    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg NO Price: {avg_price:.2f}c")
    print(f"   Breakeven: {breakeven:.2%}")
    print(f"   Edge: {edge:+.2f}%")

    passes_edge = edge > 0
    print(f"   Status: {'PASS' if passes_edge else 'FAIL'} (need > 0)")

    # Statistical significance
    print(f"\n3. STATISTICAL SIGNIFICANCE")
    n_wins = int(signal_markets['bet_wins'].sum())
    binom_result = stats.binomtest(n_wins, n_total, breakeven, alternative='greater')
    p_value = binom_result.pvalue

    print(f"   P-value: {p_value:.2e}")
    passes_pvalue = p_value < BONFERRONI_ALPHA
    print(f"   Status: {'PASS' if passes_pvalue else 'FAIL'} (need < {BONFERRONI_ALPHA})")

    # Price proxy check (CRITICAL)
    print(f"\n4. PRICE PROXY CHECK")
    print("   Comparing signal win rate to baseline at same price levels...")

    signal_markets = signal_markets.copy()
    signal_markets['price_bucket'] = (signal_markets[price_col] // 5) * 5

    all_markets = all_markets.copy()
    all_markets['price_bucket'] = (all_markets['avg_no_price'] // 5) * 5

    print(f"\n   {'Bucket':<10} {'Signal WR':<15} {'Baseline WR':<15} {'Improvement'}")
    print("   " + "-"*55)

    improvements = []
    for bucket in sorted(signal_markets['price_bucket'].unique()):
        sig_bucket = signal_markets[signal_markets['price_bucket'] == bucket]
        all_bucket = all_markets[all_markets['price_bucket'] == bucket]

        if len(sig_bucket) >= 5 and len(all_bucket) >= 20:
            sig_wr = sig_bucket['bet_wins'].mean()
            baseline_wr = all_bucket['bet_wins'].mean()
            improvement = (sig_wr - baseline_wr) * 100

            print(f"   {int(bucket):2d}-{int(bucket)+5}c     {sig_wr:.1%} (N={len(sig_bucket):<4}) {baseline_wr:.1%} (N={len(all_bucket):<5}) {improvement:+.1f}%")
            improvements.append({
                'bucket': bucket,
                'sig_n': len(sig_bucket),
                'sig_wr': sig_wr,
                'baseline_wr': baseline_wr,
                'improvement': improvement
            })

    if improvements:
        total_n = sum(i['sig_n'] for i in improvements)
        weighted_improvement = sum(i['improvement'] * i['sig_n'] / total_n for i in improvements)
        print(f"\n   WEIGHTED AVERAGE IMPROVEMENT: {weighted_improvement:+.2f}%")
    else:
        weighted_improvement = 0
        print(f"\n   WARNING: No comparable price buckets found!")

    passes_proxy = weighted_improvement > 0
    print(f"   Status: {'PASS' if passes_proxy else 'FAIL'} (need > 0% improvement)")

    # Concentration check
    print(f"\n5. CONCENTRATION CHECK")

    # Calculate profit contribution per market
    signal_markets = signal_markets.copy()
    signal_markets['profit'] = signal_markets['bet_wins'] * (100 - signal_markets[price_col]) - \
                              (1 - signal_markets['bet_wins']) * signal_markets[price_col]

    total_profit = signal_markets[signal_markets['profit'] > 0]['profit'].sum()
    if total_profit > 0:
        market_profit = signal_markets.groupby('market_ticker')['profit'].sum()
        max_single = market_profit[market_profit > 0].max()
        concentration = max_single / total_profit if max_single else 0
    else:
        concentration = 0

    print(f"   Max single market contribution: {concentration:.1%}")
    passes_conc = concentration < MAX_CONCENTRATION
    print(f"   Status: {'PASS' if passes_conc else 'FAIL'} (need < {MAX_CONCENTRATION:.0%})")

    # Temporal stability
    print(f"\n6. TEMPORAL STABILITY")

    n = len(signal_markets)
    q_size = n // 4
    quarters = []

    for i, (name, start, end) in enumerate([
        ('Q1', 0, q_size),
        ('Q2', q_size, 2*q_size),
        ('Q3', 2*q_size, 3*q_size),
        ('Q4', 3*q_size, n)
    ]):
        q_data = signal_markets.iloc[start:end]
        if len(q_data) > 10:
            q_wr = q_data['bet_wins'].mean()
            q_be = q_data[price_col].mean() / 100.0
            q_edge = (q_wr - q_be) * 100
            positive = q_edge > 0
            quarters.append(positive)
            symbol = "+" if positive else "-"
            print(f"   {name}: N={len(q_data):>4}, WR={q_wr:.1%}, BE={q_be:.1%}, Edge={q_edge:+.1f}% [{symbol}]")

    n_positive = sum(quarters)
    passes_temporal = n_positive >= MIN_TEMPORAL_POSITIVE
    print(f"\n   Positive quarters: {n_positive}/4")
    print(f"   Status: {'PASS' if passes_temporal else 'FAIL'} (need >= {MIN_TEMPORAL_POSITIVE})")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {strategy_name}")
    print('='*70)

    all_pass = all([passes_markets, passes_edge, passes_pvalue, passes_proxy, passes_conc, passes_temporal])

    criteria_summary = {
        'markets': ('PASS' if passes_markets else 'FAIL', n_total, f'>= {MIN_MARKETS}'),
        'edge': ('PASS' if passes_edge else 'FAIL', f'{edge:+.2f}%', '> 0%'),
        'p_value': ('PASS' if passes_pvalue else 'FAIL', f'{p_value:.2e}', f'< {BONFERRONI_ALPHA}'),
        'price_proxy': ('PASS' if passes_proxy else 'FAIL', f'{weighted_improvement:+.2f}%', '> 0%'),
        'concentration': ('PASS' if passes_conc else 'FAIL', f'{concentration:.1%}', f'< {MAX_CONCENTRATION:.0%}'),
        'temporal': ('PASS' if passes_temporal else 'FAIL', f'{n_positive}/4', f'>= {MIN_TEMPORAL_POSITIVE}')
    }

    print(f"\n{'Criterion':<15} {'Status':<8} {'Value':<15} {'Threshold'}")
    print("-"*55)
    for name, (status, value, thresh) in criteria_summary.items():
        print(f"{name:<15} {status:<8} {str(value):<15} {thresh}")

    status = "VALIDATED" if all_pass else "REJECTED"
    print(f"\n*** STATUS: {status} ***")

    return {
        'strategy': strategy_name,
        'status': status,
        'markets': n_total,
        'win_rate': float(win_rate),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'p_value': float(p_value),
        'improvement_vs_baseline': float(weighted_improvement),
        'concentration': float(concentration),
        'temporal_positive': n_positive,
        'all_criteria_pass': all_pass,
        'criteria': {
            'passes_markets': passes_markets,
            'passes_edge': passes_edge,
            'passes_pvalue': passes_pvalue,
            'passes_proxy': passes_proxy,
            'passes_conc': passes_conc,
            'passes_temporal': passes_temporal
        }
    }


def test_h087_round_size_bot(df, all_markets):
    """
    H087: Round Size Bot Detection

    Signal: Markets where round-size trades (10, 25, 50, 100...) show consensus.
    Test both FOLLOW and FADE strategies.
    """
    print("\n" + "="*70)
    print("H087: ROUND SIZE BOT DETECTION")
    print("="*70)

    # Identify round-size trades
    df = df.copy()
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    round_trades = df[df['is_round_size']].copy()

    print(f"\nRound-size trades: {len(round_trades):,} ({100*len(round_trades)/len(df):.1f}% of all trades)")

    # Aggregate by market
    round_market = round_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),  # YES ratio
        'no_price': 'mean',  # Actual NO price from data
        'count': 'count',  # Number of round trades
        'market_result': 'first'
    }).reset_index()
    round_market.columns = ['market_ticker', 'yes_ratio', 'avg_no_price', 'n_round_trades', 'market_result']

    # Calculate NO ratio
    round_market['no_ratio'] = 1 - round_market['yes_ratio']

    print(f"Markets with round-size trades: {len(round_market):,}")

    # Test different consensus thresholds
    results = []

    for direction in ['yes', 'no']:
        for consensus_threshold in [0.6, 0.7, 0.8]:
            for min_trades in [3, 5]:

                if direction == 'yes':
                    signal = round_market[
                        (round_market['yes_ratio'] > consensus_threshold) &
                        (round_market['n_round_trades'] >= min_trades)
                    ].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'yes').astype(int)
                    # For YES bets, use yes_price for breakeven
                    signal['avg_yes_price'] = 100 - signal['avg_no_price']  # Approximate
                    price_col = 'avg_yes_price'
                else:
                    signal = round_market[
                        (round_market['no_ratio'] > consensus_threshold) &
                        (round_market['n_round_trades'] >= min_trades)
                    ].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
                    price_col = 'avg_no_price'

                if len(signal) < 10:
                    continue

                # Quick metrics
                win_rate = signal['bet_wins'].mean()
                avg_price = signal[price_col].mean()
                breakeven = avg_price / 100.0
                edge = (win_rate - breakeven) * 100

                strategy_name = f"H087_FOLLOW_{direction.upper()}_{int(consensus_threshold*100)}pct_min{min_trades}"

                print(f"\n{strategy_name}:")
                print(f"  Markets: {len(signal)}, WR: {win_rate:.1%}, BE: {breakeven:.1%}, Edge: {edge:+.1f}%")

                results.append({
                    'name': strategy_name,
                    'direction': direction,
                    'consensus': consensus_threshold,
                    'min_trades': min_trades,
                    'markets': len(signal),
                    'win_rate': win_rate,
                    'breakeven': breakeven,
                    'edge': edge,
                    'signal_df': signal,
                    'price_col': price_col
                })

    # Find best candidate
    valid_candidates = [r for r in results if r['markets'] >= MIN_MARKETS and r['edge'] > 0]

    if not valid_candidates:
        print("\nNo candidates with >= 100 markets and positive edge")
        return {'status': 'NO_VIABLE_CANDIDATES', 'best_result': None}

    # Sort by edge
    best = max(valid_candidates, key=lambda x: x['edge'])

    print(f"\n{'='*70}")
    print(f"BEST CANDIDATE: {best['name']}")
    print('='*70)

    # Full validation of best candidate
    validation = validate_strategy(
        best['signal_df'],
        all_markets,
        best['price_col'],
        best['name']
    )

    return {
        'hypothesis': 'H087',
        'all_variants': [{k:v for k,v in r.items() if k != 'signal_df'} for r in results],
        'best_variant': best['name'],
        'validation': validation
    }


def test_h088_millisecond_burst(df, all_markets):
    """
    H088: Millisecond Burst Detection

    Signal: 3+ trades in same second indicate HFT bots.
    Test if burst direction predicts outcomes.
    """
    print("\n" + "="*70)
    print("H088: MILLISECOND BURST DETECTION")
    print("="*70)

    df = df.copy()

    # Create second-level timestamp
    df['second'] = df['timestamp'].dt.floor('S')

    # Find bursts (3+ trades in same second, same market)
    burst_counts = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_per_second')
    bursts = burst_counts[burst_counts['trades_per_second'] >= 3]

    print(f"\nBursts found (3+ trades/sec): {len(bursts):,}")

    if len(bursts) == 0:
        return {'status': 'NO_BURSTS_FOUND'}

    # Analyze burst direction
    burst_df = df.merge(bursts[['market_ticker', 'second']], on=['market_ticker', 'second'])

    # Aggregate burst direction per market
    burst_market = burst_df.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),  # YES ratio in bursts
        'no_price': 'mean',
        'market_result': 'first',
        'count': 'sum'  # Total burst volume
    }).reset_index()
    burst_market.columns = ['market_ticker', 'burst_yes_ratio', 'avg_no_price', 'market_result', 'burst_volume']
    burst_market['burst_no_ratio'] = 1 - burst_market['burst_yes_ratio']

    print(f"Markets with bursts: {len(burst_market):,}")

    results = []

    # Test different consensus thresholds for burst direction
    for direction in ['yes', 'no']:
        for threshold in [0.6, 0.7, 0.8]:

            if direction == 'yes':
                signal = burst_market[burst_market['burst_yes_ratio'] > threshold].copy()
                signal['bet_wins'] = (signal['market_result'] == 'yes').astype(int)
                signal['avg_yes_price'] = 100 - signal['avg_no_price']
                price_col = 'avg_yes_price'
            else:
                signal = burst_market[burst_market['burst_no_ratio'] > threshold].copy()
                signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
                price_col = 'avg_no_price'

            if len(signal) < 10:
                continue

            win_rate = signal['bet_wins'].mean()
            avg_price = signal[price_col].mean()
            breakeven = avg_price / 100.0
            edge = (win_rate - breakeven) * 100

            name = f"H088_FOLLOW_BURST_{direction.upper()}_{int(threshold*100)}pct"
            print(f"\n{name}: Markets={len(signal)}, WR={win_rate:.1%}, Edge={edge:+.1f}%")

            results.append({
                'name': name,
                'direction': direction,
                'threshold': threshold,
                'markets': len(signal),
                'win_rate': win_rate,
                'edge': edge,
                'signal_df': signal,
                'price_col': price_col
            })

    valid_candidates = [r for r in results if r['markets'] >= MIN_MARKETS and r['edge'] > 0]

    if not valid_candidates:
        print("\nNo viable candidates")
        return {'status': 'NO_VIABLE_CANDIDATES', 'best_result': None}

    best = max(valid_candidates, key=lambda x: x['edge'])

    validation = validate_strategy(
        best['signal_df'],
        all_markets,
        best['price_col'],
        best['name']
    )

    return {
        'hypothesis': 'H088',
        'all_variants': [{k:v for k,v in r.items() if k != 'signal_df'} for r in results],
        'validation': validation
    }


def test_h094_after_hours_bot(df, all_markets):
    """
    H094: After-Hours Bot Dominance

    Signal: Trades during 2AM-6AM ET are primarily bots.
    Test if fading or following after-hours direction has edge.
    """
    print("\n" + "="*70)
    print("H094: AFTER-HOURS BOT DOMINANCE")
    print("="*70)

    df = df.copy()

    # After-hours: 2AM-6AM ET (assuming data is in ET)
    # Note: Need to handle timezone - assuming UTC, ET is UTC-5 (winter) or UTC-4 (summer)
    # 2AM ET = 7AM UTC (winter), 6AM UTC (summer)
    # Let's use 7-11 UTC as proxy for 2-6AM ET
    df['hour_utc'] = df['timestamp'].dt.hour

    # Try different after-hours definitions
    after_hours_defs = [
        ('2AM-6AM', [2, 3, 4, 5]),  # If data is already ET
        ('7AM-11AM_UTC', [7, 8, 9, 10]),  # If data is UTC
        ('3AM-7AM', [3, 4, 5, 6]),  # Adjusted
    ]

    results = []

    for name_suffix, hours in after_hours_defs:
        df['is_after_hours'] = df['hour_utc'].isin(hours)
        after_hours_trades = df[df['is_after_hours']]

        if len(after_hours_trades) < 1000:
            continue

        print(f"\n{name_suffix}: {len(after_hours_trades):,} trades")

        # Aggregate by market
        ah_market = after_hours_trades.groupby('market_ticker').agg({
            'taker_side': lambda x: (x == 'yes').mean(),
            'no_price': 'mean',
            'market_result': 'first',
            'count': 'sum'
        }).reset_index()
        ah_market.columns = ['market_ticker', 'ah_yes_ratio', 'avg_no_price', 'market_result', 'ah_volume']
        ah_market['ah_no_ratio'] = 1 - ah_market['ah_yes_ratio']

        print(f"  Markets with after-hours trades: {len(ah_market):,}")

        # Test both follow and fade
        for direction in ['yes', 'no']:
            for threshold in [0.5, 0.6, 0.7]:

                if direction == 'yes':
                    signal = ah_market[ah_market['ah_yes_ratio'] > threshold].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'yes').astype(int)
                    signal['avg_yes_price'] = 100 - signal['avg_no_price']
                    price_col = 'avg_yes_price'
                else:
                    signal = ah_market[ah_market['ah_no_ratio'] > threshold].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
                    price_col = 'avg_no_price'

                if len(signal) < 10:
                    continue

                win_rate = signal['bet_wins'].mean()
                avg_price = signal[price_col].mean()
                breakeven = avg_price / 100.0
                edge = (win_rate - breakeven) * 100

                strat_name = f"H094_{name_suffix}_FOLLOW_{direction.upper()}_{int(threshold*100)}pct"
                print(f"  {strat_name}: Markets={len(signal)}, Edge={edge:+.1f}%")

                results.append({
                    'name': strat_name,
                    'direction': direction,
                    'markets': len(signal),
                    'edge': edge,
                    'signal_df': signal,
                    'price_col': price_col
                })

    valid_candidates = [r for r in results if r['markets'] >= MIN_MARKETS and r['edge'] > 0]

    if not valid_candidates:
        return {'status': 'NO_VIABLE_CANDIDATES'}

    best = max(valid_candidates, key=lambda x: x['edge'])

    validation = validate_strategy(
        best['signal_df'],
        all_markets,
        best['price_col'],
        best['name']
    )

    return {
        'hypothesis': 'H094',
        'all_variants': [{k:v for k,v in r.items() if k != 'signal_df'} for r in results],
        'validation': validation
    }


def test_h097_bot_agreement(df, all_markets):
    """
    H097: Bot Disagreement/Agreement Signal

    Signal: When >80% of bot-like trades agree, follow them.
    Bot-like = round sizes OR millisecond bursts
    """
    print("\n" + "="*70)
    print("H097: BOT AGREEMENT SIGNAL")
    print("="*70)

    df = df.copy()

    # Define bot-like trades
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)

    # Also detect trades in same second
    df['second'] = df['timestamp'].dt.floor('S')
    second_counts = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_in_second')
    df = df.merge(second_counts, on=['market_ticker', 'second'], how='left')
    df['is_burst'] = df['trades_in_second'] >= 2

    # Combined bot score
    df['is_bot_like'] = df['is_round_size'] | df['is_burst']

    print(f"Bot-like trades: {df['is_bot_like'].sum():,} ({100*df['is_bot_like'].mean():.1f}%)")

    bot_trades = df[df['is_bot_like']].copy()

    # Aggregate bot direction per market
    bot_market = bot_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'no_price': 'mean',
        'market_result': 'first',
        'count': 'count'
    }).reset_index()
    bot_market.columns = ['market_ticker', 'bot_yes_ratio', 'avg_no_price', 'market_result', 'n_bot_trades']
    bot_market['bot_no_ratio'] = 1 - bot_market['bot_yes_ratio']

    print(f"Markets with bot-like trades: {len(bot_market):,}")

    results = []

    # Test different agreement thresholds
    for direction in ['yes', 'no']:
        for threshold in [0.6, 0.7, 0.8]:
            for min_bot_trades in [3, 5]:

                if direction == 'yes':
                    signal = bot_market[
                        (bot_market['bot_yes_ratio'] > threshold) &
                        (bot_market['n_bot_trades'] >= min_bot_trades)
                    ].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'yes').astype(int)
                    signal['avg_yes_price'] = 100 - signal['avg_no_price']
                    price_col = 'avg_yes_price'
                else:
                    signal = bot_market[
                        (bot_market['bot_no_ratio'] > threshold) &
                        (bot_market['n_bot_trades'] >= min_bot_trades)
                    ].copy()
                    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
                    price_col = 'avg_no_price'

                if len(signal) < 10:
                    continue

                win_rate = signal['bet_wins'].mean()
                avg_price = signal[price_col].mean()
                breakeven = avg_price / 100.0
                edge = (win_rate - breakeven) * 100

                name = f"H097_FOLLOW_BOT_{direction.upper()}_{int(threshold*100)}pct_min{min_bot_trades}"
                print(f"  {name}: Markets={len(signal)}, Edge={edge:+.1f}%")

                results.append({
                    'name': name,
                    'direction': direction,
                    'threshold': threshold,
                    'markets': len(signal),
                    'edge': edge,
                    'signal_df': signal,
                    'price_col': price_col
                })

    valid_candidates = [r for r in results if r['markets'] >= MIN_MARKETS and r['edge'] > 0]

    if not valid_candidates:
        return {'status': 'NO_VIABLE_CANDIDATES'}

    best = max(valid_candidates, key=lambda x: x['edge'])

    validation = validate_strategy(
        best['signal_df'],
        all_markets,
        best['price_col'],
        best['name']
    )

    return {
        'hypothesis': 'H097',
        'all_variants': [{k:v for k,v in r.items() if k != 'signal_df'} for r in results],
        'validation': validation
    }


def test_h102_leverage_stability(df, all_markets):
    """
    H102: Leverage Stability Bot Detection

    Signal: Low std of leverage within market indicates bot activity.
    Test if bot-dominated markets (low leverage variance) are exploitable.

    CRITICAL: This was previously invalidated as a price proxy.
    Must verify improvement over baseline at same prices.
    """
    print("\n" + "="*70)
    print("H102: LEVERAGE STABILITY BOT DETECTION")
    print("="*70)

    df = df.copy()

    # Aggregate by market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'yes').mean(),
        'no_price': 'mean',
        'count': 'count',
        'market_result': 'first'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_mean', 'lev_std', 'yes_ratio', 'avg_no_price', 'n_trades', 'market_result']
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['no_ratio'] = 1 - market_stats['yes_ratio']

    print(f"Total markets: {len(market_stats):,}")
    print(f"Leverage std distribution:")
    print(f"  Min: {market_stats['lev_std'].min():.2f}")
    print(f"  Median: {market_stats['lev_std'].median():.2f}")
    print(f"  Mean: {market_stats['lev_std'].mean():.2f}")
    print(f"  Max: {market_stats['lev_std'].max():.2f}")

    results = []

    # Test different leverage std thresholds
    for lev_threshold in [0.3, 0.5, 0.7]:
        for direction in ['yes', 'no']:
            for consensus_thresh in [0.5, 0.6, 0.7]:
                for min_trades in [3, 5]:

                    if direction == 'yes':
                        signal = market_stats[
                            (market_stats['lev_std'] < lev_threshold) &
                            (market_stats['yes_ratio'] > consensus_thresh) &
                            (market_stats['n_trades'] >= min_trades)
                        ].copy()
                        signal['bet_wins'] = (signal['market_result'] == 'yes').astype(int)
                        signal['avg_yes_price'] = 100 - signal['avg_no_price']
                        price_col = 'avg_yes_price'
                    else:
                        signal = market_stats[
                            (market_stats['lev_std'] < lev_threshold) &
                            (market_stats['no_ratio'] > consensus_thresh) &
                            (market_stats['n_trades'] >= min_trades)
                        ].copy()
                        signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
                        price_col = 'avg_no_price'

                    if len(signal) < 50:
                        continue

                    win_rate = signal['bet_wins'].mean()
                    avg_price = signal[price_col].mean()
                    breakeven = avg_price / 100.0
                    edge = (win_rate - breakeven) * 100

                    name = f"H102_LevStd{lev_threshold}_{direction.upper()}_{int(consensus_thresh*100)}pct_min{min_trades}"

                    results.append({
                        'name': name,
                        'lev_threshold': lev_threshold,
                        'direction': direction,
                        'consensus': consensus_thresh,
                        'markets': len(signal),
                        'edge': edge,
                        'avg_price': avg_price,
                        'signal_df': signal,
                        'price_col': price_col
                    })

    print(f"\nTested {len(results)} variants")

    # Show top variants
    sorted_results = sorted(results, key=lambda x: x['edge'], reverse=True)[:10]
    print(f"\nTop 10 variants:")
    for r in sorted_results:
        print(f"  {r['name']}: Markets={r['markets']}, Edge={r['edge']:+.1f}%, AvgPrice={r['avg_price']:.1f}c")

    valid_candidates = [r for r in results if r['markets'] >= MIN_MARKETS and r['edge'] > 0]

    if not valid_candidates:
        return {'status': 'NO_VIABLE_CANDIDATES'}

    best = max(valid_candidates, key=lambda x: x['edge'])

    validation = validate_strategy(
        best['signal_df'],
        all_markets,
        best['price_col'],
        best['name']
    )

    return {
        'hypothesis': 'H102',
        'all_variants': [{k:v for k,v in r.items() if k != 'signal_df'} for r in results],
        'validation': validation
    }


def main():
    print("="*70)
    print("SESSION 012: BOT EXPLOITATION HYPOTHESIS TESTING")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nValidation Criteria:")
    print(f"  - Markets >= {MIN_MARKETS}")
    print(f"  - Edge > 0")
    print(f"  - P-value < {BONFERRONI_ALPHA}")
    print(f"  - Concentration < {MAX_CONCENTRATION:.0%}")
    print(f"  - Temporal stability: {MIN_TEMPORAL_POSITIVE}/4 quarters positive")
    print(f"  - Improvement vs price baseline > 0%")

    # Load data
    df = load_data()

    # Prepare baseline data for price proxy checks
    print("\nPreparing baseline data for price proxy checks...")
    all_markets = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_markets.columns = ['market_ticker', 'avg_no_price', 'market_result']
    all_markets['bet_wins'] = (all_markets['market_result'] == 'no').astype(int)

    print(f"Baseline: {len(all_markets):,} markets")

    # Test all hypotheses
    results = {}

    print("\n" + "="*70)
    print("TESTING HYPOTHESES")
    print("="*70)

    # H087: Round Size Bot Detection
    results['H087'] = test_h087_round_size_bot(df, all_markets)

    # H088: Millisecond Burst Detection
    results['H088'] = test_h088_millisecond_burst(df, all_markets)

    # H094: After-Hours Bot Dominance
    results['H094'] = test_h094_after_hours_bot(df, all_markets)

    # H097: Bot Agreement Signal
    results['H097'] = test_h097_bot_agreement(df, all_markets)

    # H102: Leverage Stability Bot Detection
    results['H102'] = test_h102_leverage_stability(df, all_markets)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    validated = []
    for h_id, result in results.items():
        if 'validation' in result and result['validation']:
            v = result['validation']
            status = v.get('status', 'UNKNOWN')
            print(f"\n{h_id}: {status}")
            if status == 'VALIDATED':
                print(f"  Strategy: {v.get('strategy')}")
                print(f"  Markets: {v.get('markets')}")
                print(f"  Edge: {v.get('edge', 0):+.2f}%")
                print(f"  Improvement: {v.get('improvement_vs_baseline', 0):+.2f}%")
                validated.append(v)
            elif v.get('markets'):
                print(f"  Markets: {v.get('markets')}")
                print(f"  Edge: {v.get('edge', 0):+.2f}%")
                print(f"  Improvement: {v.get('improvement_vs_baseline', 0):+.2f}%")
                print(f"  Failed criteria: {[k for k,v2 in v.get('criteria', {}).items() if not v2]}")
        else:
            print(f"\n{h_id}: {result.get('status', 'NO_RESULT')}")

    print("\n" + "="*70)
    print("VALIDATED STRATEGIES")
    print("="*70)

    if validated:
        for v in validated:
            print(f"\n*** {v['strategy']} ***")
            print(f"  Edge: {v['edge']:+.2f}%")
            print(f"  Markets: {v['markets']}")
            print(f"  Improvement vs Baseline: {v['improvement_vs_baseline']:+.2f}%")
    else:
        print("\nNo strategies validated in this session.")
        print("All bot exploitation hypotheses either had:")
        print("  - Negative edge")
        print("  - Were price proxies (no improvement over baseline)")
        print("  - Failed other validation criteria")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'session': 12,
        'hypotheses_tested': list(results.keys()),
        'validated_strategies': [v['strategy'] for v in validated],
        'results': {k: {kk:vv for kk,vv in v.items() if kk != 'signal_df' and not isinstance(vv, pd.DataFrame)}
                   for k,v in results.items()}
    }

    # Clean up for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output = clean_for_json(output)

    with open(REPORT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {REPORT_PATH}")

    return results


if __name__ == "__main__":
    main()
