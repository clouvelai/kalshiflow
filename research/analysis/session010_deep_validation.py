#!/usr/bin/env python3
"""
Session 010 Part 2: Deep Validation of Promising Hypotheses

Two hypotheses passed the price proxy check:
1. H072: Fade recent move in high volatility - +8.1% improvement over baseline
2. H070: Drunk sports betting (high leverage) - +1.1% improvement over baseline

Now we must validate:
1. Concentration test (< 30%)
2. Temporal stability (works in multiple periods)
3. Statistical significance with Bonferroni correction
4. Economic/behavioral explanation

SPECIAL CONCERN: H072 shows +33% edge and +24.7% baseline edge - this is TOO HIGH.
The "fade recent move" strategy itself seems to have massive edge, which is suspicious.
Need to investigate methodology.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
from pathlib import Path
import pytz

DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades/enriched_trades_resolved_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

ET = pytz.timezone('America/New_York')
BONFERRONI_THRESHOLD = 0.003  # 0.05 / 16


def load_data():
    """Load trades data."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE, low_memory=False)
    print(f"Loaded {len(trades):,} trades")
    return trades


def deep_validate_h072(trades):
    """
    Deep validation of H072: Fade recent move in high volatility markets.

    CONCERN: +33% edge and +24.7% baseline is suspicious.
    Let's understand what's happening.
    """
    print("\n" + "="*80)
    print("DEEP VALIDATION: H072 - FADE RECENT MOVE IN HIGH VOLATILITY")
    print("="*80)

    # Calculate per-market data
    market_data = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 5:
            continue

        group = group.sort_values('timestamp')
        price_std = group['trade_price'].std()

        # Recent move: compare first 3 vs last 3 trades
        first_3 = group.head(3)['trade_price'].mean()
        last_3 = group.tail(3)['trade_price'].mean()
        major_move_up = last_3 > first_3

        result = group['market_result'].iloc[0]
        last_price = group.iloc[-1]['trade_price']

        market_data.append({
            'market_ticker': ticker,
            'price_std': price_std,
            'major_move_up': major_move_up,
            'market_result': result,
            'last_price': last_price,
            'n_trades': len(group),
            'timestamp': group.iloc[-1]['timestamp']
        })

    market_df = pd.DataFrame(market_data)

    # Check: What is the distribution of "major move" direction vs actual result?
    print("\n--- Understanding the 'Fade Recent Move' Signal ---")
    print(f"Total markets analyzed: {len(market_df)}")

    # When price moved UP, how often did market settle YES?
    moved_up = market_df[market_df['major_move_up']]
    moved_down = market_df[~market_df['major_move_up']]

    up_yes = (moved_up['market_result'] == 'yes').mean()
    down_no = (moved_down['market_result'] == 'no').mean()

    print(f"\nPrice moved UP: {len(moved_up)} markets")
    print(f"  Settled YES: {up_yes*100:.1f}%")
    print(f"  -> Fade UP (bet NO) would WIN: {(1-up_yes)*100:.1f}%")

    print(f"\nPrice moved DOWN: {len(moved_down)} markets")
    print(f"  Settled NO: {down_no*100:.1f}%")
    print(f"  -> Fade DOWN (bet YES) would WIN: {(1-down_no)*100:.1f}%")

    # AH-HA: The issue is likely that "fade" win rate is being compared
    # to a breakeven calculated from the FADE price, not the original price.
    # Let me recalculate more carefully.

    print("\n--- Recalculating with Correct Methodology ---")

    # For each market, if we FADE the recent move:
    # - If moved UP: we bet NO at (100 - last_price), win if result='no'
    # - If moved DOWN: we bet YES at last_price, win if result='yes'

    market_df['fade_side'] = np.where(market_df['major_move_up'], 'no', 'yes')
    market_df['fade_price'] = np.where(
        market_df['major_move_up'],
        100 - market_df['last_price'],  # Bet NO
        market_df['last_price']          # Bet YES
    )
    market_df['fade_wins'] = (
        (market_df['major_move_up'] & (market_df['market_result'] == 'no')) |
        (~market_df['major_move_up'] & (market_df['market_result'] == 'yes'))
    )

    # Overall fade statistics
    avg_fade_win = market_df['fade_wins'].mean()
    avg_fade_price = market_df['fade_price'].mean()
    breakeven = avg_fade_price / 100.0
    overall_edge = (avg_fade_win - breakeven) * 100

    print(f"\nOVERALL 'Fade Recent Move' (all markets):")
    print(f"  Win Rate: {avg_fade_win*100:.1f}%")
    print(f"  Avg Fade Price: {avg_fade_price:.1f}c")
    print(f"  Breakeven: {breakeven*100:.1f}%")
    print(f"  Edge: {overall_edge:+.1f}%")

    # Check: What is the distribution of fade prices?
    print(f"\n  Fade Price Distribution:")
    print(f"    Min: {market_df['fade_price'].min():.1f}c")
    print(f"    Max: {market_df['fade_price'].max():.1f}c")
    print(f"    Mean: {market_df['fade_price'].mean():.1f}c")
    print(f"    Median: {market_df['fade_price'].median():.1f}c")

    # The edge is too high. Let me check if this is because fade prices are very low.
    # If avg fade price is 20c, breakeven is only 20%, so winning 50% gives +30% edge!

    # Let's check: what's the baseline when we just BET at these prices (not fade)?
    print("\n--- Baseline Check: What if we just bet at the same prices? ---")

    # For baseline, we need to compare to: betting the SAME direction as fade price
    # i.e., if fade price is 20c (we bet NO), what's the win rate for NO bets at 20c generally?

    # Actually, the issue is structural: when price moves UP, last_price is HIGH,
    # so fade_price (NO price) is LOW. Low NO prices have low breakeven but also low win rate.
    # Vice versa for DOWN moves.

    # Let me analyze by direction:
    print("\nBreakdown by recent move direction:")

    for direction, label in [(True, 'UP (fade=NO)'), (False, 'DOWN (fade=YES)')]:
        subset = market_df[market_df['major_move_up'] == direction]
        if len(subset) == 0:
            continue

        win_rate = subset['fade_wins'].mean()
        avg_price = subset['fade_price'].mean()
        be = avg_price / 100.0
        edge = (win_rate - be) * 100

        print(f"\n  {label}:")
        print(f"    Markets: {len(subset)}")
        print(f"    Win Rate: {win_rate*100:.1f}%")
        print(f"    Avg Fade Price: {avg_price:.1f}c")
        print(f"    Breakeven: {be*100:.1f}%")
        print(f"    Edge: {edge:+.1f}%")

    # Now let's look at HIGH VOLATILITY specifically
    median_vol = market_df['price_std'].median()
    high_vol = market_df[market_df['price_std'] > median_vol].copy()

    print(f"\n--- HIGH VOLATILITY Markets (above median std) ---")
    print(f"Markets: {len(high_vol)}")

    hv_win_rate = high_vol['fade_wins'].mean()
    hv_avg_price = high_vol['fade_price'].mean()
    hv_be = hv_avg_price / 100.0
    hv_edge = (hv_win_rate - hv_be) * 100

    print(f"Win Rate: {hv_win_rate*100:.1f}%")
    print(f"Avg Fade Price: {hv_avg_price:.1f}c")
    print(f"Breakeven: {hv_be*100:.1f}%")
    print(f"Edge: {hv_edge:+.1f}%")

    # Critical comparison: Does HIGH VOL have MORE edge than ALL markets?
    improvement = hv_edge - overall_edge
    print(f"\nImprovement over all markets: {improvement:+.1f}%")
    print(f"Is high-vol specific: {improvement > 0}")

    # Concentration check
    print("\n--- Concentration Check ---")
    high_vol['profit'] = np.where(
        high_vol['fade_wins'],
        100 - high_vol['fade_price'],  # Win amount
        -high_vol['fade_price']        # Loss amount
    )
    total_profit = high_vol[high_vol['profit'] > 0]['profit'].sum()
    max_single = high_vol[high_vol['profit'] > 0].groupby('market_ticker')['profit'].sum().max()
    concentration = max_single / total_profit if total_profit > 0 else 1

    print(f"Total profit: ${total_profit/100:.2f}")
    print(f"Max single market profit: ${max_single/100:.2f}")
    print(f"Concentration: {concentration*100:.1f}%")
    print(f"Passes (<30%): {concentration < 0.30}")

    # Temporal stability
    print("\n--- Temporal Stability ---")
    high_vol_sorted = high_vol.sort_values('timestamp')
    n = len(high_vol_sorted)
    quarters = [
        high_vol_sorted.iloc[:n//4],
        high_vol_sorted.iloc[n//4:n//2],
        high_vol_sorted.iloc[n//2:3*n//4],
        high_vol_sorted.iloc[3*n//4:]
    ]

    for i, q in enumerate(quarters):
        q_win = q['fade_wins'].mean()
        q_price = q['fade_price'].mean()
        q_be = q_price / 100.0
        q_edge = (q_win - q_be) * 100
        print(f"  Q{i+1}: Edge = {q_edge:+.1f}%, N = {len(q)}")

    # Statistical significance
    print("\n--- Statistical Significance ---")
    n_markets = len(high_vol)
    n_wins = int(hv_win_rate * n_markets)
    p_value = stats.binomtest(n_wins, n_markets, hv_be, alternative='greater').pvalue

    print(f"P-value: {p_value:.6f}")
    print(f"Bonferroni significant (p < {BONFERRONI_THRESHOLD}): {p_value < BONFERRONI_THRESHOLD}")

    # CRITICAL INSIGHT
    print("\n" + "!"*60)
    print("CRITICAL INSIGHT: The 'fade recent move' strategy itself has edge!")
    print(f"ALL markets fade edge: {overall_edge:+.1f}%")
    print(f"HIGH-VOL markets fade edge: {hv_edge:+.1f}%")
    print(f"High-vol specific improvement: {improvement:+.1f}%")
    print()
    print("This is likely because:")
    print("1. Price movements are mean-reverting on average")
    print("2. OR there's a methodological issue with how we calculate fade prices")
    print("3. OR the 'last_price' used for breakeven doesn't reflect actual bid/ask")
    print("!"*60)

    return {
        'overall_fade_edge': overall_edge,
        'high_vol_edge': hv_edge,
        'improvement': improvement,
        'n_markets': n_markets,
        'concentration': concentration,
        'p_value': p_value,
        'temporal_stable': all(q['fade_wins'].mean() > q['fade_price'].mean()/100 for q in quarters)
    }


def deep_validate_h070(trades):
    """
    Deep validation of H070: Drunk sports betting with high leverage.

    Initial finding: +1.1% improvement over baseline (weak but positive).
    Let's do full validation.
    """
    print("\n" + "="*80)
    print("DEEP VALIDATION: H070 - DRUNK SPORTS BETTING (HIGH LEVERAGE)")
    print("="*80)

    SPORTS_CATEGORIES = ['KXNFL', 'KXNCAAF', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAMB', 'KXSOC']

    # Parse timestamps
    trades['datetime_et'] = pd.to_datetime(trades['timestamp'], unit='ms', utc=True).dt.tz_convert(ET)
    trades['hour_et'] = trades['datetime_et'].dt.hour
    trades['day_of_week'] = trades['datetime_et'].dt.dayofweek

    trades['is_sports'] = trades['market_ticker'].apply(
        lambda x: any(cat in x for cat in SPORTS_CATEGORIES)
    )

    # Signal: Late night weekend sports + high leverage
    late_night_hours = [23, 0, 1, 2, 3]
    weekend_days = [4, 5]

    signal_trades = trades[
        trades['is_sports'] &
        trades['hour_et'].isin(late_night_hours) &
        trades['day_of_week'].isin(weekend_days) &
        (trades['leverage_ratio'] > 3)
    ].copy()

    print(f"\nSignal trades: {len(signal_trades):,}")

    # Calculate fade statistics
    signal_trades['fade_price'] = 100 - signal_trades['trade_price']
    signal_trades['fade_wins'] = ~signal_trades['is_winner']

    # Market-level aggregation
    market_stats = signal_trades.groupby('market_ticker').agg({
        'fade_wins': 'mean',
        'fade_price': 'mean',
        'actual_profit_dollars': 'sum',
        'timestamp': 'first'
    }).reset_index()

    n_markets = len(market_stats)
    print(f"Unique markets: {n_markets}")

    if n_markets < 50:
        print(f"INSUFFICIENT SAMPLE: {n_markets} < 50")
        return {'status': 'INSUFFICIENT_SAMPLE', 'n_markets': n_markets}

    avg_win_rate = market_stats['fade_wins'].mean()
    avg_price = market_stats['fade_price'].mean()
    breakeven = avg_price / 100.0
    edge = (avg_win_rate - breakeven) * 100

    print(f"\nSignal Statistics:")
    print(f"  Win Rate: {avg_win_rate*100:.1f}%")
    print(f"  Avg Fade Price: {avg_price:.1f}c")
    print(f"  Breakeven: {breakeven*100:.1f}%")
    print(f"  Edge: {edge:+.1f}%")

    # Concentration check
    print("\n--- Concentration Check ---")

    # Calculate profit per trade
    signal_trades['fade_profit'] = np.where(
        signal_trades['fade_wins'],
        (100 - signal_trades['fade_price']) * signal_trades['count'] / 100,  # Win amount in dollars
        -signal_trades['fade_price'] * signal_trades['count'] / 100  # Loss amount in dollars
    )

    profit_by_market = signal_trades.groupby('market_ticker')['fade_profit'].sum()
    positive_profits = profit_by_market[profit_by_market > 0]
    total_profit = positive_profits.sum() if len(positive_profits) > 0 else 0
    max_single = positive_profits.max() if len(positive_profits) > 0 else 0
    concentration = max_single / total_profit if total_profit > 0 else 1

    print(f"Total profit: ${total_profit:.2f}")
    print(f"Max single market: ${max_single:.2f}")
    print(f"Concentration: {concentration*100:.1f}%")
    print(f"Passes (<30%): {concentration < 0.30}")

    # Temporal stability
    print("\n--- Temporal Stability ---")
    market_stats_sorted = market_stats.sort_values('timestamp')
    n = len(market_stats_sorted)
    quarters = [
        market_stats_sorted.iloc[:n//4],
        market_stats_sorted.iloc[n//4:n//2],
        market_stats_sorted.iloc[n//2:3*n//4],
        market_stats_sorted.iloc[3*n//4:]
    ]

    quarter_edges = []
    for i, q in enumerate(quarters):
        if len(q) > 0:
            q_win = q['fade_wins'].mean()
            q_price = q['fade_price'].mean()
            q_be = q_price / 100.0
            q_edge = (q_win - q_be) * 100
            quarter_edges.append(q_edge)
            print(f"  Q{i+1}: Edge = {q_edge:+.1f}%, N = {len(q)}")
        else:
            quarter_edges.append(0)
            print(f"  Q{i+1}: No data")

    positive_quarters = sum(1 for e in quarter_edges if e > 0)
    print(f"Positive quarters: {positive_quarters}/4")
    print(f"Temporal stable (>= 2 positive): {positive_quarters >= 2}")

    # Statistical significance
    print("\n--- Statistical Significance ---")
    n_wins = int(avg_win_rate * n_markets)
    p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

    print(f"P-value: {p_value:.6f}")
    print(f"Bonferroni significant (p < {BONFERRONI_THRESHOLD}): {p_value < BONFERRONI_THRESHOLD}")

    # Compare to baseline (all sports, high lev, any time)
    print("\n--- Baseline Comparison ---")
    baseline_trades = trades[
        trades['is_sports'] &
        (trades['leverage_ratio'] > 3)
    ].copy()

    baseline_trades['fade_price'] = 100 - baseline_trades['trade_price']
    baseline_trades['fade_wins'] = ~baseline_trades['is_winner']

    # Filter to same price range
    price_min = signal_trades['fade_price'].min()
    price_max = signal_trades['fade_price'].max()
    baseline_filtered = baseline_trades[
        (baseline_trades['fade_price'] >= price_min) &
        (baseline_trades['fade_price'] <= price_max)
    ]

    baseline_market = baseline_filtered.groupby('market_ticker').agg({
        'fade_wins': 'mean',
        'fade_price': 'mean'
    }).reset_index()

    baseline_win = baseline_market['fade_wins'].mean()
    baseline_price = baseline_market['fade_price'].mean()
    baseline_be = baseline_price / 100.0
    baseline_edge = (baseline_win - baseline_be) * 100

    improvement = edge - baseline_edge

    print(f"Signal edge: {edge:+.1f}%")
    print(f"Baseline edge: {baseline_edge:+.1f}%")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"Is NOT price proxy: {improvement > 0}")

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT FOR H070")
    print("="*60)

    passes_sample = n_markets >= 50
    passes_concentration = concentration < 0.30
    passes_temporal = positive_quarters >= 2
    passes_significance = p_value < BONFERRONI_THRESHOLD
    passes_proxy = improvement > 0

    print(f"  Sample size >= 50: {passes_sample} ({n_markets})")
    print(f"  Concentration < 30%: {passes_concentration} ({concentration*100:.1f}%)")
    print(f"  Temporal stability: {passes_temporal} ({positive_quarters}/4 positive)")
    print(f"  Bonferroni significant: {passes_significance} (p={p_value:.4f})")
    print(f"  Not price proxy: {passes_proxy} (+{improvement:.1f}%)")

    all_pass = all([passes_sample, passes_concentration, passes_temporal, passes_significance, passes_proxy])
    print(f"\n  ALL CRITERIA PASS: {all_pass}")

    return {
        'edge': edge,
        'n_markets': n_markets,
        'concentration': concentration,
        'temporal_positive': positive_quarters,
        'p_value': p_value,
        'improvement': improvement,
        'passes_all': all_pass,
        'passes': {
            'sample': passes_sample,
            'concentration': passes_concentration,
            'temporal': passes_temporal,
            'significance': passes_significance,
            'not_proxy': passes_proxy
        }
    }


def main():
    """Run deep validation on promising hypotheses."""
    print("="*80)
    print("SESSION 010: DEEP VALIDATION OF PROMISING HYPOTHESES")
    print("="*80)

    trades = load_data()

    results = {}

    # Deep validate H072
    results['H072'] = deep_validate_h072(trades)

    # Deep validate H070
    results['H070'] = deep_validate_h070(trades)

    # Summary
    print("\n" + "="*80)
    print("DEEP VALIDATION SUMMARY")
    print("="*80)

    print("\nH072 (Fade Recent Move in High Volatility):")
    print(f"  Edge: {results['H072'].get('high_vol_edge', 0):+.1f}%")
    print(f"  Improvement over baseline: {results['H072'].get('improvement', 0):+.1f}%")
    print(f"  CONCERN: Baseline 'fade' has {results['H072'].get('overall_fade_edge', 0):+.1f}% edge")
    print(f"  This suggests mean reversion or methodology issue")

    print("\nH070 (Drunk Sports + High Leverage):")
    print(f"  Edge: {results['H070'].get('edge', 0):+.1f}%")
    print(f"  Improvement over baseline: {results['H070'].get('improvement', 0):+.1f}%")
    print(f"  ALL CRITERIA PASS: {results['H070'].get('passes_all', False)}")

    # Save results
    output_file = REPORTS_DIR / f"session010_deep_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_types({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
