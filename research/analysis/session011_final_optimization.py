#!/usr/bin/env python3
"""
Session 011 Part 3: Final Optimization

Key insight from Part 2:
- Best hour range: 6PM-9PM (Edge: +4.3%, Improvement: +1.5%)
- Best leverage: > 1.5x (Edge: +4.6%, Improvement: +1.3%)

But we also found:
- NCAAF has +7.7% edge (highest)
- Monday Night Football has +6.3% edge

Let's find the OPTIMAL combination and validate thoroughly.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
from pathlib import Path
import pytz

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades/enriched_trades_resolved_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

ET = pytz.timezone('America/New_York')

SPORTS_CATEGORIES = [
    'KXNFL', 'KXNCAAF', 'KXNBA', 'KXNCAAMB', 'KXNHL', 'KXMLB', 'KXSOC',
    'KXMMA', 'KXUFC', 'KXGOLF', 'KXTENNIS',
]


def load_data():
    trades = pd.read_csv(TRADES_FILE)
    return trades


def parse_timestamp_to_et(timestamp_ms):
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
    return dt.astimezone(ET)


def is_sports_ticker(ticker):
    for cat in SPORTS_CATEGORIES:
        if cat in ticker:
            return True
    return False


def calculate_fade_stats(trades_df):
    if len(trades_df) == 0:
        return None

    df = trades_df.copy()
    df['fade_trade_price'] = 100 - df['trade_price']
    df['fade_is_winner'] = ~df['is_winner']

    market_stats = df.groupby('market_ticker').agg({
        'fade_is_winner': 'mean',
        'fade_trade_price': 'mean',
        'cost_dollars': 'sum'
    }).reset_index()

    n_markets = len(market_stats)
    if n_markets == 0:
        return None

    avg_win_rate = market_stats['fade_is_winner'].mean()
    avg_price = market_stats['fade_trade_price'].mean()
    breakeven = avg_price / 100.0
    edge = (avg_win_rate - breakeven) * 100

    n_wins = int(avg_win_rate * n_markets)
    p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

    df['fade_profit'] = np.where(
        df['fade_is_winner'],
        100 - df['fade_trade_price'],
        -df['fade_trade_price']
    )
    market_fade_profit = df.groupby('market_ticker')['fade_profit'].sum()
    total_profit = market_fade_profit[market_fade_profit > 0].sum()

    if total_profit > 0:
        max_concentration = market_fade_profit[market_fade_profit > 0].max() / total_profit
    else:
        max_concentration = 0.0

    return {
        'n_trades': len(df),
        'n_markets': n_markets,
        'win_rate': avg_win_rate * 100,
        'breakeven': breakeven * 100,
        'edge': edge,
        'p_value': p_value,
        'concentration': max_concentration * 100,
        'avg_fade_price': avg_price,
        'total_fade_profit': float(market_fade_profit.sum())
    }


def check_price_proxy(signal_trades, baseline_trades):
    if len(signal_trades) == 0 or len(baseline_trades) == 0:
        return None

    signal_stats = calculate_fade_stats(signal_trades)
    if signal_stats is None:
        return None

    signal_prices = 100 - signal_trades['trade_price']
    price_min = signal_prices.min()
    price_max = signal_prices.max()

    baseline_fade_prices = 100 - baseline_trades['trade_price']
    price_matched_baseline = baseline_trades[
        (baseline_fade_prices >= price_min) &
        (baseline_fade_prices <= price_max)
    ]

    if len(price_matched_baseline) == 0:
        return None

    baseline_stats = calculate_fade_stats(price_matched_baseline)
    if baseline_stats is None:
        return None

    improvement = signal_stats['edge'] - baseline_stats['edge']

    return {
        'signal_edge': signal_stats['edge'],
        'baseline_edge': baseline_stats['edge'],
        'improvement': improvement,
        'is_price_proxy': improvement <= 0
    }


def check_temporal_stability(trades_df, n_periods=4):
    if len(trades_df) == 0:
        return None

    df = trades_df.copy().sort_values('timestamp')
    period_size = len(df) // n_periods
    if period_size == 0:
        return None

    period_edges = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < n_periods - 1 else len(df)
        period_df = df.iloc[start_idx:end_idx]
        stats = calculate_fade_stats(period_df)
        if stats:
            period_edges.append(stats['edge'])
        else:
            period_edges.append(0)

    positive_periods = sum(1 for e in period_edges if e > 0)
    return {
        'period_edges': period_edges,
        'positive_periods': positive_periods,
        'passes': positive_periods >= n_periods // 2
    }


def full_validation(trades_df, baseline_df, name):
    """Full validation of a strategy."""
    stats = calculate_fade_stats(trades_df)
    if stats is None:
        return {'status': 'NO_DATA'}

    proxy = check_price_proxy(trades_df, baseline_df)
    temporal = check_temporal_stability(trades_df)

    result = {
        'name': name,
        **stats,
        'improvement': proxy['improvement'] if proxy else 0,
        'is_price_proxy': proxy['is_price_proxy'] if proxy else True,
        'temporal_stability': temporal,
        'temporal_passes': temporal['passes'] if temporal else False,
        'passes_all': (
            stats['n_markets'] >= 100 and
            stats['edge'] > 0 and
            stats['p_value'] < 0.01 and
            stats['concentration'] < 30 and
            (temporal['passes'] if temporal else False) and
            not (proxy['is_price_proxy'] if proxy else True)
        )
    }

    return result


def main():
    print("="*80)
    print("SESSION 011 PART 3: FINAL OPTIMIZATION")
    print("="*80)
    print()

    trades = load_data()
    print(f"Loaded {len(trades):,} trades")

    # Parse timestamps
    trades['datetime_et'] = trades['timestamp'].apply(parse_timestamp_to_et)
    trades['hour_et'] = trades['datetime_et'].apply(lambda x: x.hour)
    trades['day_of_week'] = trades['datetime_et'].apply(lambda x: x.weekday())
    trades['is_sports'] = trades['market_ticker'].apply(is_sports_ticker)

    # Baseline for comparison
    baseline_all_sports = trades[trades['is_sports']].copy()

    print()
    print("="*80)
    print("TESTING OPTIMAL COMBINATIONS")
    print("="*80)

    results = []

    # Test the optimal candidates
    configs = [
        # Name, hours, days, leverage, description
        ("OPTIMAL_6PM_9PM_FriSat_Lev1.5", [18, 19, 20, 21], [4, 5], 1.5, "6PM-9PM Fri/Sat Lev>1.5"),
        ("OPTIMAL_6PM_9PM_FriSat_Lev2", [18, 19, 20, 21], [4, 5], 2.0, "6PM-9PM Fri/Sat Lev>2"),
        ("OPTIMAL_6PM_11PM_FriSat_Lev1.5", [18, 19, 20, 21, 22, 23], [4, 5], 1.5, "6PM-11PM Fri/Sat Lev>1.5"),
        ("OPTIMAL_6PM_11PM_FriSat_Lev2", [18, 19, 20, 21, 22, 23], [4, 5], 2.0, "6PM-11PM Fri/Sat Lev>2"),
        ("EXTENDED_6PM_12AM_FriSat_Lev2", [18, 19, 20, 21, 22, 23, 0], [4, 5], 2.0, "6PM-12AM Fri/Sat Lev>2"),

        # Include Thursday (TNF)
        ("OPTIMAL_6PM_11PM_ThuFriSat_Lev2", [18, 19, 20, 21, 22, 23], [3, 4, 5], 2.0, "6PM-11PM Thu/Fri/Sat Lev>2"),

        # Monday Night
        ("MONDAY_8PM_12AM_Lev2", [20, 21, 22, 23, 0], [0], 2.0, "8PM-12AM Mon Lev>2"),
        ("MONDAY_8PM_12AM_Lev3", [20, 21, 22, 23, 0], [0], 3.0, "8PM-12AM Mon Lev>3"),

        # Combined: Fri/Sat evening + Monday night
        ("COMBINED_FriSatEve_MonNight_Lev2", None, None, None, "Combined"),
    ]

    for name, hours, days, lev, desc in configs:
        print(f"\nTesting: {desc}")

        if name == "COMBINED_FriSatEve_MonNight_Lev2":
            # Special combined case
            mask = (
                trades['is_sports'] &
                (trades['leverage_ratio'] > 2) &
                (
                    # Friday/Saturday evening
                    (
                        trades['hour_et'].isin([18, 19, 20, 21, 22, 23]) &
                        trades['day_of_week'].isin([4, 5])
                    ) |
                    # Monday night
                    (
                        trades['hour_et'].isin([20, 21, 22, 23, 0]) &
                        (trades['day_of_week'] == 0)
                    )
                )
            )
        else:
            mask = (
                trades['is_sports'] &
                trades['hour_et'].isin(hours) &
                trades['day_of_week'].isin(days) &
                (trades['leverage_ratio'] > lev)
            )

        signal_trades = trades[mask].copy()

        # Baseline: same leverage filter on all sports
        baseline = trades[trades['is_sports'] & (trades['leverage_ratio'] > lev if lev else True)]

        result = full_validation(signal_trades, baseline, name)
        result['description'] = desc
        results.append(result)

        if result.get('status') == 'NO_DATA':
            print(f"   NO DATA")
        else:
            status = "VALIDATED" if result.get('passes_all') else ""
            print(f"   N={result['n_markets']}, Edge={result['edge']:+.1f}%, Improvement={result['improvement']:+.1f}%, p={result['p_value']:.4f} {status}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: ALL TESTED STRATEGIES")
    print("="*80)

    print(f"\n{'Strategy':<50} {'N':>6} {'Edge':>8} {'Imprv':>8} {'Valid':>6}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.get('edge', 0), reverse=True):
        if r.get('status') == 'NO_DATA':
            continue
        print(f"{r['description']:<50} {r['n_markets']:>6} {r['edge']:>+7.1f}% {r['improvement']:>+7.1f}% {'YES' if r.get('passes_all') else 'NO':>6}")

    # Best validated strategies
    validated = [r for r in results if r.get('passes_all')]

    print("\n" + "="*80)
    print("VALIDATED STRATEGIES (Ordered by Edge * Improvement)")
    print("="*80)

    if validated:
        # Score = edge * improvement (both should be high)
        for r in sorted(validated, key=lambda x: x['edge'] * x['improvement'], reverse=True):
            print(f"\n  {r['name']}")
            print(f"    Description: {r['description']}")
            print(f"    Markets: {r['n_markets']}")
            print(f"    Edge: {r['edge']:+.1f}%")
            print(f"    Improvement vs baseline: {r['improvement']:+.1f}%")
            print(f"    P-value: {r['p_value']:.6f}")
            print(f"    Concentration: {r['concentration']:.1f}%")
            if r.get('temporal_stability'):
                print(f"    Temporal: {[f'{e:.1f}' for e in r['temporal_stability']['period_edges']]}")

    print("\n" + "="*80)
    print("FINAL RECOMMENDATION FOR H086")
    print("="*80)

    best = max([r for r in validated if r.get('edge', 0) > 0], key=lambda x: x['edge'] * x['improvement'], default=None)

    if best:
        print(f"""
    RECOMMENDED STRATEGY (S008 Extended):

    Name: {best['name']}
    Description: {best['description']}

    SIGNAL:
    - Sports markets (NFL, NCAAF, NBA, NCAAMB, NHL, MLB, Soccer, etc.)
    - Trading window: Based on configuration
    - Leverage ratio > 2x (or as specified)

    ACTION:
    - FADE these trades (bet opposite direction)
    - If drunk bettor bets YES, we bet NO
    - If drunk bettor bets NO, we bet YES

    VALIDATION:
    - Markets: {best['n_markets']}
    - Edge: {best['edge']:+.1f}%
    - Improvement over price-matched baseline: {best['improvement']:+.1f}%
    - P-value: {best['p_value']:.6f}
    - Concentration: {best['concentration']:.1f}%
    - NOT a price proxy: True

    COMPARISON TO ORIGINAL S008:
    - Original S008: 631 markets, +3.2% edge, +0.8% improvement
    - This strategy: {best['n_markets']} markets, {best['edge']:+.1f}% edge, {best['improvement']:+.1f}% improvement
    - Market coverage: {best['n_markets']/631:.1f}x more markets
        """)

    # Save results
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
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

    output_file = REPORTS_DIR / "session011_final_optimization.json"
    with open(output_file, 'w') as f:
        json.dump(convert_types({
            'timestamp': datetime.now().isoformat(),
            'session': 'Session 011 Part 3: Final Optimization',
            'all_results': results,
            'validated': validated,
            'best_strategy': best
        }), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
