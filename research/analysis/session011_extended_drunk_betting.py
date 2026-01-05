#!/usr/bin/env python3
"""
Session 011: Extended Drunk Betting Window (H086)

INSIGHT: The original H070/S008 validated late-night (11PM-3AM ET) weekend sports betting.
BUT this may be TOO NARROW:
- Games active at 11PM-3AM ET are primarily WEST COAST games
- West coast games START at 10PM ET (7PM PT)
- Drunk bettors on the east coast start drinking earlier (6-7PM)

H086: Test EXTENDED time windows for drunk sports betting:
1. Evening + Late Night: 7PM-3AM ET (19:00-03:00)
2. Prime Time: 8PM-12AM ET (20:00-00:00)
3. Full Weekend Evening: Friday 5PM - Sunday 3AM ET
4. Compare to original H070 window (11PM-3AM)

For each window, test:
- High leverage (>3x) - the original S008 signal
- Medium-high leverage (>2x) - slightly broader
- ANY leverage (just time + sports filter)

Methodology (CRITICAL):
- Correct breakeven formula: breakeven_rate = trade_price / 100.0
- Price proxy check: Signal must improve over baseline at same prices
- Statistical significance: p < 0.01 for validation
- N >= 100 markets (higher threshold for confidence)
- Concentration < 30%
- Temporal stability
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

# Timezone
ET = pytz.timezone('America/New_York')

# Sports categories - comprehensive list
SPORTS_CATEGORIES = [
    'KXNFL', 'KXNCAAF',   # Football
    'KXNBA', 'KXNCAAMB',  # Basketball
    'KXNHL',              # Hockey
    'KXMLB',              # Baseball
    'KXSOC',              # Soccer
    'KXMMA', 'KXUFC',     # MMA/UFC
    'KXGOLF',             # Golf
    'KXTENNIS',           # Tennis
]


def load_data():
    """Load trades data."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE)
    print(f"Loaded {len(trades):,} trades")
    return trades


def parse_timestamp_to_et(timestamp_ms):
    """Convert timestamp to ET timezone datetime."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
    return dt.astimezone(ET)


def is_sports_ticker(ticker):
    """Check if ticker is a sports market."""
    for cat in SPORTS_CATEGORIES:
        if cat in ticker:
            return True
    return False


def calculate_fade_stats(trades_df):
    """
    Calculate fade statistics for a set of trades.

    When fading: we bet opposite of what the original traders bet.
    - If they bet YES at yes_price, we bet NO at no_price = 100 - yes_price
    - We win if market settles to what the original trade LOST
    - fade_is_winner = NOT is_winner
    """
    if len(trades_df) == 0:
        return None

    df = trades_df.copy()

    # Calculate fade columns
    df['fade_trade_price'] = 100 - df['trade_price']
    df['fade_is_winner'] = ~df['is_winner']

    # Aggregate to market level
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

    # P-value
    n_wins = int(avg_win_rate * n_markets)
    if n_markets > 10:
        p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue
    else:
        p_value = 1.0

    # Concentration check
    # Note: We need to calculate fade profit, not original profit
    # Fade profit = fade_trade_price if we won, -(100 - fade_trade_price) if we lost
    df['fade_profit'] = np.where(
        df['fade_is_winner'],
        100 - df['fade_trade_price'],  # Won: profit = payout - cost = 100 - price
        -df['fade_trade_price']         # Lost: profit = -cost
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
        'total_fade_profit': market_fade_profit.sum()
    }


def check_price_proxy(signal_trades, baseline_trades):
    """
    CRITICAL CHECK: Is the signal just a price proxy?

    Compare signal edge to baseline edge at the same price distribution.
    Signal must IMPROVE over baseline to be considered real.
    """
    if len(signal_trades) == 0 or len(baseline_trades) == 0:
        return None

    signal_stats = calculate_fade_stats(signal_trades)
    if signal_stats is None:
        return None

    # Get price range of signal trades
    signal_prices = 100 - signal_trades['trade_price']  # fade prices
    price_min = signal_prices.min()
    price_max = signal_prices.max()

    # Filter baseline to same fade price range
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
        'is_price_proxy': improvement <= 0,
        'price_range': [price_min, price_max],
        'signal_markets': signal_stats['n_markets'],
        'baseline_markets': baseline_stats['n_markets']
    }


def check_temporal_stability(trades_df, n_periods=4):
    """Check if strategy works across multiple time periods."""
    if len(trades_df) == 0:
        return None

    df = trades_df.copy()
    df = df.sort_values('timestamp')

    # Split into periods
    period_size = len(df) // n_periods
    if period_size == 0:
        return None

    period_edges = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < n_periods - 1 else len(df)
        period_df = df.iloc[start_idx:end_idx]

        stats = calculate_fade_stats(period_df)
        if stats is not None:
            period_edges.append(stats['edge'])
        else:
            period_edges.append(0)

    positive_periods = sum(1 for e in period_edges if e > 0)

    return {
        'period_edges': period_edges,
        'positive_periods': positive_periods,
        'total_periods': n_periods,
        'passes': positive_periods >= n_periods // 2
    }


def test_time_window(trades, window_name, hours, days, leverage_threshold=None, side_filter=None):
    """
    Test a specific time window configuration.

    Args:
        trades: DataFrame with datetime_et, hour_et, day_of_week columns
        window_name: Name for this window
        hours: List of hours (0-23) in ET that qualify
        days: List of days (0=Mon, 6=Sun) that qualify
        leverage_threshold: If set, only include trades with leverage > threshold
        side_filter: If 'yes' or 'no', only include that side
    """
    # Apply filters
    mask = (
        trades['is_sports'] &
        trades['hour_et'].isin(hours) &
        trades['day_of_week'].isin(days)
    )

    if leverage_threshold is not None:
        mask = mask & (trades['leverage_ratio'] > leverage_threshold)

    if side_filter is not None:
        mask = mask & (trades['taker_side'] == side_filter)

    signal_trades = trades[mask].copy()

    if len(signal_trades) == 0:
        return {'status': 'NO_DATA', 'n_trades': 0}

    # Calculate fade stats
    stats = calculate_fade_stats(signal_trades)
    if stats is None:
        return {'status': 'CALC_ERROR', 'n_trades': len(signal_trades)}

    stats['window_name'] = window_name

    # Check concentration
    stats['concentration_passes'] = stats['concentration'] < 30

    # Temporal stability
    temporal = check_temporal_stability(signal_trades)
    if temporal:
        stats['temporal_stability'] = temporal
        stats['temporal_passes'] = temporal['passes']
    else:
        stats['temporal_passes'] = False

    # Price proxy check (compare to all sports trades at any time)
    all_sports = trades[trades['is_sports']].copy()
    if leverage_threshold is not None:
        # Also filter baseline by leverage for fair comparison
        all_sports_leveraged = all_sports[all_sports['leverage_ratio'] > leverage_threshold]
        proxy_check = check_price_proxy(signal_trades, all_sports_leveraged)
    else:
        proxy_check = check_price_proxy(signal_trades, all_sports)

    if proxy_check:
        stats['price_proxy_check'] = proxy_check
        stats['is_price_proxy'] = proxy_check['is_price_proxy']
        stats['improvement_vs_baseline'] = proxy_check['improvement']
    else:
        stats['is_price_proxy'] = True
        stats['improvement_vs_baseline'] = 0

    # Validation summary
    stats['passes_all'] = (
        stats['n_markets'] >= 100 and
        stats['edge'] > 0 and
        stats['p_value'] < 0.01 and
        stats['concentration_passes'] and
        stats['temporal_passes'] and
        not stats['is_price_proxy']
    )

    return stats


def main():
    """Test extended drunk betting windows."""
    print("="*80)
    print("SESSION 011: EXTENDED DRUNK BETTING WINDOW (H086)")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    trades = load_data()

    # Parse timestamps to ET
    print("Parsing timestamps to ET timezone...")
    trades['datetime_et'] = trades['timestamp'].apply(parse_timestamp_to_et)
    trades['hour_et'] = trades['datetime_et'].apply(lambda x: x.hour)
    trades['day_of_week'] = trades['datetime_et'].apply(lambda x: x.weekday())

    # Identify sports trades
    trades['is_sports'] = trades['market_ticker'].apply(is_sports_ticker)
    sports_count = trades['is_sports'].sum()
    print(f"Sports trades: {sports_count:,} ({100*trades['is_sports'].mean():.1f}%)")
    print()

    # Define time windows to test
    # Day codes: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun

    time_windows = {
        # Original H070 window (baseline)
        'ORIGINAL_11PM_3AM': {
            'hours': [23, 0, 1, 2, 3],
            'days': [4, 5],  # Friday, Saturday (their nights)
            'description': 'Original: 11PM-3AM ET on Fri/Sat nights'
        },

        # Extended evening window
        'EVENING_7PM_3AM': {
            'hours': [19, 20, 21, 22, 23, 0, 1, 2, 3],
            'days': [4, 5],  # Friday, Saturday
            'description': 'Extended: 7PM-3AM ET on Fri/Sat nights'
        },

        # Prime time only
        'PRIMETIME_8PM_12AM': {
            'hours': [20, 21, 22, 23, 0],
            'days': [4, 5],  # Friday, Saturday
            'description': 'Prime Time: 8PM-12AM ET on Fri/Sat nights'
        },

        # Earlier prime time (catching more games)
        'EARLY_EVENING_6PM_11PM': {
            'hours': [18, 19, 20, 21, 22, 23],
            'days': [4, 5],  # Friday, Saturday
            'description': 'Early Evening: 6PM-11PM ET on Fri/Sat'
        },

        # Full weekend evening (Friday 5PM - Sunday 3AM)
        'FULL_WEEKEND': {
            'hours': list(range(24)),  # All hours
            'days': [4, 5, 6],  # Friday, Saturday, Sunday
            'description': 'Full Weekend: Fri 5PM - Sun 3AM ET (all hours Fri-Sun)',
            'hour_filter': lambda h, d: (d == 4 and h >= 17) or (d == 5) or (d == 6 and h < 4)
        },

        # Weekend nights only (Saturday and Sunday night)
        'SAT_SUN_NIGHTS': {
            'hours': [19, 20, 21, 22, 23, 0, 1, 2, 3],
            'days': [5, 6],  # Saturday, Sunday
            'description': 'Sat/Sun Nights: 7PM-3AM ET'
        },

        # All weekend days 7PM-3AM
        'ALL_WEEKEND_NIGHTS': {
            'hours': [19, 20, 21, 22, 23, 0, 1, 2, 3],
            'days': [4, 5, 6],  # Fri, Sat, Sun
            'description': 'All Weekend Nights: 7PM-3AM Fri/Sat/Sun'
        },

        # Thursday night too (TNF)
        'THU_FRI_SAT_NIGHTS': {
            'hours': [19, 20, 21, 22, 23, 0, 1, 2, 3],
            'days': [3, 4, 5],  # Thu, Fri, Sat
            'description': 'Extended Weekend: 7PM-3AM Thu/Fri/Sat'
        },

        # Sunday afternoon (NFL main slate)
        'SUNDAY_NFL_SLATE': {
            'hours': [13, 14, 15, 16, 17, 18, 19],
            'days': [6],  # Sunday
            'description': 'Sunday NFL Slate: 1PM-7PM ET'
        },

        # Monday Night (MNF)
        'MONDAY_NIGHT': {
            'hours': [20, 21, 22, 23, 0],
            'days': [0],  # Monday
            'description': 'Monday Night: 8PM-12AM ET'
        },
    }

    # Leverage thresholds to test
    leverage_thresholds = {
        'high_lev_3x': 3.0,
        'med_lev_2x': 2.0,
        'any_leverage': None,
    }

    # Run tests
    all_results = {}
    validated_strategies = []

    print("="*80)
    print("TESTING ALL WINDOW/LEVERAGE COMBINATIONS")
    print("="*80)

    for window_name, window_config in time_windows.items():
        print(f"\n{'='*60}")
        print(f"WINDOW: {window_name}")
        print(f"  {window_config['description']}")
        print(f"{'='*60}")

        for lev_name, lev_threshold in leverage_thresholds.items():
            test_name = f"{window_name}__{lev_name}"

            result = test_time_window(
                trades,
                window_name=test_name,
                hours=window_config['hours'],
                days=window_config['days'],
                leverage_threshold=lev_threshold
            )

            all_results[test_name] = result

            # Print summary
            if result.get('status') in ['NO_DATA', 'CALC_ERROR']:
                print(f"  {lev_name}: {result['status']} ({result.get('n_trades', 0)} trades)")
            else:
                n_markets = result['n_markets']
                edge = result['edge']
                p_val = result['p_value']
                improvement = result.get('improvement_vs_baseline', 0)
                is_proxy = result.get('is_price_proxy', True)
                passes = result.get('passes_all', False)

                status = "VALIDATED" if passes else ""
                proxy_str = "PROXY" if is_proxy else f"+{improvement:.1f}%"

                print(f"  {lev_name}: N={n_markets}, Edge={edge:+.1f}%, p={p_val:.4f}, vs_baseline={proxy_str} {status}")

                if passes:
                    validated_strategies.append({
                        'name': test_name,
                        **result
                    })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: EXTENDED DRUNK BETTING ANALYSIS (H086)")
    print("="*80)

    # Sort all results by edge
    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if isinstance(v.get('edge'), (int, float))],
        key=lambda x: x[1]['edge'],
        reverse=True
    )

    print("\nTOP 15 STRATEGIES BY EDGE:")
    print("-"*80)
    print(f"{'Strategy':<45} {'Markets':>8} {'Edge':>8} {'P-val':>10} {'Imprv':>8} {'Valid':>6}")
    print("-"*80)

    for name, result in sorted_results[:15]:
        n_markets = result.get('n_markets', 0)
        edge = result.get('edge', 0)
        p_val = result.get('p_value', 1)
        improvement = result.get('improvement_vs_baseline', 0)
        passes = result.get('passes_all', False)

        print(f"{name:<45} {n_markets:>8} {edge:>+7.1f}% {p_val:>10.4f} {improvement:>+7.1f}% {'YES' if passes else 'NO':>6}")

    print("\n" + "="*80)
    print("VALIDATED STRATEGIES (All criteria passed):")
    print("="*80)

    if validated_strategies:
        for strat in validated_strategies:
            print(f"\n  {strat['name']}")
            print(f"    Markets: {strat['n_markets']}")
            print(f"    Edge: {strat['edge']:+.1f}%")
            print(f"    P-value: {strat['p_value']:.6f}")
            print(f"    Improvement vs baseline: {strat.get('improvement_vs_baseline', 0):+.1f}%")
            print(f"    Concentration: {strat.get('concentration', 0):.1f}%")
            if 'temporal_stability' in strat:
                print(f"    Temporal stability: {strat['temporal_stability']['period_edges']}")
    else:
        print("\n  NONE - No strategies passed all validation criteria")

    # Compare to original S008
    print("\n" + "="*80)
    print("COMPARISON TO ORIGINAL S008 (H070)")
    print("="*80)

    original = all_results.get('ORIGINAL_11PM_3AM__high_lev_3x', {})
    if original and 'edge' in original:
        print(f"\nOriginal S008 (11PM-3AM, Lev>3x):")
        print(f"  Markets: {original.get('n_markets', 'N/A')}")
        print(f"  Edge: {original.get('edge', 'N/A'):+.1f}%")
        print(f"  P-value: {original.get('p_value', 'N/A'):.6f}")
        print(f"  Improvement vs baseline: {original.get('improvement_vs_baseline', 'N/A'):+.1f}%")
    else:
        print("\nOriginal S008 not found in results")

    # Find best alternative
    best_alternatives = [r for r in sorted_results if r[0] != 'ORIGINAL_11PM_3AM__high_lev_3x' and not r[1].get('is_price_proxy', True)]

    if best_alternatives:
        print(f"\nBest non-proxy alternatives:")
        for name, result in best_alternatives[:5]:
            print(f"  {name}:")
            print(f"    Markets: {result['n_markets']}, Edge: {result['edge']:+.1f}%, Improvement: {result.get('improvement_vs_baseline', 0):+.1f}%")

    # Save results
    output_file = REPORTS_DIR / f"session011_h086_results.json"

    # Convert numpy types for JSON
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

    with open(output_file, 'w') as f:
        json.dump(convert_types({
            'timestamp': datetime.now().isoformat(),
            'session': 'Session 011: Extended Drunk Betting (H086)',
            'total_trades': len(trades),
            'sports_trades': int(sports_count),
            'time_windows_tested': list(time_windows.keys()),
            'leverage_thresholds': {k: v for k, v in leverage_thresholds.items()},
            'validated_strategies': validated_strategies,
            'all_results': all_results,
            'top_10_by_edge': [{'name': k, **v} for k, v in sorted_results[:10]]
        }), f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
