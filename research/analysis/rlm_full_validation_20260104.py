"""
RLM FULL VALIDATION ON UPDATED DATA - 2026-01-04
=================================================

This script performs comprehensive validation of the RLM (Reverse Line Movement)
trading strategy on the updated dataset (Dec 5, 2025 - Jan 4, 2026).

The RLM Strategy (H123):
- Signal: >65% YES trades + YES price dropped from open + >=15 trades
- Direction: Bet NO (contrarian to retail YES bias)
- Edge mechanism: Retail bettors pile into YES, but price moves toward NO (informed flow)

Validation Criteria:
1. Win rate vs bucket-matched baseline (price proxy check)
2. Statistical significance (p < 0.05)
3. Temporal stability (positive in multiple time periods)
4. Concentration check (<30% from any single market)
5. Sample size (N >= 50 markets)

Data:
- 7,886,537 trades (Dec 5, 2025 - Jan 4, 2026)
- 316,063 market outcomes
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRADE_DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/rlm_full_validation_20260104.json'

# RLM Strategy Parameters (from production config)
RLM_YES_THRESHOLD = 0.65  # >65% YES trades
RLM_MIN_TRADES = 15       # Minimum trades to evaluate signal
RLM_MIN_PRICE_DROP = 0    # Any price drop (0 = any movement toward NO)

# Analysis constants
WHALE_THRESHOLD = 10000   # $100 in cents
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]


def load_data():
    """Load and prepare the updated trade data."""
    print("=" * 80)
    print("LOADING UPDATED DATA (Dec 5, 2025 - Jan 4, 2026)")
    print("=" * 80)

    df = pd.read_csv(TRADE_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    df['date'] = df['datetime'].dt.date
    df['week'] = df['datetime'].dt.isocalendar().week

    print(f"Loaded {len(df):,} trades")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Unique markets: {df['market_ticker'].nunique():,}")

    # Filter to resolved markets only (column is 'result' not 'market_result')
    resolved_markets = df[df['result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved_markets)]

    print(f"Resolved markets: {len(resolved_markets):,}")
    print(f"Trades in resolved markets: {len(df_resolved):,}")

    return df_resolved


def build_baseline(df):
    """Build baseline win rates at 5c price buckets for NO bets."""
    print("\n" + "=" * 80)
    print("BUILDING PRICE-BUCKET BASELINE")
    print("=" * 80)

    all_markets = df.groupby('market_ticker').agg({
        'result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'first',
        'taker_side': lambda x: (x == 'yes').sum() / len(x)  # YES trade ratio
    }).reset_index()

    all_markets.columns = ['market_ticker', 'result', 'avg_no_price',
                          'avg_yes_price', 'first_trade', 'yes_trade_ratio']
    all_markets['bucket_5c'] = (all_markets['avg_no_price'] // 5) * 5

    # Build baseline for each bucket
    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            win_rate = (bucket_markets['result'] == 'no').mean()
            baseline[bucket] = {
                'win_rate': win_rate,
                'n_markets': n,
                'expected_edge': win_rate - (bucket + 2.5) / 100  # Midpoint of bucket
            }

    print(f"Built baseline across {len(baseline)} price buckets")
    print(f"Total baseline markets: {sum(b['n_markets'] for b in baseline.values()):,}")

    return all_markets, baseline


def identify_rlm_markets(df, yes_threshold=0.65, min_trades=15, min_price_drop=0):
    """
    Identify markets meeting RLM signal criteria.

    RLM Signal:
    1. >yes_threshold of trades are YES bets (retail bias)
    2. YES price dropped from open (smart money pressure)
    3. At least min_trades total trades
    """
    print("\n" + "=" * 80)
    print(f"IDENTIFYING RLM MARKETS")
    print(f"Parameters: YES threshold={yes_threshold:.0%}, min_trades={min_trades}, min_drop={min_price_drop}c")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Aggregate per-market statistics
    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),  # YES trade ratio
        'yes_price': ['first', 'last', 'mean', 'std'],
        'no_price': ['mean', 'first', 'last'],
        'result': 'first',
        'count': ['size', 'sum', 'mean'],
        'datetime': ['first', 'last', 'min', 'max'],
        'is_whale': ['sum', 'any'],
        'trade_value_cents': ['sum', 'mean'],
        'leverage_ratio': ['mean', 'std'],
        'is_weekend': 'any',
        'is_round_size': 'sum',
        'hour': 'mean'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price', 'yes_price_std',
        'avg_no_price', 'first_no_price', 'last_no_price',
        'result',
        'n_trades', 'total_contracts', 'avg_trade_size',
        'first_trade_time', 'last_trade_time', 'min_trade_time', 'max_trade_time',
        'whale_count', 'has_whale',
        'total_value', 'avg_trade_value',
        'avg_leverage', 'lev_std',
        'has_weekend', 'round_size_count', 'avg_hour'
    ]

    # Calculate price movement
    market_stats['yes_price_moved_down'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['yes_price_drop'] = market_stats['first_yes_price'] - market_stats['last_yes_price']
    market_stats['no_price_drop'] = market_stats['last_no_price'] - market_stats['first_no_price']  # Mirror of YES drop

    # Market duration
    market_stats['market_duration_hours'] = (
        (market_stats['last_trade_time'] - market_stats['first_trade_time']).dt.total_seconds() / 3600
    )

    # Fill NaN
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['yes_price_std'] = market_stats['yes_price_std'].fillna(0)

    # Apply RLM filters
    rlm_mask = (
        (market_stats['yes_trade_ratio'] > yes_threshold) &
        (market_stats['n_trades'] >= min_trades) &
        (market_stats['yes_price_moved_down'])
    )

    if min_price_drop > 0:
        rlm_mask = rlm_mask & (market_stats['yes_price_drop'] >= min_price_drop)

    rlm_markets = market_stats[rlm_mask].copy()

    print(f"\nRLM Signal Analysis:")
    print(f"  Total markets analyzed: {len(market_stats):,}")
    print(f"  Markets with >65% YES trades: {(market_stats['yes_trade_ratio'] > yes_threshold).sum():,}")
    print(f"  Markets with >=15 trades: {(market_stats['n_trades'] >= min_trades).sum():,}")
    print(f"  Markets with YES price drop: {(market_stats['yes_price_moved_down']).sum():,}")
    print(f"  RLM signal markets: {len(rlm_markets):,}")

    return rlm_markets, market_stats


def calculate_edge_with_bucket_matching(signal_markets, baseline, min_markets=50):
    """
    Calculate edge vs bucket-matched baseline.

    This is the CRITICAL validation - ensures we're not just seeing a price effect.
    """
    n = len(signal_markets)
    if n < min_markets:
        return {
            'valid': False,
            'reason': f'insufficient_markets_{n}',
            'n_markets': n
        }

    # Add price bucket
    signal_markets = signal_markets.copy()
    signal_markets['bucket_5c'] = (signal_markets['avg_no_price'] // 5) * 5

    # Overall statistics
    wins = (signal_markets['result'] == 'no').sum()
    losses = n - wins
    win_rate = wins / n
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100
    raw_edge = win_rate - breakeven

    # Statistical significance
    z_score = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven)) if 0 < breakeven < 1 else 0
    p_value = 1 - stats.norm.cdf(z_score)

    # Bucket-by-bucket analysis
    bucket_results = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)
        if n_sig < 5:  # Need minimum for reliable comparison
            continue

        sig_win_rate = (sig_bucket['result'] == 'no').mean()
        base_win_rate = baseline[bucket]['win_rate']
        improvement = sig_win_rate - base_win_rate

        bucket_results.append({
            'bucket': int(bucket),
            'n_signal': n_sig,
            'n_baseline': baseline[bucket]['n_markets'],
            'signal_win_rate': sig_win_rate,
            'baseline_win_rate': base_win_rate,
            'improvement': improvement,
            'positive': improvement > 0
        })

    # Calculate overall improvement vs baseline
    total_improvement = 0
    total_weight = 0
    positive_buckets = 0

    for br in bucket_results:
        weight = br['n_signal']
        total_improvement += br['improvement'] * weight
        total_weight += weight
        if br['positive']:
            positive_buckets += 1

    avg_improvement = total_improvement / total_weight if total_weight > 0 else 0
    bucket_ratio = positive_buckets / len(bucket_results) if bucket_results else 0

    # Confidence interval (95%)
    std_err = np.sqrt(win_rate * (1 - win_rate) / n)
    ci_lower = win_rate - 1.96 * std_err
    ci_upper = win_rate + 1.96 * std_err
    edge_ci_lower = ci_lower - breakeven
    edge_ci_upper = ci_upper - breakeven

    return {
        'valid': True,
        'n_markets': n,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_no_price': avg_no_price,
        'breakeven': breakeven,
        'raw_edge': raw_edge,
        'raw_edge_pct': raw_edge * 100,
        'z_score': z_score,
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'bucket_analysis': {
            'n_buckets': len(bucket_results),
            'positive_buckets': positive_buckets,
            'bucket_ratio': bucket_ratio,
            'avg_improvement': avg_improvement,
            'avg_improvement_pct': avg_improvement * 100,
            'buckets': bucket_results
        },
        'confidence_interval_95': {
            'win_rate_lower': ci_lower,
            'win_rate_upper': ci_upper,
            'edge_lower': edge_ci_lower,
            'edge_upper': edge_ci_upper
        },
        'is_price_proxy': bucket_ratio < 0.6 or avg_improvement < 0
    }


def check_temporal_stability(signal_markets):
    """Check if strategy performs consistently across time periods."""
    print("\n" + "=" * 80)
    print("TEMPORAL STABILITY CHECK")
    print("=" * 80)

    signal_markets = signal_markets.copy()
    signal_markets['week'] = signal_markets['first_trade_time'].dt.isocalendar().week
    signal_markets['month'] = signal_markets['first_trade_time'].dt.month

    # Weekly breakdown
    weekly_results = []
    for week in sorted(signal_markets['week'].unique()):
        week_data = signal_markets[signal_markets['week'] == week]
        n = len(week_data)
        if n >= 10:
            wins = (week_data['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = week_data['avg_no_price'].mean()
            edge = win_rate - (avg_price / 100)
            weekly_results.append({
                'week': int(week),
                'n_markets': n,
                'win_rate': win_rate,
                'avg_no_price': avg_price,
                'edge': edge,
                'profitable': edge > 0
            })

    positive_weeks = sum(1 for w in weekly_results if w['profitable'])

    print(f"Weekly analysis: {positive_weeks}/{len(weekly_results)} weeks profitable")
    for w in weekly_results:
        status = "+" if w['profitable'] else "-"
        print(f"  Week {w['week']}: N={w['n_markets']}, WR={w['win_rate']:.1%}, Edge={w['edge']:+.1%} [{status}]")

    return {
        'weekly_results': weekly_results,
        'positive_weeks': positive_weeks,
        'total_weeks': len(weekly_results),
        'temporal_stability_ratio': positive_weeks / len(weekly_results) if weekly_results else 0
    }


def check_concentration(signal_markets):
    """Check for concentration risk (no single market should dominate profit)."""
    print("\n" + "=" * 80)
    print("CONCENTRATION CHECK")
    print("=" * 80)

    # Calculate per-market profit contribution
    signal_markets = signal_markets.copy()
    signal_markets['won'] = (signal_markets['result'] == 'no').astype(int)
    signal_markets['profit'] = signal_markets['won'] - (signal_markets['avg_no_price'] / 100)

    total_profit = signal_markets['profit'].sum()

    # Get top contributors
    top_markets = signal_markets.nlargest(10, 'profit')[['market_ticker', 'profit', 'avg_no_price', 'won']]

    max_contribution = signal_markets['profit'].max() / total_profit if total_profit > 0 else 0
    top5_contribution = signal_markets.nlargest(5, 'profit')['profit'].sum() / total_profit if total_profit > 0 else 0

    print(f"Total simulated profit: {total_profit:.2f} units")
    print(f"Max single market contribution: {max_contribution:.1%}")
    print(f"Top 5 markets contribution: {top5_contribution:.1%}")

    # Category concentration
    signal_markets['category'] = signal_markets['market_ticker'].apply(lambda x: x.split('-')[0] if '-' in x else x[:4])
    category_profit = signal_markets.groupby('category')['profit'].sum()
    max_category_contribution = category_profit.max() / total_profit if total_profit > 0 else 0

    print(f"\nCategory concentration:")
    for cat in category_profit.nlargest(5).index:
        contrib = category_profit[cat] / total_profit
        print(f"  {cat}: {contrib:.1%} of profit")

    return {
        'total_profit': total_profit,
        'max_single_market_contribution': max_contribution,
        'top5_markets_contribution': top5_contribution,
        'max_category_contribution': max_category_contribution,
        'concentration_ok': max_contribution < 0.30,
        'category_breakdown': category_profit.to_dict()
    }


def analyze_by_price_drop(signal_markets, baseline):
    """Analyze edge by price drop magnitude (position scaling validation)."""
    print("\n" + "=" * 80)
    print("PRICE DROP MAGNITUDE ANALYSIS (Position Scaling)")
    print("=" * 80)

    drop_ranges = [
        (0, 5, "0-5c"),
        (5, 10, "5-10c"),
        (10, 20, "10-20c"),
        (20, float('inf'), "20c+")
    ]

    results = []
    for low, high, label in drop_ranges:
        subset = signal_markets[
            (signal_markets['yes_price_drop'] >= low) &
            (signal_markets['yes_price_drop'] < high)
        ]

        n = len(subset)
        if n >= 30:
            wins = (subset['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = subset['avg_no_price'].mean()
            edge = win_rate - (avg_price / 100)

            results.append({
                'range': label,
                'n_markets': n,
                'win_rate': win_rate,
                'avg_no_price': avg_price,
                'edge': edge,
                'edge_pct': edge * 100
            })

            print(f"  {label}: N={n:,}, WR={win_rate:.1%}, Avg Price={avg_price:.0f}c, Edge={edge:+.1%}")

    return results


def analyze_by_category(signal_markets, baseline):
    """Analyze edge by market category."""
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    signal_markets = signal_markets.copy()
    signal_markets['category'] = signal_markets['market_ticker'].apply(
        lambda x: x.split('-')[0] if '-' in x else x[:6]
    )

    results = []
    for cat in signal_markets['category'].value_counts().head(15).index:
        subset = signal_markets[signal_markets['category'] == cat]
        n = len(subset)

        if n >= 20:
            wins = (subset['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = subset['avg_no_price'].mean()
            edge = win_rate - (avg_price / 100)

            results.append({
                'category': cat,
                'n_markets': n,
                'win_rate': win_rate,
                'avg_no_price': avg_price,
                'edge': edge,
                'edge_pct': edge * 100
            })

            print(f"  {cat}: N={n:,}, WR={win_rate:.1%}, Edge={edge:+.1%}")

    return results


def main():
    """Run full RLM validation."""
    print("\n")
    print("=" * 80)
    print("RLM STRATEGY FULL VALIDATION - 2026-01-04")
    print("=" * 80)
    print(f"Strategy: H123 - Reverse Line Movement (RLM) NO")
    print(f"Parameters: YES threshold={RLM_YES_THRESHOLD:.0%}, min_trades={RLM_MIN_TRADES}")
    print(f"Data: Dec 5, 2025 - Jan 4, 2026 (updated)")
    print("=" * 80)

    # Load data
    df = load_data()

    # Build baseline
    all_markets, baseline = build_baseline(df)

    # Identify RLM markets
    rlm_markets, market_stats = identify_rlm_markets(
        df,
        yes_threshold=RLM_YES_THRESHOLD,
        min_trades=RLM_MIN_TRADES,
        min_price_drop=RLM_MIN_PRICE_DROP
    )

    # Calculate edge with bucket matching
    print("\n" + "=" * 80)
    print("EDGE CALCULATION (Bucket-Matched)")
    print("=" * 80)

    edge_results = calculate_edge_with_bucket_matching(rlm_markets, baseline)

    if edge_results['valid']:
        print(f"\nRLM Strategy Results:")
        print(f"  Markets: {edge_results['n_markets']:,}")
        print(f"  Wins/Losses: {edge_results['wins']}/{edge_results['losses']}")
        print(f"  Win Rate: {edge_results['win_rate']:.1%}")
        print(f"  Avg NO Price: {edge_results['avg_no_price']:.1f}c")
        print(f"  Breakeven: {edge_results['breakeven']:.1%}")
        print(f"  Raw Edge: {edge_results['raw_edge_pct']:+.1f}%")
        print(f"  Z-Score: {edge_results['z_score']:.2f}")
        print(f"  P-Value: {edge_results['p_value']:.6f}")
        print(f"  Statistically Significant (p<0.05): {edge_results['statistically_significant']}")
        print(f"\nBucket-Matched Analysis:")
        print(f"  Positive Buckets: {edge_results['bucket_analysis']['positive_buckets']}/{edge_results['bucket_analysis']['n_buckets']} ({edge_results['bucket_analysis']['bucket_ratio']:.0%})")
        print(f"  Avg Improvement vs Baseline: {edge_results['bucket_analysis']['avg_improvement_pct']:+.1f}%")
        print(f"  Is Price Proxy: {edge_results['is_price_proxy']}")
        print(f"\n95% Confidence Interval:")
        print(f"  Edge: [{edge_results['confidence_interval_95']['edge_lower']:.1%}, {edge_results['confidence_interval_95']['edge_upper']:.1%}]")
    else:
        print(f"VALIDATION FAILED: {edge_results['reason']}")

    # Temporal stability
    temporal_results = check_temporal_stability(rlm_markets)

    # Concentration check
    concentration_results = check_concentration(rlm_markets)

    # Price drop analysis
    price_drop_results = analyze_by_price_drop(rlm_markets, baseline)

    # Category analysis
    category_results = analyze_by_category(rlm_markets, baseline)

    # Compile final results
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    validation_checks = {
        'sample_size_ok': edge_results.get('n_markets', 0) >= 50,
        'statistically_significant': edge_results.get('statistically_significant', False),
        'not_price_proxy': not edge_results.get('is_price_proxy', True),
        'concentration_ok': concentration_results['concentration_ok'],
        'temporal_stability_ok': temporal_results['temporal_stability_ratio'] >= 0.5
    }

    all_checks_pass = all(validation_checks.values())

    print(f"\nValidation Checks:")
    for check, passed in validation_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    recommendation = "VALIDATED - Keep in production" if all_checks_pass else "NEEDS REVIEW"
    print(f"\nRecommendation: {recommendation}")

    # Compile output
    output = {
        'metadata': {
            'strategy': 'H123 - Reverse Line Movement (RLM) NO',
            'validation_date': '2026-01-04',
            'data_range': 'Dec 5, 2025 - Jan 4, 2026',
            'total_trades': len(df),
            'total_markets': market_stats['market_ticker'].nunique(),
            'resolved_markets': len(all_markets),
            'parameters': {
                'yes_threshold': RLM_YES_THRESHOLD,
                'min_trades': RLM_MIN_TRADES,
                'min_price_drop': RLM_MIN_PRICE_DROP
            }
        },
        'edge_results': edge_results,
        'temporal_stability': temporal_results,
        'concentration': concentration_results,
        'price_drop_analysis': price_drop_results,
        'category_breakdown': category_results,
        'validation_checks': validation_checks,
        'all_checks_pass': all_checks_pass,
        'recommendation': recommendation
    }

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return output


if __name__ == '__main__':
    results = main()
