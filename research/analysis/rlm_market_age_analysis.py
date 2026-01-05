"""
RLM MARKET AGE ANALYSIS
=======================

Analyzes whether a "minimum market age" filter improves RLM edge.

Definition:
    market_age_hours = (trade_time - market_open_time).total_seconds() / 3600

Steps:
1. Load trades + market metadata with open_time
2. Calculate market age at each trade
3. Filter to RLM signal conditions (YES_ratio > 0.65, n_trades >= 15, bet NO)
4. Group by age buckets and calculate edge
5. Test minimum thresholds
6. Provide recommendation
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
MARKET_DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/markets/market_outcomes_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/rlm_market_age_analysis.json'

# RLM Strategy Parameters
RLM_YES_THRESHOLD = 0.65
RLM_MIN_TRADES = 15


def load_data():
    """Load trades and market metadata with open_time."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load trades
    df = pd.read_csv(TRADE_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df):,} trades")

    # Load market metadata
    markets_df = pd.read_csv(MARKET_DATA_PATH)
    print(f"Loaded {len(markets_df):,} market records")

    # Check columns
    print(f"\nMarket columns: {list(markets_df.columns)}")

    # Parse open_time
    markets_df['open_time'] = pd.to_datetime(markets_df['open_time'], errors='coerce')
    valid_open = markets_df['open_time'].notna().sum()
    print(f"Markets with valid open_time: {valid_open:,}")

    return df, markets_df


def merge_and_calculate_age(df, markets_df):
    """Merge trades with market open_time and calculate market age."""
    print("\n" + "=" * 80)
    print("CALCULATING MARKET AGE")
    print("=" * 80)

    # Keep only relevant columns from markets
    markets_meta = markets_df[['ticker', 'open_time', 'result', 'status']].copy()
    markets_meta = markets_meta.rename(columns={'ticker': 'market_ticker', 'open_time': 'market_open_time'})

    # Merge
    df_merged = df.merge(markets_meta[['market_ticker', 'market_open_time']],
                         on='market_ticker', how='left')

    matched = df_merged['market_open_time'].notna().sum()
    print(f"Trades with market_open_time: {matched:,} ({matched/len(df)*100:.1f}%)")

    # Normalize timezones - make both tz-naive for comparison
    df_merged['datetime'] = pd.to_datetime(df_merged['datetime']).dt.tz_localize(None)
    df_merged['market_open_time'] = pd.to_datetime(df_merged['market_open_time']).dt.tz_localize(None)

    # Calculate market age at trade time
    df_merged['market_age_hours'] = (
        (df_merged['datetime'] - df_merged['market_open_time']).dt.total_seconds() / 3600
    )

    # Filter to valid ages (positive, reasonable)
    df_valid = df_merged[
        (df_merged['market_age_hours'].notna()) &
        (df_merged['market_age_hours'] > 0) &
        (df_merged['market_age_hours'] < 8760)  # < 1 year
    ].copy()

    print(f"Trades with valid market age: {len(df_valid):,}")
    print(f"Age range: {df_valid['market_age_hours'].min():.1f}h to {df_valid['market_age_hours'].max():.1f}h")
    print(f"Median age: {df_valid['market_age_hours'].median():.1f}h")

    return df_valid


def calculate_rlm_signals_by_age(df):
    """
    Calculate RLM signals at market level with age information.

    RLM Signal:
    - >65% YES trades + YES price dropped + >=15 trades
    - Direction: Bet NO
    """
    print("\n" + "=" * 80)
    print("CALCULATING RLM SIGNALS WITH AGE")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Market-level aggregation with age info
    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean'],
        'no_price': 'mean',
        'result': 'first',
        'count': 'size',
        'datetime': ['first', 'last'],
        'market_age_hours': ['min', 'max', 'mean', 'median'],  # Age at first/last/avg trade
        'market_open_time': 'first'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price',
        'avg_no_price', 'result', 'n_trades',
        'first_trade_time', 'last_trade_time',
        'min_age_hours', 'max_age_hours', 'avg_age_hours', 'median_age_hours',
        'market_open_time'
    ]

    # Price movement
    market_stats['yes_price_dropped'] = market_stats['last_yes_price'] < market_stats['first_yes_price']

    # Apply RLM filters
    rlm_mask = (
        (market_stats['yes_trade_ratio'] > RLM_YES_THRESHOLD) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_dropped'])
    )

    rlm_markets = market_stats[rlm_mask].copy()

    print(f"Total markets: {len(market_stats):,}")
    print(f"RLM signal markets: {len(rlm_markets):,}")

    # Age distribution of RLM markets
    print(f"\nRLM market age distribution (at first trade):")
    print(f"  Min: {rlm_markets['min_age_hours'].min():.1f}h")
    print(f"  25th: {rlm_markets['min_age_hours'].quantile(0.25):.1f}h")
    print(f"  Median: {rlm_markets['min_age_hours'].median():.1f}h")
    print(f"  75th: {rlm_markets['min_age_hours'].quantile(0.75):.1f}h")
    print(f"  Max: {rlm_markets['min_age_hours'].max():.1f}h")

    return rlm_markets, market_stats


def analyze_by_age_buckets(rlm_markets):
    """Analyze edge by market age buckets."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY AGE BUCKETS")
    print("=" * 80)

    # Using min_age_hours (age at first trade - when signal was generated)
    age_buckets = [
        (0, 6, "0-6hr"),
        (6, 24, "6-24hr"),
        (24, 48, "24-48hr"),
        (48, 168, "48hr-1wk"),
        (168, float('inf'), "1wk+")
    ]

    results = []

    for low, high, label in age_buckets:
        subset = rlm_markets[
            (rlm_markets['min_age_hours'] >= low) &
            (rlm_markets['min_age_hours'] < high)
        ]

        n = len(subset)
        if n >= 10:
            wins = (subset['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = subset['avg_no_price'].mean()
            breakeven = avg_price / 100
            edge = win_rate - breakeven

            # Statistical significance
            z_score = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven)) if 0 < breakeven < 1 else 0
            p_value = 1 - stats.norm.cdf(z_score)

            result = {
                'bucket': label,
                'low_hours': low,
                'high_hours': high,
                'n_markets': n,
                'wins': wins,
                'losses': n - wins,
                'win_rate': win_rate,
                'avg_no_price': avg_price,
                'breakeven': breakeven,
                'edge': edge,
                'edge_pct': edge * 100,
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            results.append(result)

            sig_marker = "*" if p_value < 0.05 else ""
            print(f"  {label:12s}: N={n:4d}, WR={win_rate:.1%}, Avg={avg_price:.0f}c, Edge={edge:+.1%} {sig_marker}")

    return results


def test_minimum_thresholds(rlm_markets):
    """Test minimum age thresholds for RLM."""
    print("\n" + "=" * 80)
    print("MINIMUM AGE THRESHOLD ANALYSIS")
    print("=" * 80)

    thresholds = [0, 6, 12, 24, 48, 72, 168]  # hours

    results = []

    print(f"\n{'Min Age':<12} {'N Markets':>10} {'Win Rate':>10} {'Avg Price':>10} {'Edge':>10} {'P-Value':>10}")
    print("-" * 70)

    for min_age in thresholds:
        subset = rlm_markets[rlm_markets['min_age_hours'] >= min_age]

        n = len(subset)
        if n >= 20:
            wins = (subset['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = subset['avg_no_price'].mean()
            breakeven = avg_price / 100
            edge = win_rate - breakeven

            # Statistical significance
            z_score = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven)) if 0 < breakeven < 1 else 0
            p_value = 1 - stats.norm.cdf(z_score)

            result = {
                'min_age_hours': min_age,
                'n_markets': n,
                'wins': wins,
                'losses': n - wins,
                'win_rate': win_rate,
                'avg_no_price': avg_price,
                'breakeven': breakeven,
                'edge': edge,
                'edge_pct': edge * 100,
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            results.append(result)

            sig_marker = "*" if p_value < 0.05 else ""
            print(f">={min_age:3d}hr       {n:10d} {win_rate:10.1%} {avg_price:10.0f}c {edge:+10.1%} {p_value:10.4f} {sig_marker}")

    return results


def analyze_young_markets(rlm_markets):
    """Deep dive on young markets (<24hr) to understand what's happening."""
    print("\n" + "=" * 80)
    print("YOUNG MARKETS DEEP DIVE (<24hr)")
    print("=" * 80)

    young = rlm_markets[rlm_markets['min_age_hours'] < 24].copy()
    old = rlm_markets[rlm_markets['min_age_hours'] >= 24].copy()

    print(f"\nYoung markets (<24hr): {len(young):,}")
    print(f"Old markets (>=24hr): {len(old):,}")

    if len(young) >= 20:
        young_wins = (young['result'] == 'no').sum()
        young_wr = young_wins / len(young)
        young_price = young['avg_no_price'].mean()
        young_edge = young_wr - (young_price / 100)

        print(f"\nYoung market stats:")
        print(f"  Win rate: {young_wr:.1%}")
        print(f"  Avg NO price: {young_price:.0f}c")
        print(f"  Edge: {young_edge:+.1%}")
        print(f"  YES trade ratio: {young['yes_trade_ratio'].mean():.1%}")
        print(f"  Avg trades: {young['n_trades'].mean():.0f}")

    if len(old) >= 20:
        old_wins = (old['result'] == 'no').sum()
        old_wr = old_wins / len(old)
        old_price = old['avg_no_price'].mean()
        old_edge = old_wr - (old_price / 100)

        print(f"\nOld market stats:")
        print(f"  Win rate: {old_wr:.1%}")
        print(f"  Avg NO price: {old_price:.0f}c")
        print(f"  Edge: {old_edge:+.1%}")
        print(f"  YES trade ratio: {old['yes_trade_ratio'].mean():.1%}")
        print(f"  Avg trades: {old['n_trades'].mean():.0f}")

    # Category breakdown of young markets
    if len(young) >= 20:
        young['category'] = young['market_ticker'].apply(
            lambda x: x.split('-')[0][:10] if '-' in x else x[:10]
        )
        print(f"\nYoung market categories:")
        for cat in young['category'].value_counts().head(10).index:
            cat_subset = young[young['category'] == cat]
            n = len(cat_subset)
            if n >= 5:
                wr = (cat_subset['result'] == 'no').sum() / n
                print(f"  {cat}: N={n}, WR={wr:.1%}")


def main():
    """Run market age analysis."""
    print("\n")
    print("=" * 80)
    print("RLM MARKET AGE ANALYSIS")
    print("=" * 80)
    print("Question: Does minimum market age filter improve RLM edge?")
    print("=" * 80)

    # Load data
    df, markets_df = load_data()

    # Merge and calculate age
    df_with_age = merge_and_calculate_age(df, markets_df)

    # Calculate RLM signals with age
    rlm_markets, market_stats = calculate_rlm_signals_by_age(df_with_age)

    if len(rlm_markets) < 50:
        print(f"\nINSUFFICIENT DATA: Only {len(rlm_markets)} RLM markets (need 50)")
        return

    # Analyze by age buckets
    bucket_results = analyze_by_age_buckets(rlm_markets)

    # Test minimum thresholds
    threshold_results = test_minimum_thresholds(rlm_markets)

    # Deep dive on young markets
    analyze_young_markets(rlm_markets)

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find optimal threshold (highest edge with statistical significance)
    sig_results = [r for r in threshold_results if r['significant'] and r['n_markets'] >= 50]

    if sig_results:
        best = max(sig_results, key=lambda x: x['edge'])
        baseline = threshold_results[0]  # min_age = 0

        improvement = best['edge'] - baseline['edge']
        sample_loss = baseline['n_markets'] - best['n_markets']

        print(f"\nBaseline (no filter): Edge={baseline['edge']:+.1%}, N={baseline['n_markets']}")
        print(f"Best threshold: >={best['min_age_hours']}hr, Edge={best['edge']:+.1%}, N={best['n_markets']}")
        print(f"Improvement: {improvement:+.1%}")
        print(f"Sample loss: {sample_loss} markets ({sample_loss/baseline['n_markets']*100:.0f}%)")

        if improvement > 0.02 and best['n_markets'] >= 50:
            print(f"\nRECOMMENDATION: IMPLEMENT min_age >= {best['min_age_hours']}hr filter")
            print(f"  - Edge improvement: {improvement:+.1%}")
            print(f"  - Acceptable sample size maintained")
        else:
            print(f"\nRECOMMENDATION: NO CHANGE")
            print(f"  - Edge improvement too small ({improvement:+.1%}) or sample too reduced")
    else:
        print("\nRECOMMENDATION: NO CHANGE - No significant improvement found")

    # Save results
    output = {
        'metadata': {
            'analysis': 'RLM Market Age Filter Analysis',
            'date': datetime.now().isoformat(),
            'total_rlm_markets': len(rlm_markets),
            'parameters': {
                'yes_threshold': RLM_YES_THRESHOLD,
                'min_trades': RLM_MIN_TRADES
            }
        },
        'age_bucket_analysis': bucket_results,
        'threshold_analysis': threshold_results,
        'recommendation': {
            'implement_filter': False,
            'reason': 'Analysis complete - see results'
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_PATH}")

    return output


if __name__ == '__main__':
    results = main()
