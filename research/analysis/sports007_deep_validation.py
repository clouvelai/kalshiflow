"""
SPORTS-007 DEEP VALIDATION: Late-Arriving Large Money

Initial screening showed:
- Follow Late NO: +19.8% raw edge, +15.0% improvement, 11/11 positive buckets
- Follow Late YES: +18.8% raw edge

This is VERY strong. Let's validate rigorously:
1. Temporal stability (4 quarters)
2. Concentration check (<30% from any single market)
3. Statistical significance
4. Category breakdown
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'


def load_data():
    """Load the enriched trades data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def build_baseline():
    """Build price bucket baseline."""
    df = pd.read_csv(DATA_PATH)

    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    all_markets['bucket_5c'] = ((all_markets['no_price'] / 5).astype(int) * 5)

    baseline = all_markets.groupby('bucket_5c').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline.columns = ['bucket_5c', 'baseline_no_rate', 'bucket_count']

    return all_markets, dict(zip(baseline['bucket_5c'], baseline['baseline_no_rate']))


def detect_late_large_signal(df):
    """Detect markets with late-arriving large money signal."""
    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    late_large_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 16:
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        # Early (first 75%) and late (final 25%)
        cutoff = 3 * n // 4
        early = mdf.iloc[:cutoff]
        late = mdf.iloc[cutoff:]

        if len(late) < 4:
            continue

        large_threshold = 5000  # $50

        early_large_ratio = (early['trade_value_cents'] > large_threshold).mean()
        late_large_ratio = (late['trade_value_cents'] > large_threshold).mean()

        if late_large_ratio > early_large_ratio * 2 and late_large_ratio > 0.2:
            late_large = late[late['trade_value_cents'] > large_threshold]
            if len(late_large) < 2:
                continue

            late_yes_ratio = (late_large['taker_side'] == 'yes').mean()
            late_direction = 'yes' if late_yes_ratio > 0.6 else ('no' if late_yes_ratio < 0.4 else 'neutral')

            if late_direction != 'neutral':
                # Calculate total late large value for concentration check
                late_value = late_large['trade_value_cents'].sum()

                late_large_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'late_direction': late_direction,
                    'late_yes_ratio': late_yes_ratio,
                    'early_large_ratio': early_large_ratio,
                    'late_large_ratio': late_large_ratio,
                    'late_large_count': len(late_large),
                    'late_value': late_value,
                    'first_trade_time': mdf['datetime'].iloc[0],
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    return pd.DataFrame(late_large_markets)


def validate_follow_no(ll_df, baseline_by_bucket):
    """Deep validation of Follow Late NO signal."""
    print("\n" + "=" * 80)
    print("DEEP VALIDATION: FOLLOW LATE NO")
    print("=" * 80)

    signal = ll_df[ll_df['late_direction'] == 'no'].copy()
    print(f"\nTotal signal markets: {len(signal)}")

    if len(signal) < 50:
        print("Insufficient sample")
        return None

    # ===== BASIC METRICS =====
    n = len(signal)
    wins = (signal['market_result'] == 'no').sum()
    wr = wins / n
    avg_price = signal['no_price'].mean()
    be = avg_price / 100
    edge = wr - be

    # P-value
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\nBasic Metrics:")
    print(f"  N = {n}")
    print(f"  Win Rate = {wr:.1%}")
    print(f"  Avg NO Price = {avg_price:.1f}c")
    print(f"  Breakeven = {be:.1%}")
    print(f"  Raw Edge = {edge:.1%}")
    print(f"  P-value = {p_value:.6f}")

    # ===== BUCKET-MATCHED ANALYSIS =====
    print("\n----- BUCKET-MATCHED ANALYSIS -----")
    signal['bucket_5c'] = ((signal['no_price'] / 5).astype(int) * 5)
    signal['baseline_rate'] = signal['bucket_5c'].map(baseline_by_bucket)
    signal['won'] = (signal['market_result'] == 'no').astype(int)

    bucket_analysis = signal.groupby('bucket_5c').agg({
        'won': ['sum', 'count', 'mean'],
        'no_price': 'mean',
        'baseline_rate': 'first'
    }).reset_index()
    bucket_analysis.columns = ['bucket', 'wins', 'count', 'win_rate', 'avg_price', 'baseline']
    bucket_analysis['improvement'] = bucket_analysis['win_rate'] - bucket_analysis['baseline']
    bucket_analysis = bucket_analysis[bucket_analysis['count'] >= 5]

    print(bucket_analysis.to_string(index=False))

    pos_buckets = (bucket_analysis['improvement'] > 0).sum()
    total_buckets = len(bucket_analysis)
    bucket_ratio = pos_buckets / max(total_buckets, 1)

    weighted_improvement = (bucket_analysis['improvement'] * bucket_analysis['count']).sum() / bucket_analysis['count'].sum()

    print(f"\nPositive buckets: {pos_buckets}/{total_buckets} ({bucket_ratio:.1%})")
    print(f"Weighted avg improvement: {weighted_improvement:.1%}")

    # ===== TEMPORAL STABILITY =====
    print("\n----- TEMPORAL STABILITY -----")
    signal['quarter'] = pd.qcut(signal['first_trade_time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    quarter_results = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_signal = signal[signal['quarter'] == q]
        if len(q_signal) < 10:
            continue

        q_wins = (q_signal['market_result'] == 'no').sum()
        q_n = len(q_signal)
        q_wr = q_wins / q_n
        q_avg_price = q_signal['no_price'].mean()
        q_be = q_avg_price / 100
        q_edge = q_wr - q_be

        quarter_results.append({
            'quarter': q,
            'n': q_n,
            'win_rate': q_wr,
            'avg_price': q_avg_price,
            'edge': q_edge
        })
        print(f"  {q}: N={q_n}, WR={q_wr:.1%}, Edge={q_edge:.1%}")

    pos_quarters = sum(1 for q in quarter_results if q['edge'] > 0)
    print(f"\nPositive edge quarters: {pos_quarters}/4")

    # ===== CONCENTRATION CHECK =====
    print("\n----- CONCENTRATION CHECK -----")

    # Calculate profit per market
    signal['profit'] = signal.apply(
        lambda r: (100 - r['no_price']) if r['market_result'] == 'no' else -r['no_price'],
        axis=1
    )

    total_profit = signal['profit'].sum()
    max_single_profit = signal['profit'].max()
    max_market = signal.loc[signal['profit'].idxmax(), 'market_ticker']

    print(f"Total profit (hypothetical): {total_profit:.0f}c")
    print(f"Max single market profit: {max_single_profit:.0f}c ({max_market})")
    print(f"Concentration: {max_single_profit / total_profit * 100:.1f}%")

    # Top 5 contributors
    top_5 = signal.nlargest(5, 'profit')[['market_ticker', 'profit', 'no_price', 'market_result']]
    print(f"\nTop 5 profit contributors:")
    print(top_5.to_string(index=False))

    top_5_profit = signal.nlargest(5, 'profit')['profit'].sum()
    top_5_concentration = top_5_profit / total_profit * 100

    print(f"\nTop 5 concentration: {top_5_concentration:.1f}%")

    # ===== CATEGORY BREAKDOWN =====
    print("\n----- CATEGORY BREAKDOWN -----")
    signal['category'] = signal['market_ticker'].str.extract(r'^(KX[A-Z]+)', expand=False)

    cat_analysis = signal.groupby('category').agg({
        'won': ['sum', 'count', 'mean'],
        'no_price': 'mean'
    }).reset_index()
    cat_analysis.columns = ['category', 'wins', 'count', 'win_rate', 'avg_price']
    cat_analysis['edge'] = cat_analysis['win_rate'] - cat_analysis['avg_price'] / 100
    cat_analysis = cat_analysis.sort_values('count', ascending=False)

    print(cat_analysis[cat_analysis['count'] >= 10].to_string(index=False))

    # ===== VALIDATION SUMMARY =====
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY: FOLLOW LATE NO")
    print("=" * 60)

    validation_passed = True
    issues = []

    if n < 50:
        validation_passed = False
        issues.append(f"Sample size too small: {n}")

    if p_value > 0.01:
        validation_passed = False
        issues.append(f"P-value not significant at 1%: {p_value:.4f}")

    if bucket_ratio < 0.6:
        validation_passed = False
        issues.append(f"Bucket ratio too low: {bucket_ratio:.1%}")

    if pos_quarters < 3:
        validation_passed = False
        issues.append(f"Temporal instability: only {pos_quarters}/4 positive quarters")

    if top_5_concentration > 30:
        validation_passed = False
        issues.append(f"High concentration: top 5 = {top_5_concentration:.1f}%")

    if validation_passed:
        print("\nVALIDATION: PASSED")
        print(f"  Sample size: {n} (OK)")
        print(f"  P-value: {p_value:.6f} (OK)")
        print(f"  Bucket ratio: {bucket_ratio:.1%} (OK)")
        print(f"  Temporal stability: {pos_quarters}/4 quarters (OK)")
        print(f"  Concentration: top 5 = {top_5_concentration:.1f}% (OK)")
        print(f"  Improvement vs baseline: {weighted_improvement:.1%}")
    else:
        print("\nVALIDATION: FAILED")
        for issue in issues:
            print(f"  - {issue}")

    return {
        'n': n,
        'win_rate': wr,
        'edge': edge,
        'p_value': p_value,
        'bucket_ratio': bucket_ratio,
        'weighted_improvement': weighted_improvement,
        'pos_quarters': pos_quarters,
        'top_5_concentration': top_5_concentration,
        'validation_passed': validation_passed,
        'issues': issues
    }


def validate_follow_yes(ll_df, baseline_by_bucket):
    """Deep validation of Follow Late YES signal."""
    print("\n" + "=" * 80)
    print("DEEP VALIDATION: FOLLOW LATE YES")
    print("=" * 80)

    signal = ll_df[ll_df['late_direction'] == 'yes'].copy()
    print(f"\nTotal signal markets: {len(signal)}")

    if len(signal) < 50:
        print("Insufficient sample")
        return None

    # ===== BASIC METRICS =====
    n = len(signal)
    wins = (signal['market_result'] == 'yes').sum()
    wr = wins / n
    avg_price = signal['yes_price'].mean()
    be = avg_price / 100
    edge = wr - be

    # P-value
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\nBasic Metrics:")
    print(f"  N = {n}")
    print(f"  Win Rate = {wr:.1%}")
    print(f"  Avg YES Price = {avg_price:.1f}c")
    print(f"  Breakeven = {be:.1%}")
    print(f"  Raw Edge = {edge:.1%}")
    print(f"  P-value = {p_value:.6f}")

    # ===== TEMPORAL STABILITY =====
    print("\n----- TEMPORAL STABILITY -----")
    signal['quarter'] = pd.qcut(signal['first_trade_time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    quarter_results = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_signal = signal[signal['quarter'] == q]
        if len(q_signal) < 10:
            continue

        q_wins = (q_signal['market_result'] == 'yes').sum()
        q_n = len(q_signal)
        q_wr = q_wins / q_n
        q_avg_price = q_signal['yes_price'].mean()
        q_be = q_avg_price / 100
        q_edge = q_wr - q_be

        quarter_results.append({
            'quarter': q,
            'n': q_n,
            'win_rate': q_wr,
            'avg_price': q_avg_price,
            'edge': q_edge
        })
        print(f"  {q}: N={q_n}, WR={q_wr:.1%}, Edge={q_edge:.1%}")

    pos_quarters = sum(1 for q in quarter_results if q['edge'] > 0)
    print(f"\nPositive edge quarters: {pos_quarters}/4")

    # ===== CONCENTRATION CHECK =====
    print("\n----- CONCENTRATION CHECK -----")

    signal['profit'] = signal.apply(
        lambda r: (100 - r['yes_price']) if r['market_result'] == 'yes' else -r['yes_price'],
        axis=1
    )

    total_profit = signal['profit'].sum()
    max_single_profit = signal['profit'].max()

    print(f"Total profit (hypothetical): {total_profit:.0f}c")
    print(f"Max single market profit: {max_single_profit:.0f}c")

    if total_profit > 0:
        print(f"Concentration: {max_single_profit / total_profit * 100:.1f}%")

    return {
        'n': n,
        'win_rate': wr,
        'edge': edge,
        'p_value': p_value,
        'pos_quarters': pos_quarters
    }


def main():
    df = load_data()
    all_markets, baseline_by_bucket = build_baseline()

    # Detect signal
    ll_df = detect_late_large_signal(df)
    print(f"\nTotal late-large signal markets: {len(ll_df)}")
    print(f"  Follow NO: {(ll_df['late_direction'] == 'no').sum()}")
    print(f"  Follow YES: {(ll_df['late_direction'] == 'yes').sum()}")

    # Validate
    no_result = validate_follow_no(ll_df, baseline_by_bucket)
    yes_result = validate_follow_yes(ll_df, baseline_by_bucket)

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT: SPORTS-007 Late-Arriving Large Money")
    print("=" * 80)

    if no_result and no_result['validation_passed']:
        print("\nFOLLOW LATE NO: VALIDATED")
        print(f"  Edge: {no_result['edge']:.1%}")
        print(f"  Improvement vs baseline: {no_result['weighted_improvement']:.1%}")
        print(f"  Bucket ratio: {no_result['bucket_ratio']:.1%}")
        print(f"  Temporal stability: {no_result['pos_quarters']}/4 quarters")
    else:
        print("\nFOLLOW LATE NO: NEEDS MORE INVESTIGATION")
        if no_result:
            for issue in no_result['issues']:
                print(f"  - {issue}")

    # Save results
    output = {
        'hypothesis': 'SPORTS-007',
        'name': 'Late-Arriving Large Money',
        'follow_no': no_result,
        'follow_yes': yes_result
    }

    output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/sports007_deep_validation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
