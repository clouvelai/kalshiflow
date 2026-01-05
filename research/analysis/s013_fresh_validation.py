#!/usr/bin/env python3
"""
S013 Fresh Validation - Independent, Skeptical Analysis

Mission: Validate whether S013 (Low Leverage Variance NO) is a real edge or just a price proxy.

Signal: leverage_std < 0.7 AND no_ratio > 0.5
Action: Bet NO

Validation Criteria (H123 Standard):
1. Unique Markets >= 50
2. P-value < 0.05
3. Bootstrap CI excludes zero
4. Temporal stability >= 3/4 quarters
5. Concentration: max single market < 30%
6. Bucket Ratio >= 80% (STRICT - this is what separates VALIDATED from WEAK_EDGE)

Verdict Framework:
- VALIDATED: >= 80% positive buckets AND all tests pass
- WEAK_EDGE: 60-79% positive buckets
- INVALIDATED: < 60% positive buckets OR fails critical tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/s013_fresh_validation.json'


def load_data():
    """Load the trade data"""
    print("=" * 80)
    print("S013 FRESH VALIDATION - INDEPENDENT SKEPTICAL ANALYSIS")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\nData loaded: {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    return df


def compute_market_level_stats(df):
    """Compute per-market statistics needed for S013 signal"""
    print("\n" + "=" * 80)
    print("STEP 1: COMPUTING MARKET-LEVEL STATISTICS")
    print("=" * 80)

    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['std', 'mean', 'count'],
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'datetime': ['min', 'max']
    }).reset_index()

    # Flatten column names
    market_stats.columns = [
        'market_ticker',
        'leverage_std', 'leverage_mean', 'n_trades',
        'no_ratio',
        'market_result',
        'avg_no_price',
        'first_trade', 'last_trade'
    ]

    # Handle NaN leverage_std (markets with only 1 trade)
    market_stats['leverage_std'] = market_stats['leverage_std'].fillna(0)

    print(f"Computed stats for {len(market_stats):,} markets")
    print(f"  - Mean leverage_std: {market_stats['leverage_std'].mean():.3f}")
    print(f"  - Mean no_ratio: {market_stats['no_ratio'].mean():.3f}")
    print(f"  - Mean avg_no_price: {market_stats['avg_no_price'].mean():.1f}c")

    return market_stats


def identify_s013_signals(market_stats):
    """Apply S013 signal filter"""
    print("\n" + "=" * 80)
    print("STEP 2: IDENTIFYING S013 SIGNAL MARKETS")
    print("=" * 80)

    # S013 Signal: leverage_std < 0.7 AND no_ratio > 0.5 AND n_trades >= 5
    signal_mask = (
        (market_stats['leverage_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    )

    signal_markets = market_stats[signal_mask].copy()

    print(f"\nS013 Signal Criteria:")
    print(f"  - leverage_std < 0.7")
    print(f"  - no_ratio > 0.5 (majority NO trades)")
    print(f"  - n_trades >= 5 (enough data to compute std)")
    print(f"\nSignal markets found: {len(signal_markets):,}")
    print(f"  ({len(signal_markets)/len(market_stats)*100:.2f}% of all markets)")

    return signal_markets, market_stats


def test_price_proxy_correlation(signal_markets, all_markets):
    """
    CRITICAL TEST: Is leverage_std correlated with NO price?

    If low leverage_std markets tend to have higher NO prices,
    S013 might just be selecting "easy NO bets" that would win anyway.
    """
    print("\n" + "=" * 80)
    print("STEP 3: PRICE PROXY CORRELATION TEST")
    print("=" * 80)

    # Test 1: Correlation between leverage_std and avg_no_price (all markets)
    valid_all = all_markets[all_markets['n_trades'] >= 5].copy()
    corr_all = valid_all['leverage_std'].corr(valid_all['avg_no_price'])

    print(f"\nCorrelation Analysis (markets with >= 5 trades):")
    print(f"  - All markets (N={len(valid_all):,}): corr(leverage_std, no_price) = {corr_all:.4f}")

    # Test 2: Compare price distribution of signal vs non-signal markets
    non_signal = valid_all[~valid_all['market_ticker'].isin(signal_markets['market_ticker'])]

    signal_mean_price = signal_markets['avg_no_price'].mean()
    non_signal_mean_price = non_signal['avg_no_price'].mean()
    all_mean_price = valid_all['avg_no_price'].mean()

    print(f"\nPrice Distribution Comparison:")
    print(f"  - Signal markets avg NO price: {signal_mean_price:.1f}c")
    print(f"  - Non-signal markets avg NO price: {non_signal_mean_price:.1f}c")
    print(f"  - All markets avg NO price: {all_mean_price:.1f}c")
    print(f"  - Difference (signal vs all): {signal_mean_price - all_mean_price:+.1f}c")

    # Test 3: Statistical test for price difference
    t_stat, p_value = stats.ttest_ind(
        signal_markets['avg_no_price'],
        non_signal['avg_no_price']
    )

    print(f"\nT-test for price difference:")
    print(f"  - t-statistic: {t_stat:.3f}")
    print(f"  - p-value: {p_value:.6f}")

    price_proxy_warning = False
    if abs(signal_mean_price - all_mean_price) > 10:
        print(f"\n  WARNING: Signal markets have significantly different price distribution!")
        print(f"  This suggests S013 may be partially a price proxy.")
        price_proxy_warning = True
    else:
        print(f"\n  OK: Price distributions are similar (diff < 10c)")

    return {
        'correlation': corr_all,
        'signal_mean_price': signal_mean_price,
        'all_mean_price': all_mean_price,
        'price_difference': signal_mean_price - all_mean_price,
        'ttest_pvalue': p_value,
        'price_proxy_warning': price_proxy_warning
    }


def build_baseline_by_bucket(all_markets, bucket_size=5):
    """Build baseline NO win rates by price bucket"""
    print("\n" + "=" * 80)
    print(f"STEP 4: BUILDING {bucket_size}c BUCKET BASELINE")
    print("=" * 80)

    all_markets = all_markets.copy()
    all_markets['bucket'] = (all_markets['avg_no_price'] // bucket_size) * bucket_size

    baseline = {}
    print(f"\n{'Bucket':<12} {'Markets':>10} {'NO Wins':>10} {'Win Rate':>10}")
    print("-" * 45)

    for bucket in sorted(all_markets['bucket'].unique()):
        bucket_data = all_markets[all_markets['bucket'] == bucket]
        n_markets = len(bucket_data)
        no_wins = (bucket_data['market_result'] == 'no').sum()
        win_rate = no_wins / n_markets if n_markets > 0 else 0

        if n_markets >= 20:  # Minimum for reliable baseline
            baseline[bucket] = {
                'n_markets': n_markets,
                'no_wins': no_wins,
                'win_rate': win_rate
            }
            print(f"{bucket:.0f}-{bucket+bucket_size:.0f}c      {n_markets:>10,} {no_wins:>10,} {win_rate:>10.1%}")

    print(f"\nBaseline built for {len(baseline)} buckets")

    return baseline


def bucket_matched_analysis(signal_markets, baseline, bucket_size=5):
    """
    CRITICAL: Compare S013 win rate to baseline at SAME PRICES

    This is THE test that separates real edge from price proxies.
    """
    print("\n" + "=" * 80)
    print("STEP 5: BUCKET-MATCHED IMPROVEMENT ANALYSIS")
    print("=" * 80)

    signal_markets = signal_markets.copy()
    signal_markets['bucket'] = (signal_markets['avg_no_price'] // bucket_size) * bucket_size

    results = []

    print(f"\n{'Bucket':<12} {'Sig N':>8} {'Sig WR':>10} {'Base WR':>10} {'Improve':>10} {'Status':>10}")
    print("-" * 65)

    for bucket in sorted(signal_markets['bucket'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 3:  # Need minimum sample
            continue

        sig_wins = (sig_bucket['market_result'] == 'no').sum()
        sig_wr = sig_wins / n_sig
        base_wr = baseline[bucket]['win_rate']
        improvement = sig_wr - base_wr

        status = "+" if improvement > 0 else "-"

        results.append({
            'bucket': bucket,
            'n_signal': n_sig,
            'signal_wins': sig_wins,
            'signal_wr': sig_wr,
            'baseline_wr': base_wr,
            'improvement': improvement
        })

        print(f"{bucket:.0f}-{bucket+bucket_size:.0f}c      {n_sig:>8} {sig_wr:>10.1%} {base_wr:>10.1%} {improvement*100:>+10.2f}% {status:>10}")

    if not results:
        return None, 0, 0

    # Calculate summary statistics
    positive_buckets = sum(1 for r in results if r['improvement'] > 0)
    negative_buckets = sum(1 for r in results if r['improvement'] < 0)
    total_buckets = len(results)
    bucket_ratio = positive_buckets / total_buckets if total_buckets > 0 else 0

    # Weighted improvement (by signal market count)
    total_n = sum(r['n_signal'] for r in results)
    weighted_improvement = sum(r['improvement'] * r['n_signal'] for r in results) / total_n

    print(f"\n" + "=" * 65)
    print(f"BUCKET ANALYSIS SUMMARY:")
    print(f"  - Positive buckets: {positive_buckets}/{total_buckets} ({bucket_ratio*100:.1f}%)")
    print(f"  - Negative buckets: {negative_buckets}/{total_buckets}")
    print(f"  - Weighted improvement: {weighted_improvement*100:+.2f}%")

    # Verdict on bucket ratio
    if bucket_ratio >= 0.80:
        print(f"\n  BUCKET TEST: PASS (>= 80% threshold)")
    elif bucket_ratio >= 0.60:
        print(f"\n  BUCKET TEST: WEAK_EDGE (60-79% range)")
    else:
        print(f"\n  BUCKET TEST: FAIL (< 60% - likely price proxy)")

    return results, bucket_ratio, weighted_improvement


def calculate_raw_edge(signal_markets):
    """Calculate raw edge metrics"""
    print("\n" + "=" * 80)
    print("STEP 6: RAW EDGE CALCULATION")
    print("=" * 80)

    n_markets = len(signal_markets)
    no_wins = (signal_markets['market_result'] == 'no').sum()
    win_rate = no_wins / n_markets

    # Breakeven = avg NO price / 100
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100

    raw_edge = win_rate - breakeven

    print(f"\nRaw Statistics:")
    print(f"  - Total markets: {n_markets:,}")
    print(f"  - NO wins: {no_wins:,}")
    print(f"  - Win rate: {win_rate:.1%}")
    print(f"  - Avg NO price: {avg_no_price:.1f}c")
    print(f"  - Breakeven: {breakeven:.1%}")
    print(f"  - Raw edge: {raw_edge*100:+.2f}%")

    return {
        'n_markets': n_markets,
        'no_wins': no_wins,
        'win_rate': win_rate,
        'avg_no_price': avg_no_price,
        'breakeven': breakeven,
        'raw_edge': raw_edge
    }


def bootstrap_confidence_interval(signal_markets, n_bootstrap=1000):
    """Calculate bootstrap 95% CI for edge"""
    print("\n" + "=" * 80)
    print("STEP 7: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 80)

    edges = []
    n = len(signal_markets)

    for _ in range(n_bootstrap):
        sample = signal_markets.sample(n, replace=True)
        wr = (sample['market_result'] == 'no').mean()
        be = sample['avg_no_price'].mean() / 100
        edge = wr - be
        edges.append(edge)

    edges = np.array(edges)
    ci_low = np.percentile(edges, 2.5)
    ci_high = np.percentile(edges, 97.5)
    mean_edge = edges.mean()
    std_edge = edges.std()

    print(f"\nBootstrap Results ({n_bootstrap} samples):")
    print(f"  - Mean edge: {mean_edge*100:.2f}%")
    print(f"  - Std edge: {std_edge*100:.2f}%")
    print(f"  - 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

    ci_excludes_zero = ci_low > 0
    print(f"\n  CI excludes zero: {'YES (significant)' if ci_excludes_zero else 'NO (not significant)'}")

    return {
        'mean_edge': mean_edge,
        'std_edge': std_edge,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_excludes_zero': ci_excludes_zero
    }


def binomial_test(signal_markets):
    """Statistical significance via binomial test"""
    print("\n" + "=" * 80)
    print("STEP 8: BINOMIAL TEST FOR SIGNIFICANCE")
    print("=" * 80)

    n_markets = len(signal_markets)
    no_wins = (signal_markets['market_result'] == 'no').sum()
    breakeven = signal_markets['avg_no_price'].mean() / 100

    result = stats.binomtest(no_wins, n_markets, breakeven, alternative='greater')
    p_value = result.pvalue

    print(f"\nBinomial Test (H0: win_rate <= breakeven):")
    print(f"  - Successes: {no_wins}")
    print(f"  - Trials: {n_markets}")
    print(f"  - Expected rate: {breakeven:.1%}")
    print(f"  - Observed rate: {no_wins/n_markets:.1%}")
    print(f"  - P-value: {p_value:.2e}")

    significant = p_value < 0.05
    print(f"\n  Statistically significant (p < 0.05): {'YES' if significant else 'NO'}")

    return {
        'p_value': p_value,
        'significant': significant
    }


def temporal_stability_check(signal_markets):
    """Check if edge is stable across time periods"""
    print("\n" + "=" * 80)
    print("STEP 9: TEMPORAL STABILITY CHECK")
    print("=" * 80)

    signal_markets = signal_markets.sort_values('first_trade').copy()
    n = len(signal_markets)
    q_size = n // 4

    quarters = []

    print(f"\n{'Quarter':<10} {'Markets':>10} {'Win Rate':>10} {'Breakeven':>10} {'Edge':>10} {'Status':>8}")
    print("-" * 65)

    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else n
        q = signal_markets.iloc[start:end]

        q_n = len(q)
        q_wins = (q['market_result'] == 'no').sum()
        q_wr = q_wins / q_n
        q_be = q['avg_no_price'].mean() / 100
        q_edge = q_wr - q_be

        status = "+" if q_edge > 0 else "-"

        quarters.append({
            'quarter': i + 1,
            'n': q_n,
            'win_rate': q_wr,
            'breakeven': q_be,
            'edge': q_edge,
            'start': str(q['first_trade'].min()),
            'end': str(q['last_trade'].max())
        })

        print(f"Q{i+1}        {q_n:>10} {q_wr:>10.1%} {q_be:>10.1%} {q_edge*100:>+10.2f}% {status:>8}")

    positive_quarters = sum(1 for q in quarters if q['edge'] > 0)

    print(f"\nPositive quarters: {positive_quarters}/4")
    passes = positive_quarters >= 3
    print(f"Temporal stability: {'PASS (>= 3/4)' if passes else 'FAIL (< 3/4)'}")

    return {
        'quarters': quarters,
        'positive_quarters': positive_quarters,
        'passes': passes
    }


def concentration_check(signal_markets):
    """Check if profits are concentrated in few markets"""
    print("\n" + "=" * 80)
    print("STEP 10: CONCENTRATION CHECK")
    print("=" * 80)

    signal_markets = signal_markets.copy()

    # Calculate profit per market (betting NO)
    signal_markets['profit'] = signal_markets.apply(
        lambda x: (100 - x['avg_no_price']) if x['market_result'] == 'no' else -x['avg_no_price'],
        axis=1
    )

    total_profit = signal_markets['profit'].sum()

    if total_profit <= 0:
        print(f"\n  Total profit is <= 0 (${total_profit:.2f})")
        print(f"  Cannot calculate concentration - strategy loses money!")
        return {
            'total_profit': total_profit,
            'max_single_pct': 0,
            'passes': False,
            'warning': 'negative_profit'
        }

    # Top contributors
    top_markets = signal_markets.nlargest(10, 'profit')
    max_single = signal_markets['profit'].max()
    max_single_pct = max_single / total_profit
    top_10_pct = top_markets['profit'].sum() / total_profit

    print(f"\nConcentration Analysis:")
    print(f"  - Total simulated profit: ${total_profit:.2f}")
    print(f"  - Avg profit per market: ${total_profit/len(signal_markets):.2f}")
    print(f"  - Max single market: ${max_single:.2f} ({max_single_pct*100:.1f}%)")
    print(f"  - Top 10 markets: {top_10_pct*100:.1f}%")

    passes = max_single_pct < 0.30
    print(f"\n  Concentration check: {'PASS (< 30%)' if passes else 'FAIL (>= 30%)'}")

    return {
        'total_profit': total_profit,
        'avg_profit': total_profit / len(signal_markets),
        'max_single_profit': max_single,
        'max_single_pct': max_single_pct,
        'top_10_pct': top_10_pct,
        'passes': passes
    }


def final_verdict(results):
    """Determine final verdict based on all tests"""
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Collect all test results
    checks = [
        ("Unique Markets >= 50", results['raw_edge']['n_markets'] >= 50),
        ("P-value < 0.05", results['binomial']['significant']),
        ("Bootstrap CI excludes zero", results['bootstrap']['ci_excludes_zero']),
        ("Temporal Stability >= 3/4", results['temporal']['passes']),
        ("Concentration < 30%", results['concentration']['passes']),
        ("Bucket Ratio >= 80%", results['bucket_ratio'] >= 0.80),
    ]

    print("\n" + "-" * 50)
    print("VALIDATION CHECKLIST:")
    print("-" * 50)

    all_pass = True
    critical_failures = []

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False
            critical_failures.append(name)

    # Determine verdict
    bucket_ratio = results['bucket_ratio']

    if all_pass:
        verdict = "VALIDATED"
        verdict_reason = "All tests passed including 80% bucket threshold"
    elif bucket_ratio >= 0.60 and len(critical_failures) <= 1:
        verdict = "WEAK_EDGE"
        verdict_reason = f"Bucket ratio {bucket_ratio*100:.0f}% < 80% threshold (partial price proxy)"
    else:
        verdict = "INVALIDATED"
        if bucket_ratio < 0.60:
            verdict_reason = f"Bucket ratio {bucket_ratio*100:.0f}% < 60% (clear price proxy)"
        else:
            verdict_reason = f"Failed critical tests: {', '.join(critical_failures)}"

    print("\n" + "=" * 50)
    print(f"VERDICT: {verdict}")
    print(f"REASON: {verdict_reason}")
    print("=" * 50)

    # Summary stats
    print(f"\nKey Metrics:")
    print(f"  - Markets: {results['raw_edge']['n_markets']}")
    print(f"  - Raw Edge: {results['raw_edge']['raw_edge']*100:+.2f}%")
    print(f"  - Weighted Improvement: {results['weighted_improvement']*100:+.2f}%")
    print(f"  - Bucket Ratio: {bucket_ratio*100:.1f}% ({results['positive_buckets']}/{results['total_buckets']})")
    print(f"  - 95% CI: [{results['bootstrap']['ci_low']*100:.2f}%, {results['bootstrap']['ci_high']*100:.2f}%]")
    print(f"  - P-value: {results['binomial']['p_value']:.2e}")

    return {
        'verdict': verdict,
        'reason': verdict_reason,
        'checks': {name: passed for name, passed in checks},
        'critical_failures': critical_failures
    }


def main():
    # Load data
    df = load_data()

    # Compute market-level stats
    market_stats = compute_market_level_stats(df)

    # Identify S013 signals
    signal_markets, all_markets = identify_s013_signals(market_stats)

    if len(signal_markets) < 50:
        print(f"\nINSUFFICIENT SIGNAL MARKETS: {len(signal_markets)} < 50")
        print("VERDICT: INVALIDATED (insufficient sample size)")
        return

    # Run all validation tests
    results = {}

    # Price proxy correlation test
    results['price_proxy'] = test_price_proxy_correlation(signal_markets, all_markets)

    # Build baseline
    baseline = build_baseline_by_bucket(all_markets)

    # Bucket-matched analysis (CRITICAL)
    bucket_results, bucket_ratio, weighted_improvement = bucket_matched_analysis(signal_markets, baseline)
    results['bucket_results'] = bucket_results
    results['bucket_ratio'] = bucket_ratio
    results['weighted_improvement'] = weighted_improvement
    results['positive_buckets'] = sum(1 for r in bucket_results if r['improvement'] > 0) if bucket_results else 0
    results['total_buckets'] = len(bucket_results) if bucket_results else 0

    # Raw edge calculation
    results['raw_edge'] = calculate_raw_edge(signal_markets)

    # Bootstrap CI
    results['bootstrap'] = bootstrap_confidence_interval(signal_markets)

    # Binomial test
    results['binomial'] = binomial_test(signal_markets)

    # Temporal stability
    results['temporal'] = temporal_stability_check(signal_markets)

    # Concentration check
    results['concentration'] = concentration_check(signal_markets)

    # Final verdict
    results['final'] = final_verdict(results)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'verdict': results['final']['verdict'],
        'reason': results['final']['reason'],
        'summary': {
            'n_markets': results['raw_edge']['n_markets'],
            'raw_edge': results['raw_edge']['raw_edge'],
            'weighted_improvement': results['weighted_improvement'],
            'bucket_ratio': results['bucket_ratio'],
            'positive_buckets': results['positive_buckets'],
            'total_buckets': results['total_buckets'],
            'ci_low': results['bootstrap']['ci_low'],
            'ci_high': results['bootstrap']['ci_high'],
            'p_value': results['binomial']['p_value'],
            'temporal_positive': results['temporal']['positive_quarters'],
            'concentration_max': results['concentration']['max_single_pct']
        },
        'checks': results['final']['checks'],
        'price_proxy_analysis': results['price_proxy'],
        'bucket_details': bucket_results,
        'temporal_details': results['temporal']['quarters'],
        'concentration_details': {
            'total_profit': results['concentration']['total_profit'],
            'max_single_pct': results['concentration']['max_single_pct'],
            'top_10_pct': results['concentration'].get('top_10_pct', 0)
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
