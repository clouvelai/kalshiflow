"""
LSD SESSION 002: PRODUCTION VALIDATION
=======================================

Apply RIGOROUS production validation to the 3 INDEPENDENT strategies from LSD Session 002,
using the SAME methodology as H123 RLM validation.

Strategies:
1. H-LSD-207: Dollar-Weighted Direction (+12.05% raw edge)
2. H-LSD-211: Conviction Ratio NO (+11.75% raw edge)
3. H-LSD-209: Size Gradient (+10.21% raw edge)

Validation Criteria (H123 Standards):
1. Statistical Significance: p < 0.001/3 (Bonferroni corrected)
2. Not Price Proxy: >80% positive buckets
3. CI Excludes Zero: 95% bootstrap CI lower > 0
4. Temporal Stability: 3/4 quarters positive
5. Out-of-Sample: Test set improvement > 0
6. Sufficient Markets: N >= 100
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/lsd_session_002_production_validation.json'

# Validation thresholds (matching H123)
THRESHOLDS = {
    'p_value_base': 0.001,
    'n_tests': 3,  # For Bonferroni correction
    'bucket_ratio_strict': 0.80,  # H123 used 80%
    'bucket_ratio_weak': 0.60,    # Fallback for weak edge
    'min_quarters_positive': 3,
    'min_markets': 100,
    'min_bucket_n': 5,
    'bucket_size': 5
}

# Results container
results = {
    'metadata': {
        'session': 'LSD_002_PRODUCTION_VALIDATION',
        'timestamp': datetime.now().isoformat(),
        'validation_standard': 'H123_PRODUCTION',
        'thresholds': THRESHOLDS
    },
    'strategies': {},
    'summary': {
        'validated': [],
        'weak_edge': [],
        'invalidated': []
    }
}


def load_data():
    """Load and prepare trade data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000
    df['is_no'] = df['taker_side'] == 'no'
    df['is_yes'] = df['taker_side'] == 'yes'
    df['date'] = df['datetime'].dt.date

    # Filter to resolved markets
    resolved_markets = df[df['market_result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df = df[df['market_ticker'].isin(resolved_markets)]

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} resolved markets")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    return df


def build_baseline(df, bucket_size=5):
    """Build baseline NO win rates at each price bucket."""
    print("\nBuilding baseline...")

    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'min'
    }).reset_index()

    all_markets['bucket'] = (all_markets['no_price'] // bucket_size) * bucket_size

    baseline = {}
    for bucket in sorted(all_markets['bucket'].unique()):
        bucket_markets = all_markets[all_markets['bucket'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n': n
            }

    print(f"Built baseline across {len(baseline)} price buckets (min 20 markets each)")
    return all_markets, baseline


def detect_h207_signal(df):
    """
    H-LSD-207: Dollar-Weighted Direction

    Signal: Trades favor YES by >20% more than dollars (smart money going to NO)
    """
    market_stats = df.groupby('market_ticker').agg({
        'is_yes': ['sum', 'count'],
        'trade_value_cents': 'sum',
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'min'
    })
    market_stats.columns = ['yes_trades', 'total_trades', 'total_value', 'market_result', 'no_price', 'yes_price', 'datetime']
    market_stats = market_stats.reset_index()

    market_stats['yes_trade_ratio'] = market_stats['yes_trades'] / market_stats['total_trades']

    # Calculate YES dollar ratio
    yes_value = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].sum()
    market_stats['yes_value'] = market_stats['market_ticker'].map(yes_value).fillna(0)
    market_stats['yes_dollar_ratio'] = market_stats['yes_value'] / market_stats['total_value']

    # Divergence: trades favor YES more than dollars
    market_stats['divergence'] = market_stats['yes_trade_ratio'] - market_stats['yes_dollar_ratio']

    # Signal: >20% divergence (trades YES-heavy, but dollars NO-heavy)
    signal = market_stats[
        (market_stats['divergence'] > 0.20) &
        (market_stats['total_trades'] >= 5)
    ].copy()

    return signal


def detect_h211_signal(df):
    """
    H-LSD-211: Conviction Ratio NO

    Signal: Average NO trade size > 2x average YES trade size
    """
    yes_sizes = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].mean()
    no_sizes = df[df['is_no']].groupby('market_ticker')['trade_value_cents'].mean()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': ['count', 'min']
    })
    market_info.columns = ['market_result', 'no_price', 'yes_price', 'n_trades', 'datetime']
    market_info = market_info.reset_index()

    market_info['yes_avg_size'] = market_info['market_ticker'].map(yes_sizes)
    market_info['no_avg_size'] = market_info['market_ticker'].map(no_sizes)
    market_info = market_info.dropna(subset=['yes_avg_size', 'no_avg_size'])

    market_info['size_ratio'] = market_info['no_avg_size'] / (market_info['yes_avg_size'] + 0.01)

    # Signal: NO trades are 2x bigger than YES trades
    signal = market_info[
        (market_info['size_ratio'] > 2) &
        (market_info['n_trades'] >= 5)
    ].copy()

    return signal


def detect_h209_signal(df):
    """
    H-LSD-209: Size Gradient

    Signal: Correlation between trade size and NO direction > 0.3
    """
    def calc_size_direction_corr(group):
        if len(group) < 5:
            return pd.Series({'corr': np.nan, 'n_trades': len(group)})
        corr = group['trade_value_cents'].corr(group['is_no'].astype(float))
        return pd.Series({'corr': corr, 'n_trades': len(group)})

    market_corr = df.groupby('market_ticker').apply(calc_size_direction_corr).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'min'
    }).reset_index()

    combined = market_info.merge(market_corr, on='market_ticker')
    combined = combined.dropna(subset=['corr'])

    # Signal: larger trades correlate with NO direction
    signal = combined[
        (combined['corr'] > 0.3) &
        (combined['n_trades'] >= 5)
    ].copy()

    return signal


def calculate_comprehensive_stats(signal_markets, baseline, name, bucket_size=5):
    """
    Calculate comprehensive edge statistics with bucket-matched baseline.
    Returns all metrics needed for validation decision.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING: {name}")
    print(f"{'='*60}")

    n = len(signal_markets)
    if n < THRESHOLDS['min_markets']:
        return {
            'valid': False,
            'reason': f'insufficient_markets ({n} < {THRESHOLDS["min_markets"]})',
            'n': n
        }

    # Ensure bucket column exists
    signal_markets = signal_markets.copy()
    signal_markets['bucket'] = (signal_markets['no_price'] // bucket_size) * bucket_size

    # Overall statistics
    wins = (signal_markets['market_result'] == 'no').sum()
    win_rate = wins / n
    avg_no_price = signal_markets['no_price'].mean()
    breakeven = avg_no_price / 100
    raw_edge = win_rate - breakeven

    print(f"\nOverall Stats:")
    print(f"  Markets: {n:,}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Raw Edge: {raw_edge:+.2%}")

    # P-value (one-sided z-test)
    if 0 < breakeven < 1:
        z = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven))
        p_value = 1 - stats.norm.cdf(z)
    else:
        p_value = 1.0

    # Bonferroni correction
    p_bonferroni = p_value * THRESHOLDS['n_tests']
    p_bonferroni = min(p_bonferroni, 1.0)

    print(f"\nStatistical Significance:")
    print(f"  Z-score: {z:.2f}")
    print(f"  P-value (raw): {p_value:.2e}")
    print(f"  P-value (Bonferroni): {p_bonferroni:.2e}")
    print(f"  Threshold: {THRESHOLDS['p_value_base'] / THRESHOLDS['n_tests']:.2e}")

    # Bucket-by-bucket analysis
    print(f"\nBucket Analysis ({bucket_size}c increments):")
    bucket_results = []

    for bucket in sorted(signal_markets['bucket'].unique()):
        bucket_signal = signal_markets[signal_markets['bucket'] == bucket]
        n_signal = len(bucket_signal)

        if n_signal < THRESHOLDS['min_bucket_n']:
            continue

        signal_wr = (bucket_signal['market_result'] == 'no').mean()

        if bucket in baseline:
            baseline_wr = baseline[bucket]['win_rate']
            improvement = signal_wr - baseline_wr

            bucket_results.append({
                'bucket': int(bucket),
                'n': n_signal,
                'signal_wr': float(signal_wr),
                'baseline_wr': float(baseline_wr),
                'improvement': float(improvement)
            })

            marker = '+' if improvement > 0 else '-'
            print(f"  {bucket:3.0f}-{bucket+bucket_size:3.0f}c: Sig={signal_wr:.1%} vs Base={baseline_wr:.1%} -> {improvement:+.1%} [{marker}] (N={n_signal})")

    if not bucket_results:
        return {
            'valid': False,
            'reason': 'no_valid_buckets',
            'n': n
        }

    # Calculate weighted improvement and bucket ratio
    total_n = sum(b['n'] for b in bucket_results)
    weighted_improvement = sum(b['improvement'] * b['n'] / total_n for b in bucket_results)

    positive_buckets = sum(1 for b in bucket_results if b['improvement'] > 0)
    total_buckets = len(bucket_results)
    bucket_ratio = positive_buckets / total_buckets if total_buckets > 0 else 0

    print(f"\n  Summary:")
    print(f"    Positive Buckets: {positive_buckets}/{total_buckets} ({bucket_ratio:.0%})")
    print(f"    Weighted Improvement: {weighted_improvement:+.2%}")

    # Bootstrap 95% CI
    print(f"\nBootstrap Analysis (1000 samples):")
    bootstrap_edges = []
    for _ in range(1000):
        sample = signal_markets.sample(n=len(signal_markets), replace=True)
        sample_wr = (sample['market_result'] == 'no').mean()
        sample_price = sample['no_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_price)

    ci_low = np.percentile(bootstrap_edges, 2.5)
    ci_high = np.percentile(bootstrap_edges, 97.5)
    ci_excludes_zero = ci_low > 0

    print(f"  95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
    print(f"  CI Excludes Zero: {'YES' if ci_excludes_zero else 'NO'}")

    # Temporal stability (4 quarters)
    print(f"\nTemporal Stability (4 Quarters):")
    signal_sorted = signal_markets.sort_values('datetime')
    quarters = np.array_split(signal_sorted, 4)
    quarter_results = []

    for i, q in enumerate(quarters):
        if len(q) < 10:
            continue
        q_wr = (q['market_result'] == 'no').mean()
        q_price = q['no_price'].mean()
        q_edge = q_wr - q_price / 100
        quarter_results.append({
            'quarter': i + 1,
            'n': len(q),
            'edge': float(q_edge)
        })
        marker = '+' if q_edge > 0 else '-'
        print(f"  Q{i+1}: N={len(q):4d}, Edge={q_edge:+.2%} [{marker}]")

    positive_quarters = sum(1 for q in quarter_results if q['edge'] > 0)
    print(f"  Positive Quarters: {positive_quarters}/4")

    # Out-of-sample validation (80/20 split)
    print(f"\nOut-of-Sample Validation (80/20 split):")
    split_idx = int(len(signal_sorted) * 0.8)
    train = signal_sorted.iloc[:split_idx]
    test = signal_sorted.iloc[split_idx:]

    train_wr = (train['market_result'] == 'no').mean()
    train_price = train['no_price'].mean() / 100
    train_edge = train_wr - train_price

    test_wr = (test['market_result'] == 'no').mean()
    test_price = test['no_price'].mean() / 100
    test_edge = test_wr - test_price

    # Calculate test improvement vs baseline at same prices
    test['bucket'] = (test['no_price'] // bucket_size) * bucket_size
    test_improvements = []
    for bucket in test['bucket'].unique():
        if bucket not in baseline:
            continue
        bucket_test = test[test['bucket'] == bucket]
        if len(bucket_test) < 3:
            continue
        test_bucket_wr = (bucket_test['market_result'] == 'no').mean()
        base_wr = baseline[bucket]['win_rate']
        test_improvements.append({
            'bucket': bucket,
            'n': len(bucket_test),
            'improvement': test_bucket_wr - base_wr
        })

    if test_improvements:
        test_total_n = sum(t['n'] for t in test_improvements)
        test_improvement = sum(t['improvement'] * t['n'] / test_total_n for t in test_improvements)
    else:
        test_improvement = test_edge  # Fallback

    print(f"  TRAIN: N={len(train)}, Edge={train_edge:+.2%}")
    print(f"  TEST:  N={len(test)}, Edge={test_edge:+.2%}")
    print(f"  TEST Improvement vs Baseline: {test_improvement:+.2%}")

    gap = train_edge - test_edge
    print(f"  Generalization Gap: {gap:.2%} {'(acceptable)' if abs(gap) < 0.05 else '(WARNING)'}")

    # Category breakdown
    print(f"\nCategory Breakdown:")

    # Infer category from ticker
    def get_category(ticker):
        if any(x in ticker for x in ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF', 'NCAAMB', 'EPL', 'MVE', 'SOCCER']):
            return 'Sports'
        elif any(x in ticker for x in ['BTC', 'ETH', 'DOGE', 'XRP', 'SOL']):
            return 'Crypto'
        elif any(x in ticker for x in ['MENTION', 'MRBEAST', 'COLBERT', 'SNL']):
            return 'Media_Mentions'
        elif any(x in ticker for x in ['NETFLIX', 'SPOTIFY', 'BILLBOARD', 'RANK']):
            return 'Entertainment'
        elif any(x in ticker for x in ['HIGH', 'RAIN', 'SNOW', 'TEMP']):
            return 'Weather'
        elif any(x in ticker for x in ['TRUMP', 'PRES', 'APR', 'ELECT']):
            return 'Politics'
        else:
            return 'Other'

    signal_markets['category'] = signal_markets['market_ticker'].apply(get_category)
    category_results = {}

    for cat in signal_markets['category'].unique():
        cat_markets = signal_markets[signal_markets['category'] == cat]
        if len(cat_markets) < 20:
            continue
        cat_wr = (cat_markets['market_result'] == 'no').mean()
        cat_price = cat_markets['no_price'].mean() / 100
        cat_edge = cat_wr - cat_price
        category_results[cat] = {
            'n': len(cat_markets),
            'edge': float(cat_edge),
            'win_rate': float(cat_wr)
        }
        print(f"  {cat}: N={len(cat_markets)}, Edge={cat_edge:+.2%}")

    # Compile all results
    return {
        'valid': True,
        'n': n,
        'wins': int(wins),
        'win_rate': float(win_rate),
        'avg_no_price': float(avg_no_price),
        'breakeven': float(breakeven),
        'raw_edge': float(raw_edge),
        'p_value': float(p_value),
        'p_bonferroni': float(p_bonferroni),
        'z_score': float(z),
        'bucket_analysis': {
            'positive_buckets': positive_buckets,
            'total_buckets': total_buckets,
            'bucket_ratio': float(bucket_ratio),
            'weighted_improvement': float(weighted_improvement),
            'bucket_details': bucket_results
        },
        'bootstrap_ci': {
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'ci_excludes_zero': ci_excludes_zero
        },
        'temporal_stability': {
            'quarters': quarter_results,
            'positive_quarters': positive_quarters
        },
        'out_of_sample': {
            'train_n': len(train),
            'train_edge': float(train_edge),
            'test_n': len(test),
            'test_edge': float(test_edge),
            'test_improvement': float(test_improvement),
            'gap': float(gap)
        },
        'category_breakdown': category_results
    }


def determine_verdict(stats, name):
    """
    Determine validation verdict based on H123 criteria.

    VALIDATED: All 6 criteria pass
    WEAK_EDGE: 4-5 criteria pass
    INVALIDATED: <4 criteria pass
    """
    if not stats.get('valid', False):
        return 'INVALIDATED', stats.get('reason', 'unknown'), 0

    criteria = {}

    # 1. Statistical significance (Bonferroni corrected)
    p_threshold = THRESHOLDS['p_value_base'] / THRESHOLDS['n_tests']
    criteria['p_significant'] = stats['p_bonferroni'] < p_threshold

    # 2. Not price proxy (>80% positive buckets)
    criteria['not_price_proxy_strict'] = stats['bucket_analysis']['bucket_ratio'] >= THRESHOLDS['bucket_ratio_strict']
    criteria['not_price_proxy_weak'] = stats['bucket_analysis']['bucket_ratio'] >= THRESHOLDS['bucket_ratio_weak']

    # 3. CI excludes zero
    criteria['ci_excludes_zero'] = stats['bootstrap_ci']['ci_excludes_zero']

    # 4. Temporal stability (3/4 quarters positive)
    criteria['temporal_stable'] = stats['temporal_stability']['positive_quarters'] >= THRESHOLDS['min_quarters_positive']

    # 5. Out-of-sample validation (test improvement > 0)
    criteria['out_of_sample'] = stats['out_of_sample']['test_improvement'] > 0

    # 6. Sufficient markets
    criteria['sufficient_markets'] = stats['n'] >= THRESHOLDS['min_markets']

    # Count criteria met
    strict_criteria = [
        criteria['p_significant'],
        criteria['not_price_proxy_strict'],
        criteria['ci_excludes_zero'],
        criteria['temporal_stable'],
        criteria['out_of_sample'],
        criteria['sufficient_markets']
    ]

    weak_criteria = [
        criteria['p_significant'],
        criteria['not_price_proxy_weak'],
        criteria['ci_excludes_zero'],
        criteria['temporal_stable'],
        criteria['out_of_sample'],
        criteria['sufficient_markets']
    ]

    strict_count = sum(strict_criteria)
    weak_count = sum(weak_criteria)

    print(f"\n{'='*60}")
    print(f"VERDICT: {name}")
    print(f"{'='*60}")
    print(f"\nCriteria Check:")
    print(f"  [{'PASS' if criteria['p_significant'] else 'FAIL'}] P-value (Bonferroni): {stats['p_bonferroni']:.2e} < {p_threshold:.2e}")
    print(f"  [{'PASS' if criteria['not_price_proxy_strict'] else 'FAIL'}] Bucket Ratio (strict): {stats['bucket_analysis']['bucket_ratio']:.0%} >= 80%")
    print(f"  [{'PASS' if criteria['not_price_proxy_weak'] else 'FAIL'}] Bucket Ratio (weak): {stats['bucket_analysis']['bucket_ratio']:.0%} >= 60%")
    print(f"  [{'PASS' if criteria['ci_excludes_zero'] else 'FAIL'}] CI Excludes Zero: [{stats['bootstrap_ci']['ci_low']:.2%}, {stats['bootstrap_ci']['ci_high']:.2%}]")
    print(f"  [{'PASS' if criteria['temporal_stable'] else 'FAIL'}] Temporal Stability: {stats['temporal_stability']['positive_quarters']}/4 >= 3/4")
    print(f"  [{'PASS' if criteria['out_of_sample'] else 'FAIL'}] Out-of-Sample: {stats['out_of_sample']['test_improvement']:+.2%} > 0")
    print(f"  [{'PASS' if criteria['sufficient_markets'] else 'FAIL'}] Markets: {stats['n']} >= {THRESHOLDS['min_markets']}")

    print(f"\nStrict Criteria Met: {strict_count}/6")
    print(f"Weak Criteria Met: {weak_count}/6")

    # Determine verdict
    if strict_count == 6:
        verdict = 'VALIDATED'
        reason = 'All 6 strict criteria passed'
    elif weak_count >= 5:
        verdict = 'WEAK_EDGE'
        failed = []
        if not criteria['p_significant']:
            failed.append('p-value')
        if not criteria['not_price_proxy_strict']:
            failed.append(f"bucket_ratio ({stats['bucket_analysis']['bucket_ratio']:.0%} < 80%)")
        if not criteria['ci_excludes_zero']:
            failed.append('CI includes zero')
        if not criteria['temporal_stable']:
            failed.append('temporal stability')
        if not criteria['out_of_sample']:
            failed.append('out-of-sample')
        reason = f"5/6 weak criteria passed. Failed strict: {', '.join(failed)}"
    elif weak_count >= 4:
        verdict = 'WEAK_EDGE'
        reason = f"4/6 weak criteria passed"
    else:
        verdict = 'INVALIDATED'
        reason = f"Only {weak_count}/6 criteria passed"

    print(f"\n>>> VERDICT: {verdict}")
    print(f">>> REASON: {reason}")

    return verdict, reason, criteria


def main():
    print("=" * 80)
    print("LSD SESSION 002: PRODUCTION VALIDATION")
    print("Applying H123-level rigor to 3 independent strategies")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df = load_data()

    # Build baseline
    all_markets, baseline = build_baseline(df, THRESHOLDS['bucket_size'])

    # Strategy definitions
    strategies = [
        {
            'id': 'H207',
            'name': 'H-LSD-207: Dollar-Weighted Direction',
            'signal_func': detect_h207_signal,
            'description': 'Trades favor YES >20% more than dollars -> bet NO'
        },
        {
            'id': 'H211',
            'name': 'H-LSD-211: Conviction Ratio NO',
            'signal_func': detect_h211_signal,
            'description': 'NO trades are 2x bigger than YES trades -> bet NO'
        },
        {
            'id': 'H209',
            'name': 'H-LSD-209: Size Gradient',
            'signal_func': detect_h209_signal,
            'description': 'Larger trades correlate (r>0.3) with NO direction -> bet NO'
        }
    ]

    # Validate each strategy
    for strat in strategies:
        print(f"\n\n{'#'*80}")
        print(f"# STRATEGY: {strat['name']}")
        print(f"# {strat['description']}")
        print(f"{'#'*80}")

        # Detect signal
        signal_markets = strat['signal_func'](df)
        print(f"\nDetected {len(signal_markets):,} signal markets")

        # Calculate comprehensive stats
        stats = calculate_comprehensive_stats(signal_markets, baseline, strat['name'])

        # Determine verdict
        verdict, reason, criteria = determine_verdict(stats, strat['name'])

        # Store results
        results['strategies'][strat['id']] = {
            'name': strat['name'],
            'description': strat['description'],
            'stats': stats,
            'verdict': verdict,
            'reason': reason,
            'criteria': criteria if isinstance(criteria, dict) else {}
        }

        # Update summary
        if verdict == 'VALIDATED':
            results['summary']['validated'].append(strat['id'])
        elif verdict == 'WEAK_EDGE':
            results['summary']['weak_edge'].append(strat['id'])
        else:
            results['summary']['invalidated'].append(strat['id'])

    # Final Summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nVALIDATED STRATEGIES ({len(results['summary']['validated'])}):")
    for sid in results['summary']['validated']:
        s = results['strategies'][sid]
        print(f"  {sid}: {s['name']}")
        print(f"       Edge: {s['stats']['raw_edge']:+.2%}, Improvement: {s['stats']['bucket_analysis']['weighted_improvement']:+.2%}")
        print(f"       IMPLEMENT THIS STRATEGY")

    print(f"\nWEAK EDGE STRATEGIES ({len(results['summary']['weak_edge'])}):")
    for sid in results['summary']['weak_edge']:
        s = results['strategies'][sid]
        print(f"  {sid}: {s['name']}")
        print(f"       Edge: {s['stats']['raw_edge']:+.2%}, Improvement: {s['stats']['bucket_analysis']['weighted_improvement']:+.2%}")
        print(f"       Buckets: {s['stats']['bucket_analysis']['bucket_ratio']:.0%}")
        print(f"       Reason: {s['reason']}")
        print(f"       MONITOR - Potential secondary strategy")

    print(f"\nINVALIDATED STRATEGIES ({len(results['summary']['invalidated'])}):")
    for sid in results['summary']['invalidated']:
        s = results['strategies'][sid]
        print(f"  {sid}: {s['name']}")
        print(f"       Reason: {s['reason']}")
        print(f"       REJECT - Do not implement")

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {OUTPUT_PATH}")
    print(f"Session completed: {datetime.now()}")

    return results


if __name__ == '__main__':
    results = main()
