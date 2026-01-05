"""
LSD SESSION 002: DEEP VALIDATION

Top candidates from screening:
1. H-LSD-207: Dollar-Weighted Direction - 12.05% edge, N=2063
2. H-LSD-211: Conviction Ratio - 11.75% edge, N=2719
3. H-LSD-208: Whale Consensus Counter - 11.21% edge, N=101
4. H-LSD-212: Trade Count vs Dollar Imbalance - 11.10% edge, N=789
5. H-LSD-209: Size Gradient - 10.21% edge, N=1859

CRITICAL: Must validate these are NOT price proxies!
- Bucket-by-bucket baseline comparison (5c granularity)
- Temporal stability (4 quarters)
- Independence from RLM
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
RESULTS_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/lsd_session_002_deep_validation.json'


def load_data():
    """Load the enriched trades data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000
    df['is_no'] = df['taker_side'] == 'no'
    df['is_yes'] = df['taker_side'] == 'yes'
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def build_baseline(df, bucket_size=5):
    """Build baseline win rates at each price bucket."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    all_markets['bucket'] = (all_markets['no_price'] // bucket_size) * bucket_size

    baseline_no = {}
    for bucket in sorted(all_markets['bucket'].unique()):
        bucket_markets = all_markets[all_markets['bucket'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline_no[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n': n
            }

    return all_markets, baseline_no


def deep_validate_signal(signal_markets, all_markets, baseline_no, name, bucket_size=5):
    """
    Deep validation with bucket-by-bucket comparison.
    """
    print(f"\n{'='*60}")
    print(f"DEEP VALIDATION: {name}")
    print(f"{'='*60}")

    n = len(signal_markets)
    if n < 50:
        return {'name': name, 'status': 'INSUFFICIENT_DATA', 'n': n}

    # Overall stats
    signal_markets['bucket'] = (signal_markets['no_price'] // bucket_size) * bucket_size
    win_rate = (signal_markets['market_result'] == 'no').mean()
    avg_no_price = signal_markets['no_price'].mean()
    breakeven = avg_no_price / 100
    raw_edge = win_rate - breakeven

    # P-value
    wins = (signal_markets['market_result'] == 'no').sum()
    if 0 < breakeven < 1:
        z = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven))
        p_value = 1 - stats.norm.cdf(z)
    else:
        p_value = 1.0

    print(f"Overall: N={n}, WR={win_rate:.1%}, Avg NO Price={avg_no_price:.1f}c, Raw Edge={raw_edge:.2%}")
    print(f"P-value: {p_value:.6f}")

    # Bucket-by-bucket comparison
    print(f"\nBucket Analysis ({bucket_size}c):")
    bucket_results = []
    positive_buckets = 0
    total_buckets = 0

    for bucket in sorted(signal_markets['bucket'].unique()):
        bucket_signal = signal_markets[signal_markets['bucket'] == bucket]
        n_signal = len(bucket_signal)

        if n_signal < 5:
            continue

        signal_wr = (bucket_signal['market_result'] == 'no').mean()

        if bucket in baseline_no:
            baseline_wr = baseline_no[bucket]['win_rate']
            improvement = signal_wr - baseline_wr
            total_buckets += 1
            if improvement > 0:
                positive_buckets += 1
            bucket_results.append({
                'bucket': bucket,
                'n_signal': n_signal,
                'signal_wr': signal_wr,
                'baseline_wr': baseline_wr,
                'improvement': improvement
            })
            marker = '+' if improvement > 0 else '-'
            print(f"  {bucket:3.0f}-{bucket+bucket_size:3.0f}c: Signal={signal_wr:.1%} vs Base={baseline_wr:.1%} -> {improvement:+.1%} [{marker}] (N={n_signal})")

    # Weighted improvement
    if bucket_results:
        total_n = sum(b['n_signal'] for b in bucket_results)
        weighted_improvement = sum(b['improvement'] * b['n_signal'] / total_n for b in bucket_results)
    else:
        weighted_improvement = 0

    bucket_ratio = positive_buckets / total_buckets if total_buckets > 0 else 0

    print(f"\nSummary:")
    print(f"  Positive Buckets: {positive_buckets}/{total_buckets} ({bucket_ratio:.0%})")
    print(f"  Weighted Improvement: {weighted_improvement:.2%}")

    # Temporal stability (4 quarters)
    signal_markets = signal_markets.sort_values('datetime')
    quarters = np.array_split(signal_markets, 4)
    quarter_results = []
    positive_quarters = 0

    print(f"\nTemporal Stability (4 Quarters):")
    for i, q in enumerate(quarters):
        if len(q) < 10:
            continue
        q_wr = (q['market_result'] == 'no').mean()
        q_price = q['no_price'].mean()
        q_edge = q_wr - q_price / 100
        quarter_results.append({'quarter': i+1, 'n': len(q), 'edge': q_edge})
        if q_edge > 0:
            positive_quarters += 1
        marker = '+' if q_edge > 0 else '-'
        print(f"  Q{i+1}: N={len(q):4d}, Edge={q_edge:+.2%} [{marker}]")

    # Bootstrap CI
    bootstrap_edges = []
    for _ in range(1000):
        sample = signal_markets.sample(n=len(signal_markets), replace=True)
        sample_wr = (sample['market_result'] == 'no').mean()
        sample_price = sample['no_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_price)

    ci_low = np.percentile(bootstrap_edges, 2.5)
    ci_high = np.percentile(bootstrap_edges, 97.5)

    print(f"\nBootstrap 95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
    ci_excludes_zero = ci_low > 0

    # Verdict
    print(f"\n--- VERDICT ---")
    is_price_proxy = bucket_ratio < 0.6 or weighted_improvement < 0.02
    is_significant = p_value < 0.01
    is_stable = positive_quarters >= 2
    is_validated = is_significant and not is_price_proxy and is_stable and ci_excludes_zero

    if is_validated:
        print(f"  VALIDATED - Passes all checks!")
    elif is_price_proxy:
        print(f"  PRICE PROXY - Only {bucket_ratio:.0%} positive buckets, improvement={weighted_improvement:.2%}")
    elif not is_significant:
        print(f"  NOT SIGNIFICANT - p={p_value:.4f}")
    elif not is_stable:
        print(f"  UNSTABLE - Only {positive_quarters}/4 quarters positive")
    else:
        print(f"  BORDERLINE - CI includes zero")

    return {
        'name': name,
        'n': n,
        'win_rate': round(win_rate, 4),
        'avg_no_price': round(avg_no_price, 2),
        'raw_edge': round(raw_edge, 4),
        'p_value': round(p_value, 6),
        'positive_buckets': positive_buckets,
        'total_buckets': total_buckets,
        'bucket_ratio': round(bucket_ratio, 2),
        'weighted_improvement': round(weighted_improvement, 4),
        'positive_quarters': positive_quarters,
        'ci_low': round(ci_low, 4),
        'ci_high': round(ci_high, 4),
        'is_validated': is_validated,
        'status': 'VALIDATED' if is_validated else ('PRICE_PROXY' if is_price_proxy else 'WEAK')
    }


def detect_h207_signal(df):
    """H-LSD-207: Dollar-Weighted Direction"""
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

    yes_value = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].sum()
    market_stats['yes_value'] = market_stats['market_ticker'].map(yes_value).fillna(0)
    market_stats['yes_dollar_ratio'] = market_stats['yes_value'] / market_stats['total_value']

    market_stats['divergence'] = market_stats['yes_trade_ratio'] - market_stats['yes_dollar_ratio']

    # Trades favor YES by >20% more than dollars (dollars going to NO)
    signal = market_stats[
        (market_stats['divergence'] > 0.20) &
        (market_stats['total_trades'] >= 5)
    ]
    return signal


def detect_h211_signal(df):
    """H-LSD-211: Conviction Ratio (NO avg size > 2x YES avg size)"""
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

    signal = market_info[
        (market_info['size_ratio'] > 2) &
        (market_info['n_trades'] >= 5)
    ]
    return signal


def detect_h212_signal(df):
    """H-LSD-212: Trade Count vs Dollar Imbalance (Retail YES, Smart NO)"""
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

    yes_value = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].sum()
    market_stats['yes_value'] = market_stats['market_ticker'].map(yes_value).fillna(0)
    market_stats['yes_dollar_ratio'] = market_stats['yes_value'] / market_stats['total_value']

    signal = market_stats[
        (market_stats['yes_trade_ratio'] > 0.70) &
        (market_stats['yes_dollar_ratio'] < 0.60) &
        (market_stats['total_trades'] >= 5)
    ]
    return signal


def detect_h209_signal(df):
    """H-LSD-209: Size Gradient (larger trades bet NO)"""
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

    signal = combined[
        (combined['corr'] > 0.3) &
        (combined['n_trades'] >= 5)
    ]
    return signal


def detect_h210_signal(df):
    """H-LSD-210: Price Stickiness (high volume, small price move, bet NO when price dropped)"""
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'yes_price': ['first', 'last', 'count', 'mean'],
        'market_result': 'first',
        'no_price': 'mean',
        'datetime': 'min'
    })
    market_stats.columns = ['first_yes', 'last_yes', 'n_trades', 'yes_price', 'market_result', 'no_price', 'datetime']
    market_stats = market_stats.reset_index()

    market_stats['price_move'] = abs(market_stats['last_yes'] - market_stats['first_yes'])
    market_stats['price_dropped'] = market_stats['last_yes'] < market_stats['first_yes']

    signal = market_stats[
        (market_stats['n_trades'] >= 20) &
        (market_stats['price_move'] < 10) &
        (market_stats['price_dropped'] == True)
    ]
    return signal


def check_independence_from_rlm(df, signal_markets):
    """Check overlap with RLM signal."""
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Detect RLM signal
    market_stats = df_sorted.groupby('market_ticker').agg({
        'is_yes': 'mean',
        'yes_price': ['first', 'last']
    })
    market_stats.columns = ['yes_ratio', 'first_yes', 'last_yes']
    market_stats = market_stats.reset_index()

    market_stats['price_moved_no'] = market_stats['last_yes'] < market_stats['first_yes']

    rlm_markets = market_stats[
        (market_stats['yes_ratio'] > 0.70) &
        (market_stats['price_moved_no'] == True)
    ]['market_ticker'].tolist()

    signal_tickers = signal_markets['market_ticker'].tolist()

    overlap = len(set(signal_tickers) & set(rlm_markets))
    overlap_pct = overlap / len(signal_tickers) if len(signal_tickers) > 0 else 0

    return {
        'signal_count': len(signal_tickers),
        'rlm_count': len(rlm_markets),
        'overlap': overlap,
        'overlap_pct': round(overlap_pct, 3),
        'is_independent': overlap_pct < 0.5
    }


def main():
    print("=" * 80)
    print("LSD SESSION 002: DEEP VALIDATION")
    print("=" * 80)

    df = load_data()
    all_markets, baseline_no = build_baseline(df)

    results = {}

    # H-LSD-207: Dollar-Weighted Direction
    signal_207 = detect_h207_signal(df)
    results['H207'] = deep_validate_signal(signal_207, all_markets, baseline_no, "H-LSD-207: Dollar-Weighted Direction (bet NO)")
    rlm_check_207 = check_independence_from_rlm(df, signal_207)
    results['H207']['rlm_independence'] = rlm_check_207
    print(f"\nRLM Independence: {rlm_check_207['overlap_pct']:.0%} overlap ({'INDEPENDENT' if rlm_check_207['is_independent'] else 'CORRELATED'})")

    # H-LSD-211: Conviction Ratio
    signal_211 = detect_h211_signal(df)
    results['H211'] = deep_validate_signal(signal_211, all_markets, baseline_no, "H-LSD-211: Conviction Ratio NO (bet NO)")
    rlm_check_211 = check_independence_from_rlm(df, signal_211)
    results['H211']['rlm_independence'] = rlm_check_211
    print(f"\nRLM Independence: {rlm_check_211['overlap_pct']:.0%} overlap ({'INDEPENDENT' if rlm_check_211['is_independent'] else 'CORRELATED'})")

    # H-LSD-212: Trade Count vs Dollar Imbalance
    signal_212 = detect_h212_signal(df)
    results['H212'] = deep_validate_signal(signal_212, all_markets, baseline_no, "H-LSD-212: Retail YES Smart NO (bet NO)")
    rlm_check_212 = check_independence_from_rlm(df, signal_212)
    results['H212']['rlm_independence'] = rlm_check_212
    print(f"\nRLM Independence: {rlm_check_212['overlap_pct']:.0%} overlap ({'INDEPENDENT' if rlm_check_212['is_independent'] else 'CORRELATED'})")

    # H-LSD-209: Size Gradient
    signal_209 = detect_h209_signal(df)
    results['H209'] = deep_validate_signal(signal_209, all_markets, baseline_no, "H-LSD-209: Size Gradient (bet NO)")
    rlm_check_209 = check_independence_from_rlm(df, signal_209)
    results['H209']['rlm_independence'] = rlm_check_209
    print(f"\nRLM Independence: {rlm_check_209['overlap_pct']:.0%} overlap ({'INDEPENDENT' if rlm_check_209['is_independent'] else 'CORRELATED'})")

    # H-LSD-210: Price Stickiness
    signal_210 = detect_h210_signal(df)
    results['H210'] = deep_validate_signal(signal_210, all_markets, baseline_no, "H-LSD-210: Price Stickiness (bet NO)")
    rlm_check_210 = check_independence_from_rlm(df, signal_210)
    results['H210']['rlm_independence'] = rlm_check_210
    print(f"\nRLM Independence: {rlm_check_210['overlap_pct']:.0%} overlap ({'INDEPENDENT' if rlm_check_210['is_independent'] else 'CORRELATED'})")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    validated = []
    price_proxy = []

    for h_id, result in results.items():
        status = result.get('status', 'UNKNOWN')
        rlm_ind = result.get('rlm_independence', {})
        ind_status = 'INDEPENDENT' if rlm_ind.get('is_independent', True) else f"CORRELATED ({rlm_ind.get('overlap_pct', 0):.0%})"

        print(f"\n{h_id}: {result['name']}")
        print(f"  Status: {status}")
        print(f"  Raw Edge: {result['raw_edge']:.2%}")
        print(f"  Weighted Improvement: {result['weighted_improvement']:.2%}")
        print(f"  Bucket Coverage: {result['positive_buckets']}/{result['total_buckets']} ({result['bucket_ratio']:.0%})")
        print(f"  Temporal: {result['positive_quarters']}/4 quarters positive")
        print(f"  RLM Independence: {ind_status}")

        if status == 'VALIDATED':
            validated.append({
                'id': h_id,
                'name': result['name'],
                'edge': result['raw_edge'],
                'improvement': result['weighted_improvement'],
                'n': result['n'],
                'is_independent': rlm_ind.get('is_independent', True)
            })
        else:
            price_proxy.append(h_id)

    print("\n" + "=" * 80)
    print("VALIDATED STRATEGIES")
    print("=" * 80)
    if validated:
        for v in sorted(validated, key=lambda x: -x['edge']):
            ind = "INDEPENDENT" if v['is_independent'] else "CORRELATED with RLM"
            print(f"  {v['id']}: {v['name']} - Edge={v['edge']:.2%}, Improvement={v['improvement']:.2%}, N={v['n']}, {ind}")
    else:
        print("  No strategies validated")

    print("\n" + "=" * 80)
    print("REJECTED (Price Proxies)")
    print("=" * 80)
    for pp in price_proxy:
        print(f"  {pp}")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'session': 'LSD_002_DEEP_VALIDATION',
            'results': results,
            'validated': validated,
            'rejected': price_proxy
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == '__main__':
    results = main()
