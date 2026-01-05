"""
Session 012d: Re-test S010, S012, S013 with Session 012c Methodology

The critical test: At EACH price bucket, does the signal's win rate exceed
the baseline win rate for ALL markets at that same price level?

If improvement is negative at most buckets = PRICE PROXY
If improvement is positive at most buckets = GENUINE SIGNAL
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def build_baseline(df):
    """
    Build baseline win rates for ALL markets at each NO price bucket.
    This is the critical comparison - what's the expected win rate at each price?
    """
    # Get market-level aggregates
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'  # Use actual NO price column
    }).reset_index()

    all_markets['bucket'] = (all_markets['no_price'] // 10) * 10

    # Calculate baseline win rate per bucket
    baseline = {}
    for bucket in sorted(all_markets['bucket'].unique()):
        bucket_markets = all_markets[all_markets['bucket'] == bucket]
        n = len(bucket_markets)
        no_wins = (bucket_markets['market_result'] == 'no').sum()
        if n >= 20:  # Minimum for reliable estimate
            baseline[bucket] = {
                'win_rate': no_wins / n,
                'n_markets': n
            }

    return all_markets, baseline


def validate_strategy_strict(signal_markets, baseline, strategy_name):
    """
    Strict validation with bucket-by-bucket comparison.
    This is the Session 012c methodology that exposed S007, S008, S009 as price proxies.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {strategy_name}")
    print(f"{'='*80}")

    n = len(signal_markets)
    if n < 100:
        print(f"  REJECTED: Only {n} markets (need >= 100)")
        return {'status': 'rejected', 'reason': 'insufficient_markets', 'n': n}

    # Basic stats
    no_wins = (signal_markets['market_result'] == 'no').sum()
    wr = no_wins / n
    avg_no_price = signal_markets['no_price'].mean()
    be = avg_no_price / 100
    edge = wr - be

    # P-value
    z = (no_wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\n  Basic Stats:")
    print(f"    Markets: {n}")
    print(f"    NO Win Rate: {wr:.1%}")
    print(f"    Avg NO Price: {avg_no_price:.1f}c")
    print(f"    Breakeven: {be:.1%}")
    print(f"    Raw Edge: {edge*100:.2f}%")
    print(f"    P-value: {p_value:.2e}")

    if p_value > 0.01:
        print(f"  REJECTED: p-value {p_value:.4f} > 0.01 (NOT SIGNIFICANT)")
        return {'status': 'rejected', 'reason': 'not_significant', 'p': float(p_value)}

    # CRITICAL: Bucket-by-bucket comparison
    print(f"\n  Price Proxy Check (Bucket-by-Bucket):")
    signal_markets['bucket'] = (signal_markets['no_price'] // 10) * 10

    improvements = []
    print(f"  {'Bucket':<10} {'Sig WR':<10} {'Base WR':<10} {'Improve':<12} {'N Sig':<8} {'N Base':<8}")

    for bucket in sorted(signal_markets['bucket'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 10:  # Need minimum sample
            continue

        sig_wr = (sig_bucket['market_result'] == 'no').mean()
        base_wr = baseline[bucket]['win_rate']
        n_base = baseline[bucket]['n_markets']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n_sig': n_sig,
            'n_base': n_base
        })

        print(f"  {bucket:.0f}-{bucket+10:.0f}c    "
              f"{sig_wr:.1%}      "
              f"{base_wr:.1%}      "
              f"{imp*100:+.2f}%       "
              f"{n_sig:<8} "
              f"{n_base:<8}")

    if not improvements:
        print(f"  REJECTED: No buckets with sufficient data")
        return {'status': 'rejected', 'reason': 'no_buckets'}

    # Calculate weighted improvement
    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    # Count positive/negative buckets
    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    neg_buckets = sum(1 for i in improvements if i['improvement'] < 0)

    print(f"\n  Summary:")
    print(f"    Weighted Improvement: {weighted_imp*100:.2f}%")
    print(f"    Buckets with positive improvement: {pos_buckets}/{len(improvements)}")
    print(f"    Buckets with negative improvement: {neg_buckets}/{len(improvements)}")

    # Verdict
    if weighted_imp <= 0:
        print(f"\n  VERDICT: PRICE PROXY (weighted improvement <= 0)")
        return {
            'status': 'rejected',
            'reason': 'price_proxy',
            'edge': float(edge),
            'improvement': float(weighted_imp),
            'pos_buckets': pos_buckets,
            'neg_buckets': neg_buckets,
            'n': n
        }

    if pos_buckets <= neg_buckets:
        print(f"\n  VERDICT: PRICE PROXY (more negative than positive buckets)")
        return {
            'status': 'rejected',
            'reason': 'price_proxy_buckets',
            'edge': float(edge),
            'improvement': float(weighted_imp),
            'pos_buckets': pos_buckets,
            'neg_buckets': neg_buckets,
            'n': n
        }

    if weighted_imp < 0.01:
        print(f"\n  VERDICT: MARGINAL (improvement < 1%)")
        return {
            'status': 'marginal',
            'edge': float(edge),
            'improvement': float(weighted_imp),
            'pos_buckets': pos_buckets,
            'neg_buckets': neg_buckets,
            'n': n
        }

    print(f"\n  VERDICT: VALIDATED (genuine improvement over baseline)")
    return {
        'status': 'validated',
        'edge': float(edge),
        'improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'neg_buckets': neg_buckets,
        'n': n,
        'p_value': float(p_value)
    }


def test_s010_round_size_bot(df, all_markets, baseline):
    """
    S010: Round Size Bot Detection
    Signal: >60% of round-size trades (10, 25, 50, 100, 250, 500, 1000) are NO
    """
    ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]

    # Find round-size trades
    df['is_round'] = df['count'].isin(ROUND_SIZES)
    round_trades = df[df['is_round']]

    # Calculate NO ratio per market for round trades
    round_stats = round_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size'  # Number of round trades
    }).reset_index()
    round_stats.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price', 'n_round_trades']

    # Apply signal filter: >60% NO AND >= 5 round trades
    signal_markets = round_stats[
        (round_stats['no_ratio'] > 0.6) &
        (round_stats['n_round_trades'] >= 5)
    ].copy()

    print(f"\nS010 Signal Stats:")
    print(f"  Round trades found: {len(round_trades):,}")
    print(f"  Markets with >= 5 round trades: {len(round_stats[round_stats['n_round_trades'] >= 5]):,}")
    print(f"  Markets matching signal: {len(signal_markets):,}")

    return validate_strategy_strict(signal_markets, baseline, "S010: Round Size Bot NO")


def test_s012_burst_consensus(df, all_markets, baseline):
    """
    S012: Millisecond Burst Detection
    Signal: >60% of burst trades (3+ in same second) are NO
    """
    # Create second-level timestamp
    df['ts_second'] = df['datetime'].dt.floor('S')

    # Count trades per market-second
    second_counts = df.groupby(['market_ticker', 'ts_second']).size().reset_index(name='trades_in_second')

    # Find burst seconds (3+ trades)
    burst_seconds = second_counts[second_counts['trades_in_second'] >= 3]

    # Get trades from burst seconds
    df_with_counts = df.merge(second_counts, on=['market_ticker', 'ts_second'])
    burst_trades = df_with_counts[df_with_counts['trades_in_second'] >= 3]

    # Calculate NO ratio per market for burst trades
    burst_stats = burst_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size'  # Number of burst trades
    }).reset_index()
    burst_stats.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price', 'n_burst_trades']

    # Apply signal filter: >60% NO
    signal_markets = burst_stats[burst_stats['no_ratio'] > 0.6].copy()

    print(f"\nS012 Signal Stats:")
    print(f"  Burst seconds found: {len(burst_seconds):,}")
    print(f"  Burst trades found: {len(burst_trades):,}")
    print(f"  Markets with burst activity: {len(burst_stats):,}")
    print(f"  Markets matching signal: {len(signal_markets):,}")

    return validate_strategy_strict(signal_markets, baseline, "S012: Burst Consensus NO")


def test_s013_leverage_stability(df, all_markets, baseline):
    """
    S013: Low Leverage Variance NO
    Signal: leverage_std < 0.7 AND >50% NO AND >= 5 trades
    """
    # Calculate leverage stats per market
    lev_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['std', 'mean'],
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size'  # Number of trades
    }).reset_index()
    lev_stats.columns = ['market_ticker', 'lev_std', 'lev_mean', 'no_ratio', 'market_result', 'no_price', 'n_trades']

    # Apply signal filter: lev_std < 0.7 AND >50% NO AND >= 5 trades
    signal_markets = lev_stats[
        (lev_stats['lev_std'] < 0.7) &
        (lev_stats['no_ratio'] > 0.5) &
        (lev_stats['n_trades'] >= 5)
    ].copy()

    print(f"\nS013 Signal Stats:")
    print(f"  Markets with >= 5 trades: {len(lev_stats[lev_stats['n_trades'] >= 5]):,}")
    print(f"  Markets with low leverage std (<0.7): {len(lev_stats[lev_stats['lev_std'] < 0.7]):,}")
    print(f"  Markets matching signal: {len(signal_markets):,}")
    print(f"  Average leverage std in signal: {signal_markets['lev_std'].mean():.3f}")

    return validate_strategy_strict(signal_markets, baseline, "S013: Low Leverage Variance NO")


def main():
    print("="*80)
    print("SESSION 012d: RE-TEST BOT STRATEGIES WITH STRICT METHODOLOGY")
    print(f"Started: {datetime.now()}")
    print("="*80)
    print("\nCritical Test: At each price bucket, does signal WR > baseline WR?")
    print("If no -> PRICE PROXY (the signal just selects certain price levels)")
    print("If yes -> GENUINE SIGNAL (the signal provides information beyond price)")

    df = load_data()
    print(f"\nLoaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Build baseline
    print("\n" + "="*80)
    print("BUILDING BASELINE (ALL MARKETS)")
    print("="*80)
    all_markets, baseline = build_baseline(df)
    print(f"Built baseline for {len(baseline)} price buckets")
    print("\nBaseline Win Rates:")
    for bucket in sorted(baseline.keys()):
        b = baseline[bucket]
        print(f"  {bucket:.0f}-{bucket+10:.0f}c: {b['win_rate']:.1%} ({b['n_markets']} markets)")

    # Test each strategy
    results = {}

    results['s010'] = test_s010_round_size_bot(df, all_markets, baseline)
    results['s012'] = test_s012_burst_consensus(df, all_markets, baseline)
    results['s013'] = test_s013_leverage_stability(df, all_markets, baseline)

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    validated = 0
    for name, result in results.items():
        status = result['status'].upper()
        edge = result.get('edge', 0)
        imp = result.get('improvement', 0)
        n = result.get('n', 0)

        print(f"\n{name.upper()}: {status}")
        print(f"  Markets: {n}")
        print(f"  Raw Edge: {edge*100:.2f}%")
        print(f"  Improvement vs Baseline: {imp*100:.2f}%")

        if result['status'] == 'validated':
            validated += 1
            print(f"  Positive Buckets: {result.get('pos_buckets', 0)}/{result.get('pos_buckets', 0) + result.get('neg_buckets', 0)}")

    print(f"\n{'='*80}")
    print(f"VALIDATED STRATEGIES: {validated}/3")
    print(f"{'='*80}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'/Users/samuelclark/Desktop/kalshiflow/research/reports/session012d_results_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
