"""
Session 012d: Deep Validation of S013 (Low Leverage Variance NO)

The only strategy that passed initial screening. Now we do full validation:
1. Temporal stability (does it work in all time periods?)
2. Concentration check (is profit spread across markets?)
3. Finer bucket analysis (5c instead of 10c)
4. Bootstrap confidence intervals
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


def get_s013_signal_markets(df):
    """
    S013 Signal: leverage_std < 0.7 AND >50% NO AND >= 5 trades
    """
    lev_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['std', 'mean'],
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size',
        'datetime': 'first'  # For temporal analysis
    }).reset_index()
    lev_stats.columns = ['market_ticker', 'lev_std', 'lev_mean', 'no_ratio', 'market_result', 'no_price', 'n_trades', 'first_trade']

    signal_markets = lev_stats[
        (lev_stats['lev_std'] < 0.7) &
        (lev_stats['no_ratio'] > 0.5) &
        (lev_stats['n_trades'] >= 5)
    ].copy()

    return signal_markets


def build_fine_baseline(df):
    """Build 5c bucket baseline for finer analysis"""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        no_wins = (bucket_markets['market_result'] == 'no').sum()
        if n >= 20:
            baseline[bucket] = {
                'win_rate': no_wins / n,
                'n_markets': n
            }

    return all_markets, baseline


def temporal_stability_check(signal_markets):
    """Check if strategy works across different time periods"""
    print("\n" + "="*80)
    print("TEMPORAL STABILITY CHECK")
    print("="*80)

    # Sort by first trade time
    signal_markets = signal_markets.sort_values('first_trade')

    # Split into 4 quarters
    n = len(signal_markets)
    q_size = n // 4

    quarters = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else n
        q = signal_markets.iloc[start:end]

        q_n = len(q)
        q_wins = (q['market_result'] == 'no').sum()
        q_wr = q_wins / q_n
        q_be = q['no_price'].mean() / 100
        q_edge = q_wr - q_be

        quarters.append({
            'quarter': i + 1,
            'n': q_n,
            'win_rate': q_wr,
            'breakeven': q_be,
            'edge': q_edge,
            'start': q['first_trade'].min(),
            'end': q['first_trade'].max()
        })

        print(f"  Q{i+1}: {q_n} markets, WR={q_wr:.1%}, BE={q_be:.1%}, Edge={q_edge*100:+.2f}%")

    positive_quarters = sum(1 for q in quarters if q['edge'] > 0)
    print(f"\n  Quarters with positive edge: {positive_quarters}/4")

    return quarters, positive_quarters >= 3


def concentration_check(signal_markets):
    """Check if profits are concentrated in few markets"""
    print("\n" + "="*80)
    print("CONCENTRATION CHECK")
    print("="*80)

    # Calculate profit per market
    signal_markets['profit'] = signal_markets.apply(
        lambda x: (100 - x['no_price']) if x['market_result'] == 'no' else -x['no_price'],
        axis=1
    )

    total_profit = signal_markets['profit'].sum()
    signal_markets['profit_pct'] = signal_markets['profit'] / total_profit if total_profit > 0 else 0

    # Top contributors
    top_10 = signal_markets.nlargest(10, 'profit')
    top_10_pct = top_10['profit'].sum() / total_profit if total_profit > 0 else 0

    max_single = signal_markets['profit'].max()
    max_single_pct = max_single / total_profit if total_profit > 0 else 0

    print(f"  Total profit (simulated): ${total_profit:.2f}")
    print(f"  Avg profit per market: ${total_profit/len(signal_markets):.2f}")
    print(f"  Top 10 markets contribute: {top_10_pct*100:.1f}%")
    print(f"  Max single market: {max_single_pct*100:.1f}%")

    passes = max_single_pct < 0.30

    print(f"\n  Concentration check: {'PASS' if passes else 'FAIL'} (max single < 30%)")

    return passes, max_single_pct


def fine_bucket_analysis(signal_markets, baseline):
    """5c bucket analysis for more granular view"""
    print("\n" + "="*80)
    print("FINE BUCKET ANALYSIS (5c)")
    print("="*80)

    signal_markets['bucket_5c'] = (signal_markets['no_price'] // 5) * 5

    print(f"  {'Bucket':<10} {'Sig WR':<10} {'Base WR':<10} {'Improve':<12} {'N':<8}")

    improvements = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:
            continue

        sig_wr = (sig_bucket['market_result'] == 'no').mean()
        base_wr = baseline[bucket]['win_rate']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n': n_sig
        })

        print(f"  {bucket:.0f}-{bucket+5:.0f}c   "
              f"{sig_wr:.1%}      "
              f"{base_wr:.1%}      "
              f"{imp*100:+.2f}%       "
              f"{n_sig}")

    if improvements:
        pos = sum(1 for i in improvements if i['improvement'] > 0)
        neg = sum(1 for i in improvements if i['improvement'] < 0)
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['improvement'] * i['n'] for i in improvements) / total_n

        print(f"\n  Positive buckets: {pos}/{len(improvements)}")
        print(f"  Weighted improvement: {weighted_imp*100:.2f}%")

        return improvements, pos > neg, weighted_imp
    return [], False, 0


def bootstrap_confidence_interval(signal_markets, n_bootstrap=1000):
    """Calculate bootstrap CI for the improvement"""
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVAL")
    print("="*80)

    edges = []
    n = len(signal_markets)

    for _ in range(n_bootstrap):
        sample = signal_markets.sample(n, replace=True)
        wr = (sample['market_result'] == 'no').mean()
        be = sample['no_price'].mean() / 100
        edge = wr - be
        edges.append(edge)

    edges = np.array(edges)
    ci_low = np.percentile(edges, 2.5)
    ci_high = np.percentile(edges, 97.5)
    mean_edge = edges.mean()
    std_edge = edges.std()

    print(f"  Mean Edge: {mean_edge*100:.2f}%")
    print(f"  Std Edge: {std_edge*100:.2f}%")
    print(f"  95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

    # Is zero outside the CI?
    significant = ci_low > 0
    print(f"\n  Zero outside CI: {'YES (significant)' if significant else 'NO (not significant)'}")

    return mean_edge, ci_low, ci_high, significant


def check_actionability(signal_markets, df):
    """
    Critical check: Can we ACTUALLY detect this signal in real-time?

    The signal requires knowing:
    1. leverage_std < 0.7 (need multiple trades)
    2. >50% NO trades (need multiple trades)
    3. >= 5 trades in market

    This means we can only detect the signal AFTER multiple trades.
    By then, is the edge still there?
    """
    print("\n" + "="*80)
    print("ACTIONABILITY CHECK")
    print("="*80)

    # For each signal market, when could we have detected it?
    # We need at least 5 trades to calculate std
    detection_analysis = []

    for ticker in signal_markets['market_ticker'].head(50):  # Sample
        market_trades = df[df['market_ticker'] == ticker].sort_values('datetime')

        if len(market_trades) < 5:
            continue

        # Calculate rolling stats
        for i in range(5, len(market_trades) + 1):
            subset = market_trades.iloc[:i]
            lev_std = subset['leverage_ratio'].std()
            no_ratio = (subset['taker_side'] == 'no').mean()

            if lev_std < 0.7 and no_ratio > 0.5:
                # Signal detected at trade i
                detection_point = i
                remaining_trades = len(market_trades) - i
                detection_analysis.append({
                    'ticker': ticker,
                    'detection_trade': detection_point,
                    'total_trades': len(market_trades),
                    'remaining_trades': remaining_trades,
                    'pct_complete': detection_point / len(market_trades)
                })
                break

    if detection_analysis:
        df_detect = pd.DataFrame(detection_analysis)
        avg_detection = df_detect['detection_trade'].mean()
        avg_remaining = df_detect['remaining_trades'].mean()
        avg_pct = df_detect['pct_complete'].mean()

        print(f"  Analyzed {len(detection_analysis)} markets")
        print(f"  Average detection at trade: {avg_detection:.1f}")
        print(f"  Average remaining trades after detection: {avg_remaining:.1f}")
        print(f"  Average % complete at detection: {avg_pct*100:.1f}%")

        if avg_pct > 0.8:
            print("\n  WARNING: Signal detected too late in market lifecycle!")
            print("  Most markets are >80% complete before signal triggers.")
            return False

        return True

    return False


def main():
    print("="*80)
    print("SESSION 012d: DEEP VALIDATION OF S013 (Low Leverage Variance NO)")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Get signal markets
    signal_markets = get_s013_signal_markets(df)
    print(f"\nS013 Signal Markets: {len(signal_markets)}")

    # Build fine baseline
    all_markets, baseline = build_fine_baseline(df)

    # Run all validation checks
    validation = {}

    # 1. Temporal stability
    quarters, temporal_pass = temporal_stability_check(signal_markets)
    validation['temporal_stability'] = temporal_pass

    # 2. Concentration check
    conc_pass, max_conc = concentration_check(signal_markets)
    validation['concentration'] = conc_pass
    validation['max_concentration'] = max_conc

    # 3. Fine bucket analysis
    improvements, bucket_pass, weighted_imp = fine_bucket_analysis(signal_markets, baseline)
    validation['bucket_analysis'] = bucket_pass
    validation['weighted_improvement'] = weighted_imp

    # 4. Bootstrap CI
    mean_edge, ci_low, ci_high, ci_pass = bootstrap_confidence_interval(signal_markets)
    validation['bootstrap_significant'] = ci_pass
    validation['ci_low'] = ci_low
    validation['ci_high'] = ci_high

    # 5. Actionability
    actionable = check_actionability(signal_markets, df)
    validation['actionable'] = actionable

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_pass = all([
        temporal_pass,
        conc_pass,
        bucket_pass,
        ci_pass,
        actionable
    ])

    checks = [
        ("Temporal Stability (3+/4 quarters)", temporal_pass),
        ("Concentration (<30% max)", conc_pass),
        ("Bucket Analysis (pos > neg)", bucket_pass),
        ("Bootstrap CI (excludes 0)", ci_pass),
        ("Actionable (can detect in time)", actionable)
    ]

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Overall: {'VALIDATED' if all_pass else 'REJECTED'}")

    # Final stats
    print(f"\n  Final Statistics:")
    print(f"    Markets: {len(signal_markets)}")
    print(f"    Raw Edge: {mean_edge*100:.2f}%")
    print(f"    95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"    Weighted Improvement: {weighted_imp*100:.2f}%")

    # Save results
    results = {
        'status': 'validated' if all_pass else 'rejected',
        'n_markets': len(signal_markets),
        'mean_edge': float(mean_edge),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'weighted_improvement': float(weighted_imp),
        'temporal_pass': temporal_pass,
        'concentration_pass': conc_pass,
        'bucket_pass': bucket_pass,
        'ci_pass': ci_pass,
        'actionable': actionable
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'/Users/samuelclark/Desktop/kalshiflow/research/reports/session012d_s013_deep_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
