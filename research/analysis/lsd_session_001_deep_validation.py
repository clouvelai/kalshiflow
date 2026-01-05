"""
LSD SESSION 001 - DEEP VALIDATION

Validate flagged hypotheses with full rigor:
1. Bucket-by-bucket baseline comparison
2. Temporal stability check (4 quarters)
3. Concentration check (< 30% from single market)
4. Bootstrap CI

FLAGGED FROM SCREENING:
- EXT-002: Steam Cascade (6.06%)
- EXT-003 RLM NO: Reverse Line Movement (17.38%) <-- HIGHEST PRIORITY
- EXT-003 RLM YES: (10.01%)
- EXT-005: Buyback Reversal to NO (10.65%)
- LSD-001: Non-Fibonacci NO (5.24%)
- LSD-004: Mega Stack 4 (16.09%)
- LSD-004: Mega Stack 3 (11.05%)
- WILD-010: Triple Weird Stack (5.78%)
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]


def load_data():
    """Load the enriched trades data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def build_baseline_5c(df):
    """Build baseline win rates at 5c buckets for rigorous validation."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'first'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    baseline_no = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline_no[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n_markets': n
            }

    return all_markets, baseline_no


def deep_validate(signal_markets, baseline, strategy_name, side='no', df=None):
    """
    Full validation with:
    1. Bucket-by-bucket baseline comparison
    2. Temporal stability (4 quarters)
    3. Concentration check
    4. Bootstrap CI
    """
    print("\n" + "=" * 80)
    print(f"DEEP VALIDATION: {strategy_name}")
    print("=" * 80)

    n = len(signal_markets)
    if n < 50:
        print(f"REJECTED: Only {n} markets (need >= 50)")
        return {'status': 'rejected', 'reason': 'insufficient_markets', 'n': n}

    # Basic stats
    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['no_price'].mean()
    else:
        avg_price = signal_markets['yes_price'].mean()

    be = avg_price / 100
    edge = wr - be

    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\n1. BASIC STATS:")
    print(f"   Markets: {n}")
    print(f"   {side.upper()} Win Rate: {wr:.1%}")
    print(f"   Avg {side.upper()} Price: {avg_price:.1f}c")
    print(f"   Breakeven: {be:.1%}")
    print(f"   Raw Edge: {edge*100:.2f}%")
    print(f"   P-value: {p_value:.2e}")

    if p_value > 0.01:
        print(f"\n   [FAIL] NOT SIGNIFICANT (p > 0.01)")
        return {'status': 'rejected', 'reason': 'not_significant', 'n': n, 'edge': float(edge), 'p_value': float(p_value)}

    # =========================================================================
    # 2. BUCKET-BY-BUCKET COMPARISON
    # =========================================================================
    print(f"\n2. PRICE PROXY CHECK (5c Bucket Analysis):")

    signal_markets = signal_markets.copy()
    if side == 'no':
        signal_markets['bucket_5c'] = (signal_markets['no_price'] // 5) * 5
    else:
        signal_markets['bucket_5c'] = (signal_markets['yes_price'] // 5) * 5

    improvements = []
    print(f"   {'Bucket':<10} {'Sig WR':<10} {'Base WR':<10} {'Improve':<12} {'N Sig':<8}")

    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:
            continue

        sig_wr = (sig_bucket['market_result'] == side).mean()
        base_wr = baseline[bucket]['win_rate']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n_sig': n_sig
        })

        sign = '+' if imp >= 0 else ''
        print(f"   {bucket:.0f}-{bucket+5:.0f}c    {sig_wr:.1%}      {base_wr:.1%}      {sign}{imp*100:.2f}%       {n_sig}")

    if not improvements:
        print("   [FAIL] No buckets with sufficient data")
        return {'status': 'rejected', 'reason': 'no_buckets', 'n': n}

    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    neg_buckets = sum(1 for i in improvements if i['improvement'] <= 0)
    total_buckets = len(improvements)

    print(f"\n   Weighted Improvement: {weighted_imp*100:+.2f}%")
    print(f"   Positive Buckets: {pos_buckets}/{total_buckets} ({pos_buckets/total_buckets*100:.0f}%)")

    bucket_pass = weighted_imp > 0 and pos_buckets > neg_buckets
    print(f"   [{'PASS' if bucket_pass else 'FAIL'}] {'Genuine improvement' if bucket_pass else 'PRICE PROXY'}")

    # =========================================================================
    # 3. TEMPORAL STABILITY (4 quarters)
    # =========================================================================
    print(f"\n3. TEMPORAL STABILITY (Quarters):")

    if 'datetime' in signal_markets.columns:
        date_col = 'datetime'
    elif 'first_trade' in signal_markets.columns:
        date_col = 'first_trade'
    else:
        # Need to get datetime from original df
        date_col = None

    temporal_pass = True
    quarter_results = []

    if date_col:
        signal_markets['date'] = pd.to_datetime(signal_markets[date_col])
        min_date = signal_markets['date'].min()
        max_date = signal_markets['date'].max()

        date_range = max_date - min_date
        quarter_days = date_range.days // 4 if date_range.days > 0 else 1

        for q in range(4):
            q_start = min_date + pd.Timedelta(days=q * quarter_days)
            q_end = min_date + pd.Timedelta(days=(q + 1) * quarter_days)

            q_markets = signal_markets[(signal_markets['date'] >= q_start) & (signal_markets['date'] < q_end)]

            if len(q_markets) >= 10:
                q_wr = (q_markets['market_result'] == side).mean()
                if side == 'no':
                    q_be = q_markets['no_price'].mean() / 100
                else:
                    q_be = q_markets['yes_price'].mean() / 100
                q_edge = q_wr - q_be
                quarter_results.append({'q': q+1, 'n': len(q_markets), 'edge': q_edge})
                sign = '+' if q_edge >= 0 else ''
                print(f"   Q{q+1}: N={len(q_markets)}, Edge={sign}{q_edge*100:.2f}%")
            else:
                print(f"   Q{q+1}: Insufficient data (<10 markets)")

        positive_quarters = sum(1 for r in quarter_results if r['edge'] > 0)
        temporal_pass = positive_quarters >= 2
        print(f"   [{'PASS' if temporal_pass else 'FAIL'}] Positive quarters: {positive_quarters}/4")
    else:
        print("   No datetime available, skipping")

    # =========================================================================
    # 4. CONCENTRATION CHECK
    # =========================================================================
    print(f"\n4. CONCENTRATION CHECK:")

    if 'market_ticker' in signal_markets.columns:
        # Calculate profit contribution per market
        signal_markets['profit'] = (signal_markets['market_result'] == side).astype(float)

        market_profits = signal_markets.groupby('market_ticker')['profit'].sum()
        total_profit = market_profits.sum()

        if total_profit > 0:
            max_contribution = market_profits.max() / total_profit
            print(f"   Max single market contribution: {max_contribution*100:.1f}%")

            concentration_pass = max_contribution < 0.30
            print(f"   [{'PASS' if concentration_pass else 'FAIL'}] {'< 30%' if concentration_pass else '>= 30%'}")
        else:
            print("   No profitable markets to check")
            concentration_pass = True
    else:
        print("   No market_ticker available, skipping")
        concentration_pass = True

    # =========================================================================
    # 5. BOOTSTRAP CI
    # =========================================================================
    print(f"\n5. BOOTSTRAP CI (95%):")

    n_bootstrap = 1000
    bootstrap_edges = []

    for _ in range(n_bootstrap):
        sample = signal_markets.sample(n=len(signal_markets), replace=True)
        sample_wr = (sample['market_result'] == side).mean()
        if side == 'no':
            sample_be = sample['no_price'].mean() / 100
        else:
            sample_be = sample['yes_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_be)

    ci_lower = np.percentile(bootstrap_edges, 2.5)
    ci_upper = np.percentile(bootstrap_edges, 97.5)

    print(f"   95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    ci_pass = ci_lower > 0
    print(f"   [{'PASS' if ci_pass else 'FAIL'}] {'CI excludes 0' if ci_pass else 'CI includes 0'}")

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print(f"\n" + "=" * 40)
    print("VALIDATION SUMMARY:")
    print("=" * 40)

    all_pass = bucket_pass and temporal_pass and concentration_pass and ci_pass

    checks = [
        ("Statistical Significance", p_value < 0.01),
        ("Price Proxy Check", bucket_pass),
        ("Temporal Stability", temporal_pass),
        ("Concentration", concentration_pass),
        ("Bootstrap CI", ci_pass)
    ]

    for check_name, passed in checks:
        print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

    if all_pass:
        print(f"\n  VERDICT: **VALIDATED** - Genuine edge!")
    else:
        print(f"\n  VERDICT: REJECTED")

    return {
        'status': 'validated' if all_pass else 'rejected',
        'n': n,
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'edge': float(edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'total_buckets': total_buckets,
        'bucket_pass': bucket_pass,
        'temporal_pass': temporal_pass,
        'concentration_pass': concentration_pass,
        'ci_pass': ci_pass,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'quarter_results': quarter_results,
        'bucket_details': improvements
    }


def get_rlm_no_markets(df):
    """
    EXT-003: Reverse Line Movement (NO)
    >70% of trades are YES but price moved toward NO
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last'],
        'no_price': 'mean',
        'market_result': 'first',
        'count': 'size',
        'datetime': 'first'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'yes_trade_ratio', 'first_yes_price', 'last_yes_price', 'no_price', 'market_result', 'n_trades', 'datetime']

    market_stats['price_moved_no'] = market_stats['last_yes_price'] < market_stats['first_yes_price']

    rlm_no = market_stats[
        (market_stats['yes_trade_ratio'] > 0.7) &
        (market_stats['price_moved_no']) &
        (market_stats['n_trades'] >= 5)
    ].copy()

    return rlm_no


def get_buyback_reversal_no_markets(df):
    """
    EXT-005: Buyback Reversal to NO
    First half YES-heavy, second half NO-heavy with larger size
    """
    reversal_markets = []

    for market_ticker, market_df in df.groupby('market_ticker'):
        if len(market_df) < 6:
            continue

        market_df = market_df.sort_values('datetime').reset_index(drop=True)
        mid = len(market_df) // 2

        first_half = market_df.iloc[:mid]
        second_half = market_df.iloc[mid:]

        first_yes_ratio = (first_half['taker_side'] == 'yes').mean()
        second_yes_ratio = (second_half['taker_side'] == 'yes').mean()

        first_avg_size = first_half['count'].mean()
        second_avg_size = second_half['count'].mean()

        if first_yes_ratio > 0.6 and second_yes_ratio < 0.4 and second_avg_size > first_avg_size:
            reversal_markets.append({
                'market_ticker': market_ticker,
                'market_result': market_df['market_result'].iloc[0],
                'no_price': market_df['no_price'].mean(),
                'yes_price': market_df['yes_price'].mean(),
                'datetime': market_df['datetime'].iloc[0]
            })

    return pd.DataFrame(reversal_markets) if reversal_markets else pd.DataFrame()


def get_mega_stack_markets(df):
    """
    LSD-004: Mega Signal Stack (4 signals)
    lev_std < 0.7 AND weekend AND whale AND round_size AND no_ratio > 0.6
    """
    market_features = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'is_whale': 'any',
        'is_weekend': 'any',
        'is_round_size': 'any',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size',
        'datetime': 'first'
    }).reset_index()
    market_features.columns = [
        'market_ticker', 'lev_std', 'has_whale', 'is_weekend', 'has_round_size',
        'no_ratio', 'market_result', 'no_price', 'yes_price', 'n_trades', 'datetime'
    ]
    market_features['lev_std'] = market_features['lev_std'].fillna(0)

    mega_stack = market_features[
        (market_features['lev_std'] < 0.7) &
        (market_features['is_weekend'] == True) &
        (market_features['has_whale'] == True) &
        (market_features['has_round_size'] == True) &
        (market_features['no_ratio'] > 0.6) &
        (market_features['n_trades'] >= 5)
    ].copy()

    return mega_stack


def get_triple_weird_markets(df):
    """
    WILD-010: Triple Weird Stack (Fib + Weekend + Whale)
    """
    market_features = df.groupby('market_ticker').agg({
        'is_whale': 'any',
        'is_weekend': 'any',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size',
        'datetime': 'first'
    }).reset_index()
    market_features.columns = [
        'market_ticker', 'has_whale', 'is_weekend', 'no_ratio', 'market_result',
        'no_price', 'yes_price', 'n_trades', 'datetime'
    ]

    market_features['is_fibonacci'] = market_features['n_trades'].isin(FIBONACCI)

    triple_weird = market_features[
        (market_features['is_fibonacci'] == True) &
        (market_features['is_weekend'] == True) &
        (market_features['has_whale'] == True) &
        (market_features['no_ratio'] > 0.5)
    ].copy()

    return triple_weird


def get_steam_markets(df):
    """
    EXT-002: Steam Cascade
    5+ trades in same direction within 60 seconds, causing >5c price move
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    steam_markets = []

    for market_ticker, market_df in df_sorted.groupby('market_ticker'):
        if len(market_df) < 5:
            continue

        market_df = market_df.reset_index(drop=True)

        for i in range(len(market_df) - 4):
            window = market_df.iloc[i:i+5]
            time_span = (window['datetime'].max() - window['datetime'].min()).total_seconds()

            if time_span <= 60:
                sides = window['taker_side'].unique()
                if len(sides) == 1:
                    price_move = abs(window['yes_price'].iloc[-1] - window['yes_price'].iloc[0])
                    if price_move >= 5:
                        steam_markets.append({
                            'market_ticker': market_ticker,
                            'steam_direction': sides[0],
                            'market_result': market_df['market_result'].iloc[0],
                            'no_price': market_df['no_price'].mean(),
                            'yes_price': market_df['yes_price'].mean(),
                            'datetime': market_df['datetime'].iloc[0]
                        })
                        break

    return pd.DataFrame(steam_markets) if steam_markets else pd.DataFrame()


def get_non_fib_no_markets(df):
    """
    LSD-001: Non-Fibonacci NO Majority
    Markets where trade count is NOT a Fibonacci number, with NO majority
    """
    trade_counts = df.groupby('market_ticker').size().reset_index(name='n_trades')
    trade_counts = trade_counts.merge(
        df.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'taker_side': lambda x: (x == 'no').mean(),
            'datetime': 'first'
        }).reset_index(),
        on='market_ticker'
    )
    trade_counts.columns = ['market_ticker', 'n_trades', 'market_result', 'no_price', 'yes_price', 'no_ratio', 'datetime']

    non_fib = trade_counts[~trade_counts['n_trades'].isin(FIBONACCI)].copy()
    non_fib_no = non_fib[non_fib['no_ratio'] > 0.5]

    return non_fib_no


def main():
    print("=" * 80)
    print("LSD SESSION 001 - DEEP VALIDATION")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    df = load_data()
    all_markets, baseline_no = build_baseline_5c(df)

    results = {}

    # =========================================================================
    # VALIDATE TOP HYPOTHESES
    # =========================================================================

    # 1. EXT-003 RLM NO (17.38% - HIGHEST PRIORITY)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: EXT-003 RLM NO (Reverse Line Movement)")
    print("#" * 80)
    rlm_no = get_rlm_no_markets(df)
    results['EXT-003_RLM_NO'] = deep_validate(rlm_no, baseline_no, "EXT-003 RLM NO", side='no', df=df)

    # 2. LSD-004 Mega Stack 4 (16.09%)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: LSD-004 Mega Stack (4 Signals)")
    print("#" * 80)
    mega_stack = get_mega_stack_markets(df)
    results['LSD-004_MEGA_STACK'] = deep_validate(mega_stack, baseline_no, "LSD-004 Mega Stack", side='no', df=df)

    # 3. EXT-005 Buyback Reversal NO (10.65%)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: EXT-005 Buyback Reversal NO")
    print("#" * 80)
    buyback_no = get_buyback_reversal_no_markets(df)
    if len(buyback_no) > 0:
        results['EXT-005_BUYBACK_NO'] = deep_validate(buyback_no, baseline_no, "EXT-005 Buyback Reversal NO", side='no', df=df)
    else:
        results['EXT-005_BUYBACK_NO'] = {'status': 'rejected', 'reason': 'no_data'}

    # 4. EXT-002 Steam Cascade (6.06%)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: EXT-002 Steam Cascade (Follow)")
    print("#" * 80)
    steam = get_steam_markets(df)
    if len(steam) > 0:
        # Need custom validation for steam since we follow steam direction
        steam['correct'] = steam['steam_direction'] == steam['market_result']
        n = len(steam)
        wins = steam['correct'].sum()
        wr = wins / n
        edge = wr - 0.5
        print(f"\nSteam Cascade: N={n}, WR={wr:.1%}, Edge vs 50%: {edge*100:.2f}%")
        print("Note: Steam cascade uses variable direction, not suitable for standard bucket analysis")
        results['EXT-002_STEAM'] = {'status': 'custom', 'n': n, 'win_rate': float(wr), 'edge': float(edge)}
    else:
        results['EXT-002_STEAM'] = {'status': 'rejected', 'reason': 'no_data'}

    # 5. WILD-010 Triple Weird Stack (5.78%)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: WILD-010 Triple Weird Stack")
    print("#" * 80)
    triple_weird = get_triple_weird_markets(df)
    results['WILD-010_TRIPLE_WEIRD'] = deep_validate(triple_weird, baseline_no, "WILD-010 Triple Weird", side='no', df=df)

    # 6. LSD-001 Non-Fibonacci NO (5.24%)
    print("\n" + "#" * 80)
    print("# HYPOTHESIS: LSD-001 Non-Fibonacci NO")
    print("#" * 80)
    non_fib_no = get_non_fib_no_markets(df)
    results['LSD-001_NON_FIB_NO'] = deep_validate(non_fib_no, baseline_no, "LSD-001 Non-Fib NO", side='no', df=df)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    print("\n{:<30} {:>10} {:>10} {:>12} {:>15}".format(
        "Hypothesis", "N", "Edge", "Improvement", "Status"))
    print("-" * 80)

    for name, result in results.items():
        n = result.get('n', 0)
        edge = result.get('edge', 0) * 100 if result.get('edge') else 0
        imp = result.get('weighted_improvement', 0) * 100 if result.get('weighted_improvement') else 0
        status = result.get('status', 'unknown').upper()

        print("{:<30} {:>10} {:>9.2f}% {:>11.2f}% {:>15}".format(
            name, n, edge, imp, status))

    # Count validated
    validated = [k for k, v in results.items() if v.get('status') == 'validated']
    print(f"\n\nVALIDATED STRATEGIES: {len(validated)}")
    for v in validated:
        print(f"  - {v}")

    # Save results
    output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/lsd_session_001_deep_validation.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"\nSession completed: {datetime.now()}")

    return results


if __name__ == "__main__":
    results = main()
