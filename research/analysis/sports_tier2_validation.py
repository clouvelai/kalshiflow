"""
SPORTS TIER 2 VALIDATION: SPORTS-009 and SPORTS-002

SPORTS-009 (Spread Widening): +4.6% raw edge on Follow Sharp NO
SPORTS-002 (Opening Move Reversal): +3.2% raw edge

Let's validate these with full bucket-matched analysis.
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


def validate_sports009_spread_widening(df, baseline_by_bucket):
    """
    SPORTS-009: Spread Widening Before Sharp Entry

    Signal: High per-trade price volatility followed by large directional trade.
    """
    print("\n" + "=" * 80)
    print("VALIDATING SPORTS-009: SPREAD WIDENING BEFORE SHARP ENTRY")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    sharp_entry_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        mdf = mdf.reset_index(drop=True)
        prices = mdf['yes_price'].values
        sizes = mdf['count'].values
        sides = mdf['taker_side'].values

        price_moves = np.abs(np.diff(prices))

        if len(price_moves) < 5:
            continue

        # Look for spread widening + large trade
        for i in range(3, len(price_moves) - 1):
            recent_moves = price_moves[i-3:i]
            all_moves_avg = price_moves.mean()

            if recent_moves.mean() > all_moves_avg * 1.5:
                next_size = sizes[i + 1]
                avg_size = sizes.mean()

                if next_size > avg_size * 2:
                    sharp_entry_markets.append({
                        'market_ticker': market_ticker,
                        'market_result': mdf['market_result'].iloc[0],
                        'sharp_direction': sides[i + 1],
                        'volatility_ratio': recent_moves.mean() / all_moves_avg,
                        'size_ratio': next_size / avg_size,
                        'first_trade_time': mdf['datetime'].iloc[0],
                        'no_price': mdf['no_price'].mean(),
                        'yes_price': mdf['yes_price'].mean()
                    })
                    break

    se_df = pd.DataFrame(sharp_entry_markets)

    print(f"Total signal markets: {len(se_df)}")
    print(f"  Sharp NO: {(se_df['sharp_direction'] == 'no').sum()}")
    print(f"  Sharp YES: {(se_df['sharp_direction'] == 'yes').sum()}")

    # ===== VALIDATE FOLLOW SHARP NO =====
    print("\n----- FOLLOW SHARP NO -----")
    signal = se_df[se_df['sharp_direction'] == 'no'].copy()

    n = len(signal)
    wins = (signal['market_result'] == 'no').sum()
    wr = wins / n
    avg_price = signal['no_price'].mean()
    be = avg_price / 100
    edge = wr - be

    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"N = {n}")
    print(f"Win Rate = {wr:.1%}")
    print(f"Avg NO Price = {avg_price:.1f}c")
    print(f"Raw Edge = {edge:.1%}")
    print(f"P-value = {p_value:.6f}")

    # Bucket analysis
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
    bucket_analysis = bucket_analysis[bucket_analysis['count'] >= 10]

    if len(bucket_analysis) > 0:
        print("\nBucket Analysis (N >= 10):")
        print(bucket_analysis.to_string(index=False))

        pos_buckets = (bucket_analysis['improvement'] > 0).sum()
        total_buckets = len(bucket_analysis)
        print(f"\nPositive buckets: {pos_buckets}/{total_buckets}")

        weighted_improvement = (bucket_analysis['improvement'] * bucket_analysis['count']).sum() / bucket_analysis['count'].sum()
        print(f"Weighted improvement: {weighted_improvement:.1%}")

    # Temporal stability
    print("\n----- TEMPORAL STABILITY -----")
    signal['quarter'] = pd.qcut(signal['first_trade_time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    pos_quarters = 0
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_signal = signal[signal['quarter'] == q]
        if len(q_signal) < 20:
            continue

        q_wins = (q_signal['market_result'] == 'no').sum()
        q_n = len(q_signal)
        q_wr = q_wins / q_n
        q_avg_price = q_signal['no_price'].mean()
        q_edge = q_wr - q_avg_price / 100

        print(f"  {q}: N={q_n}, WR={q_wr:.1%}, Edge={q_edge:.1%}")
        if q_edge > 0:
            pos_quarters += 1

    print(f"\nPositive quarters: {pos_quarters}/4")

    return {
        'n': n,
        'edge': edge,
        'p_value': p_value
    }


def validate_sports002_opening_reversal(df, baseline_by_bucket):
    """
    SPORTS-002: Opening Move Reversal

    Signal: First 25% YES-heavy, second 25% NO-heavy (or vice versa).
    Bet: Follow the reversal direction.
    """
    print("\n" + "=" * 80)
    print("VALIDATING SPORTS-002: OPENING MOVE REVERSAL")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    reversal_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 12:
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        q1_end = n // 4
        q2_end = n // 2

        first_quarter = mdf.iloc[:q1_end]
        second_quarter = mdf.iloc[q1_end:q2_end]

        if len(first_quarter) < 3 or len(second_quarter) < 3:
            continue

        q1_yes_ratio = (first_quarter['taker_side'] == 'yes').mean()
        q2_yes_ratio = (second_quarter['taker_side'] == 'yes').mean()

        if q1_yes_ratio > 0.65 and q2_yes_ratio < 0.40:
            reversal_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'opener_direction': 'yes',
                'reversal_direction': 'no',
                'q1_yes_ratio': q1_yes_ratio,
                'q2_yes_ratio': q2_yes_ratio,
                'first_trade_time': mdf['datetime'].iloc[0],
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })
        elif q1_yes_ratio < 0.35 and q2_yes_ratio > 0.60:
            reversal_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'opener_direction': 'no',
                'reversal_direction': 'yes',
                'q1_yes_ratio': q1_yes_ratio,
                'q2_yes_ratio': q2_yes_ratio,
                'first_trade_time': mdf['datetime'].iloc[0],
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    rm_df = pd.DataFrame(reversal_markets)

    print(f"Total reversal markets: {len(rm_df)}")
    print(f"  Reversal to NO: {(rm_df['reversal_direction'] == 'no').sum()}")
    print(f"  Reversal to YES: {(rm_df['reversal_direction'] == 'yes').sum()}")

    # ===== VALIDATE REVERSAL TO NO =====
    print("\n----- FOLLOW REVERSAL TO NO -----")
    signal = rm_df[rm_df['reversal_direction'] == 'no'].copy()

    n = len(signal)
    wins = (signal['market_result'] == 'no').sum()
    wr = wins / n
    avg_price = signal['no_price'].mean()
    be = avg_price / 100
    edge = wr - be

    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"N = {n}")
    print(f"Win Rate = {wr:.1%}")
    print(f"Avg NO Price = {avg_price:.1f}c")
    print(f"Raw Edge = {edge:.1%}")
    print(f"P-value = {p_value:.6f}")

    # Bucket analysis
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

    if len(bucket_analysis) > 0:
        print("\nBucket Analysis (N >= 5):")
        print(bucket_analysis.to_string(index=False))

        pos_buckets = (bucket_analysis['improvement'] > 0).sum()
        total_buckets = len(bucket_analysis)
        print(f"\nPositive buckets: {pos_buckets}/{total_buckets}")

        weighted_improvement = (bucket_analysis['improvement'] * bucket_analysis['count']).sum() / bucket_analysis['count'].sum()
        print(f"Weighted improvement: {weighted_improvement:.1%}")

    # Temporal stability
    print("\n----- TEMPORAL STABILITY -----")
    signal['quarter'] = pd.qcut(signal['first_trade_time'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    pos_quarters = 0
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_signal = signal[signal['quarter'] == q]
        if len(q_signal) < 10:
            print(f"  {q}: N={len(q_signal)} (insufficient)")
            continue

        q_wins = (q_signal['market_result'] == 'no').sum()
        q_n = len(q_signal)
        q_wr = q_wins / q_n
        q_avg_price = q_signal['no_price'].mean()
        q_edge = q_wr - q_avg_price / 100

        print(f"  {q}: N={q_n}, WR={q_wr:.1%}, Edge={q_edge:.1%}")
        if q_edge > 0:
            pos_quarters += 1

    print(f"\nPositive quarters: {pos_quarters}/4")

    return {
        'n': n,
        'edge': edge,
        'p_value': p_value
    }


def main():
    df = load_data()
    all_markets, baseline_by_bucket = build_baseline()

    # Validate both
    sports009_result = validate_sports009_spread_widening(df, baseline_by_bucket)
    sports002_result = validate_sports002_opening_reversal(df, baseline_by_bucket)

    # Summary
    print("\n" + "=" * 80)
    print("TIER 2 VALIDATION SUMMARY")
    print("=" * 80)

    print(f"""
SPORTS-009 (Spread Widening):
  N = {sports009_result['n']}
  Raw Edge = {sports009_result['edge']:.1%}
  P-value = {sports009_result['p_value']:.4f}

SPORTS-002 (Opening Reversal):
  N = {sports002_result['n']}
  Raw Edge = {sports002_result['edge']:.1%}
  P-value = {sports002_result['p_value']:.4f}
""")


if __name__ == "__main__":
    main()
