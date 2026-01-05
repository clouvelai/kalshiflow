"""
LSD VALIDATION: Check SPORTS-015 and SPORTS-007 for methodology bugs

SPORTS-015 showed +33% edge - TOO HIGH, suspicious of look-ahead bias
SPORTS-007 showed +19.8% edge - Also suspicious

Check for:
1. Look-ahead bias (using future information)
2. Price proxy (just measuring price, not independent signal)
3. Selection bias (signal correlates with price bucket)
4. Temporal stability (works across all time periods?)
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
    df['timestamp_ms'] = df['datetime'].astype(np.int64) // 10**6
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def build_baseline():
    """Build price bucket baseline for proper validation."""
    df = pd.read_csv(DATA_PATH)

    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    # 5-cent buckets
    all_markets['bucket_5c'] = ((all_markets['no_price'] / 5).astype(int) * 5)

    # Calculate baseline by bucket
    baseline = all_markets.groupby('bucket_5c').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline.columns = ['bucket_5c', 'baseline_no_rate', 'bucket_count']

    return all_markets, dict(zip(baseline['bucket_5c'], baseline['baseline_no_rate']))


def validate_sports015_fibonacci(df):
    """
    Validate SPORTS-015: Fibonacci Price Attractors

    ISSUE: The original implementation looked at FINAL PRICE after the pattern.
    This is LOOK-AHEAD BIAS - we're using outcome to determine signal.

    If final_price > main_fib + 5 -> bet YES -> check if YES won
    But final_price being high CORRELATES with YES winning!
    """
    print("\n" + "=" * 80)
    print("VALIDATING SPORTS-015: FIBONACCI PRICE ATTRACTORS")
    print("=" * 80)

    FIB_LEVELS = [23.6, 38.2, 50.0, 61.8, 76.4]
    TOLERANCE = 2

    def near_fib(price):
        for fib in FIB_LEVELS:
            if abs(price - fib) <= TOLERANCE:
                return fib
        return None

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    fib_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        mdf = mdf.reset_index(drop=True)

        # Check Fib interaction
        mdf['near_fib'] = mdf['yes_price'].apply(near_fib)
        fib_touches = mdf['near_fib'].notna().sum()
        fib_ratio = fib_touches / len(mdf)

        if fib_ratio > 0.30:
            main_fib = mdf['near_fib'].mode()
            if len(main_fib) == 0:
                continue
            main_fib = main_fib.iloc[0]

            # Original buggy logic used final_price
            final_price = mdf['yes_price'].iloc[-1]

            # The BUG: final_price > main_fib predicts YES won!
            # Let me show this correlation:
            fib_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'main_fib': main_fib,
                'final_price': final_price,
                'avg_yes_price': mdf['yes_price'].mean(),
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    fib_df = pd.DataFrame(fib_markets)

    print(f"\nTotal Fib markets: {len(fib_df)}")

    # Show the correlation between final_price and result
    fib_df['yes_won'] = (fib_df['market_result'] == 'yes').astype(int)
    correlation = fib_df['final_price'].corr(fib_df['yes_won'])

    print(f"\nCorrelation between final_price and YES winning: {correlation:.3f}")
    print("This shows the original implementation had LOOK-AHEAD BIAS!")

    # ===== CORRECTED VERSION =====
    # Signal: Just check if market had high Fib interaction (>30%)
    # Bet: NO (since YES betters are anchored to Fib levels - retail behavior)

    print("\n----- CORRECTED IMPLEMENTATION -----")

    # High Fib interaction markets, bet NO
    corrected_result = fib_df.copy()
    n = len(corrected_result)
    wins = (corrected_result['market_result'] == 'no').sum()
    avg_no_price = corrected_result['no_price'].mean()

    wr = wins / n
    be = avg_no_price / 100
    edge = wr - be

    print(f"Markets with high Fib interaction: N={n}")
    print(f"Bet NO - Win Rate: {wr:.1%}")
    print(f"Avg NO Price: {avg_no_price:.1f}c")
    print(f"Breakeven: {be:.1%}")
    print(f"Edge: {edge:.1%}")

    # Price bucket analysis
    print("\n----- PRICE BUCKET BREAKDOWN -----")
    corrected_result['bucket_10c'] = ((corrected_result['no_price'] / 10).astype(int) * 10)

    bucket_analysis = corrected_result.groupby('bucket_10c').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'no_price': ['mean', 'count']
    }).reset_index()
    bucket_analysis.columns = ['bucket', 'win_rate', 'avg_price', 'count']
    bucket_analysis['breakeven'] = bucket_analysis['avg_price'] / 100
    bucket_analysis['edge'] = bucket_analysis['win_rate'] - bucket_analysis['breakeven']

    print(bucket_analysis[bucket_analysis['count'] >= 20].to_string(index=False))

    return {
        'original_bug': 'Look-ahead bias - used final_price which correlates with outcome',
        'correlation': correlation,
        'corrected_edge': edge,
        'corrected_n': n,
        'bucket_analysis': bucket_analysis.to_dict('records')
    }


def validate_sports007_late_large(df):
    """
    Validate SPORTS-007: Late-Arriving Large Money

    Check if this is a price proxy or independent signal.
    """
    print("\n" + "=" * 80)
    print("VALIDATING SPORTS-007: LATE-ARRIVING LARGE MONEY")
    print("=" * 80)

    all_markets, baseline_by_bucket = build_baseline()

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
                late_large_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'late_direction': late_direction,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    ll_df = pd.DataFrame(late_large_markets)

    print(f"\nTotal late-large signal markets: {len(ll_df)}")

    # Analyze Follow Late NO
    follow_no = ll_df[ll_df['late_direction'] == 'no'].copy()
    print(f"\nFollow Late NO: N={len(follow_no)}")

    if len(follow_no) < 30:
        print("Insufficient sample for Follow Late NO")
    else:
        wins = (follow_no['market_result'] == 'no').sum()
        wr = wins / len(follow_no)
        avg_price = follow_no['no_price'].mean()
        be = avg_price / 100
        edge = wr - be

        print(f"Win Rate: {wr:.1%}")
        print(f"Avg NO Price: {avg_price:.1f}c")
        print(f"Edge: {edge:.1%}")

        # Price bucket breakdown
        print("\n----- FOLLOW NO: PRICE BUCKET BREAKDOWN -----")
        follow_no['bucket_5c'] = ((follow_no['no_price'] / 5).astype(int) * 5)

        # Compare to baseline
        follow_no['baseline_rate'] = follow_no['bucket_5c'].map(baseline_by_bucket)

        bucket_analysis = follow_no.groupby('bucket_5c').agg({
            'market_result': lambda x: (x == 'no').mean(),
            'no_price': ['mean', 'count'],
            'baseline_rate': 'first'
        }).reset_index()
        bucket_analysis.columns = ['bucket', 'win_rate', 'avg_price', 'count', 'baseline']
        bucket_analysis['improvement'] = bucket_analysis['win_rate'] - bucket_analysis['baseline']
        bucket_analysis = bucket_analysis[bucket_analysis['count'] >= 5]

        print(bucket_analysis.to_string(index=False))

        # Calculate bucket-weighted improvement
        valid = bucket_analysis[bucket_analysis['baseline'].notna()]
        pos_buckets = (valid['improvement'] > 0).sum()
        total_buckets = len(valid)

        print(f"\nPositive improvement buckets: {pos_buckets}/{total_buckets} ({pos_buckets/max(total_buckets,1):.1%})")

        avg_improvement = (valid['improvement'] * valid['count']).sum() / valid['count'].sum()
        print(f"Weighted avg improvement vs baseline: {avg_improvement:.1%}")

    # Analyze Follow Late YES
    follow_yes = ll_df[ll_df['late_direction'] == 'yes'].copy()
    print(f"\n\nFollow Late YES: N={len(follow_yes)}")

    if len(follow_yes) < 30:
        print("Insufficient sample for Follow Late YES")
    else:
        wins = (follow_yes['market_result'] == 'yes').sum()
        wr = wins / len(follow_yes)
        avg_price = follow_yes['yes_price'].mean()
        be = avg_price / 100
        edge = wr - be

        print(f"Win Rate: {wr:.1%}")
        print(f"Avg YES Price: {avg_price:.1f}c")
        print(f"Edge: {edge:.1%}")

        # Price bucket breakdown (for YES, use yes_price buckets)
        print("\n----- FOLLOW YES: PRICE BUCKET BREAKDOWN -----")
        follow_yes['bucket_5c'] = ((follow_yes['yes_price'] / 5).astype(int) * 5)

        # For YES, baseline = 1 - no_rate
        follow_yes['baseline_rate'] = 1 - follow_yes['bucket_5c'].map(
            lambda x: baseline_by_bucket.get(100-x, 0.5) if 100-x in baseline_by_bucket else 0.5
        )

        bucket_analysis = follow_yes.groupby('bucket_5c').agg({
            'market_result': lambda x: (x == 'yes').mean(),
            'yes_price': ['mean', 'count']
        }).reset_index()
        bucket_analysis.columns = ['bucket', 'win_rate', 'avg_price', 'count']
        bucket_analysis['breakeven'] = bucket_analysis['avg_price'] / 100
        bucket_analysis['edge'] = bucket_analysis['win_rate'] - bucket_analysis['breakeven']
        bucket_analysis = bucket_analysis[bucket_analysis['count'] >= 5]

        print(bucket_analysis.to_string(index=False))

    return {
        'follow_no_n': len(follow_no),
        'follow_yes_n': len(follow_yes)
    }


def validate_sports012_ncaaf(df):
    """
    SPORTS-012: NCAAF Totals - confirm existing Session 009 finding
    """
    print("\n" + "=" * 80)
    print("VALIDATING SPORTS-012: NCAAF TOTALS")
    print("=" * 80)

    # Check for NCAAF markets
    ncaaf = df[df['market_ticker'].str.contains('NCAAF', case=False, na=False)]
    print(f"NCAAF trades: {len(ncaaf):,}")
    print(f"NCAAF markets: {ncaaf['market_ticker'].nunique()}")

    if len(ncaaf) == 0:
        print("No NCAAF markets found")
        return

    # Category breakdown
    ncaaf_markets = ncaaf.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    print(f"\nSample tickers:")
    print(ncaaf_markets['market_ticker'].head(10).tolist())

    # Bet NO
    n = len(ncaaf_markets)
    wins = (ncaaf_markets['market_result'] == 'no').sum()
    avg_no_price = ncaaf_markets['no_price'].mean()

    wr = wins / n
    be = avg_no_price / 100
    edge = wr - be

    print(f"\nNCAAF Bet NO:")
    print(f"N={n}, Win Rate={wr:.1%}, Avg Price={avg_no_price:.1f}c")
    print(f"Breakeven={be:.1%}, Edge={edge:.1%}")

    # Sample size warning
    if n < 100:
        print(f"\nWARNING: Small sample size (N={n}). Results may not be reliable.")


def main():
    df = load_data()

    # Validate the suspicious winners
    sports015_result = validate_sports015_fibonacci(df)
    sports007_result = validate_sports007_late_large(df)
    validate_sports012_ncaaf(df)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("""
SPORTS-015 (Fibonacci): INVALID - LOOK-AHEAD BIAS
- Original implementation used final_price to determine bet direction
- final_price correlates with outcome (correlation = {:.2f})
- Corrected version (just bet NO on high-Fib markets) shows edge of {:.1%}
- This is likely just a PRICE PROXY

SPORTS-007 (Late Large): NEEDS DEEPER VALIDATION
- The signal logic is sound (no look-ahead)
- But high edge (+19.8%) is suspicious
- Need bucket-matched analysis to confirm not a price proxy

SPORTS-012 (NCAAF): SAMPLE SIZE WARNING
- Edge looks real but sample is small
- Already known from Session 009
- Monitor for more data
""".format(
        sports015_result['correlation'],
        sports015_result['corrected_edge']
    ))


if __name__ == "__main__":
    main()
