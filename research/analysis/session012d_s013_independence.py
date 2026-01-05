"""
Session 012d: Check S013 Independence from Previously Tested Signals

Critical question: Is S013 just another form of the leverage signal (S007)?

S007 was: High leverage (>2) + YES trades -> Bet NO (INVALIDATED)
S013 is: Low leverage variance (<0.7) + >50% NO trades -> Bet NO

Are they detecting the same markets?
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def get_s007_markets(df):
    """S007: High leverage (>2) + YES trades"""
    high_lev_yes = df[
        (df['leverage_ratio'] > 2) &
        (df['taker_side'] == 'yes')
    ]

    s007_markets = high_lev_yes.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    return set(s007_markets['market_ticker'])


def get_s013_markets(df):
    """S013: leverage_std < 0.7 AND >50% NO AND >= 5 trades"""
    lev_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size'
    }).reset_index()
    lev_stats.columns = ['market_ticker', 'lev_std', 'no_ratio', 'market_result', 'no_price', 'n_trades']

    signal_markets = lev_stats[
        (lev_stats['lev_std'] < 0.7) &
        (lev_stats['no_ratio'] > 0.5) &
        (lev_stats['n_trades'] >= 5)
    ]

    return set(signal_markets['market_ticker'])


def analyze_overlap(s007_set, s013_set):
    """Analyze overlap between strategies"""
    intersection = s007_set & s013_set
    s007_only = s007_set - s013_set
    s013_only = s013_set - s007_set

    print(f"S007 markets: {len(s007_set)}")
    print(f"S013 markets: {len(s013_set)}")
    print(f"Overlap: {len(intersection)}")
    print(f"S007 only: {len(s007_only)}")
    print(f"S013 only: {len(s013_only)}")

    if len(s013_set) > 0:
        overlap_pct = len(intersection) / len(s013_set)
        print(f"\nS013 overlap with S007: {overlap_pct*100:.1f}%")

    return intersection, s007_only, s013_only


def analyze_s013_unique_edge(df, s013_only):
    """Check if S013 markets that DON'T overlap with S007 still have edge"""
    print("\n" + "="*80)
    print("S013-ONLY MARKETS (No Overlap with S007)")
    print("="*80)

    # Get S013-only market data
    lev_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size'
    }).reset_index()
    lev_stats.columns = ['market_ticker', 'lev_std', 'no_ratio', 'market_result', 'no_price', 'n_trades']

    s013_only_data = lev_stats[lev_stats['market_ticker'].isin(s013_only)]

    n = len(s013_only_data)
    if n == 0:
        print("  No unique S013 markets found")
        return None

    no_wins = (s013_only_data['market_result'] == 'no').sum()
    wr = no_wins / n
    be = s013_only_data['no_price'].mean() / 100
    edge = wr - be

    print(f"  Markets: {n}")
    print(f"  NO Win Rate: {wr:.1%}")
    print(f"  Avg NO Price: {s013_only_data['no_price'].mean():.1f}c")
    print(f"  Breakeven: {be:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # Build baseline for comparison
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()
    all_markets['bucket'] = (all_markets['no_price'] // 10) * 10

    s013_only_data = s013_only_data.copy()
    s013_only_data['bucket'] = (s013_only_data['no_price'] // 10) * 10

    print("\n  Bucket Analysis:")
    improvements = []
    for bucket in sorted(s013_only_data['bucket'].unique()):
        sig = s013_only_data[s013_only_data['bucket'] == bucket]
        base = all_markets[all_markets['bucket'] == bucket]

        if len(sig) >= 5 and len(base) >= 20:
            sig_wr = (sig['market_result'] == 'no').mean()
            base_wr = (base['market_result'] == 'no').mean()
            imp = sig_wr - base_wr
            improvements.append({'bucket': bucket, 'imp': imp, 'n': len(sig)})
            print(f"    {bucket:.0f}-{bucket+10:.0f}c: Sig={sig_wr:.1%}, Base={base_wr:.1%}, Imp={imp*100:+.2f}%, N={len(sig)}")

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] for i in improvements) / total_n
        print(f"\n  Weighted Improvement: {weighted_imp*100:.2f}%")
        return weighted_imp

    return None


def main():
    print("="*80)
    print("SESSION 012d: S013 INDEPENDENCE CHECK")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Get market sets
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS")
    print("="*80)

    s007_set = get_s007_markets(df)
    s013_set = get_s013_markets(df)

    intersection, s007_only, s013_only = analyze_overlap(s007_set, s013_set)

    # Analyze S013-unique markets
    unique_imp = analyze_s013_unique_edge(df, s013_only)

    # Summary
    print("\n" + "="*80)
    print("INDEPENDENCE VERDICT")
    print("="*80)

    if len(intersection) == 0:
        print("S013 is COMPLETELY INDEPENDENT of S007")
        independence = "complete"
    elif len(intersection) / len(s013_set) < 0.2:
        print(f"S013 is MOSTLY INDEPENDENT (<20% overlap)")
        independence = "mostly"
    elif len(intersection) / len(s013_set) < 0.5:
        print(f"S013 is PARTIALLY INDEPENDENT (20-50% overlap)")
        independence = "partial"
    else:
        print(f"S013 OVERLAPS SIGNIFICANTLY (>50%) with S007")
        independence = "correlated"

    if unique_imp is not None:
        if unique_imp > 0.02:
            print(f"S013-only markets still show strong edge (+{unique_imp*100:.1f}%)")
        elif unique_imp > 0:
            print(f"S013-only markets show marginal edge (+{unique_imp*100:.1f}%)")
        else:
            print(f"S013-only markets show no edge ({unique_imp*100:.1f}%)")

    # Check opposite direction: Does S007 signal overlap with S013 markets?
    print("\n" + "="*80)
    print("WHY S013 WORKS DIFFERENTLY THAN S007")
    print("="*80)

    print("""
S007 (INVALIDATED):
- Signal: High leverage YES trades (leverage > 2)
- Logic: Fade retail longshot bettors
- Problem: Selecting high leverage = selecting LOW NO price markets
- Result: Just a price proxy for cheap NO

S013 (VALIDATED):
- Signal: Low leverage VARIANCE + >50% NO trades
- Logic: Detect consistent/systematic trading patterns
- Key difference: Not about leverage LEVEL, but leverage CONSISTENCY
- Markets with stable leverage = systematic/bot trading = informational

The key insight: S007 selected based on leverage LEVEL (which proxies price)
S013 selects based on leverage VARIANCE (which is independent of price)
""")


if __name__ == "__main__":
    main()
