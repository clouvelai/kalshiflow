"""
Session 012: Compare "Fade YES >85c" to S007 (Leverage Fade)
Is this a new strategy or just S007 by another name?
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def analyze_overlap(df):
    """
    Compare Fade YES >85c to S007 (Fade High Leverage YES)
    """
    print("\n" + "="*80)
    print("COMPARING: Fade YES >85c vs S007 (Fade High Leverage)")
    print("="*80)

    # Strategy 1: Fade YES at >85c
    strategy1_markets = df[
        (df['taker_side'] == 'yes') &
        (df['trade_price'] > 85)
    ]['market_ticker'].unique()

    # Strategy 2: S007 - Fade High Leverage YES (leverage > 2)
    strategy2_markets = df[
        (df['taker_side'] == 'yes') &
        (df['leverage_ratio'] > 2)
    ]['market_ticker'].unique()

    # Overlap analysis
    overlap = set(strategy1_markets) & set(strategy2_markets)
    s1_only = set(strategy1_markets) - set(strategy2_markets)
    s2_only = set(strategy2_markets) - set(strategy1_markets)

    print(f"\nMarket Overlap:")
    print(f"  Fade YES >85c only: {len(s1_only)} markets")
    print(f"  S007 only: {len(s2_only)} markets")
    print(f"  Both: {len(overlap)} markets")
    print(f"  Total Fade YES >85c: {len(strategy1_markets)}")
    print(f"  Total S007: {len(strategy2_markets)}")

    overlap_pct = len(overlap) / len(strategy1_markets) * 100
    print(f"\n  Overlap: {overlap_pct:.1f}% of Fade YES >85c is also in S007")

    # Correlation between YES price and leverage
    yes_trades = df[df['taker_side'] == 'yes'].copy()

    correlation = yes_trades['trade_price'].corr(yes_trades['leverage_ratio'])
    print(f"\n  Correlation between YES price and leverage: {correlation:.3f}")

    # Edge comparison for overlapping vs non-overlapping markets
    print("\n" + "="*80)
    print("EDGE ANALYSIS BY SEGMENT")
    print("="*80)

    market_outcomes = df.groupby('market_ticker')['market_result'].first()

    def calculate_edge(market_set, df, signal_name):
        """Calculate edge for a set of markets"""
        if len(market_set) < 50:
            return None

        # Get relevant trades
        subset = df[df['market_ticker'].isin(market_set)]

        # Calculate avg NO price (when YES is traded, NO = 100 - YES)
        yes_trades = subset[subset['taker_side'] == 'yes']
        avg_no_price = (100 - yes_trades.groupby('market_ticker')['trade_price'].mean()).mean()

        # Get outcomes
        outcomes = [market_outcomes.get(m) for m in market_set if m in market_outcomes]
        no_wins = sum(1 for o in outcomes if o == 'no')
        n = len(outcomes)

        if n == 0:
            return None

        win_rate = no_wins / n
        breakeven = avg_no_price / 100

        return {
            'signal': signal_name,
            'n_markets': n,
            'no_win_rate': win_rate,
            'avg_no_price': avg_no_price,
            'breakeven': breakeven,
            'edge': win_rate - breakeven
        }

    results = []

    # Fade YES >85c only (not in S007)
    r = calculate_edge(s1_only, df, "Fade YES >85c ONLY")
    if r:
        results.append(r)
        print(f"\n{r['signal']}:")
        print(f"  Markets: {r['n_markets']}")
        print(f"  NO Win Rate: {r['no_win_rate']:.1%}")
        print(f"  Avg NO Price: {r['avg_no_price']:.1f}c")
        print(f"  Edge: {r['edge']*100:.2f}%")

    # S007 only (not in Fade YES >85c)
    r = calculate_edge(s2_only, df, "S007 ONLY")
    if r:
        results.append(r)
        print(f"\n{r['signal']}:")
        print(f"  Markets: {r['n_markets']}")
        print(f"  NO Win Rate: {r['no_win_rate']:.1%}")
        print(f"  Avg NO Price: {r['avg_no_price']:.1f}c")
        print(f"  Edge: {r['edge']*100:.2f}%")

    # Overlap
    r = calculate_edge(overlap, df, "OVERLAP (both signals)")
    if r:
        results.append(r)
        print(f"\n{r['signal']}:")
        print(f"  Markets: {r['n_markets']}")
        print(f"  NO Win Rate: {r['no_win_rate']:.1%}")
        print(f"  Avg NO Price: {r['avg_no_price']:.1f}c")
        print(f"  Edge: {r['edge']*100:.2f}%")

    # Full Fade YES >85c
    r = calculate_edge(strategy1_markets, df, "Fade YES >85c (FULL)")
    if r:
        results.append(r)
        print(f"\n{r['signal']}:")
        print(f"  Markets: {r['n_markets']}")
        print(f"  NO Win Rate: {r['no_win_rate']:.1%}")
        print(f"  Avg NO Price: {r['avg_no_price']:.1f}c")
        print(f"  Edge: {r['edge']*100:.2f}%")

    # Full S007
    r = calculate_edge(strategy2_markets, df, "S007 (FULL)")
    if r:
        results.append(r)
        print(f"\n{r['signal']}:")
        print(f"  Markets: {r['n_markets']}")
        print(f"  NO Win Rate: {r['no_win_rate']:.1%}")
        print(f"  Avg NO Price: {r['avg_no_price']:.1f}c")
        print(f"  Edge: {r['edge']*100:.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Check if they're essentially the same
    if overlap_pct > 80:
        print("\nThese strategies are essentially the SAME (>80% overlap).")
        print("Fade YES >85c is just another formulation of S007.")
        print("The high leverage comes from high YES price (cheap NO).")
    else:
        print(f"\nThese strategies are DIFFERENT ({overlap_pct:.1f}% overlap).")
        print("Each captures some unique markets.")

        # Check which non-overlapping segment has edge
        s1_only_result = [r for r in results if r['signal'] == 'Fade YES >85c ONLY']
        s2_only_result = [r for r in results if r['signal'] == 'S007 ONLY']

        if s1_only_result and s1_only_result[0]['edge'] > 0.02:
            print(f"\nFade YES >85c ONLY has edge: {s1_only_result[0]['edge']*100:.2f}%")
            print("This could be a complementary strategy to S007.")

        if s2_only_result and s2_only_result[0]['edge'] > 0.02:
            print(f"\nS007 ONLY has edge: {s2_only_result[0]['edge']*100:.2f}%")
            print("S007 captures additional markets beyond high-price YES.")


def main():
    print("="*80)
    print("SESSION 012: S007 Comparison Analysis")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    analyze_overlap(df)


if __name__ == "__main__":
    main()
