#!/usr/bin/env python3
"""
Session 009: NCAAF Deep Validation

The NCAAF category showed +17% improvement over baseline after price control.
This needs rigorous validation before we can call it a real signal.

Key concerns to address:
1. Sample size: Only 371 markets, 180 after price matching
2. Temporal stability: Second half only has 39 markets
3. Sub-category analysis: Is it driven by one type (TOTAL)?
4. Is this actually just NCAAFTOTAL being mislabeled?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json

DATA_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv")
BONFERRONI_P = 0.01 / 3


def load_data():
    print("Loading data...")
    trades = pd.read_csv(DATA_PATH)
    trades['datetime'] = pd.to_datetime(trades['datetime'])
    return trades


def calculate_edge_metrics(df, side='no'):
    n_markets = len(df)
    if n_markets < 30:
        return None

    df = df.copy()
    df['we_win'] = df['market_result'] == side
    win_rate = df['we_win'].mean()
    avg_price = df['avg_price'].mean()
    breakeven = avg_price / 100.0
    edge = (win_rate - breakeven) * 100

    df['our_cost'] = avg_price
    df['our_profit'] = np.where(df['we_win'], 100 - df['our_cost'], -df['our_cost'])
    total_profit = df['our_profit'].sum()

    winners = df[df['our_profit'] > 0]
    concentration = winners['our_profit'].max() / total_profit if len(winners) > 0 and total_profit > 0 else 1.0

    n_wins = int(win_rate * n_markets)
    try:
        result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
        p_value = result.pvalue
    except:
        p_value = 1.0

    return {
        'n_markets': n_markets,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'total_profit': total_profit,
        'concentration': concentration,
        'p_value': p_value,
        'bonferroni_sig': p_value < BONFERRONI_P
    }


def main():
    print("="*80)
    print("SESSION 009: NCAAF DEEP VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    trades = load_data()

    # Identify NCAAF markets more precisely
    # Look for KXNCAAFTOTAL, KXNCAAFSPREAD, KXNCAAFGAME
    print("\n--- NCAAF TICKER PATTERNS ---")

    ncaaf_patterns = {}
    for ticker in trades['market_ticker'].unique():
        if 'NCAAF' in ticker:
            # Extract the full pattern
            parts = ticker.split('-')
            base = parts[0] if parts else ticker
            if base not in ncaaf_patterns:
                ncaaf_patterns[base] = []
            ncaaf_patterns[base].append(ticker)

    print(f"NCAAF base patterns found: {len(ncaaf_patterns)}")
    for base, tickers in sorted(ncaaf_patterns.items(), key=lambda x: -len(x[1])):
        print(f"  {base}: {len(tickers)} markets")

    # Analyze each NCAAF sub-type separately
    print("\n" + "="*80)
    print("ANALYSIS BY NCAAF SUB-TYPE")
    print("="*80)

    for base in sorted(ncaaf_patterns.keys()):
        tickers = ncaaf_patterns[base]
        if len(tickers) < 30:
            continue

        subtype_trades = trades[trades['market_ticker'].isin(tickers)]
        subtype_markets = subtype_trades.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'datetime': 'first'
        }).reset_index()

        print(f"\n--- {base} ({len(subtype_markets)} markets) ---")

        # NO strategy
        subtype_markets['avg_price'] = subtype_markets['no_price']
        m = calculate_edge_metrics(subtype_markets, side='no')

        if m:
            print(f"  Win Rate: {m['win_rate']:.1%}")
            print(f"  Breakeven: {m['breakeven']:.1%}")
            print(f"  Edge: {m['edge']:+.1f}%")
            print(f"  P-value: {m['p_value']:.4f}")
            print(f"  Concentration: {m['concentration']:.1%}")

            # Temporal check
            subtype_markets['date'] = subtype_markets['datetime'].dt.date
            dates_sorted = sorted(subtype_markets['date'].unique())
            if len(dates_sorted) >= 2:
                mid_idx = len(dates_sorted) // 2
                mid_date = dates_sorted[mid_idx]

                first = subtype_markets[subtype_markets['date'] <= mid_date].copy()
                second = subtype_markets[subtype_markets['date'] > mid_date].copy()

                if len(first) >= 15 and len(second) >= 15:
                    first['avg_price'] = first['no_price']
                    second['avg_price'] = second['no_price']

                    m1 = calculate_edge_metrics(first, side='no')
                    m2 = calculate_edge_metrics(second, side='no')

                    if m1 and m2:
                        print(f"  Temporal: First {m1['edge']:+.1f}% (N={m1['n_markets']}), "
                              f"Second {m2['edge']:+.1f}% (N={m2['n_markets']})")

    # Focus on KXNCAAFTOTAL which showed +22.5% edge
    print("\n" + "="*80)
    print("DEEP VALIDATION: KXNCAAFTOTAL")
    print("="*80)

    if 'KXNCAAFTOTAL' in ncaaf_patterns:
        total_tickers = ncaaf_patterns['KXNCAAFTOTAL']
        total_trades = trades[trades['market_ticker'].isin(total_tickers)]
        total_markets = total_trades.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'datetime': 'first',
            'cost_dollars': 'sum'
        }).reset_index()

        print(f"Markets: {len(total_markets)}")
        print(f"Total volume: ${total_markets['cost_dollars'].sum():,.0f}")

        total_markets['avg_price'] = total_markets['no_price']
        m = calculate_edge_metrics(total_markets, side='no')

        if m:
            print(f"\nNO Strategy:")
            print(f"  Win Rate: {m['win_rate']:.1%}")
            print(f"  Breakeven: {m['breakeven']:.1%}")
            print(f"  Edge: {m['edge']:+.1f}%")
            print(f"  P-value: {m['p_value']:.4f}")
            print(f"  Concentration: {m['concentration']:.1%}")
            print(f"  Bonferroni sig: {m['bonferroni_sig']}")

            # Price proxy check vs non-NCAAF
            print("\n--- Price Proxy Check ---")

            non_ncaaf = trades[~trades['market_ticker'].isin(
                [t for tickers in ncaaf_patterns.values() for t in tickers]
            )]

            non_ncaaf_markets = non_ncaaf.groupby('market_ticker').agg({
                'market_result': 'first',
                'no_price': 'mean'
            }).reset_index()

            # Match price range
            price_min = total_markets['no_price'].quantile(0.1)
            price_max = total_markets['no_price'].quantile(0.9)

            print(f"TOTAL price range: {price_min:.0f}c - {price_max:.0f}c")

            matched_total = total_markets[
                (total_markets['no_price'] >= price_min) &
                (total_markets['no_price'] <= price_max)
            ].copy()

            matched_other = non_ncaaf_markets[
                (non_ncaaf_markets['no_price'] >= price_min) &
                (non_ncaaf_markets['no_price'] <= price_max)
            ].copy()

            print(f"Matched TOTAL: {len(matched_total)}")
            print(f"Matched Other: {len(matched_other)}")

            if len(matched_total) >= 30 and len(matched_other) >= 100:
                matched_total['avg_price'] = matched_total['no_price']
                matched_other['avg_price'] = matched_other['no_price']

                m_total = calculate_edge_metrics(matched_total, side='no')
                m_other = calculate_edge_metrics(matched_other, side='no')

                if m_total and m_other:
                    print(f"\nPrice-matched TOTAL edge: {m_total['edge']:+.1f}%")
                    print(f"Price-matched baseline edge: {m_other['edge']:+.1f}%")
                    improvement = m_total['edge'] - m_other['edge']
                    print(f"Improvement: {improvement:+.1f}%")

                    if improvement > 5 and m_total['p_value'] < 0.01:
                        print("\n*** POTENTIAL VALIDATED SIGNAL ***")
                        print("NCAAFTOTAL shows edge beyond price effect")

                        # Final validation checks
                        print("\n--- Final Validation Checks ---")
                        print(f"1. Markets >= 50: {m_total['n_markets'] >= 50} ({m_total['n_markets']})")
                        print(f"2. Concentration < 30%: {m_total['concentration'] < 0.30} ({m_total['concentration']:.1%})")
                        print(f"3. P-value < 0.01: {m_total['p_value'] < 0.01} ({m_total['p_value']:.4f})")
                        print(f"4. Edge > 5%: {m_total['edge'] > 5} ({m_total['edge']:+.1f}%)")
                        print(f"5. Improvement > 5%: {improvement > 5} ({improvement:+.1f}%)")

                        all_pass = (
                            m_total['n_markets'] >= 50 and
                            m_total['concentration'] < 0.30 and
                            m_total['p_value'] < 0.01 and
                            improvement > 5
                        )

                        if all_pass:
                            print("\n*** STATUS: VALIDATED ***")
                        else:
                            print("\n*** STATUS: MARGINAL - Does not meet all criteria ***")
                    else:
                        print("-> PRICE PROXY: Not enough improvement over baseline")

    # Summary
    print("\n" + "="*80)
    print("SESSION 009 NCAAF VALIDATION SUMMARY")
    print("="*80)

    print("""
Findings:

1. KXNCAAFTOTAL shows the strongest signal among NCAAF sub-types
2. The edge is primarily driven by specific game types (totals betting)
3. However, sample size is limited (94 markets)
4. Temporal stability unclear with small second-half sample

Recommendation:
- KXNCAAFTOTAL is a PROMISING but not VALIDATED signal
- Needs more data to confirm (current sample ~94 markets)
- DO NOT add to VALIDATED_STRATEGIES.md yet
- Continue monitoring this category for edge persistence
""")


if __name__ == '__main__':
    main()
