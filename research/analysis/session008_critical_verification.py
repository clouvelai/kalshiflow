#!/usr/bin/env python3
"""
Session 008: CRITICAL VERIFICATION
Verify that the found strategies aren't just proxies for price-based betting.

Key question: Are the leverage and flow strategies ACTUALLY different from
simple "bet NO at high prices"?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

DATA_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv")

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def verify_leverage_strategy(df):
    """
    CRITICAL: Verify the leverage strategy isn't just "bet NO at high NO prices"
    """
    print("\n" + "="*80)
    print("CRITICAL VERIFICATION: LEVERAGE STRATEGY")
    print("="*80)

    # When we "fade high-leverage YES", we're betting NO in markets where
    # someone bet YES with high leverage (low YES price)

    # Key insight: High leverage YES = low YES price = high NO price
    # So "fade high-leverage YES" might just be "bet NO at high NO prices"

    threshold = 2
    high_lev_yes = df[(df['leverage_ratio'] > threshold) & (df['taker_side'] == 'yes')]

    # What prices are these trades at?
    print(f"\nHigh-leverage YES trades (leverage > {threshold}):")
    print(f"  Count: {len(high_lev_yes):,}")
    print(f"  Mean YES price: {high_lev_yes['yes_price'].mean():.1f}c")
    print(f"  Mean NO price: {high_lev_yes['no_price'].mean():.1f}c")
    print(f"  YES price distribution:")
    print(high_lev_yes['yes_price'].describe())

    # If we bet NO in these markets, what's our cost?
    target_markets = high_lev_yes['market_ticker'].unique()

    # Get the NO prices we'd actually pay
    market_no_prices = df[df['market_ticker'].isin(target_markets)].groupby('market_ticker')['no_price'].mean()

    print(f"\n  Markets targeted: {len(target_markets):,}")
    print(f"  Our average NO cost: {market_no_prices.mean():.1f}c")

    # COMPARISON: What's the baseline for betting NO at similar prices?
    # Get all NO trades at similar price levels
    our_no_price = market_no_prices.mean()
    price_range = (our_no_price - 5, our_no_price + 5)

    baseline_no = df[(df['taker_side'] == 'no') &
                     (df['trade_price'] >= price_range[0]) &
                     (df['trade_price'] < price_range[1])]

    print(f"\nBaseline comparison: NO trades at {price_range[0]:.0f}-{price_range[1]:.0f}c")

    # Market-level stats for baseline
    baseline_markets = baseline_no.groupby('market_ticker').agg({
        'trade_price': 'mean',
        'is_winner': 'first'
    }).reset_index()

    baseline_wr = baseline_markets['is_winner'].mean()
    baseline_be = baseline_markets['trade_price'].mean() / 100
    baseline_edge = (baseline_wr - baseline_be) * 100

    print(f"  Baseline WR: {baseline_wr:.1%}")
    print(f"  Baseline BE: {baseline_be:.1%}")
    print(f"  Baseline Edge: {baseline_edge:+.1f}%")

    # Our strategy stats
    our_markets = df[df['market_ticker'].isin(target_markets)].groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    our_wr = (our_markets['market_result'] == 'no').mean()
    our_be = our_markets['no_price'].mean() / 100
    our_edge = (our_wr - our_be) * 100

    print(f"\nOur Strategy (fade high-lev YES):")
    print(f"  Our WR: {our_wr:.1%}")
    print(f"  Our BE: {our_be:.1%}")
    print(f"  Our Edge: {our_edge:+.1f}%")

    # Key comparison: Is our edge BETTER than baseline at same prices?
    edge_improvement = our_edge - baseline_edge
    print(f"\nEDGE IMPROVEMENT over baseline: {edge_improvement:+.1f}%")

    if edge_improvement > 2:
        print("  -> REAL SIGNAL: Our strategy adds value beyond price!")
    elif edge_improvement > 0:
        print("  -> MARGINAL: Slight improvement over price-only")
    else:
        print("  -> PROXY: Strategy is just a proxy for price, no additional value")

    return {
        'our_edge': our_edge,
        'baseline_edge': baseline_edge,
        'improvement': edge_improvement
    }


def verify_flow_strategy(df):
    """
    CRITICAL: Verify the flow strategy isn't just "bet the direction with higher price"
    """
    print("\n" + "="*80)
    print("CRITICAL VERIFICATION: ORDER FLOW STRATEGY")
    print("="*80)

    # Get markets with enough trades
    market_trade_counts = df.groupby('market_ticker').size()
    active_markets = market_trade_counts[market_trade_counts >= 10].index.tolist()
    df_active = df[df['market_ticker'].isin(active_markets)].copy()

    # Calculate imbalance shift
    def calc_metrics(group):
        group = group.sort_values('datetime')

        yes_flow = (group['taker_side'] == 'yes').astype(int) * group['cost_dollars']
        no_flow = (group['taker_side'] == 'no').astype(int) * group['cost_dollars']

        cum_yes = yes_flow.cumsum()
        cum_no = no_flow.cumsum()
        total = cum_yes + cum_no

        imbalance = (cum_yes - cum_no) / total.replace(0, 1)

        n = len(group)
        early_imb = imbalance.iloc[:max(1, n//3)].mean() if n > 0 else 0
        late_imb = imbalance.iloc[-max(1, n//3):].mean() if n > 0 else 0

        return pd.Series({
            'imbalance_shift': late_imb - early_imb,
            'market_result': group['market_result'].iloc[0],
            'final_yes_price': group['yes_price'].iloc[-1],
            'final_no_price': group['no_price'].iloc[-1],
            'avg_yes_price': group['yes_price'].mean(),
            'avg_no_price': group['no_price'].mean()
        })

    print("Calculating flow metrics...")
    flow_metrics = df_active.groupby('market_ticker', group_keys=False).apply(calc_metrics)
    flow_metrics = flow_metrics.reset_index()

    # For markets with flow shift toward NO
    threshold = 0.3
    shift_no = flow_metrics[flow_metrics['imbalance_shift'] < -threshold]
    shift_yes = flow_metrics[flow_metrics['imbalance_shift'] > threshold]

    print(f"\nMarkets with flow shift > {threshold} toward NO: {len(shift_no)}")
    print(f"Markets with flow shift > {threshold} toward YES: {len(shift_yes)}")

    # Key question: What are the PRICES in these markets?
    # If flow shifts toward NO, do prices also favor NO?

    print("\n--- Flow Shift Toward NO ---")
    print(f"  Avg final YES price: {shift_no['final_yes_price'].mean():.1f}c")
    print(f"  Avg final NO price: {shift_no['final_no_price'].mean():.1f}c")

    # We bet NO when flow shifts to NO
    # Our cost = final NO price
    our_cost = shift_no['final_no_price'].mean()
    we_win = (shift_no['market_result'] == 'no').mean()
    our_be = our_cost / 100
    our_edge = (we_win - our_be) * 100

    print(f"  Our WR: {we_win:.1%}")
    print(f"  Our BE (from NO price): {our_be:.1%}")
    print(f"  Our Edge: {our_edge:+.1f}%")

    # COMPARISON: What if we just bet NO at similar prices WITHOUT flow signal?
    all_markets = df_active.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    price_range = (our_cost - 5, our_cost + 5)
    baseline = all_markets[(all_markets['no_price'] >= price_range[0]) &
                           (all_markets['no_price'] < price_range[1])]

    baseline_wr = (baseline['market_result'] == 'no').mean()
    baseline_be = baseline['no_price'].mean() / 100
    baseline_edge = (baseline_wr - baseline_be) * 100

    print(f"\nBaseline: Bet NO at {price_range[0]:.0f}-{price_range[1]:.0f}c (no flow signal)")
    print(f"  Baseline WR: {baseline_wr:.1%}")
    print(f"  Baseline BE: {baseline_be:.1%}")
    print(f"  Baseline Edge: {baseline_edge:+.1f}%")

    edge_improvement = our_edge - baseline_edge
    print(f"\nEDGE IMPROVEMENT over baseline: {edge_improvement:+.1f}%")

    if edge_improvement > 2:
        print("  -> REAL SIGNAL: Flow adds value beyond price!")
    elif edge_improvement > 0:
        print("  -> MARGINAL: Slight improvement over price-only")
    else:
        print("  -> PROXY: Strategy is just a proxy for price, no additional value")

    # Also check the YES direction
    print("\n--- Flow Shift Toward YES ---")
    print(f"  Avg final YES price: {shift_yes['final_yes_price'].mean():.1f}c")
    print(f"  Avg final NO price: {shift_yes['final_no_price'].mean():.1f}c")

    yes_cost = shift_yes['final_yes_price'].mean()
    yes_win = (shift_yes['market_result'] == 'yes').mean()
    yes_be = yes_cost / 100
    yes_edge = (yes_win - yes_be) * 100

    print(f"  Our WR (bet YES): {yes_win:.1%}")
    print(f"  Our BE: {yes_be:.1%}")
    print(f"  Our Edge: {yes_edge:+.1f}%")

    return {
        'flow_no_edge': our_edge,
        'baseline_no_edge': baseline_edge,
        'no_improvement': edge_improvement,
        'flow_yes_edge': yes_edge
    }


def analyze_what_flow_actually_measures(df):
    """
    Deep dive: What does flow shift ACTUALLY correlate with?
    """
    print("\n" + "="*80)
    print("WHAT DOES FLOW SHIFT ACTUALLY MEASURE?")
    print("="*80)

    # Get markets with enough trades
    market_trade_counts = df.groupby('market_ticker').size()
    active_markets = market_trade_counts[market_trade_counts >= 10].index.tolist()
    df_active = df[df['market_ticker'].isin(active_markets)].copy()

    # Calculate flow shift and price change
    def calc_all_metrics(group):
        group = group.sort_values('datetime')

        yes_flow = (group['taker_side'] == 'yes').astype(int) * group['cost_dollars']
        no_flow = (group['taker_side'] == 'no').astype(int) * group['cost_dollars']

        cum_yes = yes_flow.cumsum()
        cum_no = no_flow.cumsum()
        total = cum_yes + cum_no

        imbalance = (cum_yes - cum_no) / total.replace(0, 1)

        n = len(group)
        early_imb = imbalance.iloc[:max(1, n//3)].mean() if n > 0 else 0
        late_imb = imbalance.iloc[-max(1, n//3):].mean() if n > 0 else 0

        # Also track price changes
        early_yes_price = group['yes_price'].iloc[:max(1, n//3)].mean()
        late_yes_price = group['yes_price'].iloc[-max(1, n//3):].mean()

        return pd.Series({
            'imbalance_shift': late_imb - early_imb,
            'price_change': late_yes_price - early_yes_price,
            'market_result': group['market_result'].iloc[0],
            'final_yes_price': group['yes_price'].iloc[-1],
            'n_trades': len(group)
        })

    print("Calculating comprehensive metrics...")
    metrics = df_active.groupby('market_ticker', group_keys=False).apply(calc_all_metrics)
    metrics = metrics.reset_index()

    # Correlation between flow shift and price change
    corr = metrics['imbalance_shift'].corr(metrics['price_change'])
    print(f"\nCorrelation: flow_shift vs price_change = {corr:.3f}")

    # If correlation is high, they're measuring the same thing
    if abs(corr) > 0.7:
        print("  -> HIGH CORRELATION: Flow shift is just a price change proxy!")
    elif abs(corr) > 0.4:
        print("  -> MODERATE CORRELATION: Some overlap with price")
    else:
        print("  -> LOW CORRELATION: Flow shift captures something different!")

    # Does flow shift predict outcome BEYOND what price predicts?
    # Run a simple analysis: control for price level

    # Bin by final price
    metrics['price_bucket'] = pd.cut(metrics['final_yes_price'],
                                     bins=[0, 30, 50, 70, 100],
                                     labels=['<30', '30-50', '50-70', '70+'])

    print("\n--- Flow Edge by Price Bucket ---")
    for bucket in ['<30', '30-50', '50-70', '70+']:
        bucket_data = metrics[metrics['price_bucket'] == bucket]
        if len(bucket_data) < 50:
            continue

        # Within this price bucket, does flow shift predict?
        high_flow_no = bucket_data[bucket_data['imbalance_shift'] < -0.3]
        low_flow_no = bucket_data[bucket_data['imbalance_shift'] > -0.3]

        if len(high_flow_no) >= 20 and len(low_flow_no) >= 20:
            hf_no_wins = (high_flow_no['market_result'] == 'no').mean()
            lf_no_wins = (low_flow_no['market_result'] == 'no').mean()

            print(f"\n  Price bucket {bucket} (N={len(bucket_data)}):")
            print(f"    High flow toward NO: {len(high_flow_no)} markets, NO wins {hf_no_wins:.1%}")
            print(f"    Low flow toward NO: {len(low_flow_no)} markets, NO wins {lf_no_wins:.1%}")
            print(f"    Difference: {(hf_no_wins - lf_no_wins)*100:+.1f}%")


def main():
    print("="*80)
    print("SESSION 008: CRITICAL VERIFICATION")
    print("Are these strategies real or just price proxies?")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    df = load_data()

    lev_results = verify_leverage_strategy(df)
    flow_results = verify_flow_strategy(df)
    analyze_what_flow_actually_measures(df)

    # Final summary
    print("\n" + "="*80)
    print("FINAL CRITICAL ASSESSMENT")
    print("="*80)

    print("\nH065 (Leverage Strategy):")
    print(f"  Edge improvement over baseline: {lev_results['improvement']:+.1f}%")
    if lev_results['improvement'] > 2:
        print("  VERDICT: REAL SIGNAL - Worth implementing")
    else:
        print("  VERDICT: PRICE PROXY - Not worth implementing")

    print("\nH052 (Flow Strategy):")
    print(f"  Edge improvement over baseline: {flow_results['no_improvement']:+.1f}%")
    if flow_results['no_improvement'] > 2:
        print("  VERDICT: REAL SIGNAL - Worth implementing")
    else:
        print("  VERDICT: PRICE PROXY - Not worth implementing")


if __name__ == '__main__':
    main()
