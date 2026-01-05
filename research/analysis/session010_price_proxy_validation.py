#!/usr/bin/env python3
"""
Session 010 Part 2: CRITICAL Price Proxy Validation

The initial tests showed 4 "validated" hypotheses with positive edge and Bonferroni significance:
1. H070 v2: Drunk Sports + High Leverage - +3.5% edge (BUT ALREADY FLAGGED AS PRICE PROXY!)
2. H078 low_mid: Fade high leverage at 30-50c - +6.5% edge
3. H084: Fade in markets with increasing leverage - +1.9% edge
4. H072: Fade recent move in high volatility - +33.0% edge (SUSPICIOUS - too high!)

CRITICAL: We must verify each is NOT just a price proxy before declaring victory.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
from pathlib import Path
import pytz

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades/enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets/market_outcomes_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

ET = pytz.timezone('America/New_York')


def load_data():
    """Load trades data."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE, low_memory=False)
    print(f"Loaded {len(trades):,} trades")
    return trades


def price_proxy_check(signal_trades, all_trades, signal_name, fade=True):
    """
    CRITICAL: Check if signal is just a price proxy.

    Compare:
    1. Signal edge at the observed prices
    2. Baseline edge at the SAME price distribution (random sample from all trades)

    If signal_edge - baseline_edge <= 0, the signal is just a price proxy.
    """
    print(f"\n{'='*60}")
    print(f"PRICE PROXY CHECK: {signal_name}")
    print('='*60)

    if len(signal_trades) == 0:
        print("  No signal trades!")
        return {'is_proxy': True, 'reason': 'no_trades'}

    # Calculate signal edge
    if fade:
        signal_trades = signal_trades.copy()
        signal_trades['fade_price'] = 100 - signal_trades['trade_price']
        signal_trades['fade_wins'] = ~signal_trades['is_winner']
        price_col = 'fade_price'
        win_col = 'fade_wins'
    else:
        price_col = 'trade_price'
        win_col = 'is_winner'

    # Market-level aggregation for signal
    signal_market = signal_trades.groupby('market_ticker').agg({
        win_col: 'mean',
        price_col: 'mean'
    }).reset_index()

    n_signal_markets = len(signal_market)
    signal_win_rate = signal_market[win_col].mean()
    signal_avg_price = signal_market[price_col].mean()
    signal_breakeven = signal_avg_price / 100.0
    signal_edge = (signal_win_rate - signal_breakeven) * 100

    print(f"\nSignal Statistics:")
    print(f"  Markets: {n_signal_markets}")
    print(f"  Win Rate: {signal_win_rate*100:.1f}%")
    print(f"  Avg Price: {signal_avg_price:.1f}c")
    print(f"  Breakeven: {signal_breakeven*100:.1f}%")
    print(f"  Edge: {signal_edge:+.1f}%")

    # Get price distribution of signal trades
    signal_prices = signal_trades[price_col].values
    price_min, price_max = signal_prices.min(), signal_prices.max()
    price_mean = signal_prices.mean()
    price_std = signal_prices.std()

    print(f"\nPrice Distribution:")
    print(f"  Range: {price_min:.1f}c - {price_max:.1f}c")
    print(f"  Mean: {price_mean:.1f}c, Std: {price_std:.1f}c")

    # Build baseline: ALL trades at the same price range (ANY time, ANY condition)
    if fade:
        all_trades = all_trades.copy()
        all_trades['fade_price'] = 100 - all_trades['trade_price']
        all_trades['fade_wins'] = ~all_trades['is_winner']

    # Filter to same price range
    baseline_trades = all_trades[
        (all_trades[price_col] >= price_min) &
        (all_trades[price_col] <= price_max)
    ].copy()

    print(f"\nBaseline (all trades at same prices):")
    print(f"  Trades: {len(baseline_trades):,}")

    if len(baseline_trades) == 0:
        print("  No baseline trades at this price range!")
        return {'is_proxy': False, 'reason': 'no_baseline', 'signal_edge': signal_edge}

    # Market-level aggregation for baseline
    baseline_market = baseline_trades.groupby('market_ticker').agg({
        win_col: 'mean',
        price_col: 'mean'
    }).reset_index()

    n_baseline_markets = len(baseline_market)
    baseline_win_rate = baseline_market[win_col].mean()
    baseline_avg_price = baseline_market[price_col].mean()
    baseline_breakeven = baseline_avg_price / 100.0
    baseline_edge = (baseline_win_rate - baseline_breakeven) * 100

    print(f"  Markets: {n_baseline_markets}")
    print(f"  Win Rate: {baseline_win_rate*100:.1f}%")
    print(f"  Avg Price: {baseline_avg_price:.1f}c")
    print(f"  Breakeven: {baseline_breakeven*100:.1f}%")
    print(f"  Edge: {baseline_edge:+.1f}%")

    # Calculate improvement
    improvement = signal_edge - baseline_edge
    is_proxy = improvement <= 0

    print(f"\n{'!'*60}")
    print(f"RESULT: Signal Edge = {signal_edge:+.1f}%, Baseline = {baseline_edge:+.1f}%")
    print(f"IMPROVEMENT: {improvement:+.1f}%")
    print(f"IS PRICE PROXY: {is_proxy}")
    print('!'*60)

    return {
        'is_proxy': is_proxy,
        'signal_edge': signal_edge,
        'baseline_edge': baseline_edge,
        'improvement': improvement,
        'n_signal_markets': n_signal_markets,
        'n_baseline_markets': n_baseline_markets,
        'price_range': [price_min, price_max],
        'signal_win_rate': signal_win_rate * 100,
        'baseline_win_rate': baseline_win_rate * 100
    }


def test_h078_low_mid(trades):
    """
    H078: Fade high leverage trades at 30-50c.

    Initial test showed +6.5% edge. Let's verify it's not a price proxy.
    """
    # Signal: high leverage (>2) trades at 30-50c
    signal_trades = trades[
        (trades['leverage_ratio'] > 2) &
        (trades['trade_price'] >= 30) &
        (trades['trade_price'] <= 50)
    ]

    return price_proxy_check(signal_trades, trades, "H078: Fade High-Lev at 30-50c", fade=True)


def test_h084_increasing_leverage(trades):
    """
    H084: Fade late trades in markets with increasing leverage.

    Initial test showed +1.9% edge. Let's verify it's not a price proxy.
    """
    print("\n" + "="*60)
    print("H084: FADE IN INCREASING LEVERAGE MARKETS")
    print("="*60)

    # Calculate leverage trend per market
    market_trends = []
    market_last_trades = []

    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 10:
            continue

        group = group.sort_values('timestamp')

        mid = len(group) // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]

        first_lev = first_half['leverage_ratio'].mean()
        second_lev = second_half['leverage_ratio'].mean()
        lev_change = second_lev - first_lev

        if lev_change > 0:  # Increasing leverage
            # Get the last trade
            last_trade = group.iloc[-1].copy()
            market_last_trades.append(last_trade)

    if len(market_last_trades) == 0:
        print("No markets with increasing leverage!")
        return {'is_proxy': True, 'reason': 'no_trades'}

    signal_trades = pd.DataFrame(market_last_trades)
    print(f"Signal trades (last trade in inc-lev markets): {len(signal_trades)}")

    # For the baseline, we need ALL last trades from ALL markets (regardless of leverage trend)
    all_last_trades = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 10:
            continue
        group = group.sort_values('timestamp')
        last_trade = group.iloc[-1].copy()
        all_last_trades.append(last_trade)

    all_last_trades = pd.DataFrame(all_last_trades)
    print(f"All last trades: {len(all_last_trades)}")

    return price_proxy_check(signal_trades, all_last_trades, "H084: Fade Last Trade in Inc-Lev Markets", fade=True)


def test_h072_fade_volatility(trades):
    """
    H072: Fade recent move in high-volatility markets.

    Initial test showed +33.0% edge - HIGHLY SUSPICIOUS, probably methodological error.
    """
    print("\n" + "="*60)
    print("H072: FADE RECENT MOVE IN HIGH VOLATILITY MARKETS")
    print("="*60)
    print("NOTE: +33% edge is highly suspicious - likely methodological error")

    # Calculate volatility and last move per market
    market_data = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 5:
            continue

        group = group.sort_values('timestamp')
        price_std = group['trade_price'].std()
        first_3 = group.head(3)['trade_price'].mean()
        last_3 = group.tail(3)['trade_price'].mean()
        major_move_up = last_3 > first_3
        result = group['market_result'].iloc[0]
        last_price = group.iloc[-1]['trade_price']

        market_data.append({
            'market_ticker': ticker,
            'price_std': price_std,
            'major_move_up': major_move_up,
            'market_result': result,
            'last_price': last_price
        })

    market_df = pd.DataFrame(market_data)
    median_vol = market_df['price_std'].median()
    high_vol = market_df[market_df['price_std'] > median_vol].copy()

    print(f"High volatility markets: {len(high_vol)}")

    # Calculate edge for fading
    # Fade UP = bet NO at (100 - last_price), win if result='no'
    # Fade DOWN = bet YES at last_price, win if result='yes'
    high_vol['fade_wins'] = (
        (~high_vol['major_move_up'] & (high_vol['market_result'] == 'yes')) |
        (high_vol['major_move_up'] & (high_vol['market_result'] == 'no'))
    )
    high_vol['fade_price'] = np.where(
        high_vol['major_move_up'],
        100 - high_vol['last_price'],  # Fade UP = bet NO
        high_vol['last_price']          # Fade DOWN = bet YES
    )

    n_markets = len(high_vol)
    avg_win_rate = high_vol['fade_wins'].mean()
    avg_price = high_vol['fade_price'].mean()
    breakeven = avg_price / 100.0
    signal_edge = (avg_win_rate - breakeven) * 100

    print(f"\nSignal Edge Calculation:")
    print(f"  Markets: {n_markets}")
    print(f"  Win Rate: {avg_win_rate*100:.1f}%")
    print(f"  Avg Fade Price: {avg_price:.1f}c")
    print(f"  Breakeven: {breakeven*100:.1f}%")
    print(f"  Edge: {signal_edge:+.1f}%")

    # For baseline, we need to compare to: what if we faded ALL markets (not just high vol)?
    all_vol = market_df.copy()
    all_vol['fade_wins'] = (
        (~all_vol['major_move_up'] & (all_vol['market_result'] == 'yes')) |
        (all_vol['major_move_up'] & (all_vol['market_result'] == 'no'))
    )
    all_vol['fade_price'] = np.where(
        all_vol['major_move_up'],
        100 - all_vol['last_price'],
        all_vol['last_price']
    )

    # Filter to same price range as high_vol
    price_min = high_vol['fade_price'].min()
    price_max = high_vol['fade_price'].max()
    baseline = all_vol[
        (all_vol['fade_price'] >= price_min) &
        (all_vol['fade_price'] <= price_max)
    ]

    baseline_win_rate = baseline['fade_wins'].mean()
    baseline_avg_price = baseline['fade_price'].mean()
    baseline_breakeven = baseline_avg_price / 100.0
    baseline_edge = (baseline_win_rate - baseline_breakeven) * 100

    print(f"\nBaseline (fade recent move in ALL markets, same prices):")
    print(f"  Markets: {len(baseline)}")
    print(f"  Win Rate: {baseline_win_rate*100:.1f}%")
    print(f"  Avg Price: {baseline_avg_price:.1f}c")
    print(f"  Edge: {baseline_edge:+.1f}%")

    improvement = signal_edge - baseline_edge

    print(f"\n{'!'*60}")
    print(f"RESULT: Signal Edge = {signal_edge:+.1f}%, Baseline = {baseline_edge:+.1f}%")
    print(f"IMPROVEMENT: {improvement:+.1f}%")
    print(f"IS PRICE PROXY: {improvement <= 0}")
    print('!'*60)

    # DEEPER CHECK: The issue might be that "fade recent move" itself has edge
    # Let's check if just "fade recent move" has edge regardless of volatility
    print(f"\nDEEPER CHECK: Does 'fade recent move' have edge in ALL markets?")
    all_fade_win_rate = all_vol['fade_wins'].mean()
    all_fade_price = all_vol['fade_price'].mean()
    all_fade_breakeven = all_fade_price / 100.0
    all_fade_edge = (all_fade_win_rate - all_fade_breakeven) * 100

    print(f"  All markets fade edge: {all_fade_edge:+.1f}%")
    print(f"  High-vol specific improvement: {signal_edge - all_fade_edge:+.1f}%")

    return {
        'signal_edge': signal_edge,
        'baseline_edge': baseline_edge,
        'improvement': improvement,
        'is_proxy': improvement <= 0,
        'all_fade_edge': all_fade_edge,
        'high_vol_improvement': signal_edge - all_fade_edge
    }


def test_h070_drunk_high_lev(trades):
    """
    H070: Drunk sports betting with high leverage.

    Already flagged as price proxy in main test (-0.2% improvement).
    Let's double-check with different methodology.
    """
    print("\n" + "="*60)
    print("H070: DRUNK SPORTS BETTING (HIGH LEVERAGE)")
    print("="*60)
    print("NOTE: Already flagged as price proxy in main test")

    SPORTS_CATEGORIES = ['KXNFL', 'KXNCAAF', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAMB', 'KXSOC']

    # Parse timestamps
    trades['datetime_et'] = pd.to_datetime(trades['timestamp'], unit='ms', utc=True).dt.tz_convert(ET)
    trades['hour_et'] = trades['datetime_et'].dt.hour
    trades['day_of_week'] = trades['datetime_et'].dt.dayofweek

    # Identify sports
    trades['is_sports'] = trades['market_ticker'].apply(
        lambda x: any(cat in x for cat in SPORTS_CATEGORIES)
    )

    # Late night weekend sports + high leverage
    late_night_hours = [23, 0, 1, 2, 3]
    weekend_days = [4, 5]

    signal_trades = trades[
        trades['is_sports'] &
        trades['hour_et'].isin(late_night_hours) &
        trades['day_of_week'].isin(weekend_days) &
        (trades['leverage_ratio'] > 3)
    ].copy()

    print(f"Signal trades: {len(signal_trades)}")

    # Compare to ALL sports trades with high leverage (any time)
    baseline_trades = trades[
        trades['is_sports'] &
        (trades['leverage_ratio'] > 3)
    ].copy()

    print(f"Baseline (all sports, high lev, any time): {len(baseline_trades)}")

    return price_proxy_check(signal_trades, baseline_trades, "H070: Drunk Sports High-Lev", fade=True)


def main():
    """Run price proxy validation for all "validated" hypotheses."""
    print("="*80)
    print("SESSION 010: CRITICAL PRICE PROXY VALIDATION")
    print("="*80)
    print("Testing whether 'validated' strategies are just price proxies")
    print()

    trades = load_data()

    results = {}

    # Test H078: Fade high leverage at 30-50c
    results['H078'] = test_h078_low_mid(trades)

    # Test H084: Fade in increasing leverage markets
    results['H084'] = test_h084_increasing_leverage(trades)

    # Test H072: Fade recent move in high volatility
    results['H072'] = test_h072_fade_volatility(trades)

    # Test H070: Drunk sports betting (double-check)
    results['H070'] = test_h070_drunk_high_lev(trades)

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for hyp, res in results.items():
        is_proxy = res.get('is_proxy', True)
        signal_edge = res.get('signal_edge', 0)
        improvement = res.get('improvement', 0)

        if is_proxy:
            status = "REJECTED - PRICE PROXY"
        else:
            status = "POTENTIAL EDGE"

        print(f"\n{hyp}:")
        print(f"  Status: {status}")
        print(f"  Signal Edge: {signal_edge:+.1f}%")
        print(f"  Improvement over baseline: {improvement:+.1f}%")

    # Save results
    output_file = REPORTS_DIR / f"session010_price_proxy_check_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_types({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
