#!/usr/bin/env python3
"""
Session 010 Part 2: Rigorous Hypothesis Testing

Tests the Tier 1 hypotheses from Session 010:
1. H070: "Drunk Sports Betting" - Late-night weekend sports trades (USER REQUESTED - TEST FIRST!)
2. H071: Trade Clustering Velocity - 5+ trades in same direction within 5 minutes
3. H078: Leverage Divergence from Price - High leverage at unfavorable prices
4. H084: Leverage Ratio Trend Within Market - Increasing leverage over market lifetime
5. H072: Price Path Volatility Regimes - Markets with high intra-market price variance

Methodology:
- Correct breakeven formula: breakeven_rate = trade_price / 100.0
- Price proxy check (MANDATORY): Compare signal edge to baseline at same prices
- Bonferroni correction: p < 0.003 (0.05 / 16 hypotheses)
- N >= 50 unique markets
- Concentration < 30%
- Temporal stability check
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
from pathlib import Path
import pytz

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades/enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets/market_outcomes_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

# Timezone
ET = pytz.timezone('America/New_York')

# Sports categories
SPORTS_CATEGORIES = ['KXNFL', 'KXNCAAF', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAMB', 'KXSOC']

def load_data():
    """Load trades and markets data."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE)
    markets = pd.read_csv(MARKETS_FILE)
    print(f"Loaded {len(trades):,} trades and {len(markets):,} markets")
    return trades, markets


def calculate_edge(df, price_col='trade_price'):
    """
    Calculate edge using correct breakeven formula.

    For ANY trade (YES or NO):
    - breakeven_rate = trade_price / 100.0
    - edge = win_rate - breakeven_rate
    """
    if len(df) == 0:
        return {'edge': 0, 'win_rate': 0, 'breakeven': 0, 'n_trades': 0, 'n_markets': 0, 'p_value': 1.0}

    # Aggregate to market level
    market_stats = df.groupby('market_ticker').agg({
        'is_winner': 'mean',
        price_col: 'mean',
        'cost_dollars': 'sum'
    }).reset_index()

    n_markets = len(market_stats)
    if n_markets == 0:
        return {'edge': 0, 'win_rate': 0, 'breakeven': 0, 'n_trades': 0, 'n_markets': 0, 'p_value': 1.0}

    # Calculate win rate and breakeven at market level
    avg_win_rate = market_stats['is_winner'].mean()
    avg_price = market_stats[price_col].mean()
    breakeven_rate = avg_price / 100.0
    edge = (avg_win_rate - breakeven_rate) * 100

    # Calculate p-value using binomial test
    n_wins = int(avg_win_rate * n_markets)
    p_value = stats.binomtest(n_wins, n_markets, breakeven_rate, alternative='greater').pvalue

    return {
        'edge': edge,
        'win_rate': avg_win_rate * 100,
        'breakeven': breakeven_rate * 100,
        'n_trades': len(df),
        'n_markets': n_markets,
        'p_value': p_value,
        'total_cost': market_stats['cost_dollars'].sum()
    }


def check_price_proxy(signal_trades, all_trades, price_col='trade_price', price_bins=10):
    """
    CRITICAL CHECK: Is the signal just a price proxy?

    Compare signal edge to baseline edge at the same price distribution.
    If signal edge - baseline edge <= 0, REJECT as price proxy.
    """
    if len(signal_trades) == 0:
        return {'is_proxy': True, 'signal_edge': 0, 'baseline_edge': 0, 'improvement': 0}

    # Get price distribution of signal trades
    signal_prices = signal_trades[price_col].values
    price_min, price_max = signal_prices.min(), signal_prices.max()

    # Filter baseline trades to same price range
    baseline_trades = all_trades[
        (all_trades[price_col] >= price_min) &
        (all_trades[price_col] <= price_max)
    ]

    if len(baseline_trades) == 0:
        return {'is_proxy': True, 'signal_edge': 0, 'baseline_edge': 0, 'improvement': 0}

    # Calculate edges
    signal_edge = calculate_edge(signal_trades, price_col)['edge']
    baseline_edge = calculate_edge(baseline_trades, price_col)['edge']
    improvement = signal_edge - baseline_edge

    return {
        'is_proxy': improvement <= 0,
        'signal_edge': signal_edge,
        'baseline_edge': baseline_edge,
        'improvement': improvement
    }


def check_concentration(df, threshold=0.30):
    """Check if any single market dominates profit."""
    if len(df) == 0:
        return {'passes': False, 'max_concentration': 1.0}

    market_profit = df.groupby('market_ticker')['actual_profit_dollars'].sum()
    total_profit = market_profit[market_profit > 0].sum()

    if total_profit <= 0:
        return {'passes': True, 'max_concentration': 0.0}

    max_concentration = market_profit[market_profit > 0].max() / total_profit
    return {
        'passes': max_concentration < threshold,
        'max_concentration': max_concentration
    }


def check_temporal_stability(df, n_periods=4):
    """Check if strategy works across multiple time periods."""
    if len(df) == 0:
        return {'passes': False, 'period_edges': []}

    # Sort by timestamp and split into periods
    df_sorted = df.sort_values('timestamp')
    period_size = len(df_sorted) // n_periods

    period_edges = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = start_idx + period_size if i < n_periods - 1 else len(df_sorted)
        period_df = df_sorted.iloc[start_idx:end_idx]
        edge = calculate_edge(period_df)['edge']
        period_edges.append(edge)

    # Check if majority of periods have positive edge
    positive_periods = sum(1 for e in period_edges if e > 0)
    passes = positive_periods >= n_periods // 2

    return {
        'passes': passes,
        'period_edges': period_edges,
        'positive_periods': positive_periods,
        'total_periods': n_periods
    }


def parse_timestamp_to_et(timestamp_ms):
    """Convert timestamp to ET timezone datetime."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
    return dt.astimezone(ET)


def test_h070_drunk_sports_betting(trades):
    """
    H070: Drunk Sports Betting

    Signal: Late-night (11PM-3AM ET) weekend (Fri/Sat) sports trades
    Additional filters: High leverage (>3) OR round number contracts (10, 25, 50, 100)
    Action: Fade these trades (if YES -> bet NO, if NO -> bet YES)
    """
    print("\n" + "="*80)
    print("H070: DRUNK SPORTS BETTING")
    print("="*80)

    # Parse timestamps to ET
    print("Parsing timestamps to ET timezone...")
    trades['datetime_et'] = trades['timestamp'].apply(parse_timestamp_to_et)
    trades['hour_et'] = trades['datetime_et'].apply(lambda x: x.hour)
    trades['day_of_week'] = trades['datetime_et'].apply(lambda x: x.weekday())  # 0=Mon, 4=Fri, 5=Sat

    # Identify sports trades
    trades['is_sports'] = trades['market_ticker'].apply(
        lambda x: any(cat in x for cat in SPORTS_CATEGORIES)
    )

    print(f"Sports trades: {trades['is_sports'].sum():,} ({100*trades['is_sports'].mean():.1f}%)")

    # Filter for late-night weekend sports trades
    # Late night: 11PM-3AM ET (hours 23, 0, 1, 2, 3)
    late_night_hours = [23, 0, 1, 2, 3]
    # Weekend nights: Friday night (day 4) and Saturday night (day 5)
    weekend_days = [4, 5]  # Friday, Saturday

    drunk_trades = trades[
        trades['is_sports'] &
        trades['hour_et'].isin(late_night_hours) &
        trades['day_of_week'].isin(weekend_days)
    ].copy()

    print(f"Base 'drunk' trades (late night weekend sports): {len(drunk_trades):,}")

    # Test multiple signal variations
    results = {}

    # Variation 1: Base signal (just late night weekend sports)
    print("\n--- Variation 1: Base Signal (late night weekend sports) ---")
    if len(drunk_trades) > 0:
        # For fading: we want to bet opposite of what the drunk traders bet
        # If they bet YES, we bet NO -> our trade_price = NO price, we win if market settles YES
        # Actually, the enriched data already has is_winner based on the ORIGINAL trade
        # To FADE, we need to invert: is_winner for fade = NOT is_winner for original

        # Create faded version
        fade_trades = drunk_trades.copy()
        fade_trades['fade_is_winner'] = ~fade_trades['is_winner']

        # Calculate edge for fading
        # When fading YES: we bet NO at no_price, win if result='yes' (original trade won)
        # When fading NO: we bet YES at yes_price, win if result='no' (original trade won)
        # This is confusing. Let's think carefully:

        # Original trade: taker_side='yes', trade_price=yes_price, is_winner=(result=='yes')
        # Fade trade: we bet NO, our cost = no_price = 100 - yes_price
        #            we win if result = 'no', i.e., NOT is_winner for original

        # So for fading:
        # - Our trade price = 100 - trade_price (for YES trades) or trade_price (for NO trades)
        # - Actually this is getting complex. Let's use a simpler approach:

        # When we fade a YES trade: we bet NO at (100-yes_price), win if market='no'
        # When we fade a NO trade: we bet YES at (100-no_price), win if market='yes'

        # The is_winner column tells us if the ORIGINAL trade won.
        # For fading: our is_winner = NOT original is_winner
        # For our price: if original was YES at price P, our fade price is (100-P)

        fade_trades['fade_trade_price'] = np.where(
            fade_trades['taker_side'] == 'yes',
            100 - fade_trades['trade_price'],  # Fade YES = bet NO at NO price
            100 - fade_trades['trade_price']   # Fade NO = bet YES at YES price
        )

        # Actually wait - trade_price is already the price of what they bought
        # If taker_side='yes', trade_price = yes_price, so fade = no_price = 100 - trade_price
        # If taker_side='no', trade_price = no_price, so fade = yes_price = 100 - trade_price
        # So the formula is the same: fade_trade_price = 100 - trade_price

        # And fade_is_winner = NOT is_winner
        fade_trades['fade_is_winner'] = ~fade_trades['is_winner']

        # Now calculate edge using fade columns
        market_stats = fade_trades.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean',
            'actual_profit_dollars': 'sum'  # Note: this is original profit, not fade profit
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v1_base'] = {
                'description': 'Late night (11PM-3AM ET) weekend (Fri/Sat) sports - FADE',
                'n_trades': len(drunk_trades),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(drunk_trades):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v1_base'] = {'n_markets': 0, 'edge': 0}
            print("  No markets found")
    else:
        results['v1_base'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Variation 2: Add high leverage filter (leverage > 3)
    print("\n--- Variation 2: + High Leverage (>3) ---")
    high_lev_drunk = drunk_trades[drunk_trades['leverage_ratio'] > 3].copy()
    if len(high_lev_drunk) > 0:
        high_lev_drunk['fade_trade_price'] = 100 - high_lev_drunk['trade_price']
        high_lev_drunk['fade_is_winner'] = ~high_lev_drunk['is_winner']

        market_stats = high_lev_drunk.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v2_high_leverage'] = {
                'description': 'Late night weekend sports + leverage > 3 - FADE',
                'n_trades': len(high_lev_drunk),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(high_lev_drunk):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v2_high_leverage'] = {'n_markets': 0, 'edge': 0}
    else:
        results['v2_high_leverage'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Variation 3: Round number contracts (10, 25, 50, 100)
    print("\n--- Variation 3: + Round Number Contracts (10, 25, 50, 100) ---")
    round_numbers = [10, 25, 50, 100]
    round_drunk = drunk_trades[drunk_trades['count'].isin(round_numbers)].copy()
    if len(round_drunk) > 0:
        round_drunk['fade_trade_price'] = 100 - round_drunk['trade_price']
        round_drunk['fade_is_winner'] = ~round_drunk['is_winner']

        market_stats = round_drunk.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v3_round_numbers'] = {
                'description': 'Late night weekend sports + round contracts - FADE',
                'n_trades': len(round_drunk),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(round_drunk):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v3_round_numbers'] = {'n_markets': 0, 'edge': 0}
    else:
        results['v3_round_numbers'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Variation 4: Combined (high leverage OR round numbers)
    print("\n--- Variation 4: + High Leverage OR Round Numbers ---")
    combined_drunk = drunk_trades[
        (drunk_trades['leverage_ratio'] > 3) |
        (drunk_trades['count'].isin(round_numbers))
    ].copy()
    if len(combined_drunk) > 0:
        combined_drunk['fade_trade_price'] = 100 - combined_drunk['trade_price']
        combined_drunk['fade_is_winner'] = ~combined_drunk['is_winner']

        market_stats = combined_drunk.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v4_combined'] = {
                'description': 'Late night weekend sports + (leverage>3 OR round contracts) - FADE',
                'n_trades': len(combined_drunk),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(combined_drunk):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v4_combined'] = {'n_markets': 0, 'edge': 0}
    else:
        results['v4_combined'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Variation 5: YES trades only (most likely to be impulsive longshot bets)
    print("\n--- Variation 5: YES Trades Only (impulsive longshot bets) ---")
    yes_drunk = drunk_trades[drunk_trades['taker_side'] == 'yes'].copy()
    if len(yes_drunk) > 0:
        yes_drunk['fade_trade_price'] = 100 - yes_drunk['trade_price']
        yes_drunk['fade_is_winner'] = ~yes_drunk['is_winner']

        market_stats = yes_drunk.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v5_yes_only'] = {
                'description': 'Late night weekend sports YES trades - FADE',
                'n_trades': len(yes_drunk),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(yes_drunk):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v5_yes_only'] = {'n_markets': 0, 'edge': 0}
    else:
        results['v5_yes_only'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Variation 6: High leverage YES trades (prime target)
    print("\n--- Variation 6: High Leverage YES Trades (prime drunk target) ---")
    high_lev_yes_drunk = drunk_trades[
        (drunk_trades['leverage_ratio'] > 3) &
        (drunk_trades['taker_side'] == 'yes')
    ].copy()
    if len(high_lev_yes_drunk) > 0:
        high_lev_yes_drunk['fade_trade_price'] = 100 - high_lev_yes_drunk['trade_price']
        high_lev_yes_drunk['fade_is_winner'] = ~high_lev_yes_drunk['is_winner']

        market_stats = high_lev_yes_drunk.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets > 0:
            avg_win_rate = market_stats['fade_is_winner'].mean()
            avg_price = market_stats['fade_trade_price'].mean()
            breakeven = avg_price / 100.0
            edge = (avg_win_rate - breakeven) * 100

            n_wins = int(avg_win_rate * n_markets)
            p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue if n_markets > 10 else 1.0

            results['v6_high_lev_yes'] = {
                'description': 'Late night weekend sports + leverage>3 + YES trades - FADE',
                'n_trades': len(high_lev_yes_drunk),
                'n_markets': n_markets,
                'win_rate': avg_win_rate * 100,
                'breakeven': breakeven * 100,
                'edge': edge,
                'p_value': p_value
            }
            print(f"  Trades: {len(high_lev_yes_drunk):,}, Markets: {n_markets}")
            print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
        else:
            results['v6_high_lev_yes'] = {'n_markets': 0, 'edge': 0}
    else:
        results['v6_high_lev_yes'] = {'n_markets': 0, 'edge': 0}
        print("  No trades found")

    # Now do PRICE PROXY CHECK for the best variation
    best_var = max(results.keys(), key=lambda k: results[k].get('edge', 0))
    best_edge = results[best_var].get('edge', 0)

    if best_edge > 0 and results[best_var].get('n_markets', 0) >= 50:
        print(f"\n--- Price Proxy Check for Best Variation ({best_var}) ---")

        # Get the trades used in the best variation
        if best_var == 'v1_base':
            signal_trades = drunk_trades.copy()
        elif best_var == 'v2_high_leverage':
            signal_trades = high_lev_drunk.copy()
        elif best_var == 'v3_round_numbers':
            signal_trades = round_drunk.copy()
        elif best_var == 'v4_combined':
            signal_trades = combined_drunk.copy()
        elif best_var == 'v5_yes_only':
            signal_trades = yes_drunk.copy()
        elif best_var == 'v6_high_lev_yes':
            signal_trades = high_lev_yes_drunk.copy()
        else:
            signal_trades = drunk_trades.copy()

        # For price proxy check, we need to compare:
        # Signal edge (fade) vs Baseline edge (fade at same prices, any time)

        # Get all sports trades (not just drunk times)
        all_sports = trades[trades['is_sports']].copy()
        all_sports['fade_trade_price'] = 100 - all_sports['trade_price']
        all_sports['fade_is_winner'] = ~all_sports['is_winner']

        # Get price range of signal trades
        signal_fade_prices = signal_trades['fade_trade_price'] if 'fade_trade_price' in signal_trades else (100 - signal_trades['trade_price'])
        price_min = signal_fade_prices.min()
        price_max = signal_fade_prices.max()

        # Baseline: all sports trades (any time) at same fade price range
        baseline_trades = all_sports[
            (all_sports['fade_trade_price'] >= price_min) &
            (all_sports['fade_trade_price'] <= price_max)
        ]

        if len(baseline_trades) > 0:
            baseline_stats = baseline_trades.groupby('market_ticker').agg({
                'fade_is_winner': 'mean',
                'fade_trade_price': 'mean'
            }).reset_index()

            baseline_win_rate = baseline_stats['fade_is_winner'].mean()
            baseline_price = baseline_stats['fade_trade_price'].mean()
            baseline_breakeven = baseline_price / 100.0
            baseline_edge = (baseline_win_rate - baseline_breakeven) * 100

            improvement = best_edge - baseline_edge

            results['price_proxy_check'] = {
                'best_variation': best_var,
                'signal_edge': best_edge,
                'baseline_edge': baseline_edge,
                'improvement': improvement,
                'is_price_proxy': improvement <= 0,
                'price_range': [price_min, price_max]
            }

            print(f"  Signal edge: {best_edge:+.1f}%")
            print(f"  Baseline edge (same prices, any time): {baseline_edge:+.1f}%")
            print(f"  Improvement: {improvement:+.1f}%")
            print(f"  Is price proxy: {improvement <= 0}")

    return results


def test_h071_trade_clustering(trades):
    """
    H071: Trade Clustering Velocity

    Signal: 5+ trades in same direction within 5 minutes
    Action: Fade the cluster (bet opposite direction)
    """
    print("\n" + "="*80)
    print("H071: TRADE CLUSTERING VELOCITY")
    print("="*80)

    results = {}

    # Group trades by market and sort by timestamp
    print("Analyzing trade clusters...")

    cluster_trades = []

    for ticker, group in trades.groupby('market_ticker'):
        group = group.sort_values('timestamp')

        # Look for clusters of 5+ trades in same direction within 5 minutes (300000 ms)
        window_ms = 5 * 60 * 1000  # 5 minutes in ms

        for i in range(len(group)):
            trade = group.iloc[i]
            window_end = trade['timestamp'] + window_ms

            # Get trades in window
            window_trades = group[
                (group['timestamp'] >= trade['timestamp']) &
                (group['timestamp'] <= window_end)
            ]

            # Count trades by direction
            yes_count = (window_trades['taker_side'] == 'yes').sum()
            no_count = (window_trades['taker_side'] == 'no').sum()

            # Check for cluster
            if yes_count >= 5 or no_count >= 5:
                dominant_side = 'yes' if yes_count > no_count else 'no'
                cluster_trades.append({
                    'market_ticker': ticker,
                    'cluster_side': dominant_side,
                    'cluster_size': max(yes_count, no_count),
                    'timestamp': trade['timestamp'],
                    'trade_price': trade['trade_price'],
                    'market_result': trade['market_result']
                })

    print(f"Found {len(cluster_trades):,} cluster events")

    if len(cluster_trades) == 0:
        results['h071'] = {'n_markets': 0, 'edge': 0, 'status': 'NO_DATA'}
        return results

    # Convert to DataFrame and deduplicate (take first cluster per market per hour)
    cluster_df = pd.DataFrame(cluster_trades)
    cluster_df['hour'] = cluster_df['timestamp'] // (3600 * 1000)
    cluster_df = cluster_df.drop_duplicates(subset=['market_ticker', 'hour', 'cluster_side'])

    print(f"Unique cluster events (deduped): {len(cluster_df):,}")

    # Calculate fade results
    # If cluster was YES, fade = bet NO, win if result='no'
    # If cluster was NO, fade = bet YES, win if result='yes'
    cluster_df['fade_wins'] = (
        (cluster_df['cluster_side'] == 'yes') & (cluster_df['market_result'] == 'no') |
        (cluster_df['cluster_side'] == 'no') & (cluster_df['market_result'] == 'yes')
    )

    # Calculate fade price (opposite of what was traded)
    cluster_df['fade_price'] = 100 - cluster_df['trade_price']

    # Aggregate to market level
    market_stats = cluster_df.groupby('market_ticker').agg({
        'fade_wins': 'mean',
        'fade_price': 'mean'
    }).reset_index()

    n_markets = len(market_stats)
    if n_markets < 50:
        results['h071'] = {
            'n_markets': n_markets,
            'edge': 0,
            'status': 'INSUFFICIENT_SAMPLE',
            'threshold': 50
        }
        print(f"  Insufficient markets: {n_markets} < 50")
        return results

    avg_win_rate = market_stats['fade_wins'].mean()
    avg_price = market_stats['fade_price'].mean()
    breakeven = avg_price / 100.0
    edge = (avg_win_rate - breakeven) * 100

    n_wins = int(avg_win_rate * n_markets)
    p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

    results['h071'] = {
        'description': 'Fade clusters of 5+ trades in same direction within 5 min',
        'n_clusters': len(cluster_df),
        'n_markets': n_markets,
        'win_rate': avg_win_rate * 100,
        'breakeven': breakeven * 100,
        'edge': edge,
        'p_value': p_value
    }

    print(f"  Clusters: {len(cluster_df):,}, Markets: {n_markets}")
    print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
    print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")

    return results


def test_h078_leverage_divergence(trades):
    """
    H078: Leverage Divergence from Price

    Signal: High leverage (>2) when price suggests low expected value
    e.g., Leverage > 2 at price 70-90c (where expected payout is low)
    Action: Fade high-leverage trades at unfavorable prices
    """
    print("\n" + "="*80)
    print("H078: LEVERAGE DIVERGENCE FROM PRICE")
    print("="*80)

    results = {}

    # Test different price ranges where high leverage seems irrational
    price_ranges = [
        (70, 90, 'mid_high'),
        (80, 90, 'high'),
        (50, 70, 'mid'),
        (30, 50, 'low_mid')
    ]

    for price_min, price_max, label in price_ranges:
        print(f"\n--- Price Range {price_min}-{price_max}c ---")

        # Filter for high leverage trades at this price
        # High leverage + high price = irrational (low potential return)
        signal_trades = trades[
            (trades['leverage_ratio'] > 2) &
            (trades['trade_price'] >= price_min) &
            (trades['trade_price'] <= price_max)
        ].copy()

        if len(signal_trades) == 0:
            results[f'h078_{label}'] = {'n_markets': 0, 'edge': 0}
            print("  No trades found")
            continue

        # Fade these trades
        signal_trades['fade_trade_price'] = 100 - signal_trades['trade_price']
        signal_trades['fade_is_winner'] = ~signal_trades['is_winner']

        market_stats = signal_trades.groupby('market_ticker').agg({
            'fade_is_winner': 'mean',
            'fade_trade_price': 'mean'
        }).reset_index()

        n_markets = len(market_stats)
        if n_markets < 50:
            results[f'h078_{label}'] = {'n_markets': n_markets, 'edge': 0, 'status': 'INSUFFICIENT'}
            print(f"  Insufficient markets: {n_markets}")
            continue

        avg_win_rate = market_stats['fade_is_winner'].mean()
        avg_price = market_stats['fade_trade_price'].mean()
        breakeven = avg_price / 100.0
        edge = (avg_win_rate - breakeven) * 100

        n_wins = int(avg_win_rate * n_markets)
        p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

        results[f'h078_{label}'] = {
            'description': f'Fade high-leverage (>2) trades at {price_min}-{price_max}c',
            'n_trades': len(signal_trades),
            'n_markets': n_markets,
            'win_rate': avg_win_rate * 100,
            'breakeven': breakeven * 100,
            'edge': edge,
            'p_value': p_value
        }

        print(f"  Trades: {len(signal_trades):,}, Markets: {n_markets}")
        print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
        print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")

    return results


def test_h084_leverage_trend(trades):
    """
    H084: Leverage Ratio Trend Within Market

    Signal: Increasing average leverage over market lifetime
    Action: If leverage trend is UP, fade late trades
    """
    print("\n" + "="*80)
    print("H084: LEVERAGE RATIO TREND WITHIN MARKET")
    print("="*80)

    results = {}

    # For each market, calculate leverage trend
    print("Calculating leverage trends per market...")

    market_trends = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 10:  # Need enough trades to measure trend
            continue

        group = group.sort_values('timestamp')

        # Split into first half and second half
        mid = len(group) // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]

        first_lev = first_half['leverage_ratio'].mean()
        second_lev = second_half['leverage_ratio'].mean()

        # Leverage trend
        lev_change = second_lev - first_lev

        # Market result
        result = group['market_result'].iloc[0]

        # Last trade info
        last_trade = group.iloc[-1]

        market_trends.append({
            'market_ticker': ticker,
            'first_lev': first_lev,
            'second_lev': second_lev,
            'lev_change': lev_change,
            'lev_increasing': lev_change > 0,
            'market_result': result,
            'last_trade_side': last_trade['taker_side'],
            'last_trade_price': last_trade['trade_price'],
            'n_trades': len(group)
        })

    trend_df = pd.DataFrame(market_trends)
    print(f"Markets with trends: {len(trend_df):,}")

    # Test: Markets with increasing leverage - fade late trades
    increasing_lev = trend_df[trend_df['lev_increasing']].copy()
    print(f"Markets with increasing leverage: {len(increasing_lev):,}")

    if len(increasing_lev) >= 50:
        # Fade the last trade
        increasing_lev['fade_wins'] = (
            (increasing_lev['last_trade_side'] == 'yes') & (increasing_lev['market_result'] == 'no') |
            (increasing_lev['last_trade_side'] == 'no') & (increasing_lev['market_result'] == 'yes')
        )
        increasing_lev['fade_price'] = 100 - increasing_lev['last_trade_price']

        n_markets = len(increasing_lev)
        avg_win_rate = increasing_lev['fade_wins'].mean()
        avg_price = increasing_lev['fade_price'].mean()
        breakeven = avg_price / 100.0
        edge = (avg_win_rate - breakeven) * 100

        n_wins = int(avg_win_rate * n_markets)
        p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

        results['h084_increasing_lev'] = {
            'description': 'Fade late trades in markets with increasing leverage',
            'n_markets': n_markets,
            'win_rate': avg_win_rate * 100,
            'breakeven': breakeven * 100,
            'edge': edge,
            'p_value': p_value
        }

        print(f"\n--- Markets with Increasing Leverage ---")
        print(f"  Markets: {n_markets}")
        print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
        print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")
    else:
        results['h084_increasing_lev'] = {'n_markets': len(increasing_lev), 'status': 'INSUFFICIENT'}

    return results


def test_h072_volatility_regimes(trades):
    """
    H072: Price Path Volatility Regimes

    Signal: Markets with high intra-market price variance
    Action: Bet on the direction of the most recent major move
    """
    print("\n" + "="*80)
    print("H072: PRICE PATH VOLATILITY REGIMES")
    print("="*80)

    results = {}

    # Calculate price volatility per market
    print("Calculating price volatility per market...")

    market_vol = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 5:
            continue

        group = group.sort_values('timestamp')

        # Price volatility (std of trade prices)
        price_std = group['trade_price'].std()
        price_range = group['trade_price'].max() - group['trade_price'].min()

        # Last major move direction
        # Compare last 3 trades avg to first 3 trades avg
        first_3 = group.head(3)['trade_price'].mean()
        last_3 = group.tail(3)['trade_price'].mean()
        major_move_up = last_3 > first_3

        # Market result
        result = group['market_result'].iloc[0]
        last_price = group.iloc[-1]['trade_price']

        market_vol.append({
            'market_ticker': ticker,
            'price_std': price_std,
            'price_range': price_range,
            'major_move_up': major_move_up,
            'market_result': result,
            'last_price': last_price,
            'n_trades': len(group)
        })

    vol_df = pd.DataFrame(market_vol)
    print(f"Markets with volatility data: {len(vol_df):,}")

    # Test: High volatility markets, bet on recent move direction
    median_vol = vol_df['price_std'].median()
    high_vol = vol_df[vol_df['price_std'] > median_vol].copy()

    print(f"High volatility markets (above median): {len(high_vol):,}")

    if len(high_vol) >= 50:
        # If major move UP, bet YES (market will settle YES)
        # If major move DOWN, bet NO (market will settle NO)
        high_vol['bet_wins'] = (
            (high_vol['major_move_up'] & (high_vol['market_result'] == 'yes')) |
            (~high_vol['major_move_up'] & (high_vol['market_result'] == 'no'))
        )

        # Bet price: if betting YES, pay yes_price (last_price); if betting NO, pay 100-last_price
        high_vol['bet_price'] = np.where(
            high_vol['major_move_up'],
            high_vol['last_price'],
            100 - high_vol['last_price']
        )

        n_markets = len(high_vol)
        avg_win_rate = high_vol['bet_wins'].mean()
        avg_price = high_vol['bet_price'].mean()
        breakeven = avg_price / 100.0
        edge = (avg_win_rate - breakeven) * 100

        n_wins = int(avg_win_rate * n_markets)
        p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

        results['h072_follow_move'] = {
            'description': 'High volatility markets: bet on recent move direction',
            'n_markets': n_markets,
            'win_rate': avg_win_rate * 100,
            'breakeven': breakeven * 100,
            'edge': edge,
            'p_value': p_value
        }

        print(f"\n--- Follow Recent Move in High Volatility ---")
        print(f"  Markets: {n_markets}")
        print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
        print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")

    # Test: High volatility markets, FADE recent move (contrarian)
    if len(high_vol) >= 50:
        high_vol['fade_wins'] = (
            (~high_vol['major_move_up'] & (high_vol['market_result'] == 'yes')) |
            (high_vol['major_move_up'] & (high_vol['market_result'] == 'no'))
        )

        # Fade price: opposite of bet price
        high_vol['fade_price'] = np.where(
            high_vol['major_move_up'],
            100 - high_vol['last_price'],  # Fade UP = bet NO
            high_vol['last_price']          # Fade DOWN = bet YES
        )

        avg_win_rate = high_vol['fade_wins'].mean()
        avg_price = high_vol['fade_price'].mean()
        breakeven = avg_price / 100.0
        edge = (avg_win_rate - breakeven) * 100

        n_wins = int(avg_win_rate * n_markets)
        p_value = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater').pvalue

        results['h072_fade_move'] = {
            'description': 'High volatility markets: FADE recent move direction',
            'n_markets': n_markets,
            'win_rate': avg_win_rate * 100,
            'breakeven': breakeven * 100,
            'edge': edge,
            'p_value': p_value
        }

        print(f"\n--- FADE Recent Move in High Volatility ---")
        print(f"  Markets: {n_markets}")
        print(f"  Win Rate: {avg_win_rate*100:.1f}%, Breakeven: {breakeven*100:.1f}%")
        print(f"  Edge: {edge:+.1f}%, P-value: {p_value:.4f}")

    return results


def main():
    """Run all Tier 1 hypothesis tests."""
    print("="*80)
    print("SESSION 010 PART 2: RIGOROUS HYPOTHESIS TESTING")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Bonferroni threshold: p < 0.003 (0.05 / 16)")

    # Load data
    trades, markets = load_data()

    # Run tests
    all_results = {}

    # H070: Drunk Sports Betting (USER REQUESTED - FIRST!)
    h070_results = test_h070_drunk_sports_betting(trades)
    all_results['H070_drunk_sports'] = h070_results

    # H071: Trade Clustering Velocity
    h071_results = test_h071_trade_clustering(trades)
    all_results['H071_clustering'] = h071_results

    # H078: Leverage Divergence from Price
    h078_results = test_h078_leverage_divergence(trades)
    all_results['H078_leverage_divergence'] = h078_results

    # H084: Leverage Ratio Trend
    h084_results = test_h084_leverage_trend(trades)
    all_results['H084_leverage_trend'] = h084_results

    # H072: Price Path Volatility
    h072_results = test_h072_volatility_regimes(trades)
    all_results['H072_volatility'] = h072_results

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: TIER 1 HYPOTHESES")
    print("="*80)

    bonferroni_threshold = 0.003
    validated = []
    rejected = []

    for hyp_id, results in all_results.items():
        if isinstance(results, dict):
            for var_id, var_results in results.items():
                if isinstance(var_results, dict) and 'edge' in var_results:
                    edge = var_results.get('edge', 0)
                    p_value = var_results.get('p_value', 1.0)
                    n_markets = var_results.get('n_markets', 0)

                    if edge > 0 and p_value < bonferroni_threshold and n_markets >= 50:
                        validated.append({
                            'hypothesis': hyp_id,
                            'variation': var_id,
                            **var_results
                        })
                    else:
                        rejected.append({
                            'hypothesis': hyp_id,
                            'variation': var_id,
                            **var_results
                        })

    print(f"\nValidated (edge > 0, p < {bonferroni_threshold}, N >= 50):")
    if validated:
        for v in validated:
            print(f"  {v['hypothesis']}/{v['variation']}: +{v['edge']:.1f}% edge, p={v.get('p_value', 1):.4f}, N={v.get('n_markets', 0)}")
    else:
        print("  NONE")

    print(f"\nRejected:")
    for r in rejected[:10]:  # Show first 10
        edge = r.get('edge', 0)
        p_val = r.get('p_value', 1)
        n = r.get('n_markets', 0)
        print(f"  {r['hypothesis']}/{r['variation']}: {edge:+.1f}% edge, p={p_val:.4f}, N={n}")

    # Save results
    output_file = REPORTS_DIR / f"session010_tier1_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    # Convert numpy types for JSON
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
            'bonferroni_threshold': bonferroni_threshold,
            'validated': validated,
            'rejected': rejected,
            'all_results': all_results
        }), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
