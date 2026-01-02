#!/usr/bin/env python3
"""
Session 008: Priority Hypothesis Testing
Test 5 Priority 1 hypotheses from Session 007 before 2026.

Hypotheses:
- H046: Closing Line Value (early vs late trades)
- H049: Recurring Market Patterns
- H065: Leverage Ratio as Fear Signal
- H052: Order Flow Imbalance Rate-of-Change
- H062: Multi-outcome Market Mispricing
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

# Configuration
DATA_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv")
OUTPUT_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")
OUTPUT_PATH.mkdir(exist_ok=True)

# Statistical thresholds
MIN_MARKETS = 50
MAX_CONCENTRATION = 0.30
ALPHA = 0.05
BONFERRONI_ALPHA = 0.05 / 20  # Adjust for multiple tests

def load_data():
    """Load the enriched trade data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df

def calculate_strategy_metrics(df, strategy_name, side_filter=None, price_range=None):
    """
    Calculate metrics for a strategy.
    Returns edge, win_rate, profit, markets, concentration, p_value
    """
    trades = df.copy()

    if side_filter:
        trades = trades[trades['taker_side'] == side_filter]

    if price_range:
        low, high = price_range
        trades = trades[(trades['trade_price'] >= low) & (trades['trade_price'] < high)]

    if len(trades) == 0:
        return None

    # Group by market
    market_stats = trades.groupby('market_ticker').agg({
        'trade_price': 'mean',
        'is_winner': 'first',
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum'
    }).reset_index()

    n_markets = len(market_stats)
    if n_markets < MIN_MARKETS:
        return {'error': f'Insufficient markets: {n_markets}'}

    # Calculate win rate and breakeven
    win_rate = market_stats['is_winner'].mean()
    avg_price = market_stats['trade_price'].mean()
    breakeven = avg_price / 100.0
    edge = (win_rate - breakeven) * 100  # As percentage points

    # Total profit
    total_profit = market_stats['actual_profit_dollars'].sum()

    # Concentration check
    profit_by_market = market_stats[market_stats['actual_profit_dollars'] > 0]['actual_profit_dollars']
    if len(profit_by_market) > 0 and total_profit > 0:
        max_concentration = profit_by_market.max() / total_profit
    else:
        max_concentration = 1.0

    # Statistical significance (binomial test)
    n_wins = int(win_rate * n_markets)
    result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
    p_value = result.pvalue

    return {
        'strategy': strategy_name,
        'markets': n_markets,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge_pct': edge,
        'total_profit': total_profit,
        'concentration': max_concentration,
        'p_value': p_value,
        'passes_validation': (
            n_markets >= MIN_MARKETS and
            max_concentration < MAX_CONCENTRATION and
            p_value < ALPHA
        ),
        'bonferroni_significant': p_value < BONFERRONI_ALPHA
    }


def test_h046_closing_line_value(df):
    """
    H046: Closing Line Value
    Compare edge of EARLY trades (first 10-20% of market lifetime) vs LATE trades (last 10-20%)

    Hypothesis: Early trades have more edge because early lines are "soft"
    """
    print("\n" + "="*80)
    print("H046: CLOSING LINE VALUE (Early vs Late Trades)")
    print("="*80)

    # Calculate market lifetime and trade position within it
    market_times = df.groupby('market_ticker').agg({
        'datetime': ['min', 'max', 'count']
    }).reset_index()
    market_times.columns = ['market_ticker', 'first_trade', 'last_trade', 'trade_count']
    market_times['duration_hours'] = (market_times['last_trade'] - market_times['first_trade']).dt.total_seconds() / 3600

    # Only markets with meaningful duration (at least 1 hour of trading)
    active_markets = market_times[market_times['duration_hours'] >= 1]['market_ticker'].tolist()
    df_active = df[df['market_ticker'].isin(active_markets)].copy()

    print(f"Markets with >= 1 hour trading duration: {len(active_markets):,}")

    # For each trade, calculate its position in market lifetime (0=first, 1=last)
    def get_position(group):
        if len(group) <= 1:
            return pd.Series([0.5] * len(group), index=group.index)
        times = group['datetime']
        min_t, max_t = times.min(), times.max()
        if min_t == max_t:
            return pd.Series([0.5] * len(group), index=group.index)
        return (times - min_t) / (max_t - min_t)

    df_active['market_position'] = df_active.groupby('market_ticker', group_keys=False).apply(get_position)

    # Define early (first 20%) and late (last 20%)
    early_trades = df_active[df_active['market_position'] <= 0.20]
    late_trades = df_active[df_active['market_position'] >= 0.80]

    print(f"Early trades (first 20%): {len(early_trades):,}")
    print(f"Late trades (last 20%): {len(late_trades):,}")

    results = []

    # Test across different price ranges
    for side in ['yes', 'no']:
        for price_range in [(50, 60), (60, 70), (70, 80), (80, 90)]:
            for period, period_df in [('early', early_trades), ('late', late_trades)]:
                metrics = calculate_strategy_metrics(
                    period_df,
                    f'{side}_{price_range[0]}_{price_range[1]}_{period}',
                    side_filter=side,
                    price_range=price_range
                )
                if metrics and 'error' not in metrics:
                    metrics['timing'] = period
                    metrics['side'] = side
                    metrics['price_range'] = price_range
                    results.append(metrics)

    if not results:
        print("No valid results - insufficient data")
        return {'status': 'REJECTED', 'reason': 'Insufficient data'}

    results_df = pd.DataFrame(results)

    # Compare early vs late for each strategy
    print("\n--- EARLY vs LATE Edge Comparison ---")
    comparisons = []
    for side in ['yes', 'no']:
        for price_range in [(50, 60), (60, 70), (70, 80), (80, 90)]:
            early = results_df[(results_df['timing'] == 'early') &
                              (results_df['side'] == side) &
                              (results_df['price_range'].apply(lambda x: x == price_range))]
            late = results_df[(results_df['timing'] == 'late') &
                             (results_df['side'] == side) &
                             (results_df['price_range'].apply(lambda x: x == price_range))]

            if len(early) > 0 and len(late) > 0:
                e = early.iloc[0]
                l = late.iloc[0]
                diff = e['edge_pct'] - l['edge_pct']
                print(f"{side.upper()} {price_range[0]}-{price_range[1]}: Early={e['edge_pct']:.1f}% Late={l['edge_pct']:.1f}% Diff={diff:.1f}%")
                comparisons.append({
                    'side': side,
                    'range': price_range,
                    'early_edge': e['edge_pct'],
                    'late_edge': l['edge_pct'],
                    'edge_diff': diff,
                    'early_markets': e['markets'],
                    'late_markets': l['markets']
                })

    # Check if early consistently beats late
    comp_df = pd.DataFrame(comparisons)
    early_wins = (comp_df['edge_diff'] > 0).sum()
    total = len(comp_df)

    print(f"\nEarly beats Late: {early_wins}/{total} comparisons")

    # Find the best CLV signal
    if len(comp_df) > 0:
        best = comp_df.loc[comp_df['edge_diff'].abs().idxmax()]
        print(f"\nLargest difference: {best['side'].upper()} {best['range']}")
        print(f"  Early edge: {best['early_edge']:.1f}%")
        print(f"  Late edge: {best['late_edge']:.1f}%")
        print(f"  Difference: {best['edge_diff']:.1f}%")

    return {
        'status': 'TESTED',
        'comparisons': comparisons,
        'early_wins': early_wins,
        'total_comparisons': total,
        'best_diff': best['edge_diff'] if len(comp_df) > 0 else 0
    }


def test_h049_recurring_markets(df):
    """
    H049: Recurring Market Patterns
    Daily/weekly markets (crypto prices, weather, etc.) - same participants, same mistakes.

    Look for systematic bias in recurring market types.
    """
    print("\n" + "="*80)
    print("H049: RECURRING MARKET PATTERNS")
    print("="*80)

    # Identify recurring market series by ticker prefix
    # Common patterns: KXBTCD (daily crypto), KXETH, weather, daily ranges
    df['series'] = df['market_ticker'].str.extract(r'^([A-Z]+)', expand=False)

    # Count markets per series
    series_counts = df.groupby('series')['market_ticker'].nunique().sort_values(ascending=False)
    print("Top series by market count:")
    print(series_counts.head(20))

    # Look for series with DAILY pattern in name
    df['is_daily'] = df['market_ticker'].str.contains('DAILY|24H|D-|DD-', case=False, na=False)

    # Also identify by date patterns in ticker (e.g., -25DEC27)
    df['has_date'] = df['market_ticker'].str.contains(r'-\d{2}[A-Z]{3}\d{2}', na=False)

    results = []

    # Test major recurring series
    major_series = ['KXBTCD', 'KXETHD', 'KXBTC', 'KXETH']

    for series in major_series:
        series_df = df[df['series'] == series]
        n_markets = series_df['market_ticker'].nunique()

        if n_markets < MIN_MARKETS:
            continue

        print(f"\n--- {series} ({n_markets} markets) ---")

        for side in ['yes', 'no']:
            for price_range in [(40, 60), (60, 80), (80, 95)]:
                metrics = calculate_strategy_metrics(
                    series_df,
                    f'{series}_{side}_{price_range[0]}_{price_range[1]}',
                    side_filter=side,
                    price_range=price_range
                )
                if metrics and 'error' not in metrics:
                    metrics['series'] = series
                    results.append(metrics)
                    if abs(metrics['edge_pct']) > 3:
                        print(f"  {side.upper()} {price_range[0]}-{price_range[1]}: Edge={metrics['edge_pct']:.1f}% WR={metrics['win_rate']:.1%} N={metrics['markets']}")

    # Also look at daily-pattern markets specifically
    daily_df = df[df['has_date']]
    print(f"\nMarkets with date patterns: {daily_df['market_ticker'].nunique()}")

    if len(results) == 0:
        print("No recurring series with sufficient markets found")
        return {'status': 'REJECTED', 'reason': 'No recurring series found'}

    # Find best opportunity
    results_df = pd.DataFrame(results)
    validated = results_df[results_df['passes_validation']]

    if len(validated) > 0:
        best = validated.loc[validated['edge_pct'].abs().idxmax()]
        print(f"\nBest validated opportunity:")
        print(f"  Series: {best['series']}")
        print(f"  Strategy: {best['strategy']}")
        print(f"  Edge: {best['edge_pct']:.1f}%")
        print(f"  Markets: {best['markets']}")
        print(f"  P-value: {best['p_value']:.4f}")
        return {
            'status': 'VALIDATED' if best['bonferroni_significant'] else 'MARGINAL',
            'best_series': best['series'],
            'best_strategy': best['strategy'],
            'edge': best['edge_pct'],
            'markets': best['markets'],
            'p_value': best['p_value']
        }

    return {'status': 'REJECTED', 'reason': 'No strategies pass validation'}


def test_h065_leverage_ratio(df):
    """
    H065: Leverage Ratio as Fear Signal
    High leverage (big $ on longshots) = desperation/retail
    Can we fade high-leverage trades?

    leverage_ratio = potential_profit / cost (already in data)
    """
    print("\n" + "="*80)
    print("H065: LEVERAGE RATIO AS FEAR SIGNAL")
    print("="*80)

    # Analyze leverage distribution
    print("\nLeverage ratio distribution:")
    print(df['leverage_ratio'].describe())

    # High leverage = betting on longshots (high potential return)
    # Low leverage = betting on favorites (low potential return)

    # Define leverage buckets
    df['leverage_bucket'] = pd.cut(df['leverage_ratio'],
                                    bins=[0, 0.5, 1, 2, 5, 10, 100],
                                    labels=['<0.5', '0.5-1', '1-2', '2-5', '5-10', '>10'])

    print("\nTrades by leverage bucket:")
    print(df['leverage_bucket'].value_counts().sort_index())

    results = []

    # Test: FADE high leverage (bet opposite side when retail bets longshots)
    # If someone bets YES at 10c (high leverage), we bet NO

    # Strategy: For high leverage YES trades (longshots), the opposite (NO) should win
    for leverage_threshold in [2, 3, 5, 10]:
        high_leverage = df[df['leverage_ratio'] > leverage_threshold]

        # These are the longshot bets
        longshot_yes = high_leverage[high_leverage['taker_side'] == 'yes']
        longshot_no = high_leverage[high_leverage['taker_side'] == 'no']

        # Fading longshot YES means betting NO
        # What was the actual outcome when retail bet high-leverage YES?
        if len(longshot_yes) > 0:
            markets_longshot_yes = longshot_yes.groupby('market_ticker').agg({
                'is_winner': 'first',  # Did their YES bet win?
                'actual_profit_dollars': 'sum'
            }).reset_index()

            n = len(markets_longshot_yes)
            if n >= MIN_MARKETS:
                # If their YES lost (is_winner=False), we would have won with NO
                fade_win_rate = 1 - markets_longshot_yes['is_winner'].mean()
                # For fading, we need to estimate our cost and breakeven
                # Roughly: if they paid 10c for YES, NO costs 90c, breakeven ~90%
                avg_their_price = longshot_yes['yes_price'].mean()
                our_price = 100 - avg_their_price  # Our NO price
                breakeven = our_price / 100.0
                edge = (fade_win_rate - breakeven) * 100

                print(f"\nFade High-Leverage YES (leverage > {leverage_threshold}):")
                print(f"  Markets: {n}")
                print(f"  Their avg YES price: {avg_their_price:.0f}c (our NO: {our_price:.0f}c)")
                print(f"  Fade win rate: {fade_win_rate:.1%}")
                print(f"  Breakeven: {breakeven:.1%}")
                print(f"  Edge: {edge:.1f}%")

                results.append({
                    'strategy': f'fade_high_leverage_yes_{leverage_threshold}',
                    'threshold': leverage_threshold,
                    'markets': n,
                    'win_rate': fade_win_rate,
                    'breakeven': breakeven,
                    'edge_pct': edge,
                    'their_price': avg_their_price,
                    'our_price': our_price
                })

    # Also test: LOW leverage trades (favorites) - are they smarter?
    print("\n--- Low Leverage Trades (Favorites) ---")
    low_leverage = df[df['leverage_ratio'] < 0.5]  # Betting favorites

    for side in ['yes', 'no']:
        side_df = low_leverage[low_leverage['taker_side'] == side]
        if len(side_df) == 0:
            continue

        markets = side_df.groupby('market_ticker').agg({
            'trade_price': 'mean',
            'is_winner': 'first'
        }).reset_index()

        n = len(markets)
        if n >= MIN_MARKETS:
            win_rate = markets['is_winner'].mean()
            breakeven = markets['trade_price'].mean() / 100
            edge = (win_rate - breakeven) * 100

            print(f"Low leverage {side.upper()}: WR={win_rate:.1%} BE={breakeven:.1%} Edge={edge:.1f}% N={n}")

            results.append({
                'strategy': f'low_leverage_{side}',
                'threshold': '<0.5',
                'markets': n,
                'win_rate': win_rate,
                'breakeven': breakeven,
                'edge_pct': edge
            })

    if not results:
        return {'status': 'REJECTED', 'reason': 'No valid results'}

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['edge_pct'].abs().idxmax()]

    return {
        'status': 'MARGINAL' if best['edge_pct'] > 2 else 'REJECTED',
        'best_strategy': best['strategy'],
        'edge': best['edge_pct'],
        'markets': best['markets'],
        'all_results': results
    }


def test_h052_order_flow_roc(df):
    """
    H052: Order Flow Imbalance Rate-of-Change
    Not just imbalance, but ACCELERATION of imbalance.
    Rapid shift in flow direction = signal?
    """
    print("\n" + "="*80)
    print("H052: ORDER FLOW IMBALANCE RATE-OF-CHANGE")
    print("="*80)

    # For each market, calculate the flow imbalance over time
    # Then look for markets with RAPID changes in imbalance

    # Get markets with enough trades to measure rate of change
    market_trade_counts = df.groupby('market_ticker').size()
    active_markets = market_trade_counts[market_trade_counts >= 10].index.tolist()

    print(f"Markets with >= 10 trades: {len(active_markets)}")

    df_active = df[df['market_ticker'].isin(active_markets)].copy()
    df_active = df_active.sort_values(['market_ticker', 'datetime'])

    # Calculate rolling flow imbalance per market
    # Imbalance = (YES volume - NO volume) / Total volume
    def calc_flow_metrics(group):
        group = group.sort_values('datetime')

        # Calculate cumulative YES/NO flow
        yes_flow = (group['taker_side'] == 'yes').astype(int) * group['cost_dollars']
        no_flow = (group['taker_side'] == 'no').astype(int) * group['cost_dollars']

        cum_yes = yes_flow.cumsum()
        cum_no = no_flow.cumsum()
        total = cum_yes + cum_no

        # Imbalance at each point (avoid division by zero)
        imbalance = (cum_yes - cum_no) / total.replace(0, 1)

        # Rate of change of imbalance
        imbalance_diff = imbalance.diff()

        # Get early vs late imbalance
        n = len(group)
        early_imb = imbalance.iloc[:max(1, n//3)].mean() if n > 0 else 0
        late_imb = imbalance.iloc[-max(1, n//3):].mean() if n > 0 else 0

        # Max acceleration (biggest single change)
        max_acceleration = imbalance_diff.abs().max() if len(imbalance_diff) > 0 else 0

        return pd.Series({
            'early_imbalance': early_imb,
            'late_imbalance': late_imb,
            'imbalance_shift': late_imb - early_imb,
            'max_acceleration': max_acceleration,
            'final_imbalance': imbalance.iloc[-1] if len(imbalance) > 0 else 0
        })

    print("Calculating flow metrics per market...")
    flow_metrics = df_active.groupby('market_ticker').apply(calc_flow_metrics)

    # Merge with outcomes
    outcomes = df_active.groupby('market_ticker').agg({
        'is_winner': 'first',
        'market_result': 'first'
    }).reset_index()

    flow_metrics = flow_metrics.reset_index()
    flow_metrics = flow_metrics.merge(outcomes, on='market_ticker')

    print(f"\nFlow metrics calculated for {len(flow_metrics)} markets")

    # Test: Do markets with large imbalance SHIFT have predictable outcomes?
    shift_threshold = 0.3  # 30% shift in imbalance

    high_shift = flow_metrics[flow_metrics['imbalance_shift'].abs() > shift_threshold]
    print(f"\nMarkets with imbalance shift > {shift_threshold}: {len(high_shift)}")

    if len(high_shift) >= MIN_MARKETS:
        # When imbalance shifts toward YES (positive), does YES win?
        shift_to_yes = high_shift[high_shift['imbalance_shift'] > 0]
        shift_to_no = high_shift[high_shift['imbalance_shift'] < 0]

        if len(shift_to_yes) >= 20:
            yes_outcome = (shift_to_yes['market_result'] == 'yes').mean()
            print(f"Shift to YES ({len(shift_to_yes)} markets): YES wins {yes_outcome:.1%}")

        if len(shift_to_no) >= 20:
            yes_outcome = (shift_to_no['market_result'] == 'yes').mean()
            print(f"Shift to NO ({len(shift_to_no)} markets): YES wins {yes_outcome:.1%} (NO wins {1-yes_outcome:.1%})")

    # Test: Maximum acceleration as signal
    accel_threshold = 0.2
    high_accel = flow_metrics[flow_metrics['max_acceleration'] > accel_threshold]
    print(f"\nMarkets with max acceleration > {accel_threshold}: {len(high_accel)}")

    results = []

    # Strategy: After high acceleration toward one side, bet that side
    for direction in ['yes', 'no']:
        if direction == 'yes':
            signal_markets = flow_metrics[flow_metrics['imbalance_shift'] > shift_threshold]
        else:
            signal_markets = flow_metrics[flow_metrics['imbalance_shift'] < -shift_threshold]

        if len(signal_markets) >= MIN_MARKETS:
            wins = (signal_markets['market_result'] == direction).sum()
            n = len(signal_markets)
            win_rate = wins / n

            # For breakeven, assume average price (rough estimate)
            breakeven = 0.5  # Neutral assumption
            edge = (win_rate - breakeven) * 100

            result = stats.binomtest(wins, n, breakeven, alternative='greater')
            p_value = result.pvalue

            print(f"\nFollow imbalance shift to {direction.upper()}:")
            print(f"  Markets: {n}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  P-value: {p_value:.4f}")

            results.append({
                'strategy': f'follow_shift_{direction}',
                'markets': n,
                'win_rate': win_rate,
                'edge_pct': edge,
                'p_value': p_value
            })

    if not results:
        return {'status': 'REJECTED', 'reason': 'No valid results'}

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['edge_pct'].idxmax()]

    return {
        'status': 'MARGINAL' if best['edge_pct'] > 5 and best['p_value'] < 0.05 else 'REJECTED',
        'results': results
    }


def test_h062_multi_outcome_mispricing(df):
    """
    H062: Multi-outcome Market Mispricing
    Do YES prices across related markets sum to reasonable probabilities?
    Internal arbitrage opportunities?
    """
    print("\n" + "="*80)
    print("H062: MULTI-OUTCOME MARKET MISPRICING")
    print("="*80)

    # Identify market series with multiple outcomes
    # Pattern: Same prefix with different endings (e.g., GAME-TEAMX, GAME-TEAMY)

    # Extract base event from ticker
    # Common patterns: KXNCAAFGAME-25DEC27LSUHOU-LSU vs KXNCAAFGAME-25DEC27LSUHOU-HOU

    df['base_event'] = df['market_ticker'].str.extract(r'^(.+)-[^-]+$', expand=False)

    # Find events with multiple market variations
    event_counts = df.groupby('base_event')['market_ticker'].nunique()
    multi_outcome_events = event_counts[event_counts > 1].index.tolist()

    print(f"Events with multiple market outcomes: {len(multi_outcome_events)}")

    if len(multi_outcome_events) == 0:
        # Try a different pattern
        df['base_event'] = df['market_ticker'].str.extract(r'^([A-Z]+\-\d+[A-Z]+\d+[A-Z]+)', expand=False)
        event_counts = df.groupby('base_event')['market_ticker'].nunique()
        multi_outcome_events = event_counts[event_counts > 1].index.tolist()
        print(f"Events with multiple outcomes (pattern 2): {len(multi_outcome_events)}")

    # For each multi-outcome event, check if probabilities sum to ~100%
    results = []

    for event in multi_outcome_events[:100]:  # Sample first 100
        event_df = df[df['base_event'] == event]
        tickers = event_df['market_ticker'].unique()

        if len(tickers) < 2:
            continue

        # Get last known YES price for each ticker in this event
        ticker_prices = []
        for ticker in tickers:
            ticker_df = event_df[event_df['market_ticker'] == ticker].sort_values('datetime')
            if len(ticker_df) > 0:
                last_yes_price = ticker_df.iloc[-1]['yes_price']
                ticker_prices.append({
                    'ticker': ticker,
                    'yes_price': last_yes_price,
                    'implied_prob': last_yes_price / 100
                })

        if len(ticker_prices) >= 2:
            total_prob = sum(t['implied_prob'] for t in ticker_prices)
            overround = total_prob - 1  # Should be close to 0 if efficient

            if abs(overround) > 0.1:  # More than 10% mispricing
                results.append({
                    'event': event,
                    'n_outcomes': len(ticker_prices),
                    'total_prob': total_prob,
                    'overround': overround,
                    'tickers': ticker_prices
                })

    print(f"\nEvents with >10% mispricing: {len(results)}")

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        print("\nTop mispriced events:")
        for _, row in results_df.nlargest(5, 'overround').iterrows():
            print(f"  {row['event']}: {row['n_outcomes']} outcomes, total prob = {row['total_prob']:.1%}")

    # This is more about arbitrage than directional trading
    # Check if overpriced markets (prob > 1) lose more often

    return {
        'status': 'MARGINAL' if len(results) > 10 else 'REJECTED',
        'mispriced_events': len(results),
        'max_overround': max([r['overround'] for r in results]) if results else 0
    }


def main():
    """Run all priority hypothesis tests."""
    print("="*80)
    print("SESSION 008: PRIORITY HYPOTHESIS TESTING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    df = load_data()

    all_results = {}

    # H046: Closing Line Value
    all_results['H046_CLV'] = test_h046_closing_line_value(df)

    # H049: Recurring Markets
    all_results['H049_Recurring'] = test_h049_recurring_markets(df)

    # H065: Leverage Ratio
    all_results['H065_Leverage'] = test_h065_leverage_ratio(df)

    # H052: Order Flow ROC
    all_results['H052_FlowROC'] = test_h052_order_flow_roc(df)

    # H062: Multi-outcome
    all_results['H062_MultiOutcome'] = test_h062_multi_outcome_mispricing(df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, result in all_results.items():
        status = result.get('status', 'UNKNOWN')
        print(f"\n{name}: {status}")
        if 'edge' in result:
            print(f"  Edge: {result['edge']:.1f}%")
        if 'markets' in result:
            print(f"  Markets: {result['markets']}")
        if 'p_value' in result:
            print(f"  P-value: {result['p_value']:.4f}")

    # Save results
    output_file = OUTPUT_PATH / f"session008_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj

    with open(output_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == '__main__':
    main()
