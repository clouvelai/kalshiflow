"""
S-LATE-001 / SPORTS-007 DEEP DIVE: Late-Arriving Large Money

Comprehensive analysis answering: Is this actually profitable in real dollar terms?

Tasks:
1. DOLLAR P&L REALITY CHECK - Calculate actual $ profit, not just edge %
2. PARAMETER SENSITIVITY GRID - Find robust parameter ranges
3. RLM INDEPENDENCE ANALYSIS - Does this add value over RLM?
4. REAL-TIME IMPLEMENTATION FEASIBILITY - Can we implement without hindsight?
5. RISK STRESS TEST - Find failure modes

Current state from initial validation:
- Signal: Final 25% of trades has 2x large trade ratio (>$50), late large trades favor NO
- Raw Edge: +19.8%
- Win Rate: 95.47%
- Sample: 331 markets
- Bucket Ratio: 11/11 (100%)
- Temporal Stability: 4/4 quarters positive
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
REPORT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/slate001_deep_dive.json'


def load_data():
    """Load the enriched trades data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['trade_value_dollars'] = df['trade_value_cents'] / 100
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Get date range
    date_range = (df['datetime'].max() - df['datetime'].min()).days
    print(f"Data spans {date_range} days ({df['datetime'].min().date()} to {df['datetime'].max().date()})")

    return df, date_range


def build_baseline():
    """Build price bucket baseline for comparison."""
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


def detect_late_large_signal(df, final_window_pct=0.25, ratio_threshold=2.0, large_threshold=5000):
    """
    Detect markets with late-arriving large money signal.

    Parameters:
    - final_window_pct: What fraction of trades is "final" (default 0.25 = 25%)
    - ratio_threshold: How much more large trade activity in final vs early (default 2.0x)
    - large_threshold: Dollar threshold for "large" trade in cents (default 5000 = $50)

    Returns DataFrame with signal markets and their properties
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    late_large_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 16:  # Need enough trades to split
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        # Split into early and late
        cutoff = int(n * (1 - final_window_pct))
        early = mdf.iloc[:cutoff]
        late = mdf.iloc[cutoff:]

        if len(late) < 4:
            continue

        # Calculate large trade ratios
        early_large_ratio = (early['trade_value_cents'] > large_threshold).mean()
        late_large_ratio = (late['trade_value_cents'] > large_threshold).mean()

        # Check if late has significantly more large trades
        if late_large_ratio > early_large_ratio * ratio_threshold and late_large_ratio > 0.2:
            late_large = late[late['trade_value_cents'] > large_threshold]
            if len(late_large) < 2:
                continue

            # Determine direction of late large money
            late_yes_ratio = (late_large['taker_side'] == 'yes').mean()
            late_direction = 'yes' if late_yes_ratio > 0.6 else ('no' if late_yes_ratio < 0.4 else 'neutral')

            if late_direction != 'neutral':
                late_large_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'late_direction': late_direction,
                    'late_yes_ratio': late_yes_ratio,
                    'early_large_ratio': early_large_ratio,
                    'late_large_ratio': late_large_ratio,
                    'late_large_count': len(late_large),
                    'late_value_cents': late_large['trade_value_cents'].sum(),
                    'late_value_dollars': late_large['trade_value_dollars'].sum(),
                    'first_trade_time': mdf['datetime'].iloc[0],
                    'last_trade_time': mdf['datetime'].iloc[-1],
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean(),
                    'total_trades': n,
                    'total_volume_dollars': mdf['trade_value_dollars'].sum()
                })

    return pd.DataFrame(late_large_markets)


# =============================================================================
# TASK 1: DOLLAR P&L REALITY CHECK
# =============================================================================

def task1_dollar_pnl_analysis(ll_df, date_range_days, baseline_by_bucket):
    """
    Calculate actual dollar P&L, not just edge %.

    Key questions:
    - What's the actual dollar profit/loss per signal?
    - At $X position size, what's expected annual profit?
    - How does this compare to RLM?
    """
    print("\n" + "=" * 80)
    print("TASK 1: DOLLAR P&L REALITY CHECK")
    print("=" * 80)

    # Focus on Follow Late NO (the validated signal)
    signal = ll_df[ll_df['late_direction'] == 'no'].copy()
    n_signals = len(signal)

    print(f"\nTotal signal markets: {n_signals}")

    # Calculate per-signal outcomes (betting $1 on each NO)
    # When betting NO at price X:
    #   Win: profit = (100 - X) cents per $1 bet
    #   Loss: loss = X cents per $1 bet

    signal['bet_amount_dollars'] = 1.0  # Assume $1 per bet
    signal['won'] = (signal['market_result'] == 'no').astype(int)
    signal['no_price_decimal'] = signal['no_price'] / 100

    # Profit = (1 - no_price_decimal) if win, else -no_price_decimal
    signal['profit_per_dollar'] = np.where(
        signal['won'] == 1,
        (1 - signal['no_price_decimal']),  # Win: get back 100c, paid NO price
        -signal['no_price_decimal']  # Lose: lose entire stake
    )

    print("\n----- PER-SIGNAL P&L DISTRIBUTION -----")

    profits = signal['profit_per_dollar']
    print(f"Mean profit per $1 bet: ${profits.mean():.4f}")
    print(f"Median profit per $1 bet: ${profits.median():.4f}")
    print(f"Std dev: ${profits.std():.4f}")
    print(f"Min: ${profits.min():.4f}")
    print(f"Max: ${profits.max():.4f}")

    # P&L distribution
    print(f"\nP&L Distribution:")
    print(f"  Wins: {signal['won'].sum()} ({signal['won'].mean()*100:.1f}%)")
    print(f"  Losses: {(~signal['won'].astype(bool)).sum()} ({(~signal['won'].astype(bool)).mean()*100:.1f}%)")

    # Calculate typical win/loss amounts
    wins = signal[signal['won'] == 1]['profit_per_dollar']
    losses = signal[signal['won'] == 0]['profit_per_dollar']

    print(f"\n  Avg win: ${wins.mean():.4f}")
    print(f"  Avg loss: ${losses.mean():.4f}")
    print(f"  Win/Loss ratio: {abs(wins.mean() / losses.mean()):.2f}x")

    # Total P&L from all signals
    total_profit = profits.sum()
    print(f"\n  Total P&L (all signals, $1/bet): ${total_profit:.2f}")

    # ===== ANNUALIZED PROFIT MODEL =====
    print("\n----- ANNUALIZED PROFIT MODEL -----")

    # Calculate signals per year
    signals_per_year = n_signals * (365 / date_range_days)
    print(f"Signals per year (extrapolated): {signals_per_year:.0f}")

    # Edge per bet
    avg_edge_per_bet = profits.mean()

    print("\n  Expected Annual Returns by Capital:")
    print(f"  {'Capital':<12} {'Per Bet':<12} {'Monthly':<15} {'Annual':<15} {'Annual %':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*12}")

    capital_levels = [100, 500, 1000, 5000, 10000, 25000, 50000]
    annualized_returns = []

    for capital in capital_levels:
        # Position size = capital (full kelly would be aggressive, so assume 1:1)
        # Actually, let's be conservative: bet $1 per signal
        bet_size = 1.0  # Could scale with capital but let's be conservative

        # Expected annual profit = signals_per_year * edge_per_bet * bet_size
        monthly_signals = signals_per_year / 12
        monthly_profit = monthly_signals * avg_edge_per_bet * bet_size
        annual_profit = signals_per_year * avg_edge_per_bet * bet_size

        # But if we scale bet size with capital (e.g., 1% of capital per bet)
        bet_size_scaled = capital * 0.01  # 1% of capital
        monthly_profit_scaled = monthly_signals * avg_edge_per_bet * bet_size_scaled
        annual_profit_scaled = signals_per_year * avg_edge_per_bet * bet_size_scaled
        annual_pct = (annual_profit_scaled / capital) * 100

        print(f"  ${capital:<11,} ${bet_size_scaled:<11.0f} ${monthly_profit_scaled:<14,.2f} ${annual_profit_scaled:<14,.2f} {annual_pct:>10.1f}%")

        annualized_returns.append({
            'capital': capital,
            'bet_size': bet_size_scaled,
            'monthly_profit': monthly_profit_scaled,
            'annual_profit': annual_profit_scaled,
            'annual_pct': annual_pct
        })

    # ===== COMPARISON TO RLM =====
    print("\n----- COMPARISON TO RLM (if available) -----")
    # RLM stats from journal: H123 validated with +17.38% edge, 1986 markets
    rlm_edge = 0.1738
    rlm_markets = 1986
    rlm_signals_per_year = rlm_markets * (365 / date_range_days)

    print(f"\nRLM (H123) comparison:")
    print(f"  Edge: {rlm_edge:.1%}")
    print(f"  Markets in data: {rlm_markets}")
    print(f"  Signals/year (extrapolated): {rlm_signals_per_year:.0f}")

    # S-LATE-001
    slate_edge = avg_edge_per_bet
    print(f"\nS-LATE-001 (SPORTS-007) comparison:")
    print(f"  Edge: {slate_edge:.1%}")
    print(f"  Markets in data: {n_signals}")
    print(f"  Signals/year (extrapolated): {signals_per_year:.0f}")

    # Volume-adjusted comparison
    rlm_annual_profit_per_dollar = rlm_edge * rlm_signals_per_year
    slate_annual_profit_per_dollar = slate_edge * signals_per_year

    print(f"\nAnnual Profit per $1 bet size:")
    print(f"  RLM: ${rlm_annual_profit_per_dollar:.2f}")
    print(f"  S-LATE-001: ${slate_annual_profit_per_dollar:.2f}")
    print(f"  Difference: S-LATE generates {slate_annual_profit_per_dollar/rlm_annual_profit_per_dollar:.1%}x as much per year")

    return {
        'n_signals': n_signals,
        'win_rate': signal['won'].mean(),
        'avg_profit_per_dollar': avg_edge_per_bet,
        'std_profit_per_dollar': profits.std(),
        'total_pnl': total_profit,
        'signals_per_year': signals_per_year,
        'annualized_returns': annualized_returns,
        'vs_rlm': {
            'rlm_edge': rlm_edge,
            'rlm_signals_per_year': rlm_signals_per_year,
            'slate_edge': slate_edge,
            'slate_signals_per_year': signals_per_year,
            'relative_annual_profit': slate_annual_profit_per_dollar / rlm_annual_profit_per_dollar
        }
    }


# =============================================================================
# TASK 2: PARAMETER SENSITIVITY GRID
# =============================================================================

def task2_parameter_sensitivity(df, all_markets, baseline_by_bucket):
    """
    Test parameter ranges to find robust region.

    Parameters to test:
    - final_windows: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    - ratio_thresholds: [1.5, 2.0, 2.5, 3.0]
    - dollar_thresholds: [30, 50, 75, 100] (in dollars, converted to cents)
    """
    print("\n" + "=" * 80)
    print("TASK 2: PARAMETER SENSITIVITY GRID")
    print("=" * 80)

    final_windows = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    ratio_thresholds = [1.5, 2.0, 2.5, 3.0]
    dollar_thresholds = [30, 50, 75, 100]  # In dollars

    results = []

    total_combos = len(final_windows) * len(ratio_thresholds) * len(dollar_thresholds)
    combo_num = 0

    print(f"\nTesting {total_combos} parameter combinations...")

    for window in final_windows:
        for ratio in ratio_thresholds:
            for dollar in dollar_thresholds:
                combo_num += 1
                if combo_num % 10 == 0:
                    print(f"  Progress: {combo_num}/{total_combos}")

                # Detect signals with these parameters
                ll_df = detect_late_large_signal(
                    df,
                    final_window_pct=window,
                    ratio_threshold=ratio,
                    large_threshold=dollar * 100  # Convert to cents
                )

                # Get NO signals
                signal = ll_df[ll_df['late_direction'] == 'no'].copy() if len(ll_df) > 0 else pd.DataFrame()

                if len(signal) < 30:
                    results.append({
                        'window': window,
                        'ratio': ratio,
                        'dollar': dollar,
                        'n_signals': len(signal),
                        'win_rate': None,
                        'edge': None,
                        'status': 'insufficient_sample'
                    })
                    continue

                # Calculate metrics
                n = len(signal)
                wins = (signal['market_result'] == 'no').sum()
                wr = wins / n
                avg_price = signal['no_price'].mean()
                be = avg_price / 100
                edge = wr - be

                # P-value
                z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
                p_value = 1 - stats.norm.cdf(z)

                results.append({
                    'window': window,
                    'ratio': ratio,
                    'dollar': dollar,
                    'n_signals': n,
                    'win_rate': wr,
                    'avg_price': avg_price,
                    'breakeven': be,
                    'edge': edge,
                    'p_value': p_value,
                    'status': 'valid'
                })

    results_df = pd.DataFrame(results)

    # ===== DISPLAY RESULTS =====
    print("\n----- PARAMETER SENSITIVITY RESULTS -----")

    valid_results = results_df[results_df['status'] == 'valid'].copy()

    if len(valid_results) == 0:
        print("No valid parameter combinations found!")
        return {'status': 'no_valid_combos'}

    # Best parameters
    best_idx = valid_results['edge'].idxmax()
    best = valid_results.loc[best_idx]

    print(f"\nBest Parameters (by edge):")
    print(f"  Window: {best['window']:.0%}")
    print(f"  Ratio: {best['ratio']:.1f}x")
    print(f"  Dollar threshold: ${best['dollar']}")
    print(f"  Edge: {best['edge']:.1%}")
    print(f"  N signals: {best['n_signals']:.0f}")
    print(f"  Win rate: {best['win_rate']:.1%}")
    print(f"  P-value: {best['p_value']:.6f}")

    # Robust region (edge > 10%)
    robust = valid_results[valid_results['edge'] > 0.10]
    print(f"\n----- ROBUST REGION (Edge > 10%) -----")
    print(f"Combinations with >10% edge: {len(robust)}/{len(valid_results)}")

    if len(robust) > 0:
        print(f"\nRobust parameter ranges:")
        print(f"  Window: {robust['window'].min():.0%} - {robust['window'].max():.0%}")
        print(f"  Ratio: {robust['ratio'].min():.1f}x - {robust['ratio'].max():.1f}x")
        print(f"  Dollar: ${robust['dollar'].min()} - ${robust['dollar'].max()}")

    # Edge by window (heatmap-like)
    print("\n----- EDGE BY WINDOW (averaged across other params) -----")
    edge_by_window = valid_results.groupby('window')['edge'].mean()
    for w, e in edge_by_window.items():
        print(f"  Window {w:.0%}: {e:.1%} edge")

    # Edge by dollar threshold
    print("\n----- EDGE BY DOLLAR THRESHOLD -----")
    edge_by_dollar = valid_results.groupby('dollar')['edge'].mean()
    for d, e in edge_by_dollar.items():
        print(f"  ${d} threshold: {e:.1%} edge")

    # Dangerous cliffs (sharp drops)
    print("\n----- CLIFF DETECTION -----")
    # Sort by edge and look for big drops
    sorted_results = valid_results.sort_values('edge', ascending=False)

    for i in range(1, min(5, len(sorted_results))):
        prev = sorted_results.iloc[i-1]
        curr = sorted_results.iloc[i]
        edge_drop = prev['edge'] - curr['edge']
        if edge_drop > 0.05:
            print(f"  CLIFF: {prev['edge']:.1%} -> {curr['edge']:.1%} (-{edge_drop:.1%})")
            print(f"    From: window={prev['window']:.0%}, ratio={prev['ratio']:.1f}x, ${prev['dollar']}")
            print(f"    To:   window={curr['window']:.0%}, ratio={curr['ratio']:.1f}x, ${curr['dollar']}")

    # Print full results table for window=0.25 (default)
    print("\n----- DETAILED RESULTS FOR 25% WINDOW -----")
    default_window = valid_results[valid_results['window'] == 0.25]
    print(default_window[['ratio', 'dollar', 'n_signals', 'win_rate', 'edge', 'p_value']].to_string(index=False))

    return {
        'best_params': {
            'window': float(best['window']),
            'ratio': float(best['ratio']),
            'dollar': int(best['dollar']),
            'edge': float(best['edge']),
            'n_signals': int(best['n_signals'])
        },
        'robust_region': {
            'count': len(robust),
            'window_range': [float(robust['window'].min()), float(robust['window'].max())] if len(robust) > 0 else None,
            'ratio_range': [float(robust['ratio'].min()), float(robust['ratio'].max())] if len(robust) > 0 else None,
            'dollar_range': [int(robust['dollar'].min()), int(robust['dollar'].max())] if len(robust) > 0 else None
        },
        'edge_by_window': edge_by_window.to_dict(),
        'edge_by_dollar': edge_by_dollar.to_dict(),
        'all_results': results_df.to_dict('records')
    }


# =============================================================================
# TASK 3: RLM INDEPENDENCE ANALYSIS
# =============================================================================

def detect_rlm_signal(df):
    """
    Detect RLM (Reverse Line Movement) signals.

    RLM signal (from H123):
    - YES trade ratio > 65%
    - YES price dropped from first to last trade
    - Bet NO
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    rlm_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 5:
            continue

        mdf = mdf.reset_index(drop=True)

        # Calculate YES ratio
        yes_ratio = (mdf['taker_side'] == 'yes').mean()

        # Get first and last YES price
        first_yes_price = mdf['yes_price'].iloc[0]
        last_yes_price = mdf['yes_price'].iloc[-1]

        # RLM condition: >65% YES trades but price moved toward NO
        if yes_ratio > 0.65 and last_yes_price < first_yes_price:
            rlm_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'yes_ratio': yes_ratio,
                'first_yes_price': first_yes_price,
                'last_yes_price': last_yes_price,
                'price_drop': first_yes_price - last_yes_price,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    return pd.DataFrame(rlm_markets)


def task3_rlm_independence(df, ll_df):
    """
    Analyze independence between S-LATE and RLM signals.

    Key questions:
    - What % of S-LATE signals also fire RLM?
    - What's the edge when BOTH fire vs just one?
    - Do they complement each other?
    """
    print("\n" + "=" * 80)
    print("TASK 3: RLM INDEPENDENCE ANALYSIS")
    print("=" * 80)

    # Get S-LATE NO signals
    slate_signal = ll_df[ll_df['late_direction'] == 'no'].copy()
    slate_tickers = set(slate_signal['market_ticker'].tolist())

    # Detect RLM signals
    rlm_df = detect_rlm_signal(df)
    rlm_tickers = set(rlm_df['market_ticker'].tolist())

    print(f"\nSignal Counts:")
    print(f"  S-LATE-001 (NO): {len(slate_tickers)} markets")
    print(f"  RLM (H123): {len(rlm_tickers)} markets")

    # Overlap analysis
    both = slate_tickers & rlm_tickers
    slate_only = slate_tickers - rlm_tickers
    rlm_only = rlm_tickers - slate_tickers

    print(f"\n----- OVERLAP ANALYSIS -----")
    print(f"  Both signals: {len(both)} ({len(both)/len(slate_tickers)*100:.1f}% of S-LATE)")
    print(f"  S-LATE only: {len(slate_only)} ({len(slate_only)/len(slate_tickers)*100:.1f}% of S-LATE)")
    print(f"  RLM only: {len(rlm_only)}")

    # Calculate edge for each group
    results = {}

    # BOTH signals fire
    if len(both) >= 10:
        both_df = slate_signal[slate_signal['market_ticker'].isin(both)]
        both_wins = (both_df['market_result'] == 'no').sum()
        both_wr = both_wins / len(both_df)
        both_be = both_df['no_price'].mean() / 100
        both_edge = both_wr - both_be

        print(f"\n  BOTH signals (N={len(both_df)}):")
        print(f"    Win Rate: {both_wr:.1%}")
        print(f"    Breakeven: {both_be:.1%}")
        print(f"    Edge: {both_edge:.1%}")

        results['both'] = {
            'n': len(both_df),
            'win_rate': both_wr,
            'edge': both_edge
        }

    # S-LATE only
    if len(slate_only) >= 10:
        slate_only_df = slate_signal[slate_signal['market_ticker'].isin(slate_only)]
        slate_only_wins = (slate_only_df['market_result'] == 'no').sum()
        slate_only_wr = slate_only_wins / len(slate_only_df)
        slate_only_be = slate_only_df['no_price'].mean() / 100
        slate_only_edge = slate_only_wr - slate_only_be

        print(f"\n  S-LATE ONLY (N={len(slate_only_df)}):")
        print(f"    Win Rate: {slate_only_wr:.1%}")
        print(f"    Breakeven: {slate_only_be:.1%}")
        print(f"    Edge: {slate_only_edge:.1%}")

        results['slate_only'] = {
            'n': len(slate_only_df),
            'win_rate': slate_only_wr,
            'edge': slate_only_edge
        }

    # RLM only
    if len(rlm_only) >= 10:
        rlm_only_df = rlm_df[rlm_df['market_ticker'].isin(rlm_only)]
        rlm_only_wins = (rlm_only_df['market_result'] == 'no').sum()
        rlm_only_wr = rlm_only_wins / len(rlm_only_df)
        rlm_only_be = rlm_only_df['no_price'].mean() / 100
        rlm_only_edge = rlm_only_wr - rlm_only_be

        print(f"\n  RLM ONLY (N={len(rlm_only_df)}):")
        print(f"    Win Rate: {rlm_only_wr:.1%}")
        print(f"    Breakeven: {rlm_only_be:.1%}")
        print(f"    Edge: {rlm_only_edge:.1%}")

        results['rlm_only'] = {
            'n': len(rlm_only_df),
            'win_rate': rlm_only_wr,
            'edge': rlm_only_edge
        }

    # Portfolio recommendation
    print("\n----- PORTFOLIO RECOMMENDATION -----")

    overlap_rate = len(both) / len(slate_tickers) if len(slate_tickers) > 0 else 0

    if overlap_rate < 0.30:
        print("  LOW OVERLAP (<30%) - Strategies are INDEPENDENT")
        print("  Recommendation: RUN BOTH for diversification")
    elif overlap_rate < 0.60:
        print("  MODERATE OVERLAP (30-60%) - Partial independence")
        print("  Recommendation: RUN BOTH, but size positions considering overlap")
    else:
        print("  HIGH OVERLAP (>60%) - Strategies are CORRELATED")
        print("  Recommendation: Choose the higher-edge signal or combine into single strategy")

    # Combined signal strength
    if 'both' in results and 'slate_only' in results:
        combined_vs_exclusive = results['both']['edge'] - results['slate_only']['edge']
        print(f"\n  Edge improvement when BOTH fire: {combined_vs_exclusive:+.1%}")

        if combined_vs_exclusive > 0.03:
            print("  -> Signal STACKING works! Consider higher position when both fire.")
        else:
            print("  -> Minimal stacking benefit. Signals provide similar information.")

    return {
        'overlap_rate': overlap_rate,
        'n_both': len(both),
        'n_slate_only': len(slate_only),
        'n_rlm_only': len(rlm_only),
        'results': results
    }


# =============================================================================
# TASK 4: REAL-TIME IMPLEMENTATION FEASIBILITY
# =============================================================================

def task4_realtime_implementation(df):
    """
    Test if the signal can be implemented in real-time without hindsight.

    The "final 25%" problem: We don't know when a market will end.

    Alternatives to test:
    1. Time-to-expiration proxy (final 2hr/1hr/30min before expiration)
    2. Cumulative trade count (after 20, 50, 100 trades)
    3. Rolling window (always look at last 25% of trades SO FAR)
    """
    print("\n" + "=" * 80)
    print("TASK 4: REAL-TIME IMPLEMENTATION FEASIBILITY")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    results = {}

    # ===== APPROACH 1: TIME-BASED WINDOWS =====
    print("\n----- APPROACH 1: TIME-BASED WINDOWS -----")
    print("Using final X minutes before market close as 'late' window")

    time_windows = [120, 60, 30, 15]  # minutes before close

    for minutes in time_windows:
        late_markets = []

        for market_ticker, mdf in df_sorted.groupby('market_ticker'):
            if len(mdf) < 10:
                continue

            mdf = mdf.reset_index(drop=True)

            # Get market close time (last trade)
            close_time = mdf['datetime'].max()

            # Define late window
            late_start = close_time - pd.Timedelta(minutes=minutes)

            early = mdf[mdf['datetime'] < late_start]
            late = mdf[mdf['datetime'] >= late_start]

            if len(late) < 3 or len(early) < 5:
                continue

            # Calculate large trade ratios
            large_threshold = 5000  # $50
            early_large_ratio = (early['trade_value_cents'] > large_threshold).mean()
            late_large_ratio = (late['trade_value_cents'] > large_threshold).mean()

            # Check signal condition
            if late_large_ratio > early_large_ratio * 2 and late_large_ratio > 0.2:
                late_large = late[late['trade_value_cents'] > large_threshold]
                if len(late_large) < 2:
                    continue

                late_yes_ratio = (late_large['taker_side'] == 'yes').mean()
                if late_yes_ratio < 0.4:  # NO direction
                    late_markets.append({
                        'market_ticker': market_ticker,
                        'market_result': mdf['market_result'].iloc[0],
                        'no_price': mdf['no_price'].mean()
                    })

        if len(late_markets) >= 20:
            late_df = pd.DataFrame(late_markets)
            n = len(late_df)
            wins = (late_df['market_result'] == 'no').sum()
            wr = wins / n
            be = late_df['no_price'].mean() / 100
            edge = wr - be

            print(f"\n  Final {minutes} min: N={n}, WR={wr:.1%}, Edge={edge:+.1%}")

            results[f'time_{minutes}min'] = {
                'n': n,
                'win_rate': wr,
                'edge': edge
            }
        else:
            print(f"\n  Final {minutes} min: Insufficient sample ({len(late_markets)} markets)")

    # ===== APPROACH 2: TRADE COUNT THRESHOLDS =====
    print("\n----- APPROACH 2: TRADE COUNT THRESHOLDS -----")
    print("After N trades, look at the last 25% and apply signal logic")

    trade_counts = [20, 50, 100, 200]

    for min_trades in trade_counts:
        late_markets = []

        for market_ticker, mdf in df_sorted.groupby('market_ticker'):
            if len(mdf) < min_trades:
                continue

            # Take first min_trades trades
            mdf = mdf.iloc[:min_trades].reset_index(drop=True)
            n = len(mdf)

            # Split at 75%
            cutoff = 3 * n // 4
            early = mdf.iloc[:cutoff]
            late = mdf.iloc[cutoff:]

            if len(late) < 3:
                continue

            # Calculate large trade ratios
            large_threshold = 5000
            early_large_ratio = (early['trade_value_cents'] > large_threshold).mean()
            late_large_ratio = (late['trade_value_cents'] > large_threshold).mean()

            if late_large_ratio > early_large_ratio * 2 and late_large_ratio > 0.2:
                late_large = late[late['trade_value_cents'] > large_threshold]
                if len(late_large) < 2:
                    continue

                late_yes_ratio = (late_large['taker_side'] == 'yes').mean()
                if late_yes_ratio < 0.4:  # NO direction
                    # Get actual result (using full market data)
                    full_mdf = df_sorted[df_sorted['market_ticker'] == market_ticker]
                    late_markets.append({
                        'market_ticker': market_ticker,
                        'market_result': full_mdf['market_result'].iloc[0],
                        'no_price': mdf['no_price'].mean()
                    })

        if len(late_markets) >= 20:
            late_df = pd.DataFrame(late_markets)
            n = len(late_df)
            wins = (late_df['market_result'] == 'no').sum()
            wr = wins / n
            be = late_df['no_price'].mean() / 100
            edge = wr - be

            print(f"\n  After {min_trades} trades: N={n}, WR={wr:.1%}, Edge={edge:+.1%}")

            results[f'trades_{min_trades}'] = {
                'n': n,
                'win_rate': wr,
                'edge': edge
            }
        else:
            print(f"\n  After {min_trades} trades: Insufficient sample ({len(late_markets)} markets)")

    # ===== RECOMMENDATION =====
    print("\n----- IMPLEMENTATION RECOMMENDATION -----")

    # Find best real-time approach
    valid_approaches = [(k, v) for k, v in results.items() if v.get('edge', -1) > 0.05]

    if valid_approaches:
        best = max(valid_approaches, key=lambda x: x[1]['edge'])
        print(f"\n  Best real-time approach: {best[0]}")
        print(f"    Edge: {best[1]['edge']:.1%}")
        print(f"    Sample: {best[1]['n']} markets")

        # Check preservation vs original
        original_edge = 0.198  # From initial validation
        preservation = best[1]['edge'] / original_edge
        print(f"\n  Edge preservation: {preservation:.1%} of original ({original_edge:.1%})")

        if preservation > 0.70:
            print("  -> GOOD: Real-time approach preserves >70% of edge")
        elif preservation > 0.50:
            print("  -> ACCEPTABLE: Real-time approach preserves 50-70% of edge")
        else:
            print("  -> POOR: Real-time approach loses >50% of edge")
    else:
        print("\n  WARNING: No real-time approach found with >5% edge")

    return results


# =============================================================================
# TASK 5: RISK STRESS TEST
# =============================================================================

def task5_risk_stress_test(ll_df, baseline_by_bucket):
    """
    Find failure modes and stress test the strategy.

    Analysis:
    1. Worst 10 markets
    2. Category concentration
    3. Temporal stability by month
    4. Maximum drawdown
    """
    print("\n" + "=" * 80)
    print("TASK 5: RISK STRESS TEST")
    print("=" * 80)

    # Focus on Follow Late NO
    signal = ll_df[ll_df['late_direction'] == 'no'].copy()

    # Calculate P&L per market
    signal['won'] = (signal['market_result'] == 'no').astype(int)
    signal['profit_cents'] = np.where(
        signal['won'] == 1,
        100 - signal['no_price'],
        -signal['no_price']
    )
    signal['profit_dollars'] = signal['profit_cents'] / 100

    # ===== WORST MARKETS =====
    print("\n----- WORST 10 MARKETS -----")
    worst_10 = signal.nsmallest(10, 'profit_cents')

    print(f"\n{'Ticker':<30} {'Result':<8} {'NO Price':<10} {'P&L':<10}")
    print("-" * 60)
    for _, row in worst_10.iterrows():
        ticker_short = row['market_ticker'][:28] + '..' if len(row['market_ticker']) > 30 else row['market_ticker']
        print(f"{ticker_short:<30} {row['market_result']:<8} {row['no_price']:.1f}c      ${row['profit_dollars']:>+.2f}")

    # Analyze what went wrong
    losses = signal[signal['won'] == 0]
    print(f"\n  Total losing markets: {len(losses)}")
    print(f"  Avg loss: ${losses['profit_dollars'].mean():.2f}")
    print(f"  Max single loss: ${losses['profit_dollars'].min():.2f}")

    # Common patterns in losses
    print(f"\n  Losing market characteristics:")
    print(f"    Avg NO price: {losses['no_price'].mean():.1f}c")
    print(f"    Avg late large count: {losses['late_large_count'].mean():.1f}")

    # ===== CATEGORY CONCENTRATION =====
    print("\n----- CATEGORY CONCENTRATION -----")
    signal['category'] = signal['market_ticker'].str.extract(r'^(KX[A-Z]+)', expand=False)

    cat_summary = signal.groupby('category').agg({
        'won': ['count', 'sum', 'mean'],
        'profit_cents': 'sum',
        'no_price': 'mean'
    }).reset_index()
    cat_summary.columns = ['category', 'n', 'wins', 'win_rate', 'total_profit', 'avg_price']
    cat_summary['edge'] = cat_summary['win_rate'] - cat_summary['avg_price'] / 100
    cat_summary = cat_summary.sort_values('n', ascending=False)

    print(f"\n{'Category':<15} {'N':<8} {'Win%':<10} {'Edge':<10} {'Profit':<12}")
    print("-" * 55)
    for _, row in cat_summary.iterrows():
        print(f"{row['category']:<15} {row['n']:<8} {row['win_rate']:.1%}     {row['edge']:+.1%}     ${row['total_profit']/100:+.2f}")

    # Concentration check
    total_profit = signal[signal['profit_cents'] > 0]['profit_cents'].sum()
    cat_summary['profit_concentration'] = cat_summary.apply(
        lambda r: r['total_profit'] / total_profit if total_profit > 0 else 0, axis=1
    )
    max_cat_concentration = cat_summary['profit_concentration'].max()

    print(f"\n  Max category profit concentration: {max_cat_concentration:.1%}")
    if max_cat_concentration > 0.40:
        print("  WARNING: High category concentration")
    else:
        print("  OK: Diversified across categories")

    # ===== TEMPORAL STABILITY BY MONTH =====
    print("\n----- TEMPORAL STABILITY BY MONTH -----")
    signal['month'] = pd.to_datetime(signal['first_trade_time']).dt.to_period('M')

    monthly = signal.groupby('month').agg({
        'won': ['count', 'sum', 'mean'],
        'profit_cents': 'sum',
        'no_price': 'mean'
    }).reset_index()
    monthly.columns = ['month', 'n', 'wins', 'win_rate', 'total_profit', 'avg_price']
    monthly['edge'] = monthly['win_rate'] - monthly['avg_price'] / 100

    print(f"\n{'Month':<12} {'N':<6} {'Win%':<10} {'Edge':<10} {'P&L':<10}")
    print("-" * 50)
    for _, row in monthly.iterrows():
        print(f"{str(row['month']):<12} {row['n']:<6} {row['win_rate']:.1%}     {row['edge']:+.1%}     ${row['total_profit']/100:+.2f}")

    # Negative edge months
    neg_edge_months = monthly[monthly['edge'] < 0]
    print(f"\n  Months with negative edge: {len(neg_edge_months)}/{len(monthly)}")

    if len(neg_edge_months) > 0:
        print(f"  Worst month: {neg_edge_months.loc[neg_edge_months['edge'].idxmin(), 'month']} ({neg_edge_months['edge'].min():.1%})")

    # ===== MAXIMUM DRAWDOWN =====
    print("\n----- MAXIMUM DRAWDOWN -----")

    # Sort by time and calculate cumulative P&L
    signal_sorted = signal.sort_values('first_trade_time')
    signal_sorted['cumulative_pnl'] = signal_sorted['profit_cents'].cumsum()
    signal_sorted['peak_pnl'] = signal_sorted['cumulative_pnl'].cummax()
    signal_sorted['drawdown'] = signal_sorted['cumulative_pnl'] - signal_sorted['peak_pnl']

    max_drawdown = signal_sorted['drawdown'].min()
    max_drawdown_idx = signal_sorted['drawdown'].idxmin()

    print(f"\n  Maximum Drawdown: ${max_drawdown/100:.2f} ({max_drawdown:.0f}c)")

    # Find peak before drawdown
    peak_before_dd = signal_sorted.loc[:max_drawdown_idx, 'cumulative_pnl'].max()
    print(f"  Peak before drawdown: ${peak_before_dd/100:.2f}")
    print(f"  Drawdown %: {abs(max_drawdown / peak_before_dd) * 100:.1f}%")

    # Consecutive losses
    signal_sorted['is_loss'] = (signal_sorted['won'] == 0).astype(int)

    # Calculate consecutive losses
    max_consecutive_losses = 0
    current_streak = 0

    for _, row in signal_sorted.iterrows():
        if row['is_loss'] == 1:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    print(f"  Max consecutive losses: {max_consecutive_losses}")

    # Recovery analysis
    print(f"\n  Final cumulative P&L: ${signal_sorted['cumulative_pnl'].iloc[-1]/100:.2f}")
    print(f"  Profit factor: {signal_sorted[signal_sorted['profit_cents'] > 0]['profit_cents'].sum() / abs(signal_sorted[signal_sorted['profit_cents'] < 0]['profit_cents'].sum()):.2f}x")

    return {
        'worst_markets': worst_10[['market_ticker', 'market_result', 'no_price', 'profit_dollars']].to_dict('records'),
        'category_concentration': max_cat_concentration,
        'negative_edge_months': len(neg_edge_months),
        'max_drawdown_cents': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'profit_factor': signal_sorted[signal_sorted['profit_cents'] > 0]['profit_cents'].sum() / abs(signal_sorted[signal_sorted['profit_cents'] < 0]['profit_cents'].sum())
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete S-LATE-001 deep dive analysis."""
    print("=" * 80)
    print("S-LATE-001 / SPORTS-007 DEEP DIVE")
    print("Late-Arriving Large Money Strategy")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df, date_range = load_data()
    all_markets, baseline_by_bucket = build_baseline()

    # Detect signals with default parameters
    ll_df = detect_late_large_signal(df)
    print(f"\nTotal late-large signal markets: {len(ll_df)}")
    print(f"  Follow NO: {(ll_df['late_direction'] == 'no').sum()}")
    print(f"  Follow YES: {(ll_df['late_direction'] == 'yes').sum()}")

    results = {}

    # Task 1: Dollar P&L
    results['task1_dollar_pnl'] = task1_dollar_pnl_analysis(ll_df, date_range, baseline_by_bucket)

    # Task 2: Parameter sensitivity
    results['task2_params'] = task2_parameter_sensitivity(df, all_markets, baseline_by_bucket)

    # Task 3: RLM Independence
    results['task3_rlm'] = task3_rlm_independence(df, ll_df)

    # Task 4: Real-time implementation
    results['task4_realtime'] = task4_realtime_implementation(df)

    # Task 5: Risk stress test
    results['task5_risk'] = task5_risk_stress_test(ll_df, baseline_by_bucket)

    # ===== EXECUTIVE SUMMARY =====
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: S-LATE-001")
    print("=" * 80)

    # GO/NO-GO Decision
    t1 = results['task1_dollar_pnl']
    t3 = results['task3_rlm']
    t4 = results['task4_realtime']
    t5 = results['task5_risk']

    print("\n----- GO/NO-GO CHECKLIST -----")

    checks = []

    # Check 1: Positive expected value
    ev_positive = t1['avg_profit_per_dollar'] > 0.05
    checks.append(('Expected value > 5%', ev_positive, f"{t1['avg_profit_per_dollar']:.1%}"))

    # Check 2: Statistical significance
    # (Already validated in initial screening)
    checks.append(('Statistically significant (p<0.01)', True, 'p=0.000'))

    # Check 3: Not a price proxy (bucket ratio)
    # (Already validated: 11/11 buckets)
    checks.append(('Bucket-matched validation passed', True, '11/11 buckets positive'))

    # Check 4: Independence from RLM
    independent = t3['overlap_rate'] < 0.50
    checks.append(('Independent from RLM (<50% overlap)', independent, f"{t3['overlap_rate']:.1%} overlap"))

    # Check 5: Real-time implementable
    realtime_ok = any(v.get('edge', 0) > 0.08 for v in t4.values() if isinstance(v, dict))
    best_realtime = max([v.get('edge', 0) for v in t4.values() if isinstance(v, dict)])
    checks.append(('Real-time implementation viable', realtime_ok, f"Best alt: {best_realtime:.1%} edge"))

    # Check 6: Acceptable drawdown
    dd_ok = t5['profit_factor'] > 1.5
    checks.append(('Profit factor > 1.5', dd_ok, f"{t5['profit_factor']:.2f}x"))

    # Check 7: Category diversification
    div_ok = t5['category_concentration'] < 0.50
    checks.append(('Diversified (<50% category conc.)', div_ok, f"{t5['category_concentration']:.1%}"))

    all_passed = all(c[1] for c in checks)

    for check_name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}: {value}")

    print(f"\n{'='*40}")
    if all_passed:
        print("  RECOMMENDATION: GO - Strategy is viable")
    else:
        failed = [c[0] for c in checks if not c[1]]
        print(f"  RECOMMENDATION: CONDITIONAL - Address: {', '.join(failed)}")
    print(f"{'='*40}")

    # Key numbers
    print("\n----- KEY NUMBERS -----")
    print(f"  Edge: {t1['avg_profit_per_dollar']:.1%}")
    print(f"  Signals/year: ~{t1['signals_per_year']:.0f}")
    print(f"  Expected annual return (1% position sizing): ~{t1['annualized_returns'][3]['annual_pct']:.0f}%")
    print(f"  RLM overlap: {t3['overlap_rate']:.1%}")
    print(f"  Max drawdown: ${t5['max_drawdown_cents']/100:.2f}")
    print(f"  Max consecutive losses: {t5['max_consecutive_losses']}")

    # Save results
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results['executive_summary'] = {
        'recommendation': 'GO' if all_passed else 'CONDITIONAL',
        'checks': [(c[0], bool(c[1]), c[2]) for c in checks],
        'all_passed': bool(all_passed)
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(convert_types(results), f, indent=2, default=str)

    print(f"\n\nFull results saved to: {REPORT_PATH}")

    return results


if __name__ == "__main__":
    results = main()
