"""
RLM_NO Optimization - LSD Mode Research Session

Analyzes:
1. Signal parameter optimization (price drop thresholds, YES thresholds)
2. Expected value tradeoffs (edge vs frequency)
3. Order pricing analysis
4. Category-specific performance
5. LSD explorations (time-of-day, velocity, signal decay)

LSD MODE: Speed over rigor, flag anything >5% raw edge!
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets" / "market_outcomes_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load trades and market outcomes."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE)
    markets = pd.read_csv(MARKETS_FILE)

    print(f"  Trades: {len(trades):,}")
    print(f"  Markets: {len(markets):,}")

    # Rename columns to expected format
    # File has: id,market_ticker,taker_side,count,yes_price,no_price,timestamp,datetime,trade_price,...
    trades = trades.rename(columns={
        'datetime': 'created_time',
        'trade_price': 'price_cents',
    })

    # Ensure timestamp parsing
    if 'created_time' in trades.columns:
        trades['created_time'] = pd.to_datetime(trades['created_time'])

    return trades, markets


def compute_rlm_signal(market_trades: pd.DataFrame,
                        yes_threshold: float = 0.65,
                        min_trades: int = 15,
                        min_price_drop: int = 0) -> Dict[str, Any]:
    """
    Compute RLM signal for a market.

    Returns signal data including:
    - yes_ratio: proportion of YES trades
    - price_drop: first_yes_price - last_yes_price
    - trade_count: total trades
    - signal_triggered: whether signal fired
    """
    if len(market_trades) < min_trades:
        return {'signal_triggered': False, 'reason': 'insufficient_trades'}

    # Compute YES ratio
    total_trades = len(market_trades)
    yes_trades = (market_trades['taker_side'] == 'yes').sum()
    yes_ratio = yes_trades / total_trades

    if yes_ratio <= yes_threshold:
        return {'signal_triggered': False, 'reason': 'low_yes_ratio', 'yes_ratio': yes_ratio}

    # YES price is already in the data
    first_yes_price = market_trades['yes_price'].iloc[0]
    last_yes_price = market_trades['yes_price'].iloc[-1]
    price_drop = first_yes_price - last_yes_price

    if price_drop < min_price_drop:
        return {
            'signal_triggered': False,
            'reason': 'insufficient_price_drop',
            'price_drop': price_drop,
            'yes_ratio': yes_ratio
        }

    # Compute NO price at signal time (for entry)
    # NO price = 100 - YES price
    no_price_entry = 100 - last_yes_price

    return {
        'signal_triggered': True,
        'yes_ratio': yes_ratio,
        'price_drop': price_drop,
        'trade_count': total_trades,
        'first_yes_price': first_yes_price,
        'last_yes_price': last_yes_price,
        'no_price_entry': no_price_entry,
    }


def compute_baseline_win_rate(no_price_bucket: int, baseline_data: pd.DataFrame) -> float:
    """Get baseline win rate for a NO price bucket."""
    bucket_data = baseline_data[baseline_data['no_price_bucket'] == no_price_bucket]
    if len(bucket_data) == 0:
        return None
    return bucket_data['no_win'].mean()


def analyze_price_drop_threshold(trades: pd.DataFrame,
                                  markets: pd.DataFrame,
                                  yes_threshold: float = 0.65,
                                  min_trades: int = 15) -> Dict[str, Any]:
    """
    Analyze different price drop thresholds.

    This is the KEY analysis: 2c vs 5c tradeoff.
    """
    print("\n" + "="*60)
    print("SIGNAL PARAMETER ANALYSIS: Price Drop Threshold")
    print("="*60)

    # Build per-market trade data
    market_signals = {}

    for ticker, group in trades.groupby('market_ticker'):
        group = group.sort_values('created_time')

        # Compute signal at all thresholds
        base_signal = compute_rlm_signal(group, yes_threshold, min_trades, 0)
        if base_signal['signal_triggered']:
            market_signals[ticker] = {
                'yes_ratio': base_signal['yes_ratio'],
                'price_drop': base_signal['price_drop'],
                'trade_count': base_signal['trade_count'],
                'no_price_entry': base_signal['no_price_entry'],
            }

    # Merge with outcomes
    signals_df = pd.DataFrame.from_dict(market_signals, orient='index')
    signals_df['market_ticker'] = signals_df.index

    # Market outcomes has 'result' column with 'yes'/'no'
    markets_subset = markets[['ticker', 'result']].copy()
    markets_subset['result_yes'] = (markets_subset['result'] == 'yes').astype(int)

    signals_df = signals_df.merge(markets_subset[['ticker', 'result_yes']],
                                   left_on='market_ticker', right_on='ticker', how='inner')
    signals_df['no_win'] = 1 - signals_df['result_yes']  # We're betting NO

    print(f"\nTotal markets with RLM signal (any price drop): {len(signals_df)}")

    # Analyze at different price drop thresholds
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    results = []

    for drop_threshold in thresholds:
        filtered = signals_df[signals_df['price_drop'] >= drop_threshold]

        if len(filtered) < 50:
            continue

        n_markets = len(filtered)
        win_rate = filtered['no_win'].mean()
        avg_no_price = filtered['no_price_entry'].mean()
        breakeven = avg_no_price / 100
        edge = win_rate - breakeven

        # Expected value per $100 bet
        # EV = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price
        ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price

        # Total expected profit = EV * N
        total_ev = ev_per_100 * n_markets

        results.append({
            'price_drop': drop_threshold,
            'markets': n_markets,
            'win_rate': win_rate,
            'avg_no_price': avg_no_price,
            'breakeven': breakeven,
            'edge': edge,
            'ev_per_100': ev_per_100,
            'total_ev': total_ev,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "-"*80)
    print("Price Drop Threshold Analysis")
    print("-"*80)
    print(f"{'Drop':>6} {'Markets':>8} {'WinRate':>8} {'AvgNO':>8} {'Edge':>8} {'EV/100':>10} {'TotalEV':>12}")
    print("-"*80)

    for _, row in results_df.iterrows():
        print(f"{row['price_drop']:>4}c {row['markets']:>8,} {row['win_rate']:>7.1%} "
              f"{row['avg_no_price']:>7.1f}c {row['edge']:>7.1%} "
              f"${row['ev_per_100']:>8.2f} ${row['total_ev']:>10,.0f}")

    # Find optimal based on total expected value
    optimal_idx = results_df['total_ev'].idxmax()
    optimal = results_df.iloc[optimal_idx]

    print("\n" + "="*60)
    print(f"OPTIMAL PRICE DROP THRESHOLD: {optimal['price_drop']}c")
    print(f"  Markets: {optimal['markets']:,}")
    print(f"  Edge: {optimal['edge']:.1%}")
    print(f"  Total Expected Value: ${optimal['total_ev']:,.0f}")
    print("="*60)

    return {
        'results': results_df.to_dict('records'),
        'optimal': optimal.to_dict(),
        'signals_df': signals_df
    }


def analyze_yes_threshold(signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze different YES ratio thresholds.
    """
    print("\n" + "="*60)
    print("SIGNAL PARAMETER ANALYSIS: YES Ratio Threshold")
    print("="*60)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    for yes_thresh in thresholds:
        filtered = signals_df[signals_df['yes_ratio'] > yes_thresh]

        if len(filtered) < 50:
            continue

        n_markets = len(filtered)
        win_rate = filtered['no_win'].mean()
        avg_no_price = filtered['no_price_entry'].mean()
        breakeven = avg_no_price / 100
        edge = win_rate - breakeven
        ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price
        total_ev = ev_per_100 * n_markets

        results.append({
            'yes_threshold': yes_thresh,
            'markets': n_markets,
            'win_rate': win_rate,
            'avg_no_price': avg_no_price,
            'edge': edge,
            'ev_per_100': ev_per_100,
            'total_ev': total_ev,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "-"*80)
    print("YES Ratio Threshold Analysis")
    print("-"*80)
    print(f"{'Thresh':>8} {'Markets':>8} {'WinRate':>8} {'AvgNO':>8} {'Edge':>8} {'EV/100':>10} {'TotalEV':>12}")
    print("-"*80)

    for _, row in results_df.iterrows():
        print(f"{row['yes_threshold']:>7.0%} {row['markets']:>8,} {row['win_rate']:>7.1%} "
              f"{row['avg_no_price']:>7.1f}c {row['edge']:>7.1%} "
              f"${row['ev_per_100']:>8.2f} ${row['total_ev']:>10,.0f}")

    return {'results': results_df.to_dict('records')}


def analyze_category_performance(trades: pd.DataFrame,
                                  markets: pd.DataFrame,
                                  signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze RLM performance by market category.
    """
    print("\n" + "="*60)
    print("CATEGORY PERFORMANCE ANALYSIS")
    print("="*60)

    # Categorize markets by ticker prefix
    def get_category(ticker: str) -> str:
        ticker = ticker.upper()
        if ticker.startswith(('KXNFL', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAF', 'KXNCAAMB', 'KXEPL', 'KXMVE')):
            return 'Sports'
        elif ticker.startswith(('KXBTC', 'KXETH', 'KXDOGE', 'KXXRP')):
            return 'Crypto'
        elif ticker.startswith(('KXMRBEAST', 'KXCOLBERT', 'KXSNL', 'KXALTMAN', 'KXSWIFT', 'KXMENTION')):
            return 'Media_Mentions'
        elif ticker.startswith(('KXNETFLIX', 'KXSPOTIFY', 'KXGG', 'KXBILLBOARD', 'KXRANK')):
            return 'Entertainment'
        elif ticker.startswith(('KXHIGH', 'KXRAIN', 'KXSNOW')):
            return 'Weather'
        elif ticker.startswith(('KXTRUMP', 'KXAPR', 'KXPRES')):
            return 'Politics'
        elif ticker.startswith(('KXNASDAQ', 'FED', 'KXCPI', 'KXPAYROLL')):
            return 'Economics'
        else:
            return 'Other'

    signals_df = signals_df.copy()
    signals_df['category'] = signals_df['market_ticker'].apply(get_category)

    results = []
    for category, group in signals_df.groupby('category'):
        if len(group) < 10:
            continue

        n_markets = len(group)
        win_rate = group['no_win'].mean()
        avg_no_price = group['no_price_entry'].mean()
        breakeven = avg_no_price / 100
        edge = win_rate - breakeven
        ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price
        total_ev = ev_per_100 * n_markets

        results.append({
            'category': category,
            'markets': n_markets,
            'win_rate': win_rate,
            'avg_no_price': avg_no_price,
            'edge': edge,
            'ev_per_100': ev_per_100,
            'total_ev': total_ev,
        })

    results_df = pd.DataFrame(results).sort_values('edge', ascending=False)

    print("\n" + "-"*90)
    print(f"{'Category':<20} {'Markets':>8} {'WinRate':>8} {'AvgNO':>8} {'Edge':>8} {'EV/100':>10} {'Verdict':<12}")
    print("-"*90)

    for _, row in results_df.iterrows():
        verdict = "INCLUDE" if row['edge'] > 0.10 else ("WEAK" if row['edge'] > 0.05 else "EXCLUDE")
        print(f"{row['category']:<20} {row['markets']:>8,} {row['win_rate']:>7.1%} "
              f"{row['avg_no_price']:>7.1f}c {row['edge']:>7.1%} "
              f"${row['ev_per_100']:>8.2f} {verdict:<12}")

    return {'results': results_df.to_dict('records')}


def analyze_time_of_day(trades: pd.DataFrame,
                         signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    LSD Exploration: Time-of-day patterns.
    """
    print("\n" + "="*60)
    print("LSD EXPLORATION: Time-of-Day Patterns")
    print("="*60)

    # Get hour of first trade for each market
    market_first_trade = trades.groupby('market_ticker').first()['created_time'].to_dict()

    signals_df = signals_df.copy()
    signals_df['first_trade_time'] = signals_df['market_ticker'].map(market_first_trade)
    signals_df['hour'] = pd.to_datetime(signals_df['first_trade_time']).dt.hour

    results = []
    for hour, group in signals_df.groupby('hour'):
        if len(group) < 20:
            continue

        n_markets = len(group)
        win_rate = group['no_win'].mean()
        avg_no_price = group['no_price_entry'].mean()
        edge = win_rate - (avg_no_price / 100)

        results.append({
            'hour': hour,
            'markets': n_markets,
            'win_rate': win_rate,
            'avg_no_price': avg_no_price,
            'edge': edge,
        })

    results_df = pd.DataFrame(results).sort_values('hour')

    print("\n" + "-"*70)
    print(f"{'Hour':>6} {'Markets':>8} {'WinRate':>8} {'AvgNO':>8} {'Edge':>8} {'Flag':<10}")
    print("-"*70)

    for _, row in results_df.iterrows():
        flag = "** HIGH **" if row['edge'] > 0.20 else ""
        print(f"{int(row['hour']):>6} {row['markets']:>8,} {row['win_rate']:>7.1%} "
              f"{row['avg_no_price']:>7.1f}c {row['edge']:>7.1%} {flag}")

    return {'results': results_df.to_dict('records')}


def analyze_signal_strength(signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    LSD Exploration: Does stronger signal = better edge?
    """
    print("\n" + "="*60)
    print("LSD EXPLORATION: Signal Strength vs Edge")
    print("="*60)

    # Create signal strength buckets
    signals_df = signals_df.copy()

    # YES ratio strength
    print("\n--- YES Ratio Strength ---")
    signals_df['yes_ratio_bucket'] = pd.cut(signals_df['yes_ratio'],
                                             bins=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
                                             labels=['65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-100%'])

    for bucket, group in signals_df.groupby('yes_ratio_bucket'):
        if len(group) < 20:
            continue
        win_rate = group['no_win'].mean()
        avg_no_price = group['no_price_entry'].mean()
        edge = win_rate - (avg_no_price / 100)
        print(f"  {bucket}: N={len(group):,}, Edge={edge:.1%}")

    # Price drop strength
    print("\n--- Price Drop Strength ---")
    signals_df['drop_bucket'] = pd.cut(signals_df['price_drop'],
                                        bins=[0, 2, 5, 10, 20, 50],
                                        labels=['0-2c', '2-5c', '5-10c', '10-20c', '20+c'])

    for bucket, group in signals_df.groupby('drop_bucket'):
        if len(group) < 20:
            continue
        win_rate = group['no_win'].mean()
        avg_no_price = group['no_price_entry'].mean()
        edge = win_rate - (avg_no_price / 100)
        print(f"  {bucket}: N={len(group):,}, Edge={edge:.1%}")

    return {}


def analyze_trade_velocity(trades: pd.DataFrame,
                            signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    LSD Exploration: Trade velocity bursts as signal amplifier.
    """
    print("\n" + "="*60)
    print("LSD EXPLORATION: Trade Velocity Patterns")
    print("="*60)

    # Compute trades per minute for each market
    velocity_data = {}
    for ticker, group in trades.groupby('market_ticker'):
        group = group.sort_values('created_time')
        if len(group) < 10:
            continue

        # Time span in minutes
        time_span = (group['created_time'].max() - group['created_time'].min()).total_seconds() / 60
        if time_span < 1:
            time_span = 1

        trades_per_min = len(group) / time_span
        velocity_data[ticker] = trades_per_min

    signals_df = signals_df.copy()
    signals_df['trades_per_min'] = signals_df['market_ticker'].map(velocity_data)
    signals_df = signals_df.dropna(subset=['trades_per_min'])

    # Bucket by velocity
    signals_df['velocity_bucket'] = pd.qcut(signals_df['trades_per_min'],
                                             q=4, labels=['Slow', 'Medium', 'Fast', 'Burst'])

    print("\n" + "-"*60)
    for bucket, group in signals_df.groupby('velocity_bucket'):
        if len(group) < 20:
            continue
        win_rate = group['no_win'].mean()
        avg_no_price = group['no_price_entry'].mean()
        edge = win_rate - (avg_no_price / 100)
        avg_velocity = group['trades_per_min'].mean()
        print(f"  {bucket} ({avg_velocity:.1f}/min): N={len(group):,}, Edge={edge:.1%}")

    return {}


def combined_parameter_grid(trades: pd.DataFrame,
                             markets: pd.DataFrame) -> Dict[str, Any]:
    """
    Grid search over combined parameters.
    Find optimal (yes_threshold, min_trades, price_drop) combination.
    """
    print("\n" + "="*60)
    print("COMBINED PARAMETER GRID SEARCH")
    print("="*60)

    yes_thresholds = [0.60, 0.65, 0.70, 0.75]
    min_trades_options = [10, 15, 20]
    price_drops = [0, 2, 3, 5]

    all_results = []

    for yes_thresh in yes_thresholds:
        for min_trades in min_trades_options:
            for price_drop in price_drops:
                # Compute signals
                market_signals = {}
                for ticker, group in trades.groupby('market_ticker'):
                    group = group.sort_values('created_time')
                    signal = compute_rlm_signal(group, yes_thresh, min_trades, price_drop)
                    if signal['signal_triggered']:
                        market_signals[ticker] = signal

                if len(market_signals) < 50:
                    continue

                # Build dataframe
                signals_df = pd.DataFrame.from_dict(market_signals, orient='index')
                signals_df['market_ticker'] = signals_df.index

                # Market outcomes has 'result' column with 'yes'/'no'
                markets_subset = markets[['ticker', 'result']].copy()
                markets_subset['result_yes'] = (markets_subset['result'] == 'yes').astype(int)

                signals_df = signals_df.merge(markets_subset[['ticker', 'result_yes']],
                                               left_on='market_ticker', right_on='ticker', how='inner')
                signals_df['no_win'] = 1 - signals_df['result_yes']

                n_markets = len(signals_df)
                win_rate = signals_df['no_win'].mean()
                avg_no_price = signals_df['no_price_entry'].mean()
                breakeven = avg_no_price / 100
                edge = win_rate - breakeven
                ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price
                total_ev = ev_per_100 * n_markets

                all_results.append({
                    'yes_threshold': yes_thresh,
                    'min_trades': min_trades,
                    'price_drop': price_drop,
                    'markets': n_markets,
                    'win_rate': win_rate,
                    'avg_no_price': avg_no_price,
                    'edge': edge,
                    'ev_per_100': ev_per_100,
                    'total_ev': total_ev,
                })

    results_df = pd.DataFrame(all_results)

    # Find optimal by total EV
    optimal_idx = results_df['total_ev'].idxmax()
    optimal = results_df.iloc[optimal_idx]

    # Also find optimal by edge
    edge_optimal_idx = results_df['edge'].idxmax()
    edge_optimal = results_df.iloc[edge_optimal_idx]

    print("\n" + "-"*100)
    print("Top 10 by Total Expected Value:")
    print("-"*100)
    print(f"{'YesT':>6} {'MinT':>6} {'Drop':>6} {'Mkts':>8} {'WinRate':>8} {'AvgNO':>8} {'Edge':>8} {'EV/100':>10} {'TotalEV':>12}")
    print("-"*100)

    for _, row in results_df.nlargest(10, 'total_ev').iterrows():
        print(f"{row['yes_threshold']:>5.0%} {row['min_trades']:>6} {row['price_drop']:>4}c "
              f"{row['markets']:>8,} {row['win_rate']:>7.1%} {row['avg_no_price']:>7.1f}c "
              f"{row['edge']:>7.1%} ${row['ev_per_100']:>8.2f} ${row['total_ev']:>10,.0f}")

    print("\n" + "="*60)
    print("OPTIMAL BY TOTAL EV:")
    print(f"  YES Threshold: {optimal['yes_threshold']:.0%}")
    print(f"  Min Trades: {optimal['min_trades']}")
    print(f"  Price Drop: {optimal['price_drop']}c")
    print(f"  Markets: {optimal['markets']:,}")
    print(f"  Edge: {optimal['edge']:.1%}")
    print(f"  Total EV: ${optimal['total_ev']:,.0f}")
    print("="*60)

    print("\nOPTIMAL BY EDGE:")
    print(f"  YES Threshold: {edge_optimal['yes_threshold']:.0%}")
    print(f"  Min Trades: {edge_optimal['min_trades']}")
    print(f"  Price Drop: {edge_optimal['price_drop']}c")
    print(f"  Markets: {edge_optimal['markets']:,}")
    print(f"  Edge: {edge_optimal['edge']:.1%}")
    print(f"  Total EV: ${edge_optimal['total_ev']:,.0f}")

    return {
        'all_results': results_df.to_dict('records'),
        'optimal_by_ev': optimal.to_dict(),
        'optimal_by_edge': edge_optimal.to_dict(),
    }


def main():
    """Run full RLM optimization analysis."""
    print("="*60)
    print("RLM_NO OPTIMIZATION - LSD MODE RESEARCH SESSION")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    trades, markets = load_data()

    # 1. Price drop threshold analysis
    price_drop_results = analyze_price_drop_threshold(trades, markets)

    # 2. YES threshold analysis (using signals from step 1)
    yes_threshold_results = analyze_yes_threshold(price_drop_results['signals_df'])

    # 3. Category analysis
    category_results = analyze_category_performance(trades, markets, price_drop_results['signals_df'])

    # 4. Combined grid search
    grid_results = combined_parameter_grid(trades, markets)

    # 5. LSD Explorations
    print("\n" + "="*60)
    print("LSD EXPLORATIONS")
    print("="*60)

    time_results = analyze_time_of_day(trades, price_drop_results['signals_df'])
    strength_results = analyze_signal_strength(price_drop_results['signals_df'])
    velocity_results = analyze_trade_velocity(trades, price_drop_results['signals_df'])

    # Compile results
    final_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'mode': 'LSD',
            'trades_analyzed': len(trades),
            'markets_analyzed': len(markets),
        },
        'price_drop_analysis': price_drop_results['results'],
        'price_drop_optimal': price_drop_results['optimal'],
        'yes_threshold_analysis': yes_threshold_results['results'],
        'category_analysis': category_results['results'],
        'grid_search': grid_results,
        'lsd_time_of_day': time_results.get('results', []),
    }

    # Save results
    output_file = REPORTS_DIR / 'rlm_optimization_lsd.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print executive summary
    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)

    ev_optimal = grid_results['optimal_by_ev']
    edge_optimal = grid_results['optimal_by_edge']

    print("\n1. SIGNAL PARAMETER RECOMMENDATION:")
    print(f"   Current: YES>65%, min_trades=15, price_drop>=2c")
    print(f"   Optimal by Total EV: YES>{ev_optimal['yes_threshold']:.0%}, min_trades={ev_optimal['min_trades']}, price_drop>={ev_optimal['price_drop']}c")
    print(f"   Optimal by Edge: YES>{edge_optimal['yes_threshold']:.0%}, min_trades={edge_optimal['min_trades']}, price_drop>={edge_optimal['price_drop']}c")

    print("\n2. 2c vs 5c PRICE DROP TRADEOFF:")
    for r in price_drop_results['results']:
        if r['price_drop'] in [2, 5]:
            print(f"   {r['price_drop']}c: {r['markets']:,} markets, {r['edge']:.1%} edge, ${r['total_ev']:,.0f} total EV")

    print("\n3. CATEGORY FILTERING:")
    for r in sorted(category_results['results'], key=lambda x: x['edge'], reverse=True)[:5]:
        verdict = "INCLUDE" if r['edge'] > 0.10 else "WEAK"
        print(f"   {r['category']}: {r['edge']:.1%} edge ({verdict})")

    return final_results


if __name__ == "__main__":
    results = main()
