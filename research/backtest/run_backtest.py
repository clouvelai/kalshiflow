#!/usr/bin/env python3
"""
Point-in-Time Backtest Runner

Runs strategies through the point-in-time backtesting engine and validates results.

Usage:
    cd research/backtest
    python run_backtest.py --strategy rlm
    python run_backtest.py --strategy slate
    python run_backtest.py --strategy all

    # Or from backend directory:
    uv run python ../research/backtest/run_backtest.py --strategy all
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Set up paths for both package and direct imports
_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
# Also add strategies directory
_STRATEGIES_DIR = _THIS_DIR / 'strategies'
if str(_STRATEGIES_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGIES_DIR))

import pandas as pd

from state import Trade
from engine import PointInTimeBacktester
from validation import validate_results, ValidationReport
from strategies import RLMNoStrategy, SLateTimingStrategy


# Data paths
DATA_PATH = Path(__file__).parent.parent / 'data' / 'trades' / 'enriched_trades_resolved_ALL.csv'
REPORTS_PATH = Path(__file__).parent.parent / 'reports'


def load_trades_and_settlements(limit: int = None) -> tuple[list[Trade], dict[str, str]]:
    """
    Load trades from CSV and extract settlements.

    Returns:
        (trades, settlements) where settlements maps market_ticker -> 'yes' or 'no'
    """
    print(f"Loading data from {DATA_PATH}...")

    df = pd.read_csv(DATA_PATH)

    if limit:
        # Get first N unique markets for testing
        unique_markets = df['market_ticker'].unique()[:limit]
        df = df[df['market_ticker'].isin(unique_markets)]

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Extract settlements
    settlements = {}
    for ticker, group in df.groupby('market_ticker'):
        result = group['result'].iloc[0]
        if pd.notna(result) and result in ('yes', 'no'):
            settlements[ticker] = result

    print(f"Found {len(settlements):,} markets with settlements")

    # Convert to Trade objects
    trades = []
    for _, row in df.iterrows():
        try:
            trades.append(Trade(
                market_ticker=row['market_ticker'],
                timestamp=pd.to_datetime(row['datetime']),
                yes_price=int(row['yes_price']),
                taker_side=row['taker_side'],
                count=int(row['count']),
                trade_id=str(row.get('id', ''))
            ))
        except Exception as e:
            # Skip invalid rows
            continue

    print(f"Converted {len(trades):,} valid trades")

    return trades, settlements


def run_rlm_backtest(trades: list[Trade], settlements: dict[str, str]) -> tuple:
    """Run RLM NO strategy backtest."""
    print("\n" + "=" * 60)
    print("RLM NO STRATEGY - POINT-IN-TIME BACKTEST")
    print("=" * 60)

    strategy = RLMNoStrategy()
    print(f"\nParameters: {strategy.get_parameters()}")

    backtester = PointInTimeBacktester(strategy)

    def progress(n, total):
        print(f"  Processed {n:,}/{total:,} trades ({100*n/total:.1f}%)")

    results = backtester.run(trades, settlements, progress_callback=progress)

    print(f"\n--- RAW RESULTS ---")
    print(f"Signals: {results.n_signals}")
    print(f"Wins: {results.n_wins}")
    print(f"Losses: {results.n_losses}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Raw Edge: {results.raw_edge:.2%}")
    print(f"Total P&L: ${results.total_pnl_cents / 100:.2f}")

    # Validate
    report = validate_results(results, settlements)

    print(f"\n--- VALIDATION ---")
    print(f"Bucket-Matched Edge: {report.bucket_matched_edge:.2%}")
    print(f"P-Value: {report.p_value:.6f}")
    print(f"Statistically Significant: {report.is_significant}")
    print(f"Max Market Concentration: {report.max_market_concentration:.2%}")
    print(f"Passes Validation: {report.passes_validation}")

    if report.failure_reasons:
        print(f"\nFailure Reasons:")
        for reason in report.failure_reasons:
            print(f"  - {reason}")

    # Bucket breakdown
    print(f"\n--- BUCKET ANALYSIS ---")
    for bucket_key in sorted(report.bucket_stats.keys(), key=lambda x: int(x.split('-')[0])):
        stats = report.bucket_stats[bucket_key]
        print(f"  {bucket_key}c: N={stats.n_signals:4}, WR={stats.win_rate:.1%}, Expected={stats.expected_win_rate:.1%}, Edge={stats.edge:+.1%}")

    return results, report


def run_slate_backtest(trades: list[Trade], settlements: dict[str, str]) -> tuple:
    """Run S-LATE TIMING strategy backtest."""
    print("\n" + "=" * 60)
    print("S-LATE TIMING STRATEGY - POINT-IN-TIME BACKTEST")
    print("=" * 60)

    strategy = SLateTimingStrategy()
    print(f"\nParameters: {strategy.get_parameters()}")

    backtester = PointInTimeBacktester(strategy)

    def progress(n, total):
        print(f"  Processed {n:,}/{total:,} trades ({100*n/total:.1f}%)")

    results = backtester.run(trades, settlements, progress_callback=progress)

    print(f"\n--- RAW RESULTS ---")
    print(f"Signals: {results.n_signals}")
    print(f"Wins: {results.n_wins}")
    print(f"Losses: {results.n_losses}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Raw Edge: {results.raw_edge:.2%}")
    print(f"Total P&L: ${results.total_pnl_cents / 100:.2f}")

    # Validate
    report = validate_results(results, settlements)

    print(f"\n--- VALIDATION ---")
    print(f"Bucket-Matched Edge: {report.bucket_matched_edge:.2%}")
    print(f"P-Value: {report.p_value:.6f}")
    print(f"Statistically Significant: {report.is_significant}")
    print(f"Max Market Concentration: {report.max_market_concentration:.2%}")
    print(f"Passes Validation: {report.passes_validation}")

    if report.failure_reasons:
        print(f"\nFailure Reasons:")
        for reason in report.failure_reasons:
            print(f"  - {reason}")

    # Bucket breakdown
    print(f"\n--- BUCKET ANALYSIS ---")
    for bucket_key in sorted(report.bucket_stats.keys(), key=lambda x: int(x.split('-')[0])):
        stats = report.bucket_stats[bucket_key]
        print(f"  {bucket_key}c: N={stats.n_signals:4}, WR={stats.win_rate:.1%}, Expected={stats.expected_win_rate:.1%}, Edge={stats.edge:+.1%}")

    return results, report


def compare_strategies(rlm_results, slate_results, settlements: dict[str, str]):
    """Compare RLM and S-LATE strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    # Signal overlap analysis
    rlm_markets = {s.market_ticker for s in rlm_results.signals}
    slate_markets = {s.market_ticker for s in slate_results.signals}

    both = rlm_markets & slate_markets
    rlm_only = rlm_markets - slate_markets
    slate_only = slate_markets - rlm_markets

    print(f"\n--- SIGNAL OVERLAP ---")
    print(f"RLM signals: {len(rlm_markets)}")
    print(f"S-LATE signals: {len(slate_markets)}")
    print(f"Both fire: {len(both)} ({100*len(both)/len(rlm_markets):.1f}% of RLM)")
    print(f"RLM only: {len(rlm_only)}")
    print(f"S-LATE only: {len(slate_only)}")

    # Combined portfolio potential
    all_unique = rlm_markets | slate_markets
    print(f"\nCombined unique signals: {len(all_unique)}")

    # Edge comparison
    print(f"\n--- EDGE COMPARISON ---")
    print(f"{'Strategy':<20} {'Signals':<10} {'Win Rate':<12} {'Raw Edge':<12} {'P&L':<12}")
    print("-" * 66)
    print(f"{'RLM NO':<20} {rlm_results.n_signals:<10} {rlm_results.win_rate:.1%}        {rlm_results.raw_edge:+.1%}        ${rlm_results.total_pnl_cents/100:>8.2f}")
    print(f"{'S-LATE TIMING':<20} {slate_results.n_signals:<10} {slate_results.win_rate:.1%}        {slate_results.raw_edge:+.1%}        ${slate_results.total_pnl_cents/100:>8.2f}")


def save_results(strategy_name: str, results, report: ValidationReport):
    """Save results to JSON report."""
    REPORTS_PATH.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = REPORTS_PATH / f'{strategy_name}_point_in_time_{timestamp}.json'

    # Build JSON-serializable report
    report_data = {
        'strategy': strategy_name,
        'timestamp': timestamp,
        'methodology': 'point_in_time',
        'results': {
            'n_signals': results.n_signals,
            'n_wins': results.n_wins,
            'n_losses': results.n_losses,
            'n_unresolved': results.n_unresolved,
            'win_rate': results.win_rate,
            'raw_edge': results.raw_edge,
            'total_pnl_cents': results.total_pnl_cents
        },
        'validation': {
            'bucket_matched_edge': report.bucket_matched_edge,
            'p_value': report.p_value,
            'is_significant': report.is_significant,
            'max_market_concentration': report.max_market_concentration,
            'concentration_ok': report.concentration_ok,
            'passes_validation': report.passes_validation,
            'failure_reasons': report.failure_reasons
        },
        'bucket_stats': {
            k: {
                'n_signals': v.n_signals,
                'n_wins': v.n_wins,
                'win_rate': v.win_rate,
                'expected_win_rate': v.expected_win_rate,
                'edge': v.edge
            }
            for k, v in report.bucket_stats.items()
        }
    }

    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run point-in-time backtests')
    parser.add_argument('--strategy', choices=['rlm', 'slate', 'all'], default='all',
                       help='Strategy to backtest')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit to N markets for testing')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON')

    args = parser.parse_args()

    # Load data
    trades, settlements = load_trades_and_settlements(limit=args.limit)

    rlm_results = None
    slate_results = None

    # Run backtests
    if args.strategy in ('rlm', 'all'):
        rlm_results, rlm_report = run_rlm_backtest(trades, settlements)
        if args.save:
            save_results('rlm_no', rlm_results, rlm_report)

    if args.strategy in ('slate', 'all'):
        slate_results, slate_report = run_slate_backtest(trades, settlements)
        if args.save:
            save_results('s_late_timing', slate_results, slate_report)

    # Compare if both ran
    if rlm_results and slate_results:
        compare_strategies(rlm_results, slate_results, settlements)

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    # Summary
    print("\n--- KEY TAKEAWAYS ---")
    if rlm_results:
        print(f"RLM NO: {rlm_results.n_signals} signals, {rlm_results.raw_edge:+.1%} raw edge")
    if slate_results:
        print(f"S-LATE: {slate_results.n_signals} signals, {slate_results.raw_edge:+.1%} raw edge")

    print("\nNOTE: These are POINT-IN-TIME results - signals fired at first trigger")
    print("Compare to flawed backtest results which used FINAL market statistics")


if __name__ == '__main__':
    main()
