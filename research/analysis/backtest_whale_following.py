#!/usr/bin/env python3
"""
Whale Following Strategy Backtester

Backtests a simple strategy: follow whale trades that match certain criteria.

Strategy Variants:
1. Follow all whales (>=100 contracts)
2. Follow whale longshots (>=100 contracts at <=15c)
3. Follow mega-whales (>=1000 contracts)
4. Follow whale extreme bets (<=15c or >=85c)

Simulates:
- Entry at trade price (assume we can get same price)
- Position sizing based on fixed dollar amount or contract count
- Simple binary outcome (win/lose at settlement)

Metrics:
- Win rate
- ROI (return on investment)
- Sharpe-like risk-adjusted return
- Drawdown analysis
- Per-trade P/L distribution

Usage:
    python backtest_whale_following.py --enriched enriched_trades.csv --strategy whale_longshot

Author: Claude Code
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade data for backtesting."""
    id: int
    market_ticker: str
    taker_side: str
    count: int
    trade_price: int  # cents
    cost_dollars: float
    timestamp: int
    is_winner: bool
    actual_profit_dollars: float

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_wagered: float
    total_pnl: float
    roi: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    trades_by_month: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pnl_series: List[float] = field(default_factory=list)


class WhaleFollowingBacktester:
    """Backtests whale following strategies."""

    STRATEGIES = {
        "all_whales": {
            "name": "All Whales (>=100 contracts)",
            "filter": lambda t: t.count >= 100,
            "description": "Follow all trades with 100+ contracts"
        },
        "mega_whales": {
            "name": "Mega Whales (>=1000 contracts)",
            "filter": lambda t: t.count >= 1000,
            "description": "Follow only the largest trades (1000+ contracts)"
        },
        "whale_longshot": {
            "name": "Whale Longshots (>=100 @ <=15c)",
            "filter": lambda t: t.count >= 100 and t.trade_price <= 15,
            "description": "Follow large bets on unlikely outcomes"
        },
        "whale_favorite": {
            "name": "Whale Favorites (>=100 @ >=85c)",
            "filter": lambda t: t.count >= 100 and t.trade_price >= 85,
            "description": "Follow large bets on likely outcomes"
        },
        "whale_extreme": {
            "name": "Whale Extremes (>=100 @ <=15c or >=85c)",
            "filter": lambda t: t.count >= 100 and (t.trade_price <= 15 or t.trade_price >= 85),
            "description": "Follow large bets at extreme prices"
        },
        "mega_longshot": {
            "name": "Mega Longshots (>=500 @ <=20c)",
            "filter": lambda t: t.count >= 500 and t.trade_price <= 20,
            "description": "Follow very large underdog bets"
        },
        "whale_moderate": {
            "name": "Whale Moderate (>=100 @ 30-70c)",
            "filter": lambda t: t.count >= 100 and 30 <= t.trade_price <= 70,
            "description": "Follow large bets on toss-up markets"
        },
    }

    def __init__(self):
        self.trades: List[Trade] = []

    def load_enriched_trades(self, csv_path: str) -> int:
        """Load enriched trades with outcome data."""
        count = 0
        skipped = 0

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip trades without outcome
                if row.get('is_winner') == '':
                    skipped += 1
                    continue

                try:
                    trade = Trade(
                        id=int(row['id']),
                        market_ticker=row['market_ticker'],
                        taker_side=row['taker_side'],
                        count=int(row['count']),
                        trade_price=int(float(row['trade_price'])),
                        cost_dollars=float(row['cost_dollars']),
                        timestamp=int(row['timestamp']),
                        is_winner=row['is_winner'].lower() == 'true',
                        actual_profit_dollars=float(row['actual_profit_dollars']),
                    )
                    self.trades.append(trade)
                    count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed row: {e}")

        # Sort by timestamp
        self.trades.sort(key=lambda t: t.timestamp)

        logger.info(f"Loaded {count:,} resolved trades from {csv_path}")
        logger.info(f"Skipped {skipped:,} unresolved trades")
        return count

    def run_backtest(
        self,
        strategy_key: str,
        position_size_dollars: float = 100.0,
        slippage_cents: int = 0,
    ) -> BacktestResult:
        """
        Run backtest for a given strategy.

        Args:
            strategy_key: Key from STRATEGIES dict
            position_size_dollars: Fixed dollar amount per trade
            slippage_cents: Assumed slippage on entry (added to cost)

        Returns:
            BacktestResult with all metrics
        """
        if strategy_key not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_key}")

        strategy = self.STRATEGIES[strategy_key]
        filter_func = strategy["filter"]

        # Filter trades for this strategy
        strategy_trades = [t for t in self.trades if filter_func(t)]

        if not strategy_trades:
            logger.warning(f"No trades matched strategy: {strategy_key}")
            return BacktestResult(
                strategy_name=strategy["name"],
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_wagered=0,
                total_pnl=0,
                roi=0,
                avg_win=0,
                avg_loss=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
            )

        # Simulate trading with fixed position size
        pnl_series = []
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0

        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0
        total_wagered = 0.0

        trades_by_month: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "pnl": 0.0}
        )

        for trade in strategy_trades:
            # Calculate position size (scale relative to original trade)
            # For simplicity, we use fixed dollar position size
            # Scale the P/L proportionally
            scale_factor = position_size_dollars / trade.cost_dollars if trade.cost_dollars > 0 else 1

            # Account for slippage (worse entry price)
            slippage_cost = (slippage_cents / 100) * scale_factor

            # Scaled P/L
            if trade.is_winner:
                # Winner: payout - cost - slippage
                # Payout is (count * $1) scaled down
                payout = (trade.count * scale_factor)
                cost = position_size_dollars + slippage_cost
                pnl = payout - cost
                winning_trades += 1
                total_wins += pnl
            else:
                # Loser: lose entire cost + slippage
                pnl = -(position_size_dollars + slippage_cost)
                losing_trades += 1
                total_losses += abs(pnl)

            total_wagered += position_size_dollars
            cumulative_pnl += pnl
            pnl_series.append(cumulative_pnl)

            # Track drawdown
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = peak_pnl - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Track by month
            month_key = trade.datetime.strftime("%Y-%m")
            trades_by_month[month_key]["trades"] += 1
            if trade.is_winner:
                trades_by_month[month_key]["wins"] += 1
            trades_by_month[month_key]["pnl"] += pnl

        # Calculate metrics
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        roi = cumulative_pnl / total_wagered if total_wagered > 0 else 0
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate Sharpe-like ratio (assuming risk-free rate = 0)
        # Use per-trade returns
        if len(pnl_series) > 1:
            per_trade_returns = []
            for i in range(1, len(pnl_series)):
                ret = (pnl_series[i] - pnl_series[i-1]) / position_size_dollars
                per_trade_returns.append(ret)

            if per_trade_returns:
                mean_return = statistics.mean(per_trade_returns)
                std_return = statistics.stdev(per_trade_returns) if len(per_trade_returns) > 1 else 1
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return BacktestResult(
            strategy_name=strategy["name"],
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_wagered=total_wagered,
            total_pnl=cumulative_pnl,
            roi=roi,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            trades_by_month=dict(trades_by_month),
            pnl_series=pnl_series,
        )

    def compare_strategies(
        self,
        position_size_dollars: float = 100.0,
    ) -> List[BacktestResult]:
        """Run backtest for all strategies and compare."""
        results = []

        for strategy_key in self.STRATEGIES:
            result = self.run_backtest(strategy_key, position_size_dollars)
            results.append(result)

        # Sort by ROI
        results.sort(key=lambda r: r.roi, reverse=True)

        return results

    def generate_report(
        self,
        position_size_dollars: float = 100.0,
    ) -> str:
        """Generate comprehensive backtest report."""
        lines = []

        lines.append("=" * 80)
        lines.append("WHALE FOLLOWING STRATEGY BACKTEST")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Historical Trades: {len(self.trades):,}")
        lines.append(f"Position Size: ${position_size_dollars:,.2f} per trade")

        # Date range
        if self.trades:
            first_date = min(t.datetime for t in self.trades)
            last_date = max(t.datetime for t in self.trades)
            lines.append(f"Date Range: {first_date.date()} to {last_date.date()}")

        lines.append("")
        lines.append("-" * 80)
        lines.append("STRATEGY COMPARISON")
        lines.append("-" * 80)

        header = f"{'Strategy':<40} {'Trades':>8} {'Win Rate':>10} {'ROI':>10} {'PnL':>12} {'MDD':>10} {'PF':>8}"
        lines.append(header)
        lines.append("-" * 100)

        results = self.compare_strategies(position_size_dollars)

        for result in results:
            if result.total_trades == 0:
                continue

            lines.append(
                f"{result.strategy_name:<40} "
                f"{result.total_trades:>8,} "
                f"{100*result.win_rate:>9.1f}% "
                f"{100*result.roi:>9.1f}% "
                f"${result.total_pnl:>10,.2f} "
                f"${result.max_drawdown:>8,.2f} "
                f"{result.profit_factor:>7.2f}"
            )

        # Detailed analysis of top strategies
        lines.append("")
        lines.append("-" * 80)
        lines.append("DETAILED ANALYSIS (Top 3 by ROI)")
        lines.append("-" * 80)

        for i, result in enumerate(results[:3], 1):
            if result.total_trades == 0:
                continue

            lines.append(f"\n{i}. {result.strategy_name}")
            lines.append("-" * 50)
            lines.append(f"  Trades: {result.total_trades:,}")
            lines.append(f"  Win Rate: {100*result.win_rate:.1f}%")
            lines.append(f"  Total Wagered: ${result.total_wagered:,.2f}")
            lines.append(f"  Total P/L: ${result.total_pnl:,.2f}")
            lines.append(f"  ROI: {100*result.roi:.1f}%")
            lines.append(f"  Avg Win: ${result.avg_win:,.2f}")
            lines.append(f"  Avg Loss: ${result.avg_loss:,.2f}")
            lines.append(f"  Max Drawdown: ${result.max_drawdown:,.2f}")
            lines.append(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
            lines.append(f"  Profit Factor: {result.profit_factor:.2f}")

            # Monthly breakdown
            if result.trades_by_month:
                lines.append("\n  Monthly Performance:")
                for month, data in sorted(result.trades_by_month.items()):
                    month_wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                    lines.append(
                        f"    {month}: {data['trades']:>4} trades, "
                        f"{month_wr:>5.1f}% WR, "
                        f"${data['pnl']:>8,.2f} PnL"
                    )

        # Risk analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("RISK ANALYSIS")
        lines.append("-" * 80)

        for result in results[:3]:
            if result.total_trades < 10:
                continue

            lines.append(f"\n{result.strategy_name}:")

            # Calculate required win rate for breakeven based on avg odds
            strategy_trades = [t for t in self.trades if self.STRATEGIES[
                next(k for k, v in self.STRATEGIES.items() if v["name"] == result.strategy_name)
            ]["filter"](t)]

            if strategy_trades:
                avg_price = sum(t.trade_price for t in strategy_trades) / len(strategy_trades)
                breakeven_wr = avg_price / 100  # Price in cents / 100 = implied probability
                lines.append(f"  Avg Entry Price: {avg_price:.1f}c")
                lines.append(f"  Breakeven Win Rate: {100*breakeven_wr:.1f}%")
                lines.append(f"  Actual Win Rate: {100*result.win_rate:.1f}%")
                lines.append(f"  Edge: {100*(result.win_rate - breakeven_wr):+.1f}%")

        # Strategy recommendations
        lines.append("")
        lines.append("-" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)

        profitable = [r for r in results if r.roi > 0 and r.total_trades >= 20]

        if profitable:
            best = profitable[0]
            lines.append(f"Best Strategy: {best.strategy_name}")
            lines.append(f"  - Historical ROI: {100*best.roi:.1f}%")
            lines.append(f"  - Win Rate: {100*best.win_rate:.1f}%")
            lines.append(f"  - Sample Size: {best.total_trades} trades")

            if best.profit_factor > 1.5:
                lines.append("  - Profit factor > 1.5: STRONG signal")
            elif best.profit_factor > 1.2:
                lines.append("  - Profit factor > 1.2: Moderate signal")

            if best.max_drawdown / best.total_pnl > 0.5 and best.total_pnl > 0:
                lines.append("  - Warning: Large drawdown relative to profit")
        else:
            lines.append("No profitable strategies found with sufficient sample size.")
            lines.append("Consider:")
            lines.append("  - Adjusting strategy filters")
            lines.append("  - Gathering more historical data")
            lines.append("  - Accounting for market microstructure effects")

        lines.append("")
        lines.append("=" * 80)
        lines.append("BACKTEST COMPLETE")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backtest whale following trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
  all_whales      - Follow all trades >=100 contracts
  mega_whales     - Follow trades >=1000 contracts
  whale_longshot  - Follow >=100 contracts at <=15c (best alpha signal)
  whale_favorite  - Follow >=100 contracts at >=85c
  whale_extreme   - Follow >=100 contracts at <=15c or >=85c
  mega_longshot   - Follow >=500 contracts at <=20c
  whale_moderate  - Follow >=100 contracts at 30-70c

Examples:
  # Full comparison report
  python backtest_whale_following.py --enriched enriched.csv --report

  # Single strategy backtest
  python backtest_whale_following.py --enriched enriched.csv --strategy whale_longshot

  # Custom position size
  python backtest_whale_following.py --enriched enriched.csv --report --position-size 50
        """
    )

    parser.add_argument('--enriched', type=str, required=True, metavar='FILE',
                       help='CSV file with enriched trades')
    parser.add_argument('--strategy', type=str, metavar='STRATEGY',
                       help='Single strategy to backtest')
    parser.add_argument('--report', action='store_true',
                       help='Generate full comparison report')
    parser.add_argument('--position-size', type=float, default=100.0,
                       help='Position size in dollars (default: 100)')
    parser.add_argument('--json', type=str, metavar='FILE',
                       help='Export results to JSON')

    args = parser.parse_args()

    # Create backtester
    bt = WhaleFollowingBacktester()

    # Load data
    count = bt.load_enriched_trades(args.enriched)

    if count == 0:
        logger.error("No resolved trades found!")
        return

    # Run specific strategy
    if args.strategy:
        if args.strategy not in bt.STRATEGIES:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {', '.join(bt.STRATEGIES.keys())}")
            return

        result = bt.run_backtest(args.strategy, args.position_size)

        print(f"\n{'='*60}")
        print(f"Strategy: {result.strategy_name}")
        print(f"{'='*60}")
        print(f"Trades: {result.total_trades:,}")
        print(f"Win Rate: {100*result.win_rate:.1f}%")
        print(f"Total P/L: ${result.total_pnl:,.2f}")
        print(f"ROI: {100*result.roi:.1f}%")
        print(f"Max Drawdown: ${result.max_drawdown:,.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")

    # Generate full report
    if args.report:
        report = bt.generate_report(args.position_size)
        print(report)

        # Save to file
        report_path = Path(args.enriched).parent / "backtest_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    # Export JSON
    if args.json:
        results = bt.compare_strategies(args.position_size)
        json_results = {
            "settings": {
                "position_size": args.position_size,
                "total_trades": len(bt.trades),
            },
            "strategies": [
                {
                    "name": r.strategy_name,
                    "trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "roi": r.roi,
                    "total_pnl": r.total_pnl,
                    "max_drawdown": r.max_drawdown,
                    "sharpe_ratio": r.sharpe_ratio,
                    "profit_factor": r.profit_factor,
                    "monthly": r.trades_by_month,
                }
                for r in results
            ],
        }

        with open(args.json, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"JSON results saved to {args.json}")


if __name__ == "__main__":
    main()
