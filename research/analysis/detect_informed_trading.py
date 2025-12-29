#!/usr/bin/env python3
"""
Informed Trading Detection - Statistical analysis to identify alpha signals.

This script analyzes enriched trade data (with outcomes) to detect patterns
that indicate informed trading. The goal is to identify signals that could
be used to follow "smart money" trades.

Key Hypotheses to Test:
1. Large trades at extreme prices win more often (informed speculation)
2. Trades closer to market close have higher win rates (event knowledge)
3. Certain market types are more predictable
4. Whales have consistently higher win rates than retail

Statistical Methods:
- Win rate comparison with confidence intervals
- Chi-squared tests for significance
- Regression analysis for factor importance
- Cohort analysis for whale tracking

Usage:
    python detect_informed_trading.py --enriched enriched_trades.csv --report

Author: Claude Code
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class EnrichedTrade:
    """Trade with outcome information."""
    id: int
    market_ticker: str
    taker_side: str
    count: int
    trade_price: int
    cost_dollars: float
    potential_profit_dollars: float
    leverage_ratio: float
    timestamp: int
    market_result: Optional[str]
    is_winner: bool
    actual_profit_dollars: float

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)

    @property
    def is_whale(self) -> bool:
        return self.count >= 100

    @property
    def is_mega_whale(self) -> bool:
        return self.count >= 1000

    @property
    def is_extreme_longshot(self) -> bool:
        return self.trade_price <= 15

    @property
    def is_extreme_favorite(self) -> bool:
        return self.trade_price >= 85


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.

    This is more accurate than normal approximation for small samples
    or extreme proportions.

    Args:
        wins: Number of successes
        total: Total trials
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)

    # Z-score for confidence level
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    p = wins / total
    denominator = 1 + (z**2 / total)

    center = (p + (z**2 / (2 * total))) / denominator
    margin = (z / denominator) * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))

    return (max(0, center - margin), min(1, center + margin))


def chi_squared_test(group1_wins: int, group1_total: int,
                      group2_wins: int, group2_total: int) -> Tuple[float, bool]:
    """
    Perform chi-squared test for independence between two groups.

    Tests if win rates are significantly different.

    Returns:
        Tuple of (chi_squared_statistic, is_significant_at_95pct)
    """
    if group1_total == 0 or group2_total == 0:
        return (0.0, False)

    # Observed values
    o11 = group1_wins
    o12 = group1_total - group1_wins
    o21 = group2_wins
    o22 = group2_total - group2_wins

    total = group1_total + group2_total
    total_wins = group1_wins + group2_wins
    total_losses = total - total_wins

    if total_wins == 0 or total_losses == 0:
        return (0.0, False)

    # Expected values under independence
    e11 = group1_total * total_wins / total
    e12 = group1_total * total_losses / total
    e21 = group2_total * total_wins / total
    e22 = group2_total * total_losses / total

    # Avoid division by zero
    if any(e == 0 for e in [e11, e12, e21, e22]):
        return (0.0, False)

    # Chi-squared statistic
    chi_sq = ((o11 - e11)**2 / e11 +
              (o12 - e12)**2 / e12 +
              (o21 - e21)**2 / e21 +
              (o22 - e22)**2 / e22)

    # Critical value for 1 degree of freedom at 95% is 3.84
    is_significant = chi_sq > 3.84

    return (chi_sq, is_significant)


class InformedTradingDetector:
    """Detects patterns that indicate informed trading."""

    def __init__(self):
        self.trades: List[EnrichedTrade] = []
        self.by_market: Dict[str, List[EnrichedTrade]] = defaultdict(list)

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
                    trade = EnrichedTrade(
                        id=int(row['id']),
                        market_ticker=row['market_ticker'],
                        taker_side=row['taker_side'],
                        count=int(row['count']),
                        trade_price=int(float(row['trade_price'])),
                        cost_dollars=float(row['cost_dollars']),
                        potential_profit_dollars=float(row['potential_profit_dollars']),
                        leverage_ratio=float(row['leverage_ratio']),
                        timestamp=int(row['timestamp']),
                        market_result=row.get('market_result') or None,
                        is_winner=row['is_winner'].lower() == 'true',
                        actual_profit_dollars=float(row['actual_profit_dollars']),
                    )
                    self.trades.append(trade)
                    self.by_market[trade.market_ticker].append(trade)
                    count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed row: {e}")

        logger.info(f"Loaded {count:,} resolved trades from {csv_path}")
        logger.info(f"Skipped {skipped:,} unresolved trades")
        return count

    def hypothesis_whale_advantage(self) -> Dict[str, Any]:
        """
        Hypothesis 1: Whales (>=100 contracts) have higher win rates.

        Test if larger trades indicate more informed trading.
        """
        whale_trades = [t for t in self.trades if t.is_whale]
        retail_trades = [t for t in self.trades if not t.is_whale]

        whale_wins = sum(1 for t in whale_trades if t.is_winner)
        retail_wins = sum(1 for t in retail_trades if t.is_winner)

        whale_wr = whale_wins / len(whale_trades) if whale_trades else 0
        retail_wr = retail_wins / len(retail_trades) if retail_trades else 0

        whale_ci = wilson_score_interval(whale_wins, len(whale_trades))
        retail_ci = wilson_score_interval(retail_wins, len(retail_trades))

        chi_sq, significant = chi_squared_test(
            whale_wins, len(whale_trades),
            retail_wins, len(retail_trades)
        )

        # Profit analysis
        whale_profit = sum(t.actual_profit_dollars for t in whale_trades)
        retail_profit = sum(t.actual_profit_dollars for t in retail_trades)
        whale_cost = sum(t.cost_dollars for t in whale_trades)
        retail_cost = sum(t.cost_dollars for t in retail_trades)

        return {
            "name": "Whale vs Retail Win Rate",
            "whale": {
                "trades": len(whale_trades),
                "wins": whale_wins,
                "win_rate": whale_wr,
                "ci_95": whale_ci,
                "total_profit": whale_profit,
                "total_cost": whale_cost,
                "roi": whale_profit / whale_cost if whale_cost > 0 else 0,
            },
            "retail": {
                "trades": len(retail_trades),
                "wins": retail_wins,
                "win_rate": retail_wr,
                "ci_95": retail_ci,
                "total_profit": retail_profit,
                "total_cost": retail_cost,
                "roi": retail_profit / retail_cost if retail_cost > 0 else 0,
            },
            "difference": whale_wr - retail_wr,
            "chi_squared": chi_sq,
            "statistically_significant": significant,
            "conclusion": (
                "SIGNIFICANT" if significant else "NOT SIGNIFICANT"
            ) + f" (whale WR {100*whale_wr:.1f}% vs retail {100*retail_wr:.1f}%)"
        }

    def hypothesis_extreme_longshot(self) -> Dict[str, Any]:
        """
        Hypothesis 2: Extreme longshot bets (<=15c) by whales are informed.

        These are bets where the market thinks an outcome is unlikely,
        but someone is putting serious money behind it.
        """
        # Whale longshots
        whale_longshots = [t for t in self.trades if t.is_whale and t.is_extreme_longshot]
        # Retail longshots
        retail_longshots = [t for t in self.trades if not t.is_whale and t.is_extreme_longshot]
        # All non-longshot trades for baseline
        baseline = [t for t in self.trades if not t.is_extreme_longshot]

        whale_ls_wins = sum(1 for t in whale_longshots if t.is_winner)
        retail_ls_wins = sum(1 for t in retail_longshots if t.is_winner)
        baseline_wins = sum(1 for t in baseline if t.is_winner)

        whale_ls_wr = whale_ls_wins / len(whale_longshots) if whale_longshots else 0
        retail_ls_wr = retail_ls_wins / len(retail_longshots) if retail_longshots else 0
        baseline_wr = baseline_wins / len(baseline) if baseline else 0

        # Expected win rate for longshots is ~15% based on price
        # If whales win more, they have edge

        whale_ci = wilson_score_interval(whale_ls_wins, len(whale_longshots))
        retail_ci = wilson_score_interval(retail_ls_wins, len(retail_longshots))

        chi_sq, significant = chi_squared_test(
            whale_ls_wins, len(whale_longshots),
            retail_ls_wins, len(retail_longshots)
        )

        # Profit analysis
        whale_profit = sum(t.actual_profit_dollars for t in whale_longshots)
        retail_profit = sum(t.actual_profit_dollars for t in retail_longshots)
        whale_cost = sum(t.cost_dollars for t in whale_longshots)
        retail_cost = sum(t.cost_dollars for t in retail_longshots)

        return {
            "name": "Extreme Longshot Trades (<=15c)",
            "whale_longshots": {
                "trades": len(whale_longshots),
                "wins": whale_ls_wins,
                "win_rate": whale_ls_wr,
                "ci_95": whale_ci,
                "expected_win_rate": 0.15,  # Based on avg price ~15c
                "edge_vs_expected": whale_ls_wr - 0.15,
                "total_profit": whale_profit,
                "roi": whale_profit / whale_cost if whale_cost > 0 else 0,
            },
            "retail_longshots": {
                "trades": len(retail_longshots),
                "wins": retail_ls_wins,
                "win_rate": retail_ls_wr,
                "ci_95": retail_ci,
                "total_profit": retail_profit,
                "roi": retail_profit / retail_cost if retail_cost > 0 else 0,
            },
            "chi_squared": chi_sq,
            "statistically_significant": significant,
            "conclusion": (
                f"Whale longshots win {100*whale_ls_wr:.1f}% vs expected ~15%. " +
                ("EDGE DETECTED" if whale_ls_wr > 0.20 else "No clear edge")
            )
        }

    def hypothesis_size_progression(self) -> Dict[str, Any]:
        """
        Hypothesis 3: Win rate increases with trade size.

        Test if larger trades are progressively more informed.
        """
        size_buckets = [
            ("1-9", lambda t: t.count < 10),
            ("10-49", lambda t: 10 <= t.count < 50),
            ("50-99", lambda t: 50 <= t.count < 100),
            ("100-499", lambda t: 100 <= t.count < 500),
            ("500-999", lambda t: 500 <= t.count < 1000),
            ("1000+", lambda t: t.count >= 1000),
        ]

        results = []
        for name, filter_func in size_buckets:
            trades = [t for t in self.trades if filter_func(t)]
            if not trades:
                continue

            wins = sum(1 for t in trades if t.is_winner)
            wr = wins / len(trades)
            ci = wilson_score_interval(wins, len(trades))
            profit = sum(t.actual_profit_dollars for t in trades)
            cost = sum(t.cost_dollars for t in trades)

            results.append({
                "bucket": name,
                "trades": len(trades),
                "wins": wins,
                "win_rate": wr,
                "ci_95": ci,
                "total_profit": profit,
                "roi": profit / cost if cost > 0 else 0,
            })

        # Check if there's a monotonic trend
        win_rates = [r["win_rate"] for r in results]
        is_monotonic = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))

        # Correlation between size index and win rate
        if len(results) >= 3:
            sizes = list(range(len(results)))
            wrs = [r["win_rate"] for r in results]
            mean_size = sum(sizes) / len(sizes)
            mean_wr = sum(wrs) / len(wrs)

            numerator = sum((s - mean_size) * (w - mean_wr) for s, w in zip(sizes, wrs))
            denom_size = sum((s - mean_size)**2 for s in sizes)
            denom_wr = sum((w - mean_wr)**2 for w in wrs)

            if denom_size > 0 and denom_wr > 0:
                correlation = numerator / (math.sqrt(denom_size) * math.sqrt(denom_wr))
            else:
                correlation = 0
        else:
            correlation = 0

        return {
            "name": "Win Rate by Trade Size",
            "size_buckets": results,
            "monotonic_increase": is_monotonic,
            "correlation": correlation,
            "conclusion": (
                f"Correlation = {correlation:.3f}. " +
                ("POSITIVE TREND" if correlation > 0.5 else "No clear trend")
            )
        }

    def hypothesis_market_type(self) -> Dict[str, Any]:
        """
        Hypothesis 4: Some market types are more predictable.

        Analyze win rates by market ticker prefix (event type).
        """
        # Group by market type (first part of ticker before hyphen)
        by_type: Dict[str, List[EnrichedTrade]] = defaultdict(list)

        for trade in self.trades:
            # Extract market type from ticker
            # Examples: KXNFLGAME-..., FED-25DEC, CONTROLH-2026
            parts = trade.market_ticker.split('-')
            market_type = parts[0] if parts else "UNKNOWN"
            by_type[market_type].append(trade)

        results = []
        for market_type, trades in sorted(by_type.items(), key=lambda x: -len(x[1])):
            if len(trades) < 10:  # Skip rare market types
                continue

            wins = sum(1 for t in trades if t.is_winner)
            wr = wins / len(trades)
            ci = wilson_score_interval(wins, len(trades))
            profit = sum(t.actual_profit_dollars for t in trades)
            cost = sum(t.cost_dollars for t in trades)

            # Whale analysis for this market type
            whale_trades = [t for t in trades if t.is_whale]
            whale_wins = sum(1 for t in whale_trades if t.is_winner)
            whale_wr = whale_wins / len(whale_trades) if whale_trades else 0

            results.append({
                "market_type": market_type,
                "trades": len(trades),
                "wins": wins,
                "win_rate": wr,
                "ci_95": ci,
                "total_profit": profit,
                "roi": profit / cost if cost > 0 else 0,
                "whale_trades": len(whale_trades),
                "whale_win_rate": whale_wr,
            })

        # Sort by ROI to find most profitable market types
        results.sort(key=lambda x: x["roi"], reverse=True)

        return {
            "name": "Win Rate by Market Type",
            "market_types": results[:20],  # Top 20
            "most_profitable": results[0]["market_type"] if results else None,
            "highest_whale_edge": max(
                ((r["market_type"], r["whale_win_rate"] - r["win_rate"])
                 for r in results if r["whale_trades"] >= 10),
                key=lambda x: x[1],
                default=("N/A", 0)
            ),
        }

    def find_alpha_signals(self) -> List[Dict[str, Any]]:
        """
        Combine hypotheses to identify actionable alpha signals.

        Returns signals sorted by expected profitability.
        """
        signals = []

        # Signal 1: Whale longshots
        whale_longshots = [t for t in self.trades if t.is_whale and t.is_extreme_longshot]
        if len(whale_longshots) >= 10:
            wins = sum(1 for t in whale_longshots if t.is_winner)
            wr = wins / len(whale_longshots)
            expected_wr = 0.15  # Market price implies ~15% probability

            if wr > expected_wr:
                edge = wr - expected_wr
                cost = sum(t.cost_dollars for t in whale_longshots)
                profit = sum(t.actual_profit_dollars for t in whale_longshots)

                signals.append({
                    "signal": "Follow whale longshot bets (>=100 contracts at <=15c)",
                    "sample_size": len(whale_longshots),
                    "win_rate": wr,
                    "expected_win_rate": expected_wr,
                    "edge": edge,
                    "historical_roi": profit / cost if cost > 0 else 0,
                    "confidence": wilson_score_interval(wins, len(whale_longshots)),
                    "statistical_significance": len(whale_longshots) >= 50 and edge > 0.05,
                })

        # Signal 2: Mega-whale trades (>=1000 contracts)
        mega_whales = [t for t in self.trades if t.is_mega_whale]
        if len(mega_whales) >= 10:
            wins = sum(1 for t in mega_whales if t.is_winner)
            wr = wins / len(mega_whales)
            cost = sum(t.cost_dollars for t in mega_whales)
            profit = sum(t.actual_profit_dollars for t in mega_whales)

            signals.append({
                "signal": "Follow mega-whale trades (>=1000 contracts)",
                "sample_size": len(mega_whales),
                "win_rate": wr,
                "historical_roi": profit / cost if cost > 0 else 0,
                "confidence": wilson_score_interval(wins, len(mega_whales)),
                "avg_cost_per_trade": cost / len(mega_whales),
                "avg_profit_per_trade": profit / len(mega_whales),
            })

        # Signal 3: Whale extreme bets (either side)
        whale_extremes = [t for t in self.trades if t.is_whale and
                          (t.is_extreme_longshot or t.is_extreme_favorite)]
        if len(whale_extremes) >= 10:
            wins = sum(1 for t in whale_extremes if t.is_winner)
            wr = wins / len(whale_extremes)
            cost = sum(t.cost_dollars for t in whale_extremes)
            profit = sum(t.actual_profit_dollars for t in whale_extremes)

            signals.append({
                "signal": "Follow whale extreme price bets (>=100 contracts at <=15c or >=85c)",
                "sample_size": len(whale_extremes),
                "win_rate": wr,
                "historical_roi": profit / cost if cost > 0 else 0,
                "confidence": wilson_score_interval(wins, len(whale_extremes)),
            })

        # Sort by historical ROI
        signals.sort(key=lambda x: x.get("historical_roi", 0), reverse=True)

        return signals

    def generate_report(self) -> str:
        """Generate comprehensive informed trading analysis report."""
        lines = []

        lines.append("=" * 80)
        lines.append("INFORMED TRADING DETECTION ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Resolved Trades Analyzed: {len(self.trades):,}")

        # Overall stats
        winners = [t for t in self.trades if t.is_winner]
        lines.append(f"Overall Win Rate: {100*len(winners)/len(self.trades):.1f}%")

        total_profit = sum(t.actual_profit_dollars for t in self.trades)
        total_cost = sum(t.cost_dollars for t in self.trades)
        lines.append(f"Total P/L: ${total_profit:,.2f}")
        lines.append(f"Overall ROI: {100*total_profit/total_cost:.1f}%")

        # Hypothesis 1: Whale advantage
        lines.append("")
        lines.append("-" * 80)
        lines.append("HYPOTHESIS 1: WHALE ADVANTAGE")
        lines.append("-" * 80)

        h1 = self.hypothesis_whale_advantage()
        lines.append(f"Whale (>=100 contracts):")
        lines.append(f"  Trades: {h1['whale']['trades']:,}")
        lines.append(f"  Win Rate: {100*h1['whale']['win_rate']:.1f}% "
                    f"(95% CI: {100*h1['whale']['ci_95'][0]:.1f}%-{100*h1['whale']['ci_95'][1]:.1f}%)")
        lines.append(f"  ROI: {100*h1['whale']['roi']:.1f}%")
        lines.append(f"  Total Profit: ${h1['whale']['total_profit']:,.2f}")

        lines.append(f"\nRetail (<100 contracts):")
        lines.append(f"  Trades: {h1['retail']['trades']:,}")
        lines.append(f"  Win Rate: {100*h1['retail']['win_rate']:.1f}% "
                    f"(95% CI: {100*h1['retail']['ci_95'][0]:.1f}%-{100*h1['retail']['ci_95'][1]:.1f}%)")
        lines.append(f"  ROI: {100*h1['retail']['roi']:.1f}%")

        lines.append(f"\nStatistical Test: Chi-squared = {h1['chi_squared']:.2f}")
        lines.append(f"Result: {h1['conclusion']}")

        # Hypothesis 2: Extreme longshots
        lines.append("")
        lines.append("-" * 80)
        lines.append("HYPOTHESIS 2: WHALE LONGSHOT BETS")
        lines.append("-" * 80)

        h2 = self.hypothesis_extreme_longshot()
        lines.append(f"Whale Longshots (>=100 contracts at <=15c):")
        lines.append(f"  Trades: {h2['whale_longshots']['trades']:,}")
        lines.append(f"  Win Rate: {100*h2['whale_longshots']['win_rate']:.1f}%")
        lines.append(f"  Expected Win Rate (from price): ~15%")
        lines.append(f"  Edge vs Expected: {100*h2['whale_longshots']['edge_vs_expected']:+.1f}%")
        lines.append(f"  ROI: {100*h2['whale_longshots']['roi']:.1f}%")

        lines.append(f"\nRetail Longshots:")
        lines.append(f"  Trades: {h2['retail_longshots']['trades']:,}")
        lines.append(f"  Win Rate: {100*h2['retail_longshots']['win_rate']:.1f}%")
        lines.append(f"  ROI: {100*h2['retail_longshots']['roi']:.1f}%")

        lines.append(f"\nConclusion: {h2['conclusion']}")

        # Hypothesis 3: Size progression
        lines.append("")
        lines.append("-" * 80)
        lines.append("HYPOTHESIS 3: WIN RATE BY TRADE SIZE")
        lines.append("-" * 80)

        h3 = self.hypothesis_size_progression()
        lines.append(f"{'Size Bucket':<15} {'Trades':>8} {'Win Rate':>10} {'ROI':>10}")
        lines.append("-" * 50)

        for bucket in h3['size_buckets']:
            lines.append(
                f"{bucket['bucket']:<15} {bucket['trades']:>8,} "
                f"{100*bucket['win_rate']:>9.1f}% {100*bucket['roi']:>9.1f}%"
            )

        lines.append(f"\nCorrelation: {h3['correlation']:.3f}")
        lines.append(f"Conclusion: {h3['conclusion']}")

        # Hypothesis 4: Market types
        lines.append("")
        lines.append("-" * 80)
        lines.append("HYPOTHESIS 4: WIN RATE BY MARKET TYPE")
        lines.append("-" * 80)

        h4 = self.hypothesis_market_type()
        lines.append(f"{'Market Type':<20} {'Trades':>8} {'Win Rate':>10} {'ROI':>10} {'Whale WR':>10}")
        lines.append("-" * 70)

        for mt in h4['market_types'][:15]:  # Top 15
            lines.append(
                f"{mt['market_type']:<20} {mt['trades']:>8,} "
                f"{100*mt['win_rate']:>9.1f}% {100*mt['roi']:>9.1f}% "
                f"{100*mt['whale_win_rate']:>9.1f}%"
            )

        # Alpha signals
        lines.append("")
        lines.append("-" * 80)
        lines.append("ALPHA SIGNALS (ACTIONABLE)")
        lines.append("-" * 80)

        signals = self.find_alpha_signals()
        for i, signal in enumerate(signals, 1):
            lines.append(f"\n{i}. {signal['signal']}")
            lines.append(f"   Sample Size: {signal['sample_size']:,}")
            lines.append(f"   Win Rate: {100*signal['win_rate']:.1f}%")
            if 'expected_win_rate' in signal:
                lines.append(f"   Expected: {100*signal['expected_win_rate']:.1f}%")
                lines.append(f"   Edge: {100*signal.get('edge', 0):+.1f}%")
            lines.append(f"   Historical ROI: {100*signal['historical_roi']:.1f}%")
            lines.append(f"   95% CI: {100*signal['confidence'][0]:.1f}%-{100*signal['confidence'][1]:.1f}%")

        # Summary recommendations
        lines.append("")
        lines.append("=" * 80)
        lines.append("SUMMARY RECOMMENDATIONS")
        lines.append("=" * 80)

        if h1['statistically_significant']:
            lines.append("- Whale trades show statistically significant edge over retail")

        if h2['whale_longshots']['edge_vs_expected'] > 0.05:
            lines.append("- Whale longshot bets show edge vs expected win rate")

        if h3['correlation'] > 0.5:
            lines.append("- Trade size positively correlates with win rate")

        if signals:
            best = signals[0]
            lines.append(f"- Best signal: {best['signal']} (ROI: {100*best['historical_roi']:.1f}%)")

        lines.append("")
        lines.append("=" * 80)
        lines.append("ANALYSIS COMPLETE")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect informed trading patterns in historical trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full report
  python detect_informed_trading.py --enriched enriched_trades.csv --report

  # Export JSON results
  python detect_informed_trading.py --enriched enriched_trades.csv --json results.json
        """
    )

    parser.add_argument('--enriched', type=str, required=True, metavar='FILE',
                       help='CSV file with enriched trades (output from analyze_trade_outcomes.py)')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive analysis report')
    parser.add_argument('--json', type=str, metavar='FILE',
                       help='Export analysis results to JSON')

    args = parser.parse_args()

    # Create detector
    detector = InformedTradingDetector()

    # Load data
    count = detector.load_enriched_trades(args.enriched)

    if count == 0:
        logger.error("No resolved trades found!")
        return

    # Generate report
    if args.report:
        report = detector.generate_report()
        print(report)

        # Save to file
        report_path = Path(args.enriched).parent / "informed_trading_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    # Export JSON
    if args.json:
        results = {
            "summary": {
                "total_trades": len(detector.trades),
                "total_profit": sum(t.actual_profit_dollars for t in detector.trades),
            },
            "whale_advantage": detector.hypothesis_whale_advantage(),
            "extreme_longshots": detector.hypothesis_extreme_longshot(),
            "size_progression": detector.hypothesis_size_progression(),
            "market_types": detector.hypothesis_market_type(),
            "alpha_signals": detector.find_alpha_signals(),
        }

        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results saved to {args.json}")

    # Print quick summary if no report
    if not args.report:
        print(f"\n{'='*60}")
        print("QUICK SUMMARY")
        print(f"{'='*60}")

        h1 = detector.hypothesis_whale_advantage()
        print(f"Whale Win Rate: {100*h1['whale']['win_rate']:.1f}%")
        print(f"Retail Win Rate: {100*h1['retail']['win_rate']:.1f}%")
        print(f"Difference: {100*h1['difference']:+.1f}%")
        print(f"Statistically Significant: {h1['statistically_significant']}")

        signals = detector.find_alpha_signals()
        if signals:
            print(f"\nTop Signal: {signals[0]['signal']}")
            print(f"Historical ROI: {100*signals[0]['historical_roi']:.1f}%")


if __name__ == "__main__":
    main()
