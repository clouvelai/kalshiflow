#!/usr/bin/env python3
"""
Trade-Outcome Analysis - Correlate historical trades with market outcomes.

This script joins trade data with market outcomes to calculate:
- Actual P/L for each trade (not just theoretical)
- Win rates by various trade characteristics
- Statistical patterns that indicate informed trading

Key Analysis Dimensions:
- Trade size (contract count)
- Trade price (extreme vs moderate)
- Timing relative to market close
- Trade side (YES vs NO)
- Market category/type

Usage:
    # Basic analysis
    python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv

    # Full analysis with report generation
    python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv --report

    # Export enriched trade data
    python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv --export enriched_trades.csv

Author: Claude Code
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
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
    """Represents a single trade with computed fields."""
    id: int
    market_ticker: str
    taker_side: str  # 'yes' or 'no'
    count: int  # number of contracts
    yes_price: int  # in cents
    no_price: int  # in cents
    timestamp: int  # milliseconds
    trade_price: int  # price paid (yes_price if taker_side='yes', else no_price)
    cost_dollars: float
    max_payout_dollars: float
    potential_profit_dollars: float
    leverage_ratio: float

    # Outcome fields (filled after joining with outcomes)
    market_result: Optional[str] = None  # 'yes', 'no', or None
    market_status: Optional[str] = None
    is_winner: Optional[bool] = None
    actual_profit_dollars: Optional[float] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)

    @property
    def is_resolved(self) -> bool:
        return self.market_result in ('yes', 'no')


@dataclass
class Outcome:
    """Market outcome data."""
    ticker: str
    status: str
    result: str  # 'yes', 'no', or ''
    close_time: Optional[str]
    settlement_value: Optional[int]
    category: Optional[str]
    volume: Optional[int]


@dataclass
class AnalysisSegment:
    """Statistics for a segment of trades."""
    name: str
    trade_count: int = 0
    resolved_count: int = 0
    win_count: int = 0
    total_cost: float = 0.0
    total_profit: float = 0.0
    total_payout_potential: float = 0.0
    avg_price: float = 0.0
    avg_contracts: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / self.resolved_count if self.resolved_count > 0 else 0.0

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_cost if self.total_cost > 0 else 0.0

    @property
    def expected_value_per_dollar(self) -> float:
        return (self.total_profit + self.total_cost) / self.total_cost if self.total_cost > 0 else 0.0


class TradeOutcomeAnalyzer:
    """Analyzes trades against their market outcomes."""

    def __init__(self):
        self.trades: List[Trade] = []
        self.outcomes: Dict[str, Outcome] = {}
        self.enriched_trades: List[Trade] = []

    def load_trades(self, csv_path: str) -> int:
        """Load trades from CSV file."""
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trade = Trade(
                        id=int(row['id']),
                        market_ticker=row['market_ticker'],
                        taker_side=row['taker_side'],
                        count=int(row['count']),
                        yes_price=int(row['yes_price']),
                        no_price=int(row['no_price']),
                        timestamp=int(row['timestamp']),
                        trade_price=int(float(row['trade_price'])),
                        cost_dollars=float(row['cost_dollars']),
                        max_payout_dollars=float(row['max_payout_dollars']),
                        potential_profit_dollars=float(row['potential_profit_dollars']),
                        leverage_ratio=float(row['leverage_ratio']),
                    )
                    self.trades.append(trade)
                    count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed row: {e}")

        logger.info(f"Loaded {count:,} trades from {csv_path}")
        return count

    def load_outcomes(self, csv_path: str) -> int:
        """Load market outcomes from CSV file."""
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    outcome = Outcome(
                        ticker=row['ticker'],
                        status=row['status'],
                        result=row['result'],
                        close_time=row.get('close_time'),
                        settlement_value=int(row['settlement_value']) if row.get('settlement_value') else None,
                        category=row.get('category'),
                        volume=int(row['volume']) if row.get('volume') else None,
                    )
                    self.outcomes[outcome.ticker] = outcome
                    count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed outcome row: {e}")

        logger.info(f"Loaded {count:,} market outcomes from {csv_path}")
        return count

    def join_trades_with_outcomes(self) -> Tuple[int, int]:
        """
        Join trades with their market outcomes to calculate actual P/L.

        Returns:
            Tuple of (total_trades, resolved_trades)
        """
        resolved_count = 0

        for trade in self.trades:
            outcome = self.outcomes.get(trade.market_ticker)

            if outcome:
                trade.market_result = outcome.result if outcome.result else None
                trade.market_status = outcome.status

                if trade.market_result in ('yes', 'no'):
                    resolved_count += 1

                    # Determine if this trade was a winner
                    # Winner if: taker_side matches result
                    trade.is_winner = (trade.taker_side == trade.market_result)

                    # Calculate actual profit
                    if trade.is_winner:
                        # Winner gets payout minus cost
                        trade.actual_profit_dollars = trade.max_payout_dollars - trade.cost_dollars
                    else:
                        # Loser loses entire cost
                        trade.actual_profit_dollars = -trade.cost_dollars

            self.enriched_trades.append(trade)

        logger.info(f"Joined {len(self.trades):,} trades with outcomes")
        logger.info(f"  Resolved: {resolved_count:,} ({100*resolved_count/len(self.trades):.1f}%)")

        return len(self.trades), resolved_count

    def analyze_by_segment(
        self,
        segment_func,
        segment_name: str
    ) -> Dict[str, AnalysisSegment]:
        """
        Analyze trades segmented by a custom function.

        Args:
            segment_func: Function that takes a Trade and returns segment key
            segment_name: Name of segmentation for logging

        Returns:
            Dict mapping segment key to AnalysisSegment
        """
        segments: Dict[str, AnalysisSegment] = {}

        for trade in self.enriched_trades:
            key = segment_func(trade)
            if key is None:
                continue

            if key not in segments:
                segments[key] = AnalysisSegment(name=str(key))

            seg = segments[key]
            seg.trade_count += 1
            seg.total_cost += trade.cost_dollars
            seg.total_payout_potential += trade.potential_profit_dollars

            if trade.is_resolved:
                seg.resolved_count += 1
                if trade.is_winner:
                    seg.win_count += 1
                seg.total_profit += trade.actual_profit_dollars or 0

        # Calculate averages
        for seg in segments.values():
            if seg.trade_count > 0:
                relevant_trades = [t for t in self.enriched_trades if segment_func(t) == seg.name]
                seg.avg_price = statistics.mean(t.trade_price for t in relevant_trades)
                seg.avg_contracts = statistics.mean(t.count for t in relevant_trades)

        logger.info(f"Analyzed {segment_name}: {len(segments)} segments")
        return segments

    def analyze_by_price_bucket(self) -> Dict[str, AnalysisSegment]:
        """Analyze trades by price bucket."""
        def price_bucket(trade: Trade) -> str:
            price = trade.trade_price
            if price <= 5:
                return "01-05c (extreme longshot)"
            elif price <= 10:
                return "06-10c (longshot)"
            elif price <= 20:
                return "11-20c (underdog)"
            elif price <= 35:
                return "21-35c (slight underdog)"
            elif price <= 50:
                return "36-50c (toss-up low)"
            elif price <= 65:
                return "51-65c (toss-up high)"
            elif price <= 80:
                return "66-80c (favorite)"
            elif price <= 90:
                return "81-90c (strong favorite)"
            else:
                return "91-99c (near-certain)"

        return self.analyze_by_segment(price_bucket, "Price Buckets")

    def analyze_by_contract_size(self) -> Dict[str, AnalysisSegment]:
        """Analyze trades by contract size."""
        def size_bucket(trade: Trade) -> str:
            count = trade.count
            if count < 10:
                return "1-9 (tiny)"
            elif count < 50:
                return "10-49 (small)"
            elif count < 100:
                return "50-99 (medium)"
            elif count < 500:
                return "100-499 (large)"
            elif count < 1000:
                return "500-999 (whale)"
            else:
                return "1000+ (mega-whale)"

        return self.analyze_by_segment(size_bucket, "Contract Size")

    def analyze_by_side(self) -> Dict[str, AnalysisSegment]:
        """Analyze trades by taker side."""
        return self.analyze_by_segment(lambda t: t.taker_side, "Taker Side")

    def analyze_whales_by_price(self) -> Dict[str, AnalysisSegment]:
        """Analyze whale trades (>=100 contracts) by price bucket."""
        def whale_price_bucket(trade: Trade) -> Optional[str]:
            if trade.count < 100:
                return None  # Skip non-whales

            price = trade.trade_price
            if price <= 10:
                return "Whale at 1-10c"
            elif price <= 20:
                return "Whale at 11-20c"
            elif price <= 35:
                return "Whale at 21-35c"
            elif price <= 50:
                return "Whale at 36-50c"
            elif price <= 65:
                return "Whale at 51-65c"
            elif price <= 80:
                return "Whale at 66-80c"
            else:
                return "Whale at 81-99c"

        return self.analyze_by_segment(whale_price_bucket, "Whale Trades by Price")

    def analyze_extreme_bets(self) -> Dict[str, AnalysisSegment]:
        """Analyze extreme price bets (<=15c or >=85c) by size."""
        def extreme_bucket(trade: Trade) -> Optional[str]:
            price = trade.trade_price

            if 15 < price < 85:
                return None  # Skip moderate prices

            side = "Longshot" if price <= 15 else "Lock"
            count = trade.count

            if count < 50:
                size = "small (<50)"
            elif count < 200:
                size = "medium (50-199)"
            elif count < 1000:
                size = "large (200-999)"
            else:
                size = "whale (1000+)"

            return f"{side} {size}"

        return self.analyze_by_segment(extreme_bucket, "Extreme Price Bets")

    def get_top_profitable_patterns(self, min_trades: int = 20) -> List[Dict]:
        """
        Identify the most profitable trading patterns.

        Returns list of patterns sorted by ROI.
        """
        # Combine multiple segmentation analyses
        all_segments = {}

        # Price buckets
        for key, seg in self.analyze_by_price_bucket().items():
            all_segments[f"Price: {key}"] = seg

        # Contract size
        for key, seg in self.analyze_by_contract_size().items():
            all_segments[f"Size: {key}"] = seg

        # Whale by price
        for key, seg in self.analyze_whales_by_price().items():
            all_segments[key] = seg

        # Extreme bets
        for key, seg in self.analyze_extreme_bets().items():
            all_segments[key] = seg

        # Filter and sort by ROI
        profitable_patterns = []
        for name, seg in all_segments.items():
            if seg.resolved_count >= min_trades:
                profitable_patterns.append({
                    "pattern": name,
                    "resolved_trades": seg.resolved_count,
                    "win_rate": seg.win_rate,
                    "total_cost": seg.total_cost,
                    "total_profit": seg.total_profit,
                    "roi": seg.roi,
                    "ev_per_dollar": seg.expected_value_per_dollar,
                })

        # Sort by ROI descending
        profitable_patterns.sort(key=lambda x: x["roi"], reverse=True)

        return profitable_patterns

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        lines = []

        lines.append("=" * 80)
        lines.append("TRADE-OUTCOME CORRELATION ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Trades: {len(self.trades):,}")

        resolved = [t for t in self.enriched_trades if t.is_resolved]
        lines.append(f"Resolved Trades: {len(resolved):,} ({100*len(resolved)/len(self.trades):.1f}%)")

        if resolved:
            winners = [t for t in resolved if t.is_winner]
            lines.append(f"Overall Win Rate: {100*len(winners)/len(resolved):.1f}%")

            total_profit = sum(t.actual_profit_dollars or 0 for t in resolved)
            total_cost = sum(t.cost_dollars for t in resolved)
            lines.append(f"Total P/L: ${total_profit:,.2f}")
            lines.append(f"Total Wagered: ${total_cost:,.2f}")
            lines.append(f"Overall ROI: {100*total_profit/total_cost:.2f}%" if total_cost > 0 else "N/A")

        # Price bucket analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("ANALYSIS BY PRICE BUCKET")
        lines.append("-" * 80)
        lines.append(f"{'Price Bucket':<30} {'Trades':>8} {'Win Rate':>10} {'P/L':>14} {'ROI':>10}")
        lines.append("-" * 80)

        for key, seg in sorted(self.analyze_by_price_bucket().items()):
            if seg.resolved_count > 0:
                lines.append(
                    f"{key:<30} {seg.resolved_count:>8,} "
                    f"{100*seg.win_rate:>9.1f}% "
                    f"${seg.total_profit:>12,.2f} "
                    f"{100*seg.roi:>9.1f}%"
                )

        # Contract size analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("ANALYSIS BY CONTRACT SIZE")
        lines.append("-" * 80)
        lines.append(f"{'Size Bucket':<30} {'Trades':>8} {'Win Rate':>10} {'P/L':>14} {'ROI':>10}")
        lines.append("-" * 80)

        for key, seg in sorted(self.analyze_by_contract_size().items()):
            if seg.resolved_count > 0:
                lines.append(
                    f"{key:<30} {seg.resolved_count:>8,} "
                    f"{100*seg.win_rate:>9.1f}% "
                    f"${seg.total_profit:>12,.2f} "
                    f"{100*seg.roi:>9.1f}%"
                )

        # Whale analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("WHALE TRADES (>=100 contracts) BY PRICE")
        lines.append("-" * 80)
        lines.append(f"{'Category':<30} {'Trades':>8} {'Win Rate':>10} {'P/L':>14} {'ROI':>10}")
        lines.append("-" * 80)

        whale_segments = self.analyze_whales_by_price()
        for key in sorted(whale_segments.keys()):
            seg = whale_segments[key]
            if seg.resolved_count > 0:
                lines.append(
                    f"{key:<30} {seg.resolved_count:>8,} "
                    f"{100*seg.win_rate:>9.1f}% "
                    f"${seg.total_profit:>12,.2f} "
                    f"{100*seg.roi:>9.1f}%"
                )

        # Extreme bets analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("EXTREME PRICE BETS (<=15c or >=85c)")
        lines.append("-" * 80)
        lines.append(f"{'Category':<30} {'Trades':>8} {'Win Rate':>10} {'P/L':>14} {'ROI':>10}")
        lines.append("-" * 80)

        extreme_segments = self.analyze_extreme_bets()
        for key in sorted(extreme_segments.keys()):
            seg = extreme_segments[key]
            if seg.resolved_count > 0:
                lines.append(
                    f"{key:<30} {seg.resolved_count:>8,} "
                    f"{100*seg.win_rate:>9.1f}% "
                    f"${seg.total_profit:>12,.2f} "
                    f"{100*seg.roi:>9.1f}%"
                )

        # Top profitable patterns
        lines.append("")
        lines.append("-" * 80)
        lines.append("TOP PROFITABLE PATTERNS (min 20 resolved trades)")
        lines.append("-" * 80)
        lines.append(f"{'Pattern':<40} {'Trades':>8} {'Win Rate':>10} {'ROI':>10}")
        lines.append("-" * 80)

        for pattern in self.get_top_profitable_patterns(min_trades=20)[:20]:
            lines.append(
                f"{pattern['pattern']:<40} {pattern['resolved_trades']:>8,} "
                f"{100*pattern['win_rate']:>9.1f}% "
                f"{100*pattern['roi']:>9.1f}%"
            )

        # Side analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("ANALYSIS BY TAKER SIDE")
        lines.append("-" * 80)

        for key, seg in self.analyze_by_side().items():
            if seg.resolved_count > 0:
                lines.append(f"  {key.upper()} side:")
                lines.append(f"    Resolved trades: {seg.resolved_count:,}")
                lines.append(f"    Win rate: {100*seg.win_rate:.1f}%")
                lines.append(f"    Total P/L: ${seg.total_profit:,.2f}")
                lines.append(f"    ROI: {100*seg.roi:.1f}%")

        lines.append("")
        lines.append("=" * 80)
        lines.append("ANALYSIS COMPLETE")
        lines.append("=" * 80)

        return "\n".join(lines)

    def export_enriched_trades(self, output_path: str):
        """Export enriched trades with outcome data to CSV."""
        columns = [
            'id', 'market_ticker', 'taker_side', 'count', 'trade_price',
            'cost_dollars', 'potential_profit_dollars', 'leverage_ratio',
            'timestamp', 'market_result', 'market_status', 'is_winner',
            'actual_profit_dollars'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for trade in self.enriched_trades:
                writer.writerow({
                    'id': trade.id,
                    'market_ticker': trade.market_ticker,
                    'taker_side': trade.taker_side,
                    'count': trade.count,
                    'trade_price': trade.trade_price,
                    'cost_dollars': trade.cost_dollars,
                    'potential_profit_dollars': trade.potential_profit_dollars,
                    'leverage_ratio': trade.leverage_ratio,
                    'timestamp': trade.timestamp,
                    'market_result': trade.market_result or '',
                    'market_status': trade.market_status or '',
                    'is_winner': trade.is_winner if trade.is_winner is not None else '',
                    'actual_profit_dollars': trade.actual_profit_dollars if trade.actual_profit_dollars is not None else '',
                })

        logger.info(f"Exported {len(self.enriched_trades):,} enriched trades to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze trade outcomes and identify profitable patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv

  # Generate full report
  python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv --report

  # Export enriched data
  python analyze_trade_outcomes.py --trades trades.csv --outcomes outcomes.csv --export enriched.csv
        """
    )

    parser.add_argument('--trades', type=str, required=True, metavar='FILE',
                       help='CSV file with historical trades')
    parser.add_argument('--outcomes', type=str, required=True, metavar='FILE',
                       help='CSV file with market outcomes')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive analysis report')
    parser.add_argument('--export', type=str, metavar='FILE',
                       help='Export enriched trades to CSV')
    parser.add_argument('--json', type=str, metavar='FILE',
                       help='Export analysis results to JSON')
    parser.add_argument('--min-resolved', type=int, default=10,
                       help='Minimum resolved trades for pattern analysis (default: 10)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = TradeOutcomeAnalyzer()

    # Load data
    analyzer.load_trades(args.trades)
    analyzer.load_outcomes(args.outcomes)

    # Join trades with outcomes
    total, resolved = analyzer.join_trades_with_outcomes()

    if resolved == 0:
        logger.warning("No resolved trades found! Cannot perform outcome analysis.")
        logger.info("This could mean:")
        logger.info("  1. Market outcomes haven't been fetched yet")
        logger.info("  2. All markets are still active (not settled)")
        logger.info("  3. No matching tickers between trades and outcomes")
        return

    # Generate report
    if args.report:
        report = analyzer.generate_report()
        print(report)

        # Also save to file
        report_path = Path(args.trades).parent / "trade_outcome_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    # Export enriched trades
    if args.export:
        analyzer.export_enriched_trades(args.export)

    # Export JSON results
    if args.json:
        results = {
            "summary": {
                "total_trades": len(analyzer.trades),
                "resolved_trades": resolved,
                "resolution_rate": resolved / len(analyzer.trades),
            },
            "price_analysis": {
                k: {
                    "resolved": s.resolved_count,
                    "win_rate": s.win_rate,
                    "total_profit": s.total_profit,
                    "roi": s.roi,
                }
                for k, s in analyzer.analyze_by_price_bucket().items()
                if s.resolved_count > 0
            },
            "size_analysis": {
                k: {
                    "resolved": s.resolved_count,
                    "win_rate": s.win_rate,
                    "total_profit": s.total_profit,
                    "roi": s.roi,
                }
                for k, s in analyzer.analyze_by_contract_size().items()
                if s.resolved_count > 0
            },
            "top_patterns": analyzer.get_top_profitable_patterns(args.min_resolved),
        }

        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON results saved to {args.json}")

    # Print quick summary if no report requested
    if not args.report:
        print(f"\n{'='*60}")
        print("QUICK SUMMARY")
        print(f"{'='*60}")
        print(f"Total trades: {len(analyzer.trades):,}")
        print(f"Resolved trades: {resolved:,} ({100*resolved/len(analyzer.trades):.1f}%)")

        winners = [t for t in analyzer.enriched_trades if t.is_winner]
        print(f"Winners: {len(winners):,} ({100*len(winners)/resolved:.1f}% win rate)")

        total_profit = sum(t.actual_profit_dollars or 0 for t in analyzer.enriched_trades if t.is_resolved)
        total_cost = sum(t.cost_dollars for t in analyzer.enriched_trades if t.is_resolved)
        print(f"Total P/L: ${total_profit:,.2f}")
        print(f"ROI: {100*total_profit/total_cost:.1f}%" if total_cost > 0 else "ROI: N/A")


if __name__ == "__main__":
    main()
