#!/usr/bin/env python3
"""
Advanced Pattern Analysis - Test creative trading hypotheses.

Builds on the basic trade-outcome analysis to test:
1. Timing patterns (trades near market close)
2. Contrarian signals (betting against strong consensus)
3. Dumb money fade (small vs large trade disagreement)
4. Category-specific edges

Usage:
    python advanced_pattern_analysis.py \
        --trades trades.csv \
        --outcomes outcomes.csv \
        --report

Author: Claude Code
"""

import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class EnrichedTrade:
    """Trade with outcome data."""
    id: int
    market_ticker: str
    taker_side: str
    count: int
    trade_price: int
    cost_dollars: float
    timestamp: int
    is_winner: bool
    actual_profit_dollars: float
    # Additional fields from outcomes
    market_close_time: Optional[datetime] = None
    category: Optional[str] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)

    @property
    def minutes_before_close(self) -> Optional[float]:
        """Minutes between trade and market close."""
        if self.market_close_time:
            delta = self.market_close_time - self.datetime
            return delta.total_seconds() / 60
        return None


@dataclass
class PatternResult:
    """Results for a pattern test."""
    name: str
    description: str
    trades: int
    win_rate: float
    breakeven_rate: float
    edge: float
    roi: float
    total_profit: float
    is_significant: bool  # >= 100 trades and positive edge


class AdvancedAnalyzer:
    """Advanced pattern analysis for trading signals."""

    def __init__(self):
        self.trades: List[EnrichedTrade] = []
        self.outcomes: Dict[str, Dict] = {}

    def load_data(self, trades_path: str, outcomes_path: str):
        """Load trades and outcomes, joining on ticker."""
        # Load outcomes first
        with open(outcomes_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row['ticker']
                close_time = None
                if row.get('close_time'):
                    try:
                        close_time = datetime.fromisoformat(
                            row['close_time'].replace('Z', '+00:00')
                        ).replace(tzinfo=None)
                    except:
                        pass

                self.outcomes[ticker] = {
                    'result': row['result'],
                    'status': row['status'],
                    'close_time': close_time,
                    'category': row.get('category', ''),
                }

        logger.info(f"Loaded {len(self.outcomes)} outcomes")

        # Load trades
        with open(trades_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row['market_ticker']
                outcome = self.outcomes.get(ticker, {})

                # Skip if no resolution
                if outcome.get('result') not in ('yes', 'no'):
                    continue

                try:
                    is_winner = row['taker_side'] == outcome['result']
                    trade_price = int(float(row['trade_price']))
                    cost = float(row['cost_dollars'])

                    if is_winner:
                        profit = float(row['max_payout_dollars']) - cost
                    else:
                        profit = -cost

                    trade = EnrichedTrade(
                        id=int(row['id']),
                        market_ticker=ticker,
                        taker_side=row['taker_side'],
                        count=int(row['count']),
                        trade_price=trade_price,
                        cost_dollars=cost,
                        timestamp=int(row['timestamp']),
                        is_winner=is_winner,
                        actual_profit_dollars=profit,
                        market_close_time=outcome.get('close_time'),
                        category=outcome.get('category', ''),
                    )
                    self.trades.append(trade)
                except Exception as e:
                    continue

        logger.info(f"Loaded {len(self.trades)} resolved trades")

    def analyze_pattern(
        self,
        name: str,
        description: str,
        filter_func,
    ) -> PatternResult:
        """Analyze a specific pattern."""
        matching = [t for t in self.trades if filter_func(t)]

        if not matching:
            return PatternResult(
                name=name,
                description=description,
                trades=0,
                win_rate=0,
                breakeven_rate=0,
                edge=0,
                roi=0,
                total_profit=0,
                is_significant=False,
            )

        wins = sum(1 for t in matching if t.is_winner)
        total_cost = sum(t.cost_dollars for t in matching)
        total_profit = sum(t.actual_profit_dollars for t in matching)
        avg_price = sum(t.trade_price for t in matching) / len(matching)

        win_rate = wins / len(matching)
        breakeven_rate = avg_price / 100
        edge = win_rate - breakeven_rate
        roi = total_profit / total_cost if total_cost > 0 else 0

        return PatternResult(
            name=name,
            description=description,
            trades=len(matching),
            win_rate=win_rate,
            breakeven_rate=breakeven_rate,
            edge=edge,
            roi=roi,
            total_profit=total_profit,
            is_significant=len(matching) >= 100 and edge > 0,
        )

    def test_timing_patterns(self) -> List[PatternResult]:
        """Test if trades closer to market close are more accurate."""
        results = []

        # Different time windows before close
        windows = [
            (0, 60, "Last 60 min before close"),
            (60, 180, "1-3 hours before close"),
            (180, 720, "3-12 hours before close"),
            (720, 1440, "12-24 hours before close"),
            (1440, 10080, "1-7 days before close"),
        ]

        for min_minutes, max_minutes, desc in windows:
            def make_filter(min_m, max_m):
                def f(t):
                    if t.minutes_before_close is None:
                        return False
                    return min_m <= t.minutes_before_close < max_m and t.count >= 100
                return f

            result = self.analyze_pattern(
                name=f"Whale timing: {desc}",
                description=f"Whales (>=100) trading {desc}",
                filter_func=make_filter(min_minutes, max_minutes),
            )
            results.append(result)

        return results

    def test_contrarian_patterns(self) -> List[PatternResult]:
        """Test contrarian bets (betting against strong consensus)."""
        results = []

        # Contrarian at different consensus levels
        levels = [
            (85, 99, "Betting against 85-99% consensus"),
            (90, 99, "Betting against 90-99% consensus"),
            (95, 99, "Betting against 95-99% consensus"),
        ]

        for min_price, max_price, desc in levels:
            # Contrarian = betting NO when price is very high (consensus is YES)
            # or betting YES when price is very low (consensus is NO)
            def make_filter(min_p, max_p):
                def f(t):
                    if t.count < 100:
                        return False
                    # High price = consensus is YES, contrarian bets NO
                    if min_p <= t.trade_price <= max_p and t.taker_side == 'no':
                        return True
                    # Low price = consensus is NO, contrarian bets YES
                    if (100 - max_p) <= t.trade_price <= (100 - min_p) and t.taker_side == 'yes':
                        return True
                    return False
                return f

            result = self.analyze_pattern(
                name=f"Contrarian whale: {desc}",
                description=desc,
                filter_func=make_filter(min_price, max_price),
            )
            results.append(result)

        return results

    def test_size_divergence(self) -> List[PatternResult]:
        """Test if small and large traders disagree, who wins."""
        results = []

        # Group trades by market and side
        market_sides = defaultdict(lambda: {'small_yes': 0, 'small_no': 0, 'large_yes': 0, 'large_no': 0, 'result': None})

        for t in self.trades:
            key = t.market_ticker
            market_sides[key]['result'] = 'yes' if t.is_winner == (t.taker_side == 'yes') else 'no'

            if t.count < 50:
                market_sides[key][f'small_{t.taker_side}'] += t.count
            else:
                market_sides[key][f'large_{t.taker_side}'] += t.count

        # Find markets where small and large disagree
        small_right = 0
        large_right = 0
        disagreements = 0

        for ticker, data in market_sides.items():
            small_pref = 'yes' if data['small_yes'] > data['small_no'] else 'no'
            large_pref = 'yes' if data['large_yes'] > data['large_no'] else 'no'

            if small_pref != large_pref and data['result']:
                disagreements += 1
                if data['result'] == small_pref:
                    small_right += 1
                else:
                    large_right += 1

        if disagreements > 0:
            results.append(PatternResult(
                name="When small/large disagree, follow large",
                description=f"Large traders beat small in {100*large_right/disagreements:.1f}% of {disagreements} disagreements",
                trades=disagreements,
                win_rate=large_right/disagreements,
                breakeven_rate=0.5,
                edge=large_right/disagreements - 0.5,
                roi=0,  # Can't calculate directly
                total_profit=0,
                is_significant=disagreements >= 50 and large_right/disagreements > 0.5,
            ))

        return results

    def test_category_patterns(self) -> List[PatternResult]:
        """Test if certain categories have different whale accuracy."""
        results = []

        # Group by category prefix
        categories = defaultdict(list)
        for t in self.trades:
            if t.count >= 100:  # Whales only
                # Extract category from ticker prefix
                parts = t.market_ticker.split('-')
                if parts:
                    cat = parts[0][:4]  # First 4 chars of ticker
                    categories[cat].append(t)

        for cat, trades in sorted(categories.items(), key=lambda x: -len(x[1])):
            if len(trades) < 50:
                continue

            wins = sum(1 for t in trades if t.is_winner)
            total_cost = sum(t.cost_dollars for t in trades)
            total_profit = sum(t.actual_profit_dollars for t in trades)
            avg_price = sum(t.trade_price for t in trades) / len(trades)

            results.append(PatternResult(
                name=f"Category: {cat}",
                description=f"Whale trades in {cat} markets",
                trades=len(trades),
                win_rate=wins/len(trades),
                breakeven_rate=avg_price/100,
                edge=wins/len(trades) - avg_price/100,
                roi=total_profit/total_cost if total_cost > 0 else 0,
                total_profit=total_profit,
                is_significant=len(trades) >= 100 and wins/len(trades) > avg_price/100,
            ))

        return results[:10]  # Top 10 categories

    def test_price_side_interaction(self) -> List[PatternResult]:
        """Test YES vs NO at different price levels."""
        results = []

        price_ranges = [
            (30, 50, "30-50c"),
            (50, 70, "50-70c"),
            (30, 70, "30-70c (sweet spot)"),
        ]

        for min_p, max_p, desc in price_ranges:
            for side in ['yes', 'no']:
                def make_filter(min_price, max_price, taker_side):
                    def f(t):
                        return (
                            t.count >= 100 and
                            min_price <= t.trade_price <= max_price and
                            t.taker_side == taker_side
                        )
                    return f

                result = self.analyze_pattern(
                    name=f"Whale {side.upper()} at {desc}",
                    description=f"Whale {side} bets in {desc} range",
                    filter_func=make_filter(min_p, max_p, side),
                )
                results.append(result)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive pattern analysis report."""
        lines = []

        lines.append("=" * 80)
        lines.append("ADVANCED PATTERN ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Resolved Trades: {len(self.trades):,}")
        lines.append("")

        # Timing patterns
        lines.append("-" * 80)
        lines.append("1. TIMING PATTERNS (Does trading close to market close help?)")
        lines.append("-" * 80)
        lines.append(f"{'Pattern':<40} {'Trades':>8} {'Win Rate':>10} {'Edge':>10} {'Significant':>12}")
        lines.append("-" * 80)

        for r in self.test_timing_patterns():
            sig = "YES" if r.is_significant else "no"
            lines.append(f"{r.name:<40} {r.trades:>8} {100*r.win_rate:>9.1f}% {100*r.edge:>+9.1f}% {sig:>12}")

        # Contrarian patterns
        lines.append("")
        lines.append("-" * 80)
        lines.append("2. CONTRARIAN PATTERNS (Betting against strong consensus)")
        lines.append("-" * 80)
        lines.append(f"{'Pattern':<40} {'Trades':>8} {'Win Rate':>10} {'Edge':>10} {'Significant':>12}")
        lines.append("-" * 80)

        for r in self.test_contrarian_patterns():
            sig = "YES" if r.is_significant else "no"
            lines.append(f"{r.name:<40} {r.trades:>8} {100*r.win_rate:>9.1f}% {100*r.edge:>+9.1f}% {sig:>12}")

        # Size divergence
        lines.append("")
        lines.append("-" * 80)
        lines.append("3. SIZE DIVERGENCE (When small/large traders disagree)")
        lines.append("-" * 80)

        for r in self.test_size_divergence():
            lines.append(f"  {r.description}")
            if r.is_significant:
                lines.append(f"  STATUS: SIGNIFICANT - Large traders have edge")
            else:
                lines.append(f"  STATUS: Not significant")

        # Category patterns
        lines.append("")
        lines.append("-" * 80)
        lines.append("4. CATEGORY PATTERNS (Which market types have whale edge?)")
        lines.append("-" * 80)
        lines.append(f"{'Category':<20} {'Trades':>8} {'Win Rate':>10} {'B/E Rate':>10} {'Edge':>10} {'ROI':>10}")
        lines.append("-" * 80)

        for r in self.test_category_patterns():
            lines.append(f"{r.name:<20} {r.trades:>8} {100*r.win_rate:>9.1f}% {100*r.breakeven_rate:>9.1f}% {100*r.edge:>+9.1f}% {100*r.roi:>+9.1f}%")

        # Price-side interaction
        lines.append("")
        lines.append("-" * 80)
        lines.append("5. PRICE-SIDE INTERACTION (YES vs NO at different prices)")
        lines.append("-" * 80)
        lines.append(f"{'Pattern':<35} {'Trades':>8} {'Win Rate':>10} {'Edge':>10} {'ROI':>10}")
        lines.append("-" * 80)

        for r in self.test_price_side_interaction():
            lines.append(f"{r.name:<35} {r.trades:>8} {100*r.win_rate:>9.1f}% {100*r.edge:>+9.1f}% {100*r.roi:>+9.1f}%")

        # Summary of significant patterns
        lines.append("")
        lines.append("=" * 80)
        lines.append("SIGNIFICANT PATTERNS SUMMARY")
        lines.append("=" * 80)

        all_patterns = (
            self.test_timing_patterns() +
            self.test_contrarian_patterns() +
            self.test_category_patterns() +
            self.test_price_side_interaction()
        )

        significant = [p for p in all_patterns if p.is_significant]
        significant.sort(key=lambda p: p.edge, reverse=True)

        if significant:
            lines.append(f"\nFound {len(significant)} significant patterns:")
            for p in significant[:10]:
                lines.append(f"  - {p.name}: +{100*p.edge:.1f}% edge ({p.trades} trades)")
        else:
            lines.append("\nNo additional significant patterns found beyond baseline.")

        lines.append("")
        lines.append("=" * 80)
        lines.append("ANALYSIS COMPLETE")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Advanced pattern analysis')
    parser.add_argument('--trades', required=True, help='Trades CSV')
    parser.add_argument('--outcomes', required=True, help='Outcomes CSV')
    parser.add_argument('--report', action='store_true', help='Generate report')

    args = parser.parse_args()

    analyzer = AdvancedAnalyzer()
    analyzer.load_data(args.trades, args.outcomes)

    if args.report:
        report = analyzer.generate_report()
        print(report)

        # Save report
        report_path = Path(args.trades).parent / "advanced_pattern_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
