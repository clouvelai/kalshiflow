#!/usr/bin/env python3
"""
All Trades Pattern Analysis - Find profitable patterns across ALL trade sizes.

This script goes beyond whale-only analysis to examine:
1. Size threshold analysis - At what contract count does edge appear?
2. Small trade patterns - Are small YES bets a fade signal?
3. Volume clustering - Do rapid consecutive trades predict outcomes?
4. Time-of-day patterns - Morning vs afternoon vs evening
5. Cascade patterns - Do same-direction sequences predict continuation?
6. Category deep-dive - Which categories have edge at any size?

Key Questions:
- Do small trades (<50 contracts) have any predictive value?
- Is there a "dumb money" signal we can fade?
- Do volume clusters predict outcomes?
- Are there category-specific patterns for non-whales?
- Is there a minimum trade size where edge appears?
- Do rapid consecutive trades signal anything?

Usage:
    python analyze_all_trades_patterns.py --enriched enriched_trades_final.csv --report

Author: Claude Code
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade data for pattern analysis."""
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

    @property
    def hour(self) -> int:
        return self.datetime.hour

    @property
    def day_of_week(self) -> int:
        return self.datetime.weekday()

    @property
    def day_name(self) -> str:
        return self.datetime.strftime('%A')

    @property
    def is_longshot(self) -> bool:
        return self.trade_price <= 20

    @property
    def is_favorite(self) -> bool:
        return self.trade_price >= 80

    @property
    def is_midrange(self) -> bool:
        return 30 <= self.trade_price <= 70

    @property
    def category_prefix(self) -> str:
        """Extract category from ticker prefix."""
        parts = self.market_ticker.split('-')
        if parts:
            return parts[0][:6]  # First 6 chars
        return "UNKNOWN"


@dataclass
class PatternStats:
    """Statistics for a pattern."""
    name: str
    description: str
    trades: int
    wins: int
    total_cost: float
    total_profit: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0

    @property
    def avg_price(self) -> float:
        return self._avg_price if hasattr(self, '_avg_price') else 50

    @avg_price.setter
    def avg_price(self, value: float):
        self._avg_price = value

    @property
    def breakeven_rate(self) -> float:
        return self.avg_price / 100

    @property
    def edge(self) -> float:
        return self.win_rate - self.breakeven_rate

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_cost if self.total_cost > 0 else 0

    @property
    def is_significant(self) -> bool:
        return self.trades >= 100

    @property
    def is_profitable(self) -> bool:
        return self.roi > 0 and self.trades >= 100


class AllTradesAnalyzer:
    """Comprehensive pattern analysis for all trade sizes."""

    # Size thresholds to test
    SIZE_THRESHOLDS = [5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

    # Time periods (hours)
    TIME_PERIODS = {
        'early_morning': (4, 8),    # 4am-8am
        'morning': (8, 12),          # 8am-12pm
        'afternoon': (12, 17),       # 12pm-5pm
        'evening': (17, 21),         # 5pm-9pm
        'night': (21, 4),            # 9pm-4am (wraps)
    }

    def __init__(self):
        self.trades: List[Trade] = []
        self.trades_by_market: Dict[str, List[Trade]] = defaultdict(list)

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
                    self.trades_by_market[trade.market_ticker].append(trade)
                    count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed row: {e}")

        # Sort by timestamp
        self.trades.sort(key=lambda t: t.timestamp)
        for ticker in self.trades_by_market:
            self.trades_by_market[ticker].sort(key=lambda t: t.timestamp)

        logger.info(f"Loaded {count:,} resolved trades from {csv_path}")
        logger.info(f"Skipped {skipped:,} unresolved trades")
        logger.info(f"Markets with resolved trades: {len(self.trades_by_market):,}")
        return count

    def compute_stats(self, trades: List[Trade], name: str, description: str) -> PatternStats:
        """Compute statistics for a set of trades."""
        if not trades:
            stats = PatternStats(
                name=name, description=description,
                trades=0, wins=0, total_cost=0, total_profit=0
            )
            stats.avg_price = 50
            return stats

        wins = sum(1 for t in trades if t.is_winner)
        total_cost = sum(t.cost_dollars for t in trades)
        total_profit = sum(t.actual_profit_dollars for t in trades)
        avg_price = sum(t.trade_price for t in trades) / len(trades)

        stats = PatternStats(
            name=name,
            description=description,
            trades=len(trades),
            wins=wins,
            total_cost=total_cost,
            total_profit=total_profit
        )
        stats.avg_price = avg_price
        return stats

    # =========================================================================
    # 1. SIZE THRESHOLD ANALYSIS
    # =========================================================================

    def analyze_size_thresholds(self) -> List[PatternStats]:
        """Test at what contract count edge appears."""
        results = []

        # Test each threshold: >= threshold
        for threshold in self.SIZE_THRESHOLDS:
            matching = [t for t in self.trades if t.count >= threshold]
            stats = self.compute_stats(
                matching,
                f"Trades >= {threshold} contracts",
                f"All trades with {threshold}+ contracts"
            )
            results.append(stats)

        # Test size ranges
        ranges = [
            (1, 10, "Micro (1-9)"),
            (10, 25, "Small (10-24)"),
            (25, 50, "Medium-Small (25-49)"),
            (50, 100, "Medium (50-99)"),
            (100, 250, "Large (100-249)"),
            (250, 500, "Very Large (250-499)"),
            (500, 1000, "Huge (500-999)"),
            (1000, 100000, "Mega (1000+)"),
        ]

        for min_size, max_size, name in ranges:
            matching = [t for t in self.trades if min_size <= t.count < max_size]
            stats = self.compute_stats(
                matching,
                f"Size: {name}",
                f"Trades with {min_size} to {max_size-1} contracts"
            )
            results.append(stats)

        return results

    def analyze_size_by_side(self) -> List[PatternStats]:
        """Analyze size effect separately for YES and NO trades."""
        results = []

        for side in ['yes', 'no']:
            for threshold in [25, 50, 100, 200]:
                matching = [t for t in self.trades if t.count >= threshold and t.taker_side == side]
                stats = self.compute_stats(
                    matching,
                    f"{side.upper()} >= {threshold}",
                    f"{side.upper()} trades with {threshold}+ contracts"
                )
                results.append(stats)

        return results

    # =========================================================================
    # 2. SMALL TRADE PATTERNS ("DUMB MONEY" SIGNALS)
    # =========================================================================

    def analyze_small_trades(self) -> List[PatternStats]:
        """Analyze small trades for fade signals."""
        results = []

        # Small trades by side
        for side in ['yes', 'no']:
            for max_size in [10, 25, 50]:
                matching = [t for t in self.trades if t.count <= max_size and t.taker_side == side]
                stats = self.compute_stats(
                    matching,
                    f"Small {side.upper()} (<={max_size})",
                    f"Small {side} bets with {max_size} or fewer contracts"
                )
                results.append(stats)

        # Small trades by price range
        for max_size in [25, 50]:
            # Small longshots (gambling?)
            matching = [t for t in self.trades
                       if t.count <= max_size and t.is_longshot]
            stats = self.compute_stats(
                matching,
                f"Small longshots (<={max_size})",
                f"Small trades at longshot prices (<=20c)"
            )
            results.append(stats)

            # Small favorites
            matching = [t for t in self.trades
                       if t.count <= max_size and t.is_favorite]
            stats = self.compute_stats(
                matching,
                f"Small favorites (<={max_size})",
                f"Small trades at favorite prices (>=80c)"
            )
            results.append(stats)

        return results

    def analyze_retail_fade(self) -> Dict[str, Any]:
        """
        Test if fading small trades is profitable.
        Calculate: when small traders favor one side, does the other side win?
        """
        # Group trades by market
        market_analysis = {}

        for ticker, trades in self.trades_by_market.items():
            small_yes = sum(t.count for t in trades if t.count < 50 and t.taker_side == 'yes')
            small_no = sum(t.count for t in trades if t.count < 50 and t.taker_side == 'no')

            # Determine outcome (from any trade's is_winner)
            yes_trades = [t for t in trades if t.taker_side == 'yes']
            no_trades = [t for t in trades if t.taker_side == 'no']

            market_result = None
            if yes_trades and yes_trades[0].is_winner:
                market_result = 'yes'
            elif no_trades and no_trades[0].is_winner:
                market_result = 'no'
            elif yes_trades:
                market_result = 'no'  # YES lost
            elif no_trades:
                market_result = 'yes'  # NO lost

            if market_result and (small_yes > 0 or small_no > 0):
                market_analysis[ticker] = {
                    'small_yes': small_yes,
                    'small_no': small_no,
                    'small_pref': 'yes' if small_yes > small_no else 'no',
                    'result': market_result,
                    'retail_right': (small_yes > small_no) == (market_result == 'yes')
                }

        # Calculate fade signal accuracy
        total_markets = len(market_analysis)
        retail_right = sum(1 for m in market_analysis.values() if m['retail_right'])
        retail_wrong = total_markets - retail_right

        # Strong preference filter (>70% one side)
        strong_pref_markets = {}
        for ticker, data in market_analysis.items():
            total = data['small_yes'] + data['small_no']
            if total > 0:
                yes_pct = data['small_yes'] / total
                if yes_pct >= 0.7 or yes_pct <= 0.3:
                    strong_pref_markets[ticker] = data

        strong_retail_right = sum(1 for m in strong_pref_markets.values() if m['retail_right'])

        return {
            'total_markets': total_markets,
            'retail_right': retail_right,
            'retail_wrong': retail_wrong,
            'retail_accuracy': retail_right / total_markets if total_markets > 0 else 0,
            'fade_accuracy': retail_wrong / total_markets if total_markets > 0 else 0,
            'strong_pref_markets': len(strong_pref_markets),
            'strong_pref_retail_right': strong_retail_right,
            'strong_fade_accuracy': (len(strong_pref_markets) - strong_retail_right) / len(strong_pref_markets) if strong_pref_markets else 0,
        }

    # =========================================================================
    # 3. VOLUME CLUSTERING PATTERNS
    # =========================================================================

    def analyze_volume_clusters(self) -> List[Dict[str, Any]]:
        """
        Analyze what happens when multiple trades occur on same market within short time.
        """
        results = []

        # Find trade clusters (5+ trades within 5 minutes)
        cluster_window_ms = 5 * 60 * 1000  # 5 minutes

        for ticker, trades in self.trades_by_market.items():
            if len(trades) < 5:
                continue

            # Find clusters
            i = 0
            while i < len(trades):
                cluster = [trades[i]]
                j = i + 1

                while j < len(trades) and trades[j].timestamp - trades[i].timestamp <= cluster_window_ms:
                    cluster.append(trades[j])
                    j += 1

                if len(cluster) >= 5:
                    # Analyze cluster
                    yes_volume = sum(t.count for t in cluster if t.taker_side == 'yes')
                    no_volume = sum(t.count for t in cluster if t.taker_side == 'no')
                    cluster_direction = 'yes' if yes_volume > no_volume else 'no'
                    total_volume = yes_volume + no_volume
                    direction_strength = max(yes_volume, no_volume) / total_volume if total_volume > 0 else 0

                    # Get outcome
                    outcome = 'yes' if cluster[0].is_winner == (cluster[0].taker_side == 'yes') else 'no'

                    results.append({
                        'ticker': ticker,
                        'cluster_size': len(cluster),
                        'total_volume': total_volume,
                        'cluster_direction': cluster_direction,
                        'direction_strength': direction_strength,
                        'outcome': outcome,
                        'direction_correct': cluster_direction == outcome,
                    })

                i = j if j > i + 1 else i + 1

        # Summarize
        if results:
            total_clusters = len(results)
            direction_wins = sum(1 for r in results if r['direction_correct'])

            # Strong clusters (>70% one direction)
            strong_clusters = [r for r in results if r['direction_strength'] >= 0.7]
            strong_wins = sum(1 for r in strong_clusters if r['direction_correct'])

            return {
                'total_clusters': total_clusters,
                'direction_wins': direction_wins,
                'direction_accuracy': direction_wins / total_clusters if total_clusters > 0 else 0,
                'strong_clusters': len(strong_clusters),
                'strong_wins': strong_wins,
                'strong_accuracy': strong_wins / len(strong_clusters) if strong_clusters else 0,
                'raw_clusters': results[:20],  # Sample for inspection
            }

        return {'total_clusters': 0, 'direction_accuracy': 0}

    # =========================================================================
    # 4. TIME OF DAY PATTERNS
    # =========================================================================

    def analyze_time_of_day(self) -> List[PatternStats]:
        """Analyze patterns by time of day (in user's local time)."""
        results = []

        for period_name, (start_hour, end_hour) in self.TIME_PERIODS.items():
            if start_hour < end_hour:
                matching = [t for t in self.trades if start_hour <= t.hour < end_hour]
            else:  # Wraps around midnight
                matching = [t for t in self.trades if t.hour >= start_hour or t.hour < end_hour]

            stats = self.compute_stats(
                matching,
                f"Time: {period_name}",
                f"Trades during {period_name} hours ({start_hour}:00-{end_hour}:00)"
            )
            results.append(stats)

        # Also analyze by specific hours
        for hour in range(24):
            matching = [t for t in self.trades if t.hour == hour]
            if len(matching) >= 100:
                stats = self.compute_stats(
                    matching,
                    f"Hour: {hour:02d}:00",
                    f"Trades at {hour:02d}:00"
                )
                results.append(stats)

        return results

    def analyze_day_of_week(self) -> List[PatternStats]:
        """Analyze patterns by day of week."""
        results = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day_idx, day_name in enumerate(day_names):
            matching = [t for t in self.trades if t.day_of_week == day_idx]
            stats = self.compute_stats(
                matching,
                f"Day: {day_name}",
                f"Trades on {day_name}"
            )
            results.append(stats)

        # Weekend vs weekday
        weekend = [t for t in self.trades if t.day_of_week >= 5]
        weekday = [t for t in self.trades if t.day_of_week < 5]

        results.append(self.compute_stats(weekend, "Weekend (Sat-Sun)", "Trades on weekends"))
        results.append(self.compute_stats(weekday, "Weekday (Mon-Fri)", "Trades on weekdays"))

        return results

    def analyze_time_by_size(self) -> List[PatternStats]:
        """Combine time and size analysis."""
        results = []

        for period_name, (start_hour, end_hour) in self.TIME_PERIODS.items():
            if start_hour < end_hour:
                period_trades = [t for t in self.trades if start_hour <= t.hour < end_hour]
            else:
                period_trades = [t for t in self.trades if t.hour >= start_hour or t.hour < end_hour]

            # Whale trades during this period
            whales = [t for t in period_trades if t.count >= 100]
            if len(whales) >= 50:
                stats = self.compute_stats(
                    whales,
                    f"Whale @ {period_name}",
                    f"Whale trades (>=100) during {period_name}"
                )
                results.append(stats)

            # Small trades during this period
            small = [t for t in period_trades if t.count < 50]
            if len(small) >= 100:
                stats = self.compute_stats(
                    small,
                    f"Small @ {period_name}",
                    f"Small trades (<50) during {period_name}"
                )
                results.append(stats)

        return results

    # =========================================================================
    # 5. CASCADE / SEQUENCE PATTERNS
    # =========================================================================

    def analyze_cascade_patterns(self) -> Dict[str, Any]:
        """
        After 3+ same-direction trades, does the move continue or reverse?
        """
        # Find sequences of 3+ same-direction trades within each market
        cascade_results = []

        for ticker, trades in self.trades_by_market.items():
            if len(trades) < 3:
                continue

            i = 0
            while i < len(trades) - 2:
                # Find consecutive same-side trades
                current_side = trades[i].taker_side
                streak = [trades[i]]

                j = i + 1
                while j < len(trades) and trades[j].taker_side == current_side:
                    streak.append(trades[j])
                    j += 1

                if len(streak) >= 3:
                    # We have a cascade - does this side win?
                    outcome = 'yes' if streak[0].is_winner == (streak[0].taker_side == 'yes') else 'no'
                    cascade_correct = current_side == outcome

                    total_volume = sum(t.count for t in streak)

                    cascade_results.append({
                        'ticker': ticker,
                        'cascade_side': current_side,
                        'streak_length': len(streak),
                        'total_volume': total_volume,
                        'outcome': outcome,
                        'cascade_correct': cascade_correct,
                    })

                i = j if j > i else i + 1

        if not cascade_results:
            return {'total_cascades': 0}

        # Summarize by streak length
        by_length = defaultdict(list)
        for r in cascade_results:
            by_length[min(r['streak_length'], 10)].append(r)  # Cap at 10+

        length_summary = {}
        for length, cascades in sorted(by_length.items()):
            wins = sum(1 for c in cascades if c['cascade_correct'])
            length_summary[length] = {
                'count': len(cascades),
                'wins': wins,
                'accuracy': wins / len(cascades) if cascades else 0,
            }

        total = len(cascade_results)
        wins = sum(1 for r in cascade_results if r['cascade_correct'])

        return {
            'total_cascades': total,
            'cascade_wins': wins,
            'cascade_accuracy': wins / total if total > 0 else 0,
            'by_length': length_summary,
        }

    # =========================================================================
    # 6. CATEGORY DEEP-DIVE
    # =========================================================================

    def analyze_categories(self) -> List[PatternStats]:
        """Analyze patterns by market category."""
        results = []

        # Group by category prefix
        by_category = defaultdict(list)
        for t in self.trades:
            by_category[t.category_prefix].append(t)

        # All trades by category
        for cat, trades in sorted(by_category.items(), key=lambda x: -len(x[1])):
            if len(trades) < 100:
                continue

            stats = self.compute_stats(
                trades,
                f"Cat: {cat}",
                f"All trades in {cat} markets"
            )
            results.append(stats)

        return results

    def analyze_categories_by_size(self) -> List[PatternStats]:
        """Analyze category patterns at different size thresholds."""
        results = []

        by_category = defaultdict(list)
        for t in self.trades:
            by_category[t.category_prefix].append(t)

        # Top 10 categories by volume
        top_categories = sorted(by_category.items(), key=lambda x: -len(x[1]))[:10]

        for cat, trades in top_categories:
            for min_size in [25, 50, 100]:
                filtered = [t for t in trades if t.count >= min_size]
                if len(filtered) >= 50:
                    stats = self.compute_stats(
                        filtered,
                        f"{cat} >= {min_size}",
                        f"{cat} trades with {min_size}+ contracts"
                    )
                    results.append(stats)

        return results

    def analyze_categories_by_side(self) -> List[PatternStats]:
        """Analyze YES vs NO performance within categories."""
        results = []

        by_category = defaultdict(list)
        for t in self.trades:
            by_category[t.category_prefix].append(t)

        top_categories = sorted(by_category.items(), key=lambda x: -len(x[1]))[:10]

        for cat, trades in top_categories:
            for side in ['yes', 'no']:
                filtered = [t for t in trades if t.taker_side == side and t.count >= 50]
                if len(filtered) >= 50:
                    stats = self.compute_stats(
                        filtered,
                        f"{cat} {side.upper()} (>=50)",
                        f"{cat} {side} trades with 50+ contracts"
                    )
                    results.append(stats)

        return results

    # =========================================================================
    # 7. PRICE BUCKET ANALYSIS
    # =========================================================================

    def analyze_price_buckets(self) -> List[PatternStats]:
        """Analyze performance by price buckets."""
        results = []

        buckets = [
            (1, 10, "1-10c (extreme longshot)"),
            (10, 20, "10-20c (longshot)"),
            (20, 30, "20-30c (underdog)"),
            (30, 40, "30-40c (lean no)"),
            (40, 50, "40-50c (toss-up low)"),
            (50, 60, "50-60c (toss-up high)"),
            (60, 70, "60-70c (lean yes)"),
            (70, 80, "70-80c (favorite)"),
            (80, 90, "80-90c (strong favorite)"),
            (90, 100, "90-99c (extreme favorite)"),
        ]

        for min_price, max_price, name in buckets:
            matching = [t for t in self.trades if min_price <= t.trade_price < max_price]
            stats = self.compute_stats(matching, f"Price: {name}", f"Trades at {name}")
            results.append(stats)

        return results

    def analyze_price_side_interaction(self) -> List[PatternStats]:
        """Analyze YES vs NO at different price levels."""
        results = []

        # Key price ranges with side breakdown
        ranges = [
            (30, 50, "30-50c"),
            (50, 70, "50-70c"),
            (70, 85, "70-85c"),
        ]

        for min_p, max_p, name in ranges:
            for side in ['yes', 'no']:
                for min_size in [25, 50, 100]:
                    matching = [t for t in self.trades
                               if min_p <= t.trade_price < max_p
                               and t.taker_side == side
                               and t.count >= min_size]
                    if len(matching) >= 50:
                        stats = self.compute_stats(
                            matching,
                            f"{side.upper()} @ {name} (>={min_size})",
                            f"{side} trades at {name} with {min_size}+ contracts"
                        )
                        results.append(stats)

        return results

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        lines = []

        lines.append("=" * 90)
        lines.append("ALL TRADES PATTERN ANALYSIS")
        lines.append("Beyond Whale-Only: Finding Edge at Every Size")
        lines.append("=" * 90)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Resolved Trades: {len(self.trades):,}")
        lines.append(f"Markets Analyzed: {len(self.trades_by_market):,}")

        # Overall stats
        wins = sum(1 for t in self.trades if t.is_winner)
        total_cost = sum(t.cost_dollars for t in self.trades)
        total_profit = sum(t.actual_profit_dollars for t in self.trades)
        lines.append(f"Overall Win Rate: {100 * wins / len(self.trades):.1f}%")
        lines.append(f"Overall ROI: {100 * total_profit / total_cost:+.1f}%")
        lines.append("")

        # =====================================================================
        # SECTION 1: SIZE THRESHOLD ANALYSIS
        # =====================================================================
        lines.append("-" * 90)
        lines.append("1. SIZE THRESHOLD ANALYSIS")
        lines.append("   At what contract count does edge appear?")
        lines.append("-" * 90)

        header = f"{'Pattern':<35} {'Trades':>8} {'Win Rate':>10} {'B/E Rate':>10} {'Edge':>10} {'ROI':>10}"
        lines.append(header)
        lines.append("-" * 90)

        for stats in self.analyze_size_thresholds():
            if stats.trades >= 50:
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        lines.append("\nSize by Side:")
        lines.append("-" * 90)
        for stats in self.analyze_size_by_side():
            if stats.trades >= 50:
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        # =====================================================================
        # SECTION 2: SMALL TRADE / DUMB MONEY ANALYSIS
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("2. SMALL TRADE PATTERNS ('Dumb Money' Analysis)")
        lines.append("   Do small trades have predictive value? Can we fade them?")
        lines.append("-" * 90)

        lines.append(header)
        lines.append("-" * 90)

        for stats in self.analyze_small_trades():
            if stats.trades >= 50:
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%"
                )

        lines.append("\nRetail Fade Signal Analysis:")
        fade_data = self.analyze_retail_fade()
        lines.append(f"  Markets analyzed: {fade_data['total_markets']:,}")
        lines.append(f"  Retail correct: {fade_data['retail_right']:,} ({100*fade_data['retail_accuracy']:.1f}%)")
        lines.append(f"  Fade signal correct: {fade_data['retail_wrong']:,} ({100*fade_data['fade_accuracy']:.1f}%)")
        lines.append(f"  Strong preference markets (>70% one side): {fade_data['strong_pref_markets']:,}")
        if fade_data['strong_pref_markets'] > 0:
            lines.append(f"  Strong fade accuracy: {100*fade_data['strong_fade_accuracy']:.1f}%")

        # =====================================================================
        # SECTION 3: VOLUME CLUSTERING
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("3. VOLUME CLUSTERING ANALYSIS")
        lines.append("   When 5+ trades occur within 5 minutes, what happens?")
        lines.append("-" * 90)

        cluster_data = self.analyze_volume_clusters()
        if cluster_data.get('total_clusters', 0) > 0:
            lines.append(f"  Total trade clusters found: {cluster_data['total_clusters']:,}")
            lines.append(f"  Cluster direction predicts outcome: {100*cluster_data['direction_accuracy']:.1f}%")
            lines.append(f"  Strong clusters (>70% one direction): {cluster_data.get('strong_clusters', 0):,}")
            if cluster_data.get('strong_clusters', 0) > 0:
                lines.append(f"  Strong cluster accuracy: {100*cluster_data.get('strong_accuracy', 0):.1f}%")
        else:
            lines.append("  No significant trade clusters found.")

        # =====================================================================
        # SECTION 4: TIME OF DAY PATTERNS
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("4. TIME OF DAY PATTERNS")
        lines.append("   Morning vs afternoon vs evening trading")
        lines.append("-" * 90)

        lines.append(header)
        lines.append("-" * 90)

        for stats in self.analyze_time_of_day():
            if stats.trades >= 100 and 'Hour' not in stats.name:  # Skip hourly for main view
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        lines.append("\nDay of Week:")
        for stats in self.analyze_day_of_week():
            if stats.trades >= 100:
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        lines.append("\nTime + Size Interaction:")
        for stats in self.analyze_time_by_size():
            if stats.trades >= 50:
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        # =====================================================================
        # SECTION 5: CASCADE PATTERNS
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("5. CASCADE / SEQUENCE PATTERNS")
        lines.append("   After 3+ same-direction trades, does the move continue?")
        lines.append("-" * 90)

        cascade_data = self.analyze_cascade_patterns()
        if cascade_data.get('total_cascades', 0) > 0:
            lines.append(f"  Total cascades found (3+ same-direction): {cascade_data['total_cascades']:,}")
            lines.append(f"  Cascade direction wins: {100*cascade_data['cascade_accuracy']:.1f}%")
            lines.append("\n  By streak length:")
            for length, data in sorted(cascade_data.get('by_length', {}).items()):
                if data['count'] >= 20:
                    lines.append(f"    {length}+ trades in a row: {data['count']:,} cascades, {100*data['accuracy']:.1f}% accuracy")
        else:
            lines.append("  No significant cascades found.")

        # =====================================================================
        # SECTION 6: CATEGORY DEEP-DIVE
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("6. CATEGORY ANALYSIS")
        lines.append("   Which market categories have edge at any size?")
        lines.append("-" * 90)

        lines.append(header)
        lines.append("-" * 90)

        for stats in self.analyze_categories():
            mark = " ***" if stats.is_profitable else ""
            lines.append(
                f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
            )

        lines.append("\nTop Categories by Size:")
        for stats in self.analyze_categories_by_size():
            mark = " ***" if stats.is_profitable else ""
            lines.append(
                f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
            )

        lines.append("\nCategory YES vs NO (>=50 contracts):")
        for stats in self.analyze_categories_by_side():
            mark = " ***" if stats.is_profitable else ""
            lines.append(
                f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
            )

        # =====================================================================
        # SECTION 7: PRICE BUCKET ANALYSIS
        # =====================================================================
        lines.append("")
        lines.append("-" * 90)
        lines.append("7. PRICE BUCKET ANALYSIS")
        lines.append("   Performance across different price ranges")
        lines.append("-" * 90)

        lines.append(header)
        lines.append("-" * 90)

        for stats in self.analyze_price_buckets():
            if stats.trades >= 50:
                mark = " ***" if stats.is_profitable else ""
                lines.append(
                    f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                    f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
                )

        lines.append("\nPrice + Side + Size Interaction:")
        for stats in self.analyze_price_side_interaction():
            mark = " ***" if stats.is_profitable else ""
            lines.append(
                f"{stats.name:<35} {stats.trades:>8,} {100*stats.win_rate:>9.1f}% "
                f"{100*stats.breakeven_rate:>9.1f}% {100*stats.edge:>+9.1f}% {100*stats.roi:>+9.1f}%{mark}"
            )

        # =====================================================================
        # SUMMARY: PROFITABLE PATTERNS
        # =====================================================================
        lines.append("")
        lines.append("=" * 90)
        lines.append("PROFITABLE PATTERNS SUMMARY")
        lines.append("Patterns with >= 100 trades and positive ROI")
        lines.append("=" * 90)

        # Collect all patterns
        all_patterns = (
            self.analyze_size_thresholds() +
            self.analyze_size_by_side() +
            self.analyze_small_trades() +
            self.analyze_time_of_day() +
            self.analyze_day_of_week() +
            self.analyze_time_by_size() +
            self.analyze_categories() +
            self.analyze_categories_by_size() +
            self.analyze_categories_by_side() +
            self.analyze_price_buckets() +
            self.analyze_price_side_interaction()
        )

        profitable = [p for p in all_patterns if p.is_profitable]
        profitable.sort(key=lambda p: p.roi, reverse=True)

        if profitable:
            lines.append(f"\nFound {len(profitable)} profitable patterns:\n")
            lines.append(f"{'Pattern':<40} {'Trades':>8} {'Edge':>10} {'ROI':>10} {'Total Profit':>15}")
            lines.append("-" * 90)

            for p in profitable[:25]:  # Top 25
                lines.append(
                    f"{p.name:<40} {p.trades:>8,} {100*p.edge:>+9.1f}% "
                    f"{100*p.roi:>+9.1f}% ${p.total_profit:>13,.2f}"
                )
        else:
            lines.append("\nNo profitable patterns found with sufficient sample size.")

        # =====================================================================
        # KEY FINDINGS
        # =====================================================================
        lines.append("")
        lines.append("=" * 90)
        lines.append("KEY FINDINGS")
        lines.append("=" * 90)

        # Size threshold finding
        size_results = self.analyze_size_thresholds()
        threshold_profitable = [s for s in size_results if s.is_profitable and '>=' in s.name]
        if threshold_profitable:
            best = max(threshold_profitable, key=lambda s: s.roi)
            lines.append(f"\n1. SIZE THRESHOLD: {best.name} has best ROI ({100*best.roi:+.1f}%)")
        else:
            lines.append("\n1. SIZE THRESHOLD: No clear size threshold shows consistent edge")

        # Retail fade finding
        fade_data = self.analyze_retail_fade()
        if fade_data['fade_accuracy'] > 0.52:
            lines.append(f"2. RETAIL FADE: Small trader fade signal shows {100*fade_data['fade_accuracy']:.1f}% accuracy")
        else:
            lines.append(f"2. RETAIL FADE: No significant fade signal ({100*fade_data['fade_accuracy']:.1f}% accuracy)")

        # Best category
        cat_results = self.analyze_categories()
        cat_profitable = [c for c in cat_results if c.is_profitable]
        if cat_profitable:
            best_cat = max(cat_profitable, key=lambda c: c.roi)
            lines.append(f"3. BEST CATEGORY: {best_cat.name} ({100*best_cat.roi:+.1f}% ROI, {best_cat.trades} trades)")
        else:
            lines.append("3. BEST CATEGORY: No category shows consistent profitability")

        # Best time
        time_results = self.analyze_time_of_day()
        time_profitable = [t for t in time_results if t.is_profitable and 'Time' in t.name]
        if time_profitable:
            best_time = max(time_profitable, key=lambda t: t.roi)
            lines.append(f"4. BEST TIME: {best_time.name} ({100*best_time.roi:+.1f}% ROI, {best_time.trades} trades)")
        else:
            lines.append("4. BEST TIME: No time period shows significant edge")

        # Cascade finding
        cascade_data = self.analyze_cascade_patterns()
        if cascade_data.get('cascade_accuracy', 0) > 0.52:
            lines.append(f"5. CASCADES: Following cascades has {100*cascade_data['cascade_accuracy']:.1f}% accuracy")
        else:
            lines.append("5. CASCADES: No significant cascade signal")

        lines.append("")
        lines.append("=" * 90)
        lines.append("ANALYSIS COMPLETE")
        lines.append("=" * 90)

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze all trade patterns for profitable strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes ALL trades (not just whales) to find additional profitable patterns.

Examples:
    # Generate full analysis report
    python analyze_all_trades_patterns.py --enriched enriched_trades_final.csv --report

    # Output to JSON for further processing
    python analyze_all_trades_patterns.py --enriched enriched_trades_final.csv --json results.json
        """
    )

    parser.add_argument('--enriched', type=str, required=True, metavar='FILE',
                       help='CSV file with enriched trades (must have is_winner column)')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive analysis report')
    parser.add_argument('--json', type=str, metavar='FILE',
                       help='Export results to JSON')

    args = parser.parse_args()

    # Create analyzer
    analyzer = AllTradesAnalyzer()

    # Load data
    count = analyzer.load_enriched_trades(args.enriched)

    if count == 0:
        logger.error("No resolved trades found!")
        return 1

    # Generate report
    if args.report:
        report = analyzer.generate_report()
        print(report)

        # Save to file
        report_path = Path(args.enriched).parent / "all_trades_pattern_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    # Export JSON
    if args.json:
        # Collect all results
        all_patterns = (
            analyzer.analyze_size_thresholds() +
            analyzer.analyze_size_by_side() +
            analyzer.analyze_small_trades() +
            analyzer.analyze_time_of_day() +
            analyzer.analyze_day_of_week() +
            analyzer.analyze_categories() +
            analyzer.analyze_price_buckets() +
            analyzer.analyze_price_side_interaction()
        )

        json_results = {
            "generated": datetime.now().isoformat(),
            "total_trades": len(analyzer.trades),
            "total_markets": len(analyzer.trades_by_market),
            "patterns": [
                {
                    "name": p.name,
                    "description": p.description,
                    "trades": p.trades,
                    "win_rate": p.win_rate,
                    "breakeven_rate": p.breakeven_rate,
                    "edge": p.edge,
                    "roi": p.roi,
                    "total_profit": p.total_profit,
                    "is_profitable": p.is_profitable,
                }
                for p in all_patterns
                if p.trades >= 50
            ],
            "retail_fade": analyzer.analyze_retail_fade(),
            "cascade_analysis": analyzer.analyze_cascade_patterns(),
            "volume_clusters": analyzer.analyze_volume_clusters(),
        }

        with open(args.json, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"JSON results saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
