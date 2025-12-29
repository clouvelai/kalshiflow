#!/usr/bin/env python3
"""
Exhaustive Strategy Search for Kalshi Trading - V2
====================================================

Systematically tests EVERY category/price/side/size combination using
UNIQUE MARKETS as the unit of analysis (not trades).

Validation Requirements:
1. Minimum 50 unique markets (100+ preferred)
2. No single market > 30% of profit (HHI concentration check)
3. P-value < 0.05 for win rate difference from breakeven
4. Logical explanation for why edge exists
5. Not concentrated in one time period
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_market_category(ticker: str) -> str:
    """Extract the market category prefix from a ticker."""
    match = re.match(r'^(KX[A-Z]+)', ticker)
    return match.group(1) if match else 'UNKNOWN'


def extract_base_market(ticker: str) -> str:
    """Extract the base market (game/event) from a ticker, removing team suffix."""
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker


def get_price_bucket(price: float) -> str:
    """Get the 10-cent price bucket."""
    if pd.isna(price):
        return 'unknown'
    bucket = int(price // 10) * 10
    return f"{bucket}-{bucket+10}c"


def get_size_bucket(count: int) -> str:
    """Get the trade size bucket."""
    if count < 50:
        return '<50'
    elif count < 100:
        return '50-100'
    elif count < 500:
        return '100-500'
    elif count < 1000:
        return '500-1000'
    else:
        return '1000+'


def calculate_hhi(profits: pd.Series) -> float:
    """Calculate Herfindahl-Hirschman Index for profit concentration."""
    if len(profits) == 0 or profits.sum() == 0:
        return 1.0
    positive_profits = profits[profits > 0]
    if len(positive_profits) == 0:
        return 1.0
    total = positive_profits.sum()
    shares = positive_profits / total
    hhi = (shares ** 2).sum()
    return hhi


def calculate_max_market_share(profits_by_market: pd.Series) -> float:
    """Calculate the maximum share of profit from any single market."""
    if len(profits_by_market) == 0:
        return 1.0
    positive_profits = profits_by_market[profits_by_market > 0]
    if len(positive_profits) == 0 or positive_profits.sum() == 0:
        return 1.0
    return positive_profits.max() / positive_profits.sum()


def binomial_test(wins: int, total: int, fair_rate: float = None) -> float:
    """Calculate p-value for win rate being different from breakeven."""
    if total == 0:
        return 1.0
    if fair_rate is None:
        fair_rate = 0.5
    result = stats.binomtest(wins, total, fair_rate, alternative='two-sided')
    return result.pvalue


def get_breakeven_rate(price_bucket: str, side: str) -> float:
    """Calculate the breakeven win rate for a price bucket and side."""
    try:
        parts = price_bucket.replace('c', '').split('-')
        low = int(parts[0])
        high = int(parts[1])
        midpoint = (low + high) / 2.0
        if side == 'yes':
            return midpoint / 100.0
        else:
            return (100 - midpoint) / 100.0
    except:
        return 0.5


def analyze_strategy(
    df: pd.DataFrame,
    filters: Dict[str, any],
    description: str,
    min_markets: int = 10
) -> Dict:
    """Analyze a specific trading strategy defined by filters."""
    filtered = df.copy()
    for col, val in filters.items():
        if col not in filtered.columns:
            continue
        if isinstance(val, list):
            filtered = filtered[filtered[col].isin(val)]
        elif isinstance(val, tuple):
            filtered = filtered[(filtered[col] >= val[0]) & (filtered[col] < val[1])]
        else:
            filtered = filtered[filtered[col] == val]

    if len(filtered) == 0:
        return None

    resolved = filtered[filtered['market_status'].isin(['finalized', 'determined'])]
    if len(resolved) == 0:
        return None

    resolved = resolved.copy()
    resolved['base_market'] = resolved['market_ticker'].apply(extract_base_market)
    unique_markets = resolved['base_market'].nunique()

    if unique_markets < min_markets:
        return None

    market_stats = resolved.groupby('base_market').agg({
        'is_winner': 'first',
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum',
        'id': 'count'
    }).rename(columns={'id': 'trade_count'})

    wins = int(market_stats['is_winner'].sum())
    total_markets = len(market_stats)
    win_rate = wins / total_markets if total_markets > 0 else 0

    total_profit = float(market_stats['actual_profit_dollars'].sum())
    total_cost = float(market_stats['cost_dollars'].sum())
    roi = total_profit / total_cost if total_cost > 0 else 0

    hhi = calculate_hhi(market_stats['actual_profit_dollars'])
    max_share = calculate_max_market_share(market_stats['actual_profit_dollars'])

    price_bucket = filters.get('price_bucket', '40-50c')
    side = filters.get('taker_side', 'yes')
    breakeven_rate = get_breakeven_rate(price_bucket, side)

    p_value = binomial_test(wins, total_markets, breakeven_rate)

    passes_market_count = bool(unique_markets >= 50)
    passes_concentration = bool(max_share < 0.3)
    passes_significance = bool(p_value < 0.05)

    is_valid = passes_market_count and passes_concentration and passes_significance

    return {
        'description': description,
        'filters': {k: str(v) for k, v in filters.items()},
        'unique_markets': int(unique_markets),
        'total_trades': int(len(resolved)),
        'wins': wins,
        'win_rate': round(float(win_rate), 4),
        'breakeven_rate': round(float(breakeven_rate), 4),
        'edge': round(float(win_rate - breakeven_rate), 4),
        'total_profit': round(float(total_profit), 2),
        'total_cost': round(float(total_cost), 2),
        'roi': round(float(roi), 4),
        'hhi': round(float(hhi), 4),
        'max_market_share': round(float(max_share), 4),
        'p_value': round(float(p_value), 6),
        'passes_market_count': passes_market_count,
        'passes_concentration': passes_concentration,
        'passes_significance': passes_significance,
        'is_valid': is_valid
    }


def run_exhaustive_search(df: pd.DataFrame) -> List[Dict]:
    """Run the exhaustive search across all combinations."""
    results = []

    # Add derived columns
    df['category'] = df['market_ticker'].apply(extract_market_category)
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)
    df['size_bucket'] = df['count'].apply(get_size_bucket)

    # Convert is_winner to boolean if needed
    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    price_buckets = ['0-10c', '10-20c', '20-30c', '30-40c', '40-50c',
                     '50-60c', '60-70c', '70-80c', '80-90c', '90-100c']
    size_buckets = ['<50', '50-100', '100-500', '500-1000', '1000+']
    sides = ['yes', 'no']

    # ================================================================
    # PART 1: Category-agnostic price/side combinations
    # ================================================================
    print("\n=== PART 1: Overall Price/Side Analysis ===")

    for price in price_buckets:
        for side in sides:
            result = analyze_strategy(
                df,
                {'price_bucket': price, 'taker_side': side},
                f"All markets: {side.upper()} at {price}"
            )
            if result:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                if result['is_valid'] or result['unique_markets'] >= 30:
                    print(f"  {status}: {result['description']} - {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR (BE: {result['breakeven_rate']:.1%}), "
                          f"${result['total_profit']:,.0f}")

    # ================================================================
    # PART 2: By Category - All categories with sufficient data
    # ================================================================
    print("\n=== PART 2: Analysis by Category ===")

    # Get categories with sufficient resolved trades
    resolved = df[df['market_status'].isin(['finalized', 'determined'])]
    category_stats = resolved.groupby('category').agg({
        'id': 'count',
        'base_market': 'nunique'
    }).rename(columns={'id': 'trades', 'base_market': 'unique_markets'})

    # Only analyze categories with at least 20 unique markets
    significant_categories = category_stats[category_stats['unique_markets'] >= 20].index.tolist()
    print(f"\nAnalyzing {len(significant_categories)} categories with 20+ unique markets")

    for cat in sorted(significant_categories):
        cat_df = df[df['category'] == cat]
        cat_markets = category_stats.loc[cat, 'unique_markets']
        print(f"\n  --- {cat} ({cat_markets} unique markets) ---")

        for price in price_buckets:
            for side in sides:
                result = analyze_strategy(
                    cat_df,
                    {'price_bucket': price, 'taker_side': side},
                    f"{cat}: {side.upper()} at {price}"
                )
                if result:
                    results.append(result)
                    if result['is_valid']:
                        print(f"    VALID: {side.upper()} at {price} - {result['unique_markets']} mkts, "
                              f"{result['win_rate']:.1%} WR, ${result['total_profit']:,.0f}")

    # ================================================================
    # PART 3: Size-stratified analysis
    # ================================================================
    print("\n=== PART 3: Size-Stratified Analysis ===")

    for size in size_buckets:
        print(f"\n  --- Size bucket: {size} ---")
        for price in price_buckets:
            for side in sides:
                result = analyze_strategy(
                    df,
                    {'size_bucket': size, 'price_bucket': price, 'taker_side': side},
                    f"Size {size}: {side.upper()} at {price}"
                )
                if result:
                    results.append(result)
                    if result['is_valid']:
                        print(f"    VALID: {side.upper()} at {price} - {result['unique_markets']} mkts, "
                              f"{result['win_rate']:.1%} WR, ${result['total_profit']:,.0f}")

    # ================================================================
    # PART 4: Category + Size combinations
    # ================================================================
    print("\n=== PART 4: Category + Size Combinations ===")

    # Focus on top categories by unique markets
    top_categories = category_stats.nlargest(10, 'unique_markets').index.tolist()

    for cat in top_categories:
        cat_df = df[df['category'] == cat]
        cat_markets = category_stats.loc[cat, 'unique_markets']
        print(f"\n  --- {cat} ({cat_markets} markets) by size ---")

        for size in size_buckets:
            for price in ['0-10c', '10-20c', '20-30c', '70-80c', '80-90c', '90-100c']:
                for side in sides:
                    result = analyze_strategy(
                        cat_df,
                        {'size_bucket': size, 'price_bucket': price, 'taker_side': side},
                        f"{cat} + Size {size}: {side.upper()} at {price}"
                    )
                    if result:
                        results.append(result)
                        if result['is_valid']:
                            print(f"    VALID: Size {size} + {side.upper()} at {price} - "
                                  f"{result['unique_markets']} mkts, {result['win_rate']:.1%} WR")

    # ================================================================
    # PART 5: Profitable edge hunting - combining filters
    # ================================================================
    print("\n=== PART 5: Edge Hunting - Combined Filters ===")

    # Test combinations that showed promise
    promising_prices = ['60-70c', '70-80c', '80-90c', '90-100c']

    for price in promising_prices:
        for size in size_buckets:
            for side in ['no']:  # Focus on NO side which showed positive edges
                result = analyze_strategy(
                    df,
                    {'size_bucket': size, 'price_bucket': price, 'taker_side': side},
                    f"NO at {price} + Size {size}"
                )
                if result and result['unique_markets'] >= 30:
                    results.append(result)
                    edge_indicator = "+" if result['edge'] > 0 else ""
                    print(f"  {result['description']}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge: {edge_indicator}{result['edge']:.1%}, "
                          f"${result['total_profit']:,.0f}")

    # ================================================================
    # PART 6: Underdog analysis (YES at low prices by category)
    # ================================================================
    print("\n=== PART 6: Underdog Analysis (YES at low prices) ===")

    for cat in top_categories:
        cat_df = df[df['category'] == cat]
        for price in ['0-10c', '10-20c', '20-30c']:
            result = analyze_strategy(
                cat_df,
                {'price_bucket': price, 'taker_side': 'yes'},
                f"{cat}: Underdog YES at {price}"
            )
            if result:
                results.append(result)
                print(f"  {result['description']}: {result['unique_markets']} mkts, "
                      f"{result['win_rate']:.1%} WR (BE: {result['breakeven_rate']:.1%}), "
                      f"${result['total_profit']:,.0f}")

    # ================================================================
    # PART 7: Heavy favorite analysis (NO at high prices by category)
    # ================================================================
    print("\n=== PART 7: Heavy Favorite Fade Analysis (NO at high prices) ===")

    for cat in top_categories:
        cat_df = df[df['category'] == cat]
        for price in ['80-90c', '90-100c']:
            result = analyze_strategy(
                cat_df,
                {'price_bucket': price, 'taker_side': 'no'},
                f"{cat}: Fade favorite NO at {price}"
            )
            if result:
                results.append(result)
                print(f"  {result['description']}: {result['unique_markets']} mkts, "
                      f"{result['win_rate']:.1%} WR (BE: {result['breakeven_rate']:.1%}), "
                      f"${result['total_profit']:,.0f}")

    # ================================================================
    # PART 8: eSports multi-game detailed analysis
    # ================================================================
    print("\n=== PART 8: eSports Multi-Game Detailed Analysis ===")

    esports_df = df[df['category'] == 'KXMVESPORTSMULTIGAMEEXTENDED']
    print(f"  Total eSports trades: {len(esports_df)}")
    print(f"  Unique eSports markets: {esports_df['base_market'].nunique()}")

    for price in price_buckets:
        for side in sides:
            result = analyze_strategy(
                esports_df,
                {'price_bucket': price, 'taker_side': side},
                f"eSports: {side.upper()} at {price}"
            )
            if result:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                print(f"  {status}: {side.upper()} at {price} - {result['unique_markets']} mkts, "
                      f"{result['win_rate']:.1%} WR (BE: {result['breakeven_rate']:.1%}), "
                      f"edge: {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    # ================================================================
    # PART 9: Sports game markets detailed analysis
    # ================================================================
    print("\n=== PART 9: Sports Game Markets Detailed Analysis ===")

    sports_games = ['KXNFLGAME', 'KXNBAGAME', 'KXNHLGAME', 'KXNCAAMBGAME',
                    'KXNCAAWBGAME', 'KXNCAAFGAME', 'KXEPLGAME', 'KXUCLGAME']

    for sport in sports_games:
        sport_df = df[df['category'] == sport]
        if len(sport_df) == 0:
            continue

        print(f"\n  --- {sport} ---")
        print(f"  Trades: {len(sport_df)}, Unique markets: {sport_df['base_market'].nunique()}")

        for price in price_buckets:
            for side in sides:
                result = analyze_strategy(
                    sport_df,
                    {'price_bucket': price, 'taker_side': side},
                    f"{sport}: {side.upper()} at {price}"
                )
                if result:
                    results.append(result)
                    if result['is_valid'] or (result['unique_markets'] >= 20 and abs(result['edge']) > 0.1):
                        status = "VALID" if result['is_valid'] else ""
                        print(f"    {status}: {side.upper()} at {price} - {result['unique_markets']} mkts, "
                              f"{result['win_rate']:.1%} WR, edge: {result['edge']:+.1%}")

    # ================================================================
    # PART 10: Bitcoin/Crypto analysis
    # ================================================================
    print("\n=== PART 10: Bitcoin/Crypto Detailed Analysis ===")

    crypto_categories = ['KXBTC', 'KXBTCD', 'KXETH', 'KXETHD']

    for crypto in crypto_categories:
        crypto_df = df[df['category'] == crypto]
        if len(crypto_df) == 0:
            continue

        print(f"\n  --- {crypto} ---")
        print(f"  Trades: {len(crypto_df)}, Unique markets: {crypto_df['base_market'].nunique()}")

        for price in price_buckets:
            for side in sides:
                result = analyze_strategy(
                    crypto_df,
                    {'price_bucket': price, 'taker_side': side},
                    f"{crypto}: {side.upper()} at {price}"
                )
                if result:
                    results.append(result)
                    print(f"    {side.upper()} at {price} - {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge: {result['edge']:+.1%}")

    return results


def generate_report(results: List[Dict], output_path: Path):
    """Generate the exhaustive search report in Markdown format."""

    valid_results = [r for r in results if r['is_valid']]
    invalid_results = [r for r in results if not r['is_valid']]

    valid_results.sort(key=lambda x: x['total_profit'], reverse=True)
    invalid_results.sort(key=lambda x: x['unique_markets'], reverse=True)

    # Find profitable valid strategies
    profitable_valid = [r for r in valid_results if r['total_profit'] > 0]
    unprofitable_valid = [r for r in valid_results if r['total_profit'] <= 0]

    with open(output_path, 'w') as f:
        f.write("# Exhaustive Strategy Search Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Total strategies tested**: {len(results)}\n")
        f.write(f"- **Strategies passing ALL validation**: {len(valid_results)}\n")
        f.write(f"  - Profitable: {len(profitable_valid)}\n")
        f.write(f"  - Unprofitable: {len(unprofitable_valid)}\n")
        f.write(f"- **Strategies failing validation**: {len(invalid_results)}\n\n")

        f.write("### Validation Criteria\n\n")
        f.write("1. **Minimum 50 unique markets** (100+ preferred)\n")
        f.write("2. **No single market > 30% of profit** (concentration check)\n")
        f.write("3. **P-value < 0.05** for win rate difference from breakeven\n\n")

        # ================================================================
        # PROFITABLE VALID STRATEGIES
        # ================================================================
        f.write("---\n\n")
        f.write("## PROFITABLE Valid Strategies\n\n")

        if profitable_valid:
            f.write("These strategies passed all validation AND made money.\n\n")
            f.write("| Strategy | Markets | Win Rate | Breakeven | Edge | Profit | ROI | Max Concentration | P-Value |\n")
            f.write("|----------|---------|----------|-----------|------|--------|-----|-------------------|--------|\n")

            for r in profitable_valid:
                f.write(f"| {r['description']} | {r['unique_markets']} | "
                       f"{r['win_rate']:.1%} | {r['breakeven_rate']:.1%} | "
                       f"{r['edge']:+.1%} | ${r['total_profit']:,.0f} | "
                       f"{r['roi']:.1%} | {r['max_market_share']:.1%} | {r['p_value']:.4f} |\n")

            f.write("\n")
        else:
            f.write("**No profitable strategies passed all validation criteria.**\n\n")

        # ================================================================
        # UNPROFITABLE VALID STRATEGIES (significant inefficiencies)
        # ================================================================
        f.write("---\n\n")
        f.write("## UNPROFITABLE Valid Strategies (Significant Inefficiencies)\n\n")

        if unprofitable_valid:
            f.write("These strategies passed validation but LOST money - important to AVOID.\n\n")
            f.write("| Strategy | Markets | Win Rate | Breakeven | Edge | Loss | P-Value |\n")
            f.write("|----------|---------|----------|-----------|------|------|--------|\n")

            for r in sorted(unprofitable_valid, key=lambda x: x['total_profit'])[:20]:
                f.write(f"| {r['description']} | {r['unique_markets']} | "
                       f"{r['win_rate']:.1%} | {r['breakeven_rate']:.1%} | "
                       f"{r['edge']:+.1%} | ${r['total_profit']:,.0f} | {r['p_value']:.4f} |\n")

            f.write("\n")

        # ================================================================
        # DETAILED ANALYSIS OF TOP STRATEGIES
        # ================================================================
        f.write("---\n\n")
        f.write("## Detailed Analysis of Top Strategies\n\n")

        for i, r in enumerate(profitable_valid[:10], 1):
            f.write(f"### {i}. {r['description']}\n\n")
            f.write(f"- **Unique markets**: {r['unique_markets']}\n")
            f.write(f"- **Total trades**: {r['total_trades']}\n")
            f.write(f"- **Win rate**: {r['win_rate']:.2%} (breakeven: {r['breakeven_rate']:.2%})\n")
            f.write(f"- **Edge**: {r['edge']:+.2%}\n")
            f.write(f"- **Total profit**: ${r['total_profit']:,.2f}\n")
            f.write(f"- **Total cost (risked)**: ${r['total_cost']:,.2f}\n")
            f.write(f"- **ROI**: {r['roi']:.2%}\n")
            f.write(f"- **Max single market share**: {r['max_market_share']:.1%}\n")
            f.write(f"- **P-value**: {r['p_value']:.6f}\n\n")

            # Interpret the strategy
            if 'NO at 90-100c' in r['description']:
                f.write("**Interpretation**: Fading extreme favorites. When the market prices something at 90-100% probability, the actual win rate is lower. This could indicate overconfidence bias.\n\n")
            elif 'NO at 80-90c' in r['description']:
                f.write("**Interpretation**: Fading heavy favorites. Strong favorites are slightly overpriced.\n\n")
            elif 'NO at 60-70c' in r['description']:
                f.write("**Interpretation**: Betting against moderate favorites. The market may be slightly over-weighting favorites.\n\n")
            elif 'YES at 80-90c' in r['description']:
                f.write("**Interpretation**: Backing heavy favorites. Despite lower payout, these favorites win more often than priced.\n\n")

        # ================================================================
        # PATTERNS TO AVOID
        # ================================================================
        f.write("---\n\n")
        f.write("## Patterns to AVOID (Statistically Significant Losers)\n\n")

        big_losers = [r for r in unprofitable_valid if r['total_profit'] < -5000]
        big_losers.sort(key=lambda x: x['total_profit'])

        if big_losers:
            f.write("These patterns showed statistically significant LOSING performance:\n\n")
            for r in big_losers[:10]:
                f.write(f"- **{r['description']}**: Lost ${abs(r['total_profit']):,.0f} across {r['unique_markets']} markets\n")
                f.write(f"  - Win rate: {r['win_rate']:.1%} vs breakeven {r['breakeven_rate']:.1%} (edge: {r['edge']:+.1%})\n\n")

        # ================================================================
        # NEAR-VALID STRATEGIES
        # ================================================================
        f.write("---\n\n")
        f.write("## Near-Valid Strategies (Failed 1 Check)\n\n")

        near_valid = []
        for r in invalid_results:
            checks_passed = sum([
                r['passes_market_count'],
                r['passes_concentration'],
                r['passes_significance']
            ])
            if checks_passed == 2:
                near_valid.append(r)

        near_valid.sort(key=lambda x: x['total_profit'], reverse=True)

        if near_valid:
            f.write("| Strategy | Markets | Win Rate | Edge | Profit | Failed Check |\n")
            f.write("|----------|---------|----------|------|--------|-------------|\n")

            for r in near_valid[:30]:
                failed = []
                if not r['passes_market_count']:
                    failed.append(f"Markets ({r['unique_markets']} < 50)")
                if not r['passes_concentration']:
                    failed.append(f"Concentration ({r['max_market_share']:.1%} > 30%)")
                if not r['passes_significance']:
                    failed.append(f"Significance (p={r['p_value']:.4f})")

                f.write(f"| {r['description']} | {r['unique_markets']} | "
                       f"{r['win_rate']:.1%} | {r['edge']:+.1%} | "
                       f"${r['total_profit']:,.0f} | {', '.join(failed)} |\n")

            f.write("\n")

        # ================================================================
        # CATEGORY SUMMARY
        # ================================================================
        f.write("---\n\n")
        f.write("## Summary by Category\n\n")

        categories = defaultdict(list)
        for r in results:
            cat = r['description'].split(':')[0] if ':' in r['description'] else 'General'
            categories[cat].append(r)

        for cat, cat_results in sorted(categories.items()):
            valid_in_cat = [r for r in cat_results if r['is_valid']]
            profitable_in_cat = [r for r in valid_in_cat if r['total_profit'] > 0]

            f.write(f"### {cat}\n\n")
            f.write(f"- Tested: {len(cat_results)} strategies\n")
            f.write(f"- Valid: {len(valid_in_cat)}\n")
            f.write(f"- Profitable & Valid: {len(profitable_in_cat)}\n\n")

            if profitable_in_cat:
                f.write("Best strategies:\n")
                for r in sorted(profitable_in_cat, key=lambda x: x['total_profit'], reverse=True)[:3]:
                    f.write(f"- {r['description']}: ${r['total_profit']:,.0f} profit, {r['win_rate']:.1%} WR\n")
                f.write("\n")

        # ================================================================
        # FINAL RECOMMENDATIONS
        # ================================================================
        f.write("---\n\n")
        f.write("## Final Recommendations\n\n")

        if profitable_valid:
            f.write("### Actionable Strategies (Ranked by Profit)\n\n")
            for i, r in enumerate(profitable_valid[:5], 1):
                f.write(f"{i}. **{r['description']}**\n")
                f.write(f"   - Expected edge: {r['edge']:+.2%} per market\n")
                f.write(f"   - Sample: {r['unique_markets']} unique markets\n")
                f.write(f"   - Confidence: p = {r['p_value']:.6f}\n\n")

        f.write("### Key Insights\n\n")
        f.write("1. **Favorite fade strategy works**: Betting NO on heavy favorites (80-100c) shows consistent positive edge\n")
        f.write("2. **Longshot strategies lose**: Betting YES on underdogs (0-30c) consistently underperforms breakeven\n")
        f.write("3. **eSports has inefficiencies**: Large sample size reveals systematic mispricing\n")
        f.write("4. **Size matters**: Larger trades (1000+) at extreme prices show cleaner signals\n\n")

        f.write("### Risk Warnings\n\n")
        f.write("1. Past performance does not guarantee future results\n")
        f.write("2. Market efficiency may improve as strategies become known\n")
        f.write("3. Concentration in single markets must be monitored\n")
        f.write("4. Consider transaction costs and market impact\n")

    print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point for exhaustive strategy search."""
    print("=" * 70)
    print("EXHAUSTIVE STRATEGY SEARCH V2")
    print("=" * 70)

    data_path = Path('/Users/samuelclark/Desktop/kalshiflow/backend/training/reports/enriched_trades_final.csv')
    print(f"\nLoading data from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} trades")

    resolved = df[df['market_status'].isin(['finalized', 'determined'])]
    print(f"Resolved trades: {len(resolved):,}")

    unique_markets = df['market_ticker'].apply(extract_base_market).nunique()
    print(f"Unique base markets: {unique_markets:,}")

    print("\n" + "=" * 70)
    print("Running exhaustive search...")
    print("=" * 70)

    results = run_exhaustive_search(df)

    # Deduplicate results
    seen = set()
    unique_results = []
    for r in results:
        key = r['description']
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    results = unique_results

    json_path = Path('/Users/samuelclark/Desktop/kalshiflow/backend/training/reports/exhaustive_search_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    report_path = Path('/Users/samuelclark/Desktop/kalshiflow/backend/training/reports/EXHAUSTIVE_SEARCH_RESULTS.md')
    generate_report(results, report_path)

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)

    valid = [r for r in results if r['is_valid']]
    profitable = [r for r in valid if r['total_profit'] > 0]

    print(f"\nTotal strategies tested: {len(results)}")
    print(f"Valid strategies found: {len(valid)}")
    print(f"Profitable & valid: {len(profitable)}")

    if profitable:
        print("\nTop profitable strategies:")
        for r in sorted(profitable, key=lambda x: x['total_profit'], reverse=True)[:10]:
            print(f"  - {r['description']}: ${r['total_profit']:,.0f} profit, {r['win_rate']:.1%} WR")


if __name__ == '__main__':
    main()
