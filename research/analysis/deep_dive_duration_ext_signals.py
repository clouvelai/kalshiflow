"""
DEEP DIVE VALIDATION: Duration Signal + External Hypotheses
============================================================

This script performs rigorous validation of:
1. Duration Signal (H-MM150): market_duration > 24hr + RLM
2. H-EXT-001: Favorite-Longshot Bias
3. H-EXT-002: Large Trade Order Imbalance
4. H-EXT-006: Steam Move Detection

Methodology:
- Bucket-matched validation (control for price level)
- Statistical significance testing (p < 0.05)
- Temporal stability (multiple quarters positive)
- Concentration check (<30% from single market)
- RLM independence check (overlap < 70%)

Data: ~7.9M trades, ~316K markets (Dec 5, 2025 - Jan 4, 2026)

Author: Quant Agent (Claude Opus 4.5)
Date: 2026-01-05
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADE_DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/deep_dive_duration_and_ext_signals.json'

# Strategy parameters
WHALE_THRESHOLD = 10000  # $100 in cents
LARGE_TRADE_THRESHOLD = 5000  # $50 for "large" trades in imbalance calc
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]

# Validation thresholds
MIN_MARKETS = 50
SIGNIFICANCE_LEVEL = 0.05
MAX_CONCENTRATION = 0.30
MIN_BUCKET_PASS_RATE = 0.60

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare trade data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(TRADE_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD
    df['is_large_trade'] = df['trade_value_cents'] >= LARGE_TRADE_THRESHOLD
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    df['week'] = df['datetime'].dt.isocalendar().week
    df['is_yes_trade'] = df['taker_side'] == 'yes'
    df['is_no_trade'] = df['taker_side'] == 'no'

    # Filter to resolved markets
    resolved_markets = df[df['result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved_markets)]

    print(f"Loaded {len(df):,} total trades")
    print(f"Resolved markets: {len(resolved_markets):,}")
    print(f"Trades in resolved markets: {len(df_resolved):,}")
    print(f"Date range: {df_resolved['datetime'].min()} to {df_resolved['datetime'].max()}")

    return df_resolved


def build_market_stats(df):
    """Build comprehensive market-level statistics."""
    print("\n" + "=" * 80)
    print("BUILDING MARKET STATISTICS")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Base aggregations
    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean', 'std'],
        'no_price': ['mean', 'first', 'last'],
        'result': 'first',
        'count': ['size', 'sum', 'mean', 'max', 'min'],
        'datetime': ['first', 'last'],
        'is_whale': ['sum', 'any'],
        'is_large_trade': 'sum',
        'trade_value_cents': ['sum', 'mean'],
        'leverage_ratio': ['mean', 'std'],
        'is_weekend': 'mean',
        'is_round_size': 'sum',
        'hour': 'mean',
        'week': 'first'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price', 'yes_price_std',
        'avg_no_price', 'first_no_price', 'last_no_price',
        'result',
        'n_trades', 'total_contracts', 'avg_trade_size', 'max_trade_size', 'min_trade_size',
        'first_trade_time', 'last_trade_time',
        'whale_count', 'has_whale',
        'large_trade_count',
        'total_value', 'avg_trade_value',
        'avg_leverage', 'lev_std',
        'weekend_ratio', 'round_size_count', 'avg_hour', 'week'
    ]

    # Derived fields
    market_stats['yes_price_moved_down'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['yes_price_drop'] = market_stats['first_yes_price'] - market_stats['last_yes_price']
    market_stats['no_trade_ratio'] = 1 - market_stats['yes_trade_ratio']

    # Market duration in hours
    market_stats['market_duration_hours'] = (
        (market_stats['last_trade_time'] - market_stats['first_trade_time']).dt.total_seconds() / 3600
    )

    # Fill NaN
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['yes_price_std'] = market_stats['yes_price_std'].fillna(0)

    # Price buckets (5c)
    market_stats['bucket_5c'] = (market_stats['avg_no_price'] // 5) * 5
    market_stats['yes_bucket_5c'] = (market_stats['avg_yes_price'] // 5) * 5

    # Outcome
    market_stats['no_won'] = market_stats['result'] == 'no'
    market_stats['yes_won'] = market_stats['result'] == 'yes'

    # Quarter for temporal stability
    market_stats['quarter'] = pd.to_datetime(market_stats['first_trade_time']).dt.to_period('Q')

    # Category
    market_stats['category'] = market_stats['market_ticker'].apply(
        lambda x: x.split('-')[0] if '-' in x else x[:8]
    )

    print(f"Built statistics for {len(market_stats):,} markets")

    return market_stats


def compute_large_trade_imbalance(df):
    """
    Compute large trade order imbalance per market.

    Definition: (Large YES trades - Large NO trades) / Total Large trades
    Large = trades >= $50 value
    """
    print("\nComputing large trade imbalance...")

    large_trades = df[df['is_large_trade']].copy()

    imbalance = large_trades.groupby('market_ticker').apply(
        lambda x: (x['is_yes_trade'].sum() - x['is_no_trade'].sum()) / len(x) if len(x) > 0 else 0
    ).reset_index()
    imbalance.columns = ['market_ticker', 'large_trade_imbalance']

    # Also count large trades by side
    large_counts = large_trades.groupby('market_ticker').agg({
        'is_yes_trade': 'sum',
        'is_no_trade': 'sum',
        'trade_value_cents': ['count', 'sum']
    }).reset_index()
    large_counts.columns = ['market_ticker', 'large_yes_trades', 'large_no_trades',
                            'total_large_trades', 'total_large_value']

    result = imbalance.merge(large_counts, on='market_ticker', how='left')
    result = result.fillna(0)

    print(f"  Markets with large trades: {len(result[result['total_large_trades'] > 0]):,}")

    return result


def detect_steam_moves(df):
    """
    Detect steam moves in each market.

    Definition: 5+ consecutive same-direction trades with price movement >= 3c
    within a short time window (< 60 seconds between trades).
    """
    print("\nDetecting steam moves...")

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    steam_results = []

    for ticker, group in df_sorted.groupby('market_ticker'):
        if len(group) < 5:
            steam_results.append({
                'market_ticker': ticker,
                'has_steam': False,
                'steam_count': 0,
                'steam_direction': None,
                'max_steam_price_move': 0
            })
            continue

        trades = group.reset_index(drop=True)

        # Find runs of same-direction trades
        steam_count = 0
        max_price_move = 0
        last_steam_direction = None

        i = 0
        while i < len(trades) - 4:
            # Check for 5+ consecutive same-direction
            direction = trades.iloc[i]['taker_side']
            run_length = 1

            for j in range(i + 1, len(trades)):
                if trades.iloc[j]['taker_side'] == direction:
                    # Check time gap (< 60 seconds)
                    time_gap = (trades.iloc[j]['datetime'] - trades.iloc[j-1]['datetime']).total_seconds()
                    if time_gap < 60:
                        run_length += 1
                    else:
                        break
                else:
                    break

            if run_length >= 5:
                # Calculate price move during run
                start_price = trades.iloc[i]['yes_price']
                end_price = trades.iloc[i + run_length - 1]['yes_price']
                price_move = abs(end_price - start_price)

                if price_move >= 3:
                    steam_count += 1
                    max_price_move = max(max_price_move, price_move)
                    last_steam_direction = direction

            i += max(1, run_length - 1)

        steam_results.append({
            'market_ticker': ticker,
            'has_steam': steam_count > 0,
            'steam_count': steam_count,
            'steam_direction': last_steam_direction,
            'max_steam_price_move': max_price_move
        })

    steam_df = pd.DataFrame(steam_results)
    print(f"  Markets with steam moves: {steam_df['has_steam'].sum():,}")

    return steam_df


def build_baseline(market_stats):
    """Build baseline win rates at 5c price buckets for NO bets."""
    print("\n" + "=" * 80)
    print("BUILDING PRICE-BUCKET BASELINE")
    print("=" * 80)

    baseline = {}
    for bucket in sorted(market_stats['bucket_5c'].unique()):
        bucket_markets = market_stats[market_stats['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            win_rate = bucket_markets['no_won'].mean()
            breakeven = (bucket + 2.5) / 100
            baseline[bucket] = {
                'win_rate': win_rate,
                'n_markets': n,
                'breakeven': breakeven,
                'expected_edge': win_rate - breakeven
            }

    print(f"Built baseline across {len(baseline)} price buckets")
    print(f"Total baseline markets: {sum(b['n_markets'] for b in baseline.values()):,}")

    return baseline


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

def validate_strategy(signal_markets, baseline, market_stats, strategy_name, bet_direction='no'):
    """
    Comprehensive strategy validation with all criteria.

    Returns dict with:
    - raw_edge: win_rate - breakeven
    - bucket_improvement: weighted improvement over baseline
    - p_value: statistical significance
    - bucket_analysis: per-bucket breakdown
    - temporal_stability: quarters positive
    - concentration: max single market profit contribution
    - is_validated: passes all criteria
    """
    print(f"\n{'-'*60}")
    print(f"VALIDATING: {strategy_name}")
    print(f"{'-'*60}")

    n_markets = len(signal_markets)
    print(f"Signal markets: {n_markets:,}")

    if n_markets < MIN_MARKETS:
        print(f"  FAIL: Insufficient markets ({n_markets} < {MIN_MARKETS})")
        return {
            'strategy_name': strategy_name,
            'n_markets': n_markets,
            'status': 'INSUFFICIENT_SAMPLE',
            'is_validated': False
        }

    # Raw win rate
    if bet_direction == 'no':
        win_rate = signal_markets['no_won'].mean()
        avg_price = signal_markets['avg_no_price'].mean()
        breakeven = avg_price / 100
    else:
        win_rate = signal_markets['yes_won'].mean()
        avg_price = signal_markets['avg_yes_price'].mean()
        breakeven = avg_price / 100

    raw_edge = win_rate - breakeven
    print(f"Win rate: {win_rate:.1%}")
    print(f"Avg price: {avg_price:.1f}c")
    print(f"Breakeven: {breakeven:.1%}")
    print(f"Raw edge: {raw_edge:+.1%}")

    # Bucket-matched analysis
    bucket_col = 'bucket_5c' if bet_direction == 'no' else 'yes_bucket_5c'
    bucket_results = []
    positive_buckets = 0
    total_buckets = 0
    weighted_improvements = []

    for bucket in sorted(signal_markets[bucket_col].unique()):
        bucket_markets = signal_markets[signal_markets[bucket_col] == bucket]
        n_bucket = len(bucket_markets)

        if n_bucket < 5:
            continue

        if bucket not in baseline:
            continue

        total_buckets += 1

        if bet_direction == 'no':
            bucket_win_rate = bucket_markets['no_won'].mean()
        else:
            bucket_win_rate = bucket_markets['yes_won'].mean()

        baseline_win_rate = baseline[bucket]['win_rate']
        improvement = bucket_win_rate - baseline_win_rate

        if improvement > 0:
            positive_buckets += 1

        bucket_results.append({
            'bucket': bucket,
            'n_markets': n_bucket,
            'win_rate': bucket_win_rate,
            'baseline_win_rate': baseline_win_rate,
            'improvement': improvement
        })

        weighted_improvements.append((improvement, n_bucket))

    # Calculate weighted average improvement
    if weighted_improvements:
        total_weight = sum(w[1] for w in weighted_improvements)
        weighted_improvement = sum(w[0] * w[1] for w in weighted_improvements) / total_weight
    else:
        weighted_improvement = 0

    bucket_pass_rate = positive_buckets / total_buckets if total_buckets > 0 else 0

    print(f"\nBucket analysis:")
    print(f"  Positive buckets: {positive_buckets}/{total_buckets} ({bucket_pass_rate:.0%})")
    print(f"  Weighted improvement: {weighted_improvement:+.1%}")

    # Statistical significance (binomial test)
    if bet_direction == 'no':
        successes = signal_markets['no_won'].sum()
    else:
        successes = signal_markets['yes_won'].sum()

    # One-sided binomial test: is win rate significantly > breakeven?
    p_value = 1 - stats.binom.cdf(successes - 1, n_markets, breakeven)

    # Also calculate z-score for confidence interval
    se = np.sqrt(win_rate * (1 - win_rate) / n_markets)
    z_score = (win_rate - breakeven) / se if se > 0 else 0
    ci_low = win_rate - 1.96 * se
    ci_high = win_rate + 1.96 * se

    print(f"\nStatistical significance:")
    print(f"  p-value: {p_value:.6f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # Temporal stability
    quarter_results = signal_markets.groupby('quarter').apply(
        lambda x: x['no_won'].mean() if bet_direction == 'no' else x['yes_won'].mean()
    )
    quarters_positive = (quarter_results > breakeven).sum()
    total_quarters = len(quarter_results)

    print(f"\nTemporal stability:")
    print(f"  Quarters positive: {quarters_positive}/{total_quarters}")
    for q, wr in quarter_results.items():
        status = "+" if wr > breakeven else "-"
        print(f"    {q}: {wr:.1%} [{status}]")

    # Concentration check
    if bet_direction == 'no':
        signal_markets = signal_markets.copy()
        signal_markets['profit'] = signal_markets['no_won'].apply(lambda x: 1 if x else -1) * signal_markets['avg_no_price'] / 100
    else:
        signal_markets = signal_markets.copy()
        signal_markets['profit'] = signal_markets['yes_won'].apply(lambda x: 1 if x else -1) * signal_markets['avg_yes_price'] / 100

    total_profit = signal_markets['profit'].sum()
    if total_profit > 0:
        max_profit_contribution = signal_markets[signal_markets['profit'] > 0]['profit'].max() / total_profit
    else:
        max_profit_contribution = 0

    print(f"\nConcentration:")
    print(f"  Max single market: {max_profit_contribution:.1%}")

    # Category breakdown
    print(f"\nTop categories:")
    category_stats = signal_markets.groupby('category').agg({
        'no_won': 'mean' if bet_direction == 'no' else 'count',
        'yes_won': 'mean' if bet_direction == 'yes' else 'count',
        'market_ticker': 'count'
    }).reset_index()
    category_stats.columns = ['category', 'win_rate', 'alt_wr', 'count']
    category_stats['win_rate'] = signal_markets.groupby('category')['no_won' if bet_direction == 'no' else 'yes_won'].mean().values
    category_stats = category_stats.sort_values('count', ascending=False).head(10)
    for _, row in category_stats.iterrows():
        print(f"    {row['category']}: {row['count']} markets, {row['win_rate']:.1%} win rate")

    # Validation criteria
    criteria = {
        'sample_size': n_markets >= MIN_MARKETS,
        'significance': p_value < SIGNIFICANCE_LEVEL,
        'not_price_proxy': bucket_pass_rate >= MIN_BUCKET_PASS_RATE,
        'concentration': max_profit_contribution < MAX_CONCENTRATION,
        'temporal_stable': quarters_positive >= total_quarters * 0.5,
        'positive_edge': raw_edge > 0.05
    }

    is_validated = all(criteria.values())

    print(f"\nValidation criteria:")
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    status = "VALIDATED" if is_validated else "REJECTED"
    print(f"\n>>> STATUS: {status} <<<")

    return {
        'strategy_name': strategy_name,
        'n_markets': n_markets,
        'win_rate': float(win_rate),
        'avg_price': float(avg_price),
        'breakeven': float(breakeven),
        'raw_edge': float(raw_edge),
        'bucket_improvement': float(weighted_improvement),
        'positive_buckets': positive_buckets,
        'total_buckets': total_buckets,
        'bucket_pass_rate': float(bucket_pass_rate),
        'p_value': float(p_value),
        'z_score': float(z_score),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'quarters_positive': int(quarters_positive),
        'total_quarters': int(total_quarters),
        'max_concentration': float(max_profit_contribution),
        'criteria': criteria,
        'is_validated': is_validated,
        'status': status,
        'bucket_results': bucket_results,
        'category_breakdown': category_stats[['category', 'count', 'win_rate']].to_dict('records')
    }


def check_rlm_independence(signal_markets, rlm_markets):
    """Check overlap with RLM strategy."""
    signal_tickers = set(signal_markets['market_ticker'])
    rlm_tickers = set(rlm_markets['market_ticker'])

    overlap = len(signal_tickers & rlm_tickers)
    overlap_pct = overlap / len(signal_tickers) if len(signal_tickers) > 0 else 0

    return {
        'signal_count': len(signal_tickers),
        'rlm_count': len(rlm_tickers),
        'overlap': overlap,
        'overlap_pct': overlap_pct,
        'is_independent': overlap_pct < 0.70
    }


# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

def get_rlm_markets(market_stats, yes_threshold=0.65, min_trades=15):
    """Identify base RLM markets."""
    return market_stats[
        (market_stats['yes_trade_ratio'] > yes_threshold) &
        (market_stats['n_trades'] >= min_trades) &
        (market_stats['yes_price_moved_down'])
    ]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("DEEP DIVE VALIDATION: Duration + External Signals")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    df = load_data()
    market_stats = build_market_stats(df)
    baseline = build_baseline(market_stats)

    # Compute additional signals
    large_trade_imbalance = compute_large_trade_imbalance(df)
    steam_moves = detect_steam_moves(df)

    # Merge additional signals
    market_stats = market_stats.merge(large_trade_imbalance, on='market_ticker', how='left')
    market_stats = market_stats.merge(steam_moves, on='market_ticker', how='left')
    market_stats = market_stats.fillna({'large_trade_imbalance': 0, 'total_large_trades': 0,
                                        'has_steam': False, 'steam_count': 0})

    # Get base RLM markets for independence check
    rlm_markets = get_rlm_markets(market_stats)
    print(f"\nBase RLM markets: {len(rlm_markets):,}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'total_markets': len(market_stats),
        'rlm_markets': len(rlm_markets),
        'strategies': {}
    }

    # ==========================================================================
    # SIGNAL 1: Duration > 24hr (H-MM150)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL 1: DURATION > 24 HOURS (H-MM150)")
    print("=" * 80)

    # Duration only
    duration_markets = market_stats[market_stats['market_duration_hours'] > 24]
    print(f"\nDuration > 24hr markets: {len(duration_markets):,}")

    results['strategies']['H-MM150_duration_only'] = validate_strategy(
        duration_markets, baseline, market_stats,
        "Duration > 24hr (NO bet)", 'no'
    )
    results['strategies']['H-MM150_duration_only']['rlm_independence'] = check_rlm_independence(
        duration_markets, rlm_markets
    )

    # Duration + RLM combined
    duration_rlm = market_stats[
        (market_stats['market_duration_hours'] > 24) &
        (market_stats['yes_trade_ratio'] > 0.65) &
        (market_stats['n_trades'] >= 15) &
        (market_stats['yes_price_moved_down'])
    ]
    print(f"\nDuration > 24hr + RLM markets: {len(duration_rlm):,}")

    results['strategies']['H-MM150_duration_rlm'] = validate_strategy(
        duration_rlm, baseline, market_stats,
        "Duration > 24hr + RLM (NO bet)", 'no'
    )

    # Duration tier analysis
    print("\n" + "-" * 40)
    print("DURATION TIER ANALYSIS")
    print("-" * 40)

    duration_tiers = [
        ('<1hr', market_stats['market_duration_hours'] < 1),
        ('1-6hr', (market_stats['market_duration_hours'] >= 1) & (market_stats['market_duration_hours'] < 6)),
        ('6-24hr', (market_stats['market_duration_hours'] >= 6) & (market_stats['market_duration_hours'] < 24)),
        ('24-72hr', (market_stats['market_duration_hours'] >= 24) & (market_stats['market_duration_hours'] < 72)),
        ('72hr+', market_stats['market_duration_hours'] >= 72)
    ]

    duration_tier_results = []
    for tier_name, tier_mask in duration_tiers:
        tier_markets = market_stats[tier_mask]
        if len(tier_markets) >= 20:
            win_rate = tier_markets['no_won'].mean()
            avg_no = tier_markets['avg_no_price'].mean()
            breakeven = avg_no / 100
            edge = win_rate - breakeven

            # RLM overlap
            tier_rlm = tier_markets[
                (tier_markets['yes_trade_ratio'] > 0.65) &
                (tier_markets['n_trades'] >= 15) &
                (tier_markets['yes_price_moved_down'])
            ]
            rlm_win_rate = tier_rlm['no_won'].mean() if len(tier_rlm) > 0 else 0
            rlm_edge = rlm_win_rate - (tier_rlm['avg_no_price'].mean() / 100) if len(tier_rlm) > 0 else 0

            tier_result = {
                'tier': tier_name,
                'n_markets': len(tier_markets),
                'win_rate': float(win_rate),
                'avg_no_price': float(avg_no),
                'edge': float(edge),
                'rlm_markets': len(tier_rlm),
                'rlm_win_rate': float(rlm_win_rate) if len(tier_rlm) > 0 else None,
                'rlm_edge': float(rlm_edge) if len(tier_rlm) > 0 else None
            }
            duration_tier_results.append(tier_result)

            print(f"  {tier_name}: {len(tier_markets):,} markets, {win_rate:.1%} win rate, {edge:+.1%} edge")
            if len(tier_rlm) > 0:
                print(f"    + RLM: {len(tier_rlm):,} markets, {rlm_win_rate:.1%} win rate, {rlm_edge:+.1%} edge")

    results['duration_tier_analysis'] = duration_tier_results

    # ==========================================================================
    # SIGNAL 2: Favorite-Longshot Bias (H-EXT-001)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL 2: FAVORITE-LONGSHOT BIAS (H-EXT-001)")
    print("=" * 80)

    # Test if high-price contracts (favorites) have positive edge
    print("\nPrice tier analysis (betting YES on favorites):")

    price_tier_results = []
    for low, high in [(5, 20), (20, 40), (40, 60), (60, 80), (80, 95)]:
        tier_markets = market_stats[
            (market_stats['avg_yes_price'] >= low) &
            (market_stats['avg_yes_price'] < high)
        ]
        if len(tier_markets) >= 50:
            win_rate = tier_markets['yes_won'].mean()
            implied_prob = (low + high) / 2 / 100
            edge = win_rate - implied_prob

            tier_result = {
                'tier': f'{low}-{high}c',
                'n_markets': len(tier_markets),
                'win_rate': float(win_rate),
                'implied_prob': float(implied_prob),
                'edge': float(edge)
            }
            price_tier_results.append(tier_result)
            print(f"  {low}-{high}c: {len(tier_markets):,} markets, {win_rate:.1%} actual vs {implied_prob:.1%} implied = {edge:+.1%} edge")

    results['favorite_longshot_analysis'] = price_tier_results

    # Specific test: YES at 80-95c (high favorites)
    favorites = market_stats[market_stats['avg_yes_price'] >= 80]
    results['strategies']['H-EXT-001_favorites_yes'] = validate_strategy(
        favorites, baseline, market_stats,
        "Favorites (YES >= 80c) - bet YES", 'yes'
    )
    results['strategies']['H-EXT-001_favorites_yes']['rlm_independence'] = check_rlm_independence(
        favorites, rlm_markets
    )

    # Also test: NO at extreme low prices (longshots)
    longshots = market_stats[market_stats['avg_yes_price'] <= 20]
    if len(longshots) >= MIN_MARKETS:
        results['strategies']['H-EXT-001_longshots_no'] = validate_strategy(
            longshots, baseline, market_stats,
            "Longshots (YES <= 20c) - bet NO", 'no'
        )
        results['strategies']['H-EXT-001_longshots_no']['rlm_independence'] = check_rlm_independence(
            longshots, rlm_markets
        )

    # ==========================================================================
    # SIGNAL 3: Large Trade Order Imbalance (H-EXT-002)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL 3: LARGE TRADE ORDER IMBALANCE (H-EXT-002)")
    print("=" * 80)

    # Markets with large trade imbalance toward NO (negative imbalance)
    has_large_trades = market_stats[market_stats['total_large_trades'] >= 3]
    print(f"\nMarkets with >= 3 large trades: {len(has_large_trades):,}")

    # Imbalance distribution
    print(f"Large trade imbalance distribution:")
    print(f"  Min: {has_large_trades['large_trade_imbalance'].min():.2f}")
    print(f"  25%: {has_large_trades['large_trade_imbalance'].quantile(0.25):.2f}")
    print(f"  50%: {has_large_trades['large_trade_imbalance'].median():.2f}")
    print(f"  75%: {has_large_trades['large_trade_imbalance'].quantile(0.75):.2f}")
    print(f"  Max: {has_large_trades['large_trade_imbalance'].max():.2f}")

    # Test: Strongly negative imbalance (large money on NO)
    no_imbalance = has_large_trades[has_large_trades['large_trade_imbalance'] <= -0.3]
    if len(no_imbalance) >= MIN_MARKETS:
        results['strategies']['H-EXT-002_large_imbalance_no'] = validate_strategy(
            no_imbalance, baseline, market_stats,
            "Large trade imbalance <= -0.3 (bet NO)", 'no'
        )
        results['strategies']['H-EXT-002_large_imbalance_no']['rlm_independence'] = check_rlm_independence(
            no_imbalance, rlm_markets
        )
    else:
        print(f"\n  WARNING: Only {len(no_imbalance)} markets with strong NO imbalance")

    # Test: Strongly positive imbalance (large money on YES)
    yes_imbalance = has_large_trades[has_large_trades['large_trade_imbalance'] >= 0.3]
    if len(yes_imbalance) >= MIN_MARKETS:
        results['strategies']['H-EXT-002_large_imbalance_yes'] = validate_strategy(
            yes_imbalance, baseline, market_stats,
            "Large trade imbalance >= 0.3 (bet YES)", 'yes'
        )
        results['strategies']['H-EXT-002_large_imbalance_yes']['rlm_independence'] = check_rlm_independence(
            yes_imbalance, rlm_markets
        )

    # Imbalance tier analysis
    imbalance_tier_results = []
    for low, high, name in [(-1, -0.5, 'Strong NO'), (-0.5, -0.2, 'Moderate NO'),
                             (-0.2, 0.2, 'Balanced'), (0.2, 0.5, 'Moderate YES'),
                             (0.5, 1, 'Strong YES')]:
        tier = has_large_trades[
            (has_large_trades['large_trade_imbalance'] > low) &
            (has_large_trades['large_trade_imbalance'] <= high)
        ]
        if len(tier) >= 20:
            no_wr = tier['no_won'].mean()
            yes_wr = tier['yes_won'].mean()
            avg_no = tier['avg_no_price'].mean()
            no_edge = no_wr - avg_no / 100

            imbalance_tier_results.append({
                'tier': name,
                'n_markets': len(tier),
                'no_win_rate': float(no_wr),
                'yes_win_rate': float(yes_wr),
                'no_edge': float(no_edge)
            })
            print(f"  {name}: {len(tier):,} markets, NO {no_wr:.1%}, YES {yes_wr:.1%}")

    results['imbalance_tier_analysis'] = imbalance_tier_results

    # ==========================================================================
    # SIGNAL 4: Steam Move Detection (H-EXT-006)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL 4: STEAM MOVE DETECTION (H-EXT-006)")
    print("=" * 80)

    steam_markets = market_stats[market_stats['has_steam']]
    print(f"\nMarkets with steam moves: {len(steam_markets):,}")

    if len(steam_markets) >= MIN_MARKETS:
        # Steam following: bet in direction of steam
        steam_yes = steam_markets[steam_markets['steam_direction'] == 'yes']
        steam_no = steam_markets[steam_markets['steam_direction'] == 'no']

        print(f"  Steam YES direction: {len(steam_yes):,}")
        print(f"  Steam NO direction: {len(steam_no):,}")

        if len(steam_yes) >= MIN_MARKETS:
            results['strategies']['H-EXT-006_follow_steam_yes'] = validate_strategy(
                steam_yes, baseline, market_stats,
                "Follow YES steam (bet YES)", 'yes'
            )
            results['strategies']['H-EXT-006_follow_steam_yes']['rlm_independence'] = check_rlm_independence(
                steam_yes, rlm_markets
            )

        if len(steam_no) >= MIN_MARKETS:
            results['strategies']['H-EXT-006_follow_steam_no'] = validate_strategy(
                steam_no, baseline, market_stats,
                "Follow NO steam (bet NO)", 'no'
            )
            results['strategies']['H-EXT-006_follow_steam_no']['rlm_independence'] = check_rlm_independence(
                steam_no, rlm_markets
            )

        # Steam fading: bet OPPOSITE of steam
        if len(steam_yes) >= MIN_MARKETS:
            results['strategies']['H-EXT-006_fade_steam_yes'] = validate_strategy(
                steam_yes, baseline, market_stats,
                "Fade YES steam (bet NO)", 'no'
            )

        if len(steam_no) >= MIN_MARKETS:
            results['strategies']['H-EXT-006_fade_steam_no'] = validate_strategy(
                steam_no, baseline, market_stats,
                "Fade NO steam (bet YES)", 'yes'
            )
    else:
        print(f"  WARNING: Only {len(steam_markets)} markets with steam moves - insufficient for validation")
        results['strategies']['H-EXT-006_steam'] = {
            'status': 'INSUFFICIENT_SAMPLE',
            'n_markets': len(steam_markets)
        }

    # ==========================================================================
    # COMBINED SIGNALS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMBINED SIGNALS")
    print("=" * 80)

    # Duration + RLM + Large NO imbalance
    combined_1 = market_stats[
        (market_stats['market_duration_hours'] > 24) &
        (market_stats['yes_trade_ratio'] > 0.65) &
        (market_stats['yes_price_moved_down']) &
        (market_stats['total_large_trades'] >= 3) &
        (market_stats['large_trade_imbalance'] <= -0.2)
    ]
    if len(combined_1) >= MIN_MARKETS:
        results['strategies']['COMBINED_duration_rlm_imbalance'] = validate_strategy(
            combined_1, baseline, market_stats,
            "Duration + RLM + Large NO imbalance", 'no'
        )
    else:
        print(f"\n  Combined (Duration + RLM + Imbalance): only {len(combined_1)} markets")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    validated = []
    rejected = []

    for name, result in results['strategies'].items():
        if result.get('is_validated', False):
            validated.append((name, result))
        else:
            rejected.append((name, result))

    print(f"\nVALIDATED STRATEGIES ({len(validated)}):")
    for name, result in validated:
        print(f"  {name}")
        print(f"    Edge: {result['raw_edge']:+.1%}, Improvement: {result.get('bucket_improvement', 0):+.1%}")
        print(f"    Markets: {result['n_markets']:,}, p={result['p_value']:.4f}")

    print(f"\nREJECTED STRATEGIES ({len(rejected)}):")
    for name, result in rejected:
        reason = result.get('status', 'Unknown')
        if reason != 'INSUFFICIENT_SAMPLE':
            failed_criteria = [k for k, v in result.get('criteria', {}).items() if not v]
            reason = ', '.join(failed_criteria) if failed_criteria else reason
        print(f"  {name}: {reason}")

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = main()
