"""
FULL VALIDATION: 6-24hr Duration Window Hypothesis (H-MM151)
=============================================================

This script performs RIGOROUS (NORMAL MODE) validation of the hypothesis that
markets open 6-24 hours show optimal edge with RLM signal.

Background:
- LSD session found 6-24hr window shows +24.2% RLM edge vs +20.8% for 24hr+
- Theory: Markets need time for price discovery (>6hr) but edge decays with
  staleness (>24hr). The "mature but fresh" window is optimal.

Validation Requirements:
1. Duration Tier Breakdown (7 tiers with RLM)
2. Category Breakdown (Sports, Politics, Crypto, Economics)
3. Temporal Stability (multiple time periods)
4. Fine-grained Window Search (4-8hr, 6-10hr, etc.)
5. Signal Interactions (RLM + whale + imbalance)
6. Risk Analysis (concentration, win rate by bucket)

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
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/duration_window_full_validation.json'

# Signal thresholds
RLM_YES_RATIO = 0.65
RLM_MIN_TRADES = 15
WHALE_THRESHOLD = 10000  # $100 in cents
LARGE_TRADE_THRESHOLD = 5000  # $50 for imbalance

# Validation thresholds
MIN_MARKETS = 50
SIGNIFICANCE_LEVEL = 0.05
MAX_CONCENTRATION = 0.30
MIN_BUCKET_PASS_RATE = 0.60

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data():
    """Load and prepare trade data with all necessary features."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(TRADE_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD
    df['is_large_trade'] = df['trade_value_cents'] >= LARGE_TRADE_THRESHOLD
    df['is_yes_trade'] = df['taker_side'] == 'yes'
    df['is_no_trade'] = df['taker_side'] == 'no'

    # Filter to resolved markets
    resolved = df[df['result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved)]

    print(f"Total trades: {len(df):,}")
    print(f"Resolved markets: {len(resolved):,}")
    print(f"Trades in resolved: {len(df_resolved):,}")
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
        'count': ['size', 'sum', 'mean', 'max'],
        'datetime': ['first', 'last'],
        'is_whale': ['sum', 'any'],
        'is_large_trade': 'sum',
        'trade_value_cents': ['sum', 'mean'],
        'leverage_ratio': ['mean', 'std'],
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price', 'yes_price_std',
        'avg_no_price', 'first_no_price', 'last_no_price',
        'result',
        'n_trades', 'total_contracts', 'avg_trade_size', 'max_trade_size',
        'first_trade_time', 'last_trade_time',
        'whale_count', 'has_whale',
        'large_trade_count',
        'total_value', 'avg_trade_value',
        'avg_leverage', 'lev_std',
    ]

    # Derived fields
    market_stats['yes_price_moved_down'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['price_move'] = market_stats['last_yes_price'] - market_stats['first_yes_price']
    market_stats['no_trade_ratio'] = 1 - market_stats['yes_trade_ratio']

    # Market duration in hours
    market_stats['market_duration_hours'] = (
        (market_stats['last_trade_time'] - market_stats['first_trade_time']).dt.total_seconds() / 3600
    )

    # Fill NaN
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['yes_price_std'] = market_stats['yes_price_std'].fillna(0)

    # Price buckets
    market_stats['bucket_5c'] = (market_stats['avg_no_price'] // 5) * 5

    # Outcome
    market_stats['no_won'] = market_stats['result'] == 'no'
    market_stats['yes_won'] = market_stats['result'] == 'yes'

    # Temporal fields
    market_stats['quarter'] = pd.to_datetime(market_stats['first_trade_time']).dt.to_period('Q')
    market_stats['month'] = pd.to_datetime(market_stats['first_trade_time']).dt.to_period('M')
    market_stats['year_half'] = market_stats['quarter'].apply(
        lambda x: f"{x.year}-H1" if x.quarter <= 2 else f"{x.year}-H2"
    )

    # Category extraction
    market_stats['category'] = market_stats['market_ticker'].apply(
        lambda x: x.split('-')[0] if '-' in x else x[:8]
    )

    # Broad category grouping
    def get_broad_category(cat):
        cat_upper = cat.upper()
        if any(s in cat_upper for s in ['NFL', 'NBA', 'NHL', 'NCAA', 'SOCCER', 'EPL', 'SPORTS', 'ESPORTS']):
            return 'Sports'
        elif any(s in cat_upper for s in ['BTC', 'ETH', 'CRYPTO', 'SOL']):
            return 'Crypto'
        elif any(s in cat_upper for s in ['TRUMP', 'BIDEN', 'ELECTION', 'CONGRESS', 'SENATE', 'PRESIDENT']):
            return 'Politics'
        elif any(s in cat_upper for s in ['GDP', 'CPI', 'FED', 'FOMC', 'JOBS', 'ECON', 'INFLATION']):
            return 'Economics'
        else:
            return 'Other'

    market_stats['broad_category'] = market_stats['category'].apply(get_broad_category)

    print(f"Built statistics for {len(market_stats):,} markets")
    print(f"\nBroad category distribution:")
    for cat, count in market_stats['broad_category'].value_counts().items():
        print(f"  {cat}: {count:,}")

    return market_stats


def compute_large_trade_imbalance(df, market_stats):
    """Compute large trade order imbalance per market."""
    print("\nComputing large trade imbalance...")

    large_trades = df[df['is_large_trade']].copy()

    imbalance = large_trades.groupby('market_ticker').apply(
        lambda x: (x['is_yes_trade'].sum() - x['is_no_trade'].sum()) / len(x) if len(x) > 0 else 0
    ).reset_index()
    imbalance.columns = ['market_ticker', 'large_trade_imbalance']

    large_counts = large_trades.groupby('market_ticker').agg({
        'is_yes_trade': 'sum',
        'is_no_trade': 'sum',
        'trade_value_cents': 'count'
    }).reset_index()
    large_counts.columns = ['market_ticker', 'large_yes_trades', 'large_no_trades', 'total_large_trades']

    result = imbalance.merge(large_counts, on='market_ticker', how='left')
    result = result.fillna(0)

    market_stats = market_stats.merge(result, on='market_ticker', how='left')
    market_stats = market_stats.fillna({'large_trade_imbalance': 0, 'total_large_trades': 0})

    return market_stats


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
    return baseline


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

def validate_strategy(signal_markets, baseline, strategy_name, bet_direction='no'):
    """Comprehensive strategy validation."""
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

    # Win rate and edge
    if bet_direction == 'no':
        win_rate = signal_markets['no_won'].mean()
        avg_price = signal_markets['avg_no_price'].mean()
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
    bucket_col = 'bucket_5c'
    bucket_results = []
    positive_buckets = 0
    total_buckets = 0
    weighted_improvements = []

    for bucket in sorted(signal_markets[bucket_col].unique()):
        bucket_markets = signal_markets[signal_markets[bucket_col] == bucket]
        n_bucket = len(bucket_markets)

        if n_bucket < 5 or bucket not in baseline:
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
            'win_rate': float(bucket_win_rate),
            'baseline_win_rate': float(baseline_win_rate),
            'improvement': float(improvement)
        })

        weighted_improvements.append((improvement, n_bucket))

    # Weighted average improvement
    if weighted_improvements:
        total_weight = sum(w[1] for w in weighted_improvements)
        weighted_improvement = sum(w[0] * w[1] for w in weighted_improvements) / total_weight
    else:
        weighted_improvement = 0

    bucket_pass_rate = positive_buckets / total_buckets if total_buckets > 0 else 0

    print(f"\nBucket analysis:")
    print(f"  Positive buckets: {positive_buckets}/{total_buckets} ({bucket_pass_rate:.0%})")
    print(f"  Weighted improvement: {weighted_improvement:+.1%}")

    # Statistical significance
    if bet_direction == 'no':
        successes = signal_markets['no_won'].sum()
    else:
        successes = signal_markets['yes_won'].sum()

    p_value = 1 - stats.binom.cdf(successes - 1, n_markets, breakeven)

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

    # Concentration check
    signal_markets = signal_markets.copy()
    if bet_direction == 'no':
        signal_markets['profit'] = signal_markets['no_won'].apply(lambda x: 1 if x else -1) * signal_markets['avg_no_price'] / 100
    else:
        signal_markets['profit'] = signal_markets['yes_won'].apply(lambda x: 1 if x else -1) * signal_markets['avg_yes_price'] / 100

    total_profit = signal_markets['profit'].sum()
    if total_profit > 0:
        max_concentration = signal_markets[signal_markets['profit'] > 0]['profit'].max() / total_profit
    else:
        max_concentration = 0

    print(f"\nConcentration: {max_concentration:.1%}")

    # Validation criteria
    criteria = {
        'sample_size': n_markets >= MIN_MARKETS,
        'significance': p_value < SIGNIFICANCE_LEVEL,
        'not_price_proxy': bucket_pass_rate >= MIN_BUCKET_PASS_RATE,
        'concentration': max_concentration < MAX_CONCENTRATION,
        'temporal_stable': quarters_positive >= total_quarters * 0.5 if total_quarters > 0 else True,
        'positive_edge': raw_edge > 0.05
    }

    is_validated = all(criteria.values())
    status = "VALIDATED" if is_validated else "REJECTED"

    print(f"\nValidation criteria:")
    for name, passed in criteria.items():
        s = "PASS" if passed else "FAIL"
        print(f"  {name}: {s}")

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
        'max_concentration': float(max_concentration),
        'criteria': criteria,
        'is_validated': is_validated,
        'status': status,
        'bucket_results': bucket_results
    }


# =============================================================================
# DURATION TIER ANALYSIS
# =============================================================================

def analyze_duration_tiers(market_stats, baseline):
    """Analyze performance across duration tiers."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: DURATION TIER BREAKDOWN")
    print("=" * 80)

    # Define 7 tiers as requested
    tiers = [
        ('Tier 0: <1hr', (0, 1)),
        ('Tier 1: 1-6hr', (1, 6)),
        ('Tier 2: 6-12hr', (6, 12)),
        ('Tier 3: 12-24hr', (12, 24)),
        ('Tier 4: 24-48hr', (24, 48)),
        ('Tier 5: 48-72hr', (48, 72)),
        ('Tier 6: 72hr+', (72, float('inf')))
    ]

    tier_results = []

    for tier_name, (low, high) in tiers:
        tier_mask = (market_stats['market_duration_hours'] >= low) & (market_stats['market_duration_hours'] < high)
        tier_markets = market_stats[tier_mask]

        if len(tier_markets) < 20:
            print(f"\n{tier_name}: Insufficient markets ({len(tier_markets)})")
            continue

        # All markets in tier
        all_win_rate = tier_markets['no_won'].mean()
        all_avg_price = tier_markets['avg_no_price'].mean()
        all_breakeven = all_avg_price / 100
        all_edge = all_win_rate - all_breakeven

        # RLM markets in tier
        rlm_mask = (
            (tier_markets['yes_trade_ratio'] > RLM_YES_RATIO) &
            (tier_markets['n_trades'] >= RLM_MIN_TRADES) &
            (tier_markets['yes_price_moved_down'])
        )
        rlm_markets = tier_markets[rlm_mask]

        rlm_n = len(rlm_markets)
        if rlm_n >= 10:
            rlm_win_rate = rlm_markets['no_won'].mean()
            rlm_avg_price = rlm_markets['avg_no_price'].mean()
            rlm_breakeven = rlm_avg_price / 100
            rlm_edge = rlm_win_rate - rlm_breakeven

            # Bucket-matched improvement for RLM
            bucket_improvements = []
            positive_buckets = 0
            total_buckets = 0
            for bucket in rlm_markets['bucket_5c'].unique():
                bm = rlm_markets[rlm_markets['bucket_5c'] == bucket]
                if len(bm) >= 5 and bucket in baseline:
                    total_buckets += 1
                    imp = bm['no_won'].mean() - baseline[bucket]['win_rate']
                    bucket_improvements.append((imp, len(bm)))
                    if imp > 0:
                        positive_buckets += 1

            if bucket_improvements:
                total_w = sum(w[1] for w in bucket_improvements)
                rlm_bucket_improvement = sum(w[0] * w[1] for w in bucket_improvements) / total_w
                rlm_bucket_pass_rate = positive_buckets / total_buckets
            else:
                rlm_bucket_improvement = 0
                rlm_bucket_pass_rate = 0
        else:
            rlm_win_rate = None
            rlm_edge = None
            rlm_bucket_improvement = None
            rlm_bucket_pass_rate = None

        tier_result = {
            'tier': tier_name,
            'duration_range': f"{low}-{high}hr" if high != float('inf') else f"{low}hr+",
            'all_markets': len(tier_markets),
            'all_win_rate': float(all_win_rate),
            'all_avg_price': float(all_avg_price),
            'all_edge': float(all_edge),
            'rlm_markets': rlm_n,
            'rlm_win_rate': float(rlm_win_rate) if rlm_win_rate else None,
            'rlm_edge': float(rlm_edge) if rlm_edge else None,
            'rlm_bucket_improvement': float(rlm_bucket_improvement) if rlm_bucket_improvement else None,
            'rlm_bucket_pass_rate': float(rlm_bucket_pass_rate) if rlm_bucket_pass_rate else None
        }
        tier_results.append(tier_result)

        print(f"\n{tier_name}:")
        print(f"  All markets: {len(tier_markets):,}, Win: {all_win_rate:.1%}, Edge: {all_edge:+.1%}")
        if rlm_n >= 10:
            print(f"  RLM markets: {rlm_n:,}, Win: {rlm_win_rate:.1%}, Edge: {rlm_edge:+.1%}")
            print(f"    Bucket improvement: {rlm_bucket_improvement:+.1%}, Pass rate: {rlm_bucket_pass_rate:.0%}")

    return tier_results


# =============================================================================
# CATEGORY ANALYSIS
# =============================================================================

def analyze_by_category(market_stats, baseline):
    """Analyze 6-24hr window performance by broad category."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: CATEGORY BREAKDOWN FOR 6-24HR WINDOW")
    print("=" * 80)

    # Filter to 6-24hr window
    window_mask = (market_stats['market_duration_hours'] >= 6) & (market_stats['market_duration_hours'] < 24)
    window_markets = market_stats[window_mask]

    # Also get RLM subset
    rlm_mask = (
        window_mask &
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )

    category_results = []

    for category in ['Sports', 'Crypto', 'Politics', 'Economics', 'Other']:
        cat_all = window_markets[window_markets['broad_category'] == category]
        cat_rlm = market_stats[rlm_mask & (market_stats['broad_category'] == category)]

        if len(cat_all) < 20:
            continue

        all_win_rate = cat_all['no_won'].mean()
        all_avg_price = cat_all['avg_no_price'].mean()
        all_edge = all_win_rate - all_avg_price / 100

        rlm_n = len(cat_rlm)
        if rlm_n >= 10:
            rlm_win_rate = cat_rlm['no_won'].mean()
            rlm_avg_price = cat_rlm['avg_no_price'].mean()
            rlm_edge = rlm_win_rate - rlm_avg_price / 100

            # Bucket improvement
            bucket_imps = []
            for bucket in cat_rlm['bucket_5c'].unique():
                bm = cat_rlm[cat_rlm['bucket_5c'] == bucket]
                if len(bm) >= 5 and bucket in baseline:
                    bucket_imps.append((bm['no_won'].mean() - baseline[bucket]['win_rate'], len(bm)))

            if bucket_imps:
                tw = sum(w[1] for w in bucket_imps)
                rlm_bucket_imp = sum(w[0] * w[1] for w in bucket_imps) / tw
            else:
                rlm_bucket_imp = None
        else:
            rlm_win_rate = None
            rlm_edge = None
            rlm_bucket_imp = None

        cat_result = {
            'category': category,
            'all_markets': len(cat_all),
            'all_win_rate': float(all_win_rate),
            'all_edge': float(all_edge),
            'rlm_markets': rlm_n,
            'rlm_win_rate': float(rlm_win_rate) if rlm_win_rate else None,
            'rlm_edge': float(rlm_edge) if rlm_edge else None,
            'rlm_bucket_improvement': float(rlm_bucket_imp) if rlm_bucket_imp else None
        }
        category_results.append(cat_result)

        print(f"\n{category}:")
        print(f"  All: {len(cat_all):,} markets, Win: {all_win_rate:.1%}, Edge: {all_edge:+.1%}")
        if rlm_n >= 10:
            print(f"  RLM: {rlm_n:,} markets, Win: {rlm_win_rate:.1%}, Edge: {rlm_edge:+.1%}")
            if rlm_bucket_imp:
                print(f"    Bucket improvement: {rlm_bucket_imp:+.1%}")

    return category_results


# =============================================================================
# TEMPORAL STABILITY
# =============================================================================

def analyze_temporal_stability(market_stats, baseline):
    """Analyze 6-24hr + RLM performance across time periods."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: TEMPORAL STABILITY")
    print("=" * 80)

    # Get 6-24hr + RLM markets
    signal_mask = (
        (market_stats['market_duration_hours'] >= 6) &
        (market_stats['market_duration_hours'] < 24) &
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )
    signal_markets = market_stats[signal_mask]

    temporal_results = []

    # By half-year
    print("\nBy Half-Year:")
    for period in sorted(signal_markets['year_half'].unique()):
        period_markets = signal_markets[signal_markets['year_half'] == period]
        if len(period_markets) < 20:
            continue

        win_rate = period_markets['no_won'].mean()
        avg_price = period_markets['avg_no_price'].mean()
        breakeven = avg_price / 100
        edge = win_rate - breakeven

        temporal_results.append({
            'period_type': 'half_year',
            'period': str(period),
            'n_markets': len(period_markets),
            'win_rate': float(win_rate),
            'edge': float(edge)
        })

        print(f"  {period}: {len(period_markets):,} markets, Win: {win_rate:.1%}, Edge: {edge:+.1%}")

    # By quarter
    print("\nBy Quarter:")
    for quarter in sorted(signal_markets['quarter'].unique()):
        q_markets = signal_markets[signal_markets['quarter'] == quarter]
        if len(q_markets) < 10:
            continue

        win_rate = q_markets['no_won'].mean()
        avg_price = q_markets['avg_no_price'].mean()
        breakeven = avg_price / 100
        edge = win_rate - breakeven

        temporal_results.append({
            'period_type': 'quarter',
            'period': str(quarter),
            'n_markets': len(q_markets),
            'win_rate': float(win_rate),
            'edge': float(edge)
        })

        print(f"  {quarter}: {len(q_markets):,} markets, Win: {win_rate:.1%}, Edge: {edge:+.1%}")

    return temporal_results


# =============================================================================
# FINE-GRAINED WINDOW SEARCH
# =============================================================================

def search_optimal_window(market_stats, baseline):
    """Search for optimal duration window with finer granularity."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: OPTIMAL WINDOW SEARCH")
    print("=" * 80)

    # Test windows with 2hr steps
    windows = [
        (2, 6), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 24),
        (6, 12), (6, 18), (6, 24), (8, 16), (10, 18), (12, 24),
        (4, 12), (4, 18), (4, 24),
    ]

    window_results = []

    for low, high in windows:
        window_mask = (
            (market_stats['market_duration_hours'] >= low) &
            (market_stats['market_duration_hours'] < high) &
            (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
            (market_stats['n_trades'] >= RLM_MIN_TRADES) &
            (market_stats['yes_price_moved_down'])
        )
        window_markets = market_stats[window_mask]

        n = len(window_markets)
        if n < 30:
            continue

        win_rate = window_markets['no_won'].mean()
        avg_price = window_markets['avg_no_price'].mean()
        breakeven = avg_price / 100
        edge = win_rate - breakeven

        # Bucket improvement
        bucket_imps = []
        positive_buckets = 0
        total_buckets = 0
        for bucket in window_markets['bucket_5c'].unique():
            bm = window_markets[window_markets['bucket_5c'] == bucket]
            if len(bm) >= 5 and bucket in baseline:
                total_buckets += 1
                imp = bm['no_won'].mean() - baseline[bucket]['win_rate']
                bucket_imps.append((imp, len(bm)))
                if imp > 0:
                    positive_buckets += 1

        if bucket_imps:
            tw = sum(w[1] for w in bucket_imps)
            bucket_improvement = sum(w[0] * w[1] for w in bucket_imps) / tw
            bucket_pass_rate = positive_buckets / total_buckets
        else:
            bucket_improvement = 0
            bucket_pass_rate = 0

        window_results.append({
            'window': f"{low}-{high}hr",
            'low_hr': low,
            'high_hr': high,
            'n_markets': n,
            'win_rate': float(win_rate),
            'avg_price': float(avg_price),
            'edge': float(edge),
            'bucket_improvement': float(bucket_improvement),
            'bucket_pass_rate': float(bucket_pass_rate)
        })

    # Sort by edge
    window_results.sort(key=lambda x: x['edge'], reverse=True)

    print("\nTop 10 Duration Windows by Edge:")
    for i, w in enumerate(window_results[:10]):
        print(f"  {i+1}. {w['window']}: N={w['n_markets']:,}, Edge={w['edge']:+.1%}, "
              f"Bucket Imp={w['bucket_improvement']:+.1%}, Pass={w['bucket_pass_rate']:.0%}")

    return window_results


# =============================================================================
# SIGNAL INTERACTIONS
# =============================================================================

def analyze_signal_interactions(market_stats, baseline):
    """Test 6-24hr combined with other signals."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: SIGNAL INTERACTIONS")
    print("=" * 80)

    # Base 6-24hr + RLM
    base_mask = (
        (market_stats['market_duration_hours'] >= 6) &
        (market_stats['market_duration_hours'] < 24) &
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )

    interaction_results = []

    # Test combinations
    combinations = [
        ('6-24hr + RLM (baseline)', base_mask, 'Base signal'),
        ('+ price_move < -5c', base_mask & (market_stats['price_move'] < -5), 'Large price drop'),
        ('+ price_move < -10c', base_mask & (market_stats['price_move'] < -10), 'Very large drop'),
        ('+ n_trades >= 30', base_mask & (market_stats['n_trades'] >= 30), 'More trades'),
        ('+ has_whale', base_mask & (market_stats['has_whale']), 'Has whale trade'),
        ('+ whale_count >= 2', base_mask & (market_stats['whale_count'] >= 2), 'Multiple whales'),
        ('+ large NO imbalance', base_mask & (market_stats['large_trade_imbalance'] <= -0.2), 'Large trades favor NO'),
        ('+ YES ratio > 70%', base_mask & (market_stats['yes_trade_ratio'] > 0.70), 'Higher YES ratio'),
        ('+ low lev_std', base_mask & (market_stats['lev_std'] < market_stats['lev_std'].median()), 'Low leverage std'),
    ]

    for name, mask, description in combinations:
        signal_markets = market_stats[mask]
        n = len(signal_markets)

        if n < 30:
            continue

        win_rate = signal_markets['no_won'].mean()
        avg_price = signal_markets['avg_no_price'].mean()
        breakeven = avg_price / 100
        edge = win_rate - breakeven

        # Bucket improvement
        bucket_imps = []
        positive_buckets = 0
        total_buckets = 0
        for bucket in signal_markets['bucket_5c'].unique():
            bm = signal_markets[signal_markets['bucket_5c'] == bucket]
            if len(bm) >= 5 and bucket in baseline:
                total_buckets += 1
                imp = bm['no_won'].mean() - baseline[bucket]['win_rate']
                bucket_imps.append((imp, len(bm)))
                if imp > 0:
                    positive_buckets += 1

        if bucket_imps:
            tw = sum(w[1] for w in bucket_imps)
            bucket_improvement = sum(w[0] * w[1] for w in bucket_imps) / tw
            bucket_pass_rate = positive_buckets / total_buckets
        else:
            bucket_improvement = 0
            bucket_pass_rate = 0

        interaction_results.append({
            'signal': name,
            'description': description,
            'n_markets': n,
            'win_rate': float(win_rate),
            'edge': float(edge),
            'bucket_improvement': float(bucket_improvement),
            'bucket_pass_rate': float(bucket_pass_rate)
        })

        print(f"\n{name}:")
        print(f"  N={n:,}, Win={win_rate:.1%}, Edge={edge:+.1%}")
        print(f"  Bucket: Imp={bucket_improvement:+.1%}, Pass={bucket_pass_rate:.0%}")

    return interaction_results


# =============================================================================
# RISK ANALYSIS
# =============================================================================

def analyze_risk(market_stats, baseline):
    """Comprehensive risk analysis for 6-24hr + RLM."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: RISK ANALYSIS")
    print("=" * 80)

    # Get signal markets
    signal_mask = (
        (market_stats['market_duration_hours'] >= 6) &
        (market_stats['market_duration_hours'] < 24) &
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )
    signal_markets = market_stats[signal_mask].copy()

    # Calculate profit per market
    signal_markets['profit'] = signal_markets.apply(
        lambda row: row['avg_no_price'] / 100 if row['no_won'] else -row['avg_no_price'] / 100,
        axis=1
    )

    total_profit = signal_markets['profit'].sum()

    # Concentration
    market_profits = signal_markets.groupby('market_ticker')['profit'].sum().sort_values(ascending=False)
    if total_profit > 0:
        top_1_pct = market_profits.head(1).sum() / total_profit
        top_5_pct = market_profits.head(5).sum() / total_profit
        top_10_pct = market_profits.head(10).sum() / total_profit
    else:
        top_1_pct = top_5_pct = top_10_pct = 0

    print(f"\nConcentration Analysis:")
    print(f"  Total profit units: {total_profit:.2f}")
    print(f"  Top 1 market: {top_1_pct:.1%}")
    print(f"  Top 5 markets: {top_5_pct:.1%}")
    print(f"  Top 10 markets: {top_10_pct:.1%}")

    # Win rate by price bucket
    print(f"\nWin Rate by Price Bucket:")
    bucket_analysis = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        bm = signal_markets[signal_markets['bucket_5c'] == bucket]
        if len(bm) >= 10:
            wr = bm['no_won'].mean()
            baseline_wr = baseline.get(bucket, {}).get('win_rate', 0)
            improvement = wr - baseline_wr if baseline_wr else 0

            bucket_analysis.append({
                'bucket': bucket,
                'n_markets': len(bm),
                'win_rate': float(wr),
                'baseline_win_rate': float(baseline_wr),
                'improvement': float(improvement)
            })

            print(f"  {bucket}c: N={len(bm):,}, Win={wr:.1%}, Baseline={baseline_wr:.1%}, Imp={improvement:+.1%}")

    # Drawdown simulation
    profits = signal_markets['profit'].values
    cumsum = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_drawdown = drawdown.max()

    print(f"\nDrawdown Analysis (random order):")
    print(f"  Max drawdown: {max_drawdown:.2f} units")

    # Category concentration
    print(f"\nCategory Distribution:")
    cat_dist = signal_markets.groupby('broad_category').agg({
        'market_ticker': 'count',
        'profit': 'sum'
    }).reset_index()
    cat_dist.columns = ['category', 'count', 'profit']
    cat_dist = cat_dist.sort_values('profit', ascending=False)
    for _, row in cat_dist.iterrows():
        pct = row['count'] / len(signal_markets) * 100
        print(f"  {row['category']}: {row['count']:,} ({pct:.1f}%), Profit={row['profit']:.2f}")

    return {
        'total_markets': len(signal_markets),
        'total_profit': float(total_profit),
        'concentration': {
            'top_1': float(top_1_pct),
            'top_5': float(top_5_pct),
            'top_10': float(top_10_pct)
        },
        'max_drawdown': float(max_drawdown),
        'bucket_analysis': bucket_analysis,
        'category_distribution': cat_dist.to_dict('records')
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FULL VALIDATION: 6-24hr Duration Window (H-MM151)")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    df = load_and_prepare_data()
    market_stats = build_market_stats(df)
    market_stats = compute_large_trade_imbalance(df, market_stats)
    baseline = build_baseline(market_stats)

    results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'H-MM151: 6-24hr Duration Window',
        'data_stats': {
            'total_trades': len(df),
            'total_markets': len(market_stats),
        }
    }

    # Analysis 1: Duration tier breakdown
    results['duration_tier_analysis'] = analyze_duration_tiers(market_stats, baseline)

    # Analysis 2: Category breakdown
    results['category_analysis'] = analyze_by_category(market_stats, baseline)

    # Analysis 3: Temporal stability
    results['temporal_stability'] = analyze_temporal_stability(market_stats, baseline)

    # Analysis 4: Optimal window search
    results['window_search'] = search_optimal_window(market_stats, baseline)

    # Analysis 5: Signal interactions
    results['signal_interactions'] = analyze_signal_interactions(market_stats, baseline)

    # Analysis 6: Risk analysis
    results['risk_analysis'] = analyze_risk(market_stats, baseline)

    # Full validation of 6-24hr + RLM
    print("\n" + "=" * 80)
    print("FULL VALIDATION: 6-24hr + RLM")
    print("=" * 80)

    signal_mask = (
        (market_stats['market_duration_hours'] >= 6) &
        (market_stats['market_duration_hours'] < 24) &
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )
    signal_markets = market_stats[signal_mask]

    results['full_validation'] = validate_strategy(
        signal_markets, baseline,
        "6-24hr Duration + RLM (NO bet)", 'no'
    )

    # Comparison with 24hr+
    print("\n" + "=" * 80)
    print("COMPARISON: 6-24hr vs 24hr+ vs Base RLM")
    print("=" * 80)

    # Base RLM (no duration filter)
    base_rlm_mask = (
        (market_stats['yes_trade_ratio'] > RLM_YES_RATIO) &
        (market_stats['n_trades'] >= RLM_MIN_TRADES) &
        (market_stats['yes_price_moved_down'])
    )
    base_rlm = market_stats[base_rlm_mask]

    # 24hr+
    dur_24_mask = base_rlm_mask & (market_stats['market_duration_hours'] >= 24)
    dur_24 = market_stats[dur_24_mask]

    comparisons = [
        ('Base RLM', base_rlm),
        ('24hr+ RLM', dur_24),
        ('6-24hr RLM', signal_markets)
    ]

    comparison_results = []
    for name, markets in comparisons:
        if len(markets) < 30:
            continue
        wr = markets['no_won'].mean()
        avg_p = markets['avg_no_price'].mean()
        be = avg_p / 100
        edge = wr - be

        comparison_results.append({
            'strategy': name,
            'n_markets': len(markets),
            'win_rate': float(wr),
            'avg_price': float(avg_p),
            'edge': float(edge)
        })

        print(f"\n{name}:")
        print(f"  N={len(markets):,}, Win={wr:.1%}, Edge={edge:+.1%}")

    results['comparison'] = comparison_results

    # Summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)

    validation = results['full_validation']
    print(f"\n6-24hr + RLM Strategy:")
    print(f"  Markets: {validation['n_markets']:,}")
    print(f"  Win Rate: {validation['win_rate']:.1%}")
    print(f"  Edge: {validation['raw_edge']:+.1%}")
    print(f"  Bucket Improvement: {validation['bucket_improvement']:+.1%}")
    print(f"  Bucket Pass Rate: {validation['bucket_pass_rate']:.0%}")
    print(f"  Status: {validation['status']}")

    # Find optimal window
    if results['window_search']:
        best = results['window_search'][0]
        print(f"\nOptimal Window: {best['window']}")
        print(f"  Edge: {best['edge']:+.1%}")
        print(f"  Markets: {best['n_markets']:,}")

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
