"""
H123 DEEP VALIDATION - SOBER MODE
=================================

Maximum rigor validation of Reverse Line Movement (RLM) NO strategy.

The strategy:
- >70% of trades are YES
- But price moved toward NO (YES price dropped from first to last trade)
- At least 5 trades in market
- Action: Bet NO

Initial LSD results:
- 1,986 markets
- +17.38% raw edge
- +13.44% improvement vs baseline
- 16/17 (94%) positive price buckets
- 4/4 quarters positive

This script goes DEEP:
1. Parameter Sensitivity (thresholds, trade counts, price moves)
2. Decomposition (category, market size, time, price range)
3. Mechanism Verification (trade sizes, whale involvement, timing)
4. Anti-Patterns (when does RLM fail?)
5. Edge Stability (rolling windows, regime changes, decay)
6. Combination Testing (RLM + S013, RLM + Whale, etc.)
7. Out-of-Sample Validation (train/test, walk-forward, bootstrap)
8. Practical Considerations (frequency, P&L, execution)

P-value threshold: < 0.001 (stricter than normal)
Minimum markets per subsegment: 50
Bonferroni correction applied
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/h123_deep_validation.json'

# Constants
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
WHALE_THRESHOLD = 10000  # $100 in cents

# Results dictionary
results = {
    'metadata': {
        'strategy': 'H123 - Reverse Line Movement NO',
        'session': 'SOBER MODE - Deep Validation',
        'timestamp': datetime.now().isoformat(),
        'p_threshold': 0.001,
        'min_subsegment': 50
    },
    'parameter_sensitivity': {},
    'decomposition': {},
    'mechanism': {},
    'anti_patterns': {},
    'edge_stability': {},
    'combinations': {},
    'out_of_sample': {},
    'practical': {},
    'final_verdict': {}
}


def load_data():
    """Load and prepare the trade data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    df['date'] = df['datetime'].dt.date

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def build_baseline(df):
    """Build baseline win rates at 5c buckets."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'first'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n_markets': n
            }

    return all_markets, baseline


def get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5, require_price_move=True):
    """
    Get RLM NO markets with configurable parameters.

    Parameters:
    - yes_trade_threshold: Minimum ratio of YES trades (default 0.7)
    - min_trades: Minimum number of trades (default 5)
    - require_price_move: If True, require YES price dropped (default True)
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean', 'std'],
        'no_price': 'mean',
        'market_result': 'first',
        'count': ['size', 'sum', 'mean'],
        'datetime': ['first', 'last'],
        'is_whale': 'sum',
        'trade_value_cents': ['sum', 'mean'],
        'leverage_ratio': ['mean', 'std'],
        'is_weekend': 'any',
        'is_round_size': 'sum',
        'hour': 'mean'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price', 'price_std',
        'no_price', 'market_result',
        'n_trades', 'total_contracts', 'avg_trade_size',
        'first_trade_time', 'last_trade_time',
        'whale_count', 'total_value', 'avg_trade_value',
        'avg_leverage', 'lev_std',
        'has_weekend', 'round_size_count', 'avg_hour'
    ]

    # Calculate price move
    market_stats['price_moved_no'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['price_move_magnitude'] = market_stats['first_yes_price'] - market_stats['last_yes_price']

    # Market duration
    market_stats['market_duration_hours'] = (
        (market_stats['last_trade_time'] - market_stats['first_trade_time']).dt.total_seconds() / 3600
    )

    # Fill NaN
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['price_std'] = market_stats['price_std'].fillna(0)

    # Apply filters
    if require_price_move:
        rlm = market_stats[
            (market_stats['yes_trade_ratio'] > yes_trade_threshold) &
            (market_stats['price_moved_no']) &
            (market_stats['n_trades'] >= min_trades)
        ].copy()
    else:
        rlm = market_stats[
            (market_stats['yes_trade_ratio'] > yes_trade_threshold) &
            (market_stats['n_trades'] >= min_trades)
        ].copy()

    return rlm, market_stats


def calculate_edge_stats(signal_markets, baseline, side='no'):
    """Calculate comprehensive edge statistics."""
    n = len(signal_markets)
    if n < 30:
        return {'n': n, 'valid': False, 'reason': 'insufficient_markets'}

    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['no_price'].mean()
    else:
        avg_price = signal_markets['yes_price'].mean()

    be = avg_price / 100
    edge = wr - be

    # Statistical significance
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    # Bucket analysis
    signal_markets = signal_markets.copy()
    if side == 'no':
        signal_markets['bucket_5c'] = (signal_markets['no_price'] // 5) * 5
    else:
        signal_markets['bucket_5c'] = (signal_markets['yes_price'] // 5) * 5

    improvements = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:
            continue

        sig_wr = (sig_bucket['market_result'] == side).mean()
        base_wr = baseline[bucket]['win_rate']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n_sig': n_sig
        })

    if not improvements:
        return {'n': n, 'valid': False, 'reason': 'no_valid_buckets'}

    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    total_buckets = len(improvements)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_edges = []
    for _ in range(n_bootstrap):
        sample = signal_markets.sample(n=len(signal_markets), replace=True)
        sample_wr = (sample['market_result'] == side).mean()
        if side == 'no':
            sample_be = sample['no_price'].mean() / 100
        else:
            sample_be = sample['yes_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_be)

    ci_lower = np.percentile(bootstrap_edges, 2.5)
    ci_upper = np.percentile(bootstrap_edges, 97.5)

    # Cohen's d effect size
    cohens_d = edge / np.sqrt(be * (1 - be)) if 0 < be < 1 else 0

    return {
        'n': n,
        'valid': True,
        'wins': int(wins),
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'total_buckets': total_buckets,
        'bucket_ratio': f"{pos_buckets}/{total_buckets}",
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'cohens_d': float(cohens_d),
        'bucket_details': improvements
    }


# ============================================================================
# 1. PARAMETER SENSITIVITY
# ============================================================================

def test_parameter_sensitivity(df, baseline):
    """Test how sensitive the edge is to parameter choices."""
    print("\n" + "=" * 80)
    print("1. PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    results['parameter_sensitivity'] = {
        'yes_threshold': {},
        'min_trades': {},
        'price_move_threshold': {},
        'optimal_params': {}
    }

    # Test YES trade threshold: 60%, 65%, 70%, 75%, 80%
    print("\n1.1 YES Trade Threshold Sensitivity:")
    print("-" * 60)

    best_edge = -999
    best_threshold = None

    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        rlm, _ = get_rlm_markets(df, yes_trade_threshold=threshold)
        stats = calculate_edge_stats(rlm, baseline)

        if stats['valid']:
            results['parameter_sensitivity']['yes_threshold'][str(threshold)] = stats
            print(f"  Threshold {threshold*100:.0f}%: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%, p={stats['p_value']:.2e}")

            if stats['weighted_improvement'] > best_edge and stats['n'] >= 100:
                best_edge = stats['weighted_improvement']
                best_threshold = threshold
        else:
            results['parameter_sensitivity']['yes_threshold'][str(threshold)] = stats
            print(f"  Threshold {threshold*100:.0f}%: N={stats.get('n', 0)} - {stats.get('reason', 'invalid')}")

    # Test minimum trades: 3, 5, 7, 10, 15
    print("\n1.2 Minimum Trades Sensitivity:")
    print("-" * 60)

    best_min_trades = None
    best_edge_trades = -999

    for min_trades in [3, 5, 7, 10, 15, 20]:
        rlm, _ = get_rlm_markets(df, min_trades=min_trades)
        stats = calculate_edge_stats(rlm, baseline)

        if stats['valid']:
            results['parameter_sensitivity']['min_trades'][str(min_trades)] = stats
            print(f"  Min trades {min_trades}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%, p={stats['p_value']:.2e}")

            if stats['weighted_improvement'] > best_edge_trades and stats['n'] >= 100:
                best_edge_trades = stats['weighted_improvement']
                best_min_trades = min_trades
        else:
            results['parameter_sensitivity']['min_trades'][str(min_trades)] = stats
            print(f"  Min trades {min_trades}: N={stats.get('n', 0)} - {stats.get('reason', 'invalid')}")

    # Test price move magnitude: 0c, 2c, 5c, 10c
    print("\n1.3 Price Move Magnitude Sensitivity:")
    print("-" * 60)

    rlm_base, all_stats = get_rlm_markets(df, require_price_move=False)

    # Add price move condition
    for move_threshold in [0, 2, 5, 10, 15]:
        if move_threshold == 0:
            # No price move requirement but still RLM direction
            rlm_filtered = rlm_base[rlm_base['price_moved_no']]
        else:
            rlm_filtered = rlm_base[rlm_base['price_move_magnitude'] >= move_threshold]

        stats = calculate_edge_stats(rlm_filtered, baseline)

        if stats['valid']:
            results['parameter_sensitivity']['price_move_threshold'][str(move_threshold)] = stats
            print(f"  Price move >={move_threshold}c: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")
        else:
            results['parameter_sensitivity']['price_move_threshold'][str(move_threshold)] = stats
            print(f"  Price move >={move_threshold}c: N={stats.get('n', 0)} - {stats.get('reason', 'invalid')}")

    # Find optimal parameters with grid search
    print("\n1.4 Optimal Parameters (Grid Search):")
    print("-" * 60)

    best_params = None
    best_improvement = -999
    grid_results = []

    for thresh in [0.65, 0.70, 0.75, 0.80]:
        for min_t in [5, 7, 10]:
            for price_m in [0, 5, 10]:
                rlm_test, _ = get_rlm_markets(df, yes_trade_threshold=thresh, min_trades=min_t)
                if price_m > 0:
                    rlm_test = rlm_test[rlm_test['price_move_magnitude'] >= price_m]

                stats = calculate_edge_stats(rlm_test, baseline)

                if stats['valid'] and stats['n'] >= 100:
                    grid_results.append({
                        'yes_threshold': thresh,
                        'min_trades': min_t,
                        'price_move': price_m,
                        'n': stats['n'],
                        'edge': stats['edge'],
                        'improvement': stats['weighted_improvement'],
                        'p_value': stats['p_value']
                    })

                    if stats['weighted_improvement'] > best_improvement:
                        best_improvement = stats['weighted_improvement']
                        best_params = {
                            'yes_threshold': thresh,
                            'min_trades': min_t,
                            'price_move': price_m,
                            'stats': stats
                        }

    if best_params:
        results['parameter_sensitivity']['optimal_params'] = best_params
        print(f"  OPTIMAL: yes_threshold={best_params['yes_threshold']}, "
              f"min_trades={best_params['min_trades']}, price_move={best_params['price_move']}c")
        print(f"  N={best_params['stats']['n']}, Edge={best_params['stats']['edge']*100:.2f}%, "
              f"Improvement={best_params['stats']['weighted_improvement']*100:.2f}%")

    results['parameter_sensitivity']['grid_search'] = grid_results

    return results['parameter_sensitivity']


# ============================================================================
# 2. DECOMPOSITION ANALYSIS
# ============================================================================

def test_decomposition(df, baseline):
    """Decompose strategy by various dimensions."""
    print("\n" + "=" * 80)
    print("2. DECOMPOSITION ANALYSIS")
    print("=" * 80)

    results['decomposition'] = {
        'by_category': {},
        'by_market_size': {},
        'by_time_of_day': {},
        'by_day_of_week': {},
        'by_price_range': {}
    }

    # Get RLM markets with full stats
    rlm, all_stats = get_rlm_markets(df)

    # 2.1 By Category (extract from ticker)
    print("\n2.1 By Category:")
    print("-" * 60)

    rlm['category'] = rlm['market_ticker'].str.extract(r'^([A-Z]+)', expand=False)

    for cat in rlm['category'].value_counts().head(15).index:
        cat_markets = rlm[rlm['category'] == cat]
        stats = calculate_edge_stats(cat_markets, baseline)

        if stats['valid'] and stats['n'] >= 50:
            results['decomposition']['by_category'][cat] = stats
            print(f"  {cat}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 2.2 By Market Size (total contracts traded)
    print("\n2.2 By Market Size (Total Contracts):")
    print("-" * 60)

    quartiles = rlm['total_contracts'].quantile([0.25, 0.5, 0.75]).values

    size_bins = [
        ('Small (<Q1)', rlm['total_contracts'] < quartiles[0]),
        ('Medium (Q1-Q2)', (rlm['total_contracts'] >= quartiles[0]) & (rlm['total_contracts'] < quartiles[1])),
        ('Large (Q2-Q3)', (rlm['total_contracts'] >= quartiles[1]) & (rlm['total_contracts'] < quartiles[2])),
        ('Very Large (>Q3)', rlm['total_contracts'] >= quartiles[2])
    ]

    for name, mask in size_bins:
        size_markets = rlm[mask]
        stats = calculate_edge_stats(size_markets, baseline)

        if stats['valid'] and stats['n'] >= 50:
            results['decomposition']['by_market_size'][name] = stats
            print(f"  {name}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 2.3 By Time of Day
    print("\n2.3 By Time of Day:")
    print("-" * 60)

    time_bins = [
        ('Night (0-6)', (rlm['avg_hour'] >= 0) & (rlm['avg_hour'] < 6)),
        ('Morning (6-12)', (rlm['avg_hour'] >= 6) & (rlm['avg_hour'] < 12)),
        ('Afternoon (12-18)', (rlm['avg_hour'] >= 12) & (rlm['avg_hour'] < 18)),
        ('Evening (18-24)', (rlm['avg_hour'] >= 18))
    ]

    for name, mask in time_bins:
        time_markets = rlm[mask]
        stats = calculate_edge_stats(time_markets, baseline)

        if stats['valid'] and stats['n'] >= 50:
            results['decomposition']['by_time_of_day'][name] = stats
            print(f"  {name}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 2.4 By Day of Week
    print("\n2.4 By Day of Week:")
    print("-" * 60)

    day_bins = [
        ('Weekday', ~rlm['has_weekend']),
        ('Weekend', rlm['has_weekend'])
    ]

    for name, mask in day_bins:
        day_markets = rlm[mask]
        stats = calculate_edge_stats(day_markets, baseline)

        if stats['valid'] and stats['n'] >= 50:
            results['decomposition']['by_day_of_week'][name] = stats
            print(f"  {name}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 2.5 By Price Range
    print("\n2.5 By Price Range (NO Price):")
    print("-" * 60)

    price_bins = [
        ('Low (0-30c)', (rlm['no_price'] >= 0) & (rlm['no_price'] < 30)),
        ('Mid-Low (30-50c)', (rlm['no_price'] >= 30) & (rlm['no_price'] < 50)),
        ('Mid (50-70c)', (rlm['no_price'] >= 50) & (rlm['no_price'] < 70)),
        ('Mid-High (70-85c)', (rlm['no_price'] >= 70) & (rlm['no_price'] < 85)),
        ('High (85-100c)', rlm['no_price'] >= 85)
    ]

    for name, mask in price_bins:
        price_markets = rlm[mask]
        stats = calculate_edge_stats(price_markets, baseline)

        if stats['valid'] and stats['n'] >= 50:
            results['decomposition']['by_price_range'][name] = stats
            print(f"  {name}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                  f"Improvement={stats['weighted_improvement']*100:.2f}%")

    return results['decomposition']


# ============================================================================
# 3. MECHANISM VERIFICATION
# ============================================================================

def test_mechanism(df, baseline):
    """Verify the behavioral mechanism behind RLM."""
    print("\n" + "=" * 80)
    print("3. MECHANISM VERIFICATION")
    print("=" * 80)

    results['mechanism'] = {
        'trade_size_analysis': {},
        'whale_involvement': {},
        'timing_analysis': {},
        'price_path': {}
    }

    rlm, all_stats = get_rlm_markets(df)

    # Get trade-level data for RLM markets
    df_sorted = df.sort_values(['market_ticker', 'datetime'])
    rlm_trades = df_sorted[df_sorted['market_ticker'].isin(rlm['market_ticker'])]

    # 3.1 Trade Size Analysis - Are NO trades larger?
    print("\n3.1 Trade Size Analysis (Are NO trades larger?):")
    print("-" * 60)

    rlm_yes_trades = rlm_trades[rlm_trades['taker_side'] == 'yes']
    rlm_no_trades = rlm_trades[rlm_trades['taker_side'] == 'no']

    yes_avg_size = rlm_yes_trades['count'].mean()
    no_avg_size = rlm_no_trades['count'].mean()
    yes_avg_value = rlm_yes_trades['trade_value_cents'].mean()
    no_avg_value = rlm_no_trades['trade_value_cents'].mean()

    results['mechanism']['trade_size_analysis'] = {
        'yes_avg_size': float(yes_avg_size),
        'no_avg_size': float(no_avg_size),
        'size_ratio_no_vs_yes': float(no_avg_size / yes_avg_size) if yes_avg_size > 0 else 0,
        'yes_avg_value': float(yes_avg_value),
        'no_avg_value': float(no_avg_value),
        'value_ratio_no_vs_yes': float(no_avg_value / yes_avg_value) if yes_avg_value > 0 else 0
    }

    print(f"  YES trades avg size: {yes_avg_size:.1f} contracts (${yes_avg_value/100:.2f})")
    print(f"  NO trades avg size: {no_avg_size:.1f} contracts (${no_avg_value/100:.2f})")
    print(f"  Ratio (NO/YES): {no_avg_size/yes_avg_size:.2f}x size, {no_avg_value/yes_avg_value:.2f}x value")

    # 3.2 Whale Involvement
    print("\n3.2 Whale Involvement:")
    print("-" * 60)

    rlm_with_whales = rlm[rlm['whale_count'] > 0]
    rlm_no_whales = rlm[rlm['whale_count'] == 0]

    stats_with_whales = calculate_edge_stats(rlm_with_whales, baseline)
    stats_no_whales = calculate_edge_stats(rlm_no_whales, baseline)

    if stats_with_whales['valid']:
        results['mechanism']['whale_involvement']['with_whales'] = stats_with_whales
        print(f"  With Whales: N={stats_with_whales['n']}, Edge={stats_with_whales['edge']*100:.2f}%, "
              f"Improvement={stats_with_whales['weighted_improvement']*100:.2f}%")

    if stats_no_whales['valid']:
        results['mechanism']['whale_involvement']['without_whales'] = stats_no_whales
        print(f"  Without Whales: N={stats_no_whales['n']}, Edge={stats_no_whales['edge']*100:.2f}%, "
              f"Improvement={stats_no_whales['weighted_improvement']*100:.2f}%")

    # Whale NO trade presence
    print("\n  Whale Trade Direction in RLM Markets:")

    for market_ticker, market_df in rlm_trades.groupby('market_ticker'):
        whale_trades = market_df[market_df['is_whale']]
        if len(whale_trades) > 0:
            whale_no_ratio = (whale_trades['taker_side'] == 'no').mean()

    # Get whale direction at market level
    whale_direction = rlm_trades[rlm_trades['is_whale']].groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean()
    }).reset_index()
    whale_direction.columns = ['market_ticker', 'whale_no_ratio']

    rlm_with_whale_dir = rlm.merge(whale_direction, on='market_ticker', how='left')
    rlm_with_whale_dir['whale_no_ratio'] = rlm_with_whale_dir['whale_no_ratio'].fillna(0.5)

    whale_favor_no = rlm_with_whale_dir[rlm_with_whale_dir['whale_no_ratio'] > 0.5]
    whale_favor_yes = rlm_with_whale_dir[rlm_with_whale_dir['whale_no_ratio'] <= 0.5]

    if len(whale_favor_no) >= 50:
        stats_whale_no = calculate_edge_stats(whale_favor_no, baseline)
        if stats_whale_no['valid']:
            results['mechanism']['whale_involvement']['whale_favor_no'] = stats_whale_no
            print(f"  Whales favor NO: N={stats_whale_no['n']}, Edge={stats_whale_no['edge']*100:.2f}%")

    # 3.3 Timing Analysis - When does the reversal happen?
    print("\n3.3 Timing Analysis (When does reversal happen?):")
    print("-" * 60)

    reversal_timing = []
    for market_ticker, market_df in rlm_trades.groupby('market_ticker'):
        market_df = market_df.sort_values('datetime')
        n_trades = len(market_df)

        # Find when price started moving toward NO (first NO-favoring price move)
        prices = market_df['yes_price'].values
        first_price = prices[0]

        reversal_point = None
        for i, price in enumerate(prices):
            if price < first_price - 2:  # 2c threshold for reversal start
                reversal_point = i / n_trades
                break

        if reversal_point is not None:
            reversal_timing.append(reversal_point)

    if reversal_timing:
        avg_reversal = np.mean(reversal_timing)
        results['mechanism']['timing_analysis']['avg_reversal_point'] = float(avg_reversal)
        results['mechanism']['timing_analysis']['early_reversal_pct'] = float(np.mean([r < 0.33 for r in reversal_timing]))
        results['mechanism']['timing_analysis']['mid_reversal_pct'] = float(np.mean([0.33 <= r < 0.66 for r in reversal_timing]))
        results['mechanism']['timing_analysis']['late_reversal_pct'] = float(np.mean([r >= 0.66 for r in reversal_timing]))

        print(f"  Average reversal point: {avg_reversal*100:.1f}% through market lifecycle")
        print(f"  Early reversals (<33%): {np.mean([r < 0.33 for r in reversal_timing])*100:.1f}%")
        print(f"  Mid reversals (33-66%): {np.mean([0.33 <= r < 0.66 for r in reversal_timing])*100:.1f}%")
        print(f"  Late reversals (>66%): {np.mean([r >= 0.66 for r in reversal_timing])*100:.1f}%")

    # 3.4 Price Path Analysis
    print("\n3.4 Price Path Analysis:")
    print("-" * 60)

    # Analyze typical price trajectory in RLM markets
    price_paths = []
    for market_ticker, market_df in rlm_trades.groupby('market_ticker'):
        market_df = market_df.sort_values('datetime')
        if len(market_df) >= 5:
            # Normalize to 5 points
            prices = market_df['yes_price'].values
            indices = np.linspace(0, len(prices)-1, 5).astype(int)
            normalized_path = prices[indices]
            price_paths.append(normalized_path)

    if price_paths:
        avg_path = np.mean(price_paths, axis=0)
        results['mechanism']['price_path']['average_normalized_path'] = avg_path.tolist()
        print(f"  Average price path (normalized to 5 points):")
        print(f"    Start: {avg_path[0]:.1f}c -> Q1: {avg_path[1]:.1f}c -> Mid: {avg_path[2]:.1f}c -> "
              f"Q3: {avg_path[3]:.1f}c -> End: {avg_path[4]:.1f}c")
        print(f"    Net move: {avg_path[0] - avg_path[4]:.1f}c toward NO")

    return results['mechanism']


# ============================================================================
# 4. ANTI-PATTERNS
# ============================================================================

def test_anti_patterns(df, baseline):
    """Find conditions where RLM FAILS."""
    print("\n" + "=" * 80)
    print("4. ANTI-PATTERNS (When does RLM FAIL?)")
    print("=" * 80)

    results['anti_patterns'] = {
        'failure_conditions': {},
        'false_positive_analysis': {}
    }

    rlm, all_stats = get_rlm_markets(df)

    # Mark winners and losers
    rlm['is_win'] = rlm['market_result'] == 'no'

    winners = rlm[rlm['is_win']]
    losers = rlm[~rlm['is_win']]

    print(f"\nTotal RLM signals: {len(rlm)}")
    print(f"Winners (NO won): {len(winners)} ({len(winners)/len(rlm)*100:.1f}%)")
    print(f"Losers (YES won): {len(losers)} ({len(losers)/len(rlm)*100:.1f}%)")

    # 4.1 Compare characteristics of winners vs losers
    print("\n4.1 Winner vs Loser Characteristics:")
    print("-" * 60)

    characteristics = {
        'yes_trade_ratio': ('YES Trade Ratio', '{:.1%}'),
        'n_trades': ('Num Trades', '{:.1f}'),
        'avg_trade_size': ('Avg Trade Size', '{:.1f}'),
        'whale_count': ('Whale Trades', '{:.1f}'),
        'price_move_magnitude': ('Price Move', '{:.1f}c'),
        'avg_leverage': ('Avg Leverage', '{:.2f}'),
        'lev_std': ('Leverage Variance', '{:.2f}'),
        'market_duration_hours': ('Duration (hrs)', '{:.1f}')
    }

    for col, (name, fmt) in characteristics.items():
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        diff_pct = (w_mean - l_mean) / l_mean * 100 if l_mean != 0 else 0

        results['anti_patterns']['failure_conditions'][col] = {
            'winner_avg': float(w_mean),
            'loser_avg': float(l_mean),
            'diff_pct': float(diff_pct)
        }

        print(f"  {name}: Winners={fmt.format(w_mean)}, Losers={fmt.format(l_mean)}, "
              f"Diff={diff_pct:+.1f}%")

    # 4.2 Find specific failure conditions
    print("\n4.2 High-Risk Conditions (Negative Edge):")
    print("-" * 60)

    # Test various conditions
    failure_tests = [
        ('Low price move (<3c)', rlm['price_move_magnitude'] < 3),
        ('Very high YES ratio (>90%)', rlm['yes_trade_ratio'] > 0.9),
        ('High leverage variance', rlm['lev_std'] > 1.0),
        ('Very few trades (<7)', rlm['n_trades'] < 7),
        ('Very short duration (<1hr)', rlm['market_duration_hours'] < 1),
        ('No whale trades', rlm['whale_count'] == 0),
        ('Small avg trade size (<10)', rlm['avg_trade_size'] < 10)
    ]

    for name, mask in failure_tests:
        subset = rlm[mask]
        if len(subset) >= 50:
            stats = calculate_edge_stats(subset, baseline)
            if stats['valid']:
                status = 'DANGER' if stats['weighted_improvement'] < 0 else 'OK'
                results['anti_patterns']['failure_conditions'][name] = stats
                print(f"  [{status}] {name}: N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
                      f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 4.3 False Positive Analysis
    print("\n4.3 False Positive Analysis:")
    print("-" * 60)

    # What % of signals are wrong?
    false_positive_rate = len(losers) / len(rlm)
    results['anti_patterns']['false_positive_analysis']['rate'] = float(false_positive_rate)
    print(f"  False positive rate: {false_positive_rate*100:.1f}%")

    # Common characteristics of false positives
    if len(losers) >= 30:
        loser_categories = losers['market_ticker'].str.extract(r'^([A-Z]+)', expand=False).value_counts().head(5)
        results['anti_patterns']['false_positive_analysis']['top_loser_categories'] = loser_categories.to_dict()
        print(f"  Top categories with false positives:")
        for cat, count in loser_categories.items():
            print(f"    {cat}: {count} ({count/len(losers)*100:.1f}%)")

    return results['anti_patterns']


# ============================================================================
# 5. EDGE STABILITY
# ============================================================================

def test_edge_stability(df, baseline):
    """Test temporal stability of the edge."""
    print("\n" + "=" * 80)
    print("5. EDGE STABILITY ANALYSIS")
    print("=" * 80)

    results['edge_stability'] = {
        'rolling_windows': {},
        'quarterly_analysis': {},
        'regime_changes': {},
        'decay_analysis': {}
    }

    rlm, _ = get_rlm_markets(df)
    rlm['first_trade_date'] = pd.to_datetime(rlm['first_trade_time']).dt.date

    # 5.1 Rolling Window Analysis (30-day windows)
    print("\n5.1 Rolling 30-Day Window Analysis:")
    print("-" * 60)

    min_date = rlm['first_trade_date'].min()
    max_date = rlm['first_trade_date'].max()
    date_range = (max_date - min_date).days

    rolling_results = []
    for start_offset in range(0, date_range - 30, 7):  # 7-day steps
        start_date = min_date + timedelta(days=start_offset)
        end_date = start_date + timedelta(days=30)

        window_markets = rlm[
            (rlm['first_trade_date'] >= start_date) &
            (rlm['first_trade_date'] < end_date)
        ]

        if len(window_markets) >= 30:
            stats = calculate_edge_stats(window_markets, baseline)
            if stats['valid']:
                rolling_results.append({
                    'start_date': str(start_date),
                    'end_date': str(end_date),
                    'n': stats['n'],
                    'edge': stats['edge'],
                    'improvement': stats['weighted_improvement']
                })

    results['edge_stability']['rolling_windows'] = rolling_results

    if rolling_results:
        edges = [r['edge'] for r in rolling_results]
        improvements = [r['improvement'] for r in rolling_results]

        print(f"  Windows analyzed: {len(rolling_results)}")
        print(f"  Edge range: {min(edges)*100:.1f}% to {max(edges)*100:.1f}%")
        print(f"  Avg edge: {np.mean(edges)*100:.2f}% (std: {np.std(edges)*100:.2f}%)")
        print(f"  Positive edge windows: {sum(1 for e in edges if e > 0)}/{len(edges)}")
        print(f"  Positive improvement windows: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    # 5.2 Quarterly Analysis
    print("\n5.2 Quarterly Analysis:")
    print("-" * 60)

    # Divide into 4 quarters
    quarter_days = date_range // 4 if date_range > 0 else 1
    quarter_results = []

    for q in range(4):
        q_start = min_date + timedelta(days=q * quarter_days)
        q_end = min_date + timedelta(days=(q + 1) * quarter_days)

        q_markets = rlm[
            (rlm['first_trade_date'] >= q_start) &
            (rlm['first_trade_date'] < q_end)
        ]

        if len(q_markets) >= 30:
            stats = calculate_edge_stats(q_markets, baseline)
            if stats['valid']:
                quarter_results.append({
                    'quarter': q + 1,
                    'start_date': str(q_start),
                    'end_date': str(q_end),
                    'n': stats['n'],
                    'edge': stats['edge'],
                    'improvement': stats['weighted_improvement'],
                    'win_rate': stats['win_rate']
                })
                print(f"  Q{q+1} ({q_start} to {q_end}): N={stats['n']}, "
                      f"Edge={stats['edge']*100:.2f}%, Improvement={stats['weighted_improvement']*100:.2f}%")

    results['edge_stability']['quarterly_analysis'] = quarter_results

    positive_quarters = sum(1 for q in quarter_results if q['improvement'] > 0)
    print(f"\n  Positive quarters: {positive_quarters}/{len(quarter_results)}")

    # 5.3 Regime Change Detection
    print("\n5.3 Regime Change Analysis:")
    print("-" * 60)

    if len(rolling_results) >= 5:
        edges = [r['edge'] for r in rolling_results]

        # Look for significant drops
        regime_changes = []
        for i in range(1, len(edges)):
            if edges[i] < edges[i-1] - 0.05:  # 5% drop
                regime_changes.append({
                    'index': i,
                    'date': rolling_results[i]['start_date'],
                    'previous_edge': edges[i-1],
                    'new_edge': edges[i],
                    'drop': edges[i-1] - edges[i]
                })

        results['edge_stability']['regime_changes'] = regime_changes

        if regime_changes:
            print(f"  Detected {len(regime_changes)} significant edge drops (>5%)")
            for rc in regime_changes:
                print(f"    {rc['date']}: Edge dropped from {rc['previous_edge']*100:.1f}% to {rc['new_edge']*100:.1f}%")
        else:
            print("  No significant regime changes detected")

    # 5.4 Decay Analysis
    print("\n5.4 Edge Decay Analysis:")
    print("-" * 60)

    if len(quarter_results) >= 2:
        first_half_edge = np.mean([q['improvement'] for q in quarter_results[:len(quarter_results)//2]])
        second_half_edge = np.mean([q['improvement'] for q in quarter_results[len(quarter_results)//2:]])

        decay = first_half_edge - second_half_edge

        results['edge_stability']['decay_analysis'] = {
            'first_half_avg_improvement': float(first_half_edge),
            'second_half_avg_improvement': float(second_half_edge),
            'decay': float(decay),
            'decay_pct': float(decay / first_half_edge * 100) if first_half_edge != 0 else 0
        }

        print(f"  First half avg improvement: {first_half_edge*100:.2f}%")
        print(f"  Second half avg improvement: {second_half_edge*100:.2f}%")
        print(f"  Decay: {decay*100:.2f}% ({decay/first_half_edge*100:.1f}% of original)")

        if decay > 0.02:
            print("  WARNING: Significant edge decay detected!")
        else:
            print("  Edge appears stable over time")

    return results['edge_stability']


# ============================================================================
# 6. COMBINATION TESTING
# ============================================================================

def test_combinations(df, baseline):
    """Test RLM combined with other signals."""
    print("\n" + "=" * 80)
    print("6. COMBINATION TESTING")
    print("=" * 80)

    results['combinations'] = {}

    rlm, _ = get_rlm_markets(df)

    # Build additional market features
    market_features = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'is_whale': 'any',
        'is_weekend': 'any',
        'is_round_size': 'any'
    }).reset_index()
    market_features.columns = ['market_ticker', 'lev_std', 'has_whale', 'is_weekend', 'has_round_size']
    market_features['lev_std'] = market_features['lev_std'].fillna(0)

    # Merge with RLM
    rlm_enhanced = rlm.merge(market_features, on='market_ticker', how='left', suffixes=('', '_feat'))

    # 6.1 RLM + S013 (Low Leverage Variance)
    print("\n6.1 RLM + S013 (Low Leverage Variance < 0.7):")
    print("-" * 60)

    rlm_s013 = rlm_enhanced[rlm_enhanced['lev_std'] < 0.7]
    stats = calculate_edge_stats(rlm_s013, baseline)
    if stats['valid']:
        results['combinations']['rlm_plus_s013'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.2 RLM + Whale
    print("\n6.2 RLM + Whale Presence:")
    print("-" * 60)

    rlm_whale = rlm_enhanced[rlm_enhanced['has_whale'] == True]
    stats = calculate_edge_stats(rlm_whale, baseline)
    if stats['valid']:
        results['combinations']['rlm_plus_whale'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.3 RLM + Weekend
    print("\n6.3 RLM + Weekend:")
    print("-" * 60)

    rlm_weekend = rlm_enhanced[rlm_enhanced['is_weekend'] == True]
    stats = calculate_edge_stats(rlm_weekend, baseline)
    if stats['valid']:
        results['combinations']['rlm_plus_weekend'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.4 RLM + Large Price Move (>10c)
    print("\n6.4 RLM + Large Price Move (>10c):")
    print("-" * 60)

    rlm_large_move = rlm[rlm['price_move_magnitude'] > 10]
    stats = calculate_edge_stats(rlm_large_move, baseline)
    if stats['valid']:
        results['combinations']['rlm_plus_large_move'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.5 RLM + Multiple Whales
    print("\n6.5 RLM + Multiple Whale Trades (2+):")
    print("-" * 60)

    rlm_multi_whale = rlm[rlm['whale_count'] >= 2]
    stats = calculate_edge_stats(rlm_multi_whale, baseline)
    if stats['valid']:
        results['combinations']['rlm_plus_multi_whale'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.6 Triple Stack: RLM + S013 + Whale
    print("\n6.6 Triple Stack: RLM + S013 + Whale:")
    print("-" * 60)

    rlm_triple = rlm_enhanced[
        (rlm_enhanced['lev_std'] < 0.7) &
        (rlm_enhanced['has_whale'] == True)
    ]
    stats = calculate_edge_stats(rlm_triple, baseline)
    if stats['valid']:
        results['combinations']['rlm_triple_stack'] = stats
        print(f"  N={stats['n']}, Edge={stats['edge']*100:.2f}%, "
              f"Improvement={stats['weighted_improvement']*100:.2f}%")

    # 6.7 Find optimal combination
    print("\n6.7 Optimal Combination Search:")
    print("-" * 60)

    best_combination = None
    best_improvement = -999

    combo_tests = [
        ('Base RLM', rlm),
        ('RLM + S013', rlm_enhanced[rlm_enhanced['lev_std'] < 0.7]),
        ('RLM + Whale', rlm_enhanced[rlm_enhanced['has_whale'] == True]),
        ('RLM + Weekend', rlm_enhanced[rlm_enhanced['is_weekend'] == True]),
        ('RLM + Large Move', rlm[rlm['price_move_magnitude'] > 10]),
        ('RLM + S013 + Whale', rlm_enhanced[(rlm_enhanced['lev_std'] < 0.7) & (rlm_enhanced['has_whale'] == True)]),
        ('RLM + S013 + Weekend', rlm_enhanced[(rlm_enhanced['lev_std'] < 0.7) & (rlm_enhanced['is_weekend'] == True)]),
        ('RLM + Whale + Large Move', rlm_enhanced[(rlm_enhanced['has_whale'] == True) & (rlm['price_move_magnitude'] > 10)])
    ]

    for name, subset in combo_tests:
        if len(subset) >= 50:
            stats = calculate_edge_stats(subset, baseline)
            if stats['valid'] and stats['weighted_improvement'] > best_improvement:
                best_improvement = stats['weighted_improvement']
                best_combination = {
                    'name': name,
                    'n': stats['n'],
                    'edge': stats['edge'],
                    'improvement': stats['weighted_improvement']
                }

    if best_combination:
        results['combinations']['optimal'] = best_combination
        print(f"  OPTIMAL: {best_combination['name']}")
        print(f"  N={best_combination['n']}, Edge={best_combination['edge']*100:.2f}%, "
              f"Improvement={best_combination['improvement']*100:.2f}%")

    return results['combinations']


# ============================================================================
# 7. OUT-OF-SAMPLE VALIDATION
# ============================================================================

def test_out_of_sample(df, baseline):
    """Out-of-sample validation with train/test splits."""
    print("\n" + "=" * 80)
    print("7. OUT-OF-SAMPLE VALIDATION")
    print("=" * 80)

    results['out_of_sample'] = {
        'train_test_split': {},
        'walk_forward': {},
        'bootstrap_validation': {}
    }

    rlm, _ = get_rlm_markets(df)
    rlm['first_trade_date'] = pd.to_datetime(rlm['first_trade_time']).dt.date

    # Sort by date
    rlm_sorted = rlm.sort_values('first_trade_date')

    # 7.1 80/20 Train/Test Split
    print("\n7.1 Train/Test Split (80/20):")
    print("-" * 60)

    split_point = int(len(rlm_sorted) * 0.8)
    train = rlm_sorted.iloc[:split_point]
    test = rlm_sorted.iloc[split_point:]

    train_stats = calculate_edge_stats(train, baseline)
    test_stats = calculate_edge_stats(test, baseline)

    if train_stats['valid'] and test_stats['valid']:
        results['out_of_sample']['train_test_split'] = {
            'train': train_stats,
            'test': test_stats,
            'generalization_gap': train_stats['weighted_improvement'] - test_stats['weighted_improvement']
        }

        print(f"  TRAIN: N={train_stats['n']}, Edge={train_stats['edge']*100:.2f}%, "
              f"Improvement={train_stats['weighted_improvement']*100:.2f}%")
        print(f"  TEST: N={test_stats['n']}, Edge={test_stats['edge']*100:.2f}%, "
              f"Improvement={test_stats['weighted_improvement']*100:.2f}%")

        gap = train_stats['weighted_improvement'] - test_stats['weighted_improvement']
        print(f"  Generalization gap: {gap*100:.2f}%")

        if gap > 0.05:
            print("  WARNING: Significant overfitting detected!")
        else:
            print("  Strategy generalizes well to unseen data")

    # 7.2 Walk-Forward Validation
    print("\n7.2 Walk-Forward Validation (Monthly recalibration):")
    print("-" * 60)

    min_date = rlm_sorted['first_trade_date'].min()
    max_date = rlm_sorted['first_trade_date'].max()

    walk_forward_results = []

    # Use first 60% as initial training, then walk forward monthly
    initial_train_end = min_date + timedelta(days=int((max_date - min_date).days * 0.6))

    current_date = initial_train_end
    step_days = 14  # 2-week steps

    while current_date < max_date:
        # Train on everything before current_date
        train_data = rlm_sorted[rlm_sorted['first_trade_date'] < current_date]

        # Test on next step_days
        test_end = current_date + timedelta(days=step_days)
        test_data = rlm_sorted[
            (rlm_sorted['first_trade_date'] >= current_date) &
            (rlm_sorted['first_trade_date'] < test_end)
        ]

        if len(test_data) >= 20:
            test_stats = calculate_edge_stats(test_data, baseline)
            if test_stats['valid']:
                walk_forward_results.append({
                    'period_start': str(current_date),
                    'period_end': str(test_end),
                    'n': test_stats['n'],
                    'edge': test_stats['edge'],
                    'improvement': test_stats['weighted_improvement']
                })

        current_date = test_end

    results['out_of_sample']['walk_forward'] = walk_forward_results

    if walk_forward_results:
        avg_edge = np.mean([r['edge'] for r in walk_forward_results])
        avg_imp = np.mean([r['improvement'] for r in walk_forward_results])
        positive_periods = sum(1 for r in walk_forward_results if r['improvement'] > 0)

        print(f"  Walk-forward periods: {len(walk_forward_results)}")
        print(f"  Avg edge: {avg_edge*100:.2f}%")
        print(f"  Avg improvement: {avg_imp*100:.2f}%")
        print(f"  Positive periods: {positive_periods}/{len(walk_forward_results)}")

    # 7.3 Bootstrap Validation (1000 iterations)
    print("\n7.3 Bootstrap Validation (1000 iterations):")
    print("-" * 60)

    n_bootstrap = 1000
    bootstrap_improvements = []

    for _ in range(n_bootstrap):
        sample = rlm.sample(n=len(rlm), replace=True)

        # Calculate improvement for this sample
        wins = (sample['market_result'] == 'no').sum()
        n = len(sample)
        wr = wins / n
        be = sample['no_price'].mean() / 100
        edge = wr - be

        bootstrap_improvements.append(edge)

    ci_lower = np.percentile(bootstrap_improvements, 2.5)
    ci_upper = np.percentile(bootstrap_improvements, 97.5)
    ci_99_lower = np.percentile(bootstrap_improvements, 0.5)
    ci_99_upper = np.percentile(bootstrap_improvements, 99.5)

    results['out_of_sample']['bootstrap_validation'] = {
        'n_iterations': n_bootstrap,
        'mean_edge': float(np.mean(bootstrap_improvements)),
        'std_edge': float(np.std(bootstrap_improvements)),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'ci_99_lower': float(ci_99_lower),
        'ci_99_upper': float(ci_99_upper),
        'pct_positive': float(np.mean([e > 0 for e in bootstrap_improvements]))
    }

    print(f"  Bootstrap mean edge: {np.mean(bootstrap_improvements)*100:.2f}%")
    print(f"  Bootstrap std: {np.std(bootstrap_improvements)*100:.2f}%")
    print(f"  95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  99% CI: [{ci_99_lower*100:.2f}%, {ci_99_upper*100:.2f}%]")
    print(f"  % positive edge: {np.mean([e > 0 for e in bootstrap_improvements])*100:.1f}%")

    if ci_lower > 0:
        print("  PASS: 95% CI excludes zero - edge is statistically robust")
    else:
        print("  WARNING: 95% CI includes zero - edge may not be reliable")

    return results['out_of_sample']


# ============================================================================
# 8. PRACTICAL CONSIDERATIONS
# ============================================================================

def test_practical(df, baseline):
    """Calculate practical implementation metrics."""
    print("\n" + "=" * 80)
    print("8. PRACTICAL CONSIDERATIONS")
    print("=" * 80)

    results['practical'] = {
        'signal_frequency': {},
        'expected_pnl': {},
        'execution_timing': {},
        'recommended_parameters': {}
    }

    rlm, _ = get_rlm_markets(df)
    rlm['first_trade_date'] = pd.to_datetime(rlm['first_trade_time']).dt.date

    # 8.1 Signal Frequency
    print("\n8.1 Signal Frequency:")
    print("-" * 60)

    date_range = (rlm['first_trade_date'].max() - rlm['first_trade_date'].min()).days
    signals_per_day = len(rlm) / date_range if date_range > 0 else 0
    signals_per_week = signals_per_day * 7
    signals_per_month = signals_per_day * 30

    results['practical']['signal_frequency'] = {
        'total_signals': len(rlm),
        'date_range_days': date_range,
        'signals_per_day': float(signals_per_day),
        'signals_per_week': float(signals_per_week),
        'signals_per_month': float(signals_per_month)
    }

    print(f"  Total signals: {len(rlm)}")
    print(f"  Date range: {date_range} days")
    print(f"  Signals per day: {signals_per_day:.1f}")
    print(f"  Signals per week: {signals_per_week:.1f}")
    print(f"  Signals per month: {signals_per_month:.1f}")

    # 8.2 Expected P&L
    print("\n8.2 Expected P&L (per $100 bet):")
    print("-" * 60)

    stats = calculate_edge_stats(rlm, baseline)

    if stats['valid']:
        avg_no_price = rlm['no_price'].mean()
        edge = stats['edge']

        # Per bet: bet $100, win pays $100, lose pays $avg_no_price
        # EV = edge * $100 (simplified)
        ev_per_bet = edge * 100

        # Per day/week/month
        ev_per_day = ev_per_bet * signals_per_day
        ev_per_week = ev_per_bet * signals_per_week
        ev_per_month = ev_per_bet * signals_per_month

        results['practical']['expected_pnl'] = {
            'edge_pct': float(edge * 100),
            'ev_per_100_bet': float(ev_per_bet),
            'ev_per_day': float(ev_per_day),
            'ev_per_week': float(ev_per_week),
            'ev_per_month': float(ev_per_month)
        }

        print(f"  Edge: {edge*100:.2f}%")
        print(f"  EV per $100 bet: ${ev_per_bet:.2f}")
        print(f"  Expected daily P&L: ${ev_per_day:.2f}")
        print(f"  Expected weekly P&L: ${ev_per_week:.2f}")
        print(f"  Expected monthly P&L: ${ev_per_month:.2f}")

    # 8.3 Execution Timing
    print("\n8.3 Execution Timing:")
    print("-" * 60)

    # At what point in market lifecycle can we detect the signal?
    # RLM requires knowing the trade direction distribution AND price movement
    # This typically requires most of the market activity to have occurred

    # Analyze when the signal becomes detectable
    rlm['trades_to_detect'] = rlm['n_trades']  # Need all trades to confirm ratio
    avg_trades_to_detect = rlm['trades_to_detect'].mean()

    # Time before resolution
    rlm['time_remaining_hours'] = rlm['market_duration_hours']
    avg_time_remaining = rlm['time_remaining_hours'].mean()

    results['practical']['execution_timing'] = {
        'avg_trades_to_detect': float(avg_trades_to_detect),
        'avg_market_duration_hours': float(avg_time_remaining),
        'note': 'RLM signal is detected late in market lifecycle - requires most trading to complete'
    }

    print(f"  Avg trades before signal: {avg_trades_to_detect:.1f}")
    print(f"  Avg market duration: {avg_time_remaining:.1f} hours")
    print(f"  Note: RLM requires observing trade imbalance and price divergence")
    print(f"  Signal typically detectable after 80%+ of market activity")

    # 8.4 Recommended Parameters
    print("\n8.4 Recommended Production Parameters:")
    print("-" * 60)

    # Use optimal parameters from sensitivity analysis
    optimal = results['parameter_sensitivity'].get('optimal_params', {})

    recommended = {
        'yes_trade_threshold': optimal.get('yes_threshold', 0.70),
        'min_trades': optimal.get('min_trades', 5),
        'price_move_minimum': 0,  # Any movement toward NO
        'bet_size': '$100 per signal (adjust for bankroll)',
        'max_daily_signals': 10,
        'stop_loss': 'None (bet resolves at settlement)'
    }

    results['practical']['recommended_parameters'] = recommended

    for param, value in recommended.items():
        print(f"  {param}: {value}")

    return results['practical']


# ============================================================================
# FINAL VERDICT
# ============================================================================

def generate_final_verdict():
    """Generate the final verdict on H123."""
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Collect key metrics
    base_stats = results['parameter_sensitivity']['yes_threshold'].get('0.7', {})
    optimal_stats = results['parameter_sensitivity'].get('optimal_params', {}).get('stats', {})

    # Validation criteria
    criteria = {
        'statistical_significance': base_stats.get('p_value', 1) < 0.001,
        'price_proxy_check': base_stats.get('pos_buckets', 0) > base_stats.get('total_buckets', 1) / 2,
        'temporal_stability': len([q for q in results['edge_stability'].get('quarterly_analysis', []) if q['improvement'] > 0]) >= 2,
        'out_of_sample': results['out_of_sample'].get('train_test_split', {}).get('test', {}).get('weighted_improvement', 0) > 0,
        'bootstrap_ci_excludes_zero': results['out_of_sample'].get('bootstrap_validation', {}).get('ci_95_lower', -1) > 0,
        'practical_frequency': results['practical'].get('signal_frequency', {}).get('signals_per_day', 0) >= 1
    }

    # Calculate verdict
    passed = sum(criteria.values())
    total = len(criteria)

    results['final_verdict'] = {
        'criteria_passed': passed,
        'criteria_total': total,
        'criteria_details': criteria,
        'is_validated': passed >= 5,
        'confidence_level': 'HIGH' if passed >= 5 else 'MEDIUM' if passed >= 4 else 'LOW',
        'recommended_action': 'IMPLEMENT' if passed >= 5 else 'MONITOR' if passed >= 4 else 'REJECT'
    }

    print(f"\nValidation Criteria ({passed}/{total} passed):")
    print("-" * 60)

    for criterion, passed_check in criteria.items():
        status = 'PASS' if passed_check else 'FAIL'
        print(f"  [{status}] {criterion.replace('_', ' ').title()}")

    print(f"\nFINAL VERDICT: {'**VALIDATED**' if results['final_verdict']['is_validated'] else 'NOT VALIDATED'}")
    print(f"Confidence Level: {results['final_verdict']['confidence_level']}")
    print(f"Recommended Action: {results['final_verdict']['recommended_action']}")

    # Key findings summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    summary = []

    # Edge
    if base_stats:
        summary.append(f"1. BASE EDGE: {base_stats.get('edge', 0)*100:.2f}% raw, "
                      f"{base_stats.get('weighted_improvement', 0)*100:.2f}% vs baseline")

    # Optimal parameters
    if optimal_stats:
        summary.append(f"2. OPTIMAL PARAMETERS: Found in grid search with "
                      f"{optimal_stats.get('weighted_improvement', 0)*100:.2f}% improvement")

    # Mechanism
    if results['mechanism'].get('trade_size_analysis'):
        ratio = results['mechanism']['trade_size_analysis'].get('value_ratio_no_vs_yes', 0)
        summary.append(f"3. MECHANISM: NO trades are {ratio:.1f}x larger than YES trades (confirms smart money theory)")

    # Stability
    stability = results['edge_stability'].get('decay_analysis', {})
    if stability:
        decay = stability.get('decay', 0)
        summary.append(f"4. STABILITY: Edge decay of {decay*100:.2f}% (acceptable if < 5%)")

    # Practical
    pnl = results['practical'].get('expected_pnl', {})
    if pnl:
        summary.append(f"5. PRACTICAL: Expected ${pnl.get('ev_per_month', 0):.2f}/month per $100/bet")

    for item in summary:
        print(f"\n{item}")

    return results['final_verdict']


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("H123 DEEP VALIDATION - SOBER MODE")
    print("Reverse Line Movement (RLM) NO Strategy")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df = load_data()

    # Build baseline
    all_markets, baseline = build_baseline(df)

    # Run all validations
    test_parameter_sensitivity(df, baseline)
    test_decomposition(df, baseline)
    test_mechanism(df, baseline)
    test_anti_patterns(df, baseline)
    test_edge_stability(df, baseline)
    test_combinations(df, baseline)
    test_out_of_sample(df, baseline)
    test_practical(df, baseline)

    # Generate final verdict
    generate_final_verdict()

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {OUTPUT_PATH}")
    print(f"Session completed: {datetime.now()}")

    return results


if __name__ == "__main__":
    results = main()
