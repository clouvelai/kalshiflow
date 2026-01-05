"""
H123 PRODUCTION VALIDATION - FINAL IMPLEMENTATION PARAMETERS
============================================================

This script performs final production validation of the RLM (Reverse Line Movement)
strategy with focus on:

1. CONFIRMATION: Re-validate the core strategy with strict methodology
2. OPTIMIZATION: Find optimal parameters for production deployment
3. IMPLEMENTATION: Generate exact entry/exit conditions for V3 trader
4. RISK MANAGEMENT: Establish position sizing and risk parameters

The goal is to produce actionable implementation specifications.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/h123_production_validation.json'

# Constants
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
WHALE_THRESHOLD = 10000  # $100 in cents

results = {
    'metadata': {
        'strategy': 'H123 - Reverse Line Movement (RLM) NO',
        'validation_type': 'PRODUCTION_VALIDATION',
        'timestamp': datetime.now().isoformat(),
        'p_threshold': 0.001
    },
    'core_confirmation': {},
    'parameter_optimization': {},
    'price_range_analysis': {},
    'signal_combinations': {},
    'implementation_spec': {},
    'risk_management': {}
}


def load_and_prepare_data():
    """Load trade data and prepare for analysis."""
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

    # Get unique resolved markets
    resolved_markets = df[df['market_result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved_markets)]
    print(f"Resolved markets: {len(resolved_markets):,}")

    return df_resolved


def build_baseline(df):
    """Build baseline win rates at 5c price buckets for NO bets."""
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

    print(f"\nBuilt baseline across {len(baseline)} price buckets")
    return all_markets, baseline


def get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5, require_price_move=True, price_move_threshold=0):
    """
    Identify RLM markets: Majority YES trades but price moved toward NO.

    Core RLM Signal:
    - >X% of trades are YES bets
    - But YES price dropped (indicating NO-side pressure won)
    - Minimum trade activity threshold
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean', 'std'],
        'no_price': ['mean', 'first', 'last'],
        'market_result': 'first',
        'count': ['size', 'sum', 'mean'],
        'datetime': ['first', 'last'],
        'is_whale': ['sum', 'any'],
        'trade_value_cents': ['sum', 'mean'],
        'leverage_ratio': ['mean', 'std'],
        'is_weekend': 'any',
        'is_round_size': 'sum',
        'hour': 'mean'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price', 'yes_price_std',
        'avg_no_price', 'first_no_price', 'last_no_price',
        'market_result',
        'n_trades', 'total_contracts', 'avg_trade_size',
        'first_trade_time', 'last_trade_time',
        'whale_count', 'has_whale',
        'total_value', 'avg_trade_value',
        'avg_leverage', 'lev_std',
        'has_weekend', 'round_size_count', 'avg_hour'
    ]

    # Calculate price movement
    market_stats['yes_price_moved_down'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['yes_price_drop'] = market_stats['first_yes_price'] - market_stats['last_yes_price']

    # Market duration
    market_stats['market_duration_hours'] = (
        (market_stats['last_trade_time'] - market_stats['first_trade_time']).dt.total_seconds() / 3600
    )

    # Fill NaN
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)
    market_stats['yes_price_std'] = market_stats['yes_price_std'].fillna(0)

    # Apply RLM filters
    conditions = (
        (market_stats['yes_trade_ratio'] > yes_trade_threshold) &
        (market_stats['n_trades'] >= min_trades)
    )

    if require_price_move:
        conditions = conditions & (market_stats['yes_price_moved_down'])
        if price_move_threshold > 0:
            conditions = conditions & (market_stats['yes_price_drop'] >= price_move_threshold)

    rlm = market_stats[conditions].copy()

    return rlm, market_stats


def calculate_edge_stats(signal_markets, baseline, side='no', min_markets=30):
    """Calculate comprehensive edge statistics with bucket-matched baseline comparison."""
    n = len(signal_markets)
    if n < min_markets:
        return {'n': n, 'valid': False, 'reason': f'insufficient_markets_{n}'}

    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['avg_no_price'].mean()
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['avg_no_price'] // 5) * 5
    else:
        avg_price = signal_markets['avg_yes_price'].mean()
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['avg_yes_price'] // 5) * 5

    be = avg_price / 100
    edge = wr - be

    # Statistical significance
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    # Bucket-by-bucket analysis (CRITICAL for price proxy check)
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
            sample_be = sample['avg_no_price'].mean() / 100
        else:
            sample_be = sample['avg_yes_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_be)

    ci_lower = np.percentile(bootstrap_edges, 2.5)
    ci_upper = np.percentile(bootstrap_edges, 97.5)

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
        'bucket_pct': float(pos_buckets / total_buckets) if total_buckets > 0 else 0,
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'bucket_details': improvements
    }


def confirm_core_strategy(df, baseline):
    """Confirm the core RLM strategy works as expected."""
    print("\n" + "=" * 80)
    print("1. CORE STRATEGY CONFIRMATION")
    print("=" * 80)

    # Base RLM with standard parameters
    rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5)
    base_stats = calculate_edge_stats(rlm, baseline)

    print(f"\n--- Base RLM (70% YES, 5+ trades) ---")
    if base_stats['valid']:
        print(f"  Markets: {base_stats['n']:,}")
        print(f"  Win Rate: {base_stats['win_rate']*100:.1f}%")
        print(f"  Avg NO Price: {base_stats['avg_price']:.1f}c")
        print(f"  Breakeven: {base_stats['breakeven']*100:.1f}%")
        print(f"  RAW EDGE: +{base_stats['edge']*100:.2f}%")
        print(f"  IMPROVEMENT vs Baseline: +{base_stats['weighted_improvement']*100:.2f}%")
        print(f"  P-value: {base_stats['p_value']:.2e}")
        print(f"  Bucket Analysis: {base_stats['bucket_ratio']} positive ({base_stats['bucket_pct']*100:.1f}%)")
        print(f"  95% CI: [{base_stats['ci_95_lower']*100:.2f}%, {base_stats['ci_95_upper']*100:.2f}%]")

    results['core_confirmation'] = {
        'base_rlm': base_stats,
        'validated': base_stats.get('valid', False) and
                     base_stats.get('p_value', 1) < 0.001 and
                     base_stats.get('bucket_pct', 0) > 0.8 and
                     base_stats.get('ci_95_lower', -1) > 0
    }

    # Validation status
    is_validated = results['core_confirmation']['validated']
    print(f"\n  VALIDATION STATUS: {'CONFIRMED' if is_validated else 'FAILED'}")

    return base_stats


def optimize_parameters(df, baseline):
    """Find optimal parameter combinations for production."""
    print("\n" + "=" * 80)
    print("2. PARAMETER OPTIMIZATION")
    print("=" * 80)

    results['parameter_optimization'] = {
        'grid_search': [],
        'optimal_params': None,
        'sensitivity': {}
    }

    # Grid search over key parameters
    print("\n--- Grid Search ---")

    best_sharpe = -999
    best_params = None

    for yes_thresh in [0.65, 0.70, 0.75, 0.80]:
        for min_trades in [5, 7, 10, 15]:
            for price_move in [0, 3, 5]:
                rlm, _ = get_rlm_markets(
                    df,
                    yes_trade_threshold=yes_thresh,
                    min_trades=min_trades,
                    price_move_threshold=price_move
                )

                stats = calculate_edge_stats(rlm, baseline, min_markets=50)

                if stats['valid']:
                    # Calculate Sharpe-like ratio: improvement / std of improvements
                    # Proxy: weighted_improvement / (1 - bucket_pct)
                    sharpe_proxy = stats['weighted_improvement'] / (0.1 + (1 - stats['bucket_pct']))

                    result = {
                        'yes_threshold': yes_thresh,
                        'min_trades': min_trades,
                        'price_move': price_move,
                        'n': stats['n'],
                        'edge': stats['edge'],
                        'improvement': stats['weighted_improvement'],
                        'bucket_pct': stats['bucket_pct'],
                        'sharpe_proxy': sharpe_proxy
                    }
                    results['parameter_optimization']['grid_search'].append(result)

                    if stats['n'] >= 100 and sharpe_proxy > best_sharpe:
                        best_sharpe = sharpe_proxy
                        best_params = {
                            'yes_threshold': yes_thresh,
                            'min_trades': min_trades,
                            'price_move': price_move,
                            'stats': stats
                        }

    if best_params:
        results['parameter_optimization']['optimal_params'] = best_params
        print(f"\n  OPTIMAL PARAMETERS:")
        print(f"    YES threshold: {best_params['yes_threshold']*100:.0f}%")
        print(f"    Min trades: {best_params['min_trades']}")
        print(f"    Price move: {best_params['price_move']}c")
        print(f"    Markets: {best_params['stats']['n']}")
        print(f"    Edge: +{best_params['stats']['edge']*100:.2f}%")
        print(f"    Improvement: +{best_params['stats']['weighted_improvement']*100:.2f}%")
        print(f"    Bucket coverage: {best_params['stats']['bucket_pct']*100:.1f}%")

    # Sensitivity analysis
    print("\n--- Parameter Sensitivity ---")

    # YES threshold sensitivity
    print("\n  YES Trade Threshold:")
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        rlm, _ = get_rlm_markets(df, yes_trade_threshold=thresh, min_trades=5)
        stats = calculate_edge_stats(rlm, baseline, min_markets=50)
        if stats['valid']:
            results['parameter_optimization']['sensitivity'][f'yes_{thresh}'] = stats
            print(f"    {thresh*100:.0f}%: N={stats['n']:,}, Edge=+{stats['edge']*100:.2f}%, "
                  f"Imp=+{stats['weighted_improvement']*100:.2f}%, Buckets={stats['bucket_ratio']}")

    return results['parameter_optimization']


def analyze_price_ranges(df, baseline):
    """Analyze edge across different NO price ranges."""
    print("\n" + "=" * 80)
    print("3. PRICE RANGE ANALYSIS")
    print("=" * 80)

    rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5)

    results['price_range_analysis'] = {}

    # Define price ranges
    price_ranges = [
        ('Very Low (0-30c)', (0, 30)),
        ('Low (30-50c)', (30, 50)),
        ('Mid-Low (50-65c)', (50, 65)),
        ('Mid-High (65-80c)', (65, 80)),
        ('High (80-90c)', (80, 90)),
        ('Very High (90-100c)', (90, 100))
    ]

    print("\n--- Edge by NO Price Range ---")

    for name, (low, high) in price_ranges:
        subset = rlm[(rlm['avg_no_price'] >= low) & (rlm['avg_no_price'] < high)]

        if len(subset) >= 30:
            stats = calculate_edge_stats(subset, baseline, min_markets=30)

            if stats['valid']:
                results['price_range_analysis'][name] = {
                    'range': f"{low}-{high}c",
                    'n': stats['n'],
                    'edge': stats['edge'],
                    'improvement': stats['weighted_improvement'],
                    'win_rate': stats['win_rate'],
                    'bucket_pct': stats['bucket_pct']
                }

                status = 'STRONG' if stats['weighted_improvement'] > 0.1 else 'MODERATE' if stats['weighted_improvement'] > 0.05 else 'WEAK'
                print(f"  {name}: N={stats['n']}, Edge=+{stats['edge']*100:.2f}%, "
                      f"Imp=+{stats['weighted_improvement']*100:.2f}%, [{status}]")

    return results['price_range_analysis']


def test_signal_combinations(df, baseline):
    """Test RLM combined with other validated signals."""
    print("\n" + "=" * 80)
    print("4. SIGNAL COMBINATIONS")
    print("=" * 80)

    rlm, all_stats = get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5)

    results['signal_combinations'] = {}

    # Test combinations
    combinations = []

    # 1. Base RLM
    base_stats = calculate_edge_stats(rlm, baseline)
    combinations.append({
        'name': 'Base RLM',
        'description': '>70% YES trades, YES price dropped',
        'stats': base_stats
    })

    # 2. RLM + S013 (Low Leverage Variance)
    rlm_s013 = rlm[rlm['lev_std'] < 0.7]
    s013_stats = calculate_edge_stats(rlm_s013, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + S013',
        'description': 'RLM + leverage_std < 0.7',
        'stats': s013_stats
    })

    # 3. RLM + Whale
    rlm_whale = rlm[rlm['has_whale'] == True]
    whale_stats = calculate_edge_stats(rlm_whale, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + Whale',
        'description': 'RLM + at least one whale trade',
        'stats': whale_stats
    })

    # 4. RLM + Large Price Move
    rlm_large_move = rlm[rlm['yes_price_drop'] >= 5]
    large_move_stats = calculate_edge_stats(rlm_large_move, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + Large Move (5c+)',
        'description': 'RLM + YES price dropped 5c+',
        'stats': large_move_stats
    })

    # 5. RLM + Weekend
    rlm_weekend = rlm[rlm['has_weekend'] == True]
    weekend_stats = calculate_edge_stats(rlm_weekend, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + Weekend',
        'description': 'RLM + any weekend trading',
        'stats': weekend_stats
    })

    # 6. RLM + Round Sizes (bot pattern)
    rlm_round = rlm[rlm['round_size_count'] >= 3]
    round_stats = calculate_edge_stats(rlm_round, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + Round Sizes',
        'description': 'RLM + 3+ round-size trades',
        'stats': round_stats
    })

    # 7. RLM + Strong Imbalance (80%+ YES trades)
    rlm_strong = rlm[rlm['yes_trade_ratio'] >= 0.8]
    strong_stats = calculate_edge_stats(rlm_strong, baseline, min_markets=30)
    combinations.append({
        'name': 'RLM + Strong Imbalance',
        'description': 'RLM + >80% YES trades',
        'stats': strong_stats
    })

    # 8. Triple Stack: RLM + S013 + Whale
    rlm_triple = rlm[(rlm['lev_std'] < 0.7) & (rlm['has_whale'] == True)]
    triple_stats = calculate_edge_stats(rlm_triple, baseline, min_markets=20)
    combinations.append({
        'name': 'RLM + S013 + Whale (Triple)',
        'description': 'RLM + lev_std<0.7 + whale',
        'stats': triple_stats
    })

    print("\n--- Combination Results ---")
    print(f"{'Combination':<30} {'N':>6} {'Edge':>8} {'Improve':>10} {'Buckets':>10}")
    print("-" * 70)

    for combo in combinations:
        stats = combo['stats']
        if stats['valid']:
            results['signal_combinations'][combo['name']] = {
                'description': combo['description'],
                'n': stats['n'],
                'edge': stats['edge'],
                'improvement': stats['weighted_improvement'],
                'bucket_pct': stats['bucket_pct'],
                'ci_lower': stats['ci_95_lower'],
                'ci_upper': stats['ci_95_upper']
            }
            print(f"{combo['name']:<30} {stats['n']:>6} {stats['edge']*100:>7.2f}% "
                  f"{stats['weighted_improvement']*100:>9.2f}% {stats['bucket_ratio']:>10}")

    # Find best combination
    valid_combos = [(c['name'], c['stats']) for c in combinations if c['stats']['valid'] and c['stats']['n'] >= 50]
    if valid_combos:
        best = max(valid_combos, key=lambda x: x[1]['weighted_improvement'])
        results['signal_combinations']['best'] = best[0]
        print(f"\n  BEST COMBINATION: {best[0]} (+{best[1]['weighted_improvement']*100:.2f}% improvement)")

    return results['signal_combinations']


def temporal_validation(df, baseline):
    """Validate edge stability across time periods."""
    print("\n" + "=" * 80)
    print("5. TEMPORAL VALIDATION")
    print("=" * 80)

    rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5)
    rlm['first_trade_date'] = pd.to_datetime(rlm['first_trade_time']).dt.date

    results['temporal_validation'] = {
        'quarterly': [],
        'train_test': {}
    }

    # Sort by date
    rlm_sorted = rlm.sort_values('first_trade_date')

    # Quarterly analysis
    min_date = rlm_sorted['first_trade_date'].min()
    max_date = rlm_sorted['first_trade_date'].max()
    date_range = (max_date - min_date).days

    if date_range > 0:
        quarter_days = max(1, date_range // 4)

        print("\n--- Quarterly Analysis ---")

        for q in range(4):
            q_start = min_date + timedelta(days=q * quarter_days)
            q_end = min_date + timedelta(days=(q + 1) * quarter_days)

            q_markets = rlm_sorted[
                (rlm_sorted['first_trade_date'] >= q_start) &
                (rlm_sorted['first_trade_date'] < q_end)
            ]

            if len(q_markets) >= 30:
                stats = calculate_edge_stats(q_markets, baseline, min_markets=30)
                if stats['valid']:
                    results['temporal_validation']['quarterly'].append({
                        'quarter': q + 1,
                        'start': str(q_start),
                        'end': str(q_end),
                        'n': stats['n'],
                        'edge': stats['edge'],
                        'improvement': stats['weighted_improvement']
                    })
                    status = 'PASS' if stats['weighted_improvement'] > 0 else 'FAIL'
                    print(f"  Q{q+1}: N={stats['n']}, Edge=+{stats['edge']*100:.2f}%, "
                          f"Imp=+{stats['weighted_improvement']*100:.2f}% [{status}]")

    # 80/20 Train/Test split
    print("\n--- Train/Test Split (80/20) ---")

    split_idx = int(len(rlm_sorted) * 0.8)
    train = rlm_sorted.iloc[:split_idx]
    test = rlm_sorted.iloc[split_idx:]

    train_stats = calculate_edge_stats(train, baseline, min_markets=50)
    test_stats = calculate_edge_stats(test, baseline, min_markets=30)

    if train_stats['valid'] and test_stats['valid']:
        results['temporal_validation']['train_test'] = {
            'train': {
                'n': train_stats['n'],
                'edge': train_stats['edge'],
                'improvement': train_stats['weighted_improvement']
            },
            'test': {
                'n': test_stats['n'],
                'edge': test_stats['edge'],
                'improvement': test_stats['weighted_improvement']
            },
            'generalization_gap': train_stats['weighted_improvement'] - test_stats['weighted_improvement']
        }

        gap = train_stats['weighted_improvement'] - test_stats['weighted_improvement']
        print(f"  TRAIN: N={train_stats['n']}, Edge=+{train_stats['edge']*100:.2f}%, "
              f"Imp=+{train_stats['weighted_improvement']*100:.2f}%")
        print(f"  TEST:  N={test_stats['n']}, Edge=+{test_stats['edge']*100:.2f}%, "
              f"Imp=+{test_stats['weighted_improvement']*100:.2f}%")
        print(f"  Gap: {gap*100:.2f}% {'(acceptable)' if gap < 0.05 else '(WARNING: possible overfit)'}")

    # Count positive quarters
    pos_quarters = sum(1 for q in results['temporal_validation']['quarterly'] if q['improvement'] > 0)
    total_quarters = len(results['temporal_validation']['quarterly'])

    print(f"\n  TEMPORAL STABILITY: {pos_quarters}/{total_quarters} quarters positive")

    return results['temporal_validation']


def generate_implementation_spec(df, baseline):
    """Generate final implementation specification for V3 trader."""
    print("\n" + "=" * 80)
    print("6. IMPLEMENTATION SPECIFICATION")
    print("=" * 80)

    # Get optimal parameters
    opt = results.get('parameter_optimization', {}).get('optimal_params', {})

    # Default to confirmed working parameters if optimization failed
    yes_thresh = opt.get('yes_threshold', 0.70)
    min_trades = opt.get('min_trades', 5)

    # Final strategy validation with optimal params
    rlm, _ = get_rlm_markets(df, yes_trade_threshold=yes_thresh, min_trades=min_trades)
    final_stats = calculate_edge_stats(rlm, baseline)

    spec = {
        'strategy_id': 'S-RLM-001',
        'strategy_name': 'Reverse Line Movement NO',
        'hypothesis_id': 'H123',

        'signal_conditions': {
            'yes_trade_ratio': f'> {yes_thresh}',
            'yes_trade_ratio_value': yes_thresh,
            'price_movement': 'last_yes_price < first_yes_price',
            'min_trades': min_trades,
            'action': 'BET NO'
        },

        'entry_logic': f"""
def detect_rlm_signal(market_trades):
    '''
    Detect RLM signal from trade stream.

    Returns True if:
    1. >{yes_thresh*100:.0f}% of trades are YES bets
    2. YES price dropped (last < first)
    3. At least {min_trades} trades
    '''
    if len(market_trades) < {min_trades}:
        return False

    yes_trades = sum(1 for t in market_trades if t.taker_side == 'yes')
    yes_ratio = yes_trades / len(market_trades)

    if yes_ratio <= {yes_thresh}:
        return False

    first_yes_price = market_trades[0].yes_price
    last_yes_price = market_trades[-1].yes_price

    if last_yes_price >= first_yes_price:
        return False

    return True
""",

        'expected_performance': {
            'edge': f"+{final_stats['edge']*100:.2f}%" if final_stats['valid'] else 'N/A',
            'improvement_vs_baseline': f"+{final_stats['weighted_improvement']*100:.2f}%" if final_stats['valid'] else 'N/A',
            'win_rate': f"{final_stats['win_rate']*100:.1f}%" if final_stats['valid'] else 'N/A',
            'avg_no_price': f"{final_stats['avg_price']:.1f}c" if final_stats['valid'] else 'N/A',
            'markets_tested': final_stats['n'] if final_stats['valid'] else 0,
            'bucket_coverage': final_stats['bucket_ratio'] if final_stats['valid'] else 'N/A',
            'ci_95': f"[{final_stats['ci_95_lower']*100:.2f}%, {final_stats['ci_95_upper']*100:.2f}%]" if final_stats['valid'] else 'N/A'
        },

        'optimal_price_ranges': [],

        'signal_combinations': {
            'recommended': 'Base RLM alone shows strong edge',
            'enhancement': 'RLM + S013 (low leverage variance) may provide additional confirmation'
        },

        'risk_parameters': {
            'max_position_per_market': '$100',
            'max_concurrent_positions': 10,
            'stop_loss': 'Hold to settlement (binary outcome)',
            'confidence_level': 'HIGH' if final_stats.get('ci_95_lower', -1) > 0.10 else 'MEDIUM'
        }
    }

    # Add price range recommendations
    for name, data in results.get('price_range_analysis', {}).items():
        if isinstance(data, dict) and data.get('improvement', 0) > 0.05:
            spec['optimal_price_ranges'].append({
                'range': name,
                'improvement': f"+{data['improvement']*100:.2f}%",
                'recommendation': 'PRIORITIZE' if data['improvement'] > 0.10 else 'INCLUDE'
            })

    results['implementation_spec'] = spec

    print("\n--- FINAL IMPLEMENTATION SPEC ---")
    print(f"\nStrategy: {spec['strategy_name']} (ID: {spec['strategy_id']})")
    print(f"\nSignal Conditions:")
    print(f"  - YES trade ratio > {yes_thresh*100:.0f}%")
    print(f"  - YES price dropped (last < first)")
    print(f"  - Minimum {min_trades} trades")
    print(f"  - ACTION: Bet NO")

    print(f"\nExpected Performance:")
    for key, value in spec['expected_performance'].items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")

    print(f"\nRisk Parameters:")
    for key, value in spec['risk_parameters'].items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")

    return spec


def calculate_risk_metrics(df, baseline):
    """Calculate risk management parameters."""
    print("\n" + "=" * 80)
    print("7. RISK MANAGEMENT ANALYSIS")
    print("=" * 80)

    rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5)
    rlm['first_trade_date'] = pd.to_datetime(rlm['first_trade_time']).dt.date

    # Calculate basic risk metrics
    win_rate = (rlm['market_result'] == 'no').mean()
    avg_no_price = rlm['avg_no_price'].mean()

    # Simulate P&L distribution
    n_simulations = 10000
    n_bets = 100  # Per simulation

    pnl_results = []
    for _ in range(n_simulations):
        # Sample markets with replacement
        sample = rlm.sample(n=n_bets, replace=True)

        # Calculate P&L: Win = 100 - no_price, Lose = -no_price
        wins = (sample['market_result'] == 'no').values
        no_prices = sample['avg_no_price'].values

        pnl = sum((100 - p) if w else -p for w, p in zip(wins, no_prices))
        pnl_results.append(pnl)

    pnl_array = np.array(pnl_results)

    risk_metrics = {
        'win_rate': float(win_rate),
        'avg_no_price': float(avg_no_price),

        'per_100_bets': {
            'mean_pnl': float(np.mean(pnl_array)),
            'median_pnl': float(np.median(pnl_array)),
            'std_pnl': float(np.std(pnl_array)),
            'pct_5_pnl': float(np.percentile(pnl_array, 5)),
            'pct_95_pnl': float(np.percentile(pnl_array, 95)),
            'max_drawdown': float(np.min(pnl_array)),
            'prob_profit': float(np.mean(pnl_array > 0))
        },

        'kelly_criterion': {
            'optimal_fraction': float((win_rate - (1-win_rate)/(avg_no_price/(100-avg_no_price)))),
            'note': 'Use fractional Kelly (0.25x) for safety'
        },

        'recommended_sizing': {
            'conservative': '$50/bet (0.25x Kelly)',
            'moderate': '$100/bet (0.5x Kelly)',
            'aggressive': '$200/bet (1x Kelly)'
        }
    }

    results['risk_management'] = risk_metrics

    print(f"\n--- Risk Metrics (Based on {len(rlm):,} historical signals) ---")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")

    print(f"\n--- P&L Distribution (Per 100 bets of $1 each) ---")
    print(f"  Mean P&L: ${risk_metrics['per_100_bets']['mean_pnl']:.2f}")
    print(f"  Median P&L: ${risk_metrics['per_100_bets']['median_pnl']:.2f}")
    print(f"  Std Dev: ${risk_metrics['per_100_bets']['std_pnl']:.2f}")
    print(f"  5th Percentile (bad scenario): ${risk_metrics['per_100_bets']['pct_5_pnl']:.2f}")
    print(f"  95th Percentile (good scenario): ${risk_metrics['per_100_bets']['pct_95_pnl']:.2f}")
    print(f"  Probability of Profit: {risk_metrics['per_100_bets']['prob_profit']*100:.1f}%")

    print(f"\n--- Position Sizing ---")
    print(f"  Kelly Optimal: {risk_metrics['kelly_criterion']['optimal_fraction']*100:.1f}% of bankroll")
    print(f"  Recommended: Use 0.25x Kelly (conservative) to 0.5x Kelly (moderate)")

    return risk_metrics


def generate_final_summary():
    """Generate final summary and verdict."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Compile validation checklist
    core = results.get('core_confirmation', {})
    temporal = results.get('temporal_validation', {})

    checklist = {
        'statistical_significance': core.get('base_rlm', {}).get('p_value', 1) < 0.001,
        'not_price_proxy': core.get('base_rlm', {}).get('bucket_pct', 0) > 0.8,
        'ci_excludes_zero': core.get('base_rlm', {}).get('ci_95_lower', -1) > 0,
        'temporal_stability': len([q for q in temporal.get('quarterly', []) if q.get('improvement', 0) > 0]) >= 2,
        'out_of_sample': temporal.get('train_test', {}).get('test', {}).get('improvement', -1) > 0,
        'sufficient_samples': core.get('base_rlm', {}).get('n', 0) >= 500
    }

    passed = sum(checklist.values())
    total = len(checklist)

    results['final_summary'] = {
        'validation_checklist': checklist,
        'criteria_passed': f"{passed}/{total}",
        'is_production_ready': passed >= 5,
        'confidence_level': 'HIGH' if passed >= 5 else 'MEDIUM' if passed >= 4 else 'LOW',
        'recommendation': 'IMPLEMENT' if passed >= 5 else 'FURTHER_TESTING' if passed >= 4 else 'REJECT'
    }

    print(f"\n--- Validation Checklist ({passed}/{total} passed) ---")
    for criterion, passed_check in checklist.items():
        status = 'PASS' if passed_check else 'FAIL'
        print(f"  [{status}] {criterion.replace('_', ' ').title()}")

    print(f"\n--- FINAL VERDICT ---")
    print(f"  Production Ready: {'YES' if results['final_summary']['is_production_ready'] else 'NO'}")
    print(f"  Confidence Level: {results['final_summary']['confidence_level']}")
    print(f"  Recommendation: {results['final_summary']['recommendation']}")

    return results['final_summary']


def main():
    print("=" * 80)
    print("H123 PRODUCTION VALIDATION")
    print("Reverse Line Movement (RLM) NO Strategy")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data()

    # Build baseline
    all_markets, baseline = build_baseline(df)

    # Run all validation steps
    confirm_core_strategy(df, baseline)
    optimize_parameters(df, baseline)
    analyze_price_ranges(df, baseline)
    test_signal_combinations(df, baseline)
    temporal_validation(df, baseline)
    generate_implementation_spec(df, baseline)
    calculate_risk_metrics(df, baseline)
    generate_final_summary()

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\n{'='*80}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Session completed: {datetime.now()}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
