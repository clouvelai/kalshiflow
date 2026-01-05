"""
H123 CATEGORY VALIDATION - Non-Sports Market Analysis
======================================================

Validates whether the H123 RLM (Reverse Line Movement) strategy
works on non-sports market categories.

Background:
- H123 validated with +17.38% edge on 1,986 markets
- Original validation primarily on sports markets
- This script tests crypto, politics, weather, economics, entertainment

Key Question: Should the V3 trader filter to sports-only or include all markets?

Author: Quant Agent
Date: 2025-12-30
Session: H123 Category Validation
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/h123_category_validation.json'

# RLM Parameters (from validated H123)
YES_TRADE_THRESHOLD = 0.70  # >70% YES trades
MIN_TRADES = 5  # Minimum trades per market
WHALE_THRESHOLD = 10000  # $100 in cents

# Sports category patterns to EXCLUDE from non-sports analysis
SPORTS_PATTERNS = [
    r'^KXMVE',          # Esports (all variants)
    r'^KXNFL',          # NFL
    r'^KXNBA',          # NBA
    r'^KXNHL',          # NHL
    r'^KXMLB',          # MLB
    r'^KXNCAAF',        # College Football
    r'^KXNCAAMB',       # College Men's Basketball
    r'^KXNCAAWB',       # College Women's Basketball
    r'^KXSOC',          # Soccer (generic)
    r'^KXUFC',          # UFC
    r'^KXPGA',          # Golf
    r'^KXEPL',          # Premier League
    r'^KXLALIGA',       # La Liga
    r'^KXSERIEA',       # Serie A
    r'^KXBUNDESLIGA',   # Bundesliga
    r'^KXLIGUE',        # Ligue 1
    r'^KXUCL',          # Champions League
    r'^KXUEL',          # Europa League
    r'^KXUECL',         # Conference League
    r'^KXCSG',          # CS:GO
    r'^KXDOTA',         # Dota
    r'^KXCOD',          # Call of Duty
    r'^KXCHESS',        # Chess
    r'^KXTENNIS',       # Tennis
    r'^KXBOXING',       # Boxing
    r'^KXCRICKET',      # Cricket
    r'^KXEURO',         # Euro competitions
    r'^KXBRASILEIR',    # Brazilian League
    r'^KXJLEAGUE',      # J-League
    r'^KXEREDIVISIE',   # Dutch League
    r'^KXLIGAPORTU',    # Portuguese League
    r'^KXBELGIAN',      # Belgian League
    r'^KXSUPERLIG',     # Turkish Super Lig
    r'^KXSCOTTISH',     # Scottish Prem
    r'^KXALEAGUE',      # A-League
    r'^KXNBL',          # NBL
    r'^KXARGP',         # Argentine Football
    r'^KXCBA',          # CBA
    r'^KXLIGAMX',       # Liga MX
    r'^KXMLS',          # MLS
    r'^KXHNL',          # HNL
    r'^KXCFP',          # College Football Playoff
    r'^KXHEISMAN',      # Heisman
    r'^KXSURF',         # Surfing
    r'^KXPUBG',         # PUBG
    r'^KXLOL',          # League of Legends
    r'^KXSTARLADDER',   # CS Major
    r'^KXF$',           # F1 (exact match for KXF prefix)
    r'^KXEFL',          # EFL
]

# Non-sports category groupings for analysis
CATEGORY_GROUPS = {
    'Crypto': [r'^KXBTC', r'^KXETH', r'^KXDOGE', r'^KXXRP', r'^KXSHIBA'],
    'Economics': [r'^KXNASDAQ', r'^KXINX', r'^FED', r'^KXCPI', r'^KXPAYROLL', r'^KXJOBLESS', r'^KXEMPLOYMENT', r'^KXRATE', r'^KXWTI', r'^KXAAA'],
    'Politics': [r'^KXTRUMP', r'^KXAPR', r'^KXPRES', r'^KXSEC(?!P)', r'^KXVANCE', r'^KXMAYO', r'^KXSENATE', r'^KXDEPUT', r'^KXBUCHA', r'^KXHONDURAS', r'^KXARGENTINA', r'^KXCHILE', r'^KXNDAA', r'^KXWHVISIT', r'^KXLEAVE', r'^KXMAMD', r'^KXJERSEY', r'^KXELECTION'],
    'Weather': [r'^KXHIGH', r'^KXRAIN', r'^KXSNOW'],
    'Entertainment': [r'^KXNETFLIX', r'^KXSPOTIFY', r'^KXGG', r'^KXGAME(?!DAY)', r'^KXBILLBOARD', r'^KXTHRASHER', r'^KXRANK(?!LIST)', r'^KXRT', r'^KXLOVE', r'^KXTHEA', r'^KXTHEVO', r'^KXCRITICS', r'^KXSTRM', r'^KXSTOCKX'],
    'Media_Mentions': [r'^KXMRBEAST', r'^KXCOLBERT', r'^KXSNL', r'^KXLATENIGHT', r'^KXSURVIVOR', r'^KXALTMAN', r'^KXSWIFT', r'^KXFEDMENTION', r'^KXNCAAMENTION', r'^KXNBAMENTION', r'^KXAWARD', r'^KXHEGSET', r'^KXARMSTRONG', r'^KXBESSENTMT', r'^KXZAKARIA', r'^KXDIMON', r'^KXVLADTENEV', r'^KXSTARMER', r'^KXGAMEDAY', r'^KXINFANTINO', r'^KXEPSTEIN', r'^KXEARNINGS'],
    'Time_Events': [r'^KXTIME', r'^KXEO'],
}

# Results container
results = {
    'metadata': {
        'analysis': 'H123 Category Validation - Non-Sports Markets',
        'session': 'H123 Category Validation',
        'timestamp': datetime.now().isoformat(),
        'rlm_parameters': {
            'yes_trade_threshold': YES_TRADE_THRESHOLD,
            'min_trades': MIN_TRADES
        }
    },
    'data_summary': {},
    'sports_baseline': {},
    'non_sports_by_group': {},
    'non_sports_by_category': {},
    'summary_table': [],
    'final_recommendation': ''
}


def load_data():
    """Load and prepare the trade data."""
    print("=" * 80)
    print("H123 CATEGORY VALIDATION - NON-SPORTS MARKET ANALYSIS")
    print("=" * 80)
    print("\nLoading data...")

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['date'] = df['datetime'].dt.date

    # Extract category from market_ticker
    df['category'] = df['market_ticker'].str.extract(r'^([A-Z]+)', expand=False)

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    print(f"Unique categories: {df['category'].nunique()}")

    return df


def is_sports_category(category):
    """Check if category matches sports patterns."""
    if category is None:
        return False
    for pattern in SPORTS_PATTERNS:
        if re.match(pattern, category):
            return True
    return False


def get_category_group(category):
    """Get the high-level group for a category."""
    if category is None:
        return 'Other'
    for group, patterns in CATEGORY_GROUPS.items():
        for pattern in patterns:
            if re.match(pattern, category):
                return group
    return 'Other'


def classify_markets(df):
    """Classify all markets as sports vs non-sports."""
    # Get unique markets with their categories
    market_cats = df.groupby('market_ticker')['category'].first().reset_index()
    market_cats['is_sports'] = market_cats['category'].apply(is_sports_category)
    market_cats['group'] = market_cats['category'].apply(get_category_group)

    sports_markets = market_cats[market_cats['is_sports']]['market_ticker'].tolist()
    non_sports_markets = market_cats[~market_cats['is_sports']]['market_ticker'].tolist()

    print(f"\nMarket Classification:")
    print(f"  Sports markets: {len(sports_markets):,}")
    print(f"  Non-sports markets: {len(non_sports_markets):,}")

    # Group breakdown for non-sports
    group_counts = market_cats[~market_cats['is_sports']].groupby('group').size()
    print(f"\nNon-Sports by Group:")
    for group, count in group_counts.sort_values(ascending=False).items():
        print(f"  {group}: {count:,} markets")

    results['data_summary'] = {
        'total_trades': len(df),
        'total_markets': df['market_ticker'].nunique(),
        'sports_markets': len(sports_markets),
        'non_sports_markets': len(non_sports_markets),
        'non_sports_by_group': group_counts.to_dict()
    }

    return market_cats, sports_markets, non_sports_markets


def get_rlm_markets(df, market_list=None):
    """
    Detect RLM (Reverse Line Movement) signal.

    RLM Signal:
    - >70% of trades are YES (public betting YES)
    - But price moved toward NO (YES price dropped from first to last)
    - At least 5 trades
    - Action: Bet NO (smart money overpowering public)
    """
    if market_list is not None:
        df = df[df['market_ticker'].isin(market_list)]

    if len(df) == 0:
        return pd.DataFrame()

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean'],
        'no_price': 'mean',
        'market_result': 'first',
        'count': ['size', 'sum'],
        'datetime': ['first', 'last'],
        'is_whale': 'sum',
        'category': 'first'
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price',
        'avg_no_price', 'market_result',
        'n_trades', 'total_contracts',
        'first_trade_time', 'last_trade_time',
        'whale_count', 'category'
    ]

    # RLM signal: majority YES trades but price moved toward NO
    market_stats['price_moved_no'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['yes_price_drop'] = market_stats['first_yes_price'] - market_stats['last_yes_price']

    # Apply RLM filters
    rlm = market_stats[
        (market_stats['yes_trade_ratio'] > YES_TRADE_THRESHOLD) &
        (market_stats['price_moved_no']) &
        (market_stats['n_trades'] >= MIN_TRADES)
    ].copy()

    return rlm


def build_baseline(df):
    """Build baseline win rates at 5c buckets using ALL markets."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 10:
            baseline[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n_markets': n
            }

    return baseline


def calculate_edge_stats(signal_markets, baseline, min_markets=30):
    """Calculate comprehensive edge statistics with bucket-matched baseline comparison."""
    n = len(signal_markets)

    if n < min_markets:
        return {
            'n': n,
            'valid': False,
            'reason': f'insufficient_markets (need {min_markets}, have {n})'
        }

    wins = (signal_markets['market_result'] == 'no').sum()
    win_rate = wins / n
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100
    edge = win_rate - breakeven

    # Statistical significance
    std_err = np.sqrt(win_rate * (1 - win_rate) / n)
    z_score = (win_rate - breakeven) / std_err if std_err > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Bucket-matched baseline comparison
    signal_markets = signal_markets.copy()
    signal_markets['bucket_5c'] = (signal_markets['avg_no_price'] // 5) * 5

    bucket_results = []
    for bucket in signal_markets['bucket_5c'].unique():
        bucket_signal = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_bucket = len(bucket_signal)
        if n_bucket < 3:
            continue

        signal_wr = (bucket_signal['market_result'] == 'no').mean()

        if bucket in baseline:
            base_wr = baseline[bucket]['win_rate']
            improvement = signal_wr - base_wr
            bucket_results.append({
                'bucket': bucket,
                'signal_wr': signal_wr,
                'baseline_wr': base_wr,
                'improvement': improvement,
                'n': n_bucket,
                'positive': improvement > 0
            })

    if len(bucket_results) == 0:
        return {
            'n': n,
            'valid': False,
            'reason': 'no_valid_buckets'
        }

    # Calculate weighted improvement
    total_n = sum(b['n'] for b in bucket_results)
    weighted_improvement = sum(b['improvement'] * b['n'] for b in bucket_results) / total_n

    pos_buckets = sum(1 for b in bucket_results if b['positive'])
    total_buckets = len(bucket_results)
    bucket_pct = pos_buckets / total_buckets if total_buckets > 0 else 0

    # 95% CI
    ci_margin = 1.96 * std_err
    ci_lower = edge - ci_margin
    ci_upper = edge + ci_margin

    return {
        'n': n,
        'valid': True,
        'wins': wins,
        'win_rate': win_rate,
        'avg_no_price': avg_no_price,
        'breakeven': breakeven,
        'edge': edge,
        'p_value': p_value,
        'weighted_improvement': weighted_improvement,
        'pos_buckets': pos_buckets,
        'total_buckets': total_buckets,
        'bucket_ratio': f'{pos_buckets}/{total_buckets}',
        'bucket_pct': bucket_pct,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'bucket_details': bucket_results
    }


def check_temporal_stability(signal_markets):
    """Check if edge is stable across time quarters."""
    if len(signal_markets) < 40:
        return {'valid': False, 'reason': 'insufficient_data'}

    signal_markets = signal_markets.copy()
    signal_markets['first_trade_date'] = pd.to_datetime(signal_markets['first_trade_time']).dt.date

    # Sort by date and split into quarters
    sorted_markets = signal_markets.sort_values('first_trade_date')
    n = len(sorted_markets)
    q_size = n // 4

    quarters = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else n
        q_markets = sorted_markets.iloc[start:end]

        if len(q_markets) >= 5:
            wins = (q_markets['market_result'] == 'no').sum()
            wr = wins / len(q_markets)
            avg_price = q_markets['avg_no_price'].mean()
            be = avg_price / 100
            edge = wr - be
            quarters.append({
                'quarter': f'Q{i+1}',
                'n': len(q_markets),
                'win_rate': wr,
                'edge': edge,
                'positive': edge > 0
            })

    positive_quarters = sum(1 for q in quarters if q['positive'])

    return {
        'valid': True,
        'quarters': quarters,
        'positive_quarters': positive_quarters,
        'total_quarters': len(quarters),
        'stable': positive_quarters >= 2
    }


def get_verdict(stats, temporal=None):
    """Determine verdict based on validation criteria."""
    if not stats.get('valid', False):
        return 'INSUFFICIENT_DATA'

    edge = stats.get('edge', 0)
    improvement = stats.get('weighted_improvement', 0)
    bucket_pct = stats.get('bucket_pct', 0)
    p_value = stats.get('p_value', 1)
    ci_lower = stats.get('ci_95_lower', -1)

    # Check temporal stability if available
    temporal_ok = True
    if temporal and temporal.get('valid'):
        temporal_ok = temporal.get('stable', False)

    # VALID: Strong edge, not a price proxy, statistically significant
    if (edge > 0.05 and
        improvement > 0.03 and
        bucket_pct > 0.6 and
        p_value < 0.05 and
        ci_lower > 0 and
        temporal_ok):
        return 'VALID'

    # WEAK: Some edge but not robust
    if (edge > 0.02 and
        improvement > 0 and
        bucket_pct > 0.4 and
        p_value < 0.1):
        return 'WEAK_EDGE'

    # PRICE_PROXY: Raw edge but no improvement vs baseline
    if edge > 0.02 and improvement <= 0:
        return 'PRICE_PROXY'

    # INVALID: No meaningful edge
    return 'NO_EDGE'


def analyze_sports_baseline(df, sports_markets, baseline):
    """Analyze RLM performance on sports markets as baseline."""
    print("\n" + "=" * 80)
    print("SPORTS BASELINE (Reference)")
    print("=" * 80)

    rlm_sports = get_rlm_markets(df, sports_markets)

    if len(rlm_sports) == 0:
        print("  No RLM signals found in sports markets!")
        return

    stats = calculate_edge_stats(rlm_sports, baseline, min_markets=50)
    temporal = check_temporal_stability(rlm_sports)

    print(f"\n  RLM Signals: {stats.get('n', 0):,}")
    if stats.get('valid'):
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Avg NO Price: {stats['avg_no_price']:.1f}c")
        print(f"  Breakeven: {stats['breakeven']*100:.1f}%")
        print(f"  Edge: {stats['edge']*100:+.2f}%")
        print(f"  Improvement vs Baseline: {stats['weighted_improvement']*100:+.2f}%")
        print(f"  Positive Buckets: {stats['bucket_ratio']} ({stats['bucket_pct']*100:.0f}%)")
        print(f"  P-value: {stats['p_value']:.6f}")
        print(f"  95% CI: [{stats['ci_95_lower']*100:.2f}%, {stats['ci_95_upper']*100:.2f}%]")

        if temporal.get('valid'):
            print(f"  Temporal: {temporal['positive_quarters']}/{temporal['total_quarters']} quarters positive")

    verdict = get_verdict(stats, temporal)
    print(f"\n  VERDICT: {verdict}")

    results['sports_baseline'] = {
        'stats': stats,
        'temporal': temporal,
        'verdict': verdict
    }

    return stats


def analyze_category_group(df, group_name, market_list, baseline):
    """Analyze RLM performance for a category group."""
    print(f"\n--- {group_name} ---")

    rlm = get_rlm_markets(df, market_list)

    if len(rlm) < 10:
        print(f"  Insufficient RLM signals: {len(rlm)} (need 10+)")
        return {
            'n_markets': len(market_list),
            'n_rlm_signals': len(rlm),
            'verdict': 'INSUFFICIENT_DATA'
        }

    stats = calculate_edge_stats(rlm, baseline, min_markets=20)
    temporal = check_temporal_stability(rlm) if len(rlm) >= 40 else {'valid': False}

    print(f"  Total markets: {len(market_list):,}")
    print(f"  RLM Signals: {stats.get('n', 0)}")

    if stats.get('valid'):
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Avg NO Price: {stats['avg_no_price']:.1f}c")
        print(f"  Edge: {stats['edge']*100:+.2f}%")
        print(f"  Improvement: {stats['weighted_improvement']*100:+.2f}%")
        print(f"  Positive Buckets: {stats['bucket_ratio']}")
        print(f"  P-value: {stats['p_value']:.4f}")
    else:
        print(f"  Stats: {stats.get('reason', 'invalid')}")

    verdict = get_verdict(stats, temporal)
    print(f"  VERDICT: {verdict}")

    return {
        'n_markets': len(market_list),
        'n_rlm_signals': stats.get('n', 0),
        'stats': stats if stats.get('valid') else None,
        'temporal': temporal if temporal.get('valid') else None,
        'verdict': verdict
    }


def analyze_individual_categories(df, non_sports_markets, baseline, top_n=15):
    """Analyze RLM for top individual non-sports categories."""
    print("\n" + "=" * 80)
    print("TOP INDIVIDUAL NON-SPORTS CATEGORIES")
    print("=" * 80)

    # Get category counts for non-sports
    market_cats = df[df['market_ticker'].isin(non_sports_markets)].groupby('market_ticker')['category'].first()
    cat_counts = market_cats.value_counts()

    top_cats = cat_counts.head(top_n)

    cat_results = {}
    for cat, count in top_cats.items():
        cat_markets = market_cats[market_cats == cat].index.tolist()
        rlm = get_rlm_markets(df, cat_markets)

        if len(rlm) < 5:
            continue

        stats = calculate_edge_stats(rlm, baseline, min_markets=10)
        verdict = get_verdict(stats)

        print(f"\n{cat}: {count} markets, {len(rlm)} RLM signals")
        if stats.get('valid'):
            print(f"  Edge: {stats['edge']*100:+.2f}%, Improvement: {stats['weighted_improvement']*100:+.2f}%, Buckets: {stats['bucket_ratio']}")
            print(f"  VERDICT: {verdict}")

        cat_results[cat] = {
            'n_markets': count,
            'n_rlm': len(rlm),
            'stats': stats if stats.get('valid') else None,
            'verdict': verdict
        }

    results['non_sports_by_category'] = cat_results
    return cat_results


def generate_summary():
    """Generate final summary table and recommendation."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    summary_rows = []

    # Sports baseline
    sports = results.get('sports_baseline', {})
    if sports.get('stats', {}).get('valid'):
        s = sports['stats']
        summary_rows.append({
            'Group': 'SPORTS (Baseline)',
            'Markets': s['n'],
            'Win_Rate': f"{s['win_rate']*100:.1f}%",
            'Edge': f"{s['edge']*100:+.1f}%",
            'Improvement': f"{s['weighted_improvement']*100:+.1f}%",
            'Buckets': s['bucket_ratio'],
            'P_value': f"{s['p_value']:.4f}",
            'Verdict': sports['verdict']
        })

    # Non-sports groups
    for group, data in results.get('non_sports_by_group', {}).items():
        if data.get('stats'):
            s = data['stats']
            summary_rows.append({
                'Group': group,
                'Markets': s['n'],
                'Win_Rate': f"{s['win_rate']*100:.1f}%",
                'Edge': f"{s['edge']*100:+.1f}%",
                'Improvement': f"{s['weighted_improvement']*100:+.1f}%",
                'Buckets': s['bucket_ratio'],
                'P_value': f"{s['p_value']:.4f}",
                'Verdict': data['verdict']
            })
        else:
            summary_rows.append({
                'Group': group,
                'Markets': data.get('n_rlm_signals', 0),
                'Win_Rate': 'N/A',
                'Edge': 'N/A',
                'Improvement': 'N/A',
                'Buckets': 'N/A',
                'P_value': 'N/A',
                'Verdict': data['verdict']
            })

    # Print summary table
    print(f"\n{'Group':<20} {'N':>6} {'WinRate':>8} {'Edge':>8} {'Improv':>8} {'Buckets':>8} {'P-val':>8} {'Verdict':<15}")
    print("-" * 100)
    for row in summary_rows:
        print(f"{row['Group']:<20} {row['Markets']:>6} {row['Win_Rate']:>8} {row['Edge']:>8} {row['Improvement']:>8} {row['Buckets']:>8} {row['P_value']:>8} {row['Verdict']:<15}")

    results['summary_table'] = summary_rows

    # Generate recommendation
    valid_groups = [r for r in summary_rows if r['Verdict'] == 'VALID']
    weak_groups = [r for r in summary_rows if r['Verdict'] == 'WEAK_EDGE']

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    if len(valid_groups) > 1:  # More than just sports
        recommendation = "INCLUDE_SELECTED: RLM shows valid edge in multiple category groups"
        valid_list = [r['Group'] for r in valid_groups]
        print(f"\n  Valid groups: {', '.join(valid_list)}")
    elif len(weak_groups) > 0:
        recommendation = "SPORTS_PRIMARY_WEAK_SECONDARY: Sports-only primary, consider weak edge categories for diversification"
        weak_list = [r['Group'] for r in weak_groups]
        print(f"\n  Weak edge groups (use cautiously): {', '.join(weak_list)}")
    else:
        recommendation = "SPORTS_ONLY: RLM edge is sports-specific, filter to sports markets only"

    print(f"\n  RECOMMENDATION: {recommendation}")
    results['final_recommendation'] = recommendation

    return summary_rows


def main():
    """Main execution function."""
    # Load data
    df = load_data()

    # Build global baseline
    print("\nBuilding baseline...")
    baseline = build_baseline(df)
    print(f"Baseline built with {len(baseline)} price buckets")

    # Classify markets
    market_cats, sports_markets, non_sports_markets = classify_markets(df)

    # Analyze sports baseline
    analyze_sports_baseline(df, sports_markets, baseline)

    # Analyze non-sports by group
    print("\n" + "=" * 80)
    print("NON-SPORTS CATEGORY GROUPS")
    print("=" * 80)

    for group, patterns in CATEGORY_GROUPS.items():
        # Get markets matching this group
        group_markets = []
        for market in non_sports_markets:
            cat = market_cats[market_cats['market_ticker'] == market]['category'].iloc[0] if len(market_cats[market_cats['market_ticker'] == market]) > 0 else None
            if cat and get_category_group(cat) == group:
                group_markets.append(market)

        if len(group_markets) > 0:
            result = analyze_category_group(df, group, group_markets, baseline)
            results['non_sports_by_group'][group] = result

    # Analyze individual categories
    analyze_individual_categories(df, non_sports_markets, baseline)

    # Generate summary
    generate_summary()

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nAnalysis complete!")

    return results


if __name__ == '__main__':
    main()
