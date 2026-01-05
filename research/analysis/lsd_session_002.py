"""
LSD SESSION 002: Exploiting the 5 Core Principles Through Novel Signals

"RLM works, but others will compete for it. We need PROPRIETARY edge."

THE 5 CORE PRINCIPLES:
1. CAPITAL WEIGHT vs TRADE COUNT - Smart money speaks in dollars
2. PUBLIC SENTIMENT vs CAPITAL CONVICTION - Retail overweights outcome confidence
3. PRICE DISCOVERY DELAY - Smart money moves before the crowd
4. SYSTEMATIC vs RANDOM BEHAVIOR - Informed traders have consistent patterns
5. UNCERTAINTY PREMIUM - Largest mispricings at highest uncertainty

15 NOVEL HYPOTHESES TO TEST - ALL NEVER TESTED BEFORE
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
RESULTS_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/lsd_session_002_results.json'

results = {}


def load_data():
    """Load the enriched trades data."""
    print("=" * 80)
    print("LSD SESSION 002: LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Core columns
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000  # >= $100
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_no'] = df['taker_side'] == 'no'
    df['is_yes'] = df['taker_side'] == 'yes'

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def quick_edge_check(signal_markets, side='no', name=""):
    """Fast edge check. Flag if edge > 5%."""
    n = len(signal_markets)
    if n < 30:
        return {'name': name, 'n': n, 'edge': None, 'flag': False, 'reason': 'insufficient_data'}

    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['no_price'].mean()
    else:
        avg_price = signal_markets['yes_price'].mean()

    be = avg_price / 100
    edge = wr - be

    # P-value
    if 0 < be < 1:
        z = (wins - n * be) / np.sqrt(n * be * (1 - be))
        p_value = 1 - stats.norm.cdf(z)
    else:
        p_value = 1.0

    flag = edge > 0.05 and p_value < 0.05

    return {
        'name': name,
        'n': n,
        'win_rate': round(float(wr), 4),
        'avg_price': round(float(avg_price), 2),
        'breakeven': round(float(be), 4),
        'edge': round(float(edge), 4),
        'p_value': float(p_value),
        'flag': flag,
        'verdict': 'PROMISING' if flag else ('WEAK' if edge > 0.02 else 'DEAD')
    }


# =============================================================================
# PRINCIPLE 1: CAPITAL WEIGHT vs TRADE COUNT
# =============================================================================

def test_h207_dollar_weighted_direction(df):
    """
    H-LSD-207: Dollar-Weighted Direction

    Signal: Dollar-weighted YES% differs from trade-count YES% by >20%
    If dollars favor NO more than trades do, bet NO.

    CORE PRINCIPLE: Smart money speaks in DOLLARS.
    """
    print("\n" + "=" * 60)
    print("H-LSD-207: DOLLAR-WEIGHTED DIRECTION (P1)")
    print("=" * 60)

    market_stats = df.groupby('market_ticker').agg({
        'is_yes': ['sum', 'count'],  # Trade count YES
        'trade_value_cents': 'sum',
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    })
    market_stats.columns = ['yes_trades', 'total_trades', 'total_value', 'market_result', 'no_price', 'yes_price']
    market_stats = market_stats.reset_index()

    # Calculate YES trade ratio
    market_stats['yes_trade_ratio'] = market_stats['yes_trades'] / market_stats['total_trades']

    # Calculate dollar-weighted YES ratio
    yes_value = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].sum()
    market_stats['yes_value'] = market_stats['market_ticker'].map(yes_value).fillna(0)
    market_stats['yes_dollar_ratio'] = market_stats['yes_value'] / market_stats['total_value']

    # Divergence: trades say YES but dollars say NO
    market_stats['divergence'] = market_stats['yes_trade_ratio'] - market_stats['yes_dollar_ratio']

    # Signal: Trades favor YES by >20% more than dollars do
    # (means dollars are going to NO)
    signal_no = market_stats[
        (market_stats['divergence'] > 0.20) &
        (market_stats['total_trades'] >= 5)
    ]

    # Inverse: Dollars favor YES more than trades
    signal_yes = market_stats[
        (market_stats['divergence'] < -0.20) &
        (market_stats['total_trades'] >= 5)
    ]

    result_no = quick_edge_check(signal_no, side='no', name="Dollar favors NO (bet NO)")
    result_yes = quick_edge_check(signal_yes, side='yes', name="Dollar favors YES (bet YES)")

    print(f"Trades favor YES but dollars favor NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Trades favor NO but dollars favor YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    # Also test more extreme divergence
    extreme_no = market_stats[
        (market_stats['divergence'] > 0.30) &
        (market_stats['total_trades'] >= 5)
    ]
    result_extreme = quick_edge_check(extreme_no, side='no', name="Extreme divergence (30%+)")
    print(f"Extreme divergence (>30%): N={result_extreme['n']}, Edge={result_extreme.get('edge', 'N/A')}")

    return {'dollar_no': result_no, 'dollar_yes': result_yes, 'extreme': result_extreme}


def test_h212_trade_count_vs_dollar_imbalance(df):
    """
    H-LSD-212: Trade Count vs Dollar Imbalance

    Signal: >70% YES trades but dollar-weighted < 60% YES
    Smart money quietly betting NO while retail is loud on YES.

    CORE PRINCIPLE: Sentiment (trade count) vs Conviction (dollars)
    """
    print("\n" + "=" * 60)
    print("H-LSD-212: TRADE COUNT vs DOLLAR IMBALANCE (P2)")
    print("=" * 60)

    market_stats = df.groupby('market_ticker').agg({
        'is_yes': ['sum', 'count'],
        'trade_value_cents': 'sum',
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    })
    market_stats.columns = ['yes_trades', 'total_trades', 'total_value', 'market_result', 'no_price', 'yes_price']
    market_stats = market_stats.reset_index()

    market_stats['yes_trade_ratio'] = market_stats['yes_trades'] / market_stats['total_trades']

    yes_value = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].sum()
    market_stats['yes_value'] = market_stats['market_ticker'].map(yes_value).fillna(0)
    market_stats['yes_dollar_ratio'] = market_stats['yes_value'] / market_stats['total_value']

    # Signal: >70% YES trades but <60% YES dollars (smart money on NO)
    signal_no = market_stats[
        (market_stats['yes_trade_ratio'] > 0.70) &
        (market_stats['yes_dollar_ratio'] < 0.60) &
        (market_stats['total_trades'] >= 5)
    ]

    # Inverse: >70% NO trades but <60% NO dollars
    signal_yes = market_stats[
        (market_stats['yes_trade_ratio'] < 0.30) &
        (market_stats['yes_dollar_ratio'] > 0.40) &
        (market_stats['total_trades'] >= 5)
    ]

    result_no = quick_edge_check(signal_no, side='no', name="Retail YES, Smart NO (bet NO)")
    result_yes = quick_edge_check(signal_yes, side='yes', name="Retail NO, Smart YES (bet YES)")

    print(f"Retail YES, Smart NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Retail NO, Smart YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    # More extreme version
    extreme_no = market_stats[
        (market_stats['yes_trade_ratio'] > 0.80) &
        (market_stats['yes_dollar_ratio'] < 0.50) &
        (market_stats['total_trades'] >= 5)
    ]
    result_extreme = quick_edge_check(extreme_no, side='no', name="Extreme imbalance (80/50)")
    print(f"Extreme (80% trades YES, <50% dollars YES): N={result_extreme['n']}, Edge={result_extreme.get('edge', 'N/A')}")

    return {'imbalance_no': result_no, 'imbalance_yes': result_yes, 'extreme': result_extreme}


def test_h208_whale_consensus_counter(df):
    """
    H-LSD-208: Whale Consensus Counter

    Signal: ALL whales bet same side, majority of small trades opposite.
    Bet with the whales.

    CORE PRINCIPLE: Capital weight (whales) vs trade count (retail)
    """
    print("\n" + "=" * 60)
    print("H-LSD-208: WHALE CONSENSUS COUNTER (P1)")
    print("=" * 60)

    # Get whale and non-whale stats per market
    whale_trades = df[df['is_whale']]
    small_trades = df[~df['is_whale']]

    whale_stats = whale_trades.groupby('market_ticker').agg({
        'is_yes': 'mean',
        'market_ticker': 'count'
    })
    whale_stats.columns = ['whale_yes_ratio', 'whale_count']

    small_stats = small_trades.groupby('market_ticker').agg({
        'is_yes': 'mean'
    })
    small_stats.columns = ['small_yes_ratio']

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    combined = market_info.merge(whale_stats, on='market_ticker', how='inner')
    combined = combined.merge(small_stats, on='market_ticker', how='inner')

    # Signal: 100% whales bet NO but >60% small trades bet YES -> bet NO
    whales_no_retail_yes = combined[
        (combined['whale_yes_ratio'] == 0) &  # All whales bet NO
        (combined['small_yes_ratio'] > 0.6) &  # Majority retail bets YES
        (combined['whale_count'] >= 2)  # At least 2 whale trades
    ]

    # Signal: 100% whales bet YES but >60% small trades bet NO -> bet YES
    whales_yes_retail_no = combined[
        (combined['whale_yes_ratio'] == 1) &  # All whales bet YES
        (combined['small_yes_ratio'] < 0.4) &  # Majority retail bets NO
        (combined['whale_count'] >= 2)
    ]

    result_no = quick_edge_check(whales_no_retail_yes, side='no', name="Whales 100% NO, Retail YES")
    result_yes = quick_edge_check(whales_yes_retail_no, side='yes', name="Whales 100% YES, Retail NO")

    print(f"Whales 100% NO vs Retail YES: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Whales 100% YES vs Retail NO: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'whale_no': result_no, 'whale_yes': result_yes}


def test_h209_size_gradient(df):
    """
    H-LSD-209: Size Gradient

    Signal: Larger trades going opposite direction from smaller trades.
    Bet with the larger trade direction.

    CORE PRINCIPLE: Capital weight reveals informed direction.
    """
    print("\n" + "=" * 60)
    print("H-LSD-209: SIZE GRADIENT (P1)")
    print("=" * 60)

    # For each market, calculate correlation between trade size and direction
    def calc_size_direction_corr(group):
        if len(group) < 5:
            return pd.Series({'corr': np.nan, 'n_trades': len(group)})
        # is_no = 1 if NO, 0 if YES
        corr = group['trade_value_cents'].corr(group['is_no'].astype(float))
        return pd.Series({'corr': corr, 'n_trades': len(group)})

    market_corr = df.groupby('market_ticker').apply(calc_size_direction_corr).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    combined = market_info.merge(market_corr, on='market_ticker')
    combined = combined.dropna(subset=['corr'])

    # Positive correlation: larger trades bet NO -> bet NO
    large_bet_no = combined[
        (combined['corr'] > 0.3) &
        (combined['n_trades'] >= 5)
    ]

    # Negative correlation: larger trades bet YES -> bet YES
    large_bet_yes = combined[
        (combined['corr'] < -0.3) &
        (combined['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(large_bet_no, side='no', name="Large trades bet NO")
    result_yes = quick_edge_check(large_bet_yes, side='yes', name="Large trades bet YES")

    print(f"Large trades bet NO (corr>0.3): N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Large trades bet YES (corr<-0.3): N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'gradient_no': result_no, 'gradient_yes': result_yes}


# =============================================================================
# PRINCIPLE 3: PRICE DISCOVERY DELAY (Time-Based)
# =============================================================================

def test_h201_opening_bell_momentum(df):
    """
    H-LSD-201: Opening Bell Momentum

    Signal: First 3 trades all same direction predicts outcome.
    Early information is most valuable.

    CORE PRINCIPLE: Smart money moves FIRST.
    """
    print("\n" + "=" * 60)
    print("H-LSD-201: OPENING BELL MOMENTUM (P3)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Get first 3 trades per market
    first_trades = df_sorted.groupby('market_ticker').head(3)

    first_stats = first_trades.groupby('market_ticker').agg({
        'is_yes': ['sum', 'count'],
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    })
    first_stats.columns = ['yes_count', 'total_count', 'market_result', 'no_price', 'yes_price']
    first_stats = first_stats.reset_index()

    # Filter to markets with exactly 3+ first trades
    first_stats = first_stats[first_stats['total_count'] >= 3]

    # All 3 YES -> bet YES
    all_yes = first_stats[first_stats['yes_count'] == 3]

    # All 3 NO -> bet NO
    all_no = first_stats[first_stats['yes_count'] == 0]

    result_yes = quick_edge_check(all_yes, side='yes', name="First 3 all YES")
    result_no = quick_edge_check(all_no, side='no', name="First 3 all NO")

    print(f"First 3 trades all YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"First 3 trades all NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    return {'open_yes': result_yes, 'open_no': result_no}


def test_h202_closing_rush_fade(df):
    """
    H-LSD-202: Closing Rush Fade

    Signal: Markets with 50%+ of trades in final hour before resolution.
    Late money is often dumb money (panic/FOMO).

    CORE PRINCIPLE: Late traders are uninformed.
    """
    print("\n" + "=" * 60)
    print("H-LSD-202: CLOSING RUSH FADE (P3)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate time position of each trade
    def calc_time_position(group):
        group = group.sort_values('datetime')
        first_time = group['datetime'].min()
        last_time = group['datetime'].max()
        duration = (last_time - first_time).total_seconds()
        if duration <= 0:
            group['time_position'] = 0.5
        else:
            group['time_position'] = (group['datetime'] - first_time).dt.total_seconds() / duration
        return group

    df_timed = df_sorted.groupby('market_ticker').apply(calc_time_position).reset_index(drop=True)

    # Trades in final 25% of market duration
    late_trades = df_timed[df_timed['time_position'] > 0.75]
    all_trades = df_timed.groupby('market_ticker')['datetime'].count().reset_index()
    all_trades.columns = ['market_ticker', 'total_trades']

    late_stats = late_trades.groupby('market_ticker').agg({
        'is_yes': 'mean',
        'datetime': 'count'
    }).reset_index()
    late_stats.columns = ['market_ticker', 'late_yes_ratio', 'late_count']

    combined = all_trades.merge(late_stats, on='market_ticker', how='inner')
    combined['late_ratio'] = combined['late_count'] / combined['total_trades']

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    combined = combined.merge(market_info, on='market_ticker')

    # Signal: >50% trades in final quarter AND late trades favor YES -> fade to NO
    late_yes_heavy = combined[
        (combined['late_ratio'] > 0.5) &
        (combined['late_yes_ratio'] > 0.6) &
        (combined['total_trades'] >= 5)
    ]

    # Inverse: late trades favor NO
    late_no_heavy = combined[
        (combined['late_ratio'] > 0.5) &
        (combined['late_yes_ratio'] < 0.4) &
        (combined['total_trades'] >= 5)
    ]

    result_fade_yes = quick_edge_check(late_yes_heavy, side='no', name="Fade late YES rush")
    result_fade_no = quick_edge_check(late_no_heavy, side='yes', name="Fade late NO rush")

    print(f"Fade late YES rush (bet NO): N={result_fade_yes['n']}, Edge={result_fade_yes.get('edge', 'N/A')}")
    print(f"Fade late NO rush (bet YES): N={result_fade_no['n']}, Edge={result_fade_no.get('edge', 'N/A')}")

    return {'fade_late_yes': result_fade_yes, 'fade_late_no': result_fade_no}


def test_h203_dead_period_signal(df):
    """
    H-LSD-203: Dead Period Signal

    Signal: Markets with long gaps (4+ hours) between trades.
    Could indicate informed traders waiting for clarity.

    CORE PRINCIPLE: Informed traders time their entries.
    """
    print("\n" + "=" * 60)
    print("H-LSD-203: DEAD PERIOD SIGNAL (P3)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate max gap between trades
    def calc_max_gap(group):
        if len(group) < 2:
            return pd.Series({'max_gap_hours': 0, 'n_trades': len(group)})
        group = group.sort_values('datetime')
        gaps = group['datetime'].diff().dt.total_seconds() / 3600
        return pd.Series({'max_gap_hours': gaps.max(), 'n_trades': len(group)})

    market_gaps = df_sorted.groupby('market_ticker').apply(calc_max_gap).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean'  # NO ratio
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    combined = market_info.merge(market_gaps, on='market_ticker')

    # Long gap + NO majority -> bet NO
    long_gap_no = combined[
        (combined['max_gap_hours'] >= 4) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # Long gap + YES majority -> bet YES
    long_gap_yes = combined[
        (combined['max_gap_hours'] >= 4) &
        (combined['no_ratio'] < 0.5) &
        (combined['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(long_gap_no, side='no', name="Long gap + NO majority")
    result_yes = quick_edge_check(long_gap_yes, side='yes', name="Long gap + YES majority")

    print(f"Long gap (4h+) + NO majority: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Long gap (4h+) + YES majority: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'gap_no': result_no, 'gap_yes': result_yes}


# =============================================================================
# PRINCIPLE 4: SYSTEMATIC vs RANDOM BEHAVIOR
# =============================================================================

def test_h206_inter_arrival_regularity(df):
    """
    H-LSD-206: Inter-Arrival Regularity

    Signal: CV of time between trades < 0.5 (clock-like timing).
    Clock-like trading = algorithmic = informed?

    CORE PRINCIPLE: Systematic behavior reveals informed trading.
    """
    print("\n" + "=" * 60)
    print("H-LSD-206: INTER-ARRIVAL REGULARITY (P4)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate CV of inter-arrival times
    def calc_arrival_cv(group):
        if len(group) < 5:
            return pd.Series({'arrival_cv': np.nan, 'n_trades': len(group)})
        group = group.sort_values('datetime')
        gaps = group['datetime'].diff().dt.total_seconds().dropna()
        if len(gaps) < 3 or gaps.mean() == 0:
            return pd.Series({'arrival_cv': np.nan, 'n_trades': len(group)})
        cv = gaps.std() / gaps.mean()
        return pd.Series({'arrival_cv': cv, 'n_trades': len(group)})

    market_cv = df_sorted.groupby('market_ticker').apply(calc_arrival_cv).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean'
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    combined = market_info.merge(market_cv, on='market_ticker')
    combined = combined.dropna(subset=['arrival_cv'])

    # Low CV (clock-like) + NO majority -> bet NO
    clock_no = combined[
        (combined['arrival_cv'] < 0.5) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # Low CV + YES majority -> bet YES
    clock_yes = combined[
        (combined['arrival_cv'] < 0.5) &
        (combined['no_ratio'] < 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # Compare with high CV (random timing)
    random_no = combined[
        (combined['arrival_cv'] > 1.5) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    result_clock_no = quick_edge_check(clock_no, side='no', name="Clock-like + NO majority")
    result_clock_yes = quick_edge_check(clock_yes, side='yes', name="Clock-like + YES majority")
    result_random_no = quick_edge_check(random_no, side='no', name="Random timing + NO majority")

    print(f"Clock-like (CV<0.5) + NO majority: N={result_clock_no['n']}, Edge={result_clock_no.get('edge', 'N/A')}")
    print(f"Clock-like (CV<0.5) + YES majority: N={result_clock_yes['n']}, Edge={result_clock_yes.get('edge', 'N/A')}")
    print(f"Random timing (CV>1.5) + NO majority: N={result_random_no['n']}, Edge={result_random_no.get('edge', 'N/A')}")

    return {'clock_no': result_clock_no, 'clock_yes': result_clock_yes, 'random_no': result_random_no}


def test_h204_leverage_consistency_within(df):
    """
    H-LSD-204: Leverage Consistency Within Market

    Signal: CV of leverage < 0.3 (all trades have similar leverage).
    Different from S013 which uses STD across all trades.

    CORE PRINCIPLE: Consistent leverage = systematic trading.
    """
    print("\n" + "=" * 60)
    print("H-LSD-204: LEVERAGE CONSISTENCY WITHIN MARKET (P4)")
    print("=" * 60)

    # Calculate CV of leverage within each market
    def calc_lev_cv(group):
        if len(group) < 5:
            return pd.Series({'lev_cv': np.nan, 'n_trades': len(group)})
        lev = group['leverage_ratio']
        if lev.mean() == 0:
            return pd.Series({'lev_cv': np.nan, 'n_trades': len(group)})
        cv = lev.std() / lev.mean()
        return pd.Series({'lev_cv': cv, 'n_trades': len(group)})

    market_cv = df.groupby('market_ticker').apply(calc_lev_cv).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean'
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    combined = market_info.merge(market_cv, on='market_ticker')
    combined = combined.dropna(subset=['lev_cv'])

    # Low CV (consistent) + NO majority -> bet NO
    consistent_no = combined[
        (combined['lev_cv'] < 0.3) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # Low CV + YES majority -> bet YES
    consistent_yes = combined[
        (combined['lev_cv'] < 0.3) &
        (combined['no_ratio'] < 0.5) &
        (combined['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(consistent_no, side='no', name="Consistent leverage + NO")
    result_yes = quick_edge_check(consistent_yes, side='yes', name="Consistent leverage + YES")

    print(f"Consistent leverage (CV<0.3) + NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Consistent leverage (CV<0.3) + YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'consistent_no': result_no, 'consistent_yes': result_yes}


def test_h205_size_clustering(df):
    """
    H-LSD-205: Size Clustering

    Signal: 80%+ of trades in same size band.
    Suggests single actor or coordinated trading.

    CORE PRINCIPLE: Clustering = systematic execution.
    """
    print("\n" + "=" * 60)
    print("H-LSD-205: SIZE CLUSTERING (P4)")
    print("=" * 60)

    # Define size bands
    def get_size_band(count):
        if count <= 10:
            return 'micro'
        elif count <= 50:
            return 'small'
        elif count <= 200:
            return 'medium'
        else:
            return 'large'

    df['size_band'] = df['count'].apply(get_size_band)

    # Calculate dominant band per market
    def calc_dominant_band(group):
        if len(group) < 5:
            return pd.Series({'dominant_ratio': np.nan, 'dominant_band': None, 'n_trades': len(group)})
        band_counts = group['size_band'].value_counts()
        dominant = band_counts.idxmax()
        ratio = band_counts.max() / len(group)
        return pd.Series({'dominant_ratio': ratio, 'dominant_band': dominant, 'n_trades': len(group)})

    market_bands = df.groupby('market_ticker').apply(calc_dominant_band).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean'
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    combined = market_info.merge(market_bands, on='market_ticker')
    combined = combined.dropna(subset=['dominant_ratio'])

    # High clustering + NO majority -> bet NO
    clustered_no = combined[
        (combined['dominant_ratio'] > 0.8) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # High clustering + YES majority -> bet YES
    clustered_yes = combined[
        (combined['dominant_ratio'] > 0.8) &
        (combined['no_ratio'] < 0.5) &
        (combined['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(clustered_no, side='no', name="Size cluster + NO")
    result_yes = quick_edge_check(clustered_yes, side='yes', name="Size cluster + YES")

    print(f"Size clustering (80%+ same band) + NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Size clustering (80%+ same band) + YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'cluster_no': result_no, 'cluster_yes': result_yes}


# =============================================================================
# PRINCIPLE 2: SENTIMENT vs CONVICTION
# =============================================================================

def test_h210_price_stickiness(df):
    """
    H-LSD-210: Price Stickiness Despite Volume

    Signal: High trade count but price barely moved.
    Smart money absorbing retail flow without moving price.

    CORE PRINCIPLE: Informed traders absorb flow quietly.
    """
    print("\n" + "=" * 60)
    print("H-LSD-210: PRICE STICKINESS DESPITE VOLUME (P2)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'yes_price': ['first', 'last', 'count', 'mean'],
        'market_result': 'first',
        'no_price': 'mean',
        'is_no': 'mean'
    })
    market_stats.columns = ['first_yes', 'last_yes', 'n_trades', 'yes_price', 'market_result', 'no_price', 'no_ratio']
    market_stats = market_stats.reset_index()

    # Price move
    market_stats['price_move'] = abs(market_stats['last_yes'] - market_stats['first_yes'])

    # High volume but small move
    market_stats['vol_per_move'] = market_stats['n_trades'] / (market_stats['price_move'] + 1)

    # Sticky: >20 trades but price moved <10c
    sticky = market_stats[
        (market_stats['n_trades'] >= 20) &
        (market_stats['price_move'] < 10)
    ].copy()

    # Bet with price direction (if YES price dropped, bet NO)
    sticky['price_dropped'] = sticky['last_yes'] < sticky['first_yes']

    sticky_bet_no = sticky[sticky['price_dropped'] == True]
    sticky_bet_yes = sticky[sticky['price_dropped'] == False]

    result_no = quick_edge_check(sticky_bet_no, side='no', name="Sticky + price dropped")
    result_yes = quick_edge_check(sticky_bet_yes, side='yes', name="Sticky + price rose")

    print(f"Sticky (20+ trades, <10c move) + price dropped: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Sticky (20+ trades, <10c move) + price rose: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'sticky_no': result_no, 'sticky_yes': result_yes}


def test_h211_conviction_ratio(df):
    """
    H-LSD-211: Conviction Ratio Extreme

    Signal: Avg NO trade size > 2x Avg YES trade size (or vice versa).
    Who's betting bigger reveals conviction.

    CORE PRINCIPLE: Size reveals conviction, not just direction.
    """
    print("\n" + "=" * 60)
    print("H-LSD-211: CONVICTION RATIO EXTREME (P2)")
    print("=" * 60)

    # Calculate avg size by side
    yes_sizes = df[df['is_yes']].groupby('market_ticker')['trade_value_cents'].mean()
    no_sizes = df[df['is_no']].groupby('market_ticker')['trade_value_cents'].mean()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'count'
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'n_trades']

    market_info['yes_avg_size'] = market_info['market_ticker'].map(yes_sizes)
    market_info['no_avg_size'] = market_info['market_ticker'].map(no_sizes)
    market_info = market_info.dropna(subset=['yes_avg_size', 'no_avg_size'])

    market_info['size_ratio'] = market_info['no_avg_size'] / (market_info['yes_avg_size'] + 0.01)

    # NO conviction > 2x YES conviction -> bet NO
    no_conviction = market_info[
        (market_info['size_ratio'] > 2) &
        (market_info['n_trades'] >= 5)
    ]

    # YES conviction > 2x NO conviction -> bet YES
    yes_conviction = market_info[
        (market_info['size_ratio'] < 0.5) &
        (market_info['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(no_conviction, side='no', name="NO conviction 2x YES")
    result_yes = quick_edge_check(yes_conviction, side='yes', name="YES conviction 2x NO")

    print(f"NO avg size > 2x YES avg size: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"YES avg size > 2x NO avg size: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'conviction_no': result_no, 'conviction_yes': result_yes}


# =============================================================================
# PRINCIPLE 5: UNCERTAINTY PREMIUM
# =============================================================================

def test_h213_leverage_spread(df):
    """
    H-LSD-213: Leverage Spread Extreme

    Signal: Wide range of leverage ratios (max-min > 5).
    High disagreement = high uncertainty = mispricing?

    CORE PRINCIPLE: Uncertainty creates opportunity.
    """
    print("\n" + "=" * 60)
    print("H-LSD-213: LEVERAGE SPREAD EXTREME (P5)")
    print("=" * 60)

    market_lev = df.groupby('market_ticker').agg({
        'leverage_ratio': ['min', 'max', 'mean'],
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean',
        'datetime': 'count'
    })
    market_lev.columns = ['lev_min', 'lev_max', 'lev_mean', 'market_result', 'no_price', 'yes_price', 'no_ratio', 'n_trades']
    market_lev = market_lev.reset_index()

    market_lev['lev_spread'] = market_lev['lev_max'] - market_lev['lev_min']

    # High spread + NO majority -> bet NO
    high_spread_no = market_lev[
        (market_lev['lev_spread'] > 5) &
        (market_lev['no_ratio'] > 0.5) &
        (market_lev['n_trades'] >= 5)
    ]

    # High spread + YES majority -> bet YES
    high_spread_yes = market_lev[
        (market_lev['lev_spread'] > 5) &
        (market_lev['no_ratio'] < 0.5) &
        (market_lev['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(high_spread_no, side='no', name="High lev spread + NO")
    result_yes = quick_edge_check(high_spread_yes, side='yes', name="High lev spread + YES")

    print(f"High leverage spread (>5) + NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"High leverage spread (>5) + YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'spread_no': result_no, 'spread_yes': result_yes}


def test_h214_midprice_whale_disagreement(df):
    """
    H-LSD-214: Mid-Price Whale Disagreement

    Signal: At 40-60c, whales split 50/50.
    Expert uncertainty at mid-price = bet with price direction?

    CORE PRINCIPLE: Expert disagreement at uncertainty prices.
    """
    print("\n" + "=" * 60)
    print("H-LSD-214: MID-PRICE WHALE DISAGREEMENT (P5)")
    print("=" * 60)

    # Filter to mid-price markets (40-60c NO price)
    mid_price_markets = df.groupby('market_ticker').agg({
        'no_price': 'mean'
    }).reset_index()
    mid_price_markets = mid_price_markets[
        (mid_price_markets['no_price'] >= 40) &
        (mid_price_markets['no_price'] <= 60)
    ]['market_ticker']

    mid_df = df[df['market_ticker'].isin(mid_price_markets)]

    # Get whale trades
    whale_trades = mid_df[mid_df['is_whale']]

    whale_stats = whale_trades.groupby('market_ticker').agg({
        'is_no': 'mean',
        'datetime': 'count'
    }).reset_index()
    whale_stats.columns = ['market_ticker', 'whale_no_ratio', 'whale_count']

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    combined = whale_stats.merge(market_info, on='market_ticker')

    # Split whales (40-60% NO) at mid-price
    # Bet with slight majority
    whale_lean_no = combined[
        (combined['whale_no_ratio'] > 0.5) &
        (combined['whale_no_ratio'] < 0.7) &
        (combined['whale_count'] >= 3)
    ]

    whale_lean_yes = combined[
        (combined['whale_no_ratio'] < 0.5) &
        (combined['whale_no_ratio'] > 0.3) &
        (combined['whale_count'] >= 3)
    ]

    result_no = quick_edge_check(whale_lean_no, side='no', name="Whale slight lean NO at mid")
    result_yes = quick_edge_check(whale_lean_yes, side='yes', name="Whale slight lean YES at mid")

    print(f"Whales lean NO (50-70%) at 40-60c: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Whales lean YES (30-50%) at 40-60c: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'whale_mid_no': result_no, 'whale_mid_yes': result_yes}


def test_h215_leverage_trend_acceleration(df):
    """
    H-LSD-215: Leverage Trend Acceleration

    Signal: Leverage increasing throughout market.
    Growing conviction/desperation as market progresses.

    CORE PRINCIPLE: Trend reveals conviction buildup.
    """
    print("\n" + "=" * 60)
    print("H-LSD-215: LEVERAGE TREND ACCELERATION (P5)")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate leverage trend
    def calc_lev_trend(group):
        if len(group) < 5:
            return pd.Series({'lev_trend': np.nan, 'n_trades': len(group)})
        group = group.sort_values('datetime')
        x = np.arange(len(group))
        y = group['leverage_ratio'].values
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
        except:
            slope = np.nan
        return pd.Series({'lev_trend': slope, 'n_trades': len(group)})

    market_trends = df_sorted.groupby('market_ticker').apply(calc_lev_trend).reset_index()

    market_info = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'is_no': 'mean'
    }).reset_index()
    market_info.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    combined = market_info.merge(market_trends, on='market_ticker')
    combined = combined.dropna(subset=['lev_trend'])

    # Increasing leverage + NO majority -> bet NO
    increasing_no = combined[
        (combined['lev_trend'] > 0.1) &
        (combined['no_ratio'] > 0.5) &
        (combined['n_trades'] >= 5)
    ]

    # Increasing leverage + YES majority -> bet YES
    increasing_yes = combined[
        (combined['lev_trend'] > 0.1) &
        (combined['no_ratio'] < 0.5) &
        (combined['n_trades'] >= 5)
    ]

    result_no = quick_edge_check(increasing_no, side='no', name="Increasing leverage + NO")
    result_yes = quick_edge_check(increasing_yes, side='yes', name="Increasing leverage + YES")

    print(f"Increasing leverage trend + NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Increasing leverage trend + YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'accel_no': result_no, 'accel_yes': result_yes}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("LSD SESSION 002: EXPLOITING THE 5 CORE PRINCIPLES")
    print("=" * 80)
    print("\nMission: Find NOVEL strategies through DIFFERENT signals")
    print("Speed over rigor - flag anything with >5% raw edge\n")

    df = load_data()

    # Run all tests
    print("\n" + "=" * 80)
    print("PRINCIPLE 1: CAPITAL WEIGHT vs TRADE COUNT")
    print("=" * 80)
    results['H207_dollar_weighted'] = test_h207_dollar_weighted_direction(df)
    results['H212_imbalance'] = test_h212_trade_count_vs_dollar_imbalance(df)
    results['H208_whale_counter'] = test_h208_whale_consensus_counter(df)
    results['H209_size_gradient'] = test_h209_size_gradient(df)

    print("\n" + "=" * 80)
    print("PRINCIPLE 3: PRICE DISCOVERY DELAY (Time-Based)")
    print("=" * 80)
    results['H201_opening_bell'] = test_h201_opening_bell_momentum(df)
    results['H202_closing_rush'] = test_h202_closing_rush_fade(df)
    results['H203_dead_period'] = test_h203_dead_period_signal(df)

    print("\n" + "=" * 80)
    print("PRINCIPLE 4: SYSTEMATIC vs RANDOM BEHAVIOR")
    print("=" * 80)
    results['H206_arrival_regularity'] = test_h206_inter_arrival_regularity(df)
    results['H204_leverage_consistency'] = test_h204_leverage_consistency_within(df)
    results['H205_size_clustering'] = test_h205_size_clustering(df)

    print("\n" + "=" * 80)
    print("PRINCIPLE 2: SENTIMENT vs CONVICTION")
    print("=" * 80)
    results['H210_price_stickiness'] = test_h210_price_stickiness(df)
    results['H211_conviction_ratio'] = test_h211_conviction_ratio(df)

    print("\n" + "=" * 80)
    print("PRINCIPLE 5: UNCERTAINTY PREMIUM")
    print("=" * 80)
    results['H213_leverage_spread'] = test_h213_leverage_spread(df)
    results['H214_whale_disagreement'] = test_h214_midprice_whale_disagreement(df)
    results['H215_leverage_trend'] = test_h215_leverage_trend_acceleration(df)

    # Summary
    print("\n" + "=" * 80)
    print("LSD SESSION 002: SUMMARY")
    print("=" * 80)

    promising = []
    all_results = []

    for test_name, test_results in results.items():
        for signal_name, result in test_results.items():
            if result.get('flag'):
                promising.append({
                    'test': test_name,
                    'signal': signal_name,
                    'edge': result['edge'],
                    'n': result['n'],
                    'p_value': result['p_value']
                })
            # Convert numpy/pandas types to native Python types
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (np.bool_, np.integer, np.floating)):
                    clean_result[k] = v.item() if hasattr(v, 'item') else bool(v) if isinstance(v, np.bool_) else v
                elif isinstance(v, bool):
                    clean_result[k] = bool(v)
                else:
                    clean_result[k] = v
            all_results.append({
                'test': test_name,
                'signal': signal_name,
                **clean_result
            })

    print("\n--- PROMISING SIGNALS (>5% edge, p<0.05) ---")
    if promising:
        for p in sorted(promising, key=lambda x: -x['edge']):
            print(f"  {p['test']} / {p['signal']}: Edge={p['edge']:.2%}, N={p['n']}, p={p['p_value']:.4f}")
    else:
        print("  No signals flagged at >5% edge")

    print("\n--- ALL SIGNALS BY EDGE ---")
    valid_results = [r for r in all_results if r.get('edge') is not None]
    for r in sorted(valid_results, key=lambda x: -x['edge'])[:20]:
        verdict = r.get('verdict', '')
        print(f"  {r['test']} / {r['signal']}: Edge={r['edge']:.2%}, N={r['n']}, {verdict}")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'session': 'LSD_002',
            'promising': promising,
            'all_results': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")

    return results, promising, all_results


if __name__ == '__main__':
    results, promising, all_results = main()
