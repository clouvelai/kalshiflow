"""
LSD SESSION - SPORTS EXPERT HYPOTHESIS SCREENING

Speed over rigor. Flag anything with raw edge > 5% OR improvement > 3%.
Test all 15 SPORTS-* hypotheses from the sports betting expert.

Priority Tier 1 First:
- SPORTS-001: Steam Exhaustion Detection
- SPORTS-005: Size Velocity Divergence
- SPORTS-008: Size Distribution Shape Change
- SPORTS-009: Spread Widening Before Sharp Entry
- SPORTS-011: Category Momentum Contagion

Then Tier 2:
- SPORTS-002: Opening Move Reversal
- SPORTS-003: Momentum Velocity Stall
- SPORTS-007: Late-Arriving Large Money
- SPORTS-012: NCAAF Totals Specialist

Then LSD-Absurd:
- SPORTS-004: Extreme Public Sentiment Fade
- SPORTS-006: Round Number Retail Clustering
- SPORTS-013: Trade Count Milestone Fading
- SPORTS-014: Bot Signature Fade
- SPORTS-015: Fibonacci Price Attractors
- SPORTS-010: Multi-Outcome Pricing Inconsistency
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

results = {}

def load_data():
    """Load the enriched trades data."""
    print("=" * 80)
    print("SPORTS EXPERT LSD SCREENING - LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Add useful columns
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000  # >= $100
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
    df['is_weekend'] = df['day_of_week'] >= 5

    # Timestamp in ms for inter-arrival calculations
    df['timestamp_ms'] = df['datetime'].astype(np.int64) // 10**6

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def quick_edge_check(signal_markets, side='no', name=""):
    """
    Fast edge check. Returns raw edge and sample size.
    Flag if edge > 5% for deeper analysis.
    """
    n = len(signal_markets)
    if n < 30:
        return {'name': name, 'n': n, 'edge': None, 'flag': False, 'reason': 'insufficient'}

    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['no_price'].mean()
    else:
        avg_price = signal_markets['yes_price'].mean()

    be = avg_price / 100
    edge = wr - be

    # P-value (quick check)
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    flag = edge > 0.05 and p_value < 0.1  # More lenient for LSD mode
    weak = edge > 0.03 and p_value < 0.15

    return {
        'name': name,
        'n': n,
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'flag': 'PROMISING' if flag else ('WEAK' if weak else 'FAIL'),
        'flag_reason': 'POTENTIAL EDGE >5%' if flag else ('Edge 3-5%' if weak else 'No edge')
    }


# ===========================================================================
# TIER 1: TEST FIRST
# ===========================================================================

def test_sports001_steam_exhaustion(df):
    """
    SPORTS-001: Steam Exhaustion Detection

    Signal: 5+ consecutive same-direction trades in <60 seconds,
    moved price >10c, then 3+ minutes with NO same-direction trades.
    Bet: FADE the steam direction.
    """
    print("\n" + "=" * 60)
    print("SPORTS-001: STEAM EXHAUSTION DETECTION")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    steam_exhaustion_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 8:  # Need enough trades
            continue

        mdf = mdf.reset_index(drop=True)
        times = mdf['timestamp_ms'].values
        sides = mdf['taker_side'].values
        prices = mdf['yes_price'].values

        # Look for steam events (5+ same direction in 60s)
        steam_found = None
        steam_direction = None
        steam_end_idx = None
        steam_end_time = None
        steam_end_price = None

        for i in range(len(mdf) - 4):
            # Check if 5 consecutive same-side trades
            run_side = sides[i]
            run_length = 1

            for j in range(i + 1, len(mdf)):
                if sides[j] == run_side and (times[j] - times[i]) < 60000:  # 60 seconds
                    run_length += 1
                else:
                    break

            if run_length >= 5:
                # Check if price moved >10c during the run
                end_idx = i + run_length - 1
                price_change = abs(prices[end_idx] - prices[i])

                if price_change > 10:
                    steam_found = True
                    steam_direction = run_side
                    steam_end_idx = end_idx
                    steam_end_time = times[end_idx]
                    steam_end_price = prices[end_idx]
                    break

        if not steam_found:
            continue

        # Check for exhaustion: 3+ minutes with NO same-direction trades
        remaining_trades = mdf.iloc[steam_end_idx + 1:]
        if len(remaining_trades) == 0:
            continue

        # Find first same-direction trade after steam
        next_same_direction = remaining_trades[remaining_trades['taker_side'] == steam_direction]

        if len(next_same_direction) == 0:
            # Never another same-direction trade = ultimate exhaustion
            gap_ms = 180001
        else:
            first_next = next_same_direction.iloc[0]
            gap_ms = first_next['timestamp_ms'] - steam_end_time

        if gap_ms >= 180000:  # 3 minutes = exhaustion
            # Signal: fade the steam direction
            fade_direction = 'no' if steam_direction == 'yes' else 'yes'

            steam_exhaustion_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'steam_direction': steam_direction,
                'fade_direction': fade_direction,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean(),
                'gap_minutes': gap_ms / 60000
            })

    if len(steam_exhaustion_markets) == 0:
        print("No steam exhaustion events detected")
        return {'name': 'SPORTS-001', 'result': 'FAIL', 'reason': 'No signals found'}

    sem_df = pd.DataFrame(steam_exhaustion_markets)

    # Check fade YES steam (bet NO)
    fade_yes = sem_df[sem_df['steam_direction'] == 'yes']
    result_fade_yes = quick_edge_check(fade_yes, side='no', name="Fade YES Steam (bet NO)")

    # Check fade NO steam (bet YES)
    fade_no = sem_df[sem_df['steam_direction'] == 'no']
    result_fade_no = quick_edge_check(fade_no, side='yes', name="Fade NO Steam (bet YES)")

    print(f"Fade YES steam: N={result_fade_yes['n']}, Edge={result_fade_yes.get('edge', 'N/A')}")
    print(f"Fade NO steam: N={result_fade_no['n']}, Edge={result_fade_no.get('edge', 'N/A')}")

    best = result_fade_yes if (result_fade_yes.get('edge') or -999) > (result_fade_no.get('edge') or -999) else result_fade_no

    return {
        'hypothesis': 'SPORTS-001',
        'name': 'Steam Exhaustion Detection',
        'fade_yes_steam': result_fade_yes,
        'fade_no_steam': result_fade_no,
        'best': best
    }


def test_sports005_size_velocity_divergence(df):
    """
    SPORTS-005: Size Velocity Divergence (Retail Pile-On Detection)

    Signal: Trade frequency INCREASING but average size DECREASING.
    This indicates retail pile-on.
    Bet: FADE the direction of increased frequency.
    """
    print("\n" + "=" * 60)
    print("SPORTS-005: SIZE VELOCITY DIVERGENCE")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    divergence_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 20:  # Need enough trades
            continue

        mdf = mdf.reset_index(drop=True)

        # Split into first and second half by TIME
        mid_time = mdf['datetime'].min() + (mdf['datetime'].max() - mdf['datetime'].min()) / 2
        first_half = mdf[mdf['datetime'] <= mid_time]
        second_half = mdf[mdf['datetime'] > mid_time]

        if len(first_half) < 5 or len(second_half) < 5:
            continue

        # Calculate velocity (trades per minute)
        first_duration = (first_half['datetime'].max() - first_half['datetime'].min()).total_seconds() / 60
        second_duration = (second_half['datetime'].max() - second_half['datetime'].min()).total_seconds() / 60

        if first_duration < 1 or second_duration < 1:
            continue

        first_velocity = len(first_half) / max(first_duration, 1)
        second_velocity = len(second_half) / max(second_duration, 1)

        # Calculate average size
        first_avg_size = first_half['count'].mean()
        second_avg_size = second_half['count'].mean()

        # Divergence: velocity UP, size DOWN
        velocity_increase = second_velocity > first_velocity * 1.3  # 30% increase
        size_decrease = second_avg_size < first_avg_size * 0.8  # 20% decrease

        if velocity_increase and size_decrease:
            # What direction is the increased activity?
            second_half_yes_ratio = (second_half['taker_side'] == 'yes').mean()
            pile_on_direction = 'yes' if second_half_yes_ratio > 0.6 else ('no' if second_half_yes_ratio < 0.4 else 'neutral')

            if pile_on_direction != 'neutral':
                fade_direction = 'no' if pile_on_direction == 'yes' else 'yes'

                divergence_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'pile_on_direction': pile_on_direction,
                    'fade_direction': fade_direction,
                    'velocity_change': second_velocity / first_velocity,
                    'size_change': second_avg_size / first_avg_size,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    if len(divergence_markets) == 0:
        print("No velocity divergence events detected")
        return {'name': 'SPORTS-005', 'result': 'FAIL', 'reason': 'No signals found'}

    div_df = pd.DataFrame(divergence_markets)

    # Check fade YES pile-on (bet NO)
    fade_yes = div_df[div_df['pile_on_direction'] == 'yes']
    result_fade_yes = quick_edge_check(fade_yes, side='no', name="Fade YES Pile-On (bet NO)")

    # Check fade NO pile-on (bet YES)
    fade_no = div_df[div_df['pile_on_direction'] == 'no']
    result_fade_no = quick_edge_check(fade_no, side='yes', name="Fade NO Pile-On (bet YES)")

    print(f"Fade YES pile-on: N={result_fade_yes['n']}, Edge={result_fade_yes.get('edge', 'N/A')}")
    print(f"Fade NO pile-on: N={result_fade_no['n']}, Edge={result_fade_no.get('edge', 'N/A')}")

    best = result_fade_yes if (result_fade_yes.get('edge') or -999) > (result_fade_no.get('edge') or -999) else result_fade_no

    return {
        'hypothesis': 'SPORTS-005',
        'name': 'Size Velocity Divergence',
        'fade_yes_pileon': result_fade_yes,
        'fade_no_pileon': result_fade_no,
        'best': best
    }


def test_sports008_size_distribution_change(df):
    """
    SPORTS-008: Size Distribution Shape Change

    Signal: Size distribution changed from first half to second half.
    Uniform -> Clustered = dominant player entered.
    Bet: Follow the direction of the NEW regime.
    """
    print("\n" + "=" * 60)
    print("SPORTS-008: SIZE DISTRIBUTION SHAPE CHANGE")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    regime_change_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 20:  # Need enough trades
            continue

        mdf = mdf.reset_index(drop=True)

        # Split into halves
        mid_idx = len(mdf) // 2
        first_half = mdf.iloc[:mid_idx]
        second_half = mdf.iloc[mid_idx:]

        if len(first_half) < 10 or len(second_half) < 10:
            continue

        # Calculate coefficient of variation (std/mean) for sizes
        first_cv = first_half['count'].std() / (first_half['count'].mean() + 1)
        second_cv = second_half['count'].std() / (second_half['count'].mean() + 1)

        # Regime change: CV decreased significantly (became more clustered)
        # OR CV increased significantly (became more dispersed)
        cv_ratio = second_cv / (first_cv + 0.01)

        if cv_ratio < 0.5:  # Became MORE clustered
            # New regime is clustered - follow the clustered direction
            second_half_yes_ratio = (second_half['taker_side'] == 'yes').mean()
            new_regime_direction = 'yes' if second_half_yes_ratio > 0.55 else ('no' if second_half_yes_ratio < 0.45 else 'neutral')

            if new_regime_direction != 'neutral':
                regime_change_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'regime_type': 'to_clustered',
                    'new_direction': new_regime_direction,
                    'cv_ratio': cv_ratio,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    if len(regime_change_markets) == 0:
        print("No regime change events detected")
        return {'name': 'SPORTS-008', 'result': 'FAIL', 'reason': 'No signals found'}

    rc_df = pd.DataFrame(regime_change_markets)

    # Check following YES regime
    follow_yes = rc_df[rc_df['new_direction'] == 'yes']
    result_yes = quick_edge_check(follow_yes, side='yes', name="Follow YES Regime")

    # Check following NO regime
    follow_no = rc_df[rc_df['new_direction'] == 'no']
    result_no = quick_edge_check(follow_no, side='no', name="Follow NO Regime")

    print(f"Follow YES regime: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Follow NO regime: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    best = result_yes if (result_yes.get('edge') or -999) > (result_no.get('edge') or -999) else result_no

    return {
        'hypothesis': 'SPORTS-008',
        'name': 'Size Distribution Shape Change',
        'follow_yes_regime': result_yes,
        'follow_no_regime': result_no,
        'best': best
    }


def test_sports009_spread_widening(df):
    """
    SPORTS-009: Spread Widening Before Sharp Entry

    Signal: Period of high per-trade price moves, followed by large directional trade.
    Bet: Follow the direction of the large post-widening trade.
    """
    print("\n" + "=" * 60)
    print("SPORTS-009: SPREAD WIDENING BEFORE SHARP ENTRY")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    sharp_entry_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        mdf = mdf.reset_index(drop=True)
        prices = mdf['yes_price'].values
        sizes = mdf['count'].values
        sides = mdf['taker_side'].values

        # Calculate per-trade price moves
        price_moves = np.abs(np.diff(prices))

        if len(price_moves) < 5:
            continue

        # Look for periods of wide moves followed by large trade
        for i in range(3, len(price_moves) - 1):
            # Check if last 3 trades had above-average price moves
            recent_moves = price_moves[i-3:i]
            all_moves_avg = price_moves.mean()

            if recent_moves.mean() > all_moves_avg * 1.5:  # 50% higher volatility
                # Check if next trade is large
                next_size = sizes[i + 1]
                avg_size = sizes.mean()

                if next_size > avg_size * 2:  # 2x larger than average
                    sharp_entry_markets.append({
                        'market_ticker': market_ticker,
                        'market_result': mdf['market_result'].iloc[0],
                        'sharp_direction': sides[i + 1],
                        'volatility_ratio': recent_moves.mean() / all_moves_avg,
                        'size_ratio': next_size / avg_size,
                        'no_price': mdf['no_price'].mean(),
                        'yes_price': mdf['yes_price'].mean()
                    })
                    break  # One signal per market

    if len(sharp_entry_markets) == 0:
        print("No spread widening + sharp entry events detected")
        return {'name': 'SPORTS-009', 'result': 'FAIL', 'reason': 'No signals found'}

    se_df = pd.DataFrame(sharp_entry_markets)

    # Check following sharp YES entry
    follow_yes = se_df[se_df['sharp_direction'] == 'yes']
    result_yes = quick_edge_check(follow_yes, side='yes', name="Follow Sharp YES Entry")

    # Check following sharp NO entry
    follow_no = se_df[se_df['sharp_direction'] == 'no']
    result_no = quick_edge_check(follow_no, side='no', name="Follow Sharp NO Entry")

    print(f"Follow sharp YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Follow sharp NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    best = result_yes if (result_yes.get('edge') or -999) > (result_no.get('edge') or -999) else result_no

    return {
        'hypothesis': 'SPORTS-009',
        'name': 'Spread Widening Before Sharp Entry',
        'follow_yes_entry': result_yes,
        'follow_no_entry': result_no,
        'best': best
    }


def test_sports011_category_momentum_contagion(df):
    """
    SPORTS-011: Category Momentum Contagion

    Signal: After 3+ underdogs won in a category, bet the favorite in the next.
    (Fade the recency bias toward underdogs)
    """
    print("\n" + "=" * 60)
    print("SPORTS-011: CATEGORY MOMENTUM CONTAGION")
    print("=" * 60)

    # We need category info - extract from market_ticker prefix
    df['category'] = df['market_ticker'].str.extract(r'^(KX[A-Z]+)', expand=False)
    df_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'category': 'first',
        'datetime': 'min',  # First trade time as proxy for market start
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    # Sort by category and time
    df_markets = df_markets.sort_values(['category', 'datetime'])

    # For each market, check preceding 3 markets in same category
    contagion_markets = []

    for category in df_markets['category'].unique():
        if pd.isna(category):
            continue

        cat_markets = df_markets[df_markets['category'] == category].reset_index(drop=True)

        if len(cat_markets) < 4:
            continue

        for i in range(3, len(cat_markets)):
            # Get last 3 markets' results
            last_3 = cat_markets.iloc[i-3:i]
            current = cat_markets.iloc[i]

            # "Underdog" = NO won (assuming YES is usually the favorite)
            # This is a simplification - in reality we'd need to know the opening price
            underdog_streak = (last_3['market_result'] == 'no').sum()

            if underdog_streak >= 3:
                # 3+ underdogs won -> bet the favorite (YES) in next
                contagion_markets.append({
                    'market_ticker': current['market_ticker'],
                    'market_result': current['market_result'],
                    'category': category,
                    'underdog_streak': underdog_streak,
                    'signal': 'bet_favorite',  # Fade the recency bias
                    'no_price': current['no_price'],
                    'yes_price': current['yes_price']
                })

    if len(contagion_markets) == 0:
        print("No category momentum contagion events detected")
        return {'name': 'SPORTS-011', 'result': 'FAIL', 'reason': 'No signals found'}

    cm_df = pd.DataFrame(contagion_markets)

    # Bet YES (favorite) after underdog streak
    result = quick_edge_check(cm_df, side='yes', name="Bet Favorite After Underdog Streak")

    print(f"Bet favorite after underdog streak: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {
        'hypothesis': 'SPORTS-011',
        'name': 'Category Momentum Contagion',
        'result': result,
        'best': result
    }


# ===========================================================================
# TIER 2: TEST SECOND
# ===========================================================================

def test_sports002_opening_move_reversal(df):
    """
    SPORTS-002: Opening Move Reversal (Fade the Opener)

    Signal: First 25% of trades YES-heavy, second 25% NO-heavy (or vice versa).
    Bet: Follow the REVERSAL direction.
    """
    print("\n" + "=" * 60)
    print("SPORTS-002: OPENING MOVE REVERSAL")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    reversal_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 12:  # Need enough for quartiles
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        # First and second quarters
        q1_end = n // 4
        q2_end = n // 2

        first_quarter = mdf.iloc[:q1_end]
        second_quarter = mdf.iloc[q1_end:q2_end]

        if len(first_quarter) < 3 or len(second_quarter) < 3:
            continue

        # Calculate YES ratios
        q1_yes_ratio = (first_quarter['taker_side'] == 'yes').mean()
        q2_yes_ratio = (second_quarter['taker_side'] == 'yes').mean()

        # Reversal: Q1 was YES-heavy (>60%), Q2 is NO-heavy (<40%) or vice versa
        if q1_yes_ratio > 0.65 and q2_yes_ratio < 0.40:
            # Opener was YES, reversal to NO -> bet NO
            reversal_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'opener_direction': 'yes',
                'reversal_direction': 'no',
                'q1_yes_ratio': q1_yes_ratio,
                'q2_yes_ratio': q2_yes_ratio,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })
        elif q1_yes_ratio < 0.35 and q2_yes_ratio > 0.60:
            # Opener was NO, reversal to YES -> bet YES
            reversal_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'opener_direction': 'no',
                'reversal_direction': 'yes',
                'q1_yes_ratio': q1_yes_ratio,
                'q2_yes_ratio': q2_yes_ratio,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    if len(reversal_markets) == 0:
        print("No opening move reversal events detected")
        return {'name': 'SPORTS-002', 'result': 'FAIL', 'reason': 'No signals found'}

    rm_df = pd.DataFrame(reversal_markets)

    # Check reversal to NO
    reversal_no = rm_df[rm_df['reversal_direction'] == 'no']
    result_no = quick_edge_check(reversal_no, side='no', name="Follow Reversal to NO")

    # Check reversal to YES
    reversal_yes = rm_df[rm_df['reversal_direction'] == 'yes']
    result_yes = quick_edge_check(reversal_yes, side='yes', name="Follow Reversal to YES")

    print(f"Follow reversal to NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Follow reversal to YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    best = result_no if (result_no.get('edge') or -999) > (result_yes.get('edge') or -999) else result_yes

    return {
        'hypothesis': 'SPORTS-002',
        'name': 'Opening Move Reversal',
        'follow_reversal_no': result_no,
        'follow_reversal_yes': result_yes,
        'best': best
    }


def test_sports003_momentum_velocity_stall(df):
    """
    SPORTS-003: Momentum Velocity Stall

    Signal: YES trade velocity dropped >50% from first third to second third,
    but price stayed high (>70c).
    Bet: NO (bearish divergence)
    """
    print("\n" + "=" * 60)
    print("SPORTS-003: MOMENTUM VELOCITY STALL")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    stall_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 15:
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        # Split into thirds
        third1 = mdf.iloc[:n//3]
        third2 = mdf.iloc[n//3:2*n//3]

        if len(third1) < 3 or len(third2) < 3:
            continue

        # Calculate YES trade velocity (YES trades per minute)
        def yes_velocity(trades):
            yes_trades = trades[trades['taker_side'] == 'yes']
            if len(yes_trades) < 2:
                return 0
            duration = (trades['datetime'].max() - trades['datetime'].min()).total_seconds() / 60
            return len(yes_trades) / max(duration, 1)

        v1 = yes_velocity(third1)
        v2 = yes_velocity(third2)

        # Check for stall (>50% drop) with high price
        if v1 > 0 and v2 < v1 * 0.5:  # Velocity dropped >50%
            avg_price = mdf['yes_price'].mean()

            if avg_price > 70:  # Price still high
                stall_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'v1': v1,
                    'v2': v2,
                    'velocity_ratio': v2 / v1 if v1 > 0 else 0,
                    'avg_yes_price': avg_price,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    if len(stall_markets) == 0:
        print("No momentum velocity stall events detected")
        return {'name': 'SPORTS-003', 'result': 'FAIL', 'reason': 'No signals found'}

    stall_df = pd.DataFrame(stall_markets)

    result = quick_edge_check(stall_df, side='no', name="Bet NO on Velocity Stall")

    print(f"Bet NO on velocity stall: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {
        'hypothesis': 'SPORTS-003',
        'name': 'Momentum Velocity Stall',
        'result': result,
        'best': result
    }


def test_sports007_late_arriving_large(df):
    """
    SPORTS-007: Late-Arriving Large Money (Closing Line Hunt)

    Signal: Final 25% of trades has 2x ratio of large trades compared to earlier.
    Bet: Follow the direction of the late large trades.
    """
    print("\n" + "=" * 60)
    print("SPORTS-007: LATE-ARRIVING LARGE MONEY")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    late_large_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 16:  # Need enough for quartiles
            continue

        mdf = mdf.reset_index(drop=True)
        n = len(mdf)

        # Early (first 75%) and late (final 25%)
        cutoff = 3 * n // 4
        early = mdf.iloc[:cutoff]
        late = mdf.iloc[cutoff:]

        if len(late) < 4:
            continue

        # Define "large" as >$50 value
        large_threshold = 5000  # $50 = 5000 cents

        early_large_ratio = (early['trade_value_cents'] > large_threshold).mean()
        late_large_ratio = (late['trade_value_cents'] > large_threshold).mean()

        # Check if late has 2x the large trade ratio
        if late_large_ratio > early_large_ratio * 2 and late_large_ratio > 0.2:
            # What direction are the late large trades?
            late_large = late[late['trade_value_cents'] > large_threshold]
            if len(late_large) < 2:
                continue

            late_yes_ratio = (late_large['taker_side'] == 'yes').mean()
            late_direction = 'yes' if late_yes_ratio > 0.6 else ('no' if late_yes_ratio < 0.4 else 'neutral')

            if late_direction != 'neutral':
                late_large_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'late_direction': late_direction,
                    'early_large_ratio': early_large_ratio,
                    'late_large_ratio': late_large_ratio,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    if len(late_large_markets) == 0:
        print("No late-arriving large money events detected")
        return {'name': 'SPORTS-007', 'result': 'FAIL', 'reason': 'No signals found'}

    ll_df = pd.DataFrame(late_large_markets)

    # Check following late YES
    follow_yes = ll_df[ll_df['late_direction'] == 'yes']
    result_yes = quick_edge_check(follow_yes, side='yes', name="Follow Late Large YES")

    # Check following late NO
    follow_no = ll_df[ll_df['late_direction'] == 'no']
    result_no = quick_edge_check(follow_no, side='no', name="Follow Late Large NO")

    print(f"Follow late large YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Follow late large NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    best = result_yes if (result_yes.get('edge') or -999) > (result_no.get('edge') or -999) else result_no

    return {
        'hypothesis': 'SPORTS-007',
        'name': 'Late-Arriving Large Money',
        'follow_yes': result_yes,
        'follow_no': result_no,
        'best': best
    }


def test_sports012_ncaaf_totals(df):
    """
    SPORTS-012: NCAAF Totals Specialist

    NCAAFTOTAL markets showed +22.5% edge in Session 009.
    Apply existing validated signals (RLM, etc) specifically to this category.
    """
    print("\n" + "=" * 60)
    print("SPORTS-012: NCAAF TOTALS SPECIALIST")
    print("=" * 60)

    # Filter to NCAAFTOTAL markets
    ncaaf_total = df[df['market_ticker'].str.contains('NCAAFTOTAL', case=False, na=False)]

    if len(ncaaf_total) == 0:
        # Try alternative patterns
        ncaaf_total = df[df['market_ticker'].str.contains('KXNCAAF', case=False, na=False)]

    if len(ncaaf_total) == 0:
        print("No NCAAF totals markets found in data")
        return {'name': 'SPORTS-012', 'result': 'FAIL', 'reason': 'No NCAAF markets in data'}

    # Get unique markets
    ncaaf_markets = ncaaf_total.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    # Just check base NO strategy for NCAAF
    result = quick_edge_check(ncaaf_markets, side='no', name="NCAAF Totals Bet NO")

    print(f"NCAAF Totals bet NO: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {
        'hypothesis': 'SPORTS-012',
        'name': 'NCAAF Totals Specialist',
        'result': result,
        'best': result
    }


# ===========================================================================
# TIER 3: LSD ABSURD
# ===========================================================================

def test_sports004_extreme_public_fade(df):
    """
    SPORTS-004: Extreme Public Sentiment Fade

    Signal: >90% of trades are YES, but price is 40-80c (not a sure thing).
    Bet: NO (fade the extreme public)
    """
    print("\n" + "=" * 60)
    print("SPORTS-004: EXTREME PUBLIC SENTIMENT FADE")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    extreme_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        yes_ratio = (mdf['taker_side'] == 'yes').mean()
        avg_yes_price = mdf['yes_price'].mean()

        # Extreme public: >90% YES trades, but price 40-80c
        if yes_ratio > 0.90 and 40 <= avg_yes_price <= 80:
            extreme_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'yes_ratio': yes_ratio,
                'avg_yes_price': avg_yes_price,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    if len(extreme_markets) == 0:
        print("No extreme public sentiment events detected")
        return {'name': 'SPORTS-004', 'result': 'FAIL', 'reason': 'No signals found'}

    ep_df = pd.DataFrame(extreme_markets)

    result = quick_edge_check(ep_df, side='no', name="Fade Extreme Public (bet NO)")

    print(f"Fade extreme public: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {
        'hypothesis': 'SPORTS-004',
        'name': 'Extreme Public Sentiment Fade',
        'result': result,
        'best': result
    }


def test_sports006_round_number_clustering(df):
    """
    SPORTS-006: Round Number Retail Clustering

    Signal: >40% of trades at round YES prices (25c, 50c, 75c within 2c).
    Bet: If retail-dominated and YES-heavy, bet NO.
    """
    print("\n" + "=" * 60)
    print("SPORTS-006: ROUND NUMBER RETAIL CLUSTERING")
    print("=" * 60)

    ROUND_PRICES = [25, 50, 75]
    TOLERANCE = 2

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    # Check if price is near a round number
    def is_near_round(price):
        for rp in ROUND_PRICES:
            if abs(price - rp) <= TOLERANCE:
                return True
        return False

    df_sorted['is_round_price'] = df_sorted['yes_price'].apply(is_near_round)

    clustering_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        round_ratio = mdf['is_round_price'].mean()
        yes_ratio = (mdf['taker_side'] == 'yes').mean()

        # Retail clustering: >40% at round prices, and YES-heavy
        if round_ratio > 0.40 and yes_ratio > 0.6:
            clustering_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'round_ratio': round_ratio,
                'yes_ratio': yes_ratio,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    if len(clustering_markets) == 0:
        print("No round number clustering events detected")
        return {'name': 'SPORTS-006', 'result': 'FAIL', 'reason': 'No signals found'}

    rc_df = pd.DataFrame(clustering_markets)

    result = quick_edge_check(rc_df, side='no', name="Fade Round Number Retail (bet NO)")

    print(f"Fade round number retail: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {
        'hypothesis': 'SPORTS-006',
        'name': 'Round Number Retail Clustering',
        'result': result,
        'best': result
    }


def test_sports013_trade_count_milestone(df):
    """
    SPORTS-013: Trade Count Milestone Fading

    Signal: At 100/500/1000 trades, track milestone direction and next 5 trades.
    Bet: Fade if momentum reversed after milestone.
    """
    print("\n" + "=" * 60)
    print("SPORTS-013: TRADE COUNT MILESTONE FADING")
    print("=" * 60)

    MILESTONES = [100, 500, 1000]

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    milestone_fades = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        mdf = mdf.reset_index(drop=True)

        for milestone in MILESTONES:
            if len(mdf) <= milestone + 5:
                continue

            # Milestone trade direction
            milestone_side = mdf.iloc[milestone - 1]['taker_side']

            # Next 5 trades after milestone
            next_5 = mdf.iloc[milestone:milestone + 5]
            next_5_ratio = (next_5['taker_side'] == milestone_side).mean()

            # If next 5 trades reversed (<40% same direction as milestone)
            if next_5_ratio < 0.40:
                fade_direction = 'no' if milestone_side == 'yes' else 'yes'

                milestone_fades.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'milestone': milestone,
                    'milestone_side': milestone_side,
                    'fade_direction': fade_direction,
                    'reversal_strength': 1 - next_5_ratio,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })
                break  # One milestone per market

    if len(milestone_fades) == 0:
        print("No milestone fade events detected")
        return {'name': 'SPORTS-013', 'result': 'FAIL', 'reason': 'No signals found'}

    mf_df = pd.DataFrame(milestone_fades)

    # Check fade to NO
    fade_no = mf_df[mf_df['fade_direction'] == 'no']
    result_no = quick_edge_check(fade_no, side='no', name="Milestone Fade to NO")

    # Check fade to YES
    fade_yes = mf_df[mf_df['fade_direction'] == 'yes']
    result_yes = quick_edge_check(fade_yes, side='yes', name="Milestone Fade to YES")

    print(f"Milestone fade to NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Milestone fade to YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    best = result_no if (result_no.get('edge') or -999) > (result_yes.get('edge') or -999) else result_yes

    return {
        'hypothesis': 'SPORTS-013',
        'name': 'Trade Count Milestone Fading',
        'fade_no': result_no,
        'fade_yes': result_yes,
        'best': best
    }


def test_sports014_bot_signature_fade(df):
    """
    SPORTS-014: Bot Signature Fade (Clock-Like Trading)

    Signal: Low coefficient of variation in inter-trade times = bot-dominated.
    Bet: FADE the bot direction.
    """
    print("\n" + "=" * 60)
    print("SPORTS-014: BOT SIGNATURE FADE")
    print("=" * 60)

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    bot_fade_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 20:
            continue

        mdf = mdf.reset_index(drop=True)

        # Calculate inter-trade times
        times = mdf['timestamp_ms'].values
        intervals = np.diff(times)

        if len(intervals) < 10:
            continue

        # Coefficient of variation
        cv = intervals.std() / (intervals.mean() + 1)

        # Bot signature: CV < 0.3 (very consistent timing)
        if cv < 0.3:
            # What direction are the bots trading?
            yes_ratio = (mdf['taker_side'] == 'yes').mean()
            bot_direction = 'yes' if yes_ratio > 0.55 else ('no' if yes_ratio < 0.45 else 'neutral')

            if bot_direction != 'neutral':
                fade_direction = 'no' if bot_direction == 'yes' else 'yes'

                bot_fade_markets.append({
                    'market_ticker': market_ticker,
                    'market_result': mdf['market_result'].iloc[0],
                    'cv': cv,
                    'bot_direction': bot_direction,
                    'fade_direction': fade_direction,
                    'no_price': mdf['no_price'].mean(),
                    'yes_price': mdf['yes_price'].mean()
                })

    if len(bot_fade_markets) == 0:
        print("No bot signature events detected")
        return {'name': 'SPORTS-014', 'result': 'FAIL', 'reason': 'No signals found'}

    bf_df = pd.DataFrame(bot_fade_markets)

    # Check fade to NO
    fade_no = bf_df[bf_df['fade_direction'] == 'no']
    result_no = quick_edge_check(fade_no, side='no', name="Fade Bot to NO")

    # Check fade to YES
    fade_yes = bf_df[bf_df['fade_direction'] == 'yes']
    result_yes = quick_edge_check(fade_yes, side='yes', name="Fade Bot to YES")

    print(f"Fade bot to NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Fade bot to YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    best = result_no if (result_no.get('edge') or -999) > (result_yes.get('edge') or -999) else result_yes

    return {
        'hypothesis': 'SPORTS-014',
        'name': 'Bot Signature Fade',
        'fade_no': result_no,
        'fade_yes': result_yes,
        'best': best
    }


def test_sports015_fibonacci_price_attractors(df):
    """
    SPORTS-015: Fibonacci Price Attractors (Magic Levels)

    Signal: Price near Fib levels (23.6c, 38.2c, 50c, 61.8c, 76.4c) within 2c.
    Bet: Based on bounce/break patterns.
    """
    print("\n" + "=" * 60)
    print("SPORTS-015: FIBONACCI PRICE ATTRACTORS")
    print("=" * 60)

    FIB_LEVELS = [23.6, 38.2, 50.0, 61.8, 76.4]
    TOLERANCE = 2

    df_sorted = df.sort_values(['market_ticker', 'datetime']).copy()

    def near_fib(price):
        for fib in FIB_LEVELS:
            if abs(price - fib) <= TOLERANCE:
                return fib
        return None

    fib_markets = []

    for market_ticker, mdf in df_sorted.groupby('market_ticker'):
        if len(mdf) < 10:
            continue

        mdf = mdf.reset_index(drop=True)

        # Check how much time price spent near Fib levels
        mdf['near_fib'] = mdf['yes_price'].apply(near_fib)
        fib_touches = mdf['near_fib'].notna().sum()
        fib_ratio = fib_touches / len(mdf)

        # High Fib interaction (>30% of trades near Fib levels)
        if fib_ratio > 0.30:
            # Did price "bounce" or "break" the main Fib level?
            main_fib = mdf['near_fib'].mode().iloc[0] if len(mdf['near_fib'].mode()) > 0 else None

            if main_fib is None:
                continue

            # Simple heuristic: if ended higher than main Fib, "bounced up"
            final_price = mdf['yes_price'].iloc[-1]

            if final_price > main_fib + 5:
                signal = 'bounced_up'
                bet_direction = 'yes'
            elif final_price < main_fib - 5:
                signal = 'bounced_down'
                bet_direction = 'no'
            else:
                continue

            fib_markets.append({
                'market_ticker': market_ticker,
                'market_result': mdf['market_result'].iloc[0],
                'main_fib': main_fib,
                'fib_ratio': fib_ratio,
                'signal': signal,
                'bet_direction': bet_direction,
                'no_price': mdf['no_price'].mean(),
                'yes_price': mdf['yes_price'].mean()
            })

    if len(fib_markets) == 0:
        print("No Fibonacci attractor events detected")
        return {'name': 'SPORTS-015', 'result': 'FAIL', 'reason': 'No signals found'}

    fib_df = pd.DataFrame(fib_markets)

    # Check bounced up (bet YES)
    bounce_up = fib_df[fib_df['bet_direction'] == 'yes']
    result_yes = quick_edge_check(bounce_up, side='yes', name="Fib Bounce Up (bet YES)")

    # Check bounced down (bet NO)
    bounce_down = fib_df[fib_df['bet_direction'] == 'no']
    result_no = quick_edge_check(bounce_down, side='no', name="Fib Bounce Down (bet NO)")

    print(f"Fib bounce up: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Fib bounce down: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    best = result_yes if (result_yes.get('edge') or -999) > (result_no.get('edge') or -999) else result_no

    return {
        'hypothesis': 'SPORTS-015',
        'name': 'Fibonacci Price Attractors',
        'bounce_up': result_yes,
        'bounce_down': result_no,
        'best': best
    }


def test_sports010_multi_outcome_inconsistency(df):
    """
    SPORTS-010: Multi-Outcome Pricing Inconsistency

    NOTE: This requires market metadata to identify related markets.
    We'll attempt a proxy using ticker patterns.
    """
    print("\n" + "=" * 60)
    print("SPORTS-010: MULTI-OUTCOME PRICING INCONSISTENCY")
    print("=" * 60)

    print("SKIPPED: Requires market metadata to identify complementary markets.")
    print("Data Issue: Cannot identify related markets from trade data alone.")

    return {
        'hypothesis': 'SPORTS-010',
        'name': 'Multi-Outcome Pricing Inconsistency',
        'result': 'SKIPPED',
        'reason': 'Requires market metadata',
        'best': {'flag': 'SKIP', 'reason': 'Data issue'}
    }


def main():
    """Run all SPORTS hypothesis screens."""
    df = load_data()

    all_results = {}

    # TIER 1
    print("\n" + "=" * 80)
    print("TIER 1: HIGHEST PRIORITY HYPOTHESES")
    print("=" * 80)

    all_results['SPORTS-001'] = test_sports001_steam_exhaustion(df)
    all_results['SPORTS-005'] = test_sports005_size_velocity_divergence(df)
    all_results['SPORTS-008'] = test_sports008_size_distribution_change(df)
    all_results['SPORTS-009'] = test_sports009_spread_widening(df)
    all_results['SPORTS-011'] = test_sports011_category_momentum_contagion(df)

    # TIER 2
    print("\n" + "=" * 80)
    print("TIER 2: GOOD POTENTIAL HYPOTHESES")
    print("=" * 80)

    all_results['SPORTS-002'] = test_sports002_opening_move_reversal(df)
    all_results['SPORTS-003'] = test_sports003_momentum_velocity_stall(df)
    all_results['SPORTS-007'] = test_sports007_late_arriving_large(df)
    all_results['SPORTS-012'] = test_sports012_ncaaf_totals(df)

    # TIER 3
    print("\n" + "=" * 80)
    print("TIER 3: LSD ABSURD HYPOTHESES")
    print("=" * 80)

    all_results['SPORTS-004'] = test_sports004_extreme_public_fade(df)
    all_results['SPORTS-006'] = test_sports006_round_number_clustering(df)
    all_results['SPORTS-013'] = test_sports013_trade_count_milestone(df)
    all_results['SPORTS-014'] = test_sports014_bot_signature_fade(df)
    all_results['SPORTS-015'] = test_sports015_fibonacci_price_attractors(df)
    all_results['SPORTS-010'] = test_sports010_multi_outcome_inconsistency(df)

    # ===========================================================================
    # SUMMARY
    # ===========================================================================

    print("\n" + "=" * 80)
    print("LSD SCREENING SUMMARY")
    print("=" * 80)

    winners = []
    promising = []
    dead = []

    for hyp_id, result in all_results.items():
        if 'best' not in result:
            dead.append({'id': hyp_id, 'name': result.get('name', 'Unknown'), 'reason': result.get('reason', 'No data')})
            continue

        best = result['best']
        if isinstance(best, dict):
            edge = best.get('edge')
            flag = best.get('flag')
            n = best.get('n', 0)

            if edge is None:
                dead.append({'id': hyp_id, 'name': result['name'], 'reason': 'Insufficient data'})
            elif flag == 'PROMISING':
                winners.append({'id': hyp_id, 'name': result['name'], 'edge': edge, 'n': n, 'p': best.get('p_value', 'N/A')})
            elif flag == 'WEAK':
                promising.append({'id': hyp_id, 'name': result['name'], 'edge': edge, 'n': n, 'p': best.get('p_value', 'N/A')})
            else:
                dead.append({'id': hyp_id, 'name': result['name'], 'edge': edge, 'n': n, 'reason': 'Low edge'})

    print("\n### WINNERS (Raw Edge > 5%) ###")
    if winners:
        for w in sorted(winners, key=lambda x: x.get('edge', 0), reverse=True):
            print(f"  {w['id']}: {w['name']} - Edge={w['edge']:.1%}, N={w['n']}")
    else:
        print("  None found")

    print("\n### PROMISING (Edge 3-5%) ###")
    if promising:
        for p in sorted(promising, key=lambda x: x.get('edge', 0), reverse=True):
            print(f"  {p['id']}: {p['name']} - Edge={p['edge']:.1%}, N={p['n']}")
    else:
        print("  None found")

    print("\n### DEAD (Failed) ###")
    for d in dead:
        edge_str = f"Edge={d.get('edge', 0):.1%}" if d.get('edge') else ""
        print(f"  {d['id']}: {d['name']} - {d.get('reason', '')} {edge_str}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'winners': winners,
        'promising': promising,
        'dead': dead,
        'full_results': {k: str(v) for k, v in all_results.items()}  # Convert for JSON
    }

    output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/sports_expert_lsd_screening.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return all_results, winners, promising, dead


if __name__ == "__main__":
    main()
