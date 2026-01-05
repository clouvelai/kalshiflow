"""
LSD SESSION 001: Lateral Strategy Discovery - MAXIMUM DOSE

The winning strategy is hiding where nobody looked. Let's get WEIRD.

This session tests:
1. 14 incoming hypotheses (EXT-001 to EXT-009, LSD-001 to LSD-005)
2. MY OWN absurd ideas that nobody would think to test

Speed over rigor. Flag anything with >5% edge for deeper investigation.

"If the winning strategy was obvious we'd all be dancing on the moon."
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

# Fibonacci numbers up to reasonable trade counts
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Prime numbers up to 100
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

PRIMES = [n for n in range(2, 200) if is_prime(n)]

# Round trade sizes (bot signatures)
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]

# Palindrome trade sizes
def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

PALINDROMES = [n for n in range(1, 10000) if is_palindrome(n)]

results = {}


def load_data():
    """Load the enriched trades data."""
    print("=" * 80)
    print("LSD SESSION 001: LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Add useful columns
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= 10000  # >= $100
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)

    # Price ratios
    df['yes_no_ratio'] = df['yes_price'] / (df['no_price'] + 0.1)  # Avoid div by 0

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def build_baseline_10c(df):
    """Build baseline win rates at 10c buckets for quick screening."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()

    all_markets['bucket_10c'] = (all_markets['no_price'] // 10) * 10

    baseline_no = {}
    baseline_yes = {}

    for bucket in sorted(all_markets['bucket_10c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_10c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline_no[bucket] = (bucket_markets['market_result'] == 'no').mean()
            baseline_yes[bucket] = (bucket_markets['market_result'] == 'yes').mean()

    return all_markets, baseline_no, baseline_yes


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

    flag = edge > 0.05 and p_value < 0.05

    return {
        'name': name,
        'n': n,
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'flag': flag,
        'flag_reason': 'POTENTIAL EDGE >5%' if flag else None
    }


# ===========================================================================
# EXT HYPOTHESES (External Research)
# ===========================================================================

def test_ext001_early_trade_premium(df):
    """
    EXT-001: Early Trade Premium (Closing Line Value)

    Trades in first 25% of market activity may have CLV edge.
    """
    print("\n" + "=" * 60)
    print("EXT-001: EARLY TRADE PREMIUM (CLV)")
    print("=" * 60)

    # Calculate trade position within each market
    df_sorted = df.sort_values(['market_ticker', 'datetime'])
    df_sorted['trade_num'] = df_sorted.groupby('market_ticker').cumcount() + 1
    df_sorted['trade_total'] = df_sorted.groupby('market_ticker')['trade_num'].transform('max')
    df_sorted['trade_position'] = df_sorted['trade_num'] / df_sorted['trade_total']

    # Early trades (first 25%)
    early_trades = df_sorted[df_sorted['trade_position'] <= 0.25]

    # Get markets with early YES majority
    early_yes_majority = early_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean() > 0.6,
        'market_result': 'first',
        'yes_price': 'mean',
        'no_price': 'mean'
    }).reset_index()
    early_yes_majority.columns = ['market_ticker', 'early_yes_majority', 'market_result', 'yes_price', 'no_price']

    signal_yes = early_yes_majority[early_yes_majority['early_yes_majority'] == True]
    signal_no = early_yes_majority[early_yes_majority['early_yes_majority'] == False]

    result_yes = quick_edge_check(signal_yes, side='yes', name="Early YES Majority")
    result_no = quick_edge_check(signal_no, side='no', name="Early NO Majority")

    print(f"Early YES Majority: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Early NO Majority: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    return {'early_yes': result_yes, 'early_no': result_no}


def test_ext002_steam_cascade(df):
    """
    EXT-002: Steam Move / Cascade Detection

    5+ trades in same direction within 60 seconds, causing >5c price move.
    """
    print("\n" + "=" * 60)
    print("EXT-002: STEAM CASCADE")
    print("=" * 60)

    # Sort by market and time
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    steam_markets = []

    for market_ticker, market_df in df_sorted.groupby('market_ticker'):
        if len(market_df) < 5:
            continue

        market_df = market_df.reset_index(drop=True)

        # Look for bursts: 5+ same-direction trades in 60 seconds
        for i in range(len(market_df) - 4):
            window = market_df.iloc[i:i+5]
            time_span = (window['datetime'].max() - window['datetime'].min()).total_seconds()

            if time_span <= 60:
                # Check if all same direction
                sides = window['taker_side'].unique()
                if len(sides) == 1:
                    # Check price move
                    price_move = abs(window['yes_price'].iloc[-1] - window['yes_price'].iloc[0])
                    if price_move >= 5:
                        steam_markets.append({
                            'market_ticker': market_ticker,
                            'steam_direction': sides[0],
                            'price_move': price_move,
                            'market_result': market_df['market_result'].iloc[0],
                            'no_price': market_df['no_price'].mean(),
                            'yes_price': market_df['yes_price'].mean()
                        })
                        break  # One steam event per market

    if not steam_markets:
        print("No steam events found")
        return {'n': 0, 'flag': False}

    steam_df = pd.DataFrame(steam_markets)

    # Follow steam direction
    steam_df['followed_correctly'] = steam_df['steam_direction'] == steam_df['market_result']

    n = len(steam_df)
    wins = steam_df['followed_correctly'].sum()
    wr = wins / n
    edge = wr - 0.5  # Assume 50% breakeven for following

    print(f"Steam events: {n}")
    print(f"Following steam WR: {wr:.1%}")
    print(f"Edge vs 50%: {edge*100:.2f}%")

    flag = edge > 0.05 and n >= 50

    return {
        'name': 'Steam Cascade (Follow)',
        'n': n,
        'win_rate': float(wr),
        'edge': float(edge),
        'flag': flag
    }


def test_ext003_reverse_line_movement(df):
    """
    EXT-003: Reverse Line Movement (RLM)

    >70% of trades are YES but price moved toward NO (or vice versa).
    Follow the price direction, not the trade count direction.
    """
    print("\n" + "=" * 60)
    print("EXT-003: REVERSE LINE MOVEMENT")
    print("=" * 60)

    # Sort by market and time to get first/last correctly
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate per-market stats
    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),  # YES trade ratio
        'yes_price': ['first', 'last', 'mean'],  # Price movement
        'no_price': 'mean',
        'market_result': 'first',
        'count': 'size'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'yes_trade_ratio', 'first_yes_price', 'last_yes_price', 'avg_yes_price', 'no_price', 'market_result', 'n_trades']

    market_stats['price_moved_yes'] = market_stats['last_yes_price'] > market_stats['first_yes_price']
    market_stats['price_moved_no'] = market_stats['last_yes_price'] < market_stats['first_yes_price']

    # Add yes_price column for quick_edge_check
    market_stats['yes_price'] = market_stats['avg_yes_price']

    # RLM: Majority YES trades but price moved toward NO
    rlm_no = market_stats[
        (market_stats['yes_trade_ratio'] > 0.7) &  # >70% YES trades
        (market_stats['price_moved_no']) &  # But price moved toward NO
        (market_stats['n_trades'] >= 5)
    ].copy()

    # RLM: Majority NO trades but price moved toward YES
    rlm_yes = market_stats[
        (market_stats['yes_trade_ratio'] < 0.3) &  # >70% NO trades
        (market_stats['price_moved_yes']) &  # But price moved toward YES
        (market_stats['n_trades'] >= 5)
    ].copy()

    result_no = quick_edge_check(rlm_no, side='no', name="RLM -> Bet NO")
    result_yes = quick_edge_check(rlm_yes, side='yes', name="RLM -> Bet YES")

    print(f"RLM NO signal: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"RLM YES signal: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'rlm_no': result_no, 'rlm_yes': result_yes}


def test_ext004_vpin_flow_toxicity(df):
    """
    EXT-004: VPIN-Style Order Flow Toxicity

    High toxicity = high probability of informed trading.
    Measure trade imbalance VELOCITY and ACCELERATION.
    """
    print("\n" + "=" * 60)
    print("EXT-004: VPIN FLOW TOXICITY (Acceleration)")
    print("=" * 60)

    # Calculate per-market flow stats
    market_stats = df.groupby('market_ticker').apply(
        lambda x: pd.Series({
            'yes_count': (x['taker_side'] == 'yes').sum(),
            'no_count': (x['taker_side'] == 'no').sum(),
            'market_result': x['market_result'].iloc[0],
            'no_price': x['no_price'].mean(),
            'yes_price': x['yes_price'].mean(),
            'n_trades': len(x)
        })
    ).reset_index()

    # Calculate imbalance ratio
    market_stats['imbalance'] = abs(market_stats['yes_count'] - market_stats['no_count']) / (market_stats['yes_count'] + market_stats['no_count'])

    # High imbalance (toxicity proxy)
    high_imbalance = market_stats[(market_stats['imbalance'] > 0.7) & (market_stats['n_trades'] >= 5)].copy()

    # Determine which side has the flow
    high_imbalance['flow_yes'] = high_imbalance['yes_count'] > high_imbalance['no_count']

    flow_yes = high_imbalance[high_imbalance['flow_yes']]
    flow_no = high_imbalance[~high_imbalance['flow_yes']]

    result_yes = quick_edge_check(flow_yes, side='yes', name="High Imbalance YES Flow")
    result_no = quick_edge_check(flow_no, side='no', name="High Imbalance NO Flow")

    print(f"High imbalance YES flow: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"High imbalance NO flow: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    return {'vpin_yes': result_yes, 'vpin_no': result_no}


def test_ext005_buyback_reversal(df):
    """
    EXT-005: Buy-Back Trap / Late Reversal Signal

    First half of trades favor side A, second half favors side B with larger size.
    """
    print("\n" + "=" * 60)
    print("EXT-005: BUYBACK REVERSAL")
    print("=" * 60)

    reversal_markets = []

    for market_ticker, market_df in df.groupby('market_ticker'):
        if len(market_df) < 6:  # Need at least 6 trades for meaningful split
            continue

        market_df = market_df.sort_values('datetime').reset_index(drop=True)
        mid = len(market_df) // 2

        first_half = market_df.iloc[:mid]
        second_half = market_df.iloc[mid:]

        first_yes_ratio = (first_half['taker_side'] == 'yes').mean()
        second_yes_ratio = (second_half['taker_side'] == 'yes').mean()

        first_avg_size = first_half['count'].mean()
        second_avg_size = second_half['count'].mean()

        # Reversal: first half YES-heavy, second half NO-heavy with larger size
        if first_yes_ratio > 0.6 and second_yes_ratio < 0.4 and second_avg_size > first_avg_size:
            reversal_markets.append({
                'market_ticker': market_ticker,
                'reversal_direction': 'to_no',
                'market_result': market_df['market_result'].iloc[0],
                'no_price': market_df['no_price'].mean(),
                'yes_price': market_df['yes_price'].mean()
            })

        # Reversal: first half NO-heavy, second half YES-heavy with larger size
        if first_yes_ratio < 0.4 and second_yes_ratio > 0.6 and second_avg_size > first_avg_size:
            reversal_markets.append({
                'market_ticker': market_ticker,
                'reversal_direction': 'to_yes',
                'market_result': market_df['market_result'].iloc[0],
                'no_price': market_df['no_price'].mean(),
                'yes_price': market_df['yes_price'].mean()
            })

    if not reversal_markets:
        print("No reversal patterns found")
        return {'n': 0, 'flag': False}

    reversal_df = pd.DataFrame(reversal_markets)

    # Follow reversal direction
    to_no = reversal_df[reversal_df['reversal_direction'] == 'to_no']
    to_yes = reversal_df[reversal_df['reversal_direction'] == 'to_yes']

    result_no = quick_edge_check(to_no, side='no', name="Reversal to NO")
    result_yes = quick_edge_check(to_yes, side='yes', name="Reversal to YES")

    print(f"Reversal to NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Reversal to YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'reversal_no': result_no, 'reversal_yes': result_yes}


def test_ext006_surprise_mispricing(df):
    """
    EXT-006: Surprise Event Mispricing

    Price moves >10c in <10 trades AGAINST the previous trend direction.
    Fade the surprise move.
    """
    print("\n" + "=" * 60)
    print("EXT-006: SURPRISE MISPRICING (Fade Surprise)")
    print("=" * 60)

    surprise_markets = []

    for market_ticker, market_df in df.groupby('market_ticker'):
        if len(market_df) < 15:  # Need enough trades for trend + surprise
            continue

        market_df = market_df.sort_values('datetime').reset_index(drop=True)

        # Look for surprise moves
        for i in range(10, len(market_df) - 10):
            # Previous 10 trades trend
            prev_window = market_df.iloc[i-10:i]
            prev_price_move = prev_window['yes_price'].iloc[-1] - prev_window['yes_price'].iloc[0]
            prev_trend = 'up' if prev_price_move > 3 else 'down' if prev_price_move < -3 else 'flat'

            if prev_trend == 'flat':
                continue

            # Next 10 trades surprise
            next_window = market_df.iloc[i:i+10]
            next_price_move = next_window['yes_price'].iloc[-1] - next_window['yes_price'].iloc[0]

            # Surprise = opposite direction and large move
            if prev_trend == 'up' and next_price_move < -10:
                surprise_markets.append({
                    'market_ticker': market_ticker,
                    'surprise_direction': 'down',  # Surprise was down, fade = bet YES
                    'market_result': market_df['market_result'].iloc[0],
                    'no_price': market_df['no_price'].mean(),
                    'yes_price': market_df['yes_price'].mean()
                })
                break
            elif prev_trend == 'down' and next_price_move > 10:
                surprise_markets.append({
                    'market_ticker': market_ticker,
                    'surprise_direction': 'up',  # Surprise was up, fade = bet NO
                    'market_result': market_df['market_result'].iloc[0],
                    'no_price': market_df['no_price'].mean(),
                    'yes_price': market_df['yes_price'].mean()
                })
                break

    if not surprise_markets:
        print("No surprise events found")
        return {'n': 0, 'flag': False}

    surprise_df = pd.DataFrame(surprise_markets)

    # Fade surprise: surprise down -> bet YES, surprise up -> bet NO
    fade_yes = surprise_df[surprise_df['surprise_direction'] == 'down']
    fade_no = surprise_df[surprise_df['surprise_direction'] == 'up']

    result_yes = quick_edge_check(fade_yes, side='yes', name="Fade Surprise (bet YES)")
    result_no = quick_edge_check(fade_no, side='no', name="Fade Surprise (bet NO)")

    print(f"Fade surprise down (bet YES): N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Fade surprise up (bet NO): N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")

    return {'fade_yes': result_yes, 'fade_no': result_no}


# ===========================================================================
# LSD ABSURD HYPOTHESES
# ===========================================================================

def test_lsd001_fibonacci_trade_count(df, all_markets):
    """
    LSD-001: Fibonacci Trade Count Signal

    Markets with exactly 5, 8, 13, 21, or 34 trades.
    Maybe the universe knows something.
    """
    print("\n" + "=" * 60)
    print("LSD-001: FIBONACCI TRADE COUNT")
    print("=" * 60)

    # Get trade counts per market
    trade_counts = df.groupby('market_ticker').size().reset_index(name='n_trades')
    trade_counts = trade_counts.merge(
        df.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'taker_side': lambda x: (x == 'no').mean()
        }).reset_index(),
        on='market_ticker'
    )
    trade_counts.columns = ['market_ticker', 'n_trades', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    # Fibonacci markets
    fib_markets = trade_counts[trade_counts['n_trades'].isin(FIBONACCI)].copy()
    non_fib_markets = trade_counts[~trade_counts['n_trades'].isin(FIBONACCI)].copy()

    # Follow majority direction in Fibonacci markets
    fib_no_majority = fib_markets[fib_markets['no_ratio'] > 0.5]
    fib_yes_majority = fib_markets[fib_markets['no_ratio'] <= 0.5]

    result_no = quick_edge_check(fib_no_majority, side='no', name="Fib + NO Majority")
    result_yes = quick_edge_check(fib_yes_majority, side='yes', name="Fib + YES Majority")

    # Non-Fibonacci comparison
    non_fib_no = quick_edge_check(non_fib_markets[non_fib_markets['no_ratio'] > 0.5], side='no', name="Non-Fib + NO Majority")

    print(f"Fibonacci markets: {len(fib_markets)}")
    print(f"Fib + NO Majority: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"Fib + YES Majority: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")
    print(f"Non-Fib + NO Majority: N={non_fib_no['n']}, Edge={non_fib_no.get('edge', 'N/A')}")

    return {'fib_no': result_no, 'fib_yes': result_yes, 'non_fib_no': non_fib_no}


def test_lsd002_prime_trade_count(df, all_markets):
    """
    LSD-002: Prime Number Trade Count

    Markets with a prime number of trades.
    """
    print("\n" + "=" * 60)
    print("LSD-002: PRIME TRADE COUNT")
    print("=" * 60)

    trade_counts = df.groupby('market_ticker').size().reset_index(name='n_trades')
    trade_counts = trade_counts.merge(
        df.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'taker_side': lambda x: (x == 'no').mean()
        }).reset_index(),
        on='market_ticker'
    )
    trade_counts.columns = ['market_ticker', 'n_trades', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    prime_markets = trade_counts[trade_counts['n_trades'].isin(PRIMES)].copy()

    prime_no_majority = prime_markets[prime_markets['no_ratio'] > 0.5]

    result = quick_edge_check(prime_no_majority, side='no', name="Prime + NO Majority")

    print(f"Prime-count markets: {len(prime_markets)}")
    print(f"Prime + NO Majority: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return result


def test_lsd003_worst_strategy_inverse(df, all_markets):
    """
    LSD-003: Inverse Worst Strategy (Curse Avoidance)

    Find the WORST possible strategy and AVOID those markets.
    """
    print("\n" + "=" * 60)
    print("LSD-003: INVERSE WORST STRATEGY")
    print("=" * 60)

    # Build market features
    market_features = df.groupby('market_ticker').agg({
        'count': ['mean', 'sum'],
        'leverage_ratio': ['mean', 'std'],
        'trade_value_cents': 'mean',
        'is_whale': 'any',
        'hour': 'mean',
        'is_weekend': 'any',
        'is_round_size': 'mean',  # Ratio of round-size trades
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()
    market_features.columns = [
        'market_ticker', 'avg_size', 'total_volume', 'avg_lev', 'lev_std',
        'avg_trade_value', 'has_whale', 'avg_hour', 'is_weekend', 'round_size_ratio',
        'no_ratio', 'market_result', 'no_price', 'yes_price'
    ]
    market_features['lev_std'] = market_features['lev_std'].fillna(0)

    # Find markets where NO bet ALWAYS LOSES
    # (These are the "cursed" markets we want to avoid)
    no_always_loses = market_features[market_features['market_result'] == 'yes'].copy()

    # Find common features of these "cursed" markets
    # High leverage, small trades, momentum chasers
    cursed_profile = {
        'avg_lev': no_always_loses['avg_lev'].mean(),
        'avg_size': no_always_loses['avg_size'].mean(),
        'lev_std': no_always_loses['lev_std'].mean(),
        'round_size_ratio': no_always_loses['round_size_ratio'].mean()
    }

    print(f"Cursed market profile (where NO loses):")
    for k, v in cursed_profile.items():
        print(f"  {k}: {v:.2f}")

    # Define "cursed" markets: high leverage + small trades + low round size ratio
    # (These look like emotional retail bets)
    cursed_markets = market_features[
        (market_features['avg_lev'] > 3) &  # High leverage
        (market_features['avg_size'] < 20) &  # Small trades
        (market_features['round_size_ratio'] < 0.1)  # Not bots
    ].copy()

    # Non-cursed markets
    blessed_markets = market_features[
        ~(
            (market_features['avg_lev'] > 3) &
            (market_features['avg_size'] < 20) &
            (market_features['round_size_ratio'] < 0.1)
        )
    ].copy()

    # Test NO on blessed markets only
    blessed_no = blessed_markets[blessed_markets['no_ratio'] > 0.5]
    cursed_no = cursed_markets[cursed_markets['no_ratio'] > 0.5]

    result_blessed = quick_edge_check(blessed_no, side='no', name="Blessed (Non-Cursed) NO")
    result_cursed = quick_edge_check(cursed_no, side='no', name="Cursed NO (Avoid)")

    print(f"Cursed markets: {len(cursed_markets)}")
    print(f"Blessed markets: {len(blessed_markets)}")
    print(f"Blessed NO: N={result_blessed['n']}, Edge={result_blessed.get('edge', 'N/A')}")
    print(f"Cursed NO (avoid): N={result_cursed['n']}, Edge={result_cursed.get('edge', 'N/A')}")

    improvement = (result_blessed.get('edge', 0) or 0) - (result_cursed.get('edge', 0) or 0)
    print(f"Improvement by avoiding cursed: {improvement*100:.2f}%")

    return {'blessed_no': result_blessed, 'cursed_no': result_cursed, 'improvement': improvement}


def test_lsd004_mega_stack(df, all_markets, baseline_no):
    """
    LSD-004: 4-Signal Mega Stack

    Stack 4+ weak signals together:
    - lev_std < 0.7 (S013 base)
    - Weekend timing
    - Whale presence
    - Round-size trades
    - no_ratio > 0.6
    """
    print("\n" + "=" * 60)
    print("LSD-004: MEGA SIGNAL STACK")
    print("=" * 60)

    # Build market features
    market_features = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'is_whale': 'any',
        'is_weekend': 'any',
        'is_round_size': 'any',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size'
    }).reset_index()
    market_features.columns = [
        'market_ticker', 'lev_std', 'has_whale', 'is_weekend', 'has_round_size',
        'no_ratio', 'market_result', 'no_price', 'yes_price', 'n_trades'
    ]
    market_features['lev_std'] = market_features['lev_std'].fillna(0)

    # Apply mega stack filter
    mega_stack = market_features[
        (market_features['lev_std'] < 0.7) &
        (market_features['is_weekend'] == True) &
        (market_features['has_whale'] == True) &
        (market_features['has_round_size'] == True) &
        (market_features['no_ratio'] > 0.6) &
        (market_features['n_trades'] >= 5)
    ].copy()

    result = quick_edge_check(mega_stack, side='no', name="Mega Stack (4 signals)")

    print(f"Mega Stack markets: {result['n']}")
    print(f"Edge: {result.get('edge', 'N/A')}")

    # Also test with 3 signals
    stack_3 = market_features[
        (market_features['lev_std'] < 0.7) &
        (market_features['has_whale'] == True) &
        (market_features['no_ratio'] > 0.6) &
        (market_features['n_trades'] >= 5)
    ].copy()

    result_3 = quick_edge_check(stack_3, side='no', name="3-Signal Stack")

    print(f"3-Signal Stack: N={result_3['n']}, Edge={result_3.get('edge', 'N/A')}")

    return {'mega_stack_4': result, 'mega_stack_3': result_3}


def test_lsd005_drunk_opposite(df, all_markets):
    """
    LSD-005: What Would A Drunk Do? (Do Opposite)

    Model impulsive/emotional trades and fade them.
    """
    print("\n" + "=" * 60)
    print("LSD-005: DRUNK TRADER FADE")
    print("=" * 60)

    # Drunk trade profile: high leverage, small size, momentum chasing, late night
    df_drunk = df[
        (df['leverage_ratio'] > 3) &  # High leverage
        (df['count'] < 10) &  # Small size
        (df['hour'].isin([23, 0, 1, 2, 3])) &  # Late night
        (df['is_weekend'] == True)  # Weekend
    ].copy()

    print(f"Drunk-looking trades: {len(df_drunk):,}")

    # Get markets with drunk activity favoring one side
    drunk_markets = df_drunk.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()
    drunk_markets.columns = ['market_ticker', 'drunk_yes_ratio', 'market_result', 'no_price', 'yes_price']

    # Fade drunk YES (bet NO)
    fade_drunk_yes = drunk_markets[drunk_markets['drunk_yes_ratio'] > 0.6]

    # Fade drunk NO (bet YES)
    fade_drunk_no = drunk_markets[drunk_markets['drunk_yes_ratio'] < 0.4]

    result_fade_yes = quick_edge_check(fade_drunk_yes, side='no', name="Fade Drunk YES (bet NO)")
    result_fade_no = quick_edge_check(fade_drunk_no, side='yes', name="Fade Drunk NO (bet YES)")

    print(f"Fade Drunk YES: N={result_fade_yes['n']}, Edge={result_fade_yes.get('edge', 'N/A')}")
    print(f"Fade Drunk NO: N={result_fade_no['n']}, Edge={result_fade_no.get('edge', 'N/A')}")

    return {'fade_drunk_yes': result_fade_yes, 'fade_drunk_no': result_fade_no}


# ===========================================================================
# MY OWN WILD ABSURD HYPOTHESES
# ===========================================================================

def test_wild001_digit_count(df, all_markets):
    """
    WILD-001: Trade Size Digit Count

    What if the NUMBER of digits in trade size matters?
    1-digit (1-9), 2-digit (10-99), 3-digit (100-999), 4-digit (1000+)
    """
    print("\n" + "=" * 60)
    print("WILD-001: TRADE SIZE DIGIT COUNT")
    print("=" * 60)

    df['digit_count'] = df['count'].apply(lambda x: len(str(int(x))))

    for digits in [1, 2, 3, 4]:
        digit_trades = df[df['digit_count'] == digits]

        digit_markets = digit_trades.groupby('market_ticker').agg({
            'taker_side': lambda x: (x == 'no').mean(),
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean'
        }).reset_index()
        digit_markets.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price', 'yes_price']

        no_majority = digit_markets[digit_markets['no_ratio'] > 0.5]
        result = quick_edge_check(no_majority, side='no', name=f"{digits}-Digit Size NO")

        print(f"{digits}-digit trades -> NO majority: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return {'tested': True}


def test_wild002_exactly_7_trades(df, all_markets):
    """
    WILD-002: Markets with EXACTLY 7 trades

    7 is considered lucky in many cultures. Maybe the market gods agree.
    """
    print("\n" + "=" * 60)
    print("WILD-002: EXACTLY 7 TRADES")
    print("=" * 60)

    trade_counts = df.groupby('market_ticker').size().reset_index(name='n_trades')
    trade_counts = trade_counts.merge(
        df.groupby('market_ticker').agg({
            'market_result': 'first',
            'no_price': 'mean',
            'yes_price': 'mean',
            'taker_side': lambda x: (x == 'no').mean()
        }).reset_index(),
        on='market_ticker'
    )
    trade_counts.columns = ['market_ticker', 'n_trades', 'market_result', 'no_price', 'yes_price', 'no_ratio']

    # Exactly 7 trades
    seven_markets = trade_counts[trade_counts['n_trades'] == 7].copy()

    seven_no = seven_markets[seven_markets['no_ratio'] > 0.5]
    seven_yes = seven_markets[seven_markets['no_ratio'] <= 0.5]

    result_no = quick_edge_check(seven_no, side='no', name="Exactly 7 + NO Majority")
    result_yes = quick_edge_check(seven_yes, side='yes', name="Exactly 7 + YES Majority")

    print(f"Markets with exactly 7 trades: {len(seven_markets)}")
    print(f"7 trades + NO: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"7 trades + YES: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'seven_no': result_no, 'seven_yes': result_yes}


def test_wild003_palindrome_sizes(df, all_markets):
    """
    WILD-003: Palindrome Trade Sizes

    Trades with palindrome sizes (11, 22, 33, 101, 111...)
    """
    print("\n" + "=" * 60)
    print("WILD-003: PALINDROME TRADE SIZES")
    print("=" * 60)

    df['is_palindrome'] = df['count'].apply(lambda x: is_palindrome(int(x)))

    palindrome_trades = df[df['is_palindrome']]
    print(f"Palindrome trades: {len(palindrome_trades):,}")

    if len(palindrome_trades) == 0:
        return {'n': 0, 'flag': False}

    # Get markets with palindrome activity
    pal_markets = palindrome_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean'
    }).reset_index()
    pal_markets.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price', 'yes_price']

    pal_no = pal_markets[pal_markets['no_ratio'] > 0.5]

    result = quick_edge_check(pal_no, side='no', name="Palindrome + NO Majority")

    print(f"Palindrome NO: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return result


def test_wild004_ticker_first_letter(df, all_markets):
    """
    WILD-004: Ticker First Letter

    Do "K" tickers behave differently from "N" tickers?
    """
    print("\n" + "=" * 60)
    print("WILD-004: TICKER FIRST LETTER")
    print("=" * 60)

    all_markets['first_letter'] = all_markets['market_ticker'].str[0]

    for letter in sorted(all_markets['first_letter'].unique()):
        letter_markets = all_markets[all_markets['first_letter'] == letter]

        n = len(letter_markets)
        if n < 50:
            continue

        no_wr = (letter_markets['market_result'] == 'no').mean()
        avg_no_price = letter_markets['no_price'].mean()
        be = avg_no_price / 100
        edge = no_wr - be

        print(f"  {letter}: N={n}, NO WR={no_wr:.1%}, Avg NO Price={avg_no_price:.1f}c, Edge={edge*100:.2f}%")

    return {'tested': True}


def test_wild005_price_never_moved(df, all_markets):
    """
    WILD-005: Price Never Moved

    Markets where all trades happened at the SAME price.
    """
    print("\n" + "=" * 60)
    print("WILD-005: PRICE NEVER MOVED")
    print("=" * 60)

    price_variance = df.groupby('market_ticker').agg({
        'yes_price': 'std',
        'market_result': 'first',
        'no_price': 'mean',
        'taker_side': lambda x: (x == 'no').mean(),
        'count': 'size'
    }).reset_index()
    price_variance.columns = ['market_ticker', 'price_std', 'market_result', 'no_price', 'no_ratio', 'n_trades']
    price_variance['price_std'] = price_variance['price_std'].fillna(0)

    # Zero price movement (all trades at same price)
    no_move = price_variance[(price_variance['price_std'] == 0) & (price_variance['n_trades'] >= 3)].copy()

    no_move_no = no_move[no_move['no_ratio'] > 0.5]

    result = quick_edge_check(no_move_no, side='no', name="No Price Move + NO Majority")

    print(f"Markets with no price move: {len(no_move)}")
    print(f"No Move + NO: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return result


def test_wild006_100_percent_consensus(df, all_markets):
    """
    WILD-006: 100% Consensus

    Markets where ALL trades are the same direction.
    """
    print("\n" + "=" * 60)
    print("WILD-006: 100% CONSENSUS")
    print("=" * 60)

    market_consensus = df.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size'
    }).reset_index()
    market_consensus.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price', 'yes_price', 'n_trades']

    # 100% NO consensus
    all_no = market_consensus[(market_consensus['no_ratio'] == 1.0) & (market_consensus['n_trades'] >= 3)].copy()

    # 100% YES consensus
    all_yes = market_consensus[(market_consensus['no_ratio'] == 0.0) & (market_consensus['n_trades'] >= 3)].copy()

    result_no = quick_edge_check(all_no, side='no', name="100% NO Consensus")
    result_yes = quick_edge_check(all_yes, side='yes', name="100% YES Consensus")

    print(f"100% NO consensus: N={result_no['n']}, Edge={result_no.get('edge', 'N/A')}")
    print(f"100% YES consensus: N={result_yes['n']}, Edge={result_yes.get('edge', 'N/A')}")

    return {'all_no': result_no, 'all_yes': result_yes}


def test_wild007_second_vs_last_trade(df, all_markets):
    """
    WILD-007: Second Trade vs Last Trade

    Does the SECOND trade predict better than the LAST trade?
    """
    print("\n" + "=" * 60)
    print("WILD-007: SECOND vs LAST TRADE")
    print("=" * 60)

    second_trade_info = []
    last_trade_info = []

    for market_ticker, market_df in df.groupby('market_ticker'):
        if len(market_df) < 3:
            continue

        market_df = market_df.sort_values('datetime').reset_index(drop=True)

        second_trade_info.append({
            'market_ticker': market_ticker,
            'second_trade_side': market_df.iloc[1]['taker_side'],
            'market_result': market_df['market_result'].iloc[0],
            'no_price': market_df['no_price'].mean(),
            'yes_price': market_df['yes_price'].mean()
        })

        last_trade_info.append({
            'market_ticker': market_ticker,
            'last_trade_side': market_df.iloc[-1]['taker_side'],
            'market_result': market_df['market_result'].iloc[0],
            'no_price': market_df['no_price'].mean(),
            'yes_price': market_df['yes_price'].mean()
        })

    second_df = pd.DataFrame(second_trade_info)
    last_df = pd.DataFrame(last_trade_info)

    # Follow second trade
    second_no = second_df[second_df['second_trade_side'] == 'no']
    second_yes = second_df[second_df['second_trade_side'] == 'yes']

    # Follow last trade
    last_no = last_df[last_df['last_trade_side'] == 'no']
    last_yes = last_df[last_df['last_trade_side'] == 'yes']

    result_second_no = quick_edge_check(second_no, side='no', name="Follow 2nd Trade (NO)")
    result_last_no = quick_edge_check(last_no, side='no', name="Follow Last Trade (NO)")

    print(f"Second trade NO: N={result_second_no['n']}, Edge={result_second_no.get('edge', 'N/A')}")
    print(f"Last trade NO: N={result_last_no['n']}, Edge={result_last_no.get('edge', 'N/A')}")

    return {'second_no': result_second_no, 'last_no': result_last_no}


def test_wild008_day_of_year(df, all_markets):
    """
    WILD-008: Day of Year Effects

    Is day 100 different from day 200?
    """
    print("\n" + "=" * 60)
    print("WILD-008: DAY OF YEAR")
    print("=" * 60)

    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Bucket by 50-day periods
    df['doy_bucket'] = (df['day_of_year'] // 50) * 50

    for bucket in sorted(df['doy_bucket'].unique()):
        bucket_trades = df[df['doy_bucket'] == bucket]

        bucket_markets = bucket_trades.groupby('market_ticker').agg({
            'taker_side': lambda x: (x == 'no').mean(),
            'market_result': 'first',
            'no_price': 'mean'
        }).reset_index()
        bucket_markets.columns = ['market_ticker', 'no_ratio', 'market_result', 'no_price']

        no_majority = bucket_markets[bucket_markets['no_ratio'] > 0.5]

        n = len(no_majority)
        if n < 30:
            continue

        wins = (no_majority['market_result'] == 'no').sum()
        wr = wins / n
        avg_price = no_majority['no_price'].mean()
        be = avg_price / 100
        edge = wr - be

        print(f"  Days {bucket}-{bucket+50}: N={n}, Edge={edge*100:.2f}%")

    return {'tested': True}


def test_wild009_yes_no_price_ratio(df, all_markets):
    """
    WILD-009: YES/NO Price Ratio

    What about the RATIO of YES to NO price, not just levels?
    """
    print("\n" + "=" * 60)
    print("WILD-009: YES/NO PRICE RATIO")
    print("=" * 60)

    all_markets['yes_no_ratio'] = all_markets['yes_price'] / (all_markets['no_price'] + 0.1)

    # Extreme ratios
    extreme_yes_heavy = all_markets[all_markets['yes_no_ratio'] > 5].copy()  # YES >> NO
    extreme_no_heavy = all_markets[all_markets['yes_no_ratio'] < 0.2].copy()  # NO >> YES

    # In extreme YES-heavy, bet NO (contrarian)
    result_extreme_yes = quick_edge_check(extreme_yes_heavy, side='no', name="Extreme YES-Heavy -> Bet NO")

    # In extreme NO-heavy, bet YES (contrarian)
    result_extreme_no = quick_edge_check(extreme_no_heavy, side='yes', name="Extreme NO-Heavy -> Bet YES")

    print(f"Extreme YES-heavy (fade): N={result_extreme_yes['n']}, Edge={result_extreme_yes.get('edge', 'N/A')}")
    print(f"Extreme NO-heavy (fade): N={result_extreme_no['n']}, Edge={result_extreme_no.get('edge', 'N/A')}")

    return {'fade_yes_heavy': result_extreme_yes, 'fade_no_heavy': result_extreme_no}


def test_wild010_triple_stack_weird(df, all_markets):
    """
    WILD-010: Triple Stack of Weird Signals

    Combine Fibonacci + Weekend + Whale into one super-weird signal.
    """
    print("\n" + "=" * 60)
    print("WILD-010: TRIPLE WEIRD STACK (Fib + Weekend + Whale)")
    print("=" * 60)

    # Build market features
    market_features = df.groupby('market_ticker').agg({
        'is_whale': 'any',
        'is_weekend': 'any',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size'
    }).reset_index()
    market_features.columns = [
        'market_ticker', 'has_whale', 'is_weekend', 'no_ratio', 'market_result',
        'no_price', 'yes_price', 'n_trades'
    ]

    # Add Fibonacci flag
    market_features['is_fibonacci'] = market_features['n_trades'].isin(FIBONACCI)

    # Triple stack
    triple_weird = market_features[
        (market_features['is_fibonacci'] == True) &
        (market_features['is_weekend'] == True) &
        (market_features['has_whale'] == True) &
        (market_features['no_ratio'] > 0.5)
    ].copy()

    result = quick_edge_check(triple_weird, side='no', name="Triple Weird Stack")

    print(f"Triple Weird Stack: N={result['n']}, Edge={result.get('edge', 'N/A')}")

    return result


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    print("=" * 80)
    print("LSD SESSION 001: LATERAL STRATEGY DISCOVERY")
    print("Maximum Dose Mode - Everything Gets Tested")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df = load_data()

    # Build baselines
    all_markets, baseline_no, baseline_yes = build_baseline_10c(df)

    results = {
        'session': 'LSD_001',
        'timestamp': datetime.now().isoformat(),
        'data_size': len(df),
        'n_markets': df['market_ticker'].nunique(),
        'hypotheses_tested': {},
        'flagged_for_deep_analysis': []
    }

    # =========================================================================
    # TEST EXT HYPOTHESES
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: EXTERNAL RESEARCH HYPOTHESES (EXT-001 to EXT-006)")
    print("=" * 80)

    results['hypotheses_tested']['EXT-001'] = test_ext001_early_trade_premium(df)
    results['hypotheses_tested']['EXT-002'] = test_ext002_steam_cascade(df)
    results['hypotheses_tested']['EXT-003'] = test_ext003_reverse_line_movement(df)
    results['hypotheses_tested']['EXT-004'] = test_ext004_vpin_flow_toxicity(df)
    results['hypotheses_tested']['EXT-005'] = test_ext005_buyback_reversal(df)
    results['hypotheses_tested']['EXT-006'] = test_ext006_surprise_mispricing(df)

    # =========================================================================
    # TEST LSD ABSURD HYPOTHESES
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: LSD ABSURD HYPOTHESES (LSD-001 to LSD-005)")
    print("=" * 80)

    results['hypotheses_tested']['LSD-001'] = test_lsd001_fibonacci_trade_count(df, all_markets)
    results['hypotheses_tested']['LSD-002'] = test_lsd002_prime_trade_count(df, all_markets)
    results['hypotheses_tested']['LSD-003'] = test_lsd003_worst_strategy_inverse(df, all_markets)
    results['hypotheses_tested']['LSD-004'] = test_lsd004_mega_stack(df, all_markets, baseline_no)
    results['hypotheses_tested']['LSD-005'] = test_lsd005_drunk_opposite(df, all_markets)

    # =========================================================================
    # TEST WILD ORIGINAL HYPOTHESES
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: WILD ORIGINAL HYPOTHESES (WILD-001 to WILD-010)")
    print("=" * 80)

    results['hypotheses_tested']['WILD-001'] = test_wild001_digit_count(df, all_markets)
    results['hypotheses_tested']['WILD-002'] = test_wild002_exactly_7_trades(df, all_markets)
    results['hypotheses_tested']['WILD-003'] = test_wild003_palindrome_sizes(df, all_markets)
    results['hypotheses_tested']['WILD-004'] = test_wild004_ticker_first_letter(df, all_markets)
    results['hypotheses_tested']['WILD-005'] = test_wild005_price_never_moved(df, all_markets)
    results['hypotheses_tested']['WILD-006'] = test_wild006_100_percent_consensus(df, all_markets)
    results['hypotheses_tested']['WILD-007'] = test_wild007_second_vs_last_trade(df, all_markets)
    results['hypotheses_tested']['WILD-008'] = test_wild008_day_of_year(df, all_markets)
    results['hypotheses_tested']['WILD-009'] = test_wild009_yes_no_price_ratio(df, all_markets)
    results['hypotheses_tested']['WILD-010'] = test_wild010_triple_stack_weird(df, all_markets)

    # =========================================================================
    # COLLECT FLAGGED HYPOTHESES
    # =========================================================================
    print("\n" + "=" * 80)
    print("FLAGGED FOR DEEPER ANALYSIS (Edge > 5% AND p < 0.05)")
    print("=" * 80)

    def check_flag(result, name):
        if isinstance(result, dict):
            if result.get('flag'):
                return [(name, result)]
            else:
                flagged = []
                for k, v in result.items():
                    if isinstance(v, dict) and v.get('flag'):
                        flagged.append((f"{name}/{k}", v))
                return flagged
        return []

    flagged = []
    for hyp_id, hyp_result in results['hypotheses_tested'].items():
        flagged.extend(check_flag(hyp_result, hyp_id))

    if flagged:
        for name, result in flagged:
            print(f"  [FLAG] {name}: N={result.get('n')}, Edge={result.get('edge')*100:.2f}%")
            results['flagged_for_deep_analysis'].append({
                'hypothesis': name,
                'n': result.get('n'),
                'edge': result.get('edge'),
                'p_value': result.get('p_value')
            })
    else:
        print("  No hypotheses flagged for deeper analysis")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_tested = 0
    total_flagged = len(flagged)

    # Count all unique tests
    for hyp_id, hyp_result in results['hypotheses_tested'].items():
        if isinstance(hyp_result, dict):
            if 'n' in hyp_result:
                total_tested += 1
            else:
                for k, v in hyp_result.items():
                    if isinstance(v, dict) and 'n' in v:
                        total_tested += 1

    print(f"\nHypotheses tested: ~{total_tested}")
    print(f"Flagged for deep analysis: {total_flagged}")

    if total_flagged > 0:
        print("\nFlagged hypotheses need:")
        print("  1. Bucket-by-bucket baseline comparison")
        print("  2. Temporal stability check")
        print("  3. Concentration check")
        print("  4. Bootstrap CI")

    # Save results
    output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/lsd_session_001_results.json'

    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    results_serializable = make_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"\nSession completed: {datetime.now()}")

    return results


if __name__ == "__main__":
    results = main()
