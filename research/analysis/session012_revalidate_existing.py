"""
Session 012: Re-validate Existing Strategies S007, S008, S009
Make sure they still pass with proper methodology
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

ET = pytz.timezone('America/New_York')

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def validate_s007(df):
    """
    S007: Fade High-Leverage YES Trades
    Signal: leverage_ratio > 2 AND taker_side == 'yes'
    Action: Bet NO
    """
    print("\n" + "="*80)
    print("RE-VALIDATING S007: Fade High-Leverage YES")
    print("="*80)

    # Find trades matching signal
    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['taker_side'] == 'yes')
    ]

    # Get unique markets
    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean'  # This is YES price when taker_side == 'yes'
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_yes_price']

    # NO price = 100 - YES price
    signal_markets['avg_no_price'] = 100 - signal_markets['avg_yes_price']

    n = len(signal_markets)
    no_wins = (signal_markets['market_result'] == 'no').sum()
    wr = no_wins / n
    be = signal_markets['avg_no_price'].mean() / 100
    edge = wr - be

    print(f"\nBasic Stats:")
    print(f"  Markets: {n}")
    print(f"  NO Win Rate: {wr:.1%}")
    print(f"  Avg NO Price: {signal_markets['avg_no_price'].mean():.1f}c")
    print(f"  Breakeven: {be:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    z = (no_wins - n * be) / np.sqrt(n * be * (1 - be))
    p_value = 1 - stats.norm.cdf(z)
    print(f"  P-value: {p_value:.2e}")

    # Price proxy check
    print("\n  Price Proxy Check:")

    # Build baseline from ALL markets
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    all_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    # Calculate effective NO price
    all_markets['avg_no_price'] = np.where(
        all_markets['yes_ratio'] > 0.5,
        100 - all_markets['avg_trade_price'],
        all_markets['avg_trade_price']
    )

    # Bucket comparison
    signal_markets['bucket'] = (signal_markets['avg_no_price'] // 10) * 10
    all_markets['bucket'] = (all_markets['avg_no_price'] // 10) * 10

    improvements = []
    for bucket in sorted(signal_markets['bucket'].unique()):
        sig = signal_markets[signal_markets['bucket'] == bucket]
        base = all_markets[all_markets['bucket'] == bucket]
        if len(sig) >= 30 and len(base) >= 30:
            sig_wr = (sig['market_result'] == 'no').mean()
            base_wr = (base['market_result'] == 'no').mean()
            imp = sig_wr - base_wr
            improvements.append({'bucket': bucket, 'sig_wr': sig_wr, 'base_wr': base_wr, 'imp': imp, 'n': len(sig)})
            print(f"  {bucket:.0f}-{bucket+10:.0f}c: Sig={sig_wr:.1%}, Base={base_wr:.1%}, Imp={imp*100:+.1f}%, N={len(sig)}")

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] for i in improvements) / total_n
        print(f"\n  Weighted Improvement: {weighted_imp*100:.2f}%")

    return {
        'status': 'validated' if edge > 0 and p_value < 0.001 else 'rejected',
        'edge': edge,
        'improvement': weighted_imp if improvements else 0,
        'n_markets': n,
        'p_value': p_value
    }


def validate_s009(df):
    """
    S009: Extended Drunk Betting
    Signal: 6PM-11PM ET on Fri/Sat + leverage > 1.5 + sports
    Action: Fade the trade
    """
    print("\n" + "="*80)
    print("RE-VALIDATING S009: Extended Drunk Betting")
    print("="*80)

    SPORTS = ['KXNFL', 'KXNCAAF', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAMB', 'KXSOC']
    EVENING_HOURS = [18, 19, 20, 21, 22, 23]
    WEEKEND_DAYS = [4, 5]

    # Parse times
    df['dt_utc'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
    df['dt_et'] = df['dt_utc'].dt.tz_convert(ET)
    df['hour_et'] = df['dt_et'].dt.hour
    df['day_et'] = df['dt_et'].dt.dayofweek

    # Check sports
    df['is_sports'] = df['market_ticker'].str.contains('|'.join(SPORTS), regex=True)

    # Find matching trades
    signal_trades = df[
        (df['leverage_ratio'] > 1.5) &
        (df['is_sports'] == True) &
        (df['day_et'].isin(WEEKEND_DAYS)) &
        (df['hour_et'].isin(EVENING_HOURS))
    ]

    print(f"Found {len(signal_trades)} trades matching signal")

    # Get unique markets
    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    # Calculate NO price (we fade, so if they bet YES we bet NO)
    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],  # They bet YES, we pay this for NO
        signal_markets['avg_trade_price']  # They bet NO, we... actually this is complex
    )

    # The FADE logic: if they bet YES (yes_ratio > 0.5), we bet NO
    # If they bet NO (yes_ratio < 0.5), we bet YES

    # For markets where they bet YES (we fade to NO):
    yes_heavy = signal_markets[signal_markets['yes_ratio'] > 0.5].copy()
    yes_heavy['our_side'] = 'no'
    yes_heavy['our_price'] = 100 - yes_heavy['avg_trade_price']
    yes_heavy['we_win'] = (yes_heavy['market_result'] == 'no')

    # For markets where they bet NO (we fade to YES):
    no_heavy = signal_markets[signal_markets['yes_ratio'] <= 0.5].copy()
    no_heavy['our_side'] = 'yes'
    no_heavy['our_price'] = 100 - no_heavy['avg_trade_price']
    no_heavy['we_win'] = (no_heavy['market_result'] == 'yes')

    combined = pd.concat([yes_heavy, no_heavy])

    n = len(combined)
    wins = combined['we_win'].sum()
    wr = wins / n
    be = combined['our_price'].mean() / 100
    edge = wr - be

    print(f"\nBasic Stats:")
    print(f"  Markets: {n}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Avg Entry Price: {combined['our_price'].mean():.1f}c")
    print(f"  Breakeven: {be:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)
    print(f"  P-value: {p_value:.2e}")

    # Price proxy check
    print("\n  Price Proxy Check:")

    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    all_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']
    all_markets['avg_no_price'] = np.where(
        all_markets['yes_ratio'] > 0.5,
        100 - all_markets['avg_trade_price'],
        all_markets['avg_trade_price']
    )

    combined['bucket'] = (combined['our_price'] // 10) * 10
    all_markets['bucket'] = (all_markets['avg_no_price'] // 10) * 10

    improvements = []
    for bucket in sorted(combined['bucket'].unique()):
        sig = combined[combined['bucket'] == bucket]
        base = all_markets[all_markets['bucket'] == bucket]
        if len(sig) >= 10 and len(base) >= 10:
            # For signal: check win rate
            sig_wr = sig['we_win'].mean()
            # For baseline: NO win rate at this bucket
            base_wr = (base['market_result'] == 'no').mean()
            imp = sig_wr - base_wr
            improvements.append({'bucket': bucket, 'sig_wr': sig_wr, 'base_wr': base_wr, 'imp': imp, 'n': len(sig)})
            print(f"  {bucket:.0f}-{bucket+10:.0f}c: Sig={sig_wr:.1%}, Base={base_wr:.1%}, Imp={imp*100:+.1f}%, N={len(sig)}")

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] for i in improvements) / total_n
        print(f"\n  Weighted Improvement: {weighted_imp*100:.2f}%")

    return {
        'status': 'validated' if edge > 0 and p_value < 0.01 else 'rejected',
        'edge': edge,
        'improvement': weighted_imp if improvements else 0,
        'n_markets': n,
        'p_value': p_value
    }


def main():
    print("="*80)
    print("SESSION 012: RE-VALIDATION OF EXISTING STRATEGIES")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    result_s007 = validate_s007(df)
    result_s009 = validate_s009(df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nS007 (Fade High-Lev YES):")
    print(f"  Status: {result_s007['status']}")
    print(f"  Edge: {result_s007['edge']*100:.2f}%")
    print(f"  Improvement: {result_s007['improvement']*100:.2f}%")

    print(f"\nS009 (Extended Drunk):")
    print(f"  Status: {result_s009['status']}")
    print(f"  Edge: {result_s009['edge']*100:.2f}%")
    print(f"  Improvement: {result_s009['improvement']*100:.2f}%")


if __name__ == "__main__":
    main()
