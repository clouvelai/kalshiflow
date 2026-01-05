"""
Session 012: Test Novel Signal Combinations
Focus on combining existing validated signals in new ways
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
REPORT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/'

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def validate_signal(signal_markets, all_markets, name, n_tests=20):
    """Validate a signal with proper price proxy check"""
    if len(signal_markets) < 100:
        return {'status': 'rejected', 'reason': 'insufficient', 'n': len(signal_markets)}

    # Calculate NO price for signal markets
    # Use avg_no_price if available, otherwise calculate
    if 'avg_no_price' not in signal_markets.columns:
        signal_markets = signal_markets.copy()
        signal_markets['avg_no_price'] = signal_markets['avg_trade_price'].apply(
            lambda x: 100 - x if x > 50 else x
        )

    n = len(signal_markets)
    no_wins = (signal_markets['market_result'] == 'no').sum()
    wr = no_wins / n
    be = signal_markets['avg_no_price'].mean() / 100
    edge = wr - be

    # P-value
    z = (no_wins - n * be) / np.sqrt(n * be * (1 - be)) if be > 0 and be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)
    bonferroni = 0.01 / n_tests

    if p_value >= bonferroni:
        return {'status': 'rejected', 'reason': 'not_significant', 'edge': edge, 'p': p_value}

    # Price proxy check
    signal_markets = signal_markets.copy()
    signal_markets['no_bucket'] = (signal_markets['avg_no_price'] // 10) * 10

    all_markets = all_markets.copy()
    all_markets['no_bucket'] = (all_markets['avg_no_price'] // 10) * 10

    improvements = []
    for bucket in signal_markets['no_bucket'].unique():
        sig = signal_markets[signal_markets['no_bucket'] == bucket]
        base = all_markets[all_markets['no_bucket'] == bucket]
        if len(sig) >= 10 and len(base) >= 10:
            sig_wr = (sig['market_result'] == 'no').mean()
            base_wr = (base['market_result'] == 'no').mean()
            improvements.append((sig_wr - base_wr, len(sig)))

    if not improvements:
        return {'status': 'rejected', 'reason': 'no_buckets'}

    weighted_imp = sum(imp * n for imp, n in improvements) / sum(n for _, n in improvements)

    if weighted_imp <= 0.01:
        return {'status': 'rejected', 'reason': 'price_proxy', 'edge': edge, 'imp': weighted_imp}

    # Concentration
    signal_markets_copy = signal_markets.copy()
    signal_markets_copy['profit'] = np.where(
        signal_markets_copy['market_result'] == 'no',
        100 - signal_markets_copy['avg_no_price'],
        -signal_markets_copy['avg_no_price']
    )
    pos_profit = signal_markets_copy[signal_markets_copy['profit'] > 0]['profit'].sum()
    max_conc = 0
    if pos_profit > 0:
        signal_markets_copy['conc'] = np.where(
            signal_markets_copy['profit'] > 0,
            signal_markets_copy['profit'] / pos_profit,
            0
        )
        max_conc = signal_markets_copy['conc'].max()

    if max_conc > 0.30:
        return {'status': 'rejected', 'reason': 'concentration', 'conc': max_conc}

    return {
        'status': 'validated',
        'n_markets': n,
        'edge': float(edge),
        'improvement': float(weighted_imp),
        'p_value': float(p_value),
        'concentration': float(max_conc)
    }


def test_high_leverage_morning(df):
    """High leverage trades during morning hours (9-12 ET)"""
    print("\nTesting: High Leverage Morning (9-12 ET)")

    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['hour'].isin([9, 10, 11, 12]))
    ]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "High Leverage Morning")
    print(f"  Result: {result}")
    return result


def test_high_leverage_evening(df):
    """High leverage trades during evening hours (6-10 PM ET)"""
    print("\nTesting: High Leverage Evening (6-10 PM ET)")

    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['hour'].isin([18, 19, 20, 21, 22]))
    ]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "High Leverage Evening")
    print(f"  Result: {result}")
    return result


def test_small_trades_high_leverage(df):
    """Small trades (< 10 contracts) with high leverage"""
    print("\nTesting: Small Trades + High Leverage")

    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['count'] < 10)
    ]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "Small Trades High Leverage")
    print(f"  Result: {result}")
    return result


def test_multiple_high_lev_trades(df):
    """Markets with 3+ high leverage trades (indicates strong retail interest)"""
    print("\nTesting: Multiple High Leverage Trades (3+)")

    high_lev_counts = df[df['leverage_ratio'] > 2].groupby('market_ticker').size()
    multi_high_lev = high_lev_counts[high_lev_counts >= 3].index

    signal_trades = df[df['market_ticker'].isin(multi_high_lev)]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "Multiple High Lev Trades")
    print(f"  Result: {result}")
    return result


def test_leverage_increasing(df):
    """Markets where leverage increases over time (desperation)"""
    print("\nTesting: Increasing Leverage Over Time")

    df_sorted = df.sort_values(['market_ticker', 'timestamp'])

    def check_increasing_lev(group):
        if len(group) < 5:
            return False
        levs = group['leverage_ratio'].values
        first_half = levs[:len(levs)//2].mean()
        second_half = levs[len(levs)//2:].mean()
        return second_half > first_half * 1.5  # 50% increase

    increasing = []
    for ticker, group in df_sorted.groupby('market_ticker'):
        if check_increasing_lev(group):
            increasing.append(ticker)

    signal_trades = df[df['market_ticker'].isin(increasing)]

    if len(increasing) < 100:
        print(f"  Only {len(increasing)} markets with increasing leverage")
        return {'status': 'rejected', 'reason': 'insufficient', 'n': len(increasing)}

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "Increasing Leverage")
    print(f"  Result: {result}")
    return result


def test_high_lev_yes_only(df):
    """High leverage on YES trades specifically (not NO)"""
    print("\nTesting: High Leverage YES Only (Fade)")

    signal_trades = df[
        (df['leverage_ratio'] > 2.5) &
        (df['taker_side'] == 'yes')
    ]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean'  # This is YES price
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price']

    # For YES trades, NO price = 100 - YES price
    signal_markets['avg_no_price'] = 100 - signal_markets['avg_trade_price']

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

    result = validate_signal(signal_markets, all_markets, "High Lev YES Only")
    print(f"  Result: {result}")
    return result


def test_very_high_leverage(df):
    """Very high leverage (>5x) trades"""
    print("\nTesting: Very High Leverage (>5x)")

    signal_trades = df[df['leverage_ratio'] > 5]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "Very High Leverage")
    print(f"  Result: {result}")
    return result


def test_sports_high_leverage(df):
    """High leverage in sports markets specifically"""
    print("\nTesting: Sports High Leverage")

    sports = ['KXNFL', 'KXNCAAF', 'KXNBA', 'KXNHL', 'KXMLB', 'KXNCAAMB', 'KXSOC']

    df['is_sports'] = df['market_ticker'].str.contains('|'.join(sports), regex=True)

    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['is_sports'] == True)
    ]

    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    signal_markets['avg_no_price'] = np.where(
        signal_markets['yes_ratio'] > 0.5,
        100 - signal_markets['avg_trade_price'],
        signal_markets['avg_trade_price']
    )

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

    result = validate_signal(signal_markets, all_markets, "Sports High Leverage")
    print(f"  Result: {result}")
    return result


def main():
    print("="*80)
    print("SESSION 012: NOVEL SIGNAL COMBINATIONS")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    results = {}

    # Test all combinations
    results['high_lev_morning'] = test_high_leverage_morning(df)
    results['high_lev_evening'] = test_high_leverage_evening(df)
    results['small_trades_high_lev'] = test_small_trades_high_leverage(df)
    results['multiple_high_lev'] = test_multiple_high_lev_trades(df)
    results['increasing_lev'] = test_leverage_increasing(df)
    results['high_lev_yes'] = test_high_lev_yes_only(df)
    results['very_high_lev'] = test_very_high_leverage(df)
    results['sports_high_lev'] = test_sports_high_leverage(df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    validated = [k for k, v in results.items() if v.get('status') == 'validated']
    rejected = [k for k, v in results.items() if v.get('status') == 'rejected']

    print(f"\nValidated: {len(validated)}")
    for k in validated:
        v = results[k]
        print(f"  {k}: Edge={v['edge']*100:.2f}%, Imp={v['improvement']*100:.2f}%, N={v['n_markets']}")

    print(f"\nRejected: {len(rejected)}")
    for k in rejected:
        v = results[k]
        reason = v.get('reason', 'unknown')
        print(f"  {k}: {reason}")

    # Save results
    output_path = f'{REPORT_PATH}session012_novel_combinations_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
