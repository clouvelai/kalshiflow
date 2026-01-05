"""
Session 012: Verify H056 Different Interpretations
The initial test showed +10% edge but deep validation showed -4% edge
Need to understand which interpretation is correct
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def test_extreme_yes_trades(df):
    """
    Original H056 from initial test: Markets with YES trades at >85c
    (i.e., expensive YES = cheap NO at <15c)
    Fade these -> bet NO
    """
    print("\n" + "="*80)
    print("H056 Interpretation 1: Markets with YES trades at >85c (cheap NO)")
    print("Signal: Someone bought YES at >85c (so NO is <15c)")
    print("Action: Bet NO (fade the YES buyer)")
    print("="*80)

    # Find markets with YES trades at high YES prices
    extreme_yes = df[
        (df['taker_side'] == 'yes') &
        (df['trade_price'] > 85)  # YES price > 85c means NO < 15c
    ]

    extreme_yes_markets = extreme_yes.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',  # This is YES price
        'count': 'sum'
    }).reset_index()
    extreme_yes_markets.columns = ['market_ticker', 'market_result', 'avg_yes_price', 'contract_count']

    # NO price = 100 - YES price
    extreme_yes_markets['avg_no_price'] = 100 - extreme_yes_markets['avg_yes_price']

    print(f"\nFound {len(extreme_yes_markets)} markets with YES trades at >85c (NO at <15c)")

    n_markets = len(extreme_yes_markets)
    no_wins = (extreme_yes_markets['market_result'] == 'no').sum()
    no_win_rate = no_wins / n_markets

    avg_no_price = extreme_yes_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100

    edge = no_win_rate - breakeven

    print(f"\nStats:")
    print(f"  Markets: {n_markets}")
    print(f"  Avg YES price: {extreme_yes_markets['avg_yes_price'].mean():.1f}c")
    print(f"  Avg NO price: {avg_no_price:.1f}c")
    print(f"  NO Win Rate: {no_win_rate:.1%}")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    z = (no_wins - n_markets * breakeven) / np.sqrt(n_markets * breakeven * (1 - breakeven))
    p_value = 1 - stats.norm.cdf(z)
    print(f"  P-value: {p_value:.2e}")

    return {
        'interpretation': 'Fade YES trades at >85c',
        'n_markets': n_markets,
        'no_win_rate': no_win_rate,
        'avg_no_price': avg_no_price,
        'breakeven': breakeven,
        'edge': edge,
        'p_value': p_value
    }


def test_extreme_no_trades(df):
    """
    Alternative interpretation: Markets with NO trades at >85c (expensive NO)
    Follow these -> bet NO
    """
    print("\n" + "="*80)
    print("H056 Interpretation 2: Markets with NO trades at >85c (expensive NO)")
    print("Signal: Someone bought NO at >85c (expensive NO)")
    print("Action: Bet NO (follow the expensive NO buyer)")
    print("="*80)

    # Find markets with NO trades at high NO prices
    extreme_no = df[
        (df['taker_side'] == 'no') &
        (df['trade_price'] > 85)  # NO price > 85c
    ]

    extreme_no_markets = extreme_no.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',  # This is NO price
        'count': 'sum'
    }).reset_index()
    extreme_no_markets.columns = ['market_ticker', 'market_result', 'avg_no_price', 'contract_count']

    print(f"\nFound {len(extreme_no_markets)} markets with NO trades at >85c")

    n_markets = len(extreme_no_markets)
    no_wins = (extreme_no_markets['market_result'] == 'no').sum()
    no_win_rate = no_wins / n_markets

    avg_no_price = extreme_no_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100

    edge = no_win_rate - breakeven

    print(f"\nStats:")
    print(f"  Markets: {n_markets}")
    print(f"  Avg NO price: {avg_no_price:.1f}c")
    print(f"  NO Win Rate: {no_win_rate:.1%}")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    z = (no_wins - n_markets * breakeven) / np.sqrt(n_markets * breakeven * (1 - breakeven))
    p_value = 1 - stats.norm.cdf(z)
    print(f"  P-value: {p_value:.2e}")

    return {
        'interpretation': 'Follow NO trades at >85c',
        'n_markets': n_markets,
        'no_win_rate': no_win_rate,
        'avg_no_price': avg_no_price,
        'breakeven': breakeven,
        'edge': edge,
        'p_value': p_value
    }


def test_fade_yes_with_price_check(df):
    """
    Interpretation 1 (Fade YES) with proper price proxy check
    """
    print("\n" + "="*80)
    print("PROPER VALIDATION: Fade YES at >85c with Price Proxy Check")
    print("="*80)

    # Find markets with YES trades at high YES prices
    extreme_yes = df[
        (df['taker_side'] == 'yes') &
        (df['trade_price'] > 85)
    ]

    signal_markets = extreme_yes.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',  # YES price
        'count': 'sum'
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'avg_yes_price', 'contract_count']
    signal_markets['avg_no_price'] = 100 - signal_markets['avg_yes_price']

    # Price proxy check: compare to all markets at same NO price
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()
    }).reset_index()
    all_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    # Calculate effective NO price
    all_markets['effective_no_price'] = np.where(
        all_markets['yes_ratio'] > 0.5,
        100 - all_markets['avg_trade_price'],
        all_markets['avg_trade_price']
    )

    # Bucket comparison
    signal_markets['no_price_bucket'] = (signal_markets['avg_no_price'] // 5) * 5
    all_markets['no_price_bucket'] = (all_markets['effective_no_price'] // 5) * 5

    signal_by_bucket = signal_markets.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    signal_by_bucket.columns = ['bucket', 'signal_wr', 'signal_n']

    baseline_by_bucket = all_markets.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline_by_bucket.columns = ['bucket', 'baseline_wr', 'baseline_n']

    comparison = signal_by_bucket.merge(baseline_by_bucket, on='bucket', how='left')
    comparison['improvement'] = comparison['signal_wr'] - comparison['baseline_wr']
    comparison['weight'] = comparison['signal_n'] / comparison['signal_n'].sum()

    print(f"\n{'Bucket':<10} {'Signal WR':<12} {'Base WR':<12} {'Improve':<10} {'N':<8}")
    for _, row in comparison.iterrows():
        if pd.notna(row['baseline_wr']):
            print(f"{row['bucket']:.0f}-{row['bucket']+5:.0f}c   "
                  f"{row['signal_wr']:.1%}        "
                  f"{row['baseline_wr']:.1%}        "
                  f"{row['improvement']*100:+.1f}%      "
                  f"{row['signal_n']:.0f}")

    weighted_improvement = (comparison['improvement'] * comparison['weight']).sum()
    print(f"\nWeighted Improvement: {weighted_improvement*100:.2f}%")

    return {
        'interpretation': 'Fade YES at >85c',
        'n_markets': len(signal_markets),
        'weighted_improvement': weighted_improvement
    }


def main():
    print("="*80)
    print("SESSION 012: H056 INTERPRETATION VERIFICATION")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Test both interpretations
    result1 = test_extreme_yes_trades(df)
    result2 = test_extreme_no_trades(df)

    # Price proxy check for interpretation 1
    result1_validated = test_fade_yes_with_price_check(df)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nInterpretation 1 (Fade YES at >85c):")
    print(f"  Raw Edge: {result1['edge']*100:.2f}%")
    print(f"  After Price Check: {result1_validated['weighted_improvement']*100:.2f}% improvement")

    print("\nInterpretation 2 (Follow NO at >85c):")
    print(f"  Raw Edge: {result2['edge']*100:.2f}%")

    print("\nConclusion:")
    if result1_validated['weighted_improvement'] > 0.01:
        print("  Interpretation 1 (Fade YES at >85c) PASSES price proxy check")
    else:
        print("  Interpretation 1 (Fade YES at >85c) FAILS price proxy check")

    if result2['edge'] > 0:
        print(f"  Interpretation 2 (Follow NO at >85c) has positive raw edge but needs full validation")
    else:
        print(f"  Interpretation 2 (Follow NO at >85c) has NEGATIVE raw edge")


if __name__ == "__main__":
    main()
