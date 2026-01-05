"""
Session 012: Correct Methodology Validation
Use the correct columns from the data: yes_price, no_price
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


def validate_s007_correct(df):
    """
    S007: Fade High-Leverage YES Trades
    Use the correct columns: yes_price, no_price (NOT derived from trade_price)
    """
    print("\n" + "="*80)
    print("CORRECT VALIDATION: S007 (Fade High-Leverage YES)")
    print("="*80)

    # Find high-leverage YES trades
    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['taker_side'] == 'yes')
    ]

    # Get market-level aggregates using CORRECT columns
    signal_markets = signal_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',  # USE no_price directly!
        'yes_price': 'mean',
        'count': 'sum'
    }).reset_index()

    n = len(signal_markets)
    no_wins = (signal_markets['market_result'] == 'no').sum()
    wr = no_wins / n
    avg_no_price = signal_markets['no_price'].mean()
    be = avg_no_price / 100
    edge = wr - be

    print(f"\nBasic Stats (USING CORRECT no_price COLUMN):")
    print(f"  Markets: {n}")
    print(f"  NO Win Rate: {wr:.1%}")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")
    print(f"  Breakeven: {be:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    z = (no_wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)
    print(f"  P-value: {p_value:.2e}")

    # Price proxy check
    print("\n  Price Proxy Check:")

    # Build baseline from ALL markets using no_price
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    # Bucket comparison
    signal_markets['bucket'] = (signal_markets['no_price'] // 10) * 10
    all_markets['bucket'] = (all_markets['no_price'] // 10) * 10

    improvements = []
    print(f"\n  {'Bucket':<10} {'Signal WR':<12} {'Base WR':<12} {'Improve':<10} {'N':<8}")
    for bucket in sorted(signal_markets['bucket'].unique()):
        sig = signal_markets[signal_markets['bucket'] == bucket]
        base = all_markets[all_markets['bucket'] == bucket]
        if len(sig) >= 30 and len(base) >= 30:
            sig_wr = (sig['market_result'] == 'no').mean()
            base_wr = (base['market_result'] == 'no').mean()
            imp = sig_wr - base_wr
            improvements.append({'bucket': bucket, 'sig_wr': sig_wr, 'base_wr': base_wr, 'imp': imp, 'n': len(sig)})
            print(f"  {bucket:.0f}-{bucket+10:.0f}c   "
                  f"{sig_wr:.1%}        "
                  f"{base_wr:.1%}        "
                  f"{imp*100:+.1f}%      "
                  f"{len(sig)}")

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] for i in improvements) / total_n
        print(f"\n  Weighted Improvement: {weighted_imp*100:.2f}%")

        return {
            'status': 'validated' if edge > 0 and p_value < 0.001 and weighted_imp > 0.01 else 'rejected',
            'edge': float(edge),
            'improvement': float(weighted_imp),
            'n_markets': n,
            'p_value': float(p_value)
        }

    return {'status': 'rejected', 'reason': 'no_buckets'}


def compare_methodology(df):
    """
    Compare old (derived no_price) vs new (actual no_price column)
    """
    print("\n" + "="*80)
    print("METHODOLOGY COMPARISON")
    print("="*80)

    # Old method: Calculate NO price from trade_price and taker_side
    old_method = df.copy()
    old_method['derived_no_price'] = np.where(
        old_method['taker_side'] == 'yes',
        100 - old_method['trade_price'],  # YES trade: NO = 100 - YES price
        old_method['trade_price']  # NO trade: NO = trade_price
    )

    # Compare to actual no_price column
    old_method['price_diff'] = old_method['derived_no_price'] - old_method['no_price']

    print(f"\nPrice derivation comparison:")
    print(f"  Mean diff: {old_method['price_diff'].mean():.2f}c")
    print(f"  Std diff: {old_method['price_diff'].std():.2f}c")
    print(f"  Max diff: {old_method['price_diff'].max():.2f}c")
    print(f"  Min diff: {old_method['price_diff'].min():.2f}c")

    # Check correlation
    corr = old_method['derived_no_price'].corr(old_method['no_price'])
    print(f"  Correlation: {corr:.6f}")

    # Count exact matches
    exact_yes = (old_method[old_method['taker_side'] == 'yes']['price_diff'] == 0).sum()
    total_yes = len(old_method[old_method['taker_side'] == 'yes'])
    exact_no = (old_method[old_method['taker_side'] == 'no']['price_diff'] == 0).sum()
    total_no = len(old_method[old_method['taker_side'] == 'no'])

    print(f"\n  YES trades: {exact_yes}/{total_yes} ({exact_yes/total_yes*100:.1f}%) exact match")
    print(f"  NO trades: {exact_no}/{total_no} ({exact_no/total_no*100:.1f}%) exact match")


def analyze_what_s007_actually_does(df):
    """
    Deep dive: What EXACTLY does S007 select?
    """
    print("\n" + "="*80)
    print("DEEP DIVE: WHAT S007 ACTUALLY SELECTS")
    print("="*80)

    # S007 signal: High leverage YES trades
    signal_trades = df[
        (df['leverage_ratio'] > 2) &
        (df['taker_side'] == 'yes')
    ]

    print(f"\nS007 Signal Trades:")
    print(f"  Total trades: {len(signal_trades)}")
    print(f"  Unique markets: {signal_trades['market_ticker'].nunique()}")

    print(f"\n  YES Price (trade_price when taker_side=yes):")
    print(f"    Mean: {signal_trades['trade_price'].mean():.1f}c")
    print(f"    Median: {signal_trades['trade_price'].median():.1f}c")
    print(f"    Distribution: 10th={signal_trades['trade_price'].quantile(0.1):.0f}c, "
          f"90th={signal_trades['trade_price'].quantile(0.9):.0f}c")

    print(f"\n  NO Price (no_price column):")
    print(f"    Mean: {signal_trades['no_price'].mean():.1f}c")
    print(f"    Median: {signal_trades['no_price'].median():.1f}c")
    print(f"    Distribution: 10th={signal_trades['no_price'].quantile(0.1):.0f}c, "
          f"90th={signal_trades['no_price'].quantile(0.9):.0f}c")

    print(f"\n  Leverage Ratio:")
    print(f"    Mean: {signal_trades['leverage_ratio'].mean():.2f}")
    print(f"    Median: {signal_trades['leverage_ratio'].median():.2f}")

    # Key insight: NO price should be 100 - YES price
    signal_trades = signal_trades.copy()
    signal_trades['expected_no'] = 100 - signal_trades['trade_price']
    signal_trades['no_price_diff'] = signal_trades['no_price'] - signal_trades['expected_no']

    print(f"\n  NO Price vs Expected (100 - YES price):")
    print(f"    Mean diff: {signal_trades['no_price_diff'].mean():.2f}c")
    print(f"    These should be exactly 0 if YES price + NO price = 100")

    # Most S007 trades are at low YES prices (high NO prices)
    print(f"\n  Price Distribution (YES price buckets):")
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        bucket = signal_trades[(signal_trades['trade_price'] >= lo) & (signal_trades['trade_price'] < hi)]
        print(f"    {lo}-{hi}c: {len(bucket):,} trades ({len(bucket)/len(signal_trades)*100:.1f}%)")


def main():
    print("="*80)
    print("SESSION 012: CORRECT METHODOLOGY VALIDATION")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # First, understand what columns mean
    compare_methodology(df)

    # Deep dive on S007
    analyze_what_s007_actually_does(df)

    # Validate with correct columns
    result = validate_s007_correct(df)

    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"S007 Status: {result['status']}")
    print(f"Edge: {result.get('edge', 0)*100:.2f}%")
    print(f"Improvement over baseline: {result.get('improvement', 0)*100:.2f}%")


if __name__ == "__main__":
    main()
