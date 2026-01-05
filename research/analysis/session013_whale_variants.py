"""
Session 013: Whale Follower Strategy Variants Analysis

Test all whale-based strategies with Session 012c strict methodology:
- Bucket-by-bucket baseline comparison
- If improvement is 0 or negative at most buckets = PRICE PROXY = REJECT

Strategies tested:
1. Whale Baseline (follow whale's side)
2. Whale NO Only
3. Whale YES Only
4. Whale Low Leverage (follow and fade)
5. Whale + S013 Filter
6. S013 Signal Frequency Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'

# Whale definition: trade value >= $100 (10000 cents)
WHALE_THRESHOLD_CENTS = 10000


def load_data():
    """Load and prepare the enriched trades data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Calculate trade value in cents
    df['trade_value_cents'] = df['count'] * df['trade_price']
    df['is_whale'] = df['trade_value_cents'] >= WHALE_THRESHOLD_CENTS

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Whale trades (>=$100): {df['is_whale'].sum():,} ({df['is_whale'].mean()*100:.1f}%)")

    return df


def build_baseline_5c(df):
    """
    Build baseline win rates for ALL markets at each 5c NO price bucket.
    This is the critical comparison standard.
    """
    print("\nBuilding 5c bucket baseline...")

    # Get market-level aggregates
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'first'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    # Calculate baseline win rates
    baseline_no = {}  # For NO bets
    baseline_yes = {}  # For YES bets

    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        no_wins = (bucket_markets['market_result'] == 'no').sum()
        yes_wins = (bucket_markets['market_result'] == 'yes').sum()

        if n >= 20:
            baseline_no[bucket] = {
                'win_rate': no_wins / n,
                'n_markets': n
            }
            baseline_yes[bucket] = {
                'win_rate': yes_wins / n,
                'n_markets': n
            }

    print(f"Built baseline for {len(baseline_no)} price buckets")
    return all_markets, baseline_no, baseline_yes


def validate_strategy_strict(signal_markets, baseline, strategy_name, side='no'):
    """
    Strict validation with bucket-by-bucket comparison.

    Args:
        signal_markets: DataFrame with 'market_result' and 'no_price' or 'yes_price'
        baseline: dict of bucket -> {win_rate, n_markets}
        strategy_name: Name for display
        side: 'no' or 'yes' - which side we're betting
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {strategy_name}")
    print(f"{'='*80}")

    n = len(signal_markets)
    if n < 50:
        print(f"  REJECTED: Only {n} markets (need >= 50)")
        return {'status': 'rejected', 'reason': 'insufficient_markets', 'n': n}

    # Calculate win rate for our side
    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    # Calculate breakeven based on side
    if side == 'no':
        avg_price = signal_markets['no_price'].mean()
    else:
        avg_price = signal_markets['yes_price'].mean()

    be = avg_price / 100
    edge = wr - be

    # P-value
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\n  Basic Stats:")
    print(f"    Markets: {n}")
    print(f"    {side.upper()} Win Rate: {wr:.1%}")
    print(f"    Avg {side.upper()} Price: {avg_price:.1f}c")
    print(f"    Breakeven: {be:.1%}")
    print(f"    Raw Edge: {edge*100:.2f}%")
    print(f"    P-value: {p_value:.2e}")

    # CRITICAL: Bucket-by-bucket comparison
    print(f"\n  Price Proxy Check (5c Bucket Analysis):")

    if side == 'no':
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['no_price'] // 5) * 5
    else:
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['yes_price'] // 5) * 5

    improvements = []
    print(f"  {'Bucket':<10} {'Sig WR':<10} {'Base WR':<10} {'Improve':<12} {'N Sig':<8}")

    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:  # Minimum sample
            continue

        sig_wr = (sig_bucket['market_result'] == side).mean()
        base_wr = baseline[bucket]['win_rate']
        n_base = baseline[bucket]['n_markets']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n_sig': n_sig,
            'n_base': n_base
        })

        sign = '+' if imp >= 0 else ''
        print(f"  {bucket:.0f}-{bucket+5:.0f}c    "
              f"{sig_wr:.1%}      "
              f"{base_wr:.1%}      "
              f"{sign}{imp*100:.2f}%       "
              f"{n_sig:<8}")

    if not improvements:
        print(f"  REJECTED: No buckets with sufficient data")
        return {'status': 'rejected', 'reason': 'no_buckets', 'n': n}

    # Calculate weighted improvement
    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    # Count positive/negative buckets
    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    neg_buckets = sum(1 for i in improvements if i['improvement'] <= 0)
    total_buckets = len(improvements)

    print(f"\n  Summary:")
    print(f"    Weighted Improvement: {weighted_imp*100:+.2f}%")
    print(f"    Positive Buckets: {pos_buckets}/{total_buckets}")
    print(f"    Negative/Zero Buckets: {neg_buckets}/{total_buckets}")

    # Determine verdict
    result = {
        'n': n,
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'neg_buckets': neg_buckets,
        'total_buckets': total_buckets,
        'bucket_details': improvements
    }

    if p_value > 0.01:
        print(f"\n  VERDICT: NOT SIGNIFICANT (p={p_value:.4f} > 0.01)")
        result['status'] = 'rejected'
        result['reason'] = 'not_significant'
        result['verdict'] = 'REJECTED - NOT SIGNIFICANT'
    elif weighted_imp <= 0:
        print(f"\n  VERDICT: PRICE PROXY (weighted improvement <= 0)")
        result['status'] = 'rejected'
        result['reason'] = 'price_proxy'
        result['verdict'] = 'REJECTED - PRICE PROXY'
    elif pos_buckets <= neg_buckets:
        print(f"\n  VERDICT: PRICE PROXY (more negative than positive buckets)")
        result['status'] = 'rejected'
        result['reason'] = 'price_proxy_buckets'
        result['verdict'] = 'REJECTED - PRICE PROXY'
    elif weighted_imp < 0.02:
        print(f"\n  VERDICT: MARGINAL (improvement < 2%)")
        result['status'] = 'marginal'
        result['verdict'] = 'MARGINAL'
    else:
        print(f"\n  VERDICT: VALIDATED (genuine improvement over baseline)")
        result['status'] = 'validated'
        result['verdict'] = 'VALIDATED'

    return result


def test_whale_baseline(df, all_markets, baseline_no, baseline_yes):
    """
    Strategy 1: Whale Baseline - Follow whale's side
    """
    print("\n" + "="*80)
    print("STRATEGY 1: WHALE BASELINE (Follow Whale's Side)")
    print("="*80)

    whale_trades = df[df['is_whale']]
    print(f"Total whale trades: {len(whale_trades):,}")

    # Get markets with whale activity and their dominant whale direction
    whale_markets = whale_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: x.mode()[0] if len(x) > 0 else 'unknown',  # Most common side
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'trade_value_cents': 'sum',  # Total whale volume
        'count': 'size'  # Number of whale trades
    }).reset_index()
    whale_markets.columns = ['market_ticker', 'whale_side', 'market_result', 'no_price', 'yes_price', 'whale_volume', 'n_whale_trades']

    print(f"Markets with whale activity: {len(whale_markets):,}")

    # Calculate win rate when following whale's side
    whale_markets['followed_correctly'] = whale_markets['whale_side'] == whale_markets['market_result']

    n = len(whale_markets)
    wins = whale_markets['followed_correctly'].sum()
    wr = wins / n

    print(f"\n  Following Whale Side Stats:")
    print(f"    Markets: {n}")
    print(f"    Win Rate (whale was right): {wr:.1%}")
    print(f"    Whale bet YES: {(whale_markets['whale_side'] == 'yes').sum()}")
    print(f"    Whale bet NO: {(whale_markets['whale_side'] == 'no').sum()}")

    # This is harder to bucket-test since we're following variable sides
    # Let's report but note it's not directly comparable

    return {
        'strategy': 'Whale Baseline (Follow Side)',
        'n': n,
        'win_rate': float(wr),
        'whale_yes': int((whale_markets['whale_side'] == 'yes').sum()),
        'whale_no': int((whale_markets['whale_side'] == 'no').sum()),
        'verdict': 'NOT DIRECTLY TESTABLE - Variable sides',
        'status': 'baseline'
    }


def test_whale_no_only(df, all_markets, baseline_no):
    """
    Strategy 2: Whale NO Only - Only follow when whale bets NO
    """
    print("\n" + "="*80)
    print("STRATEGY 2: WHALE NO ONLY")
    print("="*80)

    # Find whale NO trades
    whale_no_trades = df[(df['is_whale']) & (df['taker_side'] == 'no')]
    print(f"Whale NO trades: {len(whale_no_trades):,}")

    # Get unique markets with whale NO activity
    whale_no_markets = whale_no_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'trade_value_cents': 'sum',
        'count': 'size'
    }).reset_index()
    whale_no_markets.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'whale_volume', 'n_whale_trades']

    return validate_strategy_strict(whale_no_markets, baseline_no, "Whale NO Only", side='no')


def test_whale_yes_only(df, all_markets, baseline_yes):
    """
    Strategy 3: Whale YES Only - Only follow when whale bets YES
    """
    print("\n" + "="*80)
    print("STRATEGY 3: WHALE YES ONLY")
    print("="*80)

    # Find whale YES trades
    whale_yes_trades = df[(df['is_whale']) & (df['taker_side'] == 'yes')]
    print(f"Whale YES trades: {len(whale_yes_trades):,}")

    # Get unique markets with whale YES activity
    whale_yes_markets = whale_yes_trades.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'trade_value_cents': 'sum',
        'count': 'size'
    }).reset_index()
    whale_yes_markets.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'whale_volume', 'n_whale_trades']

    return validate_strategy_strict(whale_yes_markets, baseline_yes, "Whale YES Only", side='yes')


def test_whale_low_leverage_follow(df, all_markets, baseline_no):
    """
    Strategy 4a: Whale Low Leverage FOLLOW - Whale with leverage < 2, follow their NO
    """
    print("\n" + "="*80)
    print("STRATEGY 4a: WHALE LOW LEVERAGE FOLLOW (NO)")
    print("="*80)

    # Find whale NO trades with low leverage
    whale_low_lev_no = df[(df['is_whale']) & (df['taker_side'] == 'no') & (df['leverage_ratio'] < 2)]
    print(f"Whale NO trades with leverage < 2: {len(whale_low_lev_no):,}")

    # Get unique markets
    signal_markets = whale_low_lev_no.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'leverage_ratio': 'mean',
        'count': 'size'
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'avg_leverage', 'n_trades']

    return validate_strategy_strict(signal_markets, baseline_no, "Whale Low Leverage Follow (NO)", side='no')


def test_whale_low_leverage_fade(df, all_markets, baseline_no):
    """
    Strategy 4b: Whale Low Leverage FADE - Whale bets YES with low leverage, we bet NO
    """
    print("\n" + "="*80)
    print("STRATEGY 4b: WHALE LOW LEVERAGE FADE (Bet NO when whale bets YES)")
    print("="*80)

    # Find whale YES trades with low leverage (we'll fade them by betting NO)
    whale_low_lev_yes = df[(df['is_whale']) & (df['taker_side'] == 'yes') & (df['leverage_ratio'] < 2)]
    print(f"Whale YES trades with leverage < 2: {len(whale_low_lev_yes):,}")

    # Get unique markets (we bet NO to fade)
    signal_markets = whale_low_lev_yes.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'leverage_ratio': 'mean',
        'count': 'size'
    }).reset_index()
    signal_markets.columns = ['market_ticker', 'market_result', 'no_price', 'yes_price', 'avg_leverage', 'n_trades']

    return validate_strategy_strict(signal_markets, baseline_no, "Whale Low Leverage Fade (bet NO)", side='no')


def test_whale_plus_s013(df, all_markets, baseline_no):
    """
    Strategy 5: Whale + S013 Filter
    Combine whale NO signal with S013 conditions (leverage_std < 0.7, no_ratio > 0.5)
    """
    print("\n" + "="*80)
    print("STRATEGY 5: WHALE + S013 FILTER")
    print("="*80)

    # First, calculate S013 conditions per market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['std', 'mean'],
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_std', 'lev_mean', 'no_ratio', 'market_result', 'no_price', 'yes_price', 'n_trades']

    # Find markets with whale NO activity
    whale_no_trades = df[(df['is_whale']) & (df['taker_side'] == 'no')]
    whale_no_markets = set(whale_no_trades['market_ticker'].unique())

    # Apply combined filter: whale NO + S013 conditions
    signal_markets = market_stats[
        (market_stats['market_ticker'].isin(whale_no_markets)) &
        (market_stats['lev_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    ].copy()

    print(f"Markets with whale NO: {len(whale_no_markets):,}")
    print(f"Markets passing S013 filter: {len(market_stats[(market_stats['lev_std'] < 0.7) & (market_stats['no_ratio'] > 0.5) & (market_stats['n_trades'] >= 5)]):,}")
    print(f"Markets with BOTH (whale NO + S013): {len(signal_markets):,}")

    return validate_strategy_strict(signal_markets, baseline_no, "Whale + S013 Filter (NO)", side='no')


def analyze_s013_frequency(df):
    """
    Strategy 6: S013 Signal Frequency Analysis
    How often does S013 trigger? Is it practical for continuous trading?
    """
    print("\n" + "="*80)
    print("STRATEGY 6: S013 SIGNAL FREQUENCY ANALYSIS")
    print("="*80)

    # Calculate S013 conditions per market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'count': 'size',
        'datetime': ['min', 'max']
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_std', 'no_ratio', 'market_result', 'no_price', 'n_trades', 'first_trade', 'last_trade']

    # Apply S013 filter
    s013_markets = market_stats[
        (market_stats['lev_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    ].copy()

    print(f"\nS013 Signal Markets: {len(s013_markets):,}")

    # Calculate frequency
    if len(s013_markets) > 0:
        date_range = (market_stats['last_trade'].max() - market_stats['first_trade'].min()).days
        signals_per_day = len(s013_markets) / max(date_range, 1)

        print(f"Data spans: {date_range} days")
        print(f"S013 signals per day (avg): {signals_per_day:.1f}")
        print(f"S013 signals per week (avg): {signals_per_day * 7:.1f}")

        # Win rate and edge
        wins = (s013_markets['market_result'] == 'no').sum()
        wr = wins / len(s013_markets)
        avg_no = s013_markets['no_price'].mean()
        be = avg_no / 100
        edge = wr - be

        print(f"\nS013 Performance:")
        print(f"  Win Rate: {wr:.1%}")
        print(f"  Avg NO Price: {avg_no:.1f}c")
        print(f"  Breakeven: {be:.1%}")
        print(f"  Edge: {edge*100:+.2f}%")

        return {
            'n_signals': len(s013_markets),
            'date_range_days': int(date_range),
            'signals_per_day': float(signals_per_day),
            'signals_per_week': float(signals_per_day * 7),
            'win_rate': float(wr),
            'avg_no_price': float(avg_no),
            'edge': float(edge)
        }

    return {'n_signals': 0, 'error': 'No S013 signals found'}


def compare_s013_standalone(df, baseline_no):
    """
    Re-validate pure S013 for comparison
    """
    print("\n" + "="*80)
    print("COMPARISON: PURE S013 (Low Leverage Variance NO)")
    print("="*80)

    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': 'std',
        'taker_side': lambda x: (x == 'no').mean(),
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'size'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_std', 'no_ratio', 'market_result', 'no_price', 'yes_price', 'n_trades']

    s013_markets = market_stats[
        (market_stats['lev_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    ].copy()

    return validate_strategy_strict(s013_markets, baseline_no, "Pure S013 (Baseline Comparison)", side='no')


def main():
    print("="*80)
    print("SESSION 013: WHALE FOLLOWER STRATEGY VARIANTS ANALYSIS")
    print(f"Started: {datetime.now()}")
    print("="*80)
    print("\nObjective: Find trade-feed-based improvements over pure S013")
    print("Methodology: Session 012c bucket-by-bucket baseline comparison")
    print("Criterion: Signal must beat baseline at SAME price levels")

    # Load data
    df = load_data()

    # Build baselines
    all_markets, baseline_no, baseline_yes = build_baseline_5c(df)

    # Print baseline summary
    print("\n" + "="*80)
    print("BASELINE WIN RATES (ALL MARKETS)")
    print("="*80)
    print("\nNO Baseline (5c buckets):")
    for bucket in sorted(baseline_no.keys()):
        b = baseline_no[bucket]
        print(f"  {bucket:.0f}-{bucket+5:.0f}c: {b['win_rate']:.1%} ({b['n_markets']} markets)")

    # Run all tests
    results = {}

    # Strategy 1: Whale Baseline
    results['whale_baseline'] = test_whale_baseline(df, all_markets, baseline_no, baseline_yes)

    # Strategy 2: Whale NO Only
    results['whale_no_only'] = test_whale_no_only(df, all_markets, baseline_no)

    # Strategy 3: Whale YES Only
    results['whale_yes_only'] = test_whale_yes_only(df, all_markets, baseline_yes)

    # Strategy 4a: Whale Low Leverage Follow
    results['whale_low_lev_follow'] = test_whale_low_leverage_follow(df, all_markets, baseline_no)

    # Strategy 4b: Whale Low Leverage Fade
    results['whale_low_lev_fade'] = test_whale_low_leverage_fade(df, all_markets, baseline_no)

    # Strategy 5: Whale + S013 Filter
    results['whale_plus_s013'] = test_whale_plus_s013(df, all_markets, baseline_no)

    # Strategy 6: S013 Frequency
    results['s013_frequency'] = analyze_s013_frequency(df)

    # Comparison: Pure S013
    results['pure_s013'] = compare_s013_standalone(df, baseline_no)

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\n{:<35} {:>10} {:>10} {:>12} {:>12} {:>20}".format(
        "Strategy", "N Markets", "Raw Edge", "Improve", "Pos Buckets", "Verdict"))
    print("-"*100)

    for name, result in results.items():
        if name == 's013_frequency':
            continue

        n = result.get('n', 0)
        edge = result.get('edge', 0) * 100 if result.get('edge') else 0
        imp = result.get('improvement', 0) * 100 if result.get('improvement') else 0
        pos = result.get('pos_buckets', '-')
        total = result.get('total_buckets', '-')
        verdict = result.get('verdict', 'N/A')

        bucket_str = f"{pos}/{total}" if pos != '-' else '-'

        print("{:<35} {:>10} {:>10.2f}% {:>11.2f}% {:>12} {:>20}".format(
            name, n, edge, imp, bucket_str, verdict[:20]))

    # Print S013 frequency
    freq = results['s013_frequency']
    print(f"\nS013 Signal Frequency:")
    print(f"  Signals per day: {freq.get('signals_per_day', 0):.1f}")
    print(f"  Signals per week: {freq.get('signals_per_week', 0):.1f}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find best strategy
    validated = [(k, v) for k, v in results.items()
                 if v.get('status') == 'validated' and k != 's013_frequency']

    if validated:
        best = max(validated, key=lambda x: x[1].get('improvement', 0))
        print(f"\nBest validated strategy: {best[0]}")
        print(f"  Edge: {best[1].get('edge', 0)*100:+.2f}%")
        print(f"  Improvement vs baseline: {best[1].get('improvement', 0)*100:+.2f}%")
        print(f"  Positive buckets: {best[1].get('pos_buckets', 0)}/{best[1].get('total_buckets', 0)}")
    else:
        print("\nNo whale-based strategy provides validated improvement over baseline.")
        print("RECOMMENDATION: Implement pure S013 (Low Leverage Variance NO)")
        print(f"\nPure S013 Performance:")
        s013 = results['pure_s013']
        print(f"  Markets: {s013.get('n', 0)}")
        print(f"  Edge: {s013.get('edge', 0)*100:+.2f}%")
        print(f"  Improvement vs baseline: {s013.get('improvement', 0)*100:+.2f}%")
        print(f"  Signal frequency: {freq.get('signals_per_day', 0):.1f} per day")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'/Users/samuelclark/Desktop/kalshiflow/research/reports/session013_whale_variants.json'

    # Convert bucket_details to serializable format
    for key in results:
        if 'bucket_details' in results[key]:
            results[key]['bucket_details'] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in b.items()}
                for b in results[key]['bucket_details']
            ]

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*80}")
    print(f"Session 013 completed: {datetime.now()}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
