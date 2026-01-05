"""
RLM Signal Reliability Analysis - YES% x min_trades Grid Search

Objective: Find the MOST RELIABLE signal configuration, NOT the highest Total EV.

Key Metrics:
1. False Positive Rate: P(trigger | random 50/50 trades) using binomial distribution
2. Win Rate: Actual market-level win rate
3. Edge: Average profit per signal
4. Statistical Significance: P-value for edge > 0
5. 95% Confidence Interval on edge

The goal is to answer: What configuration gives us highest CONFIDENCE the signal is real?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from scipy import stats
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets" / "market_outcomes_ALL.csv"
REPORTS_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load trades and market outcomes."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE)
    markets = pd.read_csv(MARKETS_FILE)

    print(f"  Trades: {len(trades):,}")
    print(f"  Markets: {len(markets):,}")

    # Rename columns to expected format
    trades = trades.rename(columns={
        'datetime': 'created_time',
        'trade_price': 'price_cents',
    })

    # Ensure timestamp parsing
    if 'created_time' in trades.columns:
        trades['created_time'] = pd.to_datetime(trades['created_time'])

    return trades, markets


def compute_false_positive_rate(min_trades: int, yes_threshold: float) -> float:
    """
    Compute false positive rate using binomial distribution.

    If trades were truly random (50/50), what's the probability of
    having >= (yes_threshold * min_trades) YES trades out of min_trades?

    This is P(X >= k) where X ~ Binomial(n=min_trades, p=0.5) and k = ceil(yes_threshold * min_trades)
    """
    # Required YES trades to trigger (must be > yes_threshold, so use floor + 1)
    required_yes = int(np.floor(yes_threshold * min_trades)) + 1

    # P(X >= required_yes) = 1 - P(X < required_yes) = 1 - CDF(required_yes - 1)
    fp_rate = 1 - binom.cdf(required_yes - 1, min_trades, 0.5)

    return fp_rate


def compute_rlm_signal(market_trades: pd.DataFrame,
                        yes_threshold: float = 0.65,
                        min_trades: int = 15,
                        min_price_drop: int = 2) -> Dict[str, Any]:
    """
    Compute RLM signal for a market.
    """
    if len(market_trades) < min_trades:
        return {'signal_triggered': False, 'reason': 'insufficient_trades'}

    # Compute YES ratio
    total_trades = len(market_trades)
    yes_trades = (market_trades['taker_side'] == 'yes').sum()
    yes_ratio = yes_trades / total_trades

    if yes_ratio <= yes_threshold:
        return {'signal_triggered': False, 'reason': 'low_yes_ratio', 'yes_ratio': yes_ratio}

    # YES price is already in the data
    first_yes_price = market_trades['yes_price'].iloc[0]
    last_yes_price = market_trades['yes_price'].iloc[-1]
    price_drop = first_yes_price - last_yes_price

    if price_drop < min_price_drop:
        return {
            'signal_triggered': False,
            'reason': 'insufficient_price_drop',
            'price_drop': price_drop,
            'yes_ratio': yes_ratio
        }

    # Compute NO price at signal time
    no_price_entry = 100 - last_yes_price

    return {
        'signal_triggered': True,
        'yes_ratio': yes_ratio,
        'price_drop': price_drop,
        'trade_count': total_trades,
        'first_yes_price': first_yes_price,
        'last_yes_price': last_yes_price,
        'no_price_entry': no_price_entry,
        'yes_trades': yes_trades,
    }


def compute_confidence_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for win rate.
    More accurate than normal approximation for extreme proportions.
    """
    if total == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total

    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

    return (max(0, center - margin), min(1, center + margin))


def compute_edge_statistics(win_rate: float, breakeven: float, n_markets: int) -> Dict[str, float]:
    """
    Compute edge statistics including p-value and confidence interval.

    H0: True win rate = breakeven (no edge)
    H1: True win rate > breakeven (positive edge)
    """
    edge = win_rate - breakeven

    # Standard error under H0
    se_h0 = np.sqrt(breakeven * (1 - breakeven) / n_markets)

    # Z-score
    if se_h0 > 0:
        z_score = edge / se_h0
        p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test
    else:
        z_score = 0
        p_value = 0.5

    # 95% CI on edge using actual win rate variance
    se_actual = np.sqrt(win_rate * (1 - win_rate) / n_markets)
    edge_ci_lower = edge - 1.96 * se_actual
    edge_ci_upper = edge + 1.96 * se_actual

    return {
        'edge': edge,
        'z_score': z_score,
        'p_value': p_value,
        'edge_ci_lower': edge_ci_lower,
        'edge_ci_upper': edge_ci_upper,
        'se': se_actual,
    }


def reliability_grid_search(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """
    Full grid search over YES% x min_trades combinations.
    Focus on RELIABILITY metrics, not Total EV.
    """
    print("\n" + "="*80)
    print("SIGNAL RELIABILITY GRID SEARCH: YES% x min_trades")
    print("="*80)

    # Grid parameters
    yes_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    min_trades_options = [10, 15, 20, 25, 30, 40, 50]

    # Fixed price drop = 2c (can be changed)
    min_price_drop = 2

    all_results = []

    for yes_thresh in yes_thresholds:
        for min_trades in min_trades_options:
            # Compute false positive rate (theoretical)
            fp_rate = compute_false_positive_rate(min_trades, yes_thresh)

            # Compute signals for all markets
            market_signals = {}
            for ticker, group in trades.groupby('market_ticker'):
                group = group.sort_values('created_time')
                signal = compute_rlm_signal(group, yes_thresh, min_trades, min_price_drop)
                if signal['signal_triggered']:
                    market_signals[ticker] = signal

            if len(market_signals) < 30:  # Need minimum sample
                continue

            # Build dataframe and merge with outcomes
            signals_df = pd.DataFrame.from_dict(market_signals, orient='index')
            signals_df['market_ticker'] = signals_df.index

            markets_subset = markets[['ticker', 'result']].copy()
            markets_subset['result_yes'] = (markets_subset['result'] == 'yes').astype(int)

            signals_df = signals_df.merge(markets_subset[['ticker', 'result_yes']],
                                           left_on='market_ticker', right_on='ticker', how='inner')
            signals_df['no_win'] = 1 - signals_df['result_yes']

            # Compute statistics
            n_markets = len(signals_df)
            wins = signals_df['no_win'].sum()
            win_rate = wins / n_markets
            avg_no_price = signals_df['no_price_entry'].mean()
            breakeven = avg_no_price / 100

            # Compute edge with statistics
            edge_stats = compute_edge_statistics(win_rate, breakeven, n_markets)

            # Compute win rate confidence interval
            wr_ci_lower, wr_ci_upper = compute_confidence_interval(wins, n_markets)

            # EV calculations
            ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price
            total_ev = ev_per_100 * n_markets

            # Required YES trades to trigger
            required_yes = int(np.floor(yes_thresh * min_trades)) + 1

            all_results.append({
                'yes_threshold': yes_thresh,
                'min_trades': min_trades,
                'required_yes': required_yes,
                'false_positive_rate': fp_rate,
                'markets': n_markets,
                'wins': wins,
                'win_rate': win_rate,
                'wr_ci_lower': wr_ci_lower,
                'wr_ci_upper': wr_ci_upper,
                'avg_no_price': avg_no_price,
                'breakeven': breakeven,
                'edge': edge_stats['edge'],
                'edge_ci_lower': edge_stats['edge_ci_lower'],
                'edge_ci_upper': edge_stats['edge_ci_upper'],
                'p_value': edge_stats['p_value'],
                'z_score': edge_stats['z_score'],
                'ev_per_100': ev_per_100,
                'total_ev': total_ev,
            })

    results_df = pd.DataFrame(all_results)
    return results_df


def create_reliability_matrix(results_df: pd.DataFrame) -> str:
    """
    Create a formatted reliability matrix for the report.
    """
    output = []
    output.append("\n" + "="*120)
    output.append("RELIABILITY MATRIX: YES Threshold vs Min Trades")
    output.append("="*120)
    output.append("\nCell format: FP Rate | Win Rate | Edge | p-value | Markets")
    output.append("-"*120)

    # Pivot for each metric
    min_trades_values = sorted(results_df['min_trades'].unique())
    yes_thresh_values = sorted(results_df['yes_threshold'].unique())

    # Header
    header = f"{'YES>':<8}"
    for mt in min_trades_values:
        header += f"| min_trades={mt:<4} "
    output.append(header)
    output.append("-"*120)

    for yt in yes_thresh_values:
        row = f"{yt:.0%}    "
        for mt in min_trades_values:
            cell = results_df[(results_df['yes_threshold'] == yt) &
                              (results_df['min_trades'] == mt)]
            if len(cell) > 0:
                c = cell.iloc[0]
                fp = c['false_positive_rate']
                wr = c['win_rate']
                edge = c['edge']
                pval = c['p_value']
                n = c['markets']

                # Color coding via text markers
                fp_mark = "*" if fp < 0.25 else (" " if fp < 0.35 else "!")
                wr_mark = "*" if wr > 0.93 else " "
                p_mark = "*" if pval < 0.01 else (" " if pval < 0.05 else "!")

                row += f"| {fp_mark}{fp:>4.1%} {wr_mark}{wr:>4.1%} {edge:>+5.1%} {p_mark}{pval:.3f} {n:>4} "
            else:
                row += f"| {'N/A':<28} "
        output.append(row)

    output.append("-"*120)
    output.append("Legend: * = Good (FP<25%, WR>93%, p<0.01)  ! = Warning (FP>35%, p>0.05)")

    return "\n".join(output)


def find_optimal_reliable_config(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Find the optimal configuration based on RELIABILITY, not Total EV.

    Criteria (in priority order):
    1. False Positive Rate < 25%
    2. Win Rate > 93%
    3. P-value < 0.01
    4. Markets > 500 (sufficient sample)
    5. Highest Edge among qualifying
    """
    print("\n" + "="*80)
    print("FINDING OPTIMAL RELIABLE CONFIGURATION")
    print("="*80)

    # Tier 1: Strictest criteria
    tier1 = results_df[
        (results_df['false_positive_rate'] < 0.25) &
        (results_df['win_rate'] > 0.93) &
        (results_df['p_value'] < 0.01) &
        (results_df['markets'] > 500)
    ].copy()

    # Tier 2: Relaxed criteria
    tier2 = results_df[
        (results_df['false_positive_rate'] < 0.30) &
        (results_df['win_rate'] > 0.92) &
        (results_df['p_value'] < 0.05) &
        (results_df['markets'] > 300)
    ].copy()

    # Tier 3: Most relaxed
    tier3 = results_df[
        (results_df['false_positive_rate'] < 0.35) &
        (results_df['win_rate'] > 0.90) &
        (results_df['p_value'] < 0.05) &
        (results_df['markets'] > 100)
    ].copy()

    recommendations = {}

    print("\n--- Tier 1 (Strictest): FP<25%, WR>93%, p<0.01, N>500 ---")
    if len(tier1) > 0:
        # Sort by edge (highest), then by lowest FP rate
        tier1_sorted = tier1.sort_values(['edge', 'false_positive_rate'],
                                          ascending=[False, True])
        best = tier1_sorted.iloc[0]
        recommendations['tier1_best'] = best.to_dict()
        print(f"  FOUND {len(tier1)} qualifying configurations!")
        print(f"  Best: YES>{best['yes_threshold']:.0%}, min_trades={best['min_trades']}")
        print(f"    FP Rate: {best['false_positive_rate']:.1%}")
        print(f"    Win Rate: {best['win_rate']:.1%}")
        print(f"    Edge: {best['edge']:.1%} [{best['edge_ci_lower']:.1%}, {best['edge_ci_upper']:.1%}]")
        print(f"    p-value: {best['p_value']:.4f}")
        print(f"    Markets: {best['markets']:,}")
    else:
        print("  No configurations meet Tier 1 criteria.")
        recommendations['tier1_best'] = None

    print("\n--- Tier 2 (Moderate): FP<30%, WR>92%, p<0.05, N>300 ---")
    if len(tier2) > 0:
        tier2_sorted = tier2.sort_values(['edge', 'false_positive_rate'],
                                          ascending=[False, True])
        best = tier2_sorted.iloc[0]
        recommendations['tier2_best'] = best.to_dict()
        print(f"  FOUND {len(tier2)} qualifying configurations!")
        print(f"  Best: YES>{best['yes_threshold']:.0%}, min_trades={best['min_trades']}")
        print(f"    FP Rate: {best['false_positive_rate']:.1%}")
        print(f"    Win Rate: {best['win_rate']:.1%}")
        print(f"    Edge: {best['edge']:.1%}")
        print(f"    p-value: {best['p_value']:.4f}")
        print(f"    Markets: {best['markets']:,}")
    else:
        print("  No configurations meet Tier 2 criteria.")
        recommendations['tier2_best'] = None

    print("\n--- Tier 3 (Relaxed): FP<35%, WR>90%, p<0.05, N>100 ---")
    if len(tier3) > 0:
        tier3_sorted = tier3.sort_values(['edge', 'false_positive_rate'],
                                          ascending=[False, True])
        best = tier3_sorted.iloc[0]
        recommendations['tier3_best'] = best.to_dict()
        print(f"  FOUND {len(tier3)} qualifying configurations!")
        print(f"  Best: YES>{best['yes_threshold']:.0%}, min_trades={best['min_trades']}")
        print(f"    FP Rate: {best['false_positive_rate']:.1%}")
        print(f"    Win Rate: {best['win_rate']:.1%}")
        print(f"    Edge: {best['edge']:.1%}")
        print(f"    p-value: {best['p_value']:.4f}")
        print(f"    Markets: {best['markets']:,}")
    else:
        print("  No configurations meet Tier 3 criteria.")
        recommendations['tier3_best'] = None

    return recommendations


def rank_by_reliability(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all configurations by a composite reliability score.

    Score = (1 - FP_rate) * 0.4 + Win_rate * 0.3 + (1 - p_value) * 0.2 + CI_width_inverse * 0.1
    """
    df = results_df.copy()

    # Normalize metrics to [0, 1]
    df['fp_score'] = 1 - df['false_positive_rate']  # Lower FP = better
    df['wr_score'] = df['win_rate']
    df['p_score'] = 1 - df['p_value'].clip(upper=1)  # Lower p = better

    # CI width (tighter = better)
    df['ci_width'] = df['edge_ci_upper'] - df['edge_ci_lower']
    max_width = df['ci_width'].max()
    df['ci_score'] = 1 - (df['ci_width'] / max_width) if max_width > 0 else 1

    # Composite reliability score
    df['reliability_score'] = (
        df['fp_score'] * 0.40 +
        df['wr_score'] * 0.30 +
        df['p_score'] * 0.20 +
        df['ci_score'] * 0.10
    )

    return df.sort_values('reliability_score', ascending=False)


def print_detailed_results(results_df: pd.DataFrame):
    """Print detailed results table."""
    print("\n" + "="*140)
    print("DETAILED RESULTS: All YES% x min_trades Combinations")
    print("="*140)
    print(f"{'YES>':<6} {'MinT':<6} {'ReqY':<5} {'FP%':<7} {'N':<6} {'WR%':<7} {'Edge':<7} "
          f"{'CI_Low':<7} {'CI_Up':<7} {'p-val':<8} {'EV/$100':<10} {'TotalEV':<12}")
    print("-"*140)

    for _, row in results_df.sort_values(['yes_threshold', 'min_trades']).iterrows():
        fp_flag = "*" if row['false_positive_rate'] < 0.25 else ""
        p_flag = "*" if row['p_value'] < 0.01 else ""

        print(f"{row['yes_threshold']:<5.0%} {row['min_trades']:<6} {row['required_yes']:<5} "
              f"{row['false_positive_rate']:<6.1%}{fp_flag} {row['markets']:<6} "
              f"{row['win_rate']:<6.1%} {row['edge']:<+6.1%} "
              f"{row['edge_ci_lower']:<+6.1%} {row['edge_ci_upper']:<+6.1%} "
              f"{row['p_value']:<7.4f}{p_flag} ${row['ev_per_100']:<9.2f} ${row['total_ev']:<11,.0f}")


def compare_to_current(results_df: pd.DataFrame):
    """Compare configurations to current settings."""
    print("\n" + "="*80)
    print("COMPARISON TO CURRENT CONFIGURATION")
    print("="*80)

    # Current: YES>65%, min_trades=15
    current = results_df[(results_df['yes_threshold'] == 0.65) &
                         (results_df['min_trades'] == 15)]

    if len(current) == 0:
        print("Current configuration not found in results.")
        return

    curr = current.iloc[0]

    print(f"\nCurrent (YES>65%, min_trades=15):")
    print(f"  False Positive Rate: {curr['false_positive_rate']:.1%}")
    print(f"  Win Rate: {curr['win_rate']:.1%}")
    print(f"  Edge: {curr['edge']:.1%} [{curr['edge_ci_lower']:.1%}, {curr['edge_ci_upper']:.1%}]")
    print(f"  p-value: {curr['p_value']:.4f}")
    print(f"  Markets: {curr['markets']:,}")

    # Find configurations with lower FP rate but similar edge
    better_fp = results_df[
        (results_df['false_positive_rate'] < curr['false_positive_rate']) &
        (results_df['edge'] >= curr['edge'] * 0.9)  # Within 10% of current edge
    ].sort_values('false_positive_rate')

    print(f"\nConfigurations with lower FP rate (within 10% edge):")
    if len(better_fp) > 0:
        for _, row in better_fp.head(5).iterrows():
            print(f"  YES>{row['yes_threshold']:.0%}, min_trades={row['min_trades']}: "
                  f"FP={row['false_positive_rate']:.1%}, Edge={row['edge']:.1%}, N={row['markets']:,}")
    else:
        print("  None found - current config is already optimal for FP rate at this edge level.")


def main():
    """Run full reliability grid search analysis."""
    print("="*80)
    print("RLM SIGNAL RELIABILITY ANALYSIS")
    print("YES% x min_trades Grid Search")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nObjective: Find the MOST RELIABLE signal, not highest Total EV")

    # Load data
    trades, markets = load_data()

    # Run grid search
    results_df = reliability_grid_search(trades, markets)

    # Print detailed results
    print_detailed_results(results_df)

    # Create reliability matrix
    matrix = create_reliability_matrix(results_df)
    print(matrix)

    # Rank by reliability
    ranked_df = rank_by_reliability(results_df)

    print("\n" + "="*80)
    print("TOP 10 BY COMPOSITE RELIABILITY SCORE")
    print("="*80)
    print(f"{'Rank':<5} {'YES>':<6} {'MinT':<6} {'FP%':<7} {'WR%':<7} {'Edge':<7} {'p-val':<8} {'Score':<7}")
    print("-"*80)

    for i, (_, row) in enumerate(ranked_df.head(10).iterrows(), 1):
        print(f"{i:<5} {row['yes_threshold']:<5.0%} {row['min_trades']:<6} "
              f"{row['false_positive_rate']:<6.1%} {row['win_rate']:<6.1%} "
              f"{row['edge']:<+6.1%} {row['p_value']:<7.4f} {row['reliability_score']:<6.3f}")

    # Find optimal reliable configuration
    recommendations = find_optimal_reliable_config(results_df)

    # Compare to current
    compare_to_current(results_df)

    # Executive Summary
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY: MOST RELIABLE CONFIGURATION")
    print("="*80)

    if recommendations.get('tier1_best'):
        best = recommendations['tier1_best']
        print(f"\nRECOMMENDED (Tier 1 - Highest Reliability):")
        print(f"  YES Threshold: >{best['yes_threshold']:.0%}")
        print(f"  Min Trades: {best['min_trades']}")
        print(f"  Required YES Trades: {best['required_yes']}/{best['min_trades']} ({best['required_yes']/best['min_trades']:.0%})")
        print(f"\nReliability Metrics:")
        print(f"  False Positive Rate: {best['false_positive_rate']:.1%} (if random, chance of false trigger)")
        print(f"  Win Rate: {best['win_rate']:.1%}")
        print(f"  Edge: {best['edge']:.1%}")
        print(f"  95% CI on Edge: [{best['edge_ci_lower']:.1%}, {best['edge_ci_upper']:.1%}]")
        print(f"  p-value: {best['p_value']:.4f}")
        print(f"  Sample Size: {best['markets']:,} markets")
    elif recommendations.get('tier2_best'):
        best = recommendations['tier2_best']
        print(f"\nRECOMMENDED (Tier 2 - Moderate Reliability):")
        print(f"  YES Threshold: >{best['yes_threshold']:.0%}")
        print(f"  Min Trades: {best['min_trades']}")
        print(f"  False Positive Rate: {best['false_positive_rate']:.1%}")
        print(f"  Win Rate: {best['win_rate']:.1%}")
        print(f"  Edge: {best['edge']:.1%}")
        print(f"  p-value: {best['p_value']:.4f}")
        print(f"  Sample Size: {best['markets']:,} markets")
    else:
        print("\nNo configuration meets reliability criteria. Consider collecting more data.")

    # Save results
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'objective': 'Signal Reliability Analysis',
            'price_drop_threshold': 2,
            'trades_analyzed': len(trades),
            'markets_analyzed': len(markets),
        },
        'grid_results': results_df.to_dict('records'),
        'ranked_by_reliability': ranked_df.head(20).to_dict('records'),
        'recommendations': {
            'tier1': recommendations.get('tier1_best'),
            'tier2': recommendations.get('tier2_best'),
            'tier3': recommendations.get('tier3_best'),
        }
    }

    output_file = REPORTS_DIR / 'rlm_reliability_grid_search.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results_df, recommendations


if __name__ == "__main__":
    results_df, recommendations = main()
