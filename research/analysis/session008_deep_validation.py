#!/usr/bin/env python3
"""
Session 008: Deep Validation of Promising Findings
Rigorously validate H065 (Leverage) and H052 (Flow ROC)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

DATA_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv")
OUTPUT_PATH = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports")

MIN_MARKETS = 50
MAX_CONCENTRATION = 0.30
ALPHA = 0.05
BONFERRONI_ALPHA = 0.05 / 20

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def rigorous_validation(df, strategy_trades, strategy_name, description):
    """
    Apply full validation criteria:
    1. N >= 50 markets
    2. Concentration < 30%
    3. Statistical significance (p < 0.05)
    4. Temporal stability (works in different periods)
    5. Economic explanation
    """
    print(f"\n{'='*80}")
    print(f"RIGOROUS VALIDATION: {strategy_name}")
    print(f"Description: {description}")
    print("="*80)

    if len(strategy_trades) == 0:
        print("NO TRADES - REJECTED")
        return None

    # Market-level aggregation
    market_stats = strategy_trades.groupby('market_ticker').agg({
        'trade_price': 'mean',
        'is_winner': 'first',
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum',
        'datetime': 'first'  # For temporal analysis
    }).reset_index()

    n_markets = len(market_stats)
    print(f"\n1. SAMPLE SIZE: {n_markets} markets")
    if n_markets < MIN_MARKETS:
        print(f"   FAIL: Need >= {MIN_MARKETS}")
        return {'status': 'REJECTED', 'reason': 'Insufficient markets', 'n': n_markets}
    print(f"   PASS: {n_markets} >= {MIN_MARKETS}")

    # Calculate metrics
    win_rate = market_stats['is_winner'].mean()
    avg_price = market_stats['trade_price'].mean()
    breakeven = avg_price / 100.0
    edge = (win_rate - breakeven) * 100

    print(f"\n2. EDGE CALCULATION:")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Avg Trade Price: {avg_price:.0f}c")
    print(f"   Breakeven: {breakeven:.1%}")
    print(f"   Edge: {edge:+.1f}%")

    if edge <= 0:
        print("   FAIL: Negative or zero edge")
        return {'status': 'REJECTED', 'reason': 'No edge', 'edge': edge}
    print("   PASS: Positive edge")

    # Concentration check
    total_profit = market_stats['actual_profit_dollars'].sum()
    print(f"\n3. CONCENTRATION CHECK:")
    print(f"   Total Profit: ${total_profit:,.0f}")

    if total_profit > 0:
        profit_by_market = market_stats[market_stats['actual_profit_dollars'] > 0]['actual_profit_dollars']
        if len(profit_by_market) > 0:
            max_single = profit_by_market.max()
            concentration = max_single / total_profit
            top5 = profit_by_market.nlargest(5).sum() / total_profit

            print(f"   Max single market: ${max_single:,.0f} ({concentration:.1%})")
            print(f"   Top 5 markets: {top5:.1%}")

            if concentration > MAX_CONCENTRATION:
                print(f"   FAIL: Concentration {concentration:.1%} > {MAX_CONCENTRATION:.0%}")
                return {'status': 'REJECTED', 'reason': 'Concentration too high', 'concentration': concentration}
            print(f"   PASS: Concentration {concentration:.1%} < {MAX_CONCENTRATION:.0%}")
        else:
            concentration = 0
            print("   WARNING: No profitable markets")
    else:
        concentration = 0
        print("   WARNING: Non-positive total profit")

    # Statistical significance
    n_wins = int(win_rate * n_markets)
    result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
    p_value = result.pvalue

    print(f"\n4. STATISTICAL SIGNIFICANCE:")
    print(f"   P-value: {p_value:.6f}")
    print(f"   Alpha: {ALPHA}")
    print(f"   Bonferroni Alpha: {BONFERRONI_ALPHA:.6f}")

    if p_value >= ALPHA:
        print(f"   FAIL: p={p_value:.4f} >= {ALPHA}")
        return {'status': 'REJECTED', 'reason': 'Not significant', 'p_value': p_value}

    if p_value < BONFERRONI_ALPHA:
        print(f"   PASS: p={p_value:.6f} < {BONFERRONI_ALPHA:.6f} (Bonferroni)")
        significance_level = 'bonferroni'
    else:
        print(f"   MARGINAL: p={p_value:.4f} < {ALPHA} (nominal) but >= {BONFERRONI_ALPHA:.6f} (Bonferroni)")
        significance_level = 'nominal'

    # Temporal stability
    print(f"\n5. TEMPORAL STABILITY:")
    market_stats['date'] = market_stats['datetime'].dt.date
    dates = market_stats['date'].sort_values()

    if len(dates) > 0:
        min_date = dates.min()
        max_date = dates.max()
        mid_date = min_date + (max_date - min_date) / 2

        early_period = market_stats[market_stats['date'] <= mid_date]
        late_period = market_stats[market_stats['date'] > mid_date]

        results_by_period = []
        for period_name, period_df in [('Early', early_period), ('Late', late_period)]:
            if len(period_df) >= 20:
                wr = period_df['is_winner'].mean()
                ap = period_df['trade_price'].mean()
                be = ap / 100.0
                e = (wr - be) * 100
                results_by_period.append({
                    'period': period_name,
                    'n': len(period_df),
                    'win_rate': wr,
                    'edge': e
                })
                print(f"   {period_name}: N={len(period_df)}, WR={wr:.1%}, Edge={e:+.1f}%")

        # Check if edge is positive in both periods
        if len(results_by_period) == 2:
            early_edge = results_by_period[0]['edge']
            late_edge = results_by_period[1]['edge']

            if early_edge > 0 and late_edge > 0:
                print("   PASS: Positive edge in both periods")
                temporal_stable = True
            else:
                print("   WARNING: Edge not positive in all periods")
                temporal_stable = False
        else:
            print("   WARNING: Not enough data for temporal analysis")
            temporal_stable = None
    else:
        temporal_stable = None

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT:")

    if edge > 3 and p_value < BONFERRONI_ALPHA and (concentration < MAX_CONCENTRATION or total_profit <= 0):
        status = 'VALIDATED'
        print(f"   STATUS: **VALIDATED** - Robust edge detected")
    elif edge > 2 and p_value < ALPHA:
        status = 'MARGINAL'
        print(f"   STATUS: **MARGINAL** - Edge exists but not Bonferroni robust")
    else:
        status = 'WEAK'
        print(f"   STATUS: **WEAK** - Insufficient evidence")

    return {
        'status': status,
        'strategy': strategy_name,
        'description': description,
        'n_markets': n_markets,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge_pct': edge,
        'total_profit': total_profit,
        'concentration': concentration if total_profit > 0 else None,
        'p_value': p_value,
        'bonferroni_significant': p_value < BONFERRONI_ALPHA,
        'temporal_stable': temporal_stable
    }


def deep_validate_h065(df):
    """
    H065: Leverage Ratio as Fear Signal
    Deep validation of "Fade High-Leverage YES" strategy
    """
    print("\n" + "#"*80)
    print("# H065: LEVERAGE RATIO - DEEP VALIDATION")
    print("#"*80)

    results = []

    # The signal: When someone bets YES with high leverage (big potential return),
    # they are betting on a longshot. We fade by betting NO.

    for threshold in [2, 3, 5]:
        # Find high-leverage YES trades
        high_lev_yes = df[(df['leverage_ratio'] > threshold) & (df['taker_side'] == 'yes')]

        # Get unique markets where these trades occurred
        target_markets = high_lev_yes['market_ticker'].unique()

        # For each market, what was the outcome?
        # We would bet NO in these markets
        market_outcomes = df[df['market_ticker'].isin(target_markets)].groupby('market_ticker').agg({
            'market_result': 'first',
            'yes_price': 'mean',  # Average YES price in market
            'no_price': 'mean',   # Average NO price in market
        }).reset_index()

        # Our strategy: Bet NO at the prevailing NO price when high-lev YES occurs
        # We win if market_result == 'no'
        market_outcomes['we_win'] = market_outcomes['market_result'] == 'no'

        # Calculate our cost and profit
        market_outcomes['our_cost'] = market_outcomes['no_price']  # We pay NO price
        market_outcomes['our_profit'] = np.where(
            market_outcomes['we_win'],
            100 - market_outcomes['our_cost'],  # Win: get $1, paid cost
            -market_outcomes['our_cost']  # Lose: lose cost
        )

        # Create synthetic trade dataframe for validation
        synthetic_trades = market_outcomes.copy()
        synthetic_trades['trade_price'] = synthetic_trades['our_cost']
        synthetic_trades['is_winner'] = synthetic_trades['we_win']
        synthetic_trades['actual_profit_dollars'] = synthetic_trades['our_profit']
        synthetic_trades['cost_dollars'] = synthetic_trades['our_cost']
        synthetic_trades['datetime'] = df.groupby('market_ticker')['datetime'].first().reindex(synthetic_trades['market_ticker']).values

        result = rigorous_validation(
            df,
            synthetic_trades,
            f'fade_high_leverage_yes_gt{threshold}',
            f'Bet NO when retail bets YES with leverage > {threshold}'
        )
        if result:
            result['threshold'] = threshold
            results.append(result)

    return results


def deep_validate_h052(df):
    """
    H052: Order Flow Imbalance Rate-of-Change
    Deep validation of "Follow imbalance shift" strategy
    """
    print("\n" + "#"*80)
    print("# H052: ORDER FLOW ROC - DEEP VALIDATION")
    print("#"*80)

    # Get markets with enough trades
    market_trade_counts = df.groupby('market_ticker').size()
    active_markets = market_trade_counts[market_trade_counts >= 10].index.tolist()

    df_active = df[df['market_ticker'].isin(active_markets)].copy()
    df_active = df_active.sort_values(['market_ticker', 'datetime'])

    # Calculate flow metrics per market
    def calc_flow_metrics(group):
        group = group.sort_values('datetime')

        yes_flow = (group['taker_side'] == 'yes').astype(int) * group['cost_dollars']
        no_flow = (group['taker_side'] == 'no').astype(int) * group['cost_dollars']

        cum_yes = yes_flow.cumsum()
        cum_no = no_flow.cumsum()
        total = cum_yes + cum_no

        imbalance = (cum_yes - cum_no) / total.replace(0, 1)

        n = len(group)
        early_imb = imbalance.iloc[:max(1, n//3)].mean() if n > 0 else 0
        late_imb = imbalance.iloc[-max(1, n//3):].mean() if n > 0 else 0

        return pd.Series({
            'imbalance_shift': late_imb - early_imb,
            'market_result': group['market_result'].iloc[0],
            'avg_yes_price': group['yes_price'].mean(),
            'avg_no_price': group['no_price'].mean(),
            'first_datetime': group['datetime'].iloc[0]
        })

    print("Calculating flow metrics...")
    flow_metrics = df_active.groupby('market_ticker', group_keys=False).apply(calc_flow_metrics)
    flow_metrics = flow_metrics.reset_index()

    results = []

    for direction in ['yes', 'no']:
        for threshold in [0.3, 0.4, 0.5]:
            if direction == 'yes':
                signal_markets = flow_metrics[flow_metrics['imbalance_shift'] > threshold]
            else:
                signal_markets = flow_metrics[flow_metrics['imbalance_shift'] < -threshold]

            if len(signal_markets) < MIN_MARKETS:
                continue

            # We bet the direction indicated by the flow shift
            signal_markets['we_win'] = signal_markets['market_result'] == direction

            if direction == 'yes':
                signal_markets['our_cost'] = signal_markets['avg_yes_price']
            else:
                signal_markets['our_cost'] = signal_markets['avg_no_price']

            signal_markets['our_profit'] = np.where(
                signal_markets['we_win'],
                100 - signal_markets['our_cost'],
                -signal_markets['our_cost']
            )

            # Create synthetic trades for validation
            synthetic = signal_markets.copy()
            synthetic['trade_price'] = synthetic['our_cost']
            synthetic['is_winner'] = synthetic['we_win']
            synthetic['actual_profit_dollars'] = synthetic['our_profit']
            synthetic['cost_dollars'] = synthetic['our_cost']
            synthetic['datetime'] = synthetic['first_datetime']

            result = rigorous_validation(
                df,
                synthetic,
                f'follow_flow_shift_{direction}_gt{threshold}',
                f'Bet {direction.upper()} when flow shifts > {threshold} toward {direction}'
            )
            if result:
                result['direction'] = direction
                result['threshold'] = threshold
                results.append(result)

    return results


def compare_to_baseline(df, strategy_results):
    """
    Compare strategy edge to baseline (random betting at same prices)
    """
    print("\n" + "="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)

    # Baseline: What edge does a random NO strategy have at different prices?
    for price_range in [(80, 95), (70, 90), (60, 85)]:
        low, high = price_range
        baseline = df[(df['taker_side'] == 'no') &
                     (df['trade_price'] >= low) &
                     (df['trade_price'] < high)]

        if len(baseline) == 0:
            continue

        market_stats = baseline.groupby('market_ticker').agg({
            'trade_price': 'mean',
            'is_winner': 'first'
        }).reset_index()

        wr = market_stats['is_winner'].mean()
        be = market_stats['trade_price'].mean() / 100
        edge = (wr - be) * 100

        print(f"Baseline NO {low}-{high}c: N={len(market_stats)}, WR={wr:.1%}, BE={be:.1%}, Edge={edge:+.1f}%")


def main():
    print("="*80)
    print("SESSION 008: DEEP VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    df = load_data()

    all_results = {
        'H065_leverage': [],
        'H052_flow_roc': []
    }

    # H065: Leverage
    leverage_results = deep_validate_h065(df)
    all_results['H065_leverage'] = leverage_results

    # H052: Flow ROC
    flow_results = deep_validate_h052(df)
    all_results['H052_flow_roc'] = flow_results

    # Compare to baseline
    compare_to_baseline(df, all_results)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    validated = []
    marginal = []

    for category, results in all_results.items():
        for r in results:
            if r['status'] == 'VALIDATED':
                validated.append(r)
            elif r['status'] == 'MARGINAL':
                marginal.append(r)

    print(f"\nVALIDATED STRATEGIES: {len(validated)}")
    for r in validated:
        print(f"  - {r['strategy']}: Edge={r['edge_pct']:+.1f}%, N={r['n_markets']}, p={r['p_value']:.6f}")

    print(f"\nMARGINAL STRATEGIES: {len(marginal)}")
    for r in marginal:
        print(f"  - {r['strategy']}: Edge={r['edge_pct']:+.1f}%, N={r['n_markets']}, p={r['p_value']:.4f}")

    # Save results
    output_file = OUTPUT_PATH / f"session008_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    with open(output_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == '__main__':
    main()
