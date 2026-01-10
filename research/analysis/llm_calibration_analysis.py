#!/usr/bin/env python3
"""
LLM Calibration Analysis - Measure prediction accuracy of agentic research system.

This script:
1. Queries research_decisions for LLM probability estimates
2. Fetches settlement outcomes for each market
3. Calculates calibration metrics: Brier score, reliability diagram, per-confidence accuracy

Usage:
    cd backend && ENVIRONMENT=paper uv run python ../research/analysis/llm_calibration_analysis.py

Output:
    Calibration report with:
    - Probability buckets with actual win rates
    - Brier score (lower is better, <0.25 is decent)
    - Per-confidence-level accuracy
    - Detailed per-decision breakdown
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor

# Add backend/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'backend' / 'src'))

# Must set ENVIRONMENT=paper before importing config
os.environ.setdefault('ENVIRONMENT', 'paper')

from kalshiflow_rl.traderv3.clients.demo_client import KalshiDemoTradingClient

# Database connection
LOCAL_DATABASE_URL = 'postgresql://postgres:postgres@localhost:54322/postgres'

OUTPUT_PATH = Path(__file__).parent.parent / 'reports' / 'llm_calibration_analysis.json'


async def get_market_result(client: KalshiDemoTradingClient, ticker: str) -> Optional[str]:
    """
    Get market result from Kalshi API.

    Returns: 'yes', 'no', 'void', or None if not settled
    """
    try:
        market = await client.get_market(ticker)
        market_data = market.get('market', {})

        status = market_data.get('status', '')
        result = market_data.get('result', '')

        if status == 'finalized' and result:
            return result.lower()  # 'yes', 'no', or 'void'

        return None
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def get_research_decisions(conn) -> List[Dict[str, Any]]:
    """
    Fetch all research decisions with probability estimates.
    """
    query = """
    SELECT
        id,
        created_at,
        market_ticker,
        event_ticker,
        action,
        ai_probability,
        market_probability,
        edge,
        confidence,
        recommendation,
        price_guess_cents,
        price_guess_error_cents,
        evidence_quality,
        traded
    FROM research_decisions
    WHERE ai_probability IS NOT NULL
    ORDER BY created_at DESC
    """

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        return list(cur.fetchall())


def calculate_calibration_metrics(
    decisions: List[Dict[str, Any]],
    results: Dict[str, str]
) -> Dict[str, Any]:
    """
    Calculate calibration metrics given decisions and their outcomes.

    Args:
        decisions: List of research decisions with ai_probability
        results: Dict mapping market_ticker -> 'yes' | 'no' | 'void' | None

    Returns:
        Dict with calibration metrics
    """
    # Filter to settled markets (exclude void and unsettled)
    settled_decisions = []
    for d in decisions:
        ticker = d['market_ticker']
        result = results.get(ticker)
        if result in ['yes', 'no']:
            settled_decisions.append({
                **d,
                'actual_outcome': 1 if result == 'yes' else 0
            })

    if not settled_decisions:
        return {
            'error': 'No settled decisions found',
            'total_decisions': len(decisions),
            'settled_decisions': 0
        }

    # === Brier Score ===
    # Brier = mean((probability - outcome)^2)
    # Lower is better, 0.25 is random guessing
    brier_sum = 0.0
    for d in settled_decisions:
        prob = float(d['ai_probability'])
        outcome = d['actual_outcome']
        brier_sum += (prob - outcome) ** 2
    brier_score = brier_sum / len(settled_decisions)

    # === Probability Bucket Calibration ===
    # Group by probability bucket, compute actual win rate
    buckets = defaultdict(lambda: {'count': 0, 'sum_prob': 0.0, 'wins': 0})
    bucket_size = 0.20  # 5 buckets: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%

    for d in settled_decisions:
        prob = float(d['ai_probability'])
        bucket_idx = min(int(prob / bucket_size), 4)  # 0-4
        bucket_name = f"{int(bucket_idx * 20)}-{int((bucket_idx + 1) * 20)}%"

        buckets[bucket_name]['count'] += 1
        buckets[bucket_name]['sum_prob'] += prob
        buckets[bucket_name]['wins'] += d['actual_outcome']

    calibration_by_bucket = {}
    for bucket_name, data in sorted(buckets.items()):
        avg_prob = data['sum_prob'] / data['count'] if data['count'] > 0 else 0
        actual_rate = data['wins'] / data['count'] if data['count'] > 0 else 0
        gap = actual_rate - avg_prob

        calibration_by_bucket[bucket_name] = {
            'count': data['count'],
            'avg_predicted': round(avg_prob, 3),
            'actual_win_rate': round(actual_rate, 3),
            'calibration_gap': round(gap, 3),
            'calibration_quality': 'good' if abs(gap) < 0.10 else ('overconfident' if gap < 0 else 'underconfident')
        }

    # === Per-Confidence Accuracy ===
    # Group by confidence level, compute accuracy
    by_confidence = defaultdict(lambda: {'count': 0, 'correct': 0})

    for d in settled_decisions:
        conf = d['confidence'] or 'unknown'
        predicted_yes = float(d['ai_probability']) > 0.5
        actual_yes = d['actual_outcome'] == 1
        correct = predicted_yes == actual_yes

        by_confidence[conf]['count'] += 1
        by_confidence[conf]['correct'] += 1 if correct else 0

    confidence_accuracy = {}
    for conf, data in by_confidence.items():
        accuracy = data['correct'] / data['count'] if data['count'] > 0 else 0
        confidence_accuracy[conf] = {
            'count': data['count'],
            'accuracy': round(accuracy, 3),
            'correct': data['correct']
        }

    # === Per-Recommendation Accuracy ===
    by_recommendation = defaultdict(lambda: {'count': 0, 'profitable': 0})

    for d in settled_decisions:
        rec = d['recommendation'] or 'unknown'
        actual_yes = d['actual_outcome'] == 1

        # Profitable means: BUY_YES and outcome is YES, or BUY_NO and outcome is NO
        profitable = False
        if rec == 'BUY_YES' and actual_yes:
            profitable = True
        elif rec == 'BUY_NO' and not actual_yes:
            profitable = True

        by_recommendation[rec]['count'] += 1
        by_recommendation[rec]['profitable'] += 1 if profitable else 0

    recommendation_accuracy = {}
    for rec, data in by_recommendation.items():
        accuracy = data['profitable'] / data['count'] if data['count'] > 0 else 0
        recommendation_accuracy[rec] = {
            'count': data['count'],
            'win_rate': round(accuracy, 3),
            'profitable': data['profitable']
        }

    # === Price Guess Calibration ===
    # How well does the LLM guess the market price?
    price_guesses = [d for d in settled_decisions if d.get('price_guess_cents') is not None]
    if price_guesses:
        total_error = sum(abs(d['price_guess_error_cents'] or 0) for d in price_guesses)
        mean_abs_error = total_error / len(price_guesses)
    else:
        mean_abs_error = None

    # === Calibration Slope ===
    # Linear regression: actual = slope * predicted + intercept
    # Perfect calibration: slope = 1, intercept = 0
    if len(settled_decisions) >= 5:
        probs = [float(d['ai_probability']) for d in settled_decisions]
        outcomes = [d['actual_outcome'] for d in settled_decisions]

        mean_prob = sum(probs) / len(probs)
        mean_outcome = sum(outcomes) / len(outcomes)

        numerator = sum((p - mean_prob) * (o - mean_outcome) for p, o in zip(probs, outcomes))
        denominator = sum((p - mean_prob) ** 2 for p in probs)

        slope = numerator / denominator if denominator > 0 else 0
        intercept = mean_outcome - slope * mean_prob
    else:
        slope = None
        intercept = None

    return {
        'sample_size': len(settled_decisions),
        'total_decisions': len(decisions),
        'unsettled_or_void': len(decisions) - len(settled_decisions),

        'brier_score': round(brier_score, 4),
        'brier_interpretation': (
            'excellent' if brier_score < 0.15 else
            'good' if brier_score < 0.20 else
            'decent' if brier_score < 0.25 else
            'poor' if brier_score < 0.30 else
            'bad'
        ),

        'calibration_by_bucket': calibration_by_bucket,
        'confidence_accuracy': confidence_accuracy,
        'recommendation_accuracy': recommendation_accuracy,

        'calibration_slope': round(slope, 3) if slope is not None else None,
        'calibration_intercept': round(intercept, 3) if intercept is not None else None,
        'slope_interpretation': (
            'well-calibrated' if slope and 0.8 <= slope <= 1.2 else
            'overconfident (spread too wide)' if slope and slope < 0.8 else
            'underconfident (spread too narrow)' if slope and slope > 1.2 else
            'insufficient data'
        ),

        'price_guess_mean_abs_error_cents': round(mean_abs_error, 1) if mean_abs_error else None,
    }


async def main():
    print("=" * 60)
    print("LLM CALIBRATION ANALYSIS")
    print("=" * 60)
    print()

    # Connect to database
    print("[1] Connecting to database...")
    try:
        conn = psycopg2.connect(LOCAL_DATABASE_URL)
        print("    Connected to local Supabase")
    except Exception as e:
        print(f"    ERROR: Could not connect to database: {e}")
        print("    Make sure Supabase is running: cd backend && supabase start")
        return

    # Fetch research decisions
    print("[2] Fetching research decisions...")
    decisions = get_research_decisions(conn)
    print(f"    Found {len(decisions)} decisions with probability estimates")

    if not decisions:
        print("    No decisions found. Run the agentic research system first.")
        conn.close()
        return

    # Get unique market tickers
    tickers = list(set(d['market_ticker'] for d in decisions))
    print(f"    Unique markets: {len(tickers)}")

    # Initialize Kalshi client
    print("[3] Initializing Kalshi client...")
    try:
        client = KalshiDemoTradingClient()
        await client.async_init()
        print("    Client initialized")
    except Exception as e:
        print(f"    ERROR: Could not initialize client: {e}")
        conn.close()
        return

    # Fetch settlement results
    print("[4] Fetching settlement results from Kalshi API...")
    results = {}
    settled_count = 0
    unsettled_count = 0

    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(tickers)}")

        result = await get_market_result(client, ticker)
        results[ticker] = result

        if result in ['yes', 'no']:
            settled_count += 1
        else:
            unsettled_count += 1

    print(f"    Settled: {settled_count}, Unsettled/Void: {unsettled_count}")

    # Calculate calibration metrics
    print("[5] Calculating calibration metrics...")
    metrics = calculate_calibration_metrics(decisions, results)

    # Print report
    print()
    print("=" * 60)
    print("CALIBRATION REPORT")
    print("=" * 60)
    print()

    print(f"Sample Size: {metrics['sample_size']} settled decisions")
    print(f"Total Decisions: {metrics['total_decisions']}")
    print(f"Unsettled/Void: {metrics['unsettled_or_void']}")
    print()

    if 'error' in metrics:
        print(f"ERROR: {metrics['error']}")
        conn.close()
        return

    print(f"Brier Score: {metrics['brier_score']} ({metrics['brier_interpretation']})")
    print(f"    (lower is better, <0.25 is decent, <0.20 is good)")
    print()

    if metrics['calibration_slope']:
        print(f"Calibration Slope: {metrics['calibration_slope']} ({metrics['slope_interpretation']})")
        print(f"    (1.0 is perfect, <0.8 = overconfident, >1.2 = underconfident)")
        print()

    print("PROBABILITY BUCKET CALIBRATION:")
    print("-" * 60)
    print(f"{'Bucket':<12} | {'Count':>6} | {'Predicted':>10} | {'Actual':>8} | {'Gap':>8} | {'Quality':<15}")
    print("-" * 60)
    for bucket, data in sorted(metrics['calibration_by_bucket'].items()):
        print(f"{bucket:<12} | {data['count']:>6} | {data['avg_predicted']:>10.1%} | {data['actual_win_rate']:>8.1%} | {data['calibration_gap']:>+8.1%} | {data['calibration_quality']:<15}")
    print()

    print("CONFIDENCE LEVEL ACCURACY:")
    print("-" * 40)
    for conf, data in sorted(metrics['confidence_accuracy'].items()):
        print(f"  {conf.upper()}: {data['accuracy']:.1%} accurate (n={data['count']})")
    print()

    print("RECOMMENDATION WIN RATES:")
    print("-" * 40)
    for rec, data in sorted(metrics['recommendation_accuracy'].items()):
        print(f"  {rec}: {data['win_rate']:.1%} win rate (n={data['count']})")
    print()

    if metrics['price_guess_mean_abs_error_cents']:
        print(f"Price Guess MAE: {metrics['price_guess_mean_abs_error_cents']:.1f} cents")
        print()

    # Save to file
    output = {
        'generated_at': datetime.now().isoformat(),
        'metrics': metrics,
        'per_decision': [
            {
                'market_ticker': d['market_ticker'],
                'ai_probability': float(d['ai_probability']) if d['ai_probability'] else None,
                'market_probability': float(d['market_probability']) if d['market_probability'] else None,
                'confidence': d['confidence'],
                'recommendation': d['recommendation'],
                'result': results.get(d['market_ticker']),
                'correct': (
                    (float(d['ai_probability']) > 0.5 and results.get(d['market_ticker']) == 'yes') or
                    (float(d['ai_probability']) <= 0.5 and results.get(d['market_ticker']) == 'no')
                ) if d['ai_probability'] and results.get(d['market_ticker']) in ['yes', 'no'] else None
            }
            for d in decisions
        ]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Report saved to: {OUTPUT_PATH}")

    # Cleanup
    conn.close()
    print()
    print("Done!")


if __name__ == '__main__':
    asyncio.run(main())
