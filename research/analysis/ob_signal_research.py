"""
Orderbook Signal Research for RLM_NO Strategy Enhancement

Investigates whether orderbook signals can improve RLM_NO strategy's profitability.

Current Baseline:
- Edge: +17.38%
- Win Rate: 90.2% (1,791 wins / 1,986 markets)

Research Questions:
1. Can we filter out losing trades using orderbook signals?
2. Is there correlation between orderbook state and trade outcomes?
3. Can spread, imbalance, or depth signals improve win rate?

Author: Quant Research Agent
Date: 2025-01-05
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

import asyncpg
from dotenv import load_dotenv

# Load paper environment for database connection
load_dotenv(Path(__file__).parent.parent.parent / "backend" / ".env.paper")


async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    return await asyncpg.create_pool(database_url, min_size=1, max_size=5)


class OrderbookSignalResearch:
    """Research class for analyzing orderbook signals impact on RLM_NO strategy."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.report = {
            "research_id": "ob_signal_research_20250105",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phases": {}
        }

    async def phase0_data_availability(self) -> Dict[str, Any]:
        """
        Phase 0: Validate data availability.

        Checks:
        1. Total order_contexts for RLM_NO strategy
        2. Settled trades count
        3. Orderbook signals coverage
        4. JOIN feasibility between tables
        """
        print("\n" + "="*60)
        print("PHASE 0: Data Availability Validation")
        print("="*60)

        results = {
            "phase": "0_data_availability",
            "status": "unknown",
            "checks": {},
            "decision": None
        }

        async with self.pool.acquire() as conn:
            # Check 1: Order contexts for RLM_NO
            print("\n1. Checking order_contexts table for RLM_NO strategy...")

            rlm_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN settled_at IS NOT NULL THEN 1 END) as settled_trades,
                    COUNT(CASE WHEN market_result = 'yes' THEN 1 END) as losses,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    COUNT(DISTINCT market_ticker) as unique_markets,
                    MIN(filled_at) as first_trade,
                    MAX(filled_at) as last_trade
                FROM order_contexts
                WHERE strategy = 'rlm_no'
            """)

            results["checks"]["order_contexts"] = {
                "total_trades": rlm_stats["total_trades"],
                "settled_trades": rlm_stats["settled_trades"],
                "wins": rlm_stats["wins"],
                "losses": rlm_stats["losses"],
                "unique_markets": rlm_stats["unique_markets"],
                "first_trade": str(rlm_stats["first_trade"]) if rlm_stats["first_trade"] else None,
                "last_trade": str(rlm_stats["last_trade"]) if rlm_stats["last_trade"] else None,
            }

            if rlm_stats["settled_trades"] > 0:
                win_rate = rlm_stats["wins"] / rlm_stats["settled_trades"] * 100
                results["checks"]["order_contexts"]["win_rate_pct"] = round(win_rate, 2)

            print(f"   Total RLM_NO trades: {rlm_stats['total_trades']}")
            print(f"   Settled trades: {rlm_stats['settled_trades']}")
            print(f"   Wins: {rlm_stats['wins']}, Losses: {rlm_stats['losses']}")

            # Check 2: Orderbook signals coverage
            print("\n2. Checking orderbook_signals coverage...")

            ob_stats = await conn.fetch("""
                SELECT
                    session_id,
                    COUNT(DISTINCT market_ticker) as markets,
                    COUNT(*) as signal_buckets,
                    MIN(bucket_timestamp) as first_bucket,
                    MAX(bucket_timestamp) as last_bucket
                FROM orderbook_signals
                GROUP BY session_id
                ORDER BY session_id DESC
                LIMIT 10
            """)

            results["checks"]["orderbook_signals_by_session"] = [
                {
                    "session_id": row["session_id"],
                    "markets": row["markets"],
                    "signal_buckets": row["signal_buckets"],
                    "first_bucket": str(row["first_bucket"]),
                    "last_bucket": str(row["last_bucket"])
                }
                for row in ob_stats
            ]

            total_ob_signals = sum(row["signal_buckets"] for row in ob_stats)
            total_ob_markets = sum(row["markets"] for row in ob_stats)
            print(f"   Total orderbook signal buckets: {total_ob_signals}")
            print(f"   Total markets with signals: {total_ob_markets}")

            # Check 3: JOIN feasibility - exact timestamp match within 10-second window
            print("\n3. Testing JOIN between order_contexts and orderbook_signals...")

            # First, check session_id overlap
            session_overlap = await conn.fetchrow("""
                SELECT COUNT(DISTINCT oc.session_id) as overlapping_sessions
                FROM order_contexts oc
                INNER JOIN orderbook_signals os ON oc.session_id::text = os.session_id::text
                WHERE oc.strategy = 'rlm_no' AND oc.settled_at IS NOT NULL
            """)

            results["checks"]["session_overlap"] = session_overlap["overlapping_sessions"]
            print(f"   Sessions with overlap: {session_overlap['overlapping_sessions']}")

            # Try different join strategies
            # Strategy 1: Match by session + ticker + time window
            matched_trades = await conn.fetchrow("""
                SELECT COUNT(*) as matched_trades
                FROM order_contexts oc
                JOIN orderbook_signals os
                    ON oc.session_id::text = os.session_id::text
                    AND oc.market_ticker = os.market_ticker
                    AND os.bucket_timestamp <= oc.filled_at
                    AND os.bucket_timestamp > oc.filled_at - INTERVAL '10 seconds'
                WHERE oc.strategy = 'rlm_no' AND oc.settled_at IS NOT NULL
            """)

            results["checks"]["join_method_1_exact_window"] = matched_trades["matched_trades"]
            print(f"   Method 1 (exact 10s window): {matched_trades['matched_trades']} matched trades")

            # Strategy 2: Match by session + ticker + nearest bucket
            matched_trades_nearest = await conn.fetchrow("""
                WITH ranked_signals AS (
                    SELECT
                        os.*,
                        oc.order_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY oc.order_id
                            ORDER BY ABS(EXTRACT(EPOCH FROM (oc.filled_at - os.bucket_timestamp)))
                        ) as rn
                    FROM order_contexts oc
                    JOIN orderbook_signals os
                        ON oc.session_id::text = os.session_id::text
                        AND oc.market_ticker = os.market_ticker
                        AND os.bucket_timestamp BETWEEN oc.filled_at - INTERVAL '60 seconds'
                                                    AND oc.filled_at + INTERVAL '10 seconds'
                    WHERE oc.strategy = 'rlm_no' AND oc.settled_at IS NOT NULL
                )
                SELECT COUNT(*) as matched_trades
                FROM ranked_signals WHERE rn = 1
            """)

            results["checks"]["join_method_2_nearest_60s"] = matched_trades_nearest["matched_trades"]
            print(f"   Method 2 (nearest within 60s): {matched_trades_nearest['matched_trades']} matched trades")

            # Strategy 3: Match by ticker only (ignore session_id)
            matched_by_ticker = await conn.fetchrow("""
                SELECT COUNT(DISTINCT oc.order_id) as matched_trades
                FROM order_contexts oc
                JOIN orderbook_signals os
                    ON oc.market_ticker = os.market_ticker
                    AND os.bucket_timestamp BETWEEN oc.filled_at - INTERVAL '60 seconds'
                                                AND oc.filled_at + INTERVAL '10 seconds'
                WHERE oc.strategy = 'rlm_no' AND oc.settled_at IS NOT NULL
            """)

            results["checks"]["join_method_3_ticker_only"] = matched_by_ticker["matched_trades"]
            print(f"   Method 3 (ticker only, 60s window): {matched_by_ticker['matched_trades']} matched trades")

            # Check 4: What's the session_id format mismatch?
            print("\n4. Investigating session_id format...")

            oc_session_samples = await conn.fetch("""
                SELECT DISTINCT session_id, filled_at::date as trade_date
                FROM order_contexts
                WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
                ORDER BY filled_at DESC
                LIMIT 5
            """)

            os_session_samples = await conn.fetch("""
                SELECT DISTINCT session_id, bucket_timestamp::date as signal_date
                FROM orderbook_signals
                ORDER BY bucket_timestamp DESC
                LIMIT 5
            """)

            results["checks"]["session_id_samples"] = {
                "order_contexts": [
                    {"session_id": row["session_id"], "date": str(row["trade_date"])}
                    for row in oc_session_samples
                ],
                "orderbook_signals": [
                    {"session_id": row["session_id"], "date": str(row["signal_date"])}
                    for row in os_session_samples
                ]
            }

            print("   Order contexts session_ids:", [r["session_id"] for r in oc_session_samples])
            print("   Orderbook signals session_ids:", [r["session_id"] for r in os_session_samples])

            # Decision gate
            best_match_count = max(
                matched_trades["matched_trades"],
                matched_trades_nearest["matched_trades"],
                matched_by_ticker["matched_trades"]
            )

            if best_match_count >= 10:
                results["status"] = "DATA_AVAILABLE"
                results["decision"] = f"Proceed to Phase 1 - {best_match_count} trades can be matched"
                results["best_join_method"] = (
                    "ticker_only_60s" if matched_by_ticker["matched_trades"] == best_match_count
                    else "nearest_60s" if matched_trades_nearest["matched_trades"] == best_match_count
                    else "exact_10s"
                )
            else:
                results["status"] = "INSUFFICIENT_DATA"
                results["decision"] = f"Only {best_match_count} matched trades - need alternative approach"

            print(f"\n   DECISION: {results['decision']}")

        self.report["phases"]["phase0"] = results
        return results

    async def phase0b_alternative_analysis(self) -> Dict[str, Any]:
        """
        Phase 0b: Alternative analysis using order_contexts BBO data.

        If orderbook_signals JOIN doesn't work, analyze using the BBO data
        already captured in order_contexts table.
        """
        print("\n" + "="*60)
        print("PHASE 0b: Alternative Analysis Using order_contexts BBO Data")
        print("="*60)

        results = {
            "phase": "0b_alternative_analysis",
            "status": "unknown",
            "checks": {}
        }

        async with self.pool.acquire() as conn:
            # Check what BBO data exists in order_contexts
            print("\n1. Analyzing BBO data availability in order_contexts...")

            bbo_coverage = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_settled,
                    COUNT(best_bid_cents) as has_bid,
                    COUNT(best_ask_cents) as has_ask,
                    COUNT(bid_ask_spread_cents) as has_spread,
                    COUNT(spread_tier) as has_spread_tier,
                    COUNT(bid_size_contracts) as has_bid_size,
                    COUNT(ask_size_contracts) as has_ask_size
                FROM order_contexts
                WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
            """)

            results["checks"]["bbo_coverage"] = {
                "total_settled": bbo_coverage["total_settled"],
                "has_bid": bbo_coverage["has_bid"],
                "has_ask": bbo_coverage["has_ask"],
                "has_spread": bbo_coverage["has_spread"],
                "has_spread_tier": bbo_coverage["has_spread_tier"],
                "has_bid_size": bbo_coverage["has_bid_size"],
                "has_ask_size": bbo_coverage["has_ask_size"]
            }

            print(f"   Total settled: {bbo_coverage['total_settled']}")
            print(f"   Has spread: {bbo_coverage['has_spread']} ({100*bbo_coverage['has_spread']/max(1,bbo_coverage['total_settled']):.1f}%)")
            print(f"   Has spread tier: {bbo_coverage['has_spread_tier']} ({100*bbo_coverage['has_spread_tier']/max(1,bbo_coverage['total_settled']):.1f}%)")

            # Win rate by spread tier
            print("\n2. Win rate analysis by spread tier...")

            spread_analysis = await conn.fetch("""
                SELECT
                    spread_tier,
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    COUNT(CASE WHEN market_result = 'yes' THEN 1 END) as losses,
                    ROUND(AVG(slippage_cents)::numeric, 2) as avg_slippage,
                    ROUND(AVG(bid_ask_spread_cents)::numeric, 2) as avg_spread_cents
                FROM order_contexts
                WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
                GROUP BY spread_tier
                ORDER BY total DESC
            """)

            results["checks"]["spread_tier_analysis"] = []
            print(f"\n   {'Spread Tier':<12} {'Total':>7} {'Wins':>7} {'Losses':>7} {'Win%':>8} {'AvgSlip':>8} {'AvgSprd':>8}")
            print("   " + "-"*65)

            for row in spread_analysis:
                tier = row["spread_tier"] or "NULL"
                total = row["total"]
                wins = row["wins"]
                losses = row["losses"]
                win_pct = 100 * wins / total if total > 0 else 0

                results["checks"]["spread_tier_analysis"].append({
                    "spread_tier": tier,
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct": round(win_pct, 2),
                    "avg_slippage_cents": float(row["avg_slippage"]) if row["avg_slippage"] else None,
                    "avg_spread_cents": float(row["avg_spread_cents"]) if row["avg_spread_cents"] else None
                })

                print(f"   {tier:<12} {total:>7} {wins:>7} {losses:>7} {win_pct:>7.1f}% {row['avg_slippage'] or '-':>8} {row['avg_spread_cents'] or '-':>8}")

            # Analyze losing trades characteristics
            print("\n3. Analyzing losing trade characteristics...")

            losing_profile = await conn.fetchrow("""
                SELECT
                    AVG(bid_ask_spread_cents) as avg_spread,
                    AVG(slippage_cents) as avg_slippage,
                    AVG(no_price_at_signal) as avg_no_price,
                    AVG(bid_size_contracts) as avg_bid_size,
                    AVG(ask_size_contracts) as avg_ask_size,
                    COUNT(*) as count
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND market_result = 'yes'  -- Losses (we bet NO, market resolved YES)
            """)

            winning_profile = await conn.fetchrow("""
                SELECT
                    AVG(bid_ask_spread_cents) as avg_spread,
                    AVG(slippage_cents) as avg_slippage,
                    AVG(no_price_at_signal) as avg_no_price,
                    AVG(bid_size_contracts) as avg_bid_size,
                    AVG(ask_size_contracts) as avg_ask_size,
                    COUNT(*) as count
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND market_result = 'no'  -- Wins (we bet NO, market resolved NO)
            """)

            results["checks"]["trade_profiles"] = {
                "losing_trades": {
                    "count": losing_profile["count"],
                    "avg_spread_cents": float(losing_profile["avg_spread"]) if losing_profile["avg_spread"] else None,
                    "avg_slippage_cents": float(losing_profile["avg_slippage"]) if losing_profile["avg_slippage"] else None,
                    "avg_no_price_at_signal": float(losing_profile["avg_no_price"]) if losing_profile["avg_no_price"] else None,
                    "avg_bid_size": float(losing_profile["avg_bid_size"]) if losing_profile["avg_bid_size"] else None,
                    "avg_ask_size": float(losing_profile["avg_ask_size"]) if losing_profile["avg_ask_size"] else None,
                },
                "winning_trades": {
                    "count": winning_profile["count"],
                    "avg_spread_cents": float(winning_profile["avg_spread"]) if winning_profile["avg_spread"] else None,
                    "avg_slippage_cents": float(winning_profile["avg_slippage"]) if winning_profile["avg_slippage"] else None,
                    "avg_no_price_at_signal": float(winning_profile["avg_no_price"]) if winning_profile["avg_no_price"] else None,
                    "avg_bid_size": float(winning_profile["avg_bid_size"]) if winning_profile["avg_bid_size"] else None,
                    "avg_ask_size": float(winning_profile["avg_ask_size"]) if winning_profile["avg_ask_size"] else None,
                }
            }

            print(f"\n   Losing trades ({losing_profile['count']} total):")
            print(f"     Avg spread: {losing_profile['avg_spread']:.1f}c" if losing_profile['avg_spread'] else "     Avg spread: N/A")
            print(f"     Avg slippage: {losing_profile['avg_slippage']:.1f}c" if losing_profile['avg_slippage'] else "     Avg slippage: N/A")
            print(f"     Avg NO price: {losing_profile['avg_no_price']:.1f}c" if losing_profile['avg_no_price'] else "     Avg NO price: N/A")

            print(f"\n   Winning trades ({winning_profile['count']} total):")
            print(f"     Avg spread: {winning_profile['avg_spread']:.1f}c" if winning_profile['avg_spread'] else "     Avg spread: N/A")
            print(f"     Avg slippage: {winning_profile['avg_slippage']:.1f}c" if winning_profile['avg_slippage'] else "     Avg slippage: N/A")
            print(f"     Avg NO price: {winning_profile['avg_no_price']:.1f}c" if winning_profile['avg_no_price'] else "     Avg NO price: N/A")

            # Check signal_params JSONB for additional signals
            print("\n4. Analyzing signal_params for additional signals...")

            signal_param_keys = await conn.fetch("""
                SELECT DISTINCT jsonb_object_keys(signal_params) as key
                FROM order_contexts
                WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
                LIMIT 20
            """)

            results["checks"]["available_signal_params"] = [row["key"] for row in signal_param_keys]
            print(f"   Available signal params: {results['checks']['available_signal_params']}")

            # Analyze yes_ratio distribution for wins vs losses
            print("\n5. Analyzing yes_ratio for wins vs losses...")

            yes_ratio_analysis = await conn.fetch("""
                SELECT
                    market_result,
                    COUNT(*) as count,
                    AVG((signal_params->>'yes_ratio')::numeric) as avg_yes_ratio,
                    MIN((signal_params->>'yes_ratio')::numeric) as min_yes_ratio,
                    MAX((signal_params->>'yes_ratio')::numeric) as max_yes_ratio,
                    AVG((signal_params->>'price_drop_cents')::numeric) as avg_price_drop
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND signal_params ? 'yes_ratio'
                GROUP BY market_result
            """)

            results["checks"]["yes_ratio_by_outcome"] = [
                {
                    "outcome": row["market_result"],
                    "count": row["count"],
                    "avg_yes_ratio": float(row["avg_yes_ratio"]) if row["avg_yes_ratio"] else None,
                    "min_yes_ratio": float(row["min_yes_ratio"]) if row["min_yes_ratio"] else None,
                    "max_yes_ratio": float(row["max_yes_ratio"]) if row["max_yes_ratio"] else None,
                    "avg_price_drop": float(row["avg_price_drop"]) if row["avg_price_drop"] else None
                }
                for row in yes_ratio_analysis
            ]

            for row in yes_ratio_analysis:
                print(f"\n   {row['market_result'].upper()} outcomes ({row['count']} trades):")
                print(f"     Avg yes_ratio: {row['avg_yes_ratio']:.3f}" if row['avg_yes_ratio'] else "     Avg yes_ratio: N/A")
                print(f"     Yes_ratio range: {row['min_yes_ratio']:.3f} - {row['max_yes_ratio']:.3f}" if row['min_yes_ratio'] else "     Yes_ratio range: N/A")
                print(f"     Avg price_drop: {row['avg_price_drop']:.1f}c" if row['avg_price_drop'] else "     Avg price_drop: N/A")

            results["status"] = "ANALYSIS_COMPLETE"

        self.report["phases"]["phase0b"] = results
        return results

    async def phase1_hypothesis_testing(self) -> Dict[str, Any]:
        """
        Phase 1: Test orderbook-based filter hypotheses.

        Hypotheses:
        OB-001: Skip when spread > 4c (wide spreads indicate uncertainty)
        OB-002: Skip when ask_size < 50 contracts (thin liquidity)
        OB-003: Higher yes_ratio threshold (>70% vs >65%)
        OB-004: Skip when slippage > 2c (bad execution)
        """
        print("\n" + "="*60)
        print("PHASE 1: Hypothesis Testing (Using order_contexts Data)")
        print("="*60)

        results = {
            "phase": "1_hypothesis_testing",
            "hypotheses": {},
            "baseline": {}
        }

        async with self.pool.acquire() as conn:
            # Establish baseline
            print("\n1. Establishing baseline performance...")

            baseline = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    COUNT(CASE WHEN market_result = 'yes' THEN 1 END) as losses,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
            """)

            baseline_win_rate = baseline["wins"] / baseline["total"] * 100 if baseline["total"] > 0 else 0
            baseline_edge = baseline_win_rate - (float(baseline["avg_no_price"]) if baseline["avg_no_price"] else 0)

            results["baseline"] = {
                "total_trades": baseline["total"],
                "wins": baseline["wins"],
                "losses": baseline["losses"],
                "win_rate_pct": round(baseline_win_rate, 2),
                "avg_no_price_pct": round(float(baseline["avg_no_price"]), 2) if baseline["avg_no_price"] else None,
                "edge_pct": round(baseline_edge, 2)
            }

            print(f"   Baseline: {baseline['total']} trades, {baseline_win_rate:.1f}% win rate, {baseline_edge:.1f}% edge")

            # OB-001: Skip wide spreads
            print("\n2. Testing OB-001: Skip when spread > 4c...")

            ob001 = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND (bid_ask_spread_cents IS NULL OR bid_ask_spread_cents <= 4)
            """)

            ob001_win_rate = ob001["wins"] / ob001["total"] * 100 if ob001["total"] > 0 else 0
            ob001_edge = ob001_win_rate - (float(ob001["avg_no_price"]) if ob001["avg_no_price"] else 0)
            ob001_retention = ob001["total"] / baseline["total"] * 100 if baseline["total"] > 0 else 0

            results["hypotheses"]["OB-001_skip_wide_spread"] = {
                "description": "Skip when spread > 4c",
                "total_trades": ob001["total"],
                "wins": ob001["wins"],
                "win_rate_pct": round(ob001_win_rate, 2),
                "edge_pct": round(ob001_edge, 2),
                "retention_pct": round(ob001_retention, 2),
                "edge_improvement_pct": round(ob001_edge - baseline_edge, 2)
            }

            print(f"   OB-001: {ob001['total']} trades ({ob001_retention:.1f}% retention), {ob001_win_rate:.1f}% win rate, {ob001_edge:.1f}% edge")
            print(f"   Edge improvement: {ob001_edge - baseline_edge:+.2f}%")

            # OB-002: Skip thin liquidity
            print("\n3. Testing OB-002: Skip when ask_size < 50 contracts...")

            ob002 = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND (ask_size_contracts IS NULL OR ask_size_contracts >= 50)
            """)

            ob002_win_rate = ob002["wins"] / ob002["total"] * 100 if ob002["total"] > 0 else 0
            ob002_edge = ob002_win_rate - (float(ob002["avg_no_price"]) if ob002["avg_no_price"] else 0)
            ob002_retention = ob002["total"] / baseline["total"] * 100 if baseline["total"] > 0 else 0

            results["hypotheses"]["OB-002_skip_thin_liquidity"] = {
                "description": "Skip when ask_size < 50 contracts",
                "total_trades": ob002["total"],
                "wins": ob002["wins"],
                "win_rate_pct": round(ob002_win_rate, 2),
                "edge_pct": round(ob002_edge, 2),
                "retention_pct": round(ob002_retention, 2),
                "edge_improvement_pct": round(ob002_edge - baseline_edge, 2)
            }

            print(f"   OB-002: {ob002['total']} trades ({ob002_retention:.1f}% retention), {ob002_win_rate:.1f}% win rate, {ob002_edge:.1f}% edge")
            print(f"   Edge improvement: {ob002_edge - baseline_edge:+.2f}%")

            # OB-003: Higher yes_ratio threshold
            print("\n4. Testing OB-003: Higher yes_ratio threshold (>70% vs >65%)...")

            ob003 = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND (signal_params->>'yes_ratio')::numeric >= 0.70
            """)

            ob003_win_rate = ob003["wins"] / ob003["total"] * 100 if ob003["total"] > 0 else 0
            ob003_edge = ob003_win_rate - (float(ob003["avg_no_price"]) if ob003["avg_no_price"] else 0)
            ob003_retention = ob003["total"] / baseline["total"] * 100 if baseline["total"] > 0 else 0

            results["hypotheses"]["OB-003_higher_yes_ratio"] = {
                "description": "Higher yes_ratio threshold >= 70%",
                "total_trades": ob003["total"],
                "wins": ob003["wins"],
                "win_rate_pct": round(ob003_win_rate, 2),
                "edge_pct": round(ob003_edge, 2),
                "retention_pct": round(ob003_retention, 2),
                "edge_improvement_pct": round(ob003_edge - baseline_edge, 2)
            }

            print(f"   OB-003: {ob003['total']} trades ({ob003_retention:.1f}% retention), {ob003_win_rate:.1f}% win rate, {ob003_edge:.1f}% edge")
            print(f"   Edge improvement: {ob003_edge - baseline_edge:+.2f}%")

            # OB-004: Skip high slippage
            print("\n5. Testing OB-004: Skip when slippage > 2c...")

            ob004 = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND (slippage_cents IS NULL OR slippage_cents <= 2)
            """)

            ob004_win_rate = ob004["wins"] / ob004["total"] * 100 if ob004["total"] > 0 else 0
            ob004_edge = ob004_win_rate - (float(ob004["avg_no_price"]) if ob004["avg_no_price"] else 0)
            ob004_retention = ob004["total"] / baseline["total"] * 100 if baseline["total"] > 0 else 0

            results["hypotheses"]["OB-004_skip_high_slippage"] = {
                "description": "Skip when slippage > 2c",
                "total_trades": ob004["total"],
                "wins": ob004["wins"],
                "win_rate_pct": round(ob004_win_rate, 2),
                "edge_pct": round(ob004_edge, 2),
                "retention_pct": round(ob004_retention, 2),
                "edge_improvement_pct": round(ob004_edge - baseline_edge, 2)
            }

            print(f"   OB-004: {ob004['total']} trades ({ob004_retention:.1f}% retention), {ob004_win_rate:.1f}% win rate, {ob004_edge:.1f}% edge")
            print(f"   Edge improvement: {ob004_edge - baseline_edge:+.2f}%")

            # Combined filters
            print("\n6. Testing Combined Filters (best performing)...")

            # Combine non-overlapping beneficial filters
            combined = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(no_price_at_signal) as avg_no_price
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND (bid_ask_spread_cents IS NULL OR bid_ask_spread_cents <= 4)
                  AND (slippage_cents IS NULL OR slippage_cents <= 2)
            """)

            combined_win_rate = combined["wins"] / combined["total"] * 100 if combined["total"] > 0 else 0
            combined_edge = combined_win_rate - (float(combined["avg_no_price"]) if combined["avg_no_price"] else 0)
            combined_retention = combined["total"] / baseline["total"] * 100 if baseline["total"] > 0 else 0

            results["hypotheses"]["COMBINED_spread_slippage"] = {
                "description": "Combined: spread <= 4c AND slippage <= 2c",
                "total_trades": combined["total"],
                "wins": combined["wins"],
                "win_rate_pct": round(combined_win_rate, 2),
                "edge_pct": round(combined_edge, 2),
                "retention_pct": round(combined_retention, 2),
                "edge_improvement_pct": round(combined_edge - baseline_edge, 2)
            }

            print(f"   Combined: {combined['total']} trades ({combined_retention:.1f}% retention), {combined_win_rate:.1f}% win rate, {combined_edge:.1f}% edge")
            print(f"   Edge improvement: {combined_edge - baseline_edge:+.2f}%")

        self.report["phases"]["phase1"] = results
        return results

    async def phase2_price_bucket_analysis(self) -> Dict[str, Any]:
        """
        Phase 2: Analyze performance by NO price bucket.

        This is critical for understanding if orderbook signals
        provide independent value vs just being price proxies.
        """
        print("\n" + "="*60)
        print("PHASE 2: Price Bucket Analysis")
        print("="*60)

        results = {
            "phase": "2_price_bucket_analysis",
            "buckets": []
        }

        async with self.pool.acquire() as conn:
            # Analyze by 5c price bucket
            print("\n1. Performance by NO price bucket (5c increments)...")

            bucket_analysis = await conn.fetch("""
                SELECT
                    bucket_5c,
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins,
                    AVG(bid_ask_spread_cents) as avg_spread,
                    AVG(slippage_cents) as avg_slippage
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND bucket_5c IS NOT NULL
                GROUP BY bucket_5c
                ORDER BY bucket_5c
            """)

            print(f"\n   {'Bucket':<8} {'Total':>7} {'Wins':>7} {'Win%':>8} {'Edge%':>8} {'AvgSprd':>8} {'AvgSlip':>8}")
            print("   " + "-"*60)

            for row in bucket_analysis:
                bucket = row["bucket_5c"]
                total = row["total"]
                wins = row["wins"]
                win_rate = wins / total * 100 if total > 0 else 0
                # Edge = win_rate - breakeven_price (bucket represents NO price, so breakeven is bucket/100)
                edge = win_rate - bucket

                bucket_data = {
                    "bucket_5c": bucket,
                    "total": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate_pct": round(win_rate, 2),
                    "edge_pct": round(edge, 2),
                    "avg_spread_cents": float(row["avg_spread"]) if row["avg_spread"] else None,
                    "avg_slippage_cents": float(row["avg_slippage"]) if row["avg_slippage"] else None
                }
                results["buckets"].append(bucket_data)

                print(f"   {bucket:>5}c  {total:>7} {wins:>7} {win_rate:>7.1f}% {edge:>+7.1f}% {row['avg_spread'] or '-':>8} {row['avg_slippage'] or '-':>8}")

            # Within-bucket analysis: spread impact
            print("\n2. Within-bucket spread analysis (bucket 25-35c where most trades)...")

            within_bucket = await conn.fetch("""
                SELECT
                    bucket_5c,
                    CASE
                        WHEN bid_ask_spread_cents IS NULL THEN 'unknown'
                        WHEN bid_ask_spread_cents <= 2 THEN 'tight'
                        WHEN bid_ask_spread_cents <= 4 THEN 'normal'
                        ELSE 'wide'
                    END as spread_group,
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_result = 'no' THEN 1 END) as wins
                FROM order_contexts
                WHERE strategy = 'rlm_no'
                  AND settled_at IS NOT NULL
                  AND bucket_5c BETWEEN 25 AND 35
                GROUP BY bucket_5c, spread_group
                ORDER BY bucket_5c, spread_group
            """)

            results["within_bucket_spread"] = []
            print(f"\n   {'Bucket':<8} {'Spread':<10} {'Total':>7} {'Wins':>7} {'Win%':>8}")
            print("   " + "-"*50)

            for row in within_bucket:
                win_rate = row["wins"] / row["total"] * 100 if row["total"] > 0 else 0
                results["within_bucket_spread"].append({
                    "bucket_5c": row["bucket_5c"],
                    "spread_group": row["spread_group"],
                    "total": row["total"],
                    "wins": row["wins"],
                    "win_rate_pct": round(win_rate, 2)
                })
                print(f"   {row['bucket_5c']:>5}c  {row['spread_group']:<10} {row['total']:>7} {row['wins']:>7} {win_rate:>7.1f}%")

        self.report["phases"]["phase2"] = results
        return results

    async def generate_report(self) -> str:
        """Generate final research report."""
        report_path = Path(__file__).parent.parent / "reports" / "ob_signal_research.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Add summary and recommendations
        self.report["summary"] = {
            "research_question": "Can orderbook signals improve RLM_NO strategy profitability?",
            "data_status": self.report["phases"].get("phase0", {}).get("status", "unknown"),
            "key_findings": [],
            "recommendations": []
        }

        # Extract key findings
        if "phase0b" in self.report["phases"]:
            p0b = self.report["phases"]["phase0b"]
            if "spread_tier_analysis" in p0b.get("checks", {}):
                for tier in p0b["checks"]["spread_tier_analysis"]:
                    if tier["total"] >= 10:
                        self.report["summary"]["key_findings"].append(
                            f"Spread tier '{tier['spread_tier']}': {tier['win_rate_pct']}% win rate ({tier['total']} trades)"
                        )

        if "phase1" in self.report["phases"]:
            p1 = self.report["phases"]["phase1"]
            for hyp_id, hyp_data in p1.get("hypotheses", {}).items():
                if hyp_data.get("edge_improvement_pct", 0) > 1.0:
                    self.report["summary"]["key_findings"].append(
                        f"{hyp_id}: +{hyp_data['edge_improvement_pct']:.1f}% edge improvement ({hyp_data['retention_pct']:.0f}% retention)"
                    )

        # Generate recommendations
        if "phase1" in self.report["phases"]:
            best_hyp = max(
                self.report["phases"]["phase1"].get("hypotheses", {}).items(),
                key=lambda x: x[1].get("edge_improvement_pct", -999) if x[1].get("retention_pct", 0) >= 75 else -999,
                default=(None, None)
            )
            if best_hyp[0] and best_hyp[1].get("edge_improvement_pct", 0) > 0:
                self.report["summary"]["recommendations"].append(
                    f"Implement {best_hyp[0]} filter: {best_hyp[1]['description']} (+{best_hyp[1]['edge_improvement_pct']:.1f}% edge)"
                )

        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        print(f"\n\nReport saved to: {report_path}")
        return str(report_path)


async def main():
    """Run the orderbook signal research."""
    print("="*60)
    print("ORDERBOOK SIGNAL RESEARCH FOR RLM_NO STRATEGY")
    print("="*60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    pool = await get_db_pool()

    try:
        research = OrderbookSignalResearch(pool)

        # Phase 0: Data availability
        phase0_results = await research.phase0_data_availability()

        # Phase 0b: Alternative analysis using order_contexts BBO data
        await research.phase0b_alternative_analysis()

        # Phase 1: Hypothesis testing (always run with order_contexts data)
        await research.phase1_hypothesis_testing()

        # Phase 2: Price bucket analysis
        await research.phase2_price_bucket_analysis()

        # Generate report
        report_path = await research.generate_report()

        print("\n" + "="*60)
        print("RESEARCH COMPLETE")
        print("="*60)
        print(f"\nReport: {report_path}")
        print(f"\nSummary:")
        print(f"  Key Findings: {len(research.report['summary']['key_findings'])}")
        for finding in research.report['summary']['key_findings'][:5]:
            print(f"    - {finding}")
        print(f"\n  Recommendations: {len(research.report['summary']['recommendations'])}")
        for rec in research.report['summary']['recommendations'][:3]:
            print(f"    - {rec}")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
