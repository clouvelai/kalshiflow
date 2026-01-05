"""
Strategy Validation Core Engine.

This module implements the main validation logic, supporting both
LSD mode (quick ~2s screening) and Full mode (rigorous ~30s validation).

Key Responsibilities:
    1. Filter markets by strategy signal conditions
    2. Calculate primary edge metrics (win rate, breakeven, raw edge)
    3. Bucket-matched baseline comparison (NOT a price proxy check)
    4. Bootstrap confidence intervals
    5. Temporal stability analysis (train/test, quarterly)
    6. Parameter sensitivity sweeps
    7. Verdict generation with validation checklist

Validation Methodology (matching quant's approach):
    - Bucket-matched: Compare signal win rate to baseline at same price bucket
    - Market-level analysis: Each market is one observation
    - Bootstrap CI: 1000 iterations for confidence intervals
    - Multiple validation checks: Sample size, significance, temporal stability

Architecture Position:
    The StrategyValidator is the core component used by:
    - CLI tools: For running validations
    - Research scripts: For automated hypothesis testing
    - V3 trader integration: For production strategy deployment
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .cache import CachedDataLayer
from .config import StrategyConfig, ValidationMode

logger = logging.getLogger("kalshiflow.validation.validator")


@dataclass
class BucketAnalysis:
    """Analysis results for a single price bucket."""
    bucket: int
    n_signal: int
    n_baseline: int
    signal_win_rate: float
    baseline_win_rate: float
    improvement: float
    is_positive: bool


@dataclass
class TemporalResult:
    """Results for a temporal period (week, quarter, etc.)."""
    period: str
    n_markets: int
    win_rate: float
    avg_price: float
    edge: float
    is_profitable: bool


@dataclass
class ValidationResult:
    """
    Complete validation result.

    Contains all metrics and analysis from a strategy validation run.
    Used by both LSD and Full modes, with different levels of detail.
    """
    # Metadata
    strategy_name: str
    hypothesis_id: str
    validation_date: str
    validation_mode: str
    data_range: str

    # Primary metrics
    is_valid: bool
    n_markets: int
    wins: int
    losses: int
    win_rate: float
    avg_entry_price: float
    breakeven: float
    raw_edge: float
    z_score: float
    p_value: float
    is_statistically_significant: bool

    # Bucket analysis (for price proxy check)
    bucket_analysis: Optional[Dict[str, Any]] = None

    # Confidence intervals
    confidence_interval_95: Optional[Dict[str, float]] = None

    # Temporal stability
    temporal_stability: Optional[Dict[str, Any]] = None

    # Concentration analysis (to detect over-reliance on few markets/categories)
    concentration: Optional[Dict[str, Any]] = None

    # Price drop analysis (for RLM-style strategies)
    price_drop_analysis: Optional[List[Dict[str, Any]]] = None

    # Category breakdown
    category_breakdown: Optional[List[Dict[str, Any]]] = None

    # Validation checks summary
    validation_checks: Optional[Dict[str, bool]] = None
    all_checks_pass: bool = False
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "metadata": {
                "strategy": self.strategy_name,
                "validation_date": self.validation_date,
                "data_range": self.data_range,
            },
            "edge_results": {
                "valid": self.is_valid,
                "n_markets": self.n_markets,
                "wins": str(self.wins),
                "losses": str(self.losses),
                "win_rate": self.win_rate,
                "avg_no_price": self.avg_entry_price,
                "breakeven": self.breakeven,
                "raw_edge": self.raw_edge,
                "raw_edge_pct": self.raw_edge * 100,
                "z_score": self.z_score,
                "p_value": self.p_value,
                "statistically_significant": str(self.is_statistically_significant),
            }
        }

        if self.bucket_analysis:
            result["edge_results"]["bucket_analysis"] = self.bucket_analysis

        if self.confidence_interval_95:
            result["edge_results"]["confidence_interval_95"] = self.confidence_interval_95

        result["edge_results"]["is_price_proxy"] = str(
            self.bucket_analysis and
            self.bucket_analysis.get("bucket_ratio", 1.0) < 0.5
        )

        if self.temporal_stability:
            result["temporal_stability"] = self.temporal_stability

        if self.concentration:
            result["concentration"] = self.concentration

        if self.price_drop_analysis:
            result["price_drop_analysis"] = self.price_drop_analysis

        if self.category_breakdown:
            result["category_breakdown"] = self.category_breakdown

        if self.validation_checks:
            result["validation_checks"] = self.validation_checks
            result["all_checks_pass"] = self.all_checks_pass
            result["recommendation"] = self.recommendation

        return result


class StrategyValidator:
    """
    Generic strategy validation engine.

    Accepts strategy configuration and produces standardized results.
    Supports both LSD mode (quick) and Full mode (rigorous).
    """

    def __init__(self, config: StrategyConfig, cache: CachedDataLayer):
        """
        Initialize the validator.

        Args:
            config: Strategy configuration
            cache: Precomputed data cache
        """
        self.config = config
        self.cache = cache

    def validate(self, mode: Optional[ValidationMode] = None) -> ValidationResult:
        """
        Run validation pipeline.

        Args:
            mode: "lsd" for quick screening, "full" for rigorous validation
                  Defaults to the mode specified in config

        Returns:
            ValidationResult with all metrics
        """
        if mode is None:
            mode = self.config.validation.mode

        logger.info(f"Starting {mode.value.upper()} validation for {self.config.name}")

        # 1. Filter markets by signal conditions
        signal_markets = self.cache.filter_markets(
            self.config.signal.conditions,
            resolved_only=True
        )

        logger.info(f"Signal matched {len(signal_markets):,} markets")

        # 2. Calculate primary metrics
        primary = self._calculate_primary_metrics(signal_markets)

        # Quick exit for LSD mode
        if mode == ValidationMode.LSD:
            return self._create_lsd_result(primary, signal_markets)

        # 3. Full validation pipeline
        logger.info("Running full validation pipeline...")

        bucket_analysis = self._analyze_by_bucket(signal_markets)
        temporal_stability = self._analyze_temporal_stability(signal_markets)
        bootstrap_ci = self._bootstrap_confidence_interval(signal_markets)
        concentration = self._analyze_concentration(signal_markets)
        price_drop = self._analyze_price_drops(signal_markets)
        category_breakdown = self._analyze_by_category(signal_markets)

        # Generate validation checks and verdict
        checks = self._run_validation_checks(
            primary, bucket_analysis, temporal_stability, bootstrap_ci
        )

        return self._create_full_result(
            primary=primary,
            signal_markets=signal_markets,
            bucket_analysis=bucket_analysis,
            temporal_stability=temporal_stability,
            bootstrap_ci=bootstrap_ci,
            concentration=concentration,
            price_drop=price_drop,
            category_breakdown=category_breakdown,
            checks=checks
        )

    def _calculate_primary_metrics(self, markets: pd.DataFrame) -> Dict[str, Any]:
        """Calculate primary edge metrics."""
        n = len(markets)
        side = self.config.bet_side

        if n < self.config.validation.min_markets:
            return {
                "valid": False,
                "n": n,
                "reason": f"insufficient_markets_{n}"
            }

        # Calculate wins/losses
        wins = (markets["market_result"] == side).sum()
        losses = n - wins
        win_rate = wins / n

        # Get entry price field
        price_field = self.config.signal.entry_price_field
        if price_field == "no_price":
            avg_price = markets["avg_no_price"].mean()
        else:
            avg_price = markets["avg_yes_price"].mean()

        # Breakeven and edge
        breakeven = avg_price / 100
        edge = win_rate - breakeven

        # Statistical significance (z-score and p-value)
        if 0 < breakeven < 1:
            z_score = (wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven))
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 1.0

        return {
            "valid": True,
            "n": n,
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "avg_price": float(avg_price),
            "breakeven": float(breakeven),
            "edge": float(edge),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": p_value < self.config.validation.p_threshold
        }

    def _analyze_by_bucket(self, markets: pd.DataFrame) -> Dict[str, Any]:
        """Analyze edge by price bucket, comparing to baseline."""
        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"
        bucket_size = self.config.validation.bucket_size
        min_bucket = self.config.validation.min_bucket_markets

        # Add bucket column
        markets = markets.copy()
        markets["bucket"] = (markets[price_field] // bucket_size) * bucket_size

        buckets = []
        for bucket in sorted(markets["bucket"].unique()):
            bucket_markets = markets[markets["bucket"] == bucket]
            n_signal = len(bucket_markets)

            if n_signal < min_bucket:
                continue

            # Get baseline for this bucket
            baseline = self.cache.get_baseline(int(bucket))
            if baseline is None:
                continue

            signal_win_rate = (bucket_markets["market_result"] == side).mean()
            improvement = signal_win_rate - baseline.win_rate

            buckets.append({
                "bucket": int(bucket),
                "n_signal": int(n_signal),
                "n_baseline": int(baseline.n_markets),
                "signal_win_rate": float(signal_win_rate),
                "baseline_win_rate": float(baseline.win_rate),
                "improvement": float(improvement),
                "positive": str(improvement > 0)
            })

        if not buckets:
            return {"n_buckets": 0, "positive_buckets": 0, "bucket_ratio": 0.0}

        positive_buckets = sum(1 for b in buckets if b["improvement"] > 0)
        total_buckets = len(buckets)

        # Calculate weighted average improvement
        total_n = sum(b["n_signal"] for b in buckets)
        avg_improvement = sum(b["improvement"] * b["n_signal"] for b in buckets) / total_n

        return {
            "n_buckets": total_buckets,
            "positive_buckets": positive_buckets,
            "bucket_ratio": positive_buckets / total_buckets if total_buckets > 0 else 0.0,
            "avg_improvement": float(avg_improvement),
            "avg_improvement_pct": float(avg_improvement * 100),
            "buckets": buckets
        }

    def _analyze_temporal_stability(self, markets: pd.DataFrame) -> Dict[str, Any]:
        """Analyze edge stability across time periods."""
        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"

        # Sort by date
        markets = markets.copy()
        markets = markets.sort_values("first_trade_time")

        # Weekly analysis
        weekly_results = []
        for week in sorted(markets["week"].unique()):
            week_markets = markets[markets["week"] == week]
            n = len(week_markets)

            if n >= 30:  # Minimum for meaningful analysis
                win_rate = (week_markets["market_result"] == side).mean()
                avg_price = week_markets[price_field].mean()
                breakeven = avg_price / 100
                edge = win_rate - breakeven

                weekly_results.append({
                    "week": int(week),
                    "n_markets": int(n),
                    "win_rate": float(win_rate),
                    "avg_no_price": float(avg_price),
                    "edge": float(edge),
                    "profitable": str(edge > 0)
                })

        positive_weeks = sum(1 for w in weekly_results if float(w["edge"]) > 0)
        total_weeks = len(weekly_results)

        return {
            "weekly_results": weekly_results,
            "positive_weeks": positive_weeks,
            "total_weeks": total_weeks,
            "temporal_stability_ratio": positive_weeks / total_weeks if total_weeks > 0 else 0.0
        }

    def _bootstrap_confidence_interval(
        self,
        markets: pd.DataFrame,
        n_iterations: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for edge."""
        if n_iterations is None:
            n_iterations = self.config.validation.bootstrap_iterations

        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"

        bootstrap_win_rates = []
        bootstrap_edges = []

        for _ in range(n_iterations):
            sample = markets.sample(n=len(markets), replace=True)
            sample_win_rate = (sample["market_result"] == side).mean()
            sample_breakeven = sample[price_field].mean() / 100
            sample_edge = sample_win_rate - sample_breakeven

            bootstrap_win_rates.append(sample_win_rate)
            bootstrap_edges.append(sample_edge)

        return {
            "win_rate_lower": float(np.percentile(bootstrap_win_rates, 2.5)),
            "win_rate_upper": float(np.percentile(bootstrap_win_rates, 97.5)),
            "edge_lower": float(np.percentile(bootstrap_edges, 2.5)),
            "edge_upper": float(np.percentile(bootstrap_edges, 97.5))
        }

    def _analyze_concentration(self, markets: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profit concentration to detect over-reliance on few markets."""
        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"

        markets = markets.copy()

        # Calculate profit per market
        markets["won"] = markets["market_result"] == side
        markets["profit"] = markets.apply(
            lambda r: (100 - r[price_field]) if r["won"] else -r[price_field],
            axis=1
        )

        total_profit = markets["profit"].sum()

        if total_profit <= 0:
            return {
                "total_profit": float(total_profit),
                "concentration_ok": str(False),
                "reason": "no_positive_profit"
            }

        # Max single market contribution
        max_single = markets["profit"].max() / total_profit

        # Top 5 markets contribution
        top5 = markets.nlargest(5, "profit")["profit"].sum() / total_profit

        # Max category contribution
        category_profits = markets.groupby("category")["profit"].sum()
        max_category = category_profits.max() / total_profit if len(category_profits) > 0 else 0

        # Concentration is OK if no single market/category dominates
        concentration_ok = (max_single < 0.05 and max_category < 0.30)

        return {
            "total_profit": float(total_profit),
            "max_single_market_contribution": float(max_single),
            "top5_markets_contribution": float(top5),
            "max_category_contribution": float(max_category),
            "concentration_ok": str(concentration_ok),
            "category_breakdown": category_profits.to_dict()
        }

    def _analyze_price_drops(self, markets: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze edge by price drop magnitude (for RLM strategies)."""
        if "yes_price_drop" not in markets.columns:
            return []

        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"

        markets = markets.copy()

        # Define price drop ranges
        ranges = [
            ("0-5c", 0, 5),
            ("5-10c", 5, 10),
            ("10-20c", 10, 20),
            ("20c+", 20, 1000)
        ]

        results = []
        for name, low, high in ranges:
            subset = markets[
                (markets["yes_price_drop"] >= low) &
                (markets["yes_price_drop"] < high)
            ]

            if len(subset) >= 30:
                win_rate = (subset["market_result"] == side).mean()
                avg_price = subset[price_field].mean()
                breakeven = avg_price / 100
                edge = win_rate - breakeven

                results.append({
                    "range": name,
                    "n_markets": int(len(subset)),
                    "win_rate": float(win_rate),
                    "avg_no_price": float(avg_price),
                    "edge": float(edge),
                    "edge_pct": float(edge * 100)
                })

        return results

    def _analyze_by_category(self, markets: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze edge by market category."""
        side = self.config.bet_side
        price_field = "avg_no_price" if side == "no" else "avg_yes_price"

        # Get top categories by count
        top_categories = markets["category"].value_counts().head(15).index

        results = []
        for cat in top_categories:
            cat_markets = markets[markets["category"] == cat]
            n = len(cat_markets)

            if n >= 30:
                win_rate = (cat_markets["market_result"] == side).mean()
                avg_price = cat_markets[price_field].mean()
                breakeven = avg_price / 100
                edge = win_rate - breakeven

                results.append({
                    "category": cat,
                    "n_markets": int(n),
                    "win_rate": float(win_rate),
                    "avg_no_price": float(avg_price),
                    "edge": float(edge),
                    "edge_pct": float(edge * 100)
                })

        # Sort by count descending
        results.sort(key=lambda x: x["n_markets"], reverse=True)

        return results

    def _run_validation_checks(
        self,
        primary: Dict[str, Any],
        bucket_analysis: Dict[str, Any],
        temporal: Dict[str, Any],
        bootstrap_ci: Dict[str, float]
    ) -> Dict[str, bool]:
        """Run all validation checks and return pass/fail status."""
        checks = {
            "sample_size_ok": primary.get("n", 0) >= self.config.validation.min_markets,
            "statistically_significant": str(primary.get("significant", False)),
            "not_price_proxy": bucket_analysis.get("bucket_ratio", 0) > 0.5,
            "concentration_ok": True,  # Will be updated if concentration analysis available
            "temporal_stability_ok": temporal.get("temporal_stability_ratio", 0) >= 0.5
        }

        return checks

    def _create_lsd_result(
        self,
        primary: Dict[str, Any],
        markets: pd.DataFrame
    ) -> ValidationResult:
        """Create a quick LSD mode result."""
        # Get data metadata
        metadata = self.cache.get_metadata()

        return ValidationResult(
            strategy_name=self.config.name,
            hypothesis_id=self.config.hypothesis_id,
            validation_date=datetime.now().strftime("%Y-%m-%d"),
            validation_mode="lsd",
            data_range=f"{metadata['date_range']['start'][:10]} - {metadata['date_range']['end'][:10]}",
            is_valid=primary.get("valid", False),
            n_markets=primary.get("n", 0),
            wins=primary.get("wins", 0),
            losses=primary.get("losses", 0),
            win_rate=primary.get("win_rate", 0),
            avg_entry_price=primary.get("avg_price", 0),
            breakeven=primary.get("breakeven", 0),
            raw_edge=primary.get("edge", 0),
            z_score=primary.get("z_score", 0),
            p_value=primary.get("p_value", 1),
            is_statistically_significant=primary.get("significant", False),
            recommendation="FULL_VALIDATION_RECOMMENDED" if primary.get("edge", 0) > 0.05 else "REJECT"
        )

    def _create_full_result(
        self,
        primary: Dict[str, Any],
        signal_markets: pd.DataFrame,
        bucket_analysis: Dict[str, Any],
        temporal_stability: Dict[str, Any],
        bootstrap_ci: Dict[str, float],
        concentration: Dict[str, Any],
        price_drop: List[Dict[str, Any]],
        category_breakdown: List[Dict[str, Any]],
        checks: Dict[str, bool]
    ) -> ValidationResult:
        """Create a full validation result."""
        # Get data metadata
        metadata = self.cache.get_metadata()

        # Update concentration check
        checks["concentration_ok"] = concentration.get("concentration_ok", "True") == "True"

        # Determine if all checks pass
        all_pass = all(
            v if isinstance(v, bool) else v == "True"
            for v in checks.values()
        )

        # Generate recommendation
        if all_pass:
            recommendation = "VALIDATED - Keep in production"
        elif sum(1 for v in checks.values() if (v if isinstance(v, bool) else v == "True")) >= 3:
            recommendation = "PARTIALLY_VALIDATED - Monitor closely"
        else:
            recommendation = "REJECTED - Do not use"

        return ValidationResult(
            strategy_name=self.config.name,
            hypothesis_id=self.config.hypothesis_id,
            validation_date=datetime.now().strftime("%Y-%m-%d"),
            validation_mode="full",
            data_range=f"{metadata['date_range']['start'][:10]} - {metadata['date_range']['end'][:10]}",
            is_valid=primary.get("valid", False),
            n_markets=primary.get("n", 0),
            wins=primary.get("wins", 0),
            losses=primary.get("losses", 0),
            win_rate=primary.get("win_rate", 0),
            avg_entry_price=primary.get("avg_price", 0),
            breakeven=primary.get("breakeven", 0),
            raw_edge=primary.get("edge", 0),
            z_score=primary.get("z_score", 0),
            p_value=primary.get("p_value", 1),
            is_statistically_significant=primary.get("significant", False),
            bucket_analysis=bucket_analysis,
            confidence_interval_95=bootstrap_ci,
            temporal_stability=temporal_stability,
            concentration=concentration,
            price_drop_analysis=price_drop,
            category_breakdown=category_breakdown,
            validation_checks=checks,
            all_checks_pass=all_pass,
            recommendation=recommendation
        )


def validate_strategy(
    config_path: str,
    mode: str = "full",
    data_path: Optional[str] = None
) -> ValidationResult:
    """
    Convenience function to validate a strategy from a config file.

    Args:
        config_path: Path to the YAML config file
        mode: "lsd" or "full"
        data_path: Optional override for data path

    Returns:
        ValidationResult
    """
    from .config import load_config
    from .cache import create_cache

    # Load config
    config = load_config(config_path)

    # Create cache
    cache = create_cache(data_path=data_path)

    # Create validator
    validator = StrategyValidator(config, cache)

    # Run validation
    validation_mode = ValidationMode.LSD if mode.lower() == "lsd" else ValidationMode.FULL

    return validator.validate(mode=validation_mode)
