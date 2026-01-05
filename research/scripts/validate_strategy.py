#!/usr/bin/env python3
"""
Strategy Validation CLI Tool.

This script provides a command-line interface for validating trading strategies
using the automated validation framework.

Usage:
    # Quick LSD screening
    python validate_strategy.py --config strategies/configs/h123_rlm.yaml --mode lsd

    # Full validation
    python validate_strategy.py --config strategies/configs/h123_rlm.yaml --mode full

    # List available strategies
    python validate_strategy.py --list

    # Save results to JSON
    python validate_strategy.py --config h123_rlm.yaml --output results.json

Examples:
    # Run H123 RLM validation in LSD mode (quick ~2s)
    python validate_strategy.py -c h123_rlm.yaml -m lsd

    # Run full validation with custom output
    python validate_strategy.py -c h123_rlm.yaml -m full -o reports/h123_validation.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.scripts.validation.config import load_config, list_configs
from research.scripts.validation.cache import create_cache
from research.scripts.validation.validator import StrategyValidator, ValidationMode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_strategy")


def print_banner():
    """Print a banner for the CLI."""
    print("=" * 70)
    print("  STRATEGY VALIDATION FRAMEWORK")
    print("  Kalshi Flow Research - Automated Hypothesis Testing")
    print("=" * 70)


def print_lsd_result(result):
    """Print LSD mode results in a compact format."""
    print("\n" + "-" * 50)
    print("LSD MODE SCREENING RESULTS")
    print("-" * 50)

    print(f"\nStrategy: {result.strategy_name}")
    print(f"Hypothesis: {result.hypothesis_id}")
    print(f"Data Range: {result.data_range}")

    print(f"\n--- Quick Metrics ---")
    print(f"  Markets: {result.n_markets:,}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Breakeven: {result.breakeven*100:.1f}%")
    print(f"  Raw Edge: {result.raw_edge*100:+.2f}%")
    print(f"  P-value: {result.p_value:.2e}")
    print(f"  Significant: {'YES' if result.is_statistically_significant else 'NO'}")

    print(f"\n--- Recommendation ---")
    if result.raw_edge > 0.05 and result.n_markets >= 50:
        print("  [PASS] Edge > 5% with sufficient sample")
        print("  -> RECOMMEND: Run FULL validation")
    elif result.raw_edge > 0:
        print("  [MARGINAL] Positive edge but below threshold")
        print("  -> Consider: Run FULL validation for deeper analysis")
    else:
        print("  [FAIL] No positive edge detected")
        print("  -> REJECT: Do not proceed")


def print_full_result(result):
    """Print full validation results in detailed format."""
    print("\n" + "=" * 70)
    print("FULL VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nStrategy: {result.strategy_name}")
    print(f"Hypothesis: {result.hypothesis_id}")
    print(f"Validation Date: {result.validation_date}")
    print(f"Data Range: {result.data_range}")

    # Primary Metrics
    print("\n" + "-" * 50)
    print("PRIMARY METRICS")
    print("-" * 50)
    print(f"  Markets: {result.n_markets:,}")
    print(f"  Wins/Losses: {result.wins}/{result.losses}")
    print(f"  Win Rate: {result.win_rate*100:.2f}%")
    print(f"  Avg Entry Price: {result.avg_entry_price:.1f}c")
    print(f"  Breakeven: {result.breakeven*100:.2f}%")
    print(f"  RAW EDGE: {result.raw_edge*100:+.2f}%")
    print(f"  Z-Score: {result.z_score:.2f}")
    print(f"  P-value: {result.p_value:.2e}")

    # Bucket Analysis
    if result.bucket_analysis:
        ba = result.bucket_analysis
        print("\n" + "-" * 50)
        print("BUCKET-MATCHED ANALYSIS (Price Proxy Check)")
        print("-" * 50)
        print(f"  Buckets Analyzed: {ba['n_buckets']}")
        print(f"  Positive Buckets: {ba['positive_buckets']}/{ba['n_buckets']} ({ba['bucket_ratio']*100:.1f}%)")
        print(f"  Avg Improvement vs Baseline: {ba['avg_improvement_pct']:+.2f}%")

        # Show bucket details if not too many
        if ba.get('buckets') and len(ba['buckets']) <= 20:
            print("\n  Bucket Details:")
            print(f"  {'Bucket':>8} {'N':>6} {'Signal':>8} {'Base':>8} {'Diff':>8}")
            print("  " + "-" * 44)
            for b in ba['buckets']:
                diff_str = f"{b['improvement']*100:+.1f}%"
                marker = "+" if b['positive'] == "True" else "-"
                print(f"  {b['bucket']:>8} {b['n_signal']:>6} {b['signal_win_rate']*100:>7.1f}% "
                      f"{b['baseline_win_rate']*100:>7.1f}% {diff_str:>8} {marker}")

    # Confidence Intervals
    if result.confidence_interval_95:
        ci = result.confidence_interval_95
        print("\n" + "-" * 50)
        print("CONFIDENCE INTERVALS (95%)")
        print("-" * 50)
        print(f"  Win Rate: [{ci['win_rate_lower']*100:.2f}%, {ci['win_rate_upper']*100:.2f}%]")
        print(f"  Edge: [{ci['edge_lower']*100:.2f}%, {ci['edge_upper']*100:.2f}%]")

    # Temporal Stability
    if result.temporal_stability:
        ts = result.temporal_stability
        print("\n" + "-" * 50)
        print("TEMPORAL STABILITY")
        print("-" * 50)
        print(f"  Positive Weeks: {ts['positive_weeks']}/{ts['total_weeks']} "
              f"({ts['temporal_stability_ratio']*100:.1f}%)")

        if ts.get('weekly_results'):
            print("\n  Weekly Results:")
            print(f"  {'Week':>6} {'N':>6} {'Win%':>8} {'Edge':>8}")
            print("  " + "-" * 32)
            for w in ts['weekly_results'][:8]:  # Show first 8 weeks
                print(f"  {w['week']:>6} {w['n_markets']:>6} {w['win_rate']*100:>7.1f}% "
                      f"{w['edge']*100:>+7.2f}%")

    # Price Drop Analysis
    if result.price_drop_analysis:
        print("\n" + "-" * 50)
        print("PRICE DROP ANALYSIS")
        print("-" * 50)
        print(f"  {'Range':>10} {'N':>6} {'Win%':>8} {'Edge':>8}")
        print("  " + "-" * 36)
        for pd in result.price_drop_analysis:
            print(f"  {pd['range']:>10} {pd['n_markets']:>6} {pd['win_rate']*100:>7.1f}% "
                  f"{pd['edge_pct']:>+7.2f}%")

    # Category Breakdown
    if result.category_breakdown:
        print("\n" + "-" * 50)
        print("TOP CATEGORIES")
        print("-" * 50)
        print(f"  {'Category':<25} {'N':>6} {'Win%':>8} {'Edge':>8}")
        print("  " + "-" * 52)
        for cat in result.category_breakdown[:10]:  # Show top 10
            print(f"  {cat['category']:<25} {cat['n_markets']:>6} {cat['win_rate']*100:>7.1f}% "
                  f"{cat['edge_pct']:>+7.2f}%")

    # Validation Checks
    if result.validation_checks:
        print("\n" + "-" * 50)
        print("VALIDATION CHECKLIST")
        print("-" * 50)
        for check, passed in result.validation_checks.items():
            passed_bool = passed if isinstance(passed, bool) else passed == "True"
            status = "[PASS]" if passed_bool else "[FAIL]"
            print(f"  {status} {check.replace('_', ' ').title()}")

        print(f"\n  Overall: {sum(1 for v in result.validation_checks.values() if (v if isinstance(v, bool) else v == 'True'))}/{len(result.validation_checks)} checks passed")

    # Final Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    verdict_color = "PASS" if result.all_checks_pass else "REVIEW"
    print(f"  [{verdict_color}] {result.recommendation}")
    print("=" * 70)


def list_available_strategies():
    """List all available strategy configs."""
    # Default configs directory
    configs_dir = Path(__file__).parent.parent / "strategies" / "configs"

    print("\nAvailable Strategy Configs:")
    print("-" * 50)

    configs = list_configs(configs_dir)

    if not configs:
        print("  No configs found in strategies/configs/")
        print(f"  (looked in: {configs_dir})")
        return

    for config_path in configs:
        try:
            config = load_config(config_path)
            print(f"  {config_path.name}")
            print(f"    Name: {config.name}")
            print(f"    Hypothesis: {config.hypothesis_id}")
            print(f"    Action: {config.action.value}")
            print()
        except Exception as e:
            print(f"  {config_path.name} (ERROR: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Validate trading strategies using the automated framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick LSD screening
  python validate_strategy.py --config h123_rlm.yaml --mode lsd

  # Full validation
  python validate_strategy.py --config h123_rlm.yaml --mode full

  # List available strategies
  python validate_strategy.py --list

  # Save results to JSON
  python validate_strategy.py --config h123_rlm.yaml --output results.json
        """
    )

    parser.add_argument(
        "-c", "--config",
        help="Path to strategy config YAML file (relative to strategies/configs/ or absolute)"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["lsd", "full"],
        default="full",
        help="Validation mode: 'lsd' for quick screening, 'full' for rigorous (default: full)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results (optional)"
    )

    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available strategy configs"
    )

    parser.add_argument(
        "-d", "--data",
        help="Override path to trade data CSV (optional)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output, only write to file"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Handle --list
    if args.list:
        print_banner()
        list_available_strategies()
        return 0

    # Require --config if not --list
    if not args.config:
        parser.error("--config is required (or use --list to see available configs)")

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to strategies/configs/
        base_dir = Path(__file__).parent.parent / "strategies" / "configs"
        if (base_dir / args.config).exists():
            config_path = base_dir / args.config
        elif not config_path.exists():
            # Try with .yaml extension
            if (base_dir / f"{args.config}.yaml").exists():
                config_path = base_dir / f"{args.config}.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"  Looked in: {Path(__file__).parent.parent / 'strategies' / 'configs'}")
        return 1

    if not args.quiet:
        print_banner()

    try:
        # Load config
        logger.info(f"Loading config from {config_path}")
        config = load_config(config_path)

        # Create cache
        logger.info("Initializing data cache...")
        cache = create_cache(data_path=args.data)

        # Create validator
        validator = StrategyValidator(config, cache)

        # Run validation
        mode = ValidationMode.LSD if args.mode == "lsd" else ValidationMode.FULL
        logger.info(f"Running {mode.value.upper()} validation...")

        result = validator.validate(mode=mode)

        # Print results
        if not args.quiet:
            if mode == ValidationMode.LSD:
                print_lsd_result(result)
            else:
                print_full_result(result)

        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            if not args.quiet:
                print(f"\nResults saved to: {output_path}")

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        logger.exception("Validation failed")
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
