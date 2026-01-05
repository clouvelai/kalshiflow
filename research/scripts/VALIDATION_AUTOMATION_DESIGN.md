# Strategy Validation Automation Framework Design

## Document Information
- **Created**: 2026-01-04
- **Purpose**: Design document for automating the strategy validation process
- **Status**: Design Phase (not implemented)

---

## Executive Summary

The current strategy validation process involves manually writing Python scripts for each hypothesis, with significant code duplication across ~80+ analysis scripts in `research/analysis/`. This document proposes a validation automation framework that standardizes the process, reduces duplication, and accelerates hypothesis testing by 3-5x.

---

## Part 1: Current Pain Points Analysis

### 1.1 Repetitive Code Patterns

After analyzing the existing scripts, I identified these repetitive patterns across nearly all validation scripts:

| Pattern | Occurrences | LOC per Instance |
|---------|-------------|------------------|
| Data loading & preprocessing | Every script | ~20-40 lines |
| Baseline win rate calculation | Every validation | ~30-50 lines |
| Bucket-matched edge calculation | Most validations | ~80-120 lines |
| Bootstrap confidence intervals | ~60% of scripts | ~30-40 lines |
| Temporal stability (train/test split) | ~70% of scripts | ~40-60 lines |
| P-value calculation | Every validation | ~15-25 lines |
| Report generation & JSON export | Every script | ~30-50 lines |

**Estimated duplication**: 250-400 lines of boilerplate per validation script.

### 1.2 Inconsistent Methodology

Different validation scripts use slightly different methodologies:

1. **Bucket sizes vary**: Some use 5-cent buckets, others 10-cent
2. **Minimum market thresholds differ**: 30, 50, or 100 markets per subsegment
3. **P-value thresholds inconsistent**: 0.05, 0.01, 0.001, or 0.0005
4. **Temporal splits vary**: 50/50, 80/20, or rolling windows
5. **Baseline definitions differ**: All markets vs. same-price markets

### 1.3 Manual Execution Overhead

Current workflow requires:
1. Write custom Python script (~15-30 min)
2. Run script and wait for results (~5-20 min depending on dataset)
3. Interpret raw output (~5-10 min)
4. Copy results to hypothesis documentation (~5-10 min)
5. Repeat for parameter sensitivity (~30-60 min)

**Total time per hypothesis**: 60-120 minutes in "normal mode", 10-15 minutes in "LSD mode"

### 1.4 No Caching or Incremental Updates

When the trade dataset grows (currently 7.9M trades), every validation recomputes:
- Market-level aggregations
- Baseline win rates per bucket
- All market features (leverage, whale count, etc.)

This results in ~30-60 second startup time per script.

---

## Part 2: Proposed Automation Architecture

### 2.1 High-Level Design

```
+-------------------+     +-------------------+     +-------------------+
|  Strategy Config  | --> |  Validation Core  | --> |  Report Generator |
|     (YAML/JSON)   |     |     (Python)      |     |   (JSON/MD/HTML)  |
+-------------------+     +-------------------+     +-------------------+
                                  |
                                  v
                          +-------------------+
                          |  Cached Data Layer|
                          |  (Parquet/SQLite) |
                          +-------------------+
```

### 2.2 Core Components

#### Component 1: Strategy Configuration Schema

```yaml
# strategy_config.yaml
strategy:
  name: "H123 - Reverse Line Movement NO"
  hypothesis_id: "H123"
  action: "bet_no"  # bet_yes | bet_no

signal:
  # Signal conditions (ALL must be true)
  conditions:
    - field: "yes_trade_ratio"
      operator: ">"
      value: 0.70
    - field: "price_dropped"
      operator: "=="
      value: true
    - field: "trade_count"
      operator: ">="
      value: 5

  # Price field for breakeven calculation
  entry_price_field: "no_price"

validation:
  mode: "full"  # lsd | full
  min_markets: 50
  p_threshold: 0.001
  bucket_size: 5  # cents

parameter_sensitivity:
  enabled: true
  parameters:
    yes_trade_ratio:
      values: [0.60, 0.65, 0.70, 0.75, 0.80]
    trade_count:
      values: [3, 5, 7, 10, 15, 20]
```

#### Component 2: Cached Data Layer

**Purpose**: Precompute and cache expensive aggregations.

```python
# Cached artifacts:
class CachedDataLayer:
    """
    Precomputed data for fast validation.

    Cached at startup, invalidated when source data changes.
    """

    # Market-level aggregations (precomputed once)
    markets_df: pd.DataFrame  # One row per market with all features

    # Baseline win rates by bucket
    baseline_cache: Dict[int, BaselineStats]  # bucket -> stats

    # Trade-level data (lazy-loaded)
    _trades_df: Optional[pd.DataFrame] = None

    def get_baseline(self, bucket: int) -> BaselineStats:
        """Get cached baseline for a price bucket."""
        pass

    def filter_markets(self, conditions: List[Condition]) -> pd.DataFrame:
        """Filter markets by conditions, using vectorized operations."""
        pass
```

**Estimated speedup**: 10-20x for repeat validations (from ~30s to ~2s).

#### Component 3: Validation Core

```python
class StrategyValidator:
    """
    Generic strategy validation engine.

    Accepts strategy configuration and produces standardized results.
    """

    def __init__(self, config: StrategyConfig, cache: CachedDataLayer):
        self.config = config
        self.cache = cache

    def validate(self, mode: str = "full") -> ValidationResult:
        """
        Run validation pipeline.

        Args:
            mode: "lsd" for quick screening, "full" for rigorous validation

        Returns:
            ValidationResult with all metrics
        """
        # 1. Filter markets by signal conditions
        signal_markets = self.cache.filter_markets(self.config.signal.conditions)

        # 2. Calculate primary metrics
        primary = self._calculate_primary_metrics(signal_markets)

        # 3. Quick exit for LSD mode
        if mode == "lsd":
            return ValidationResult(
                quick_edge=primary.edge,
                sample_size=len(signal_markets),
                passes_threshold=primary.edge > 0.05,
                full_validation_recommended=primary.edge > 0.05 and len(signal_markets) > 50
            )

        # 4. Full validation pipeline
        bucket_analysis = self._analyze_by_bucket(signal_markets)
        temporal_stability = self._analyze_temporal_stability(signal_markets)
        bootstrap_ci = self._bootstrap_confidence_interval(signal_markets)
        parameter_sensitivity = self._sweep_parameters(signal_markets)

        return ValidationResult(
            primary=primary,
            bucket_analysis=bucket_analysis,
            temporal_stability=temporal_stability,
            bootstrap_ci=bootstrap_ci,
            parameter_sensitivity=parameter_sensitivity,
            verdict=self._generate_verdict(...)
        )
```

#### Component 4: Report Generator

```python
class ReportGenerator:
    """
    Generate standardized reports from validation results.
    """

    def to_json(self, result: ValidationResult) -> str:
        """Export to JSON for programmatic use."""
        pass

    def to_markdown(self, result: ValidationResult) -> str:
        """Export to Markdown for documentation."""
        pass

    def to_console(self, result: ValidationResult) -> None:
        """Print formatted console output."""
        pass
```

### 2.3 CLI Interface

```bash
# Quick LSD screening
python -m research.validate --config strategy.yaml --mode lsd

# Full validation
python -m research.validate --config strategy.yaml --mode full

# Batch validation (multiple strategies)
python -m research.validate --batch strategies/ --mode lsd --parallel 4

# Generate comparison report
python -m research.validate --compare H123 H124 H125
```

---

## Part 3: Implementation Phases

### Phase 1: Core Framework (Estimated: 2-3 days)

**Deliverables**:
1. `StrategyConfig` dataclass and YAML parser
2. `CachedDataLayer` with market aggregations
3. `StrategyValidator.validate()` for LSD mode only
4. Basic console output

**Files to create**:
```
research/
  validation/
    __init__.py
    config.py         # StrategyConfig schema
    cache.py          # CachedDataLayer
    validator.py      # StrategyValidator
    metrics.py        # Statistical calculations
    cli.py            # Command-line interface
```

### Phase 2: Full Validation Pipeline (Estimated: 2-3 days)

**Deliverables**:
1. Bucket-matched baseline comparison
2. Temporal stability analysis
3. Bootstrap confidence intervals
4. Parameter sensitivity sweeps
5. Verdict generation logic

**Additional files**:
```
research/
  validation/
    analysis/
      bucket_analysis.py
      temporal.py
      bootstrap.py
      sensitivity.py
    verdict.py
```

### Phase 3: Reporting & Integration (Estimated: 1-2 days)

**Deliverables**:
1. JSON report export
2. Markdown report generation
3. Integration with existing `research/reports/` structure
4. Migration guide for existing scripts

### Phase 4: Advanced Features (Estimated: 2-3 days)

**Deliverables**:
1. Parallel batch validation
2. Incremental cache updates when data changes
3. Web UI for validation results (optional)
4. Integration with hypothesis tracking system

---

## Part 4: Key Design Decisions

### Decision 1: Configuration Format

**Recommendation**: YAML with JSON Schema validation

**Rationale**:
- YAML is human-readable and supports comments
- JSON Schema provides validation and IDE autocomplete
- Can be version-controlled alongside code

### Decision 2: Caching Strategy

**Recommendation**: Parquet files with hash-based invalidation

**Rationale**:
- Parquet is fast for columnar operations (common in validation)
- Hash of source CSV determines cache validity
- ~10-20x speedup for repeat validations

### Decision 3: Signal Definition Language

**Recommendation**: Simple condition-based DSL (not full Python)

**Rationale**:
- Covers 90%+ of validation needs with simple conditions
- Enables non-programmers to define strategies
- Complex signals can still use custom Python via "hooks"

### Decision 4: LSD vs Full Mode

**Recommendation**: Separate code paths with shared infrastructure

```python
if mode == "lsd":
    # Skip: bucket analysis, temporal, bootstrap, parameter sweep
    # Calculate: raw edge, sample size, pass/fail threshold
    return quick_result

# Full mode continues with all analyses
```

---

## Part 5: Estimated Complexity

### Lines of Code Estimate

| Component | Estimated LOC | Complexity |
|-----------|---------------|------------|
| Config schema & parser | 200-300 | Low |
| Cached data layer | 300-400 | Medium |
| Validator core | 400-500 | Medium |
| Bucket analysis | 150-200 | Low |
| Temporal analysis | 100-150 | Low |
| Bootstrap CI | 80-120 | Low |
| Parameter sensitivity | 150-200 | Medium |
| Report generator | 200-300 | Low |
| CLI interface | 150-200 | Low |
| **Total** | **1,700-2,400** | Medium |

### Time Estimate (with tests)

| Phase | Estimated Time |
|-------|----------------|
| Phase 1 (Core) | 2-3 days |
| Phase 2 (Full validation) | 2-3 days |
| Phase 3 (Reporting) | 1-2 days |
| Phase 4 (Advanced) | 2-3 days |
| **Total** | **7-11 days** |

### Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Edge cases in signal DSL | Medium | Support custom Python hooks |
| Cache invalidation bugs | Medium | Always allow cache bypass flag |
| Performance regression | Low | Benchmark against existing scripts |
| Adoption resistance | Low | Provide migration guide, keep old scripts working |

---

## Part 6: Comparison with Current Approach

### Before (Current State)

```python
# ~400 lines per validation script
def main():
    # Load data (30 lines, duplicated in every script)
    df = pd.read_csv(TRADES_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    # ... preprocessing ...

    # Build baseline (50 lines, duplicated)
    baseline = build_baseline(df)

    # Custom signal logic (varies)
    signal_markets = df[
        (df['yes_ratio'] > 0.7) &
        (df['price_dropped']) &
        (df['n_trades'] >= 5)
    ]

    # Calculate stats (80 lines, duplicated)
    # Bootstrap (40 lines, duplicated)
    # Temporal split (40 lines, duplicated)
    # Report (50 lines, duplicated)
```

### After (With Framework)

```yaml
# 20 lines of configuration
strategy:
  name: "H123 - Reverse Line Movement NO"
  action: "bet_no"

signal:
  conditions:
    - field: "yes_ratio"
      operator: ">"
      value: 0.70
    - field: "price_dropped"
      value: true
    - field: "n_trades"
      operator: ">="
      value: 5
```

```bash
# One command
python -m research.validate --config h123.yaml --mode full
```

**Reduction**: 400 lines -> 20 lines config + 1 command

---

## Part 7: Open Questions

1. **Should we support Python hooks for complex signals?**
   - Yes, via `custom_filter` field pointing to Python function

2. **How do we handle strategies that need trade-level data (not just market-level)?**
   - Add `granularity: trade` option that uses trades_df instead of markets_df

3. **Should we auto-generate hypothesis IDs?**
   - No, hypothesis IDs should be manually assigned for traceability

4. **How do we handle multi-strategy comparisons?**
   - Phase 4 feature: `--compare` flag generates side-by-side report

5. **Integration with existing RLM service?**
   - Validated strategies export to `traderv3/services/` format (separate tool)

---

## Part 8: Recommended Next Steps

1. **Immediate**: Review this design with the team
2. **If approved**: Start with Phase 1 (core framework)
3. **Validation**: Test against existing H123 validation to ensure parity
4. **Iteration**: Gather feedback after first 5-10 strategies validated
5. **Documentation**: Create user guide and migration path for existing scripts

---

## Appendix A: Example Strategy Configs

### RLM (Reverse Line Movement)

```yaml
strategy:
  name: "RLM - Reverse Line Movement NO"
  hypothesis_id: "H123"
  action: "bet_no"

signal:
  conditions:
    - field: "yes_trade_ratio"
      operator: ">"
      value: 0.70
    - field: "price_move_toward_no"
      operator: ">"
      value: 0  # Any drop
    - field: "trade_count"
      operator: ">="
      value: 5
  entry_price_field: "no_price"

validation:
  mode: "full"
  p_threshold: 0.001
  min_markets: 50
```

### Whale Following

```yaml
strategy:
  name: "Whale Following - Bet with $1k+ traders"
  hypothesis_id: "H045"
  action: "bet_no"

signal:
  conditions:
    - field: "has_whale_no_trade"
      value: true
    - field: "whale_no_ratio"
      operator: ">"
      value: 0.5
  entry_price_field: "no_price"

validation:
  mode: "full"
```

### LSD Screening Example

```yaml
# Quick screening config
strategy:
  name: "Fibonacci Trade Count"
  hypothesis_id: "L002"
  action: "bet_no"

signal:
  conditions:
    - field: "trade_count"
      operator: "in"
      value: [8, 13, 21, 34]
  entry_price_field: "no_price"

validation:
  mode: "lsd"
  min_markets: 30
```

---

## Appendix B: Existing Scripts to Consolidate

The following scripts in `research/analysis/` contain reusable patterns that should be consolidated into the framework:

### High-Value Scripts (Good Patterns)
- `h123_deep_validation.py` - Comprehensive validation template
- `rlm_reliability_grid_search.py` - Parameter sensitivity pattern
- `session012_deep_validation.py` - Bucket analysis pattern

### Scripts with Unique Logic (Need Custom Hooks)
- `session011_bot_exploitation.py` - Complex bot detection logic
- `exhaustive_strategy_search.py` - Multi-strategy search
- `detect_informed_trading.py` - Trade sequence analysis

### Scripts to Deprecate (After Framework Ready)
- Most `session00X_*.py` scripts - One-off explorations
- Duplicate validation scripts with minor differences

---

*End of Design Document*
