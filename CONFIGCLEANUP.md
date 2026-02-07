# V3 Config Cleanup Plan

**Created:** 2026-02-05
**Status:** Pending validation
**Branch:** sam/arb-mvp

## Objective

Remove dead/unused configuration settings so only single-arb (Captain) and core V3 trader settings remain.

---

## Summary of Changes

| Category | Items to Remove | Files Affected |
|----------|-----------------|----------------|
| Unused config fields | 7 fields | environment.py |
| Deprecated env vars | 5 vars | .env.paper |
| Outdated comments | 3 lines | run-v3.sh |

---

## Detailed Changes

### 1. `backend/src/kalshiflow_rl/traderv3/config/environment.py`

**Remove these unused fields from V3Config dataclass:**

| Field | Line | Env Variable | Reason |
|-------|------|--------------|--------|
| `orderbook_depth` | 46 | `V3_ORDERBOOK_DEPTH` | Never used in code |
| `snapshot_interval` | 47 | `V3_SNAPSHOT_INTERVAL` | Only used in validation, no business logic |
| `allow_multiple_positions_per_market` | 62 | `V3_ALLOW_MULTIPLE_POSITIONS` | Only logged, never checked in trading logic |
| `allow_multiple_orders_per_market` | 63 | `V3_ALLOW_MULTIPLE_ORDERS` | Only logged, never checked in trading logic |
| `single_arb_max_contracts` | 108 | `V3_SINGLE_ARB_MAX_CONTRACTS` | Defined but never used in arb coordinator |
| `error_recovery_delay` | 115 | `V3_ERROR_RECOVERY_DELAY` | Defined but never used |
| `ws_ping_interval` | 120 | `V3_WS_PING_INTERVAL` | Defined but never used |

**Code locations to edit:**

1. **Field definitions** (remove):
   - Lines 46-47: `orderbook_depth`, `snapshot_interval`
   - Lines 62-63: `allow_multiple_positions_per_market`, `allow_multiple_orders_per_market`
   - Line 108: `single_arb_max_contracts`
   - Line 115: `error_recovery_delay`
   - Line 120: `ws_ping_interval`

2. **from_env() loading** (remove):
   - Lines 171-172: Loading `orderbook_depth`, `snapshot_interval`
   - Lines 193-194: Loading `allow_multiple_positions_per_market`, `allow_multiple_orders_per_market`
   - Line 241: Loading `single_arb_max_contracts`
   - Line 247: Loading `error_recovery_delay`
   - Line 251: Loading `ws_ping_interval`

3. **Config construction** (remove from cls() call):
   - Lines 265-266: `orderbook_depth=`, `snapshot_interval=`
   - Lines 273-274: `allow_multiple_positions_per_market=`, `allow_multiple_orders_per_market=`
   - Line 299: `single_arb_max_contracts=`
   - Line 304: `error_recovery_delay=`
   - Line 307: `ws_ping_interval=`

4. **Logging statements** (remove):
   - Lines 338-341: Warning logs for `allow_multiple_positions_per_market` and `allow_multiple_orders_per_market`

5. **Validation** (remove):
   - Lines 395-396: Validation for `snapshot_interval`

---

### 2. `scripts/run-v3.sh`

**Update outdated comments:**

| Line | Current | Change To |
|------|---------|-----------|
| 5 | `"Runs the V3 Trader with Kalshi-Polymarket arbitrage strategy."` | `"Runs the V3 Trader with single-event arbitrage (Captain system)."` |
| 13 | `"# Arb system enabled via V3_ARB_ENABLED=true and V3_POLYMARKET_ENABLED=true"` | Remove this line |
| 29 | `"Kalshi-Polymarket Arbitrage"` | `"Single-Event Arbitrage (Captain)"` |

---

### 3. `backend/.env.paper`

**Remove deprecated env vars:**

```bash
# REMOVE lines 95-97:
# Testing Flags (allow multiple positions per market for paper testing)
V3_ALLOW_MULTIPLE_POSITIONS=true
V3_ALLOW_MULTIPLE_ORDERS=true

# REMOVE lines 99-102:
# Arb / Polymarket
V3_ARB_ENABLED=true
V3_POLYMARKET_ENABLED=true
V3_ARB_ORCHESTRATOR_ENABLED=true
```

---

## What NOT to Remove

**These fields ARE actively used (verified by grep):**

| Field | Used In |
|-------|---------|
| `sports_allowed_prefixes` | coordinator.py (lines 452, 704) |
| `lifecycle_sync_interval` | coordinator.py (line 473) |
| `api_discovery_interval` | coordinator.py (lines 705, 715) |
| `api_discovery_batch_size` | coordinator.py (line 706) |
| `discovery_min_hours_to_settlement` | coordinator.py (line 707) |
| `discovery_max_days_to_settlement` | coordinator.py (line 708) |
| `dormant_volume_threshold` | tracked_markets_syncer.py (line 355) |
| `dormant_grace_period_hours` | tracked_markets_syncer.py (line 364) |
| `event_exposure_action` | event_position_tracker.py (lines 26, 165, 533, 560) |
| `health_check_interval` | health_monitor.py (lines 208, 498) |
| `order_ttl_seconds` | trading_state_syncer.py (lines 265, 268, 291) |

**RL_ prefix variables** belong to a separate RL training subsystem (RLConfig class in `config.py`) and should NOT be touched.

---

## Verification Steps

After making changes:

1. **Config load test:**
   ```bash
   cd backend
   uv run python -c "from kalshiflow_rl.traderv3.config.environment import V3Config; import os; os.environ.setdefault('KALSHI_API_URL','x'); os.environ.setdefault('KALSHI_WS_URL','x'); os.environ.setdefault('KALSHI_API_KEY_ID','x'); os.environ.setdefault('KALSHI_PRIVATE_KEY_CONTENT','x'); c = V3Config.from_env(); print('Config loaded successfully')"
   ```

2. **Start V3 trader:**
   ```bash
   ./scripts/run-v3.sh paper
   ```

3. **Check startup logs:**
   - No errors about missing config fields
   - Single-arb system initializes correctly
   - Captain starts if enabled

4. **Verify no regressions:**
   - Orderbook subscriptions work
   - Trading client connects
   - Event tracking functions
   - Health endpoint responds at http://localhost:8005/v3/health

---

## Notes

- The `V3_CALIBRATION_DURATION` legacy fallback (line 245) is kept for backwards compatibility
- Root `.env.paper` file should also be checked for deprecated vars if it exists
- No changes to RL subsystem configuration
