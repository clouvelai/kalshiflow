# M1-M2 Foundation Issues - High and Medium Priority

This document captures high and medium priority issues identified during the M1-M2 foundation review. These issues should be addressed in future sessions after the critical issues are resolved.

## High Priority Issues

### 1. Model Loaded But Never Used

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:148-149`

**Problem**: 
- Model is loaded if `model_path` is provided, but `ActionSelectorStub` ignores it
- Wastes memory (model stays in RAM unused)
- Wastes startup time loading model that won't be used
- Creates confusion about whether model is actually being used

**Impact**: 
- Performance: Unnecessary memory usage and startup delay
- Developer confusion: Model appears loaded but isn't used

**Recommended Fix**:
- Don't load model if using stub selector
- Add check/warning if model_path provided but stub selector configured
- Or defer model loading until ActionSelector actually needs it

---

### 2. Position Updates Don't Account for Async Fills

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:601-637`

**Problem**:
- `_update_positions()` reads positions immediately after order execution
- Fills are processed asynchronously in a separate queue (`KalshiMultiMarketOrderManager._process_fills()`)
- Position may not reflect the fill yet when read
- Portfolio value calculation may be stale

**Impact**:
- Position tracking temporarily incorrect
- Portfolio value calculations may be wrong
- Observation features may use stale position data

**Recommended Fix**:
- Document this limitation clearly
- Consider adding delay/retry mechanism for position reads
- Or expose fill events to ActorService for immediate position updates
- Or use eventual consistency model with clear documentation

---

### 3. Dual Observation Adapter Paths - Potential Inconsistency

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:485-518`

**Problem**:
- Two code paths for observation building:
  1. Injected adapter (with portfolio data) - `self._injected_observation_adapter`
  2. Callback adapter (may not support portfolio data) - `self._observation_adapter`
- Which path is used depends on initialization order
- Callback adapter may not receive portfolio/position data

**Impact**:
- Inconsistent observations depending on configuration
- Portfolio features may be missing in some cases
- Hard to debug which path is being used

**Recommended Fix**:
- Standardize on one path (prefer injected adapter)
- Deprecate callback adapter path
- Or add clear documentation about which path is used when
- Ensure both paths support same interface

---

## Medium Priority Issues

### 4. Missing Validation for Required Dependencies

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:_process_market_update()`

**Problem**:
- Pipeline can proceed with missing dependencies, leading to silent failures:
  - No observation adapter → returns None, continues
  - No action selector → returns None, continues  
  - No order manager → returns None, continues
- No validation in `initialize()` to ensure required dependencies are set
- Errors only surface at runtime during processing

**Impact**:
- Silent failures - events processed but nothing happens
- Hard to debug why actions aren't executing
- No early detection of misconfiguration

**Recommended Fix**:
- Add validation in `initialize()` to check required dependencies
- Fail fast if critical dependencies missing
- Or add comprehensive logging when dependencies missing
- Consider using dependency injection container to enforce dependencies

---

### 5. Error Circuit Breaker Removes Markets Permanently

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:439-440`

**Problem**:
- On too many errors, market is removed from `self.market_tickers` list
- Permanent removal (until restart)
- No way to re-enable market
- Could remove all markets if errors persist

**Impact**:
- Markets permanently disabled until restart
- No recovery mechanism
- Could disable entire trading system

**Recommended Fix**:
- Use disabled markets set instead of modifying active list
- Add re-enable mechanism (time-based or manual)
- Or add exponential backoff with automatic retry
- Consider circuit breaker pattern with half-open state

---

### 6. Hardcoded Default Portfolio Values

**Location**: `backend/src/kalshiflow_rl/trading/actor_service.py:461-462`

**Problem**:
- Default portfolio/cash values (10000.0) are hardcoded
- Should come from OrderManager
- Could mismatch actual values if OrderManager has different initial cash

**Impact**:
- Incorrect portfolio values in observations
- Mismatch between actual and observed portfolio state
- Training/inference inconsistency if defaults differ

**Recommended Fix**:
- Get defaults from OrderManager if available
- Or use config values for initial portfolio
- Ensure consistency between OrderManager initial cash and observation defaults
- Document expected initial values

---

## Summary

These issues don't prevent basic functionality but should be addressed to improve:
- **Reliability**: Position tracking accuracy, dependency validation
- **Consistency**: Observation building paths, portfolio values
- **Debuggability**: Model usage clarity, error recovery
- **Performance**: Unnecessary model loading

Priority order for addressing:
1. Position updates with async fills (affects trading accuracy)
2. Dual observation adapter paths (affects observation consistency)
3. Missing dependency validation (affects debuggability)
4. Model loading optimization (affects performance)
5. Circuit breaker improvements (affects reliability)
6. Hardcoded defaults (affects consistency)

