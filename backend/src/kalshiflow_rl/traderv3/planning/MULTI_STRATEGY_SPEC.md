# Multi-Strategy Implementation Specification

**Status**: Ready for Implementation
**Date**: 2024-12-31
**Strategies**: RLM_NO + S013 (concurrent)

---

## 1. Core Data Structures

### 1.1 StrategyEntry (Individual Trade Record)

Each entry is a SEPARATE record - no blending across strategies or re-entries.

```python
# traderv3/state/strategy_tracking.py

@dataclass
class StrategyEntry:
    """
    Individual trade entry - NEVER blended.

    Each order gets its own entry. Re-entries create new records.
    P&L calculated per-entry, then aggregated to strategy level.
    """
    entry_id: str              # Unique: f"{strategy}:{order_id}"
    order_id: str              # Kalshi order ID
    strategy: str              # "RLM_NO" or "S013"
    market_ticker: str
    side: str                  # "yes" or "no"
    contracts: int             # This entry's contract count
    entry_price: int           # THIS entry's price (cents) - NEVER averaged
    cost_basis: int            # contracts * entry_price (cents)
    created_at: float
    signal_id: str             # Strategy-specific signal reference

    # Fill tracking
    status: str                # "pending", "filled", "partial", "cancelled"
    fill_price: Optional[int] = None
    filled_contracts: int = 0
    filled_at: Optional[float] = None

    # Settlement (when position closes)
    settlement_price: Optional[int] = None  # 0 or 100 for binary
    realized_pnl: Optional[int] = None      # Calculated on settlement
    settled_at: Optional[float] = None


@dataclass
class StrategySignal:
    """
    Records ALL signals including skipped ones.

    Critical for analyzing strategy behavior and skip patterns.
    """
    signal_id: str             # Unique: f"{strategy}:{market}:{timestamp_ms}"
    strategy: str
    market_ticker: str
    signal_type: str           # "entry", "reentry", "exit"
    timestamp: float

    # Signal strength
    confidence: float
    signal_data: Dict[str, Any]  # Strategy-specific (yes_ratio, price_drop, etc.)

    # Decision
    executed: bool
    skip_reason: Optional[str] = None  # "position_limit", "combined_limit", "category_limit", etc.

    # Context at signal time
    context: Dict[str, Any] = field(default_factory=dict)
    # Includes: concurrent_strategies, market_positions, category_exposure, etc.


@dataclass
class ConsensusEvent:
    """
    When multiple strategies fire on same market within window.

    HIGH-CONVICTION signal worth tracking separately.
    """
    event_id: str
    market_ticker: str
    timestamp: float
    strategies: List[str]      # ["RLM_NO", "S013"]
    signal_ids: List[str]      # References to StrategySignal records
    combined_confidence: float # Max or weighted average
```

### 1.2 Risk Limits Configuration

```python
# traderv3/config/risk_limits.py

@dataclass
class RiskLimits:
    """
    Portfolio-level risk constraints.

    Applied BEFORE strategy-specific limits.
    """
    # Per-market limits
    combined_max_contracts_per_market: int = 75   # Total across all strategies
    max_single_market_pct: float = 0.10           # 10% of portfolio value

    # Category limits
    max_category_pct: float = 0.40                # 40% in any single category

    # Strategy-specific (loaded from config)
    strategy_limits: Dict[str, int] = field(default_factory=lambda: {
        "RLM_NO": 50,   # Max contracts per market for RLM
        "S013": 50,     # Max contracts per market for S013
    })

    # Global
    max_total_positions: int = 100                # Total open positions
    min_cash_reserve_pct: float = 0.20            # Keep 20% cash


@dataclass
class RiskCheckResult:
    """Result of pre-trade risk check."""
    allowed: bool
    max_contracts: int         # Maximum allowed given limits
    limiting_factor: str       # Which limit is binding
    details: Dict[str, Any]    # Full breakdown
```

### 1.3 Strategy Position Summary

```python
@dataclass
class StrategyPositionSummary:
    """
    Aggregated view of a strategy's position in a market.

    Computed from individual StrategyEntry records.
    """
    strategy: str
    market_ticker: str

    # Aggregated from entries
    total_contracts: int
    total_cost_basis: int      # Sum of entry cost bases
    entry_count: int           # Number of separate entries

    # Per-entry tracking (for accurate P&L)
    entries: List[str]         # List of entry_ids

    # Computed
    avg_entry_price: float     # For display only, NOT used for P&L calc

    @property
    def unrealized_pnl(self) -> int:
        """Must be computed from individual entries, not avg price."""
        raise NotImplementedError("Use compute_unrealized_pnl(entries, current_price)")
```

---

## 2. StateContainer Extensions

```python
# traderv3/core/state_container.py (additions)

class V3StateContainer:
    def __init__(self):
        # ... existing fields ...

        # Multi-strategy tracking
        self._strategy_entries: Dict[str, StrategyEntry] = {}      # entry_id -> entry
        self._strategy_signals: deque[StrategySignal] = deque(maxlen=500)  # All signals
        self._consensus_events: deque[ConsensusEvent] = deque(maxlen=100)
        self._risk_limits: RiskLimits = RiskLimits()

        # Indices for fast lookup
        self._entries_by_market: Dict[str, List[str]] = defaultdict(list)   # ticker -> [entry_ids]
        self._entries_by_strategy: Dict[str, List[str]] = defaultdict(list) # strategy -> [entry_ids]

    # ===== Entry Management =====

    def record_strategy_entry(self, entry: StrategyEntry) -> None:
        """Record a new strategy entry. Called after successful order."""
        self._strategy_entries[entry.entry_id] = entry
        self._entries_by_market[entry.market_ticker].append(entry.entry_id)
        self._entries_by_strategy[entry.strategy].append(entry.entry_id)
        self._trading_state_version += 1

    def get_strategy_entries(
        self,
        market_ticker: Optional[str] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[StrategyEntry]:
        """Query entries with optional filters."""
        entries = list(self._strategy_entries.values())

        if market_ticker:
            entries = [e for e in entries if e.market_ticker == market_ticker]
        if strategy:
            entries = [e for e in entries if e.strategy == strategy]
        if status:
            entries = [e for e in entries if e.status == status]

        return entries

    def get_strategy_contracts(self, market_ticker: str, strategy: str) -> int:
        """Get filled contracts for strategy in market."""
        entries = self.get_strategy_entries(
            market_ticker=market_ticker,
            strategy=strategy,
            status="filled"
        )
        return sum(e.filled_contracts for e in entries)

    def get_combined_contracts(self, market_ticker: str) -> int:
        """Get total contracts across ALL strategies in market."""
        entries = self.get_strategy_entries(market_ticker=market_ticker, status="filled")
        return sum(e.filled_contracts for e in entries)

    # ===== Signal Logging =====

    def record_signal(self, signal: StrategySignal) -> None:
        """Record ALL signals including skipped ones."""
        self._strategy_signals.append(signal)

        # Check for consensus
        self._check_for_consensus(signal)

    def _check_for_consensus(self, new_signal: StrategySignal, window_seconds: float = 60.0) -> None:
        """Detect if multiple strategies fired on same market recently."""
        recent_cutoff = new_signal.timestamp - window_seconds

        # Find other executed signals on same market within window
        other_signals = [
            s for s in self._strategy_signals
            if s.market_ticker == new_signal.market_ticker
            and s.strategy != new_signal.strategy
            and s.timestamp > recent_cutoff
            and s.executed
        ]

        if other_signals and new_signal.executed:
            all_strategies = list(set([new_signal.strategy] + [s.strategy for s in other_signals]))
            all_signal_ids = [new_signal.signal_id] + [s.signal_id for s in other_signals]

            consensus = ConsensusEvent(
                event_id=f"consensus:{new_signal.market_ticker}:{int(new_signal.timestamp * 1000)}",
                market_ticker=new_signal.market_ticker,
                timestamp=new_signal.timestamp,
                strategies=all_strategies,
                signal_ids=all_signal_ids,
                combined_confidence=max(new_signal.confidence, max(s.confidence for s in other_signals))
            )
            self._consensus_events.append(consensus)

            logger.info(
                f"CONSENSUS: {new_signal.market_ticker} - "
                f"strategies={all_strategies}, confidence={consensus.combined_confidence:.2f}"
            )

    # ===== Risk Checks =====

    def check_risk_limits(
        self,
        market_ticker: str,
        strategy: str,
        requested_contracts: int,
        category: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Pre-trade risk check against all limits.

        Returns max allowed contracts and limiting factor.
        """
        limits = self._risk_limits
        trading_state = self._trading_state

        if not trading_state:
            return RiskCheckResult(allowed=False, max_contracts=0,
                                   limiting_factor="no_trading_state", details={})

        portfolio_value = trading_state.balance + trading_state.portfolio_value

        # 1. Strategy-specific limit
        current_strategy = self.get_strategy_contracts(market_ticker, strategy)
        strategy_limit = limits.strategy_limits.get(strategy, 50)
        strategy_remaining = strategy_limit - current_strategy

        # 2. Combined market limit
        current_combined = self.get_combined_contracts(market_ticker)
        combined_remaining = limits.combined_max_contracts_per_market - current_combined

        # 3. Single market % limit (approximate, using entry price)
        # Assumes ~50c per contract for estimation
        estimated_cost = requested_contracts * 50
        max_market_value = int(portfolio_value * limits.max_single_market_pct)
        current_market_value = self._get_market_exposure(market_ticker)
        market_pct_remaining = (max_market_value - current_market_value) // 50

        # 4. Category limit
        category_remaining = float('inf')
        if category:
            max_category_value = int(portfolio_value * limits.max_category_pct)
            current_category_value = self._get_category_exposure(category)
            category_remaining = (max_category_value - current_category_value) // 50

        # Find binding constraint
        max_contracts = min(
            strategy_remaining,
            combined_remaining,
            market_pct_remaining,
            category_remaining,
            requested_contracts
        )

        # Determine limiting factor
        if max_contracts <= 0:
            if strategy_remaining <= 0:
                limiting = "strategy_limit"
            elif combined_remaining <= 0:
                limiting = "combined_limit"
            elif market_pct_remaining <= 0:
                limiting = "market_pct_limit"
            elif category_remaining <= 0:
                limiting = "category_limit"
            else:
                limiting = "unknown"
        elif max_contracts < requested_contracts:
            limiting = "partial_fill_allowed"
        else:
            limiting = "none"

        return RiskCheckResult(
            allowed=max_contracts > 0,
            max_contracts=max(0, int(max_contracts)),
            limiting_factor=limiting,
            details={
                "strategy_remaining": strategy_remaining,
                "combined_remaining": combined_remaining,
                "market_pct_remaining": market_pct_remaining,
                "category_remaining": category_remaining if category else None,
                "portfolio_value": portfolio_value,
            }
        )

    def _get_market_exposure(self, market_ticker: str) -> int:
        """Get current value exposure to a market (cents)."""
        entries = self.get_strategy_entries(market_ticker=market_ticker, status="filled")
        return sum(e.cost_basis for e in entries)

    def _get_category_exposure(self, category: str) -> int:
        """Get current value exposure to a category (cents)."""
        # Requires category lookup from tracked_markets_state
        # TODO: Implement with category mapping
        return 0

    # ===== P&L Computation =====

    def compute_strategy_pnl(self, strategy: str, current_prices: Dict[str, int]) -> Dict[str, Any]:
        """
        Compute P&L for a strategy from individual entries.

        CRITICAL: Uses per-entry prices, NOT averaged.
        """
        entries = self.get_strategy_entries(strategy=strategy, status="filled")

        total_cost_basis = 0
        total_contracts = 0
        unrealized_pnl = 0
        realized_pnl = 0

        for entry in entries:
            total_cost_basis += entry.cost_basis
            total_contracts += entry.filled_contracts

            if entry.settled_at:
                # Settled - use actual settlement
                realized_pnl += entry.realized_pnl or 0
            else:
                # Open - compute unrealized from THIS entry's cost basis
                current_price = current_prices.get(entry.market_ticker, 50)
                # For NO position, profit if current_price drops
                if entry.side == "no":
                    current_value = entry.filled_contracts * (100 - current_price)
                else:
                    current_value = entry.filled_contracts * current_price
                entry_unrealized = current_value - entry.cost_basis
                unrealized_pnl += entry_unrealized

        return {
            "strategy": strategy,
            "total_contracts": total_contracts,
            "total_cost_basis": total_cost_basis,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "net_pnl": realized_pnl + unrealized_pnl,
            "entry_count": len(entries),
            "roi_pct": ((realized_pnl + unrealized_pnl) / total_cost_basis * 100) if total_cost_basis > 0 else 0,
        }
```

---

## 3. Strategy Service Pattern

### 3.1 Base Pattern (RLMService update)

```python
# In RLMService._execute_signal() - key changes

async def _execute_signal(self, signal: RLMSignal) -> None:
    """Execute signal with full tracking."""

    signal_id = f"RLM_NO:{signal.market_ticker}:{int(signal.detected_at * 1000)}"

    # Build context for signal logging
    context = {
        "concurrent_strategies": self._get_active_strategies(),
        "rlm_position": self._state_container.get_strategy_contracts(signal.market_ticker, "RLM_NO"),
        "combined_position": self._state_container.get_combined_contracts(signal.market_ticker),
        "category": self._get_market_category(signal.market_ticker),
    }

    # Check risk limits FIRST
    risk_check = self._state_container.check_risk_limits(
        market_ticker=signal.market_ticker,
        strategy="RLM_NO",
        requested_contracts=self._contracts_per_trade,
        category=context.get("category"),
    )

    # Record signal (even if skipped)
    strategy_signal = StrategySignal(
        signal_id=signal_id,
        strategy="RLM_NO",
        market_ticker=signal.market_ticker,
        signal_type="reentry" if signal.is_reentry else "entry",
        timestamp=signal.detected_at,
        confidence=min(0.9, 0.7 + signal.price_drop * 0.01),
        signal_data=signal.to_dict(),
        executed=False,  # Updated if executed
        skip_reason=None,
        context=context,
    )

    if not risk_check.allowed:
        strategy_signal.skip_reason = risk_check.limiting_factor
        self._state_container.record_signal(strategy_signal)
        self._stats["signals_skipped"] += 1
        logger.info(f"RLM signal skipped: {signal.market_ticker} - {risk_check.limiting_factor}")
        return

    # Adjust contracts if partially limited
    actual_contracts = min(self._contracts_per_trade, risk_check.max_contracts)

    # Get entry price from orderbook
    entry_price = await self._get_entry_price(signal.market_ticker)
    if entry_price is None:
        strategy_signal.skip_reason = "no_orderbook"
        self._state_container.record_signal(strategy_signal)
        return

    # Create trading decision
    decision = TradingDecision(
        action="buy",
        market=signal.market_ticker,
        side="no",
        quantity=actual_contracts,
        price=entry_price,
        reason=f"RLM:{signal_id}",  # Embed signal_id for tracking
        confidence=strategy_signal.confidence,
        strategy=TradingStrategy.RLM_NO,
    )

    # Execute
    success = await self._trading_service.execute_decision(decision)

    if success:
        strategy_signal.executed = True
        self._state_container.record_signal(strategy_signal)

        # Record entry (order_id comes from execute_decision response)
        order_id = self._trading_service.last_order_id  # Need to expose this
        entry = StrategyEntry(
            entry_id=f"RLM_NO:{order_id}",
            order_id=order_id,
            strategy="RLM_NO",
            market_ticker=signal.market_ticker,
            side="no",
            contracts=actual_contracts,
            entry_price=entry_price,
            cost_basis=actual_contracts * entry_price,
            created_at=time.time(),
            signal_id=signal_id,
            status="pending",  # Updated on fill
        )
        self._state_container.record_strategy_entry(entry)

        self._stats["signals_executed"] += 1
    else:
        strategy_signal.skip_reason = "execution_failed"
        self._state_container.record_signal(strategy_signal)
```

### 3.2 TradingDecisionService Update

```python
# In TradingDecisionService._execute_buy() - expose order_id

async def _execute_buy(self, decision: TradingDecision) -> bool:
    """Execute buy with order_id tracking."""
    try:
        response = await self._trading_client.place_order(...)

        order_id = response.get("order", {}).get("order_id", "unknown")

        # Expose for strategy services
        self._last_order_id = order_id
        self._last_order_response = response

        # ... rest of method ...

        return True
    except Exception as e:
        self._last_order_id = None
        return False

@property
def last_order_id(self) -> Optional[str]:
    """Get order_id from most recent execution."""
    return getattr(self, '_last_order_id', None)
```

---

## 4. Event Bus Extensions

```python
# traderv3/core/event_bus.py (new event types)

class EventType(Enum):
    # ... existing ...

    # Multi-strategy events
    STRATEGY_SIGNAL = "strategy_signal"           # All signals (executed or skipped)
    STRATEGY_ENTRY_CREATED = "strategy_entry"     # New entry recorded
    STRATEGY_CONSENSUS = "strategy_consensus"     # Multiple strategies fired
    RISK_LIMIT_HIT = "risk_limit_hit"            # Position limit reached
```

---

## 5. WebSocket Status Extensions

```python
# In get_trading_summary() additions

def get_trading_summary(self, order_group_id: Optional[str] = None) -> Dict[str, Any]:
    summary = {
        # ... existing fields ...

        # Multi-strategy P&L (NEW)
        "strategy_pnl": {
            strategy: self.compute_strategy_pnl(strategy, current_prices)
            for strategy in ["RLM_NO", "S013"]
        },

        # Recent signals (NEW)
        "recent_signals": [s.__dict__ for s in list(self._strategy_signals)[-20:]],

        # Consensus events (NEW)
        "consensus_events": [c.__dict__ for c in list(self._consensus_events)[-10:]],

        # Risk status (NEW)
        "risk_status": {
            "markets_at_combined_limit": self._count_markets_at_limit(),
            "categories_near_limit": self._get_categories_near_limit(),
        },
    }
    return summary
```

---

## 6. Implementation Order

### Phase 1: Data Structures (Day 1)
1. Create `traderv3/state/strategy_tracking.py` with all dataclasses
2. Create `traderv3/config/risk_limits.py`
3. Add fields to `StateContainer.__init__()`

### Phase 2: StateContainer Methods (Day 1-2)
1. Implement entry management methods
2. Implement signal logging with consensus detection
3. Implement risk check method
4. Implement P&L computation

### Phase 3: RLMService Integration (Day 2)
1. Update `_execute_signal()` with full tracking
2. Add skip reason logging
3. Add context capture
4. Test single-strategy P&L tracking

### Phase 4: S013Service (Day 3)
1. Create following RLM pattern
2. Wire into coordinator
3. Test consensus detection
4. Test combined risk limits

### Phase 5: Frontend/WebSocket (Day 4)
1. Extend status messages
2. Add strategy P&L breakdown
3. Add signal history display
4. Add consensus event highlighting

---

## 7. Key Invariants

1. **NEVER blend entry prices** - Each StrategyEntry has its own entry_price
2. **ALL signals logged** - Including skipped ones with skip_reason
3. **Risk checks BEFORE execution** - check_risk_limits() called before place_order()
4. **Consensus is HIGH-CONVICTION** - Log and potentially increase position size
5. **P&L from entries** - Always compute from individual StrategyEntry records
