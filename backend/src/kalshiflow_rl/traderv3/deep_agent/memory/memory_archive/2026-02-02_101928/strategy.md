<!-- version:17 updated:2026-02-02T10:18 -->
# Trading Strategy

## CRITICAL EXECUTION RULES
**EX1: Mandatory Execution Verification** - After every trade() call, wait minimum 10 seconds then call get_true_performance() to verify position creation. If position not created, investigate before next trade.

**EX2: Cycle Time Minimum** - Minimum 45 seconds between trade decisions. High-frequency cycles (>3/minute) cause execution failure rates >90%.

**EX3: One Trade Per Cycle Rule** - Maximum one trade() call per cycle. Multiple rapid trades overwhelm order processing.

**EX4: Position Tracking Halt Rule** - When position tracking shows impossible data (1820c prices, phantom positions), halt trading until systems stabilize. Cannot manage risk with unreliable position data.

**EX5: System Crisis Backup Protocol** - During extended system failures (5+ cycles), identify signals in events with NO existing positions for minimal speculative trades. Prioritize system integrity over anti-stagnation pressure.

## Entry Rules
**E1: Signal Threshold** - Enter only with extraction signals showing magnitude >30 AND engagement >5000, OR coordinated whale activity (100% flow + 2x normal volume).

**E2: Information Asymmetry Gap** - Heavy whale activity without extraction signals suggests breaking news or institutional edge. Enter when microstructure shows 100% flow + escalating trade counts (2x+ normal) across multiple related tickers.

**E3: Event-Wide Whale Patterns** - When whale activity expands across entire event (7+ tickers) with perfect directional consensus and 1000+ trades per ticker, this indicates major unreported news. Enter positions across multiple related markets.

**E4: Signal Persistence** - Signals persisting across 2+ cycles indicate sustained narrative momentum. Increase conviction on persistent signals.

**E5: Fill Aggregation Check** - Verify fills aggregate correctly into position tracking. When recent_fills show trades but positions don't update, trading systems are compromised.

**E6: Signal Stability Warning** - When identical signals persist >4 cycles without growth, suspect signal decay disguised as stability. Reduce new entries but maintain existing positions.

## Position Sizing
**S1: Signal Strength** - $25-50 for single extraction signal, $50-75 for extraction + microstructure alignment.

**S2: Event Diversification** - Max 3 positions per event to prevent overconcentration. Avoid 4+ positions in single event resolving same day.

## Data Quality Checks
**D1: Microstructure Staleness** - When identical microstructure patterns persist >5 cycles (same trade counts, ratios, whale activity), suspect data staleness. Reduce conviction in affected signals.

**D2: Position Reconciliation** - Compare true_performance() vs session_state positions at session start. Discrepancies >2 positions or impossible pricing trigger EX4 halt.

## Golden Rules (MUST preserve in strategy)
# Golden Rules
