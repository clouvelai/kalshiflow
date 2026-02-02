<!-- version:22 updated:2026-02-02T11:11 -->
# Trading Strategy

## CRITICAL EXECUTION RULES
**EX1: Mandatory Execution Verification** - After every trade() call, wait minimum 10 seconds then call get_true_performance() to verify position creation. If position not created, investigate before next trade.

**EX2: Cycle Time Minimum** - Minimum 45 seconds between trade decisions. High-frequency cycles (>3/minute) cause execution failure rates >90%.

**EX3: One Trade Per Cycle Rule** - Maximum one trade() call per cycle. Multiple rapid trades overwhelm order processing.

**EX4: Position Tracking Halt Rule** - When position tracking shows impossible data (impossible prices >100c, phantom positions), halt trading until systems stabilize. Cannot manage risk with unreliable position data.

**EX5: System Crisis Backup Protocol** - During extended system failures (5+ cycles), identify signals in events with NO existing positions for minimal speculative trades. Prioritize system integrity over anti-stagnation pressure.

**EX6: Anti-Stagnation Override** - After 8+ consecutive cycles without trade, make one speculative $25 position on strongest available signal to maintain system engagement, unless EX4 halt is active.

**EX7: Data Staleness Detection** - When identical microstructure patterns persist >5 cycles (same trade counts, whale activity), suspect data staleness. Reduce conviction in stale patterns by 50%.

**EX8: Persistent System Failure Protocol** - When position tracking APIs show consistent disconnect across entire session (5+ cycles), maintain trading halt regardless of signal quality. No edge justifies trading with broken risk management infrastructure.

## Entry Rules
**E1: Signal Threshold** - Enter only with extraction signals showing magnitude >30 AND engagement >5000, OR coordinated whale activity (100% flow + 2x normal volume).

**E2: Information Asymmetry Gap** - Heavy whale activity without extraction signals suggests breaking news or institutional edge. Enter when microstructure shows 100% flow + escalating trade counts (2x+ normal) across multiple related tickers.

**E3: Event-Wide Whale Patterns** - When whale activity expands across entire event (7+ tickers) with perfect directional consensus and 1000+ trades per ticker, this indicates major unreported news. Enter positions across multiple related markets.

**E4: Whale Escalation Pattern** - When whale activity escalates from 1T → 2T → 4T per ticker with sustained directional consensus, this suggests time-critical institutional information. Increase conviction but verify position tracking before trading.

## Position Sizing
**S1: Signal Strength** - $25-50 for single extraction signal, $50-75 for extraction + microstructure alignment.

**S2: Event Diversification** - Max 3 positions per event to prevent overconcentration.