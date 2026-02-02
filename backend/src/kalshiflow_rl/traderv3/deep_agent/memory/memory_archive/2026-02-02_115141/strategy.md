<!-- version:24 updated:2026-02-02T11:51 -->
# Trading Strategy

## CRITICAL EXECUTION RULES
**EX1: Mandatory Execution Verification** - After every trade() call, wait minimum 10 seconds then call get_true_performance() to verify position creation. If position not created, investigate before next trade.

**EX2: Cycle Time Minimum** - Minimum 45 seconds between trade decisions. High-frequency cycles (>3/minute) cause execution failure rates >90%.

**EX3: One Trade Per Cycle Rule** - Maximum one trade() call per cycle. Multiple rapid trades overwhelm order processing.

**EX4: Position Tracking Halt Rule** - When position tracking shows impossible data (impossible prices >100c, phantom positions), halt trading until systems stabilize. Cannot manage risk with unreliable position data.

**EX5: System Crisis Backup Protocol** - During extended system failures (5+ cycles), identify signals in events with NO existing positions for minimal speculative trades. Prioritize system integrity over anti-stagnation pressure.

**EX6: Anti-Stagnation Override** - After 8+ consecutive cycles without trade, make one speculative $25 position on strongest available signal to maintain system engagement, unless EX4/EX8 halt is active.

**EX7: Data Staleness Detection** - When identical microstructure patterns persist >5 cycles (same trade counts, whale activity), suspect data staleness. Reduce conviction in stale patterns by 50%. Patterns persisting >8 cycles are likely frozen data.

**EX8: Persistent System Failure Protocol** - When position tracking APIs show consistent disconnect across entire session (5+ cycles), maintain trading halt regardless of signal quality. No edge justifies trading with broken risk management infrastructure. VALIDATED: 26-cycle session with API disconnect - conservative monitoring preserved capital during system instability.

**EX9: Infrastructure-First Trading** - System reliability is foundational to edge execution. Portfolio performance during stable periods validates extraction signal accuracy (+61.7% on TRUMPSAY), but infrastructure failures negate any edge advantage. Always prioritize system stability over profit maximization.
