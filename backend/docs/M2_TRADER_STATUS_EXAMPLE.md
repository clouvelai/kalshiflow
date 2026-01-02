# M2 Trader Status Console - Example Output

This document shows what the trader status console will look like at the end of M2. This is what you would see when viewing the Trader Status component in the UI.

---

## Simple Overview: Actor + Trading Loop Flow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADER LIFECYCLE                          │
└─────────────────────────────────────────────────────────────┘

STARTUP → [initializing] → [trading] ←┐
                                      │
                                      │ (every 60s)
                                      │
                ┌─────────────────────┘
                │
                │ State transition: trading → calibrating
                │
                ▼
        [calibrating] (4 steps, < 5s total)
                │
                │ 1. syncing state
                │ 2. closing positions
                │ 3. monitoring markets
                │ 4. cash recovery (if needed)
                │
                │ State transition: calibrating → trading
                │
                └─────────────────────┐
                                      │
                                      ▼
                                [trading]
```

### Trading State: Event Processing Loop

```
Event Queue → [trading] state active
    │
    ├─→ Event 1: INXD-25JAN03 delta
    │   ├─ build_observation() → 52 features
    │   ├─ select_action() → BUY_YES_LIMIT (action 1)
    │   ├─ execute_action() → Order placed
    │   └─ update_positions() → Portfolio updated
    │
    ├─→ Event 2: INXD-25JAN03 delta
    │   ├─ build_observation()
    │   ├─ select_action() → HOLD (action 0)
    │   └─ (no order placed)
    │
    └─→ Event 3: INXD-25JAN03 delta
        ├─ build_observation()
        ├─ select_action() → SELL_YES_LIMIT (action 2)
        ├─ execute_action() → Order placed
        └─ update_positions()
```

**Key Point:** Events are processed serially, one at a time, through the 4-step pipeline.

### Calibration State: Recalibration Loop

```
Every 60s: State transition trading → calibrating

[calibrating] state active
    │
    ├─→ Step 1: calibrating -> syncing state
    │   ├─ Sync orders with Kalshi API
    │   ├─ Sync positions with Kalshi API
    │   ├─ Sync cash balance
    │   └─ Duration: ~1.2s
    │
    ├─→ Step 2: calibrating -> closing positions
    │   ├─ Check position health (P&L, time)
    │   ├─ Close positions meeting criteria
    │   └─ Duration: ~0.8s
    │
    ├─→ Step 3: calibrating -> monitoring markets
    │   ├─ Check market states
    │   ├─ Close positions in closing markets
    │   └─ Duration: ~0.5s
    │
    └─→ Step 4: calibrating -> cash recovery (if needed)
        ├─ Check cash vs reserve
        ├─ Close worst positions if low
        └─ Duration: ~0.3s (or skipped if not needed)

Total Duration: ~2.8s (under 5s target)

State transition: calibrating → trading (resume event processing)
```

**Key Point:** During calibration, event processing is paused. Events queue up and resume after calibration completes.

---

## Example: Trader Status Console Output

Here's what the trader status console would look like during normal operation:

```
═══════════════════════════════════════════════════════════════════
🔄 TRADER STATUS
═══════════════════════════════════════════════════════════════════

Current Status: trading
───────────────────────────────────────────────────────────────────

Status History (23 entries) [📋 Copy]

03:45:12 PM  trading
03:45:02 PM  calibrating -> cash recovery       cash $1,250.00 -> $1,250.00 (no action needed) (0.1s)
03:45:02 PM  calibrating -> monitoring markets  no markets closing (0.4s)
03:45:02 PM  calibrating -> closing positions   no positions to close (3 active) (0.7s)
03:45:02 PM  calibrating -> syncing state       no changes (1.1s)
03:45:02 PM  calibrating                        starting recalibration
03:44:12 PM  trading
03:44:02 PM  calibrating -> cash recovery       cash $1,245.00 -> $1,245.00 (no action needed) (0.1s)
03:44:02 PM  calibrating -> monitoring markets  no markets closing (0.4s)
03:44:02 PM  calibrating -> closing positions   closed 1 (3 -> 2) +$12.50 P&L | 1 active positions | 1 above profit threshold: closed 1 -> +$12.50 P&L (0.9s)
03:44:02 PM  calibrating -> syncing state       cash $1,232.50 -> $1,245.00, portfolio $125.00 -> $112.50 (1.2s)
03:44:02 PM  calibrating                        starting recalibration
03:43:12 PM  trading
03:43:05 PM  trading                            last action: BUY_YES_LIMIT @ INXD-25JAN03
03:42:58 PM  trading                            last action: HOLD @ INXD-25JAN03
03:42:51 PM  trading                            last action: SELL_YES_LIMIT @ INXD-25JAN03
03:42:45 PM  trading                            last action: BUY_NO_LIMIT @ INXD-25JAN03
03:42:38 PM  trading                            last action: HOLD @ INXD-25JAN03
03:42:02 PM  calibrating -> cash recovery       cash $1,220.00 -> $1,232.50 (no action needed) (0.2s)
03:42:02 PM  calibrating -> monitoring markets  no markets closing (0.5s)
03:42:02 PM  calibrating -> closing positions   no positions to close (2 active) (0.8s)
03:42:02 PM  calibrating -> syncing state       no changes (1.0s)
03:42:02 PM  calibrating                        starting recalibration
03:41:12 PM  trading
03:41:05 PM  trading                            last action: BUY_YES_LIMIT @ INXD-25JAN03
03:40:58 PM  trading                            last action: HOLD @ INXD-25JAN03
```

### Example: During Active Calibration

```
═══════════════════════════════════════════════════════════════════
🔄 TRADER STATUS
═══════════════════════════════════════════════════════════════════

Current Status: calibrating -> closing positions
───────────────────────────────────────────────────────────────────

Status History (25 entries) [📋 Copy]

03:45:15 PM  calibrating -> closing positions   checking position health... (in progress)
03:45:14 PM  calibrating -> syncing state       cash $1,250.00 -> $1,250.00, portfolio $125.00 -> $125.00 (1.2s)
03:45:14 PM  calibrating                        starting recalibration
03:45:12 PM  trading
03:45:02 PM  calibrating -> cash recovery       cash $1,245.00 -> $1,250.00 (no action needed) (0.1s)
03:45:02 PM  calibrating -> monitoring markets  no markets closing (0.4s)
03:45:02 PM  calibrating -> closing positions   no positions to close (3 active) (0.7s)
03:45:02 PM  calibrating -> syncing state       no changes (1.1s)
03:45:02 PM  calibrating                        starting recalibration
```

### Example: After Position Closure

```
═══════════════════════════════════════════════════════════════════
🔄 TRADER STATUS
═══════════════════════════════════════════════════════════════════

Current Status: trading
───────────────────────────────────────────────────────────────────

Status History (26 entries) [📋 Copy]

03:45:18 PM  trading
03:45:17 PM  calibrating -> cash recovery       cash $1,250.00 -> $1,250.00 (no action needed) (0.1s)
03:45:16 PM  calibrating -> monitoring markets  no markets closing (0.4s)
03:45:16 PM  calibrating -> closing positions   closed 2 (4 -> 2) +$25.30 P&L | 4 active positions | 2 above profit threshold: closed 2 -> +$25.30 P&L (1.2s)
03:45:15 PM  calibrating -> syncing state       cash $1,224.70 -> $1,250.00, portfolio $150.00 -> $125.00, positions 4 -> 2 (1.3s)
03:45:15 PM  calibrating                        starting recalibration
03:45:12 PM  trading
```

### Example: Error State (Paused)

```
═══════════════════════════════════════════════════════════════════
🔄 TRADER STATUS
═══════════════════════════════════════════════════════════════════

Current Status: paused
───────────────────────────────────────────────────────────────────

Status History (30 entries) [📋 Copy]

03:45:25 PM  paused                             error: Kalshi API timeout during position sync
03:45:24 PM  calibrating -> syncing state       error: Request timeout (2.0s)
03:45:24 PM  calibrating                        starting recalibration
03:45:22 PM  trading
03:45:12 PM  trading
```

---

## Key Behaviors Visible in Status

### 1. **State Transitions**
- Clear state names: `trading`, `calibrating`, `paused`, `stopping`
- Sub-states for calibration steps: `calibrating -> syncing state`
- Every transition logged with timestamp

### 2. **Calibration Progress**
- Each calibration step shown with result
- Duration tracked for each step
- Total calibration time visible in history

### 3. **Position Closing Details**
- Shows positions closed count: `closed 2 (4 -> 2)`
- Shows P&L: `+$25.30 P&L`
- Shows closing reasons: `above profit threshold`

### 4. **State Changes**
- Cash changes: `cash $1,224.70 -> $1,250.00`
- Portfolio changes: `portfolio $150.00 -> $125.00`
- Position count changes: `positions 4 -> 2`

### 5. **Trading Activity**
- Last action shown during trading: `last action: BUY_YES_LIMIT @ INXD-25JAN03`
- Action frequency visible in history

### 6. **Performance Metrics**
- Calibration duration: `(1.2s)`, `(2.8s)`
- Individual step durations tracked
- Total duration visible for each calibration cycle

---

## What This Tells Us

1. **Current State:** Always know what the trader is doing right now
2. **Calibration Frequency:** See calibrations every 60s (or configured interval)
3. **Calibration Speed:** See that calibrations are under 5s (target met)
4. **Position Management:** See when positions are closed and why
5. **State Changes:** See cash, portfolio, and position changes during sync
6. **Trading Activity:** See recent trading actions and frequency
7. **Error Handling:** See when errors occur and how state transitions to `paused`

---

## Implementation Notes

- **State is mutually exclusive:** Only one state active at a time
- **History is scrollable:** Last 20 entries shown, full history (50) available
- **Copy functionality:** Click clipboard icon to copy full history for debugging
- **Real-time updates:** Status updates via WebSocket as state changes
- **Duration tracking:** Every status update with duration shows execution time
- **Result messages:** Each step includes result summary (what happened, counts, changes)
