---
name: iterate-captain
description: Automated improvement loop for the Captain single-arb strategy. Starts the system, monitors logs + browser for issues, implements fixes, simplifies code, and validates. Use when you want to systematically find and fix issues in the single-event arb system.
argument-hint: [cycles]
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, TaskCreate, TaskUpdate, TaskList, AskUserQuestion
---

# Iterate Captain - Automated Improvement Loop

You are running an automated improvement loop on the Captain single-event arb system. Each iteration:
1. Start the system (run-v3.sh)
2. Observe via logs + browser (in parallel)
3. Stop after target cycles
4. Fix the top issues
5. Simplify changed code
6. Restart and validate

**Target cycles per iteration**: `$ARGUMENTS` (default: 3)

---

## Key System Info

- **Backend**: Port 8005
- **Frontend**: http://localhost:5173/arb
- **Health**: http://localhost:8005/v3/health
- **Status**: http://localhost:8005/v3/status
- **Log file**: backend/logs/v3-trader.log
- **Log markers**: `[SINGLE_ARB:*]` prefixes

### Frontend Data-TestID Reference

The frontend has been optimized with data-testid attributes for automation:

- `data-testid="arb-dashboard"` - Main dashboard container
- `data-testid="arb-header"` - Top header bar
- `data-testid="connection-status"` with `data-connected` attribute
- `data-testid="system-state"` with `data-state` attribute
- `data-testid="agent-panel"` with `data-running` attribute
- `data-testid="agent-cycle-count"` - Shows current cycle number
- `data-testid="thinking-stream"` - Agent thinking/reasoning output
- `data-testid="tool-calls-section"` with `data-count` attribute
- `data-testid="trades-section"` with `data-count` attribute
- `data-testid="event-index-panel"` with `data-event-count`, `data-market-count`
- `data-testid="event-row-{ticker}"` with `data-edge`, `data-edge-direction`
- `data-testid="metric-balance"`, `data-testid="metric-pnl"`, etc.

---

## Step 1: Setup & Start System

```
TARGET_CYCLES = $ARGUMENTS or 3
LOG_FILE = backend/logs/v3-trader.log
ITERATION = 1
```

### 1a. Kill existing process
```bash
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
sleep 2
```

### 1b. Clear log file
```bash
> backend/logs/v3-trader.log
```

### 1c. Start system in background
```bash
cd /Users/samuelclark/Desktop/kalshiflow
./scripts/run-v3.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use the Bash tool with `run_in_background: true` for this command.

### 1d. Wait for startup
Poll until you see `[SINGLE_ARB:STARTUP]` in the log file (check every 5s, timeout 90s):
```bash
for i in $(seq 1 18); do
  grep -q "SINGLE_ARB:STARTUP\|SINGLE_ARB:CAPTAIN_START" backend/logs/v3-trader.log 2>/dev/null && echo "READY" && break
  sleep 5
done
```

If the system doesn't start, read the log file to diagnose and fix the startup error before proceeding.

---

## Step 2: Observe (Parallel Agents)

Launch TWO background agents simultaneously using a single message with multiple Task tool calls:

### 2a. Log Monitor Agent
Use the Task tool with `subagent_type: "general-purpose"` and `run_in_background: true`:

```
Prompt: "You are monitoring Captain single-arb logs for issues and improvements.

LOG FILE: /Users/samuelclark/Desktop/kalshiflow/backend/logs/v3-trader.log

KEY LOG MARKERS TO WATCH:
- [SINGLE_ARB:STARTUP] - System startup
- [SINGLE_ARB:SHUTDOWN] - System shutdown
- [SINGLE_ARB:CAPTAIN_START] - Captain agent started
- [SINGLE_ARB:CAPTAIN_STOP] - Captain agent stopped
- [SINGLE_ARB:CYCLE_START] cycle=N - Captain cycle began
- [SINGLE_ARB:CYCLE_END] cycle=N duration=Xs - Captain cycle completed
- [SINGLE_ARB:CAPTAIN_ERROR] - Captain encountered an error
- [SINGLE_ARB:SUBAGENT_START] subagent=X - TradeCommando/ChevalDeTroie invoked
- [SINGLE_ARB:SUBAGENT_COMPLETE] subagent=X - Subagent finished
- [SINGLE_ARB:TRADE] - Trade order placed
- [SINGLE_ARB:TRADE_ERROR] - Trade failed
- [SINGLE_ARB:TRADE_RESULT] - Trade execution result

Your job:
1. Read the log file every 30 seconds (use Read tool with offset to only read new lines)
2. Track how many captain cycles have completed by counting '[SINGLE_ARB:CYCLE_END]' markers
3. Collect ALL of the following into a structured list:
   - ERRORS: Any ERROR, Exception, or [SINGLE_ARB:*_ERROR] lines with full context
   - WARNINGS: Any WARNING lines that indicate real problems
   - TRADE_ISSUES: Failed trades, aborted arbs, cancelled orders with reasons
   - AGENT_ISSUES: Captain failures, subagent errors, tool timeouts
   - PERFORMANCE: Slow cycles (>120s), connection issues, queue drops
   - IMPROVEMENTS: Patterns you notice that could be improved

4. After TARGET_CYCLES captain cycles (look for [SINGLE_ARB:CYCLE_END] count), OR if you've found 15+ issues, write your findings to:
   /Users/samuelclark/Desktop/kalshiflow/backend/logs/iteration-issues-logs.md

Format the output as:
# Log Monitor Issues - Iteration N

## Cycle Summary
- Cycles completed: X
- Total errors: X
- Total warnings: X
- Trades attempted: X
- Subagent invocations: X

## Critical Issues (must fix)
1. [ERROR] Description - file:line if identifiable
   Context: ...

## Warnings (should fix)
1. [WARNING] Description
   Context: ...

## Trade Issues
1. Description
   Context: ...

## Improvement Opportunities
1. Description
   Suggestion: ...

Then stop monitoring."
```

### 2b. Browser Validation Agent
Use the Task tool with `subagent_type: "puppeteer-e2e-validator"` and `run_in_background: true`:

```
Prompt: "Validate the Captain single-arb frontend at http://localhost:5173/arb

Wait 30 seconds for the system to initialize, then:

1. SCREENSHOTS: Take screenshots of each major section:
   - Full page overview
   - Agent panel (data-testid='agent-panel') - shows captain thinking/tool calls
   - Event index (data-testid='event-index-panel') - shows events and edge calculations
   - Metrics bar (data-testid='arb-metrics-bar') - balance, P&L, trade counts

2. CHECK these things using data-testid selectors:
   - Connection status: data-testid='connection-status' should have data-connected='true'
   - System state: data-testid='system-state' should show 'READY' or 'TRADING'
   - Agent running: data-testid='agent-panel' should have data-running='true' during cycles
   - Cycle count: data-testid='agent-cycle-count' should increment
   - Events loaded: data-testid='event-index-panel' should have data-event-count > 0
   - Edge calculations: Check for event rows with data-edge values

3. VERIFY DATA FLOW:
   - Agent thinking stream (data-testid='thinking-stream') should show content
   - Tool calls section (data-testid='tool-calls-section') should have data-count > 0
   - Balance metric (data-testid='metric-balance') should show a value

4. WAIT and re-check after 2 minutes:
   - Has the cycle count increased?
   - Are new tool calls appearing?
   - Any UI freezes or stale data?

5. Write findings to:
   /Users/samuelclark/Desktop/kalshiflow/backend/logs/iteration-issues-browser.md

Format:
# Browser Validation Issues - Iteration N

## Connection & State
- WebSocket: [Connected/Disconnected]
- System State: [state value]
- Agent Running: [Yes/No]
- Cycles Observed: X

## UI Issues
1. [SEVERITY] Description
   Screenshot: path
   Element: data-testid

## Data Flow Issues
1. Description

## Working Correctly
1. What's working fine

Then stop."
```

---

## Step 3: Wait for Completion

Poll until both agents have written their issue files:
```bash
for i in $(seq 1 60); do
  LOGS_DONE=$(test -f backend/logs/iteration-issues-logs.md && echo 1 || echo 0)
  BROWSER_DONE=$(test -f backend/logs/iteration-issues-browser.md && echo 1 || echo 0)
  if [ "$LOGS_DONE" = "1" ] && [ "$BROWSER_DONE" = "1" ]; then
    echo "Both agents done"
    break
  fi
  # Also check if enough cycles have passed
  CYCLES=$(grep -c 'SINGLE_ARB:CYCLE_END' backend/logs/v3-trader.log 2>/dev/null || echo 0)
  echo "Waiting... cycles=$CYCLES logs_done=$LOGS_DONE browser_done=$BROWSER_DONE"
  sleep 15
done
```

If the log monitor agent hasn't finished but enough cycles have passed, you can proceed with whatever issues have been collected so far.

---

## Step 4: Stop System & Collect Issues

### 4a. Stop system
```bash
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
sleep 2
```

### 4b. Read and merge issue lists
Read both issue files:
- `backend/logs/iteration-issues-logs.md`
- `backend/logs/iteration-issues-browser.md`

Merge into a single prioritized list:
1. **Critical** (errors, crashes, data corruption, failed trades)
2. **High** (agent failures, broken UI, missing data, edge miscalculations)
3. **Medium** (warnings, performance, UX issues)
4. **Low** (improvements, cosmetics)

---

## Step 5: Implement Fixes

For each issue starting from highest priority:

1. **Locate the source**: Use Grep/Glob to find the relevant code
   - Captain logic: `backend/src/kalshiflow_rl/traderv3/single_arb/captain.py`
   - Coordinator: `backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py`
   - Tools: `backend/src/kalshiflow_rl/traderv3/single_arb/tools.py`
   - Index/Monitor: `backend/src/kalshiflow_rl/traderv3/single_arb/index.py`, `monitor.py`
   - Frontend: `frontend/src/components/arb/`

2. **Understand the root cause**: Read the file and surrounding context

3. **Implement the fix**: Use Edit tool for minimal, targeted changes

4. **Track what you changed**: Note which files were modified

**Rules**:
- Fix at most 5 issues per iteration (stay focused)
- Each fix should be minimal and targeted
- Don't refactor surrounding code
- Don't add features beyond what's needed for the fix
- If a fix requires architectural changes, note it for a future iteration

Use the `kalshi-flow-trader-specialist` agent (Task tool) for complex trader-specific fixes.

---

## Step 6: Simplify Changed Code

After implementing fixes, run the code-simplifier agent on changed files:

Use the Task tool with `subagent_type: "code-simplifier"`:
```
Prompt: "Simplify and clean up the following recently modified files. Focus on clarity, consistency, and maintainability while preserving all functionality:
[list the files you modified in Step 5]"
```

---

## Step 7: Validate Fixes

### 7a. Restart system
Repeat Step 1 (kill existing, clear logs, start fresh, wait for startup).

### 7b. Quick validation
Run a shorter observation cycle (2 cycles) to verify:
- Previous errors are gone
- No new errors introduced
- Captain cycles completing successfully
- Agent tool calls working
- System is stable

### 7c. Report
Create a brief iteration report:
```
## Iteration N Complete

### Issues Found: X
### Issues Fixed: X
### Issues Deferred: X

### Changes Made:
- file1.py: description of change
- file2.py: description of change

### Remaining Issues:
- issue1 (deferred because...)
- issue2 (deferred because...)

### Next Iteration Focus:
- What to prioritize next
```

---

## Step 8: Repeat or Stop

If there are remaining issues and the user hasn't asked to stop:
- Increment ITERATION
- Go to Step 1

If the system is stable and all major issues are fixed:
- Report final status
- Ask the user if they want another iteration or are satisfied

---

## Important Notes

- **Never commit automatically** - only commit when the user explicitly asks
- **Keep the system running during observation** - don't stop it prematurely
- **Log file location**: `backend/logs/v3-trader.log` (structured markers prefixed with `[SINGLE_ARB:*]`)
- **Frontend URL**: `http://localhost:5173/arb`
- **Backend health**: `http://localhost:8005/v3/health`
- **Backend status**: `http://localhost:8005/v3/status`
- The system must have `V3_SINGLE_ARB_ENABLED=true` and `V3_SINGLE_ARB_CAPTAIN_ENABLED=true` in `.env.paper` for captain cycles to run
