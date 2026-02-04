---
name: iterate-trader
description: Automated improvement loop for the V3 trader. Starts the trader, monitors logs + browser for issues, implements fixes, simplifies code, and validates. Use when you want to systematically find and fix issues in the V3 trader system.
argument-hint: [cycles]
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, TaskCreate, TaskUpdate, TaskList, AskUserQuestion
---

# Iterate Trader - Automated Improvement Loop

You are running an automated improvement loop on the V3 trader. Each iteration:
1. Start the trader
2. Observe via logs + browser (in parallel)
3. Stop after target cycles
4. Fix the top issues
5. Simplify changed code
6. Restart and validate

**Target cycles per iteration**: `$ARGUMENTS` (default: 5)

---

## Step 1: Setup & Start Trader

```
TARGET_CYCLES = $ARGUMENTS or 5
LOG_FILE = backend/logs/v3-trader.log
ITERATION = 1
```

### 1a. Kill existing trader
```bash
# Kill any process on port 8005
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
sleep 2
```

### 1b. Clear log file
```bash
> backend/logs/v3-trader.log
```

### 1c. Start trader in background
```bash
cd /Users/samuelclark/Desktop/kalshiflow
./scripts/run-v3.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use the Bash tool with `run_in_background: true` for this command.

### 1d. Wait for startup
Poll until you see `TRADER V3 STARTED SUCCESSFULLY` or `[V3:STARTUP]` in the log file (check every 5s, timeout 60s):
```bash
for i in $(seq 1 12); do
  grep -q "TRADER V3 STARTED SUCCESSFULLY\|V3:STARTUP" backend/logs/v3-trader.log 2>/dev/null && echo "READY" && break
  sleep 5
done
```

If the trader doesn't start, read the log file to diagnose and fix the startup error before proceeding.

---

## Step 2: Observe (Parallel Agents)

Launch TWO background agents simultaneously using a single message with multiple Task tool calls:

### 2a. Log Monitor Agent
Use the Task tool with `subagent_type: "general-purpose"` and `run_in_background: true`:

```
Prompt: "You are monitoring V3 trader logs for issues and improvements.

LOG FILE: /Users/samuelclark/Desktop/kalshiflow/backend/logs/v3-trader.log

Your job:
1. Read the log file every 30 seconds (use Read tool with offset to only read new lines)
2. Track how many agent cycles have completed by counting '[V3:CYCLE_END]' markers
3. Collect ALL of the following into a structured list:
   - ERRORS: Any ERROR or Exception lines with full context (2 lines before/after)
   - WARNINGS: Any WARNING lines that indicate real problems (not suppressed loggers)
   - TRADE_ISSUES: Failed trades, aborted trades, cancelled trades with reasons
   - AGENT_ISSUES: Agent failures, tool errors, subagent timeouts
   - PERFORMANCE: Slow operations (>30s), queue drops, connection issues
   - IMPROVEMENTS: Patterns you notice that could be improved

4. After TARGET_CYCLES agent cycles (look for [V3:CYCLE_END] count), OR if you've found 15+ issues, write your findings to:
   /Users/samuelclark/Desktop/kalshiflow/backend/logs/iteration-issues-logs.md

Format the output as:
# Log Monitor Issues - Iteration N

## Cycle Summary
- Cycles completed: X
- Total errors: X
- Total warnings: X

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
Prompt: "Validate the V3 trader frontend at http://localhost:5173/v3-trader

Wait 30 seconds for the trader to initialize, then:

1. SCREENSHOTS: Take screenshots of each major section:
   - Full page overview
   - Agent panel (right side - shows agent thinking/tool calls)
   - Spread monitor / pairs display
   - Any trade activity visible

2. CHECK these things:
   - Is the WebSocket connected? (should show 'Connected' or green indicator)
   - Are agent messages flowing in the agent panel?
   - Are spreads/pairs being displayed?
   - Are there any rendering errors, blank panels, or broken layouts?
   - Do the spread values look reasonable (not NaN, not all zeros)?
   - Is the trade execution UI working (if trades happen)?

3. WAIT and re-check after 2 minutes:
   - Has new data appeared?
   - Are agent cycles progressing?
   - Any UI freezes or stale data?

4. Write findings to:
   /Users/samuelclark/Desktop/kalshiflow/backend/logs/iteration-issues-browser.md

Format:
# Browser Validation Issues - Iteration N

## UI Issues
1. [SEVERITY] Description
   Screenshot: path

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
  CYCLES=$(grep -c 'V3:CYCLE_END' backend/logs/v3-trader.log 2>/dev/null || echo 0)
  echo "Waiting... cycles=$CYCLES logs_done=$LOGS_DONE browser_done=$BROWSER_DONE"
  sleep 15
done
```

If the log monitor agent hasn't finished but enough cycles have passed, you can proceed with whatever issues have been collected so far.

---

## Step 4: Stop Trader & Collect Issues

### 4a. Stop trader
```bash
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
sleep 2
```

### 4b. Read and merge issue lists
Read both issue files:
- `backend/logs/iteration-issues-logs.md`
- `backend/logs/iteration-issues-browser.md`

Merge into a single prioritized list:
1. **Critical** (errors, crashes, data corruption)
2. **High** (failed trades, broken UI, missing data)
3. **Medium** (warnings, performance, UX issues)
4. **Low** (improvements, cosmetics)

---

## Step 5: Implement Fixes

For each issue starting from highest priority:

1. **Locate the source**: Use Grep/Glob to find the relevant code
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

### 7a. Restart trader
Repeat Step 1 (kill existing, clear logs, start fresh, wait for startup).

### 7b. Quick validation
Run a shorter observation cycle (2-3 cycles) to verify:
- Previous errors are gone
- No new errors introduced
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
- **Keep the trader running during observation** - don't stop it prematurely
- **Log file location**: `backend/logs/v3-trader.log` (structured markers prefixed with `[V3:*]`)
- **Frontend URL**: `http://localhost:5173/v3-trader`
- **Backend health**: `http://localhost:8005/v3/health`
- **Backend status**: `http://localhost:8005/v3/status`
- The trader must be started with `ARB_ORCHESTRATOR_ENABLED=true` in `.env.paper` for agent cycles to run
