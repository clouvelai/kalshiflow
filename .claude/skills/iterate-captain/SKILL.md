---
name: iterate-captain
description: Step-by-step Captain debugging. Runs one cycle, pauses, debugs issues, then resumes or restarts based on what changed.
argument-hint: [iterations]
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, TaskCreate, TaskUpdate, TaskList, AskUserQuestion
---

# Iterate Captain - Step-by-Step Debugging

Debug the Captain in 2-cycle batches. After every 2 cycles:
1. Auto-pause
2. Analyze logs for issues
3. Fix issues
4. Resume or Restart (if Python files changed)
5. Validate system is working

**Iterations**: `$ARGUMENTS` (default: 3 batches = 6 cycles)

---

## System Info

- **Backend**: Port 8005
- **Frontend**: http://localhost:5173/arb
- **Log file**: backend/logs/v3-trader.log
- **Log markers**: `[SINGLE_ARB:*]`

---

## Helper: Send WebSocket Command

Use this Python snippet to send pause/resume commands:

```bash
cd /Users/samuelclark/Desktop/kalshiflow/backend && uv run python -c "
import asyncio, websockets, json
async def send(cmd):
    try:
        async with websockets.connect('ws://localhost:8005/v3/ws') as ws:
            await ws.send(json.dumps({'type': cmd}))
            print(f'{cmd} sent')
    except Exception as e:
        print(f'Error: {e}')
asyncio.run(send('$CMD'))
"
```

Replace `$CMD` with `captain_pause` or `captain_resume`.

---

## Step 1: Start System

### 1a. Kill existing and clear logs
```bash
lsof -ti:8005 | xargs kill -9 2>/dev/null || true
sleep 2
> backend/logs/v3-trader.log
```

### 1b. Start system in background
```bash
cd /Users/samuelclark/Desktop/kalshiflow && ./scripts/run-v3.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use `run_in_background: true`.

### 1c. Wait for startup
Poll until `[SINGLE_ARB:STARTUP]` or `[SINGLE_ARB:CAPTAIN_START]` appears (timeout 90s):
```bash
for i in $(seq 1 18); do
  grep -q "SINGLE_ARB:STARTUP\|SINGLE_ARB:CAPTAIN_START" backend/logs/v3-trader.log 2>/dev/null && echo "READY" && break
  sleep 5
done
```

If startup fails, read the log file to diagnose.

---

## Step 2: Debug Loop (Repeat per Iteration)

### Track State
```
ITERATION = 1
MAX_ITERATIONS = $ARGUMENTS or 3
MODIFIED_PY_FILES = []
CYCLES_BEFORE = (count of CYCLE_END in logs)
```

### 2a. Wait for 2 cycles to complete
Poll until 2 new `[SINGLE_ARB:CYCLE_END]` markers appear:
```bash
START_COUNT=$(grep -c 'SINGLE_ARB:CYCLE_END' backend/logs/v3-trader.log 2>/dev/null || echo 0)
TARGET=$((START_COUNT + 2))
for i in $(seq 1 120); do
  CURRENT=$(grep -c 'SINGLE_ARB:CYCLE_END' backend/logs/v3-trader.log 2>/dev/null || echo 0)
  if [ "$CURRENT" -ge "$TARGET" ]; then
    echo "2 cycles complete (total: $CURRENT)"
    break
  fi
  echo "Waiting... cycles=$CURRENT target=$TARGET"
  sleep 5
done
```

### 2b. Pause after 2 cycles
Send `captain_pause` via WebSocket (use helper above).

### 2c. Extract the last 2 cycles' log segments
Get lines from the start of the first of the 2 cycles to the end:
```bash
# Get line numbers for the last 2 cycle starts and the final cycle end
STARTS=$(grep -n 'SINGLE_ARB:CYCLE_START' backend/logs/v3-trader.log | tail -2 | head -1 | cut -d: -f1)
LAST_END=$(grep -n 'SINGLE_ARB:CYCLE_END' backend/logs/v3-trader.log | tail -1 | cut -d: -f1)
sed -n "${STARTS},${LAST_END}p" backend/logs/v3-trader.log
```

### 2d. Analyze for issues
Look for in the cycle segment:
- `ERROR` or `Exception` lines
- `[SINGLE_ARB:*_ERROR]` markers
- `WARNING` that indicate real problems
- Failed tool calls or API errors
- Timeouts or connection issues

**Mentions-specific issues to watch for:**
- `mentions_specialist` subagent not being invoked for mentions markets
- `simulate_probability` or `trigger_simulation` failures
- Missing `gather_mention_contexts` or `gather_blind_context` calls
- Edge computation errors (spread/fee awareness issues)
- Context cache failures (4-hour TTL system)
- LLM simulation timeouts or rate limits

**How to trigger mentions testing:**
The Captain should auto-detect mentions markets (markets with "mention" rules).
If not seeing mentions activity, check:
1. Are mentions markets in the index? (`get_events_summary` output)
2. Is `mentions_enabled` config True?
3. Does `get_mentions_rules(event_ticker)` return parsed rules?

List issues by priority:
1. **Critical**: Crashes, exceptions, failed trades
2. **High**: Tool failures, missing data, mentions not detected
3. **Medium**: Warnings, slow performance, simulation timeouts
4. **Low**: Improvements

### 2e. Fix issues (if any)

**MAJOR vs MINOR changes:**
- **MINOR**: Single-line fixes, typos, simple bug fixes, logging changes → Apply directly
- **MAJOR**: Architectural changes, new functions, refactors, multi-file changes, prompt rewrites → **STOP and create plan file**

**For MINOR fixes:**
1. Locate source file using Grep/Glob
2. Read and understand context
3. Apply minimal fix using Edit
4. Track: `MODIFIED_PY_FILES.append(file_path)` if it's a `.py` file

**For MAJOR changes:**
1. **DO NOT apply the change**
2. Create a plan file at `backend/logs/iterate-captain-plan.md`:
```markdown
# Iterate Captain - Proposed Major Change

## Issue
[Description of the issue]

## Proposed Change
[What needs to change and why]

## Files Affected
- file1.py: [what changes]
- file2.py: [what changes]

## Risk Assessment
[Low/Medium/High] - [Why]

## Waiting for approval...
```
3. Tell the user: "Found a major change needed. Plan written to `backend/logs/iterate-captain-plan.md`. Review and approve before I proceed."
4. **STOP the iteration loop** and wait for user response
5. Only proceed with the change after explicit user approval

Key files:
- Captain: `backend/src/kalshiflow_rl/traderv3/single_arb/captain.py`
- Tools: `backend/src/kalshiflow_rl/traderv3/single_arb/tools.py`
- Coordinator: `backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py`
- Index/Monitor: `backend/src/kalshiflow_rl/traderv3/single_arb/index.py`, `monitor.py`
- Mentions Tools: `backend/src/kalshiflow_rl/traderv3/single_arb/mentions_tools.py`
- Mentions Context: `backend/src/kalshiflow_rl/traderv3/single_arb/mentions_context.py`
- Mentions Simulator: `backend/src/kalshiflow_rl/traderv3/single_arb/mentions_simulator.py`

For complex fixes, use:
```
Task tool with subagent_type: "kalshi-flow-trader-specialist"
```

### 2f. Decide: Resume or Restart

**If MODIFIED_PY_FILES is not empty** → RESTART
- Python changes require process restart to take effect
- Go to Step 3 (Restart) which includes validation

**If no .py files changed** → RESUME
- Frontend, config, or no changes
- Send `captain_resume` via WebSocket
- Go to Step 2g (Validate Resume)

### 2g. Validate Resume
After sending `captain_resume`, verify system is healthy:
```bash
# Wait 10s for system to resume
sleep 10

# Check health endpoint
curl -s http://localhost:8005/v3/health | jq -e '.healthy == true'

# Check captain is running (not paused)
curl -s http://localhost:8005/v3/status | jq -e '.captain.paused == false'

# Watch for next CYCLE_START to confirm agent resumed
for i in $(seq 1 12); do
  grep -q "SINGLE_ARB:CYCLE_START" backend/logs/v3-trader.log 2>/dev/null && echo "Captain resumed OK" && break
  sleep 5
done
```

If validation fails, investigate before continuing.

### 2h. Increment and check
```
ITERATION += 1
if ITERATION > MAX_ITERATIONS:
    Stop and report
else:
    Continue to Step 2a (wait for next 2 cycles)
```

---

## Step 3: Restart (When Python Files Changed)

### 3a. Graceful stop (faster than SIGKILL)
```bash
# Try graceful shutdown first (SIGTERM allows cleanup)
PID=$(lsof -ti:8005)
if [ -n "$PID" ]; then
  kill -TERM $PID 2>/dev/null
  sleep 1
  # Only SIGKILL if still running
  lsof -ti:8005 | xargs kill -9 2>/dev/null || true
fi
```

### 3b. Clear logs and modified files list
```bash
> backend/logs/v3-trader.log
```
```
MODIFIED_PY_FILES = []
```

### 3c. Start system fresh
```bash
cd /Users/samuelclark/Desktop/kalshiflow && ./scripts/run-v3.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use `run_in_background: true`.

### 3d. Fast Startup Validation
Optimized for speed - poll aggressively, fail fast:
```bash
# Fast poll for startup (check every 2s, timeout 60s)
for i in $(seq 1 30); do
  if grep -q "SINGLE_ARB:CAPTAIN_START" backend/logs/v3-trader.log 2>/dev/null; then
    echo "Captain started in $((i*2))s"
    break
  fi
  # Check for startup errors early
  if grep -q "ERROR\|Failed to start" backend/logs/v3-trader.log 2>/dev/null; then
    echo "Startup error detected!"
    tail -20 backend/logs/v3-trader.log
    break
  fi
  sleep 2
done

# Quick health check
curl -sf http://localhost:8005/v3/health > /dev/null && echo "Health OK" || echo "Health FAIL"
```

If validation fails, read logs and diagnose before continuing.

Then go to Step 2a (wait for 2 cycles).

---

## Fast Restart Optimizations

The restart is inherently slow due to:
1. Python process startup + imports (~5-10s)
2. Market discovery via REST API (~3-5s)
3. Orderbook WebSocket connections (~5-10s)
4. Index initialization waiting for data (~10-30s)

**Current optimizations in skill:**
- Graceful SIGTERM before SIGKILL (faster cleanup)
- Aggressive polling (2s intervals vs 5s)
- Early error detection (fail fast on startup errors)
- Skip unnecessary waits (60s timeout vs 90s)

**Future optimizations (not in this PR):**
1. **Hot module reload** - Reload only changed Python modules without full restart
2. **Market cache** - Cache discovered markets to skip REST call on restart
3. **Persistent orderbook connections** - Keep WS connections across restarts
4. **Index warmup from DB** - Load last known orderbook state from database

---

## Step 4: Final Report

After all iterations complete:
```markdown
## Iterate Captain Complete

### Summary
- Iterations: X / Y
- Cycles run: N
- Restarts required: M

### Issues Fixed
1. [file.py:line] Description

### Remaining Issues
1. Description (deferred because...)

### Files Modified
- file1.py
- file2.py
```

---

## Important Notes

- **Never commit automatically** - only when user asks
- **Major changes need approval** - create plan file at `backend/logs/iterate-captain-plan.md` and STOP
- **2 cycles per batch** - pause after every 2 cycles for analysis
- **Always validate** - check health after BOTH resume and restart
- **Restart if .py changed** - Python needs process restart
- **Resume if only frontend/config** - faster iteration
- **Focus on real issues** - don't over-engineer fixes
