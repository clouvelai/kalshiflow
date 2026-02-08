---
name: iterate-captain
description: Step-by-step Captain debugging. Runs cycles, pauses, debugs issues, then resumes or restarts based on what changed.
argument-hint: [iterations]
user-invocable: true
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, TaskCreate, TaskUpdate, TaskList, AskUserQuestion
---

# Iterate Captain - Step-by-Step Debugging

Debug the Captain in 2-cycle batches. After every 2 cycles:
1. Auto-pause via REST endpoint
2. Analyze logs for issues
3. Fix issues
4. Resume or Restart (if Python files changed)

**Iterations**: `$ARGUMENTS` (default: 3 batches = 6 cycles)

---

## System Info

- **Backend**: Port 8005
- **Log file**: `backend/logs/v3-trader.log`
- **Pause**: `curl -s -X POST -H 'Content-Type: application/json' -d '{"type":"captain_pause"}' http://localhost:8005/v3/captain/control`
- **Resume**: `curl -s -X POST -H 'Content-Type: application/json' -d '{"type":"captain_resume"}' http://localhost:8005/v3/captain/control`
- **Status**: `curl -s http://localhost:8005/v3/status`

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
cd /Users/samuelclark/Desktop/kalshiflow && ./scripts/run-captain.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use `run_in_background: true`.

### 1c. Wait for Captain to be running
Poll `/v3/status` until Captain is running (timeout 90s):
```bash
for i in $(seq 1 45); do
  RUNNING=$(curl -sf http://localhost:8005/v3/status 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d.get('components', {}).get('single_arb_coordinator', {}).get('captain', {})
    print(c.get('running', False))
except: print('False')
" 2>/dev/null || echo "False")
  if [ "$RUNNING" = "True" ]; then
    echo "Captain running"
    break
  fi
  sleep 2
done
```

If startup fails, read the log file to diagnose.

**Note**: Captain starts its run loop immediately but waits for background initialization
(understanding, lifecycle, causal models) before executing the first cycle. Look for
`[DEFERRED_INIT]` log markers to track background init progress.

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
```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"type":"captain_pause"}' http://localhost:8005/v3/captain/control
```

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

List issues by priority:
1. **Critical**: Crashes, exceptions, failed trades
2. **High**: Tool failures, missing data
3. **Medium**: Warnings, slow performance
4. **Low**: Improvements

### 2e. Fix issues (if any)

**MAJOR vs MINOR changes:**
- **MINOR**: Single-line fixes, typos, simple bug fixes, logging changes -> Apply directly
- **MAJOR**: Architectural changes, new functions, refactors, multi-file changes, prompt rewrites -> **STOP and create plan file**

**For MINOR fixes:**
1. Locate source file using Grep/Glob
2. Read and understand context
3. Apply minimal fix using Edit
4. Track: `MODIFIED_PY_FILES.append(file_path)` if it's a `.py` file

**For MAJOR changes:**
1. **DO NOT apply the change**
2. Create a plan file at `backend/logs/iterate-captain-plan.md`
3. Tell the user: "Found a major change needed. Plan written to `backend/logs/iterate-captain-plan.md`. Review and approve before I proceed."
4. **STOP the iteration loop** and wait for user response

Key files:
- Captain: `backend/src/kalshiflow_rl/traderv3/single_arb/captain.py`
- Tools: `backend/src/kalshiflow_rl/traderv3/single_arb/tools.py`
- Coordinator: `backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py`
- Index/Monitor: `backend/src/kalshiflow_rl/traderv3/single_arb/index.py`, `monitor.py`
- Mentions: `backend/src/kalshiflow_rl/traderv3/single_arb/mentions_tools.py`

### 2f. Decide: Resume or Restart

**If MODIFIED_PY_FILES is not empty** -> RESTART (go to Step 3)
**If no .py files changed** -> RESUME:

```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"type":"captain_resume"}' http://localhost:8005/v3/captain/control
```

Then validate resume:
```bash
sleep 10
curl -sf http://localhost:8005/v3/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Health OK' if d.get('healthy') else 'Health FAIL')"
curl -sf http://localhost:8005/v3/status | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('components',{}).get('single_arb_coordinator',{}).get('captain',{}); print('Captain OK' if not c.get('paused') else 'Captain still paused')"
```

### 2g. Increment and check
```
ITERATION += 1
if ITERATION > MAX_ITERATIONS:
    Stop and report
else:
    Continue to Step 2a
```

---

## Step 3: Restart (When Python Files Changed)

### 3a. Graceful stop
```bash
PID=$(lsof -ti:8005)
if [ -n "$PID" ]; then
  kill -TERM $PID 2>/dev/null
  sleep 1
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
cd /Users/samuelclark/Desktop/kalshiflow && ./scripts/run-captain.sh paper 2>&1 | tee backend/logs/v3-trader.log &
```
Use `run_in_background: true`.

### 3d. Fast Startup Validation
```bash
for i in $(seq 1 45); do
  RUNNING=$(curl -sf http://localhost:8005/v3/status 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d.get('components', {}).get('single_arb_coordinator', {}).get('captain', {})
    print(c.get('running', False))
except: print('False')
" 2>/dev/null || echo "False")
  if [ "$RUNNING" = "True" ]; then
    echo "Captain started in $((i*2))s"
    break
  fi
  if grep -q "Failed to start\|CRITICAL" backend/logs/v3-trader.log 2>/dev/null; then
    echo "Startup error!"
    tail -20 backend/logs/v3-trader.log
    break
  fi
  sleep 2
done

curl -sf http://localhost:8005/v3/health > /dev/null && echo "Health OK" || echo "Health FAIL"
```

Then go to Step 2a (wait for 2 cycles).

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
- **Major changes need approval** - create plan file and STOP
- **2 cycles per batch** - pause after every 2 cycles for analysis
- **Always validate** - check health after BOTH resume and restart
- **Restart if .py changed** - Python needs process restart (no --reload)
- **Resume if only frontend/config** - faster iteration
- **Deferred init**: Captain starts fast, but first cycle waits for `[DEFERRED_INIT]` completion
