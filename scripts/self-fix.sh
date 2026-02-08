#!/usr/bin/env bash
#
# self-fix.sh - Autonomous Captain self-improvement system.
#
# Reads open issues from issues.jsonl, creates fix branches, invokes Claude Code
# to fix each issue, validates with E2E tests, and auto-merges successful fixes.
#
# Usage:
#   ./scripts/self-fix.sh              # Fix all open issues
#   ./scripts/self-fix.sh --dry-run    # Show issues without fixing
#   ./scripts/self-fix.sh --limit N    # Fix at most N issues
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
READ_ISSUES="$SCRIPT_DIR/read_issues.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
LIMIT=0
FIXED=0
FAILED=0
SKIPPED=0

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${BLUE}[self-fix]${NC} Working branch: $CURRENT_BRANCH"

# Read open issues
ISSUES=$(python3 "$READ_ISSUES")
ISSUE_COUNT=$(echo "$ISSUES" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$ISSUE_COUNT" -eq 0 ]; then
    echo -e "${GREEN}[self-fix]${NC} No open issues. System is healthy."
    exit 0
fi

echo -e "${BLUE}[self-fix]${NC} Found $ISSUE_COUNT open issue(s)"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[self-fix]${NC} DRY RUN - showing issues only:"
    echo "$ISSUES" | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    print(f\"  [{i['severity']}] {i['id']}: {i['title']}\")
    if i.get('proposed_fix'):
        print(f\"    proposed: {i['proposed_fix']}\")
"
    exit 0
fi

# Check for clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}[self-fix]${NC} Stashing uncommitted changes..."
    git stash push -m "self-fix: auto-stash $(date +%Y%m%dT%H%M%S)"
    STASHED=true
else
    STASHED=false
fi

PROCESSED=0

# Process each issue
echo "$ISSUES" | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    print(f\"{i['id']}|{i['severity']}|{i['category']}|{i['title']}|{i.get('description','')}|{i.get('proposed_fix','')}\")
" | while IFS='|' read -r ID SEVERITY CATEGORY TITLE DESCRIPTION PROPOSED_FIX; do

    # Check limit
    if [ "$LIMIT" -gt 0 ] && [ "$PROCESSED" -ge "$LIMIT" ]; then
        echo -e "${YELLOW}[self-fix]${NC} Reached limit of $LIMIT issues"
        break
    fi

    PROCESSED=$((PROCESSED + 1))
    SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-' | head -c 30)
    FIX_BRANCH="fix/issue-${ID}-${SLUG}"

    echo ""
    echo -e "${BLUE}[self-fix]${NC} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}[self-fix]${NC} Issue ${ID}: [${SEVERITY}] ${TITLE}"
    echo -e "${BLUE}[self-fix]${NC} Branch: ${FIX_BRANCH}"
    echo -e "${BLUE}[self-fix]${NC} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Create fix branch
    git checkout -b "$FIX_BRANCH" 2>/dev/null || {
        echo -e "${YELLOW}[self-fix]${NC} Branch $FIX_BRANCH already exists, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    }

    # Invoke Claude Code
    echo -e "${BLUE}[self-fix]${NC} Invoking Claude Code..."

    CLAUDE_RESULT=0
    claude --dangerously-skip-permissions -p \
        "Fix this issue in the Kalshi Flow Captain trading system.

Title: ${TITLE}
Severity: ${SEVERITY}
Category: ${CATEGORY}
Description: ${DESCRIPTION}
Proposed Fix: ${PROPOSED_FIX}

Rules:
- Make minimal, focused changes
- Do NOT modify test files to make them pass - fix the actual code
- After making changes, run: cd backend && uv run pytest tests/test_backend_e2e_regression.py -v
- If you can't fix it, exit cleanly without making any changes
- Do not create new files unless absolutely necessary
- Commit your changes with a descriptive message" || CLAUDE_RESULT=$?

    if [ "$CLAUDE_RESULT" -ne 0 ]; then
        echo -e "${RED}[self-fix]${NC} Claude Code exited with error ($CLAUDE_RESULT)"
    fi

    # Check if Claude made any changes
    if git diff --quiet && git diff --cached --quiet; then
        echo -e "${YELLOW}[self-fix]${NC} No changes made, skipping"
        git checkout "$CURRENT_BRANCH"
        git branch -D "$FIX_BRANCH" 2>/dev/null || true
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Run E2E tests
    echo -e "${BLUE}[self-fix]${NC} Running E2E tests..."
    TEST_RESULT=0
    cd backend && uv run pytest tests/test_backend_e2e_regression.py -v 2>&1 || TEST_RESULT=$?
    cd "$PROJECT_DIR"

    if [ "$TEST_RESULT" -eq 0 ]; then
        echo -e "${GREEN}[self-fix]${NC} Tests PASSED"

        # Ensure changes are committed (Claude may have already committed)
        if ! git diff --quiet || ! git diff --cached --quiet; then
            git add -A
            git commit -m "fix(captain): auto-fix issue ${ID} - ${TITLE}

Self-fix: ${DESCRIPTION}
Severity: ${SEVERITY}
Category: ${CATEGORY}

Co-Authored-By: Claude Code <noreply@anthropic.com>"
        fi

        # Merge back to working branch
        git checkout "$CURRENT_BRANCH"
        git merge --no-edit "$FIX_BRANCH"
        git branch -d "$FIX_BRANCH"

        # Mark issue as resolved
        python3 "$READ_ISSUES" --resolve "$ID" "Auto-fixed by self-fix.sh (branch: $FIX_BRANCH)"
        FIXED=$((FIXED + 1))
        echo -e "${GREEN}[self-fix]${NC} Issue ${ID} FIXED and merged"
    else
        echo -e "${RED}[self-fix]${NC} Tests FAILED - abandoning fix"

        # Hard reset and go back to working branch
        git checkout -- .
        git clean -fd
        git checkout "$CURRENT_BRANCH"
        git branch -D "$FIX_BRANCH" 2>/dev/null || true

        # Mark attempt timestamp
        python3 "$READ_ISSUES" --mark-attempted "$ID"
        FAILED=$((FAILED + 1))
    fi
done

# Restore stashed changes
if [ "$STASHED" = true ]; then
    echo -e "${BLUE}[self-fix]${NC} Restoring stashed changes..."
    git stash pop || echo -e "${YELLOW}[self-fix]${NC} Stash pop had conflicts - resolve manually"
fi

echo ""
echo -e "${BLUE}[self-fix]${NC} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}[self-fix]${NC} Summary:"
echo -e "  ${GREEN}Fixed:${NC}   $FIXED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo -e "${BLUE}[self-fix]${NC} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
