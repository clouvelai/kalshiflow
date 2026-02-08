# Captain Memory Architect - Key Learnings

See root-level MEMORY.md for full notes:
`/Users/samuelclark/Desktop/kalshiflow/.claude/agent-memory/captain-memory-architect/MEMORY.md`

## Quick Reference
- Auto-curator: substring matching for sections, dedup via SequenceMatcher(0.82)
- LESSONS section: keep_limit=10 (was unbounded -- root cause of AGENTS.md bloat)
- Captain prompt: 84 lines (~3864 chars), reduced from 227 lines
- Memory files per cycle: AGENTS.md(60) + SIGNALS.md(30) + PLAYBOOK.md(40) + THESES.md(30) = ~160 lines
- E2E test: `cd backend && uv run pytest tests/test_backend_e2e_regression.py -v`
- Anti-narration: prompt explicitly prohibits "Cycle #N: PASS" entries + dedup catches them
