# Captain Memory Architect - Key Learnings

See root-level MEMORY.md for full notes:
`/Users/samuelclark/Desktop/kalshiflow/.claude/agent-memory/captain-memory-architect/MEMORY.md`

## Quick Reference
- Auto-curator: substring matching for sections, dedup via SequenceMatcher(0.82)
- LESSONS section: keep_limit=10 (was unbounded -- root cause of AGENTS.md bloat)
- Captain prompt: ~126 lines after 3-subagent refactor (was 84 after first reduction)
- Memory files per cycle: AGENTS.md(60) + SIGNALS.md(30) + PLAYBOOK.md(60) + THESES.md(30) = ~180 lines
- E2E test: `cd backend && uv run pytest tests/test_backend_e2e_regression.py -v`
- Unit tests: `cd backend && uv run pytest tests/traderv3/ -v --tb=short` (254 tests)
- Anti-narration: prompt explicitly prohibits "Cycle #N: PASS" entries + dedup catches them

## Critical Pattern: Legacy Global Wiring
- When removing legacy code paths (like _setup_legacy_tools), ALWAYS grep for ALL module-level globals they set
- _setup_gateway_tools MUST call set_tool_dependencies() with ALL globals that legacy tools still use
- record_learning, recall_context, search_event_news, etc. still use tools.py module globals
- The new agent_tools/ layer only covers captain_tools + commando_tools + mentions_tools

## Embeddings Architecture (2026-02-08)
- Subagent outputs now fire-and-forget embedded via DualMemoryStore.append(memory_type="subagent_output")
- Cross-cycle recall: _recall_prior_context() queries vector store for recent subagent_output entries
- Only analyst/intelligence outputs embedded (commando already uses record_learning)
- Prior context injected into cycle briefing, giving Captain continuity across fresh-thread cycles
