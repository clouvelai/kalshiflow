# Deep Agent as the System: Replace V3 Trader with a Learning Trader from Scratch

**Goal**: Replace the entire V3 trader with a deep-agent-based system where the **top-level entity is a deep agent** that builds its own profitable learning trader from scratch, aligned with [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview).

---

## Vision

Today:

- **Coordinator** is the top-level orchestrator: state machine, event bus, orderbook, trading client, listeners, syncers, health, status.
- **Deep Agent** is one **strategy plugin** among others; the coordinator starts it and it runs an observe–act–reflect loop inside the larger system.

Target:

- **Deep agent is the system.** It is the single main loop. It has **tools** to discover markets, read orderbooks, place orders, reflect on outcomes, and manage memory. It **plans** (todos), **delegates** (subagents), and **persists** (file + vector memory). The “trader” is whatever the agent learns and builds by using those tools over time.

No coordinator-driven strategy selection; no state machine that decides “which strategy runs.” The agent decides what to do next via planning and tool use.

---

## LangChain Deep Agents → Our Mapping

| LangChain Deep Agents | Current V3 / Deep Agent | Notes |
|-----------------------|-------------------------|--------|
| **Planning (write_todos)** | `write_todos` / `read_todos` in `tools.py` | Already present; agent can break “become profitable” into steps. |
| **Context (file system)** | `read_memory`, `write_memory`, `append_memory`, `edit_memory` + `memory/` dir | Already present; strategy, learnings, mistakes, patterns, etc. |
| **Subagent spawning (task)** | `task(agent, input)` + `subagents/` (e.g. issue_reporter) | Already present; can add reflection, event research, microstructure. |
| **Long-term memory** | File memory + `vector_memory.py` (Store-like) | Present; can standardize on LangGraph Store if we adopt that stack. |
| **Graph execution** | Custom `_main_loop()` in `agent.py` | Could be replaced by LangGraph if we adopt `deepagents`. |

So: **planning, context, subagents, and persistence are already there.** The shift is **inverting control**: deep agent at the top, coordinator and “strategy plugin” removed or reduced to “infra the agent uses via tools.”

---

## What to Remove vs Keep

### Remove or drastically reduce

- **Coordinator as orchestrator**  
  No single component that “runs the event loop,” “starts strategies,” “drives state machine.” The deep agent’s loop is the main loop.

- **State machine as main loop**  
  No explicit V3 state machine deciding “discovery → trading → …”. The agent’s plan (todos) and tool use replace that.

- **Strategy plugin pattern**  
  No `DeepAgentStrategy` plugin that the coordinator “starts.” The app starts infra + deep agent; the agent is the only “strategy.”

- **Optional**: Other strategy plugins (Reddit, price impact, etc.) as **separate strategies**. They can become **tools or subagents** the deep agent calls when it wants that input (e.g. `task(agent='event_researcher', input=...)` or tools that pull Reddit/GDELT).

### Keep (as agent environment / tools)

- **Orderbook integration** – Agent tool: e.g. `get_orderbook`, `get_microstructure`.
- **Trading client** – Agent tool: place/cancel orders (already `trade`, `TradeExecutor`).
- **State container** – Agent tools: positions, orders, PnL, session state (already `get_session_state`, `get_true_performance`, etc.).
- **Fills / lifecycle / positions** – Feed state container; agent reads via tools. Optional: push salient events into agent context (e.g. “fill received”, “position closed”).
- **Event bus** – Can be simplified to “event feed the agent can subscribe to or poll via tools” (e.g. `get_recent_events`).
- **Market discovery / tracked markets** – Agent tools: `get_markets`, discovery sync as a background process that updates data the agent reads.
- **WebSocket manager** – Keep for frontend; agent (or a small adapter) emits status and messages.
- **Health / status** – Can be a small side process or agent tools (e.g. `get_health`) rather than coordinator-owned loops.

So: **same capabilities, but exposed to the agent as tools and data**, not as code paths that “run the agent.”

---

## Two Implementation Paths

### Path A: Evolve current SelfImprovingAgent (recommended first step)

- **No new framework.** Keep `SelfImprovingAgent`, Anthropic, existing tools.
- **Change app entrypoint**:  
  - Boot: create orderbook integration, trading client, state container, event bus (or minimal event feed), WebSocket manager, any syncers that populate data.  
  - Start **only** the deep agent; its `_main_loop()` is the main loop.  
  - No coordinator `_run_event_loop()` that starts strategies; no `DeepAgentStrategy.start()` from coordinator.
- **Agent owns the loop**: It wakes on an interval (or event-driven later), loads context (memory files, todos), uses tools (markets, orderbook, trade, reflect), updates todos and memory, and optionally delegates to subagents.
- **Incremental**: You can keep the coordinator code but unused, or delete it stepwise. Frontend and APIs can stay; only the “who starts what” changes.

**Pros**: Reuses all current tools, prompts, subagents, reflection. Minimal dependency change.  
**Cons**: Still custom loop; no LangGraph benefits (checkpointing, persistence, multi-step graphs) unless added later.

### Path B: Adopt LangChain Deep Agents (LangGraph)

- **Rebuild the agent** using [deepagents](https://pypi.org/project/deepagents/): LangGraph-based, with `write_todos`, file tools, `task` for subagents, Store for long-term memory.
- **Implement “trader” as tools** that wrap current orderbook, trading client, state container (same as today, but registered as LangChain tools and called from the graph).
- **Deploy** the deep agent as the main process; infra (orderbook, Kalshi client, DB) stays outside the graph as services the tools call.

**Pros**: Standardized planning/subagent/memory model; LangSmith observability; possible use of LangGraph checkpointing and branching.  
**Cons**: Migration of current agent logic and prompts; new dependency; need to rewire all tools to LangChain tool interface.

---

## Suggested Phased Migration (Path A first)

| Phase | What | Outcome |
|-------|------|---------|
| **1** | New entrypoint: “deep-agent-first” app that starts only infra + SelfImprovingAgent (no coordinator, no strategy plugin). | Single process: infra + agent loop. Agent is the only “strategy.” |
| **2** | Remove or stub coordinator’s event loop and strategy startup in that code path. Keep health/status as lightweight side logic or agent tools. | No duplicate “main loop”; agent loop is canonical. |
| **3** | Promote “build a profitable trader” explicitly in agent prompt and todos: e.g. high-level todo “Improve edge and risk-adjusted returns” with sub-todos the agent refines. | Agent’s plan drives discovery, sizing, reflection, and learning. |
| **4** | (Optional) Add more subagents (e.g. async reflection, event research) per your existing [sub-agent proposal](./deep-supagent-propsoal.md). | Better cost and context isolation without changing “agent is the system.” |
| **5** | (Optional) Evaluate Path B: prototype one small agent in LangGraph/deepagents; if it pays off, migrate tool-by-tool. | Decision point: stay custom or adopt LangChain stack. |

---

## Success Criteria

- **Single main loop**: The deep agent’s loop is the only “strategy” loop; no coordinator-driven strategy selection.
- **Agent builds the trader**: The agent has tools for discovery, execution, and reflection; profitability comes from its plans and memory, not from hard-coded strategy logic.
- **LangChain-aligned capabilities**: Planning (todos), context (files), subagents (task), long-term memory (files + optional Store). These are already in place; the rest is control inversion and cleanup.
- **Operational continuity**: Frontend, WebSocket, health, and paper/live trading still work; only the internal “who is in charge” changes.

---

## References

- [LangChain Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview) – planning, context, subagents, long-term memory.
- [Deep Agent sub-agent proposal](./deep-supagent-propsoal.md) – async reflection, event research, microstructure; fits as subagents under this architecture.
- Current implementation: `SelfImprovingAgent` in `agent.py`, tools in `tools.py`, subagents in `subagents/`, strategy plugin in `strategies/plugins/deep_agent.py`.
