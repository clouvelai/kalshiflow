-- Extend agent_memories memory_type to support signal store, research, and thesis types.
-- These new types allow the deep agent to embed extraction signals, GDELT analysis results,
-- and event theses for semantic recall across cycles.

ALTER TABLE agent_memories DROP CONSTRAINT IF EXISTS agent_memories_memory_type_check;
ALTER TABLE agent_memories ADD CONSTRAINT agent_memories_memory_type_check
  CHECK (memory_type IN (
    'learning', 'mistake', 'pattern', 'journal',
    'market_knowledge', 'consolidation',
    'signal', 'research', 'thesis',
    'trade', 'strategy', 'observation', 'trade_result'
  ));
