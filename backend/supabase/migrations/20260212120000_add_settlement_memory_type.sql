-- Add 'settlement_outcome' and 'trade_outcome' to agent_memories memory_type constraint.
-- settlement_outcome: written by AccountHealthService when settlements are discovered.
-- trade_outcome: recalled by Captain during deep_scan for learning from past trades.

ALTER TABLE agent_memories DROP CONSTRAINT IF EXISTS agent_memories_memory_type_check;
ALTER TABLE agent_memories ADD CONSTRAINT agent_memories_memory_type_check
  CHECK (memory_type IN (
    'learning', 'mistake', 'pattern', 'journal',
    'market_knowledge', 'consolidation',
    'signal', 'research', 'thesis',
    'trade', 'strategy', 'observation', 'trade_result',
    'news', 'news_digest', 'thesis_archived',
    'settlement_outcome', 'trade_outcome'
  ));
