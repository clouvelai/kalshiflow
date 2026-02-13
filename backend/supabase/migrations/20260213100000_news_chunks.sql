-- News chunks: parent-child document pattern for chunked article storage.
-- Enables semantic search over article fragments while linking back to parent.

-- Add parent_id for chunk -> article linking
ALTER TABLE agent_memories
  ADD COLUMN IF NOT EXISTS parent_memory_id UUID REFERENCES agent_memories(id);

-- Expand memory_type constraint with 'news_chunk'
ALTER TABLE agent_memories DROP CONSTRAINT IF EXISTS agent_memories_memory_type_check;
ALTER TABLE agent_memories ADD CONSTRAINT agent_memories_memory_type_check
  CHECK (memory_type IN (
    'learning', 'mistake', 'pattern', 'journal',
    'market_knowledge', 'consolidation',
    'signal', 'research', 'thesis',
    'trade', 'strategy', 'observation', 'trade_result',
    'news', 'news_digest', 'thesis_archived',
    'settlement_outcome', 'trade_outcome',
    'news_chunk', 'swing_news'
  ));

-- Index for finding chunks of a parent article
CREATE INDEX IF NOT EXISTS idx_memories_parent
  ON agent_memories(parent_memory_id) WHERE parent_memory_id IS NOT NULL;
