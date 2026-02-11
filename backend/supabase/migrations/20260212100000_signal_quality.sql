-- Signal Quality Feedback Loop: Add signal_quality to agent_memories.
-- Scale: 0.0 = confirmed noise, 0.5 = unknown/neutral, 1.0 = strong signal.
-- Backfilled by _backfill_news_impacts() based on measured price changes.

ALTER TABLE agent_memories
  ADD COLUMN IF NOT EXISTS signal_quality FLOAT DEFAULT 0.5;

ALTER TABLE agent_memories
  ADD CONSTRAINT agent_memories_signal_quality_range
  CHECK (signal_quality >= 0.0 AND signal_quality <= 1.0);

-- Sparse index: only index memories that are active AND have proven signal value.
-- Used by search_agent_memories signal boost term.
CREATE INDEX IF NOT EXISTS idx_memories_signal_quality
  ON agent_memories(signal_quality DESC)
  WHERE signal_quality > 0.6 AND is_active = TRUE;
