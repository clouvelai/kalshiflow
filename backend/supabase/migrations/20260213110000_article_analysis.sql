-- Article analysis: structured LLM analysis stored alongside news memories.
-- Stores sentiment, relevance, entities, key claims, probability direction.

ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS article_analysis JSONB;
