-- Knowledge Base tables for entity accumulation and signal generation
-- Part of the Deep Agent KB Redesign

-- Accumulated entity state (the KB)
-- Updated by EntityAccumulator on every mention, triggers Supabase Realtime
CREATE TABLE IF NOT EXISTS kb_entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_category TEXT NOT NULL,  -- person | organization | objective
    current_sentiment FLOAT DEFAULT 0,
    mention_count INT DEFAULT 0,
    unique_sources INT DEFAULT 0,
    signal_strength FLOAT DEFAULT 0,
    max_reddit_score INT DEFAULT 0,
    total_reddit_comments INT DEFAULT 0,
    source_types TEXT[] DEFAULT '{}',
    categories TEXT[] DEFAULT '{}',
    linked_market_tickers TEXT[] DEFAULT '{}',
    latest_context TEXT DEFAULT '',
    first_mention_at TIMESTAMPTZ,
    last_mention_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_entities_signal ON kb_entities(signal_strength DESC);
CREATE INDEX IF NOT EXISTS idx_kb_entities_updated ON kb_entities(updated_at DESC);

-- Enable Realtime for frontend KB visualization
ALTER PUBLICATION supabase_realtime ADD TABLE kb_entities;

-- Objective entities from Kalshi events
-- Generated during EntityMarketIndex refresh for outcome-type events
CREATE TABLE IF NOT EXISTS objective_entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    market_tickers TEXT[] DEFAULT '{}',
    keywords TEXT[] DEFAULT '{}',
    related_entities TEXT[] DEFAULT '{}',
    categories TEXT[] DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_obj_entities_event ON objective_entities(event_ticker);

-- Entity mentions (observation log)
-- Each extraction from any source creates a mention record
CREATE TABLE IF NOT EXISTS entity_mentions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_post_id TEXT,
    sentiment_score INT,
    confidence FLOAT,
    categories TEXT[] DEFAULT '{}',
    reddit_score INT DEFAULT 0,
    reddit_comments INT DEFAULT 0,
    context_snippet TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentions_created ON entity_mentions(created_at DESC);

-- Entity relations (extracted via LLM REL)
-- Stores directional relationships between named entities
CREATE TABLE IF NOT EXISTS entity_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_entity_id TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    relation TEXT NOT NULL,  -- SUPPORTS | OPPOSES | CAUSES | AFFECTED_BY | MEMBER_OF
    object_entity_id TEXT NOT NULL,
    object_name TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.7,
    source_post_id TEXT,
    context_snippet TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relations_subject ON entity_relations(subject_entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_relations_object ON entity_relations(object_entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_relations_created ON entity_relations(created_at DESC);
