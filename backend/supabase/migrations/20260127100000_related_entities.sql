-- Related Entities Migration
-- Stores non-market entities discovered via general NER for second-hand signal analysis
-- Used by DeepAgent for contextual queries

-- Related entities table
-- Stores PERSON, ORG, GPE, EVENT entities that aren't directly linked to markets
-- but provide contextual signals for trading decisions
CREATE TABLE IF NOT EXISTS related_entities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),

    -- Entity identification
    entity_text TEXT NOT NULL,           -- Original text: "Putin", "Taiwan"
    entity_type TEXT NOT NULL,           -- "PERSON", "ORG", "GPE", "EVENT"
    normalized_id TEXT NOT NULL,         -- Normalized: "vladimir_putin", "taiwan"

    -- Sentiment from LLM
    sentiment_score INTEGER NOT NULL,    -- -100 to +100
    confidence FLOAT DEFAULT 1.0,

    -- Source context
    source_post_id TEXT NOT NULL,        -- Reddit post ID
    source_subreddit TEXT,
    context_snippet TEXT,                -- Surrounding text for context

    -- Co-occurrence with market entities
    -- Enables queries like "what PERSON entities appear with 'pam_bondi'?"
    co_occurring_market_entities TEXT[], -- ["pam_bondi", "pete_hegseth"]

    -- Constraints
    CONSTRAINT related_entities_sentiment_check
        CHECK (sentiment_score >= -100 AND sentiment_score <= 100),
    CONSTRAINT related_entities_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT related_entities_type_check
        CHECK (entity_type IN ('PERSON', 'ORG', 'GPE', 'EVENT', 'OTHER'))
);

-- Indexes for efficient querying
-- Primary lookups by normalized_id
CREATE INDEX IF NOT EXISTS idx_related_entities_normalized
    ON related_entities(normalized_id);

-- Filter by entity type (PERSON, ORG, GPE, EVENT)
CREATE INDEX IF NOT EXISTS idx_related_entities_type
    ON related_entities(entity_type);

-- Temporal queries (recent entities first)
CREATE INDEX IF NOT EXISTS idx_related_entities_created
    ON related_entities(created_at DESC);

-- Co-occurrence queries using GIN for array containment
CREATE INDEX IF NOT EXISTS idx_related_entities_cooccur
    ON related_entities USING GIN(co_occurring_market_entities);

-- Composite index for common query pattern: type + time
CREATE INDEX IF NOT EXISTS idx_related_entities_type_time
    ON related_entities(entity_type, created_at DESC);

-- Composite index for sentiment magnitude filtering
-- Enables queries for strong sentiment (positive or negative)
CREATE INDEX IF NOT EXISTS idx_related_entities_sentiment
    ON related_entities(ABS(sentiment_score) DESC);

-- Source tracking
CREATE INDEX IF NOT EXISTS idx_related_entities_post
    ON related_entities(source_post_id);

CREATE INDEX IF NOT EXISTS idx_related_entities_subreddit
    ON related_entities(source_subreddit);

-- Enable realtime for DeepAgent subscriptions
ALTER TABLE related_entities REPLICA IDENTITY FULL;

-- Add to realtime publication (allows subscriptions)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
        AND tablename = 'related_entities'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE related_entities;
    END IF;
END $$;

-- Comments for documentation
COMMENT ON TABLE related_entities IS 'Non-market entities (PERSON, ORG, GPE, EVENT) for second-hand signal analysis';
COMMENT ON COLUMN related_entities.entity_text IS 'Original entity text as found in source';
COMMENT ON COLUMN related_entities.entity_type IS 'NER label: PERSON, ORG, GPE, EVENT';
COMMENT ON COLUMN related_entities.normalized_id IS 'Normalized identifier for deduplication';
COMMENT ON COLUMN related_entities.sentiment_score IS 'Sentiment toward entity (-100 to +100)';
COMMENT ON COLUMN related_entities.co_occurring_market_entities IS 'Market entities mentioned in same post';
