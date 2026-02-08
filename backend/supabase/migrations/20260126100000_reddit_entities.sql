-- Create reddit_entities table for storing extracted entities from Reddit posts
-- This stores raw sentiment per entity, which is then transformed into price impacts

CREATE TABLE IF NOT EXISTS reddit_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id TEXT NOT NULL UNIQUE,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    author TEXT,
    score INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    post_created_utc TIMESTAMPTZ,

    -- Extracted entities stored as JSONB array
    -- Example: [{"entity_id": "pam_bondi", "canonical_name": "Pam Bondi",
    --            "entity_type": "person", "sentiment_score": -87, "confidence": 0.94,
    --            "context_snippet": "Bondi faces scrutiny..."}]
    entities JSONB NOT NULL DEFAULT '[]',

    -- Aggregate sentiment across all entities in this post
    aggregate_sentiment INTEGER,

    -- Processing timestamps
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Realtime for this table
ALTER PUBLICATION supabase_realtime ADD TABLE reddit_entities;

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_reddit_entities_created ON reddit_entities(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reddit_entities_subreddit ON reddit_entities(subreddit);
CREATE INDEX IF NOT EXISTS idx_reddit_entities_post_id ON reddit_entities(post_id);

-- GIN index for JSONB entity queries
CREATE INDEX IF NOT EXISTS idx_reddit_entities_entities ON reddit_entities USING GIN (entities);

-- Comment describing the table
COMMENT ON TABLE reddit_entities IS 'Stores Reddit posts with extracted entities and sentiment scores for entity-based trading';
COMMENT ON COLUMN reddit_entities.entities IS 'JSONB array of ExtractedEntity objects with entity_id, canonical_name, entity_type, sentiment_score, confidence, context_snippet';
