-- Create news_entities table for the news source pipeline.
-- News articles flow through a separate table but feed into the same
-- _process_entity_record() processing path as Reddit entities.
--
-- Key design decisions:
-- - article_url UNIQUE provides natural dedup (same article from two searches = no-op)
-- - Lineage fields (search_query, triggered_by_entity, triggered_by_signal) enable debugging
-- - No Reddit-specific baggage (subreddit, score, num_comments)

CREATE TABLE IF NOT EXISTS news_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_url TEXT NOT NULL UNIQUE,
    headline TEXT NOT NULL,
    publisher TEXT,
    source_domain TEXT,
    published_at TIMESTAMPTZ,

    entities JSONB NOT NULL DEFAULT '[]',
    aggregate_sentiment INTEGER,

    -- Lineage: what triggered this search
    search_query TEXT,
    triggered_by_entity TEXT,
    triggered_by_signal TEXT,

    content_type TEXT DEFAULT 'news',
    extraction_source TEXT DEFAULT 'llm_extraction',
    extraction_success BOOLEAN DEFAULT FALSE,

    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER PUBLICATION supabase_realtime ADD TABLE news_entities;

CREATE INDEX IF NOT EXISTS idx_news_entities_created ON news_entities(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_entities_article_url ON news_entities(article_url);
CREATE INDEX IF NOT EXISTS idx_news_entities_entities ON news_entities USING GIN (entities);

-- Enable RLS (Row Level Security) for Supabase
ALTER TABLE news_entities ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (service role)
CREATE POLICY "Allow all for service role" ON news_entities
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant permissions
GRANT ALL ON news_entities TO authenticated;
GRANT ALL ON news_entities TO service_role;
