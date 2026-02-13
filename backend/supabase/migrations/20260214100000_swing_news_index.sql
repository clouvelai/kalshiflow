-- Swing-News Association Index
-- Links price swings to their causal news articles for predictive recall.
-- The key RPC find_similar_impact_patterns() enables: new news → embed → find
-- historically similar articles that moved prices → predict impact.

-- ============================================================
-- Table: swing_news_associations
-- ============================================================
CREATE TABLE IF NOT EXISTS swing_news_associations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_ticker TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up', 'down')),
    change_cents FLOAT NOT NULL,
    price_before FLOAT,
    price_after FLOAT,
    swing_start_ts TIMESTAMPTZ NOT NULL,
    swing_end_ts TIMESTAMPTZ NOT NULL,
    volume_during INT DEFAULT 0,
    source TEXT NOT NULL CHECK (source IN ('candlestick', 'live')),

    -- Causal news link
    news_memory_id UUID REFERENCES agent_memories(id) ON DELETE SET NULL,
    news_title TEXT,
    news_url TEXT,
    news_published_at TIMESTAMPTZ,
    causal_confidence FLOAT DEFAULT 0.0 CHECK (causal_confidence >= 0 AND causal_confidence <= 1),
    article_analysis JSONB,
    corroborating_count INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sna_event_ticker ON swing_news_associations(event_ticker);
CREATE INDEX IF NOT EXISTS idx_sna_market_ticker ON swing_news_associations(market_ticker);
CREATE INDEX IF NOT EXISTS idx_sna_news_memory_id ON swing_news_associations(news_memory_id);
CREATE INDEX IF NOT EXISTS idx_sna_created_at ON swing_news_associations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sna_confidence ON swing_news_associations(causal_confidence DESC)
    WHERE news_memory_id IS NOT NULL;

-- ============================================================
-- RPC: find_swing_news
-- Query past swing-news associations for an event.
-- ============================================================
CREATE OR REPLACE FUNCTION find_swing_news(
    p_event_ticker TEXT,
    p_min_change FLOAT DEFAULT 3.0,
    p_min_confidence FLOAT DEFAULT 0.5,
    p_limit INT DEFAULT 10
) RETURNS TABLE(
    id UUID,
    event_ticker TEXT,
    market_ticker TEXT,
    direction TEXT,
    change_cents FLOAT,
    news_title TEXT,
    news_url TEXT,
    causal_confidence FLOAT,
    source TEXT,
    swing_start_ts TIMESTAMPTZ,
    created_at TIMESTAMPTZ
) AS $$
    SELECT
        sna.id,
        sna.event_ticker,
        sna.market_ticker,
        sna.direction,
        sna.change_cents,
        sna.news_title,
        sna.news_url,
        sna.causal_confidence,
        sna.source,
        sna.swing_start_ts,
        sna.created_at
    FROM swing_news_associations sna
    WHERE sna.event_ticker = p_event_ticker
      AND sna.change_cents >= p_min_change
      AND sna.causal_confidence >= p_min_confidence
      AND sna.news_memory_id IS NOT NULL
    ORDER BY sna.change_cents DESC, sna.causal_confidence DESC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- ============================================================
-- RPC: find_similar_impact_patterns
-- THE KEY RPC: semantic search against swing-news embeddings to
-- find articles with similar content that previously moved prices.
-- Powers predictive recall.
-- ============================================================
CREATE OR REPLACE FUNCTION find_similar_impact_patterns(
    query_embedding vector(1536),
    p_min_confidence FLOAT DEFAULT 0.5,
    p_limit INT DEFAULT 5
) RETURNS TABLE(
    news_title TEXT,
    news_url TEXT,
    direction TEXT,
    change_cents FLOAT,
    causal_confidence FLOAT,
    event_ticker TEXT,
    market_ticker TEXT,
    similarity FLOAT
) AS $$
    SELECT
        sna.news_title,
        sna.news_url,
        sna.direction,
        sna.change_cents,
        sna.causal_confidence,
        sna.event_ticker,
        sna.market_ticker,
        1 - (am.embedding <=> query_embedding) AS similarity
    FROM swing_news_associations sna
    JOIN agent_memories am ON am.id = sna.news_memory_id
    WHERE sna.causal_confidence >= p_min_confidence
      AND sna.news_memory_id IS NOT NULL
      AND am.embedding IS NOT NULL
    ORDER BY am.embedding <=> query_embedding
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;
