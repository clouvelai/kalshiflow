-- News Intelligence: Extend agent_memories with news-specific columns and
-- create news_price_impacts table for tracking news-price correlation.

-- 1. Add news-specific columns to agent_memories
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS news_url TEXT;
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS news_title TEXT;
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS news_published_at TIMESTAMPTZ;
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS news_source TEXT;
ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS price_snapshot JSONB;

-- 2. Index for URL dedup (unique constraint on non-null URLs)
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_news_url
  ON agent_memories(news_url) WHERE news_url IS NOT NULL;

-- 3. Index for time-range queries on news
CREATE INDEX IF NOT EXISTS idx_memories_news_published
  ON agent_memories(news_published_at DESC) WHERE news_published_at IS NOT NULL;

-- 4. Extend memory_type CHECK to include news types
ALTER TABLE agent_memories DROP CONSTRAINT IF EXISTS agent_memories_memory_type_check;
ALTER TABLE agent_memories ADD CONSTRAINT agent_memories_memory_type_check
  CHECK (memory_type IN (
    'learning', 'mistake', 'pattern', 'journal',
    'market_knowledge', 'consolidation',
    'signal', 'research', 'thesis',
    'trade', 'strategy', 'observation', 'trade_result',
    'news', 'news_digest', 'thesis_archived'
  ));

-- 5. Update search_agent_memories to include temporal decay scoring
-- Score = cosine_similarity * temporal_decay + access_boost
-- temporal_decay = exp(-0.05 * hours_old) — half-life ~14 hours
-- Must DROP first because return type is changing (adding news columns)
DROP FUNCTION IF EXISTS search_agent_memories;
CREATE OR REPLACE FUNCTION search_agent_memories(
    query_embedding extensions.vector(1536),
    p_memory_types TEXT[] DEFAULT NULL,
    p_market_ticker TEXT DEFAULT NULL,
    p_event_ticker TEXT DEFAULT NULL,
    p_min_recency_hours FLOAT DEFAULT NULL,
    p_limit INT DEFAULT 10,
    p_similarity_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID, memory_type TEXT, content TEXT, confidence TEXT,
    market_tickers TEXT[], event_tickers TEXT[],
    trade_result TEXT, pnl_cents INT,
    access_count INT,
    created_at TIMESTAMPTZ, similarity FLOAT,
    news_url TEXT, news_title TEXT, news_published_at TIMESTAMPTZ,
    news_source TEXT, price_snapshot JSONB
) AS $$
    SELECT m.id, m.memory_type, m.content, m.confidence,
           m.market_tickers, m.event_tickers,
           m.trade_result, m.pnl_cents,
           m.access_count,
           m.created_at,
           -- Composite score: cosine similarity with temporal decay + access boost
           (1 - (m.embedding <=> query_embedding))
             * exp(-0.05 * EXTRACT(EPOCH FROM (NOW() - m.created_at)) / 3600)
             + ln(m.access_count + 1) / ln(2) * 0.02
             AS similarity,
           m.news_url, m.news_title, m.news_published_at,
           m.news_source, m.price_snapshot
    FROM agent_memories m
    WHERE m.is_active = TRUE
        AND m.embedding IS NOT NULL
        AND m.superseded_by IS NULL
        AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
        AND (p_market_ticker IS NULL OR p_market_ticker = ANY(m.market_tickers))
        AND (p_event_ticker IS NULL OR p_event_ticker = ANY(m.event_tickers))
        AND (p_min_recency_hours IS NULL
             OR m.created_at >= NOW() - (p_min_recency_hours || ' hours')::interval)
        AND (1 - (m.embedding <=> query_embedding)) >= p_similarity_threshold
    ORDER BY similarity DESC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- 6. News price impacts table (Phase 3)
CREATE TABLE IF NOT EXISTS news_price_impacts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    news_memory_id UUID REFERENCES agent_memories(id),
    market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,

    price_at_news JSONB,        -- {yes_bid, yes_ask, yes_mid, ts}
    price_after_1h JSONB,       -- same structure
    price_after_4h JSONB,
    price_after_24h JSONB,

    change_1h_cents INT,
    change_4h_cents INT,
    change_24h_cents INT,

    magnitude TEXT CHECK (magnitude IN ('none', 'small', 'medium', 'large')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_price_impacts_event
  ON news_price_impacts(event_ticker);
CREATE INDEX IF NOT EXISTS idx_price_impacts_magnitude
  ON news_price_impacts(magnitude) WHERE magnitude != 'none';
CREATE INDEX IF NOT EXISTS idx_price_impacts_news_memory
  ON news_price_impacts(news_memory_id);

-- 7. RPC: Find market-moving articles
CREATE OR REPLACE FUNCTION find_market_movers(
    p_event_ticker TEXT,
    p_min_change INT DEFAULT 5,
    p_limit INT DEFAULT 10
)
RETURNS TABLE(
    news_title TEXT,
    news_url TEXT,
    change_cents INT,
    direction TEXT,
    delay_hours INT,
    published_at TIMESTAMPTZ,
    price_at_news JSONB,
    market_ticker TEXT
) AS $$
    SELECT
        am.news_title,
        am.news_url,
        GREATEST(
            ABS(COALESCE(npi.change_1h_cents, 0)),
            ABS(COALESCE(npi.change_4h_cents, 0)),
            ABS(COALESCE(npi.change_24h_cents, 0))
        ) AS change_cents,
        CASE
            WHEN GREATEST(ABS(COALESCE(npi.change_1h_cents, 0)),
                         ABS(COALESCE(npi.change_4h_cents, 0)),
                         ABS(COALESCE(npi.change_24h_cents, 0)))
                = ABS(COALESCE(npi.change_1h_cents, 0))
            THEN CASE WHEN npi.change_1h_cents > 0 THEN 'up' ELSE 'down' END
            WHEN GREATEST(ABS(COALESCE(npi.change_1h_cents, 0)),
                         ABS(COALESCE(npi.change_4h_cents, 0)),
                         ABS(COALESCE(npi.change_24h_cents, 0)))
                = ABS(COALESCE(npi.change_4h_cents, 0))
            THEN CASE WHEN npi.change_4h_cents > 0 THEN 'up' ELSE 'down' END
            ELSE CASE WHEN npi.change_24h_cents > 0 THEN 'up' ELSE 'down' END
        END AS direction,
        CASE
            WHEN ABS(COALESCE(npi.change_24h_cents, 0)) >= p_min_change THEN 24
            WHEN ABS(COALESCE(npi.change_4h_cents, 0)) >= p_min_change THEN 4
            ELSE 1
        END AS delay_hours,
        am.news_published_at AS published_at,
        npi.price_at_news,
        npi.market_ticker
    FROM news_price_impacts npi
    JOIN agent_memories am ON am.id = npi.news_memory_id
    WHERE npi.event_ticker = p_event_ticker
        AND GREATEST(
            ABS(COALESCE(npi.change_1h_cents, 0)),
            ABS(COALESCE(npi.change_4h_cents, 0)),
            ABS(COALESCE(npi.change_24h_cents, 0))
        ) >= p_min_change
    ORDER BY change_cents DESC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;
