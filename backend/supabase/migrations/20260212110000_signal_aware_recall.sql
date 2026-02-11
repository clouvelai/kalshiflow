-- Signal-Aware Recall: Add signal_quality boost to search_agent_memories scoring.
-- Effect: signal_quality=1.0 gets +0.075 boost, 0.3 gets -0.03 penalty, 0.5/NULL = zero.

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
           -- Composite score: cosine similarity with temporal decay + access boost + signal boost
           -- decay=0.01 gives ~69h half-life
           -- signal boost: (signal_quality - 0.5) * 0.15 → range [-0.075, +0.075]
           (1 - (m.embedding <=> query_embedding))
             * exp(-0.01 * EXTRACT(EPOCH FROM (NOW() - m.created_at)) / 3600)
             + ln(m.access_count + 1) / ln(2) * 0.02
             + COALESCE(m.signal_quality - 0.5, 0) * 0.15
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
