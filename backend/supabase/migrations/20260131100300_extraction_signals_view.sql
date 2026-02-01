-- Extraction Signals: Aggregated view per market
-- Materialized from extractions table, queried by deep agent

CREATE OR REPLACE FUNCTION get_extraction_signals(
    p_window_hours FLOAT DEFAULT 4.0,
    p_market_ticker TEXT DEFAULT NULL
)
RETURNS TABLE (
    market_ticker TEXT,
    extraction_class TEXT,
    direction TEXT,
    occurrence_count BIGINT,
    unique_sources BIGINT,
    avg_engagement NUMERIC,
    max_engagement INT,
    total_comments BIGINT,
    avg_magnitude NUMERIC,
    avg_sentiment NUMERIC,
    last_seen_at TIMESTAMPTZ,
    first_seen_at TIMESTAMPTZ,
    oldest_source_at TIMESTAMPTZ,
    newest_source_at TIMESTAMPTZ
)
LANGUAGE sql STABLE
AS $$
    SELECT
        unnest(e.market_tickers) AS market_ticker,
        e.extraction_class,
        e.attributes->>'direction' AS direction,
        COUNT(*) AS occurrence_count,
        COUNT(DISTINCT e.source_id) AS unique_sources,
        AVG(e.engagement_score) AS avg_engagement,
        MAX(e.engagement_score) AS max_engagement,
        SUM(e.engagement_comments) AS total_comments,
        AVG((e.attributes->>'magnitude')::float) AS avg_magnitude,
        AVG((e.attributes->>'sentiment')::float) AS avg_sentiment,
        MAX(e.created_at) AS last_seen_at,
        MIN(e.created_at) AS first_seen_at,
        MIN(e.source_created_at) AS oldest_source_at,
        MAX(e.source_created_at) AS newest_source_at
    FROM extractions e
    WHERE e.created_at > NOW() - (p_window_hours || ' hours')::INTERVAL
      AND e.market_tickers != '{}'
      AND (p_market_ticker IS NULL OR p_market_ticker = ANY(e.market_tickers))
    GROUP BY unnest(e.market_tickers), e.extraction_class,
             e.attributes->>'direction';
$$;
