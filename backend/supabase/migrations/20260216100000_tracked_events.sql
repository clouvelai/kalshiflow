-- Tracked Events: event-level grouping of markets for lifecycle tracking
CREATE TABLE IF NOT EXISTS tracked_events (
    event_ticker VARCHAR(100) PRIMARY KEY,
    title TEXT DEFAULT '',
    category VARCHAR(50) DEFAULT '',
    series_ticker VARCHAR(100) DEFAULT '',
    mutually_exclusive BOOLEAN DEFAULT TRUE,
    status VARCHAR(30) DEFAULT 'pending',
    earliest_open_ts BIGINT DEFAULT 0,
    latest_close_ts BIGINT DEFAULT 0,
    first_seen_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    discovery_source VARCHAR(30) DEFAULT 'lifecycle_ws',
    market_tickers JSONB DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_te_status ON tracked_events(status) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_te_close ON tracked_events(latest_close_ts) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_te_category ON tracked_events(category) WHERE is_active;
