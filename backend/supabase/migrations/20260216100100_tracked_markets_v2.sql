-- Tracked Markets V2: persistent market state for lifecycle tracking with restart recovery
CREATE TABLE IF NOT EXISTS tracked_markets_v2 (
    ticker VARCHAR(100) PRIMARY KEY,
    event_ticker VARCHAR(100) NOT NULL,
    title TEXT DEFAULT '',
    category VARCHAR(50) DEFAULT '',
    status VARCHAR(30) DEFAULT 'pending',
    open_ts BIGINT DEFAULT 0,
    close_ts BIGINT DEFAULT 0,
    determined_ts BIGINT,
    settled_ts BIGINT,
    tracked_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    discovery_source VARCHAR(30) DEFAULT 'lifecycle_ws',
    market_info JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_tmv2_event ON tracked_markets_v2(event_ticker);
CREATE INDEX IF NOT EXISTS idx_tmv2_status ON tracked_markets_v2(status) WHERE is_active;
