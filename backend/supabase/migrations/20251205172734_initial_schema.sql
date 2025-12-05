-- Initial PostgreSQL schema migration for Kalshi Flowboard
-- Converted from SQLite schema with PostgreSQL optimizations
-- Combined with indexes and constraint fixes for Supabase compatibility

-- Create markets table first (for foreign key reference)
CREATE TABLE IF NOT EXISTS markets (
    ticker VARCHAR(100) PRIMARY KEY,  -- Increased length for long market identifiers
    title TEXT NOT NULL,
    category VARCHAR(100),
    liquidity_dollars DECIMAL(15,2),  -- More precise than REAL
    open_interest INTEGER,
    latest_expiration_time TIMESTAMPTZ,  -- PostgreSQL timezone-aware timestamp
    raw_market_data JSONB,  -- PostgreSQL native JSON storage instead of TEXT
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create trades table with PostgreSQL optimizations
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,  -- BIGSERIAL for PostgreSQL auto-increment
    market_ticker VARCHAR(100) NOT NULL,  -- Increased length to match markets
    yes_price INTEGER NOT NULL,
    no_price INTEGER NOT NULL,
    yes_price_dollars DECIMAL(10,4) NOT NULL,  -- More precise than REAL
    no_price_dollars DECIMAL(10,4) NOT NULL,
    count INTEGER NOT NULL,
    taker_side VARCHAR(10) NOT NULL,
    ts BIGINT NOT NULL,  -- Keep as BIGINT for millisecond timestamps
    received_at BIGINT NOT NULL
);

-- Add check constraints for data validation
ALTER TABLE trades ADD CONSTRAINT chk_taker_side 
    CHECK (taker_side IN ('yes', 'no'));

ALTER TABLE trades ADD CONSTRAINT chk_count_positive 
    CHECK (count > 0);

ALTER TABLE trades ADD CONSTRAINT chk_ts_valid 
    CHECK (ts > 1577836800000);  -- After 2020-01-01

-- Add updated_at trigger function for markets table
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for automatic updated_at updates
CREATE TRIGGER update_markets_updated_at 
    BEFORE UPDATE ON markets 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Performance indexes for trades table
CREATE INDEX IF NOT EXISTS idx_trades_ts 
    ON trades(ts DESC);  -- Most queries order by timestamp descending

CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts 
    ON trades(market_ticker, ts DESC);  -- Ticker-specific queries with time ordering

CREATE INDEX IF NOT EXISTS idx_trades_received_at 
    ON trades(received_at DESC);  -- Recovery queries use received_at

-- Covering index for common aggregation queries
CREATE INDEX IF NOT EXISTS idx_trades_ticker_window_agg 
    ON trades(market_ticker, ts) 
    INCLUDE (count, taker_side, yes_price_dollars, no_price_dollars);

-- Taker side performance for volume calculations
CREATE INDEX IF NOT EXISTS idx_trades_taker_side_ts 
    ON trades(taker_side, ts DESC);

-- Index for performance on market_ticker (no foreign key constraint to allow trades before metadata fetch)
CREATE INDEX IF NOT EXISTS idx_trades_market_ticker 
    ON trades(market_ticker);

-- Markets table indexes
CREATE INDEX IF NOT EXISTS idx_markets_category 
    ON markets(category) WHERE category IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_markets_updated_at 
    ON markets(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_markets_liquidity 
    ON markets(liquidity_dollars DESC) WHERE liquidity_dollars IS NOT NULL;

-- Partial indexes for active markets (optimization)
-- Note: Using NOW() instead of CURRENT_TIMESTAMP for index predicate compatibility
CREATE INDEX IF NOT EXISTS idx_markets_active_liquidity 
    ON markets(liquidity_dollars DESC) 
    WHERE liquidity_dollars > 0;

-- GIN index for JSON search in raw market data
CREATE INDEX IF NOT EXISTS idx_markets_raw_data_gin 
    ON markets USING GIN(raw_market_data) 
    WHERE raw_market_data IS NOT NULL;

-- Add comments for documentation
COMMENT ON TABLE markets IS 'Market metadata and information';
COMMENT ON TABLE trades IS 'Public trades from Kalshi WebSocket stream';
COMMENT ON COLUMN trades.ts IS 'Trade timestamp in milliseconds from Kalshi';
COMMENT ON COLUMN trades.received_at IS 'When we received the trade in milliseconds';
COMMENT ON COLUMN trades.market_ticker IS 'Market ticker - no foreign key constraint to allow trades before metadata fetch';
COMMENT ON COLUMN markets.ticker IS 'Market ticker - primary key for metadata lookup';
COMMENT ON COLUMN markets.raw_market_data IS 'Raw JSON market data from Kalshi API';

-- Analyze tables for optimal query planning
ANALYZE markets;
ANALYZE trades;