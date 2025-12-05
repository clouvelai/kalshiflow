-- PostgreSQL indexes for optimal query performance
-- Based on SQLite indexes but optimized for PostgreSQL query planner

-- Primary performance indexes for trades table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_ts 
    ON trades(ts DESC);  -- Most queries order by timestamp descending

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_ticker_ts 
    ON trades(market_ticker, ts DESC);  -- Ticker-specific queries with time ordering

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_received_at 
    ON trades(received_at DESC);  -- Recovery queries use received_at

-- Covering index for common aggregation queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_ticker_window_agg 
    ON trades(market_ticker, ts) 
    INCLUDE (count, taker_side, yes_price_dollars, no_price_dollars);

-- Taker side performance for volume calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_taker_side_ts 
    ON trades(taker_side, ts DESC);

-- Markets table indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_markets_category 
    ON markets(category) WHERE category IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_markets_updated_at 
    ON markets(updated_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_markets_liquidity 
    ON markets(liquidity_dollars DESC) WHERE liquidity_dollars IS NOT NULL;

-- Partial indexes for active markets (optimization)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_markets_active_liquidity 
    ON markets(liquidity_dollars DESC) 
    WHERE liquidity_dollars > 0 AND latest_expiration_time > CURRENT_TIMESTAMP;

-- GIN index for JSON search in raw market data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_markets_raw_data_gin 
    ON markets USING GIN(raw_market_data) 
    WHERE raw_market_data IS NOT NULL;

-- Analyze tables for optimal query planning
ANALYZE markets;
ANALYZE trades;