-- Production schema optimizations for Kalshi Flowboard (Fixed for Supabase)
-- Consolidated migration combining index optimizations, constraints, and monitoring
-- NOTE: Removed CONCURRENTLY keywords for Supabase compatibility

-- ========================================
-- SECTION 1: INDEX OPTIMIZATIONS
-- ========================================

-- Drop existing indexes that will be replaced with optimized versions
DROP INDEX IF EXISTS idx_trades_ticker_window_agg;
DROP INDEX IF EXISTS idx_markets_active_liquidity;

-- Optimized covering index for get_ticker_stats aggregation query
-- This query: WHERE market_ticker = ? AND ts >= ? with SUM(count), COUNT(*), AVG(prices)
CREATE INDEX IF NOT EXISTS idx_trades_ticker_stats_optimized
    ON trades(market_ticker, ts DESC)
    INCLUDE (count, taker_side, yes_price_dollars, no_price_dollars);

-- Optimized index for recovery queries with dual timestamp filtering
CREATE INDEX IF NOT EXISTS idx_trades_recovery_optimized
    ON trades(ts ASC) 
    WHERE ts >= 1577836800000;  -- Pre-filtered for valid timestamps (after 2020-01-01)

-- Composite index for ticker + time window queries (most common pattern)
CREATE INDEX IF NOT EXISTS idx_trades_ticker_time_window
    ON trades(market_ticker, ts DESC)
    WHERE ts >= 1577836800000;  -- Exclude corrupted timestamps

-- Partial index for high-liquidity active markets (production optimization)
-- Note: Removed CURRENT_TIMESTAMP predicate for immutability
CREATE INDEX IF NOT EXISTS idx_markets_high_liquidity_active
    ON markets(liquidity_dollars DESC, updated_at DESC)
    WHERE liquidity_dollars >= 1000.0;

-- Index for cleanup operations (old trade removal)  
-- Note: Simplified predicate using fixed timestamp
CREATE INDEX IF NOT EXISTS idx_trades_cleanup
    ON trades(ts ASC)
    WHERE ts >= 1577836800000;  -- After 2020-01-01

-- Partial unique constraint to prevent duplicate trades in the same millisecond
-- Note: Simplified predicate using fixed timestamp
CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_dedup_safety
    ON trades(market_ticker, ts, yes_price, no_price, count, taker_side)
    WHERE ts >= 1577836800000;  -- After 2020-01-01

-- ========================================
-- SECTION 2: PRODUCTION CONSTRAINTS
-- ========================================

-- Add constraints for price consistency (yes_price + no_price should equal 100)
ALTER TABLE trades ADD CONSTRAINT chk_price_consistency
    CHECK (yes_price + no_price = 100);

-- Add constraint for dollar price accuracy (should match cent prices)
ALTER TABLE trades ADD CONSTRAINT chk_dollar_price_accuracy
    CHECK (
        ABS(yes_price_dollars - (yes_price::decimal / 100.0)) < 0.0001 AND
        ABS(no_price_dollars - (no_price::decimal / 100.0)) < 0.0001
    );

-- Add constraint for reasonable trade size (prevent obvious data errors)
ALTER TABLE trades ADD CONSTRAINT chk_reasonable_count
    CHECK (count <= 1000000);  -- Max 1M shares per trade

-- Add constraint for recent timestamps (prevent far future dates)
-- Note: Using reasonable upper bound instead of dynamic CURRENT_TIMESTAMP
ALTER TABLE trades ADD CONSTRAINT chk_timestamp_not_future
    CHECK (ts <= 2000000000000);  -- Year 2033 upper bound

-- Add constraint for received_at timestamp ordering
ALTER TABLE trades ADD CONSTRAINT chk_received_at_reasonable
    CHECK (received_at >= (ts - 300000));  -- Allow 5 minutes of clock skew

-- Market constraints for data integrity
ALTER TABLE markets ADD CONSTRAINT chk_ticker_format
    CHECK (ticker ~ '^[A-Z0-9_-]+$');  -- Alphanumeric, underscore, dash only

ALTER TABLE markets ADD CONSTRAINT chk_liquidity_non_negative
    CHECK (liquidity_dollars IS NULL OR liquidity_dollars >= 0);

ALTER TABLE markets ADD CONSTRAINT chk_open_interest_non_negative
    CHECK (open_interest IS NULL OR open_interest >= 0);

-- ========================================
-- SECTION 3: PERFORMANCE TUNING
-- ========================================

-- Optimize PostgreSQL query planner with updated statistics
ALTER TABLE trades ALTER COLUMN market_ticker SET STATISTICS 1000;
ALTER TABLE trades ALTER COLUMN ts SET STATISTICS 1000;
ALTER TABLE trades ALTER COLUMN taker_side SET STATISTICS 100;

-- Update statistics for optimal query planning
ANALYZE trades;
ANALYZE markets;

-- ========================================
-- SECTION 4: SCHEMA VERSION TRACKING
-- ========================================

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100) DEFAULT CURRENT_USER
);

-- Record current schema state
INSERT INTO schema_info (version, description) 
VALUES ('1.6.0-production-fixed', 'Production schema optimizations: indexes, constraints, performance tuning')
ON CONFLICT DO NOTHING;