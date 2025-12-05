-- Fix PostgreSQL schema constraints for production compatibility
-- Issue 1: Increase ticker length for long market identifiers
-- Issue 2: Remove foreign key constraint to allow trades without metadata

-- Drop foreign key constraint temporarily
ALTER TABLE trades DROP CONSTRAINT IF EXISTS fk_trades_market_ticker;

-- Increase ticker length to handle long market identifiers
ALTER TABLE markets ALTER COLUMN ticker TYPE VARCHAR(100);
ALTER TABLE trades ALTER COLUMN market_ticker TYPE VARCHAR(100);

-- Add index for performance on the now-unconstrained foreign key
CREATE INDEX IF NOT EXISTS idx_trades_market_ticker 
    ON trades(market_ticker);

-- Add comments explaining the design decision
COMMENT ON COLUMN trades.market_ticker IS 'Market ticker - no foreign key constraint to allow trades before metadata fetch';
COMMENT ON COLUMN markets.ticker IS 'Market ticker - primary key for metadata lookup';