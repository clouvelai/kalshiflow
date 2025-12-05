-- Initial PostgreSQL schema migration for Kalshi Flowboard
-- Converted from SQLite schema with PostgreSQL optimizations

-- Create markets table first (for foreign key reference)
CREATE TABLE IF NOT EXISTS markets (
    ticker VARCHAR(50) PRIMARY KEY,
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
    market_ticker VARCHAR(50) NOT NULL,
    yes_price INTEGER NOT NULL,
    no_price INTEGER NOT NULL,
    yes_price_dollars DECIMAL(10,4) NOT NULL,  -- More precise than REAL
    no_price_dollars DECIMAL(10,4) NOT NULL,
    count INTEGER NOT NULL,
    taker_side VARCHAR(10) NOT NULL,
    ts BIGINT NOT NULL,  -- Keep as BIGINT for millisecond timestamps
    received_at BIGINT NOT NULL,
    CONSTRAINT fk_trades_market_ticker 
        FOREIGN KEY(market_ticker) 
        REFERENCES markets(ticker) 
        ON DELETE CASCADE
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

-- Add comments for documentation
COMMENT ON TABLE markets IS 'Market metadata and information';
COMMENT ON TABLE trades IS 'Public trades from Kalshi WebSocket stream';
COMMENT ON COLUMN trades.ts IS 'Trade timestamp in milliseconds from Kalshi';
COMMENT ON COLUMN trades.received_at IS 'When we received the trade in milliseconds';
COMMENT ON COLUMN markets.raw_market_data IS 'Raw JSON market data from Kalshi API';