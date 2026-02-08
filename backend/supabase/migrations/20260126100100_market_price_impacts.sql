-- Create market_price_impacts table for storing transformed price impact signals
-- This transforms entity sentiment into market-specific price impact scores

CREATE TABLE IF NOT EXISTS market_price_impacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source reference back to reddit_entities
    reddit_entity_id UUID REFERENCES reddit_entities(id) ON DELETE CASCADE,
    source_post_id TEXT NOT NULL,
    source_subreddit TEXT NOT NULL,

    -- Entity information
    entity_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,

    -- Market information
    market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    market_type TEXT NOT NULL,  -- OUT, WIN, CONFIRM, NOMINEE, etc.

    -- Score transformation
    sentiment_score INTEGER NOT NULL,      -- Original entity sentiment: -100 to +100
    price_impact_score INTEGER NOT NULL,   -- Transformed for market type: -100 to +100
    confidence FLOAT NOT NULL,             -- Signal confidence: 0.0 to 1.0

    -- Transformation metadata
    transformation_logic TEXT,  -- e.g., "OUT market: inverted sentiment"

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Realtime for this table
ALTER PUBLICATION supabase_realtime ADD TABLE market_price_impacts;

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_price_impacts_market ON market_price_impacts(market_ticker);
CREATE INDEX IF NOT EXISTS idx_price_impacts_entity ON market_price_impacts(entity_id);
CREATE INDEX IF NOT EXISTS idx_price_impacts_created ON market_price_impacts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_price_impacts_event ON market_price_impacts(event_ticker);
CREATE INDEX IF NOT EXISTS idx_price_impacts_market_type ON market_price_impacts(market_type);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_price_impacts_market_created
    ON market_price_impacts(market_ticker, created_at DESC);

-- Comments
COMMENT ON TABLE market_price_impacts IS 'Stores price impact signals derived from entity sentiment, transformed based on market type';
COMMENT ON COLUMN market_price_impacts.sentiment_score IS 'Original sentiment from entity extraction (-100 to +100)';
COMMENT ON COLUMN market_price_impacts.price_impact_score IS 'Transformed score for this market type (e.g., inverted for OUT markets)';
COMMENT ON COLUMN market_price_impacts.market_type IS 'Market type: OUT, WIN, CONFIRM, NOMINEE - determines transformation logic';
