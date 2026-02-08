-- Migration: Add source_created_at to market_price_impacts
-- Purpose: Preserve original Reddit post creation time for temporal analysis

ALTER TABLE market_price_impacts
ADD COLUMN source_created_at TIMESTAMPTZ;

COMMENT ON COLUMN market_price_impacts.source_created_at IS
    'Original Reddit post creation time (UTC). NULL for legacy records.';

-- Create index for temporal queries
CREATE INDEX IF NOT EXISTS idx_market_price_impacts_source_created_at
ON market_price_impacts(source_created_at DESC)
WHERE source_created_at IS NOT NULL;
