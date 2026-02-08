-- Add source_title and context_snippet columns to market_price_impacts table
-- These columns surface the Reddit post title and entity context in the UX
-- so traders can understand WHY a signal was generated

ALTER TABLE market_price_impacts
ADD COLUMN IF NOT EXISTS source_title TEXT,
ADD COLUMN IF NOT EXISTS context_snippet TEXT;

-- Add comment for documentation
COMMENT ON COLUMN market_price_impacts.source_title IS 'Reddit post title that triggered this signal';
COMMENT ON COLUMN market_price_impacts.context_snippet IS 'Text context around the entity mention for understanding signal origin';
