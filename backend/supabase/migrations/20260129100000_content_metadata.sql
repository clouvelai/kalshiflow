-- Add content metadata columns to track extraction source and success
-- This enables visibility into what content is being fed to spaCy

-- Add content metadata to reddit_entities table
ALTER TABLE reddit_entities
ADD COLUMN IF NOT EXISTS content_type TEXT,           -- text, video, link, image, social, unknown
ADD COLUMN IF NOT EXISTS source_domain TEXT,          -- foxnews.com, youtube.com, reddit.com
ADD COLUMN IF NOT EXISTS extraction_source TEXT,      -- selftext, whisper, llm_extraction
ADD COLUMN IF NOT EXISTS extraction_success BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS extraction_error TEXT;

-- Add content metadata to market_price_impacts for Deep Agent visibility
ALTER TABLE market_price_impacts
ADD COLUMN IF NOT EXISTS content_type TEXT,
ADD COLUMN IF NOT EXISTS source_domain TEXT;

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_reddit_entities_content_type ON reddit_entities(content_type);
CREATE INDEX IF NOT EXISTS idx_reddit_entities_source_domain ON reddit_entities(source_domain);
CREATE INDEX IF NOT EXISTS idx_reddit_entities_extraction_success ON reddit_entities(extraction_success);

CREATE INDEX IF NOT EXISTS idx_price_impacts_content_type ON market_price_impacts(content_type);
CREATE INDEX IF NOT EXISTS idx_price_impacts_source_domain ON market_price_impacts(source_domain);

-- Comments for documentation
COMMENT ON COLUMN reddit_entities.content_type IS 'Type of content: text, video, link, image, social, unknown';
COMMENT ON COLUMN reddit_entities.source_domain IS 'Domain of source URL: youtube.com, foxnews.com, reddit.com, etc.';
COMMENT ON COLUMN reddit_entities.extraction_source IS 'Method of extraction: selftext, whisper, llm_extraction';
COMMENT ON COLUMN reddit_entities.extraction_success IS 'Whether content extraction succeeded';
COMMENT ON COLUMN reddit_entities.extraction_error IS 'Error message if extraction failed';

COMMENT ON COLUMN market_price_impacts.content_type IS 'Type of source content: text, video, link, image, social';
COMMENT ON COLUMN market_price_impacts.source_domain IS 'Domain of source URL: youtube.com, foxnews.com, reddit.com';
