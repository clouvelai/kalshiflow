-- Extractions: Stream of individual extractions from langextract
-- Each row = one extraction from one source
-- Many extractions per source (a post about Trump + tariffs = 2+ extractions)

CREATE TABLE IF NOT EXISTS extractions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source tracking (for engagement refresh)
    source_type TEXT NOT NULL,              -- reddit_post | reddit_comment | news | video
    source_id TEXT NOT NULL,                -- Reddit submission/comment ID, article URL
    source_url TEXT,
    source_subreddit TEXT,

    -- langextract output
    extraction_class TEXT NOT NULL,         -- market_signal | entity_mention | context_factor
    extraction_text TEXT NOT NULL,          -- Verbatim text from source
    attributes JSONB NOT NULL DEFAULT '{}', -- Class-specific attributes

    -- Market linking (many-to-many: this extraction can impact N markets)
    market_tickers TEXT[] DEFAULT '{}',     -- Which markets this impacts
    event_tickers TEXT[] DEFAULT '{}',      -- Which events this relates to

    -- Engagement snapshot at extraction time
    engagement_score INT DEFAULT 0,         -- Reddit score at extraction time
    engagement_comments INT DEFAULT 0,      -- Comment count at extraction time
    engagement_updated_at TIMESTAMPTZ,      -- When engagement was last refreshed

    -- Timestamps
    source_created_at TIMESTAMPTZ,          -- When the source was published
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Realtime for downstream processing
ALTER PUBLICATION supabase_realtime ADD TABLE extractions;

CREATE INDEX IF NOT EXISTS idx_extractions_created ON extractions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_extractions_source ON extractions(source_id);
CREATE INDEX IF NOT EXISTS idx_extractions_class ON extractions(extraction_class);
CREATE INDEX IF NOT EXISTS idx_extractions_markets ON extractions USING GIN(market_tickers);
CREATE INDEX IF NOT EXISTS idx_extractions_events ON extractions USING GIN(event_tickers);
