-- Extraction Examples: Learning cycle for langextract
-- Stores real and AI-generated examples for continuous improvement

CREATE TABLE IF NOT EXISTS extraction_examples (
    id BIGSERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    extractions JSONB NOT NULL,          -- Expected langextract output
    source TEXT NOT NULL CHECK (source IN ('real', 'claude')),
    quality_score FLOAT DEFAULT 0.5,
    event_ticker TEXT,                   -- Optional: event-specific example
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
