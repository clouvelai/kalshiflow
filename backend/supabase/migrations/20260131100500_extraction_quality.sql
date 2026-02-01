-- Add quality_score to extractions for extraction learning loop feedback
-- Set by evaluate_extractions tool after trade settlement
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS quality_score FLOAT;

CREATE INDEX IF NOT EXISTS idx_extractions_quality
    ON extractions(quality_score) WHERE quality_score IS NOT NULL;
