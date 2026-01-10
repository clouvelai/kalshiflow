-- Migration: Add v2 calibration fields to research_decisions table
-- Purpose: Persist calibration data for LLM probability estimation analysis

-- Add v2 calibration columns
ALTER TABLE research_decisions
    ADD COLUMN IF NOT EXISTS evidence_cited JSONB,             -- Which evidence points support this estimate
    ADD COLUMN IF NOT EXISTS what_would_change_mind TEXT,      -- What would most change this estimate
    ADD COLUMN IF NOT EXISTS assumption_flags JSONB,           -- Assumptions made due to missing info
    ADD COLUMN IF NOT EXISTS calibration_notes TEXT,           -- Notes on confidence calibration
    ADD COLUMN IF NOT EXISTS evidence_quality TEXT;            -- Quality: high, medium, low

-- Comment for documentation
COMMENT ON COLUMN research_decisions.evidence_cited IS 'List of evidence points that support the probability estimate';
COMMENT ON COLUMN research_decisions.what_would_change_mind IS 'What information would most change this estimate';
COMMENT ON COLUMN research_decisions.assumption_flags IS 'Assumptions made due to missing information';
COMMENT ON COLUMN research_decisions.calibration_notes IS 'Notes on confidence calibration methodology';
COMMENT ON COLUMN research_decisions.evidence_quality IS 'Quality rating of supporting evidence: high, medium, low';
