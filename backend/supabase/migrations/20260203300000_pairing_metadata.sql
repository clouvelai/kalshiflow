-- Add match provenance metadata to paired_markets table
-- Tracks how each pair was discovered and validated

ALTER TABLE paired_markets
ADD COLUMN IF NOT EXISTS text_score REAL,
ADD COLUMN IF NOT EXISTS embedding_score REAL,
ADD COLUMN IF NOT EXISTS event_title_score REAL,
ADD COLUMN IF NOT EXISTS match_signals TEXT[],
ADD COLUMN IF NOT EXISTS validated_by TEXT DEFAULT 'manual';
