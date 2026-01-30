-- =============================================================================
-- Entity Event Entities Table
-- =============================================================================
-- Caches LLM-classified entities from events. Each event gets one LLM call
-- that classifies ALL yes_sub_titles (person/org/outcome) and generates
-- search aliases per entity. Also extracts the event-level entity if any
-- (e.g., "Donald Trump" from "Trump Tariff Rate by May 31").
--
-- The `classifications` JSONB stores per-yes_sub_title:
--   {"Wind": {"type": "outcome", "aliases": ["wind energy", "wind power"]},
--    "Solar": {"type": "outcome", "aliases": ["solar energy", "solar power"]}}
--
-- Created: 2026-01-30
-- =============================================================================

-- Create entity_event_entities table
CREATE TABLE IF NOT EXISTS entity_event_entities (
    id BIGSERIAL PRIMARY KEY,
    event_ticker TEXT NOT NULL UNIQUE,       -- Kalshi event ticker (e.g., "KXTARIFF")
    event_title TEXT NOT NULL,               -- Event title used for extraction
    canonical_name TEXT,                     -- Extracted event-level entity name (NULL if none)
    entity_type TEXT DEFAULT 'person',       -- Event entity type: person, organization, position
    classifications JSONB DEFAULT '{}',      -- Per-yes_sub_title: {"yst": {"type": "...", "aliases": [...]}}
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookups by event_ticker
CREATE INDEX IF NOT EXISTS idx_entity_event_entities_event_ticker ON entity_event_entities(event_ticker);

-- Index for lookups by canonical_name (find all events for an entity)
CREATE INDEX IF NOT EXISTS idx_entity_event_entities_canonical_name ON entity_event_entities(canonical_name);

-- Add comments
COMMENT ON TABLE entity_event_entities IS 'Caches LLM event classification: per-yes_sub_title types + aliases, and event-level entity extraction. Populated by EntityMarketIndex._classify_event()';
COMMENT ON COLUMN entity_event_entities.event_ticker IS 'Kalshi event ticker (unique per event)';
COMMENT ON COLUMN entity_event_entities.event_title IS 'Event title text that was analyzed';
COMMENT ON COLUMN entity_event_entities.canonical_name IS 'Extracted event-level entity canonical name (NULL if no entity found)';
COMMENT ON COLUMN entity_event_entities.entity_type IS 'Event entity type: person, organization, position';
COMMENT ON COLUMN entity_event_entities.classifications IS 'Per-yes_sub_title classification: {"yst": {"type": "person"|"outcome"|"organization", "aliases": [...]}}';

-- Enable RLS (Row Level Security) for Supabase
ALTER TABLE entity_event_entities ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (service role)
CREATE POLICY "Allow all for service role" ON entity_event_entities
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant permissions
GRANT ALL ON entity_event_entities TO authenticated;
GRANT ALL ON entity_event_entities TO service_role;
GRANT USAGE, SELECT ON SEQUENCE entity_event_entities_id_seq TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE entity_event_entities_id_seq TO service_role;
