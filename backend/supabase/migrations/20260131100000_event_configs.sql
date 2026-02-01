-- Event Configs: Per-event langextract specifications
-- Each event has its own extraction config produced by understand_event

CREATE TABLE IF NOT EXISTS event_configs (
    event_ticker TEXT PRIMARY KEY,
    event_title TEXT NOT NULL,

    -- Event understanding (context for the deep agent)
    primary_entity TEXT,
    primary_entity_type TEXT,          -- person | org | policy | event
    description TEXT,                  -- What is this event about?
    key_drivers JSONB DEFAULT '[]',    -- What determines outcomes?
    outcome_descriptions JSONB DEFAULT '{}',  -- {yes_sub_title: meaning}

    -- langextract spec for this event
    prompt_description TEXT,           -- Event-specific extraction instructions
    extraction_classes JSONB DEFAULT '[]',    -- [{class_name, description, attributes}]
    examples JSONB DEFAULT '[]',       -- ExampleData objects as JSON
    watchlist JSONB DEFAULT '{}',      -- {entities: [], keywords: [], aliases: {}}

    -- Markets in this event
    markets JSONB DEFAULT '[]',        -- [{ticker, yes_sub_title, type}]

    -- Lifecycle
    is_active BOOLEAN DEFAULT true,    -- False when event closes
    last_researched_at TIMESTAMPTZ,
    research_version INT DEFAULT 1,    -- Bumps on each understand_event call
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_configs_active
    ON event_configs(is_active) WHERE is_active = true;
