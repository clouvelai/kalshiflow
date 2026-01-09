-- Migration: Semantic Frames Table
-- Purpose: Store extracted semantic understanding of prediction market events
-- This enables caching of event structure to avoid repeated LLM extraction

-- Create the semantic_frames table
CREATE TABLE IF NOT EXISTS semantic_frames (
    -- Primary key: event ticker (one frame per event)
    event_ticker TEXT PRIMARY KEY,

    -- Frame type classification
    frame_type TEXT NOT NULL DEFAULT 'unknown',

    -- Semantic structure
    question_template TEXT,
    primary_relation TEXT,

    -- Semantic roles (JSONB arrays of SemanticRole objects)
    actors JSONB NOT NULL DEFAULT '[]',
    objects JSONB NOT NULL DEFAULT '[]',
    candidates JSONB NOT NULL DEFAULT '[]',

    -- Frame constraints
    mutual_exclusivity BOOLEAN DEFAULT TRUE,
    actor_controls_outcome BOOLEAN DEFAULT FALSE,
    resolution_trigger TEXT,

    -- Search optimization
    primary_search_queries TEXT[] DEFAULT '{}',
    signal_keywords TEXT[] DEFAULT '{}',

    -- Full context for recovery (denormalized for performance)
    event_context JSONB,
    key_driver_analysis JSONB,

    -- Metadata
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    event_title TEXT,
    event_category TEXT
);

-- Index for frame type queries (e.g., find all NOMINATION events)
CREATE INDEX IF NOT EXISTS idx_semantic_frames_type
ON semantic_frames(frame_type);

-- Index for category-based queries
CREATE INDEX IF NOT EXISTS idx_semantic_frames_category
ON semantic_frames(event_category);

-- Index for recent extractions (useful for cache management)
CREATE INDEX IF NOT EXISTS idx_semantic_frames_extracted_at
ON semantic_frames(extracted_at DESC);

-- GIN index for full-text search on signal keywords
CREATE INDEX IF NOT EXISTS idx_semantic_frames_keywords
ON semantic_frames USING GIN (signal_keywords);

-- Comment explaining the table structure
COMMENT ON TABLE semantic_frames IS
'Cached semantic understanding of prediction market events.
Each event has a frame_type (NOMINATION, COMPETITION, etc.)
and semantic roles (actors, objects, candidates) that describe
the structure of the prediction question.';

COMMENT ON COLUMN semantic_frames.actors IS
'Entities with agency/decision power (e.g., Trump for Fed Chair nomination)';

COMMENT ON COLUMN semantic_frames.candidates IS
'Possible outcomes linked to specific markets (e.g., Warsh -> KXFEDCHAIRNOM-WARSH)';

COMMENT ON COLUMN semantic_frames.primary_search_queries IS
'Targeted search queries for finding news about this event';

COMMENT ON COLUMN semantic_frames.signal_keywords IS
'Keywords that indicate important news (frontrunner, shortlist, etc.)';
