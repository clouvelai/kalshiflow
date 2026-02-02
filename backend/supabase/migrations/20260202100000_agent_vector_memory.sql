-- Agent Vector Memory: pgvector-backed semantic memory for the deep agent.
-- Enables semantic similarity search, access-based ranking, consolidation, and retention.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

CREATE TABLE agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Classification
    memory_type TEXT NOT NULL CHECK (memory_type IN (
        'learning', 'mistake', 'pattern', 'journal',
        'market_knowledge', 'consolidation'
    )),

    -- Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- SHA-256 for fast dedup

    -- Vector embedding
    embedding extensions.vector(1536),
    embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small',

    -- Metadata for hybrid filtering
    market_tickers TEXT[] DEFAULT '{}',
    event_tickers TEXT[] DEFAULT '{}',
    confidence TEXT DEFAULT 'medium' CHECK (confidence IN ('low', 'medium', 'high')),

    -- Source tracking
    source_cycle INT,
    source_session TEXT,
    source_file TEXT,

    -- Trade linkage (for learnings from reflections)
    trade_id TEXT,
    trade_result TEXT,      -- win/loss/break_even
    pnl_cents INT,

    -- Access tracking (memory strength signal)
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,

    -- Consolidation lifecycle
    superseded_by UUID REFERENCES agent_memories(id),

    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search (sub-10ms at 100K rows)
CREATE INDEX idx_memories_embedding ON agent_memories
    USING hnsw (embedding extensions.vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Metadata filter indexes
CREATE INDEX idx_memories_type ON agent_memories(memory_type);
CREATE INDEX idx_memories_active ON agent_memories(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_memories_created ON agent_memories(created_at DESC);
CREATE INDEX idx_memories_markets ON agent_memories USING GIN(market_tickers);
CREATE INDEX idx_memories_events ON agent_memories USING GIN(event_tickers);
CREATE INDEX idx_memories_hash ON agent_memories(content_hash);
CREATE INDEX idx_memories_superseded ON agent_memories(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX idx_memories_access ON agent_memories(access_count DESC) WHERE is_active = TRUE;

-- RPC: Hybrid search (vector similarity + access-boosted ranking + metadata filters)
CREATE OR REPLACE FUNCTION search_agent_memories(
    query_embedding extensions.vector(1536),
    p_memory_types TEXT[] DEFAULT NULL,
    p_market_ticker TEXT DEFAULT NULL,
    p_event_ticker TEXT DEFAULT NULL,
    p_min_recency_hours FLOAT DEFAULT NULL,
    p_limit INT DEFAULT 10,
    p_similarity_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID, memory_type TEXT, content TEXT, confidence TEXT,
    market_tickers TEXT[], event_tickers TEXT[],
    trade_result TEXT, pnl_cents INT,
    access_count INT,
    created_at TIMESTAMPTZ, similarity FLOAT
) AS $$
    SELECT m.id, m.memory_type, m.content, m.confidence,
           m.market_tickers, m.event_tickers,
           m.trade_result, m.pnl_cents,
           m.access_count,
           m.created_at,
           -- Composite score: cosine similarity boosted by access frequency
           -- access_boost: log2(access_count + 1) * 0.02 gives +0.02 per doubling of access
           (1 - (m.embedding <=> query_embedding))
             + ln(m.access_count + 1) / ln(2) * 0.02
             AS similarity
    FROM agent_memories m
    WHERE m.is_active = TRUE
        AND m.embedding IS NOT NULL
        AND m.superseded_by IS NULL
        AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
        AND (p_market_ticker IS NULL OR p_market_ticker = ANY(m.market_tickers))
        AND (p_event_ticker IS NULL OR p_event_ticker = ANY(m.event_tickers))
        AND (p_min_recency_hours IS NULL
             OR m.created_at >= NOW() - (p_min_recency_hours || ' hours')::interval)
        AND (1 - (m.embedding <=> query_embedding)) >= p_similarity_threshold
    ORDER BY similarity DESC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- RPC: Find near-duplicates for dedup
CREATE OR REPLACE FUNCTION find_similar_memories(
    query_embedding extensions.vector(1536),
    p_memory_type TEXT,
    p_threshold FLOAT DEFAULT 0.88
)
RETURNS TABLE (id UUID, content TEXT, similarity FLOAT) AS $$
    SELECT m.id, m.content, 1 - (m.embedding <=> query_embedding) AS similarity
    FROM agent_memories m
    WHERE m.is_active = TRUE AND m.memory_type = p_memory_type
        AND m.embedding IS NOT NULL
        AND m.superseded_by IS NULL
        AND (1 - (m.embedding <=> query_embedding)) >= p_threshold
    ORDER BY (m.embedding <=> query_embedding) ASC LIMIT 5;
$$ LANGUAGE sql STABLE;

-- RPC: Batch update access tracking (called after every recall)
CREATE OR REPLACE FUNCTION touch_memories(p_ids UUID[])
RETURNS VOID AS $$
    UPDATE agent_memories
    SET access_count = access_count + 1,
        last_accessed_at = NOW(),
        updated_at = NOW()
    WHERE id = ANY(p_ids);
$$ LANGUAGE sql;

-- RPC: Find consolidation candidates
CREATE OR REPLACE FUNCTION find_consolidation_candidates(
    p_memory_type TEXT,
    p_cluster_threshold FLOAT DEFAULT 0.80,
    p_min_cluster_size INT DEFAULT 3,
    p_limit INT DEFAULT 50
)
RETURNS TABLE (
    id UUID, content TEXT, embedding extensions.vector(1536),
    access_count INT, created_at TIMESTAMPTZ
) AS $$
    SELECT m.id, m.content, m.embedding, m.access_count, m.created_at
    FROM agent_memories m
    WHERE m.is_active = TRUE
        AND m.memory_type = p_memory_type
        AND m.embedding IS NOT NULL
        AND m.superseded_by IS NULL
    ORDER BY m.created_at ASC
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- RPC: Mark memories as superseded by a consolidated entry
CREATE OR REPLACE FUNCTION supersede_memories(
    p_old_ids UUID[],
    p_new_id UUID
)
RETURNS INT AS $$
    WITH updated AS (
        UPDATE agent_memories
        SET superseded_by = p_new_id,
            is_active = FALSE,
            updated_at = NOW()
        WHERE id = ANY(p_old_ids)
            AND is_active = TRUE
        RETURNING id
    )
    SELECT count(*)::INT FROM updated;
$$ LANGUAGE sql;

-- RPC: Enforce retention policy (soft-delete stale low-value memories)
CREATE OR REPLACE FUNCTION enforce_retention_policy()
RETURNS TABLE (deleted_count INT, memory_type TEXT) AS $$
    WITH deleted AS (
        UPDATE agent_memories
        SET is_active = FALSE, updated_at = NOW()
        WHERE is_active = TRUE
            AND superseded_by IS NULL
            AND trade_id IS NULL  -- Never expire trade-linked memories
            AND (
                -- Low confidence, never accessed, older than 14 days
                (confidence = 'low' AND access_count = 0
                 AND created_at < NOW() - INTERVAL '14 days')
                OR
                -- Medium confidence, never accessed, older than 60 days
                (confidence = 'medium' AND access_count = 0
                 AND created_at < NOW() - INTERVAL '60 days')
                OR
                -- Any confidence, never accessed, older than 120 days
                (access_count = 0 AND created_at < NOW() - INTERVAL '120 days')
            )
        RETURNING agent_memories.memory_type
    )
    SELECT count(*)::INT AS deleted_count, d.memory_type
    FROM deleted d
    GROUP BY d.memory_type;
$$ LANGUAGE sql;
