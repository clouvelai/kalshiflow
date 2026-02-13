-- HNSW tuning: rebuild index with higher ef_construction for better recall.
-- Also set aggressive autovacuum for high-write memory table.

-- Rebuild HNSW with higher ef_construction for better recall
DROP INDEX IF EXISTS idx_memories_embedding;
CREATE INDEX idx_memories_embedding ON agent_memories
    USING hnsw (embedding extensions.vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Set aggressive autovacuum for high-write memory table
ALTER TABLE agent_memories SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.05
);
