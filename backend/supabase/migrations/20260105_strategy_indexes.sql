-- Migration: Add indexes for per-strategy queries on order_contexts
-- Purpose: Enable efficient filtering/aggregation by strategy for plugin-based trading system

-- Index for filtering by strategy
CREATE INDEX IF NOT EXISTS idx_order_contexts_strategy
    ON order_contexts(strategy);

-- Composite index for settled orders by strategy (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_order_contexts_strategy_settled
    ON order_contexts(strategy, settled_at)
    WHERE settled_at IS NOT NULL;

-- Composite index for strategy + market analysis
CREATE INDEX IF NOT EXISTS idx_order_contexts_strategy_ticker
    ON order_contexts(strategy, market_ticker);

-- Comment explaining usage
COMMENT ON INDEX idx_order_contexts_strategy IS 'Filter orders by strategy plugin (rlm_no, s013, etc.)';
COMMENT ON INDEX idx_order_contexts_strategy_settled IS 'Per-strategy settlement performance queries';
COMMENT ON INDEX idx_order_contexts_strategy_ticker IS 'Per-strategy per-market analysis';
