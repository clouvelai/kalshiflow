-- Add dedicated columns for AI Research strategy tracking
-- These enable direct calibration queries without JSONB extraction

ALTER TABLE order_contexts ADD COLUMN IF NOT EXISTS ai_probability DECIMAL(5,4);
ALTER TABLE order_contexts ADD COLUMN IF NOT EXISTS ai_confidence VARCHAR(10);
ALTER TABLE order_contexts ADD COLUMN IF NOT EXISTS price_guess_cents INT;
ALTER TABLE order_contexts ADD COLUMN IF NOT EXISTS price_guess_error_cents INT;
ALTER TABLE order_contexts ADD COLUMN IF NOT EXISTS event_ticker VARCHAR(100);

-- Indexes for calibration analysis queries
CREATE INDEX IF NOT EXISTS idx_order_contexts_ai_probability
    ON order_contexts(ai_probability) WHERE ai_probability IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_order_contexts_ai_confidence
    ON order_contexts(ai_confidence) WHERE ai_confidence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_order_contexts_event_ticker
    ON order_contexts(event_ticker);

COMMENT ON COLUMN order_contexts.ai_probability IS 'AI agent estimated YES probability (0.0-1.0)';
COMMENT ON COLUMN order_contexts.ai_confidence IS 'AI confidence level: high, medium, low';
COMMENT ON COLUMN order_contexts.price_guess_cents IS 'LLM blind price estimate in cents';
COMMENT ON COLUMN order_contexts.price_guess_error_cents IS 'price_guess - actual_mid_price';
COMMENT ON COLUMN order_contexts.event_ticker IS 'Event ticker for grouping related markets';
