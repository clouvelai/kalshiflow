-- Migration: research_decisions table
-- Purpose: Persist ALL agentic research decisions (traded + skipped) for offline analysis

CREATE TABLE IF NOT EXISTS research_decisions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Session context
    session_id VARCHAR(100),              -- Links to trading session
    strategy_id VARCHAR(50) NOT NULL,     -- 'agentic_research'

    -- Market identification
    market_ticker VARCHAR(100) NOT NULL,
    event_ticker VARCHAR(100),

    -- Decision outcome
    action VARCHAR(30) NOT NULL,          -- TRADE_YES, TRADE_NO, SKIP, etc.
    reason TEXT NOT NULL,                 -- Full reason (not truncated)
    traded BOOLEAN DEFAULT FALSE,

    -- Research outputs (at decision time)
    ai_probability DECIMAL(5,4),          -- LLM's YES probability
    market_probability DECIMAL(5,4),      -- Actual market price
    edge DECIMAL(5,4),                    -- Mispricing magnitude
    confidence VARCHAR(10),               -- high/medium/low
    recommendation VARCHAR(20),           -- BUY_YES/BUY_NO/HOLD

    -- Calibration data
    price_guess_cents INT,
    price_guess_error_cents INT,

    -- Full reasoning (not truncated)
    edge_explanation TEXT,                -- Full LLM reasoning
    key_driver TEXT,
    key_evidence JSONB,                   -- Array of evidence points

    -- Trade details (if traded)
    entry_price_cents INT,
    order_id VARCHAR(100)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_research_decisions_market ON research_decisions(market_ticker);
CREATE INDEX IF NOT EXISTS idx_research_decisions_event ON research_decisions(event_ticker);
CREATE INDEX IF NOT EXISTS idx_research_decisions_action ON research_decisions(action);
CREATE INDEX IF NOT EXISTS idx_research_decisions_traded ON research_decisions(traded);
CREATE INDEX IF NOT EXISTS idx_research_decisions_created ON research_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_research_decisions_session ON research_decisions(session_id);

-- Composite index for session + action analysis
CREATE INDEX IF NOT EXISTS idx_research_decisions_session_action ON research_decisions(session_id, action);

-- Comment for documentation
COMMENT ON TABLE research_decisions IS 'Tracks ALL agentic research decisions (traded + skipped) for offline analysis and iteration';
