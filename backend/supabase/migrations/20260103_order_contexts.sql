-- Order Context Capture for V3 Trader
-- Enables quant analysis of settled trades with full signal/strategy/orderbook context
-- Only stores FILLED orders (staged in memory, persisted on fill confirmation)

CREATE TABLE IF NOT EXISTS order_contexts (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL UNIQUE,
    market_ticker VARCHAR(100) NOT NULL,
    session_id VARCHAR(50),

    -- Strategy & Signal
    strategy VARCHAR(50) NOT NULL,
    signal_id VARCHAR(200),
    signal_detected_at TIMESTAMPTZ,
    signal_params JSONB DEFAULT '{}',

    -- Market Context (P0 - Critical for validation)
    market_category VARCHAR(50),
    market_close_ts TIMESTAMPTZ,
    hours_to_settlement DECIMAL(8,2),
    trades_in_market INT,

    -- Price Context (P0 - Required for bucket-matched baseline)
    no_price_at_signal INT,           -- 100 - yes_price, for edge calculation
    bucket_5c INT,                    -- 45, 50, 55, ..., 95

    -- Orderbook (at order time)
    best_bid_cents INT,
    best_ask_cents INT,
    bid_ask_spread_cents INT,
    spread_tier VARCHAR(10),          -- "tight", "normal", "wide"
    bid_size_contracts INT,
    ask_size_contracts INT,

    -- Position (at order time)
    existing_position_count INT DEFAULT 0,
    existing_position_side VARCHAR(10),
    is_reentry BOOLEAN DEFAULT FALSE,
    entry_number INT DEFAULT 1,
    balance_cents INT,
    open_position_count INT DEFAULT 0,

    -- Order Details
    action VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_price_cents INT,
    order_quantity INT NOT NULL,
    order_type VARCHAR(20) DEFAULT 'limit',

    -- Timing
    placed_at TIMESTAMPTZ NOT NULL,
    hour_of_day_utc INT,
    day_of_week INT,
    calendar_week VARCHAR(10),        -- "2025-W01"

    -- Fill (populated when order fills)
    fill_count INT NOT NULL,
    fill_avg_price_cents INT,
    filled_at TIMESTAMPTZ NOT NULL,
    time_to_fill_ms BIGINT,
    slippage_cents INT,               -- fill_price - signal_price

    -- Settlement (updated when market settles)
    market_result VARCHAR(10),
    settled_at TIMESTAMPTZ,
    realized_pnl_cents INT,

    -- Metadata
    strategy_version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_action CHECK (action IN ('buy', 'sell')),
    CONSTRAINT chk_side CHECK (side IN ('yes', 'no'))
);

-- Indexes for quant queries
CREATE INDEX IF NOT EXISTS idx_order_contexts_strategy_settled
    ON order_contexts(strategy, settled_at) WHERE settled_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_order_contexts_bucket
    ON order_contexts(bucket_5c) WHERE settled_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_order_contexts_category
    ON order_contexts(market_category);
CREATE INDEX IF NOT EXISTS idx_order_contexts_week
    ON order_contexts(calendar_week);
CREATE INDEX IF NOT EXISTS idx_order_contexts_ticker
    ON order_contexts(market_ticker);
CREATE INDEX IF NOT EXISTS idx_order_contexts_filled_at
    ON order_contexts(filled_at DESC);

-- GIN index for JSONB signal params querying
CREATE INDEX IF NOT EXISTS idx_order_contexts_signal_params
    ON order_contexts USING GIN(signal_params);

-- Comments for documentation
COMMENT ON TABLE order_contexts IS 'Captures order context at fill time for post-hoc quant analysis';
COMMENT ON COLUMN order_contexts.signal_params IS 'Strategy-specific signal parameters in JSONB format (RLM: yes_ratio, price_drop, etc.)';
COMMENT ON COLUMN order_contexts.no_price_at_signal IS 'NO price at signal time (100 - yes_price), used for edge calculation: edge = win_rate - no_price/100';
COMMENT ON COLUMN order_contexts.bucket_5c IS 'Price bucket rounded to nearest 5 cents (45, 50, 55, ..., 95) for bucket-matched baseline';
COMMENT ON COLUMN order_contexts.spread_tier IS 'Spread classification at order time: tight (<=2c), normal (<=4c), wide (>4c)';
