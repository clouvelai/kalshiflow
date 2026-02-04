-- Cross-venue arbitrage: paired markets and price ticks
-- Maps Kalshi markets to Polymarket markets for spread monitoring

CREATE TABLE IF NOT EXISTS paired_markets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kalshi_ticker TEXT NOT NULL,
    kalshi_event_ticker TEXT,
    poly_condition_id TEXT NOT NULL,
    poly_token_id_yes TEXT NOT NULL,
    poly_token_id_no TEXT,
    question TEXT NOT NULL,
    match_method TEXT DEFAULT 'manual',
    match_confidence FLOAT DEFAULT 1.0,
    threshold_override_cents INT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(kalshi_ticker, poly_condition_id)
);

CREATE TABLE IF NOT EXISTS price_ticks (
    id BIGSERIAL PRIMARY KEY,
    pair_id UUID REFERENCES paired_markets(id),
    kalshi_yes_bid INT,
    kalshi_yes_ask INT,
    poly_yes_cents INT,
    poly_no_cents INT,
    spread_cents INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts ON price_ticks(pair_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_spread ON price_ticks(spread_cents, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_paired_markets_status ON paired_markets(status);
CREATE INDEX IF NOT EXISTS idx_paired_markets_kalshi ON paired_markets(kalshi_ticker);
