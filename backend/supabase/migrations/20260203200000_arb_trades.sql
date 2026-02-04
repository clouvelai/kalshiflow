-- Arbitrage trade execution log
-- Records all trades executed by the spread monitor hot path

CREATE TABLE IF NOT EXISTS arb_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair_id UUID REFERENCES paired_markets(id),
    kalshi_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    contracts INT NOT NULL,
    price_cents INT NOT NULL,
    spread_at_entry INT,
    kalshi_mid INT,
    poly_mid INT,
    kalshi_order_id TEXT,
    status TEXT DEFAULT 'pending',
    reasoning TEXT,
    pnl_cents INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_arb_trades_pair ON arb_trades(pair_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arb_trades_status ON arb_trades(status);
CREATE INDEX IF NOT EXISTS idx_arb_trades_ticker ON arb_trades(kalshi_ticker, created_at DESC);
