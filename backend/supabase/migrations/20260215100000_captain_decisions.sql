-- Decision ledger for Captain order quality tracking.
-- Records every order with production market context at decision time,
-- then backfills production price evolution to compute hypothetical P&L.

CREATE TABLE IF NOT EXISTS captain_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Decision metadata
    order_id TEXT,
    source TEXT NOT NULL CHECK (source IN ('captain', 'sniper')),
    event_ticker TEXT,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    action TEXT NOT NULL CHECK (action IN ('buy', 'sell')),
    contracts INT NOT NULL,
    limit_price_cents INT NOT NULL,
    reasoning TEXT,
    cycle_mode TEXT,

    -- Production snapshot at decision time
    prod_yes_bid INT,
    prod_yes_ask INT,
    prod_yes_mid REAL,
    prod_spread INT,
    prod_volume_5m INT,
    prod_book_imbalance REAL,

    -- Demo order outcome (updated by backfill)
    demo_status TEXT,
    demo_fill_count INT DEFAULT 0,

    -- Production price evolution (backfilled at 1m/5m/15m/1h)
    prod_mid_1m REAL,
    prod_mid_5m REAL,
    prod_mid_15m REAL,
    prod_mid_1h REAL,

    -- Computed metrics (backfilled)
    would_have_filled BOOLEAN,
    direction_correct BOOLEAN,
    hypothetical_pnl_cents REAL,
    max_favorable_cents REAL,
    max_adverse_cents REAL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_captain_decisions_created
    ON captain_decisions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_captain_decisions_market
    ON captain_decisions (market_ticker, created_at DESC);

-- Aggregated accuracy statistics function
CREATE OR REPLACE FUNCTION get_captain_accuracy_stats(
    p_hours_back INT DEFAULT 24,
    p_source TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    cutoff TIMESTAMPTZ;
BEGIN
    cutoff := now() - (p_hours_back || ' hours')::INTERVAL;

    WITH filtered AS (
        SELECT *
        FROM captain_decisions
        WHERE created_at >= cutoff
          AND (p_source IS NULL OR source = p_source)
    ),
    totals AS (
        SELECT
            COUNT(*) AS total_decisions,
            COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) AS decisions_with_outcomes,
            COUNT(*) FILTER (WHERE direction_correct = TRUE) AS direction_correct_count,
            ROUND(
                CASE WHEN COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) > 0
                THEN COUNT(*) FILTER (WHERE direction_correct = TRUE)::NUMERIC
                     / COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) * 100
                ELSE 0 END, 1
            ) AS direction_accuracy_pct,
            ROUND(COALESCE(AVG(hypothetical_pnl_cents) FILTER (WHERE hypothetical_pnl_cents IS NOT NULL), 0)::NUMERIC, 2) AS avg_hypothetical_pnl,
            ROUND(COALESCE(SUM(hypothetical_pnl_cents) FILTER (WHERE hypothetical_pnl_cents IS NOT NULL), 0)::NUMERIC, 2) AS total_hypothetical_pnl,
            COUNT(*) FILTER (WHERE would_have_filled = TRUE) AS would_have_filled_count,
            ROUND(
                CASE WHEN COUNT(*) FILTER (WHERE would_have_filled IS NOT NULL) > 0
                THEN COUNT(*) FILTER (WHERE would_have_filled = TRUE)::NUMERIC
                     / COUNT(*) FILTER (WHERE would_have_filled IS NOT NULL) * 100
                ELSE 0 END, 1
            ) AS would_have_filled_pct
        FROM filtered
    ),
    by_source AS (
        SELECT
            source,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE direction_correct = TRUE) AS correct,
            ROUND(
                CASE WHEN COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) > 0
                THEN COUNT(*) FILTER (WHERE direction_correct = TRUE)::NUMERIC
                     / COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) * 100
                ELSE 0 END, 1
            ) AS accuracy_pct,
            ROUND(COALESCE(AVG(hypothetical_pnl_cents) FILTER (WHERE hypothetical_pnl_cents IS NOT NULL), 0)::NUMERIC, 2) AS avg_pnl
        FROM filtered
        GROUP BY source
    ),
    by_mode AS (
        SELECT
            COALESCE(cycle_mode, 'unknown') AS mode,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE direction_correct = TRUE) AS correct,
            ROUND(
                CASE WHEN COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) > 0
                THEN COUNT(*) FILTER (WHERE direction_correct = TRUE)::NUMERIC
                     / COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) * 100
                ELSE 0 END, 1
            ) AS accuracy_pct
        FROM filtered
        GROUP BY COALESCE(cycle_mode, 'unknown')
    )
    SELECT jsonb_build_object(
        'total_decisions', t.total_decisions,
        'decisions_with_outcomes', t.decisions_with_outcomes,
        'direction_correct_count', t.direction_correct_count,
        'direction_accuracy_pct', t.direction_accuracy_pct,
        'avg_hypothetical_pnl', t.avg_hypothetical_pnl,
        'total_hypothetical_pnl', t.total_hypothetical_pnl,
        'would_have_filled_count', t.would_have_filled_count,
        'would_have_filled_pct', t.would_have_filled_pct,
        'by_source', COALESCE((
            SELECT jsonb_object_agg(s.source, jsonb_build_object(
                'total', s.total, 'correct', s.correct,
                'accuracy_pct', s.accuracy_pct, 'avg_pnl', s.avg_pnl
            ))
            FROM by_source s
        ), '{}'::jsonb),
        'by_cycle_mode', COALESCE((
            SELECT jsonb_object_agg(m.mode, jsonb_build_object(
                'total', m.total, 'correct', m.correct,
                'accuracy_pct', m.accuracy_pct
            ))
            FROM by_mode m
        ), '{}'::jsonb)
    ) INTO result
    FROM totals t;

    RETURN result;
END;
$$;
