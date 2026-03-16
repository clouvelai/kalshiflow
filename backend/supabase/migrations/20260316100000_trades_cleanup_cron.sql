-- Automatic cleanup of trades older than 7 days.
-- Runs daily at 04:00 UTC via pg_cron.
-- Deletes in batches of 500k to avoid long locks and WAL bloat.
-- Designed for steady-state maintenance (~1 day of new trades per run),
-- NOT for the initial 34GB backlog cleanup (handle that separately).

-- Enable pg_cron (Supabase has it pre-installed, just needs enabling)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Grant usage so cron jobs can hit the public schema
GRANT USAGE ON SCHEMA cron TO postgres;

-- Batched delete function: deletes up to 500k rows per call,
-- returns how many were deleted so callers can loop if needed.
CREATE OR REPLACE FUNCTION public.cleanup_old_trades(retention_days INTEGER DEFAULT 7, batch_size INTEGER DEFAULT 500000)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    cutoff_ms BIGINT;
    deleted_count INTEGER;
BEGIN
    -- trades.ts is BIGINT milliseconds since epoch
    cutoff_ms := (EXTRACT(EPOCH FROM now()) - retention_days * 86400)::BIGINT * 1000;

    -- Delete a bounded batch to keep lock time short
    WITH to_delete AS (
        SELECT ctid FROM trades
        WHERE ts < cutoff_ms
        LIMIT batch_size
    )
    DELETE FROM trades
    WHERE ctid IN (SELECT ctid FROM to_delete);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE LOG 'cleanup_old_trades: deleted % rows (cutoff=%, retention=% days)',
        deleted_count, cutoff_ms, retention_days;

    RETURN deleted_count;
END;
$$;

-- Wrapper that loops batches until done (max 20 iterations = 10M rows per run).
-- This keeps each individual DELETE small while still clearing a full day's worth.
CREATE OR REPLACE FUNCTION public.cleanup_old_trades_loop()
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    batch_deleted INTEGER;
    total_deleted INTEGER := 0;
    max_iterations INTEGER := 20;
    i INTEGER := 0;
BEGIN
    LOOP
        SELECT public.cleanup_old_trades(7, 500000) INTO batch_deleted;
        total_deleted := total_deleted + batch_deleted;
        i := i + 1;

        EXIT WHEN batch_deleted = 0 OR i >= max_iterations;

        -- Brief pause between batches to let other queries through
        PERFORM pg_sleep(1);
    END LOOP;

    RAISE LOG 'cleanup_old_trades_loop: total deleted % rows in % batches', total_deleted, i;
END;
$$;

-- Schedule: daily at 04:00 UTC
SELECT cron.schedule(
    'cleanup-old-trades',           -- job name
    '0 4 * * *',                    -- cron expression: 4am UTC daily
    'SELECT public.cleanup_old_trades_loop()'
);

-- Note: VACUUM cannot run via pg_cron (it wraps commands in a transaction block).
-- Autovacuum handles dead tuple cleanup automatically. The batched deletes with
-- pg_sleep(1) pauses give autovacuum enough breathing room to keep up with the
-- daily cleanup volume (~10M rows). For the aggressive backlog cleanup script
-- (scripts/db_cleanup.py), VACUUM is run explicitly between batches.
