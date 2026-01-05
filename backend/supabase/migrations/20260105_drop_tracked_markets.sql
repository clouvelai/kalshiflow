-- Migration: Drop tracked_markets table
-- Reason: Tracked markets are now discovered fresh each startup via lifecycle WebSocket
--         and API discovery. DB persistence caused stale data issues (prod vs demo mismatch).
-- Date: 2026-01-05
--
-- This migration removes:
-- 1. The tracked_markets table
-- 2. All associated indexes
-- 3. All associated constraints

-- Drop indexes first (they depend on the table)
DROP INDEX IF EXISTS idx_tracked_markets_status;
DROP INDEX IF EXISTS idx_tracked_markets_category;
DROP INDEX IF EXISTS idx_tracked_markets_status_category;

-- Drop the table (CASCADE will handle any remaining dependencies)
DROP TABLE IF EXISTS tracked_markets CASCADE;

-- Verify cleanup
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tracked_markets') THEN
        RAISE EXCEPTION 'tracked_markets table still exists after migration';
    END IF;
    RAISE NOTICE 'Successfully dropped tracked_markets table';
END $$;
