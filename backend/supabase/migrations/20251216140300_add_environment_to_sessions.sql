-- Add environment field to rl_orderbook_sessions table
-- Track environment (local/production/paper) for data source identification
-- NOTE: This migration is conditional - only runs if the table exists

DO $$
BEGIN
    -- Only add column if table exists
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'rl_orderbook_sessions') THEN
        -- Add environment column if it doesn't exist
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'rl_orderbook_sessions' AND column_name = 'environment') THEN
            ALTER TABLE rl_orderbook_sessions ADD COLUMN environment TEXT;

            -- Update existing sessions with assumed local environment
            UPDATE rl_orderbook_sessions SET environment = 'local' WHERE environment IS NULL;

            -- Add constraint for valid environment values
            ALTER TABLE rl_orderbook_sessions
            ADD CONSTRAINT chk_sessions_environment
            CHECK (environment IN ('local', 'production', 'paper', 'test'));

            -- Create index for environment-based queries
            CREATE INDEX IF NOT EXISTS idx_sessions_environment
            ON rl_orderbook_sessions(environment, started_at DESC);

            -- Add comment for the new field
            COMMENT ON COLUMN rl_orderbook_sessions.environment IS 'Environment where session was collected (local/production/paper)';
        END IF;
    END IF;
END $$;
