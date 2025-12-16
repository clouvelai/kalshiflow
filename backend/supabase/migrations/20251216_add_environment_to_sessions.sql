-- Add environment field to rl_orderbook_sessions table
-- Track environment (local/production/paper) for data source identification

ALTER TABLE rl_orderbook_sessions 
ADD COLUMN IF NOT EXISTS environment TEXT;

-- Add comment for the new field
COMMENT ON COLUMN rl_orderbook_sessions.environment IS 'Environment where session was collected (local/production/paper)';

-- Update existing sessions with assumed local environment
-- Safe assumption since production hasn't been deployed yet
UPDATE rl_orderbook_sessions 
SET environment = 'local'
WHERE environment IS NULL;

-- Add constraint for valid environment values
ALTER TABLE rl_orderbook_sessions
ADD CONSTRAINT chk_sessions_environment 
CHECK (environment IN ('local', 'production', 'paper', 'test'));

-- Create index for environment-based queries
CREATE INDEX IF NOT EXISTS idx_sessions_environment 
ON rl_orderbook_sessions(environment, started_at DESC);