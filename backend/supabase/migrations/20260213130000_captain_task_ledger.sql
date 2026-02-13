-- Captain task ledger persistence
-- Stores TaskLedger snapshots per cycle for debugging and analysis

CREATE TABLE IF NOT EXISTS captain_task_ledger (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    cycle INT,
    tasks JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_task_ledger_session ON captain_task_ledger(session_id, created_at DESC);
