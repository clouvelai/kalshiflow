-- Fix ticker format constraint to allow periods in Kalshi tickers
-- Issue: Original constraint was too restrictive, blocking legitimate tickers like KXBTCD-25DEC0515-T89499.99

-- Drop the existing overly restrictive constraint
ALTER TABLE markets DROP CONSTRAINT IF EXISTS chk_ticker_format;

-- Add updated constraint that allows periods (.) in ticker symbols
-- Pattern now allows: Letters, numbers, periods, underscores, dashes
ALTER TABLE markets ADD CONSTRAINT chk_ticker_format
    CHECK (ticker ~ '^[A-Z0-9._-]+$');

-- Document the change
INSERT INTO schema_info (version, description) 
VALUES ('1.6.1-ticker-fix', 'Fixed ticker format constraint to allow periods for price/decimal tickers')
ON CONFLICT DO NOTHING;