-- Add 24h volume columns to paired_markets for pair index sorting
ALTER TABLE paired_markets
ADD COLUMN IF NOT EXISTS kalshi_volume_24h INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS poly_volume_24h INT DEFAULT 0;
