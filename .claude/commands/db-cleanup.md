# Supabase Production DB Status & Cleanup

Check the production Supabase database health and optionally run aggressive backlog cleanup on the `trades` table.

## What This Does

1. Connects to the production Supabase DB (KalshiFlow project `fnsbruyvocdefnhzjiyk`)
2. Reports table sizes, row counts, and the `trades` table age distribution
3. Checks the `cleanup-old-trades` cron job status
4. Optionally runs aggressive batched cleanup to clear old trades

## How to Run

Run the status check and cleanup script:

```bash
cd /Users/samuelclark/Desktop/kalshiflow/backend && \
  export $(grep -E '^DATABASE_URL=' .env.production | head -1) && \
  uv run python ../scripts/db_cleanup.py
```

### Arguments

- `--status-only` — Just report DB status, don't delete anything
- `--cleanup` — Run aggressive batched cleanup (default: 7-day retention, 500K batch size)
- `--retention-days N` — Override retention period (default: 7)
- `--batch-size N` — Override batch size (default: 500000)
- `--max-batches N` — Max batches to run (default: 200, i.e. up to 100M rows)
- `--pause N` — Seconds to pause between batches (default: 0.5)

### Examples

```bash
# Just check status
uv run python ../scripts/db_cleanup.py --status-only

# Aggressive cleanup with defaults (7 days, 500K batches, up to 200 batches)
uv run python ../scripts/db_cleanup.py --cleanup

# Custom: 3-day retention, 1M batch, 50 batches max
uv run python ../scripts/db_cleanup.py --cleanup --retention-days 3 --batch-size 1000000 --max-batches 50
```

## Production Context

- **Database**: Supabase PostgreSQL (West US Oregon)
- **Main table**: `trades` (~87M rows, ~34GB as of 2026-03-16)
- **Cron job**: `cleanup-old-trades` runs daily at 4:00 AM UTC (10M rows max/run)
- **Backlog note**: The daily cron handles steady-state (~1 day of new trades). For the initial 34GB backlog, use `--cleanup` with high `--max-batches` to clear aggressively.
- **`ts` column**: BIGINT milliseconds since epoch (Kalshi trade timestamp)

## Safety

- Each batch deletes at most `batch_size` rows using `ctid`-based bounded deletes
- Pauses between batches to avoid locking out production queries
- Reports progress after every batch so you can Ctrl+C safely
- The `cleanup_old_trades()` function is already deployed to production via migration `20260316100000`
