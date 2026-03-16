"""
Production Supabase DB status check and vacuum-aware trades cleanup.

Connects to the production database, reports table sizes and trade age
distribution, checks the pg_cron cleanup job, and optionally runs batched
deletes with periodic VACUUM to reclaim dead tuple space.

Usage:
    cd backend && export $(grep -E '^DATABASE_URL=' .env.production | head -1)
    uv run python ../scripts/db_cleanup.py --status-only
    uv run python ../scripts/db_cleanup.py --cleanup
    uv run python ../scripts/db_cleanup.py --cleanup --retention-days 7 --batch-size 500000 --max-batches 200
    uv run python ../scripts/db_cleanup.py --cleanup --vacuum-every 10 --dead-tuple-limit 5000000
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone


async def get_connection():
    import asyncpg

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set. Run:")
        print("  export $(grep -E '^DATABASE_URL=' .env.production | head -1)")
        sys.exit(1)
    conn = await asyncpg.connect(url, timeout=30, command_timeout=600)
    # Supabase has a server-side statement_timeout that can kill long queries.
    # Override for this session so VACUUM and large batch deletes don't get canceled.
    await conn.execute("SET statement_timeout = '30min'")
    return conn


async def report_status(conn):
    print("=" * 65)
    print("  KALSHIFLOW PRODUCTION DB STATUS")
    print("=" * 65)

    # Table sizes
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    )
    print(f"\n{'Table':<30} {'Size':<15} {'~Rows':<15}")
    print("-" * 60)
    total_bytes = 0
    for t in tables:
        name = t["tablename"]
        size = await conn.fetchval(
            f"SELECT pg_size_pretty(pg_total_relation_size('{name}'))"
        )
        size_bytes = await conn.fetchval(
            f"SELECT pg_total_relation_size('{name}')"
        )
        rows = await conn.fetchval(
            f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{name}' AND relkind = 'r'"
        )
        total_bytes += size_bytes or 0
        print(f"{name:<30} {size:<15} {rows if rows else 0:<15,}")

    total_pretty = await conn.fetchval(
        f"SELECT pg_size_pretty({total_bytes}::bigint)"
    )
    print(f"\n{'TOTAL':<30} {total_pretty}")

    # Trades age distribution — use estimates to avoid full table scans
    print("\n" + "-" * 60)
    print("TRADES TABLE DETAILS")
    print("-" * 60)

    try:
        approx_rows = await conn.fetchval(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = 'trades' AND relkind = 'r'"
        )
        size_bytes = await conn.fetchval(
            "SELECT pg_total_relation_size('trades')"
        )
        size_pretty = await conn.fetchval(
            "SELECT pg_size_pretty(pg_total_relation_size('trades'))"
        )
        print(f"  Approx rows:   {approx_rows:,}")
        print(f"  Table size:    {size_pretty}")

        # Sample newest trade (fast — reads last page)
        newest = await conn.fetchval(
            "SELECT ts FROM trades ORDER BY ts DESC LIMIT 1"
        )
        if newest:
            newest_dt = datetime.fromtimestamp(newest / 1000, tz=timezone.utc)
            print(f"  Newest trade:  {newest_dt.strftime('%Y-%m-%d %H:%M UTC')}")

        # Estimate rows to delete using table stats
        # (exact count is too slow on 87M rows without an index)
        if approx_rows and size_bytes:
            avg_row_bytes = size_bytes / approx_rows if approx_rows > 0 else 0
            print(f"  Avg row size:  ~{avg_row_bytes:.0f} bytes")

        # Dead tuples and DB size
        dead_tup = await conn.fetchval(
            "SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = 'trades'"
        )
        db_size = await conn.fetchval(
            "SELECT pg_size_pretty(pg_database_size(current_database()))"
        )
        print(f"  Dead tuples:   {dead_tup:,}" if dead_tup else "  Dead tuples:   0")
        print(f"  Database size: {db_size}")
    except Exception as e:
        print(f"  Error querying trades: {e}")

    # Cron job status
    print("\n" + "-" * 60)
    print("CRON JOBS")
    print("-" * 60)
    try:
        ext = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_cron')"
        )
        if ext:
            jobs = await conn.fetch(
                "SELECT jobid, jobname, schedule, active FROM cron.job"
            )
            if jobs:
                for j in jobs:
                    status = "ACTIVE" if j["active"] else "INACTIVE"
                    print(f"  [{j['jobid']}] {j['jobname']} | {j['schedule']} | {status}")
            else:
                print("  No cron jobs scheduled")

            # Recent cron history
            try:
                history = await conn.fetch(
                    "SELECT jobid, status, start_time, end_time "
                    "FROM cron.job_run_details "
                    "ORDER BY start_time DESC LIMIT 5"
                )
                if history:
                    print("\n  Recent cron runs:")
                    for h in history:
                        start = h["start_time"].strftime("%Y-%m-%d %H:%M UTC") if h["start_time"] else "?"
                        end = h["end_time"].strftime("%H:%M") if h["end_time"] else "running"
                        print(f"    job {h['jobid']} | {h['status']} | {start} -> {end}")
            except Exception:
                pass  # cron.job_run_details may not exist
        else:
            print("  pg_cron extension not installed")
    except Exception as e:
        print(f"  Error checking cron: {e}")

    # Cleanup function check
    print("\n" + "-" * 60)
    print("CLEANUP FUNCTIONS")
    print("-" * 60)
    for fn in ["cleanup_old_trades", "cleanup_old_trades_loop"]:
        exists = await conn.fetchval(
            f"SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = '{fn}')"
        )
        print(f"  {fn}: {'deployed' if exists else 'MISSING'}")

    print("\n" + "=" * 65)


async def get_dead_tuples(conn):
    """Query dead tuple count for the trades table."""
    return await conn.fetchval(
        "SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = 'trades'"
    ) or 0


async def get_db_size(conn):
    """Get human-readable database size."""
    return await conn.fetchval(
        "SELECT pg_size_pretty(pg_database_size(current_database()))"
    )


async def run_vacuum(conn, reason="scheduled"):
    """Run VACUUM on trades table. Disables statement_timeout since VACUUM on
    large dead tuple counts can take 10+ minutes, then restores it."""
    print(f"\n  >>> VACUUM trades ({reason})...")
    vac_start = time.time()
    # Supabase has statement_timeout=2min; VACUUM needs much longer
    await conn.execute("SET statement_timeout = 0")
    try:
        await conn.execute("VACUUM trades", timeout=3600)
    finally:
        await conn.execute("SET statement_timeout = '30min'")
    vac_elapsed = time.time() - vac_start
    dead_after = await get_dead_tuples(conn)
    db_size = await get_db_size(conn)
    print(f"  >>> VACUUM complete in {vac_elapsed:.1f}s | dead tuples: {dead_after:,} | DB size: {db_size}\n")


async def run_cleanup(conn, retention_days, batch_size, max_batches, pause_secs,
                      vacuum_every=10, dead_tuple_limit=5_000_000):
    # Verify function exists
    exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'cleanup_old_trades')"
    )
    if not exists:
        print("ERROR: cleanup_old_trades() function not found in production DB.")
        print("Push the migration first: supabase db push")
        return

    # Count rows to delete
    cutoff_ms = int((time.time() - retention_days * 86400) * 1000)
    cutoff_dt = datetime.fromtimestamp(cutoff_ms / 1000, tz=timezone.utc)

    db_size = await get_db_size(conn)
    dead_tuples = await get_dead_tuples(conn)

    print(f"\nCLEANUP CONFIG:")
    print(f"  Retention:        {retention_days} days (cutoff: {cutoff_dt.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"  Batch size:       {batch_size:,}")
    print(f"  Max batches:      {max_batches}")
    print(f"  Pause:            {pause_secs}s between batches")
    print(f"  Vacuum every:     {vacuum_every} batches")
    print(f"  Dead tuple limit: {dead_tuple_limit:,}")
    print(f"  DB size (start):  {db_size}")
    print(f"  Dead tuples:      {dead_tuples:,}")
    print()

    total_deleted = 0
    batch_num = 0
    start_time = time.time()

    for i in range(max_batches):
        batch_num = i + 1

        # Pre-batch dead tuple check — force VACUUM if over limit
        if i > 0:
            dead_tuples = await get_dead_tuples(conn)
            if dead_tuples > dead_tuple_limit:
                await run_vacuum(conn, reason=f"dead tuples {dead_tuples:,} > limit {dead_tuple_limit:,}")

        batch_start = time.time()

        deleted = await conn.fetchval(
            "SELECT public.cleanup_old_trades($1, $2)",
            retention_days,
            batch_size,
        )

        batch_elapsed = time.time() - batch_start
        total_deleted += deleted
        total_elapsed = time.time() - start_time

        rate = total_deleted / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  Batch {batch_num:>3}: deleted {deleted:>10,} rows "
            f"({batch_elapsed:.1f}s) | "
            f"Total: {total_deleted:>12,} | "
            f"Rate: {rate:,.0f} rows/s"
        )

        if deleted == 0:
            print("\n  No more rows to delete — cleanup complete!")
            break

        # Periodic VACUUM after every N batches
        if batch_num % vacuum_every == 0:
            await run_vacuum(conn, reason=f"periodic (every {vacuum_every} batches)")

        if i < max_batches - 1:
            await asyncio.sleep(pause_secs)
    else:
        print(f"\n  Reached max batches ({max_batches}). More rows may remain.")

    # Final VACUUM to clean up remaining dead tuples
    await run_vacuum(conn, reason="final cleanup")

    elapsed = time.time() - start_time
    db_size_end = await get_db_size(conn)
    print(f"{'=' * 60}")
    print(f"CLEANUP SUMMARY")
    print(f"  Total deleted:  {total_deleted:,} rows")
    print(f"  Batches run:    {batch_num}")
    print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Avg rate:       {total_deleted/elapsed:,.0f} rows/s" if elapsed > 0 else "")
    print(f"  DB size (end):  {db_size_end}")
    print(f"{'=' * 60}")


async def main():
    parser = argparse.ArgumentParser(description="KalshiFlow production DB status & cleanup")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status-only", action="store_true", help="Just report DB status")
    group.add_argument("--cleanup", action="store_true", help="Run aggressive batched cleanup")
    parser.add_argument("--retention-days", type=int, default=7, help="Days of data to keep (default: 7)")
    parser.add_argument("--batch-size", type=int, default=500000, help="Rows per batch (default: 500000)")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches to run (default: 200)")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between batches (default: 0.5)")
    parser.add_argument("--vacuum-every", type=int, default=10, help="Run VACUUM after every N batches (default: 10)")
    parser.add_argument("--dead-tuple-limit", type=int, default=5_000_000, help="Force VACUUM if dead tuples exceed this (default: 5000000)")
    args = parser.parse_args()

    conn = await get_connection()
    try:
        await report_status(conn)

        if args.cleanup:
            await run_cleanup(conn, args.retention_days, args.batch_size, args.max_batches, args.pause,
                              args.vacuum_every, args.dead_tuple_limit)
            # Report final status after cleanup
            print("\n--- POST-CLEANUP STATUS ---\n")
            await report_status(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
