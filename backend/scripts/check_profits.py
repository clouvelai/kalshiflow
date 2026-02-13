"""Quick check for profitable Captain decisions."""
import asyncio
import asyncpg

async def check():
    conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:54322/postgres")
    rows = await conn.fetch("""
        SELECT market_ticker, side, action, contracts, limit_price_cents,
               direction_correct, would_have_filled, hypothetical_pnl_cents,
               demo_status, demo_fill_count, cycle_mode, created_at
        FROM captain_decisions
        WHERE created_at >= now() - INTERVAL '3 hours'
        ORDER BY created_at DESC
    """)
    profitable = sum(1 for r in rows if r["hypothetical_pnl_cents"] and r["hypothetical_pnl_cents"] > 0)
    print(profitable)
    await conn.close()

asyncio.run(check())
