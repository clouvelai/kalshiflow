#!/usr/bin/env python3
"""
Historical Trades Analysis - Safe Read-Only Production Data Export

Analyzes all trades from production database to identify:
- Whale trades (high-value trades)
- High-leverage outliers (low price, high contract count)
- Profitable trading patterns

Safety Features:
- Read-only database access (no INSERT/UPDATE/DELETE)
- Minimal connection pool (1 connection, 30s timeout)
- Exports to CSV for local analysis

Usage:
    # Export all trades to CSV
    python analyze_historical_trades.py --export trades.csv

    # Show top 100 trades by size
    python analyze_historical_trades.py --top-trades 100

    # Find high leverage outliers (price < 20 cents, count > 100)
    python analyze_historical_trades.py --leverage-outliers

    # Full analysis report
    python analyze_historical_trades.py --analyze

    # Quick stats only
    python analyze_historical_trades.py --stats
"""

import asyncio
import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import asyncpg

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def get_read_only_connection():
    """
    Create a minimal, read-only database connection.

    Safety features:
    - Single connection (no pool)
    - 30 second command timeout
    - Read-only intent (no write methods exposed)
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Try loading from backend/.env (primary) then .env.production
        env_files = [
            Path(__file__).parent.parent.parent.parent / ".env",
            Path(__file__).parent.parent.parent.parent / ".env.production",
        ]
        for env_file in env_files:
            if env_file.exists():
                print(f"  Loading from {env_file.name}...")
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("DATABASE_URL=") and not line.startswith("#"):
                            database_url = line.split("=", 1)[1].strip()
                            break
                if database_url:
                    break

    if not database_url:
        raise ValueError("DATABASE_URL not found in environment or .env files")

    print(f"Connecting to database (read-only)...")
    conn = await asyncpg.connect(
        database_url,
        command_timeout=30,
        statement_cache_size=0,  # Disable cache for safety
    )

    # Verify connection with simple query
    result = await conn.fetchval("SELECT 1")
    print(f"Connected successfully.")

    return conn


async def get_database_stats(conn):
    """Get basic database statistics."""
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total_trades,
            COUNT(DISTINCT market_ticker) as unique_markets,
            MIN(ts) as oldest_ts,
            MAX(ts) as newest_ts,
            SUM(count) as total_contracts,
            SUM(count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0) as total_volume_dollars
        FROM trades
    """)

    oldest = datetime.fromtimestamp(stats['oldest_ts'] / 1000) if stats['oldest_ts'] else None
    newest = datetime.fromtimestamp(stats['newest_ts'] / 1000) if stats['newest_ts'] else None

    print(f"\n  Total trades:        {stats['total_trades']:,}")
    print(f"  Unique markets:      {stats['unique_markets']:,}")
    print(f"  Total contracts:     {stats['total_contracts']:,}")
    print(f"  Total volume:        ${stats['total_volume_dollars']:,.2f}")
    print(f"  Date range:          {oldest} to {newest}")
    if oldest and newest:
        print(f"  Span:                {(newest - oldest).days} days")

    return stats


async def export_trades_to_csv(conn, output_path: str, limit: int = None):
    """Export all trades to CSV with computed fields."""
    print(f"\n" + "=" * 80)
    print(f"EXPORTING TRADES TO CSV: {output_path}")
    print("=" * 80)

    query = """
        SELECT
            id,
            market_ticker,
            taker_side,
            count,
            yes_price,
            no_price,
            ts,
            -- Computed fields
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END as trade_price,
            count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 as cost_dollars,
            count as max_payout_dollars,
            count * (100 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 as potential_profit_dollars,
            CASE
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END > 0
                THEN (100.0 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) /
                     CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END
                ELSE 0
            END as leverage_ratio
        FROM trades
        ORDER BY ts DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    print(f"  Fetching trades...")
    rows = await conn.fetch(query)
    print(f"  Retrieved {len(rows):,} trades")

    # Write to CSV
    print(f"  Writing to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'id', 'market_ticker', 'taker_side', 'count', 'yes_price', 'no_price',
            'timestamp', 'datetime', 'trade_price', 'cost_dollars', 'max_payout_dollars',
            'potential_profit_dollars', 'leverage_ratio'
        ])
        # Data
        for row in rows:
            dt = datetime.fromtimestamp(row['ts'] / 1000)
            writer.writerow([
                row['id'],
                row['market_ticker'],
                row['taker_side'],
                row['count'],
                row['yes_price'],
                row['no_price'],
                row['ts'],
                dt.isoformat(),
                row['trade_price'],
                float(row['cost_dollars']),
                float(row['max_payout_dollars']),
                float(row['potential_profit_dollars']),
                float(row['leverage_ratio'])
            ])

    print(f"  Exported to {output_path}")
    return len(rows)


async def get_top_trades(conn, limit: int = 100, order_by: str = 'cost'):
    """Get top trades by various metrics."""
    print(f"\n" + "=" * 80)
    print(f"TOP {limit} TRADES BY {order_by.upper()}")
    print("=" * 80)

    order_clause = {
        'cost': 'cost_dollars DESC',
        'contracts': 'count DESC',
        'profit_potential': 'potential_profit_dollars DESC',
        'leverage': 'leverage_ratio DESC',
    }.get(order_by, 'cost_dollars DESC')

    query = f"""
        SELECT
            id,
            market_ticker,
            taker_side,
            count,
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END as trade_price,
            count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 as cost_dollars,
            count as max_payout_dollars,
            count * (100 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 as potential_profit_dollars,
            CASE
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END > 0
                THEN (100.0 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) /
                     CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END
                ELSE 0
            END as leverage_ratio,
            ts
        FROM trades
        ORDER BY {order_clause}
        LIMIT {limit}
    """

    rows = await conn.fetch(query)

    print(f"\n{'#':>4} {'Market':<30} {'Side':<4} {'Count':>8} {'Price':>6} {'Cost':>12} {'Profit Pot':>12} {'Leverage':>8} {'Time'}")
    print("-" * 120)

    for i, row in enumerate(rows, 1):
        dt = datetime.fromtimestamp(row['ts'] / 1000)
        print(f"{i:>4} {row['market_ticker'][:30]:<30} {row['taker_side']:<4} "
              f"{row['count']:>8,} {row['trade_price']:>5}c ${float(row['cost_dollars']):>10,.2f} "
              f"${float(row['potential_profit_dollars']):>10,.2f} {float(row['leverage_ratio']):>7.1f}x "
              f"{dt.strftime('%m/%d %H:%M')}")

    return rows


async def find_leverage_outliers(conn, max_price: int = 20, min_contracts: int = 100):
    """Find high-leverage trades (low price, high contract count)."""
    print(f"\n" + "=" * 80)
    print(f"HIGH LEVERAGE OUTLIERS (price <= {max_price}c, contracts >= {min_contracts})")
    print("=" * 80)

    query = f"""
        SELECT
            id,
            market_ticker,
            taker_side,
            count,
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END as trade_price,
            count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 as cost_dollars,
            count as max_payout_dollars,
            count * (100 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0 as potential_profit_dollars,
            CASE
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END > 0
                THEN (100.0 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) /
                     CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END
                ELSE 0
            END as leverage_ratio,
            ts
        FROM trades
        WHERE
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= {max_price}
            AND count >= {min_contracts}
        ORDER BY potential_profit_dollars DESC
        LIMIT 100
    """

    rows = await conn.fetch(query)

    print(f"\n  Found {len(rows)} high-leverage trades")
    print(f"\n{'#':>4} {'Market':<35} {'Side':<4} {'Count':>8} {'Price':>6} {'Cost':>10} {'Max Profit':>12} {'Leverage':>8} {'Time'}")
    print("-" * 130)

    for i, row in enumerate(rows, 1):
        dt = datetime.fromtimestamp(row['ts'] / 1000)
        print(f"{i:>4} {row['market_ticker'][:35]:<35} {row['taker_side']:<4} "
              f"{row['count']:>8,} {row['trade_price']:>5}c ${float(row['cost_dollars']):>8,.2f} "
              f"${float(row['potential_profit_dollars']):>10,.2f} {float(row['leverage_ratio']):>7.1f}x "
              f"{dt.strftime('%m/%d %H:%M')}")

    # Summary stats
    if rows:
        total_cost = sum(float(r['cost_dollars']) for r in rows)
        total_profit_potential = sum(float(r['potential_profit_dollars']) for r in rows)
        total_contracts = sum(r['count'] for r in rows)

        print(f"\n  SUMMARY:")
        print(f"    Total trades:           {len(rows)}")
        print(f"    Total contracts:        {total_contracts:,}")
        print(f"    Total cost:             ${total_cost:,.2f}")
        print(f"    Total profit potential: ${total_profit_potential:,.2f}")
        print(f"    Average leverage:       {total_profit_potential/total_cost:.1f}x" if total_cost > 0 else "")

    return rows


async def find_whale_million_trade(conn):
    """Search for trades approaching or exceeding $1M in size."""
    print(f"\n" + "=" * 80)
    print("SEARCHING FOR 1M+ TRADES")
    print("=" * 80)

    # Check by cost (amount risked)
    print("\n  Checking by COST (amount risked)...")
    query = """
        SELECT
            market_ticker,
            taker_side,
            count,
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END as trade_price,
            count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 as cost_dollars,
            count as max_payout_dollars,
            ts
        FROM trades
        WHERE count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 >= 100000
        ORDER BY cost_dollars DESC
        LIMIT 50
    """

    cost_rows = await conn.fetch(query)
    print(f"  Found {len(cost_rows)} trades with cost >= $100k")

    # Check by payout potential (contracts = max payout in dollars)
    print("\n  Checking by PAYOUT (max payout if win)...")
    query2 = """
        SELECT
            market_ticker,
            taker_side,
            count,
            CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END as trade_price,
            count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0 as cost_dollars,
            count as max_payout_dollars,
            ts
        FROM trades
        WHERE count >= 100000
        ORDER BY count DESC
        LIMIT 50
    """

    payout_rows = await conn.fetch(query2)
    print(f"  Found {len(payout_rows)} trades with payout potential >= $100k")

    # Combined output of biggest trades
    all_big = list(cost_rows) + list(payout_rows)
    # Dedupe by id-like characteristics
    seen = set()
    unique_big = []
    for row in all_big:
        key = (row['market_ticker'], row['count'], row['ts'])
        if key not in seen:
            seen.add(key)
            unique_big.append(row)

    # Sort by max of cost or payout
    unique_big.sort(key=lambda x: max(float(x['cost_dollars']), float(x['max_payout_dollars'])), reverse=True)

    print(f"\n  TOP 20 BIGGEST TRADES (by max(cost, payout)):")
    print(f"\n{'#':>3} {'Market':<40} {'Side':<4} {'Contracts':>12} {'Price':>6} {'Cost':>14} {'Payout':>14} {'Time'}")
    print("-" * 120)

    for i, row in enumerate(unique_big[:20], 1):
        dt = datetime.fromtimestamp(row['ts'] / 1000)
        print(f"{i:>3} {row['market_ticker'][:40]:<40} {row['taker_side']:<4} "
              f"{row['count']:>12,} {row['trade_price']:>5}c ${float(row['cost_dollars']):>12,.2f} "
              f"${float(row['max_payout_dollars']):>12,.2f} {dt.strftime('%m/%d %H:%M')}")

    return unique_big


async def analyze_whale_patterns(conn):
    """Analyze patterns in whale trades."""
    print(f"\n" + "=" * 80)
    print("WHALE TRADE PATTERN ANALYSIS")
    print("=" * 80)

    # Define whale as top 1% by size
    print("\n  1. WHALE THRESHOLD ANALYSIS")
    print("  " + "-" * 40)

    percentiles = await conn.fetch("""
        SELECT
            percentile_cont(0.90) WITHIN GROUP (ORDER BY count) as p90,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY count) as p95,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY count) as p99,
            percentile_cont(0.999) WITHIN GROUP (ORDER BY count) as p999,
            MAX(count) as max_count,
            AVG(count) as avg_count
        FROM trades
    """)

    p = percentiles[0]
    print(f"    Average contracts:    {float(p['avg_count']):,.1f}")
    print(f"    90th percentile:      {float(p['p90']):,.0f} contracts")
    print(f"    95th percentile:      {float(p['p95']):,.0f} contracts")
    print(f"    99th percentile:      {float(p['p99']):,.0f} contracts")
    print(f"    99.9th percentile:    {float(p['p999']):,.0f} contracts")
    print(f"    Maximum:              {float(p['max_count']):,.0f} contracts")

    # Whale trades by market
    print("\n  2. MARKETS WITH MOST WHALE ACTIVITY (>1000 contracts)")
    print("  " + "-" * 40)

    whale_markets = await conn.fetch("""
        SELECT
            market_ticker,
            COUNT(*) as whale_count,
            SUM(count) as total_contracts,
            AVG(count) as avg_contracts,
            SUM(count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0) as total_cost
        FROM trades
        WHERE count >= 1000
        GROUP BY market_ticker
        ORDER BY whale_count DESC
        LIMIT 20
    """)

    print(f"\n    {'Market':<40} {'Whale Trades':>12} {'Total Contracts':>15} {'Avg Size':>10} {'Total Cost':>14}")
    print("    " + "-" * 95)

    for row in whale_markets:
        print(f"    {row['market_ticker'][:40]:<40} {row['whale_count']:>12,} "
              f"{row['total_contracts']:>15,} {float(row['avg_contracts']):>10,.0f} "
              f"${float(row['total_cost']):>12,.2f}")

    # Time of day analysis
    print("\n  3. WHALE TRADE TIMING (>1000 contracts)")
    print("  " + "-" * 40)

    timing = await conn.fetch("""
        SELECT
            EXTRACT(HOUR FROM to_timestamp(ts / 1000)) as hour,
            COUNT(*) as trade_count,
            SUM(count) as total_contracts,
            AVG(count) as avg_size
        FROM trades
        WHERE count >= 1000
        GROUP BY hour
        ORDER BY hour
    """)

    print(f"\n    {'Hour (UTC)':>10} {'Trades':>10} {'Contracts':>15} {'Avg Size':>12}")
    print("    " + "-" * 50)

    for row in timing:
        hour = int(row['hour'])
        print(f"    {hour:>10}:00 {row['trade_count']:>10,} {row['total_contracts']:>15,} {float(row['avg_size']):>12,.0f}")

    # Side bias in whale trades
    print("\n  4. WHALE TRADE SIDE ANALYSIS (>1000 contracts)")
    print("  " + "-" * 40)

    sides = await conn.fetch("""
        SELECT
            taker_side,
            COUNT(*) as trade_count,
            SUM(count) as total_contracts,
            AVG(CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) as avg_price,
            SUM(count * CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END / 100.0) as total_cost
        FROM trades
        WHERE count >= 1000
        GROUP BY taker_side
    """)

    for row in sides:
        print(f"\n    {row['taker_side'].upper()} side:")
        print(f"      Trades:         {row['trade_count']:,}")
        print(f"      Contracts:      {row['total_contracts']:,}")
        print(f"      Avg price:      {float(row['avg_price']):.1f} cents")
        print(f"      Total cost:     ${float(row['total_cost']):,.2f}")

    # Price distribution
    print("\n  5. WHALE PRICE DISTRIBUTION (>1000 contracts)")
    print("  " + "-" * 40)

    price_dist = await conn.fetch("""
        SELECT
            CASE
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 10 THEN '1-10c (extreme longshot)'
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 25 THEN '11-25c (longshot)'
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 40 THEN '26-40c (underdog)'
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 60 THEN '41-60c (toss-up)'
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 75 THEN '61-75c (favorite)'
                WHEN CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END <= 90 THEN '76-90c (strong favorite)'
                ELSE '91-100c (near-certain)'
            END as price_bucket,
            COUNT(*) as trade_count,
            SUM(count) as total_contracts,
            SUM(count * (100 - CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END) / 100.0) as profit_potential
        FROM trades
        WHERE count >= 1000
        GROUP BY price_bucket
        ORDER BY MIN(CASE WHEN taker_side = 'yes' THEN yes_price ELSE no_price END)
    """)

    print(f"\n    {'Price Bucket':<30} {'Trades':>10} {'Contracts':>15} {'Profit Potential':>18}")
    print("    " + "-" * 75)

    for row in price_dist:
        print(f"    {row['price_bucket']:<30} {row['trade_count']:>10,} "
              f"{row['total_contracts']:>15,} ${float(row['profit_potential']):>16,.2f}")


async def generate_full_analysis(conn):
    """Generate complete analysis report."""
    print("\n" + "=" * 80)
    print("FULL HISTORICAL TRADES ANALYSIS")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")

    # Stats
    await get_database_stats(conn)

    # Big trades search
    await find_whale_million_trade(conn)

    # High leverage
    await find_leverage_outliers(conn)

    # Patterns
    await analyze_whale_patterns(conn)

    # Top trades
    await get_top_trades(conn, limit=50, order_by='cost')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze historical trades from production database (READ-ONLY)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_historical_trades.py --stats              # Quick database stats
  python analyze_historical_trades.py --export trades.csv  # Export to CSV
  python analyze_historical_trades.py --top-trades 100     # Top 100 trades
  python analyze_historical_trades.py --leverage-outliers  # High leverage trades
  python analyze_historical_trades.py --million            # Find 1M+ trades
  python analyze_historical_trades.py --analyze            # Full analysis
        """
    )

    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics only')
    parser.add_argument('--export', type=str, metavar='FILE',
                       help='Export trades to CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of trades to export')
    parser.add_argument('--top-trades', type=int, metavar='N',
                       help='Show top N trades by cost')
    parser.add_argument('--order-by', type=str, default='cost',
                       choices=['cost', 'contracts', 'profit_potential', 'leverage'],
                       help='Order top trades by (default: cost)')
    parser.add_argument('--leverage-outliers', action='store_true',
                       help='Find high leverage outliers')
    parser.add_argument('--million', action='store_true',
                       help='Search for million dollar trades')
    parser.add_argument('--analyze', action='store_true',
                       help='Run full analysis')
    parser.add_argument('--patterns', action='store_true',
                       help='Analyze whale patterns only')

    args = parser.parse_args()

    # Default to stats if no args
    if not any([args.stats, args.export, args.top_trades, args.leverage_outliers,
                args.million, args.analyze, args.patterns]):
        args.stats = True

    try:
        conn = await get_read_only_connection()

        try:
            if args.stats:
                await get_database_stats(conn)

            if args.export:
                await export_trades_to_csv(conn, args.export, args.limit)

            if args.top_trades:
                await get_top_trades(conn, args.top_trades, args.order_by)

            if args.leverage_outliers:
                await find_leverage_outliers(conn)

            if args.million:
                await find_whale_million_trade(conn)

            if args.patterns:
                await analyze_whale_patterns(conn)

            if args.analyze:
                await generate_full_analysis(conn)

        finally:
            await conn.close()
            print("\n  Database connection closed.")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
