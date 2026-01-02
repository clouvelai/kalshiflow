#!/bin/bash

# Kalshi Flow - Unified Local Database Backup Script
# Creates timestamped compressed backup of local Supabase database
# Supports both Docker and direct pg_dump methods

set -e

# Configuration
BACKUP_DIR="./backups/database_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="kalshi_backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "🗄️  Local Database Backup"
echo "========================="

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL environment variable is not set"
    echo "Please set DATABASE_URL or source your .env file first"
    exit 1
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Show what we're backing up
echo "📊 Pre-backup statistics:"
echo -n "   Sessions: "
psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM rl_orderbook_sessions;" 2>/dev/null || echo "0"
echo -n "   Snapshots: "
psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM rl_orderbook_snapshots;" 2>/dev/null || echo "0"
echo -n "   Deltas: "
psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM rl_orderbook_deltas;" 2>/dev/null || echo "0"
echo -n "   Trades: "
psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM trades;" 2>/dev/null || echo "0"

# Create compressed backup
echo ""
echo "🔄 Creating compressed backup..."

# Try Docker method first (if Supabase is running in Docker)
if docker ps | grep -q supabase_db_backend; then
    echo "   Using Docker container method..."
    if docker exec supabase_db_backend pg_dump -U postgres postgres | gzip > "$BACKUP_PATH"; then
        echo "✅ Backup created via Docker"
    else
        echo "❌ Docker method failed"
        exit 1
    fi
# Try Supabase CLI method
elif command -v supabase &> /dev/null && [ -d "backend" ]; then
    echo "   Using Supabase CLI method..."
    cd backend && supabase db dump --local | gzip > "../$BACKUP_PATH" && cd ..
    if [ $? -eq 0 ]; then
        echo "✅ Backup created via Supabase CLI"
    else
        echo "❌ Supabase CLI method failed"
        exit 1
    fi
# Fall back to direct pg_dump
else
    echo "   Using direct pg_dump method..."
    if pg_dump "$DATABASE_URL" | gzip > "$BACKUP_PATH"; then
        echo "✅ Backup created via direct pg_dump"
    else
        echo "❌ Direct pg_dump failed"
        exit 1
    fi
fi

# Verify backup file exists and has content
if [ ! -f "$BACKUP_PATH" ]; then
    echo "❌ Backup file was not created"
    exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
if [ "$BACKUP_SIZE" = "0B" ]; then
    echo "❌ Backup file is empty"
    rm "$BACKUP_PATH"
    exit 1
fi

# Show backup information
echo ""
echo "📦 Backup completed:"
echo "   Location: $BACKUP_PATH"
echo "   Size: $BACKUP_SIZE"

# Clean up old backups (keep last 7 days or last 10 backups, whichever is more)
echo ""
echo "🧹 Cleaning up old backups..."
# Delete backups older than 7 days
find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" -mtime +7 -delete 2>/dev/null || true
# Keep at least 10 most recent backups regardless of age
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" | wc -l)
if [ $BACKUP_COUNT -gt 10 ]; then
    find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" -type f -printf "%T+ %p\n" | \
        sort | head -n -10 | cut -d' ' -f2 | xargs rm -f 2>/dev/null || true
fi
REMAINING=$(find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" | wc -l)
echo "   Kept $REMAINING recent backup(s)"

echo ""
echo "✅ Backup complete!"
echo ""
echo "💡 To restore this backup:"
echo "   gunzip -c $BACKUP_PATH | psql \$DATABASE_URL"
echo ""
echo "   Or restore to a fresh database:"
echo "   createdb kalshi_restored"
echo "   gunzip -c $BACKUP_PATH | psql kalshi_restored"