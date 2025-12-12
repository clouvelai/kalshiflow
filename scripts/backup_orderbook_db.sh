#!/bin/bash

# Simple database backup script for orderbook training data
# Creates a timestamped backup that can be restored if needed

set -e

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL environment variable not set"
    echo "Please source your .env file first: source .env"
    exit 1
fi

# Create backups directory if it doesn't exist
BACKUP_DIR="./database_backups"
mkdir -p "$BACKUP_DIR"

# Generate timestamp for backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/kalshiflow_backup_${TIMESTAMP}.sql"

# Perform the backup
echo "📦 Starting database backup..."
echo "   Backing up to: $BACKUP_FILE"

# Use supabase CLI for proper version compatibility with local database
cd backend && supabase db dump --data-only --local > "../$BACKUP_FILE" && cd ..

# Check if backup was successful
if [ $? -eq 0 ]; then
    # Get file size for verification
    SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
    
    # Quick verification - count tables
    TABLES=$(grep -c "CREATE TABLE" "$BACKUP_FILE" || true)
    
    echo "✅ Backup completed successfully!"
    echo "   File: $BACKUP_FILE"
    echo "   Size: $SIZE"
    echo "   Tables: $TABLES"
    
    # Count training data records for verification
    echo ""
    echo "📊 Training data summary:"
    psql "$DATABASE_URL" -t -c "SELECT 'Sessions: ' || COUNT(*) FROM rl_orderbook_sessions;" 2>/dev/null || true
    psql "$DATABASE_URL" -t -c "SELECT 'Snapshots: ' || COUNT(*) FROM rl_orderbook_snapshots;" 2>/dev/null || true
    psql "$DATABASE_URL" -t -c "SELECT 'Deltas: ' || COUNT(*) FROM rl_orderbook_deltas;" 2>/dev/null || true
    
    echo ""
    echo "💡 To restore this backup later, run:"
    echo "   psql \$DATABASE_URL < $BACKUP_FILE"
else
    echo "❌ Backup failed!"
    exit 1
fi