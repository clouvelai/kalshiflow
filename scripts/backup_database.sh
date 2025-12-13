#!/bin/bash

# Kalshi Flow - Simple Database Backup Script
# Creates timestamped compressed backup of database

set -e

# Configuration
BACKUP_DIR="/Users/samuelclark/Desktop/kalshiflow/backups/database_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="kalshi_backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "🗄️  Database Backup Script"
echo "=========================="

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL environment variable is not set"
    echo "Please set DATABASE_URL before running backup script."
    exit 1
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create compressed backup
echo "🔄 Creating compressed backup..."
if docker exec supabase_db_backend pg_dump -U postgres postgres | gzip > "$BACKUP_PATH"; then
    echo "✅ Backup created successfully"
else
    echo "❌ Docker method failed, trying direct pg_dump..."
    if pg_dump "$DATABASE_URL" | gzip > "$BACKUP_PATH"; then
        echo "✅ Backup created with direct method"
    else
        echo "❌ Backup failed"
        exit 1
    fi
fi

# Verify backup file exists
if [ ! -f "$BACKUP_PATH" ]; then
    echo "❌ Backup file was not created"
    exit 1
fi

# Show backup information
BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
echo "📊 Backup completed:"
echo "   Location: $BACKUP_PATH"
echo "   Size: $BACKUP_SIZE"

# Clean up old backups (keep last 7 days)
echo "🧹 Cleaning up old backups..."
find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" -mtime +7 -delete 2>/dev/null || true
REMAINING=$(find "$BACKUP_DIR" -name "kalshi_backup_*.sql.gz" | wc -l)
echo "   Kept $REMAINING recent backup(s)"

echo "✅ Backup complete!"
echo ""
echo "💡 To restore: gunzip -c $BACKUP_PATH | psql \$DATABASE_URL"