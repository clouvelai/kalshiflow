# Supabase Setup Guide for Kalshi Flowboard

This guide explains how to set up and use Supabase with the Kalshi Flowboard application, following best practices for environment separation.

## Overview

The application now supports both **local development** with Supabase local instance and **production deployment** with remote Supabase, providing a clean separation between environments.

### Architecture

- **Local Development**: Supabase local instance (Docker-based)
- **Production**: Remote Supabase project (managed cloud)
- **Environment Separation**: Clean `.env` file management
- **Migration Management**: Supabase CLI handles schema updates

## Quick Start

### 1. Choose Your Environment

```bash
# For local development
./scripts/switch-env.sh local

# For production deployment
./scripts/switch-env.sh production

# Check current environment
./scripts/switch-env.sh current
```

### 2. Local Development Setup

```bash
# Switch to local environment
./scripts/switch-env.sh local

# Start local Supabase (from backend directory)
cd backend && supabase start

# Run the application
uv run uvicorn kalshiflow.app:app --reload
```

### 3. Production Setup

```bash
# Switch to production environment
./scripts/switch-env.sh production

# Run the application (uses remote Supabase)
cd backend && uv run uvicorn kalshiflow.app:app --reload
```

## Detailed Setup Instructions

### Prerequisites

1. **Supabase CLI**: Already installed (`supabase --version`)
2. **Docker**: Required for local Supabase instance
3. **PostgreSQL**: Already configured in the project

### Local Development Environment

#### Features
- **Fast setup**: No internet connection required after initial setup
- **Isolated data**: Completely separate from production
- **Full feature parity**: All Supabase features available locally
- **Reset capability**: Easy to reset and start fresh

#### Setup Steps

1. **Initialize local environment**:
   ```bash
   ./scripts/switch-env.sh local
   ```

2. **Start local Supabase**:
   ```bash
   cd backend
   supabase start
   ```

3. **Local Supabase URLs**:
   - API: `http://localhost:54321`
   - Database: `postgresql://postgres:postgres@localhost:54322/postgres`
   - Studio (Dashboard): `http://localhost:54323`

4. **Run application**:
   ```bash
   uv run uvicorn kalshiflow.app:app --reload
   ```

5. **Stop when done**:
   ```bash
   supabase stop
   ```

### Production Environment

#### Features
- **Managed infrastructure**: Supabase handles scaling, backups, security
- **Connection pooling**: Optimized for production workloads
- **Monitoring**: Built-in metrics and logging
- **High availability**: Automatic failover and redundancy

#### Setup Steps

1. **Switch to production environment**:
   ```bash
   ./scripts/switch-env.sh production
   ```

2. **Verify connection**:
   ```bash
   cd backend
   uv run python -c "
   import asyncio
   from kalshiflow.database_factory import get_current_database
   async def test(): 
       db = get_current_database()
       await db.initialize()
       stats = await db.get_db_stats()
       print('Production DB connected:', stats)
       await db.close()
   asyncio.run(test())
   "
   ```

3. **Run application**:
   ```bash
   uv run uvicorn kalshiflow.app:app --reload
   ```

## Environment Files

### `.env.local` (Local Development)
```bash
# Local Supabase Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres
DATABASE_URL_POOLED=postgresql://postgres:postgres@localhost:54322/postgres
SUPABASE_URL=http://localhost:54321
USE_POSTGRESQL=true
ENVIRONMENT=local
```

### `.env.production` (Production)
```bash
# Production Supabase Configuration
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres
DATABASE_URL_POOLED=postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:6543/postgres
SUPABASE_URL=https://[PROJECT_ID].supabase.co
USE_POSTGRESQL=true
ENVIRONMENT=production
```

### `.env.example` (Template)
Contains examples for both local and production configurations.

## Database Migration Management

### Migration Files Location
```
backend/supabase/migrations/
├── 20251205172734_initial_schema.sql
└── [future migrations...]
```

### Local Development Migrations
- **Automatic**: Supabase CLI applies migrations when starting local instance
- **Manual**: `supabase db reset` to reset and reapply all migrations
- **New migrations**: `supabase migration new migration_name`

### Production Migrations
- **Manual deployment**: Use Supabase dashboard or CLI
- **Automatic**: Application skips migration execution for remote Supabase

### Schema Structure

#### Tables
- **`markets`**: Market metadata and information
- **`trades`**: Public trades from Kalshi WebSocket stream

#### Key Features
- **JSONB storage**: Efficient JSON data handling
- **Optimized indexes**: Performance-tuned for time-series queries
- **Constraints**: Data validation and integrity
- **Triggers**: Automatic timestamp updates

## Best Practices

### Development Workflow

1. **Start with local**: Always develop and test locally first
2. **Environment switching**: Use `./scripts/switch-env.sh` consistently
3. **Clean commits**: Don't commit real `.env` files
4. **Migration testing**: Test migrations locally before production

### Production Deployment

1. **Environment variables**: Set in Render dashboard, not in code
2. **Connection pooling**: Use `DATABASE_URL_POOLED` for production
3. **Monitoring**: Check Supabase dashboard for performance metrics
4. **Backups**: Verify automatic backup settings

### Security

1. **Credentials**: Never commit real credentials to Git
2. **Environment separation**: Keep local and production completely separate
3. **Access control**: Use Supabase Row Level Security (RLS) if needed
4. **API keys**: Rotate regularly and monitor usage

## Troubleshooting

### Common Issues

#### Local Supabase won't start
```bash
# Check Docker is running
docker ps

# Reset Supabase
supabase stop
supabase start --ignore-health-check
```

#### Migration errors
```bash
# Reset local database
supabase db reset

# Check migration file syntax
cd backend/supabase/migrations/
cat *.sql
```

#### Connection timeouts
```bash
# Check environment configuration
./scripts/switch-env.sh current

# Test database connection
cd backend
uv run python test_db_connection.py
```

### Performance Issues

#### Local Development
- Local Supabase should be very fast
- If slow, check Docker resource allocation
- Consider `supabase db reset` to clean up data

#### Production
- Monitor connection pool usage
- Check Supabase dashboard for metrics
- Consider upgrading Supabase plan if needed

## Environment Switcher Script

### Usage
```bash
./scripts/switch-env.sh [local|production|current]
```

### Features
- **Automatic backup**: Saves current `.env` as `.env.backup`
- **Validation**: Checks for required files
- **Status display**: Shows current configuration
- **Next steps**: Provides guidance for each environment

## Integration with Deployment

### Render Deployment
- Uses `.env.production` configuration automatically
- Environment variables set in Render dashboard
- Automatic deployments on Git push

### Local Testing
- Full feature parity with production
- Fast iteration and debugging
- Isolated from production data

## Monitoring and Maintenance

### Local Development
- Check `supabase status` for service health
- Use `http://localhost:54323` for database management
- Monitor Docker container resources

### Production
- Supabase dashboard: `https://supabase.com/dashboard`
- Monitor connection counts and query performance
- Set up alerts for critical metrics
- Regular backup verification

## Summary

This Supabase setup provides:

✅ **Clean environment separation** between local and production  
✅ **Easy switching** between environments with single command  
✅ **Production-ready** PostgreSQL with connection pooling  
✅ **Local development** with full Supabase feature set  
✅ **Proper migration management** via Supabase CLI  
✅ **Security best practices** with credential management  
✅ **Performance optimization** with indexes and connection pooling  

The setup follows Supabase best practices and is ready for production deployment to Render with proper scaling and monitoring capabilities.