#!/bin/bash

# Environment Switcher Script for Kalshi Flowboard
# Switches between local and production environment configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_usage() {
    echo -e "${BLUE}Usage: $0 [local|production|current]${NC}"
    echo ""
    echo "Commands:"
    echo "  local      - Switch to local development environment (.env.local)"
    echo "  production - Switch to production environment (.env.production)" 
    echo "  current    - Show current environment configuration"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Switch to local Supabase development"
    echo "  $0 production # Switch to production Supabase"
    echo "  $0 current    # Show current .env file"
}

function show_current_env() {
    echo -e "${BLUE}Current Environment Configuration:${NC}"
    echo ""
    if [ -f .env ]; then
        # Extract key environment indicators
        echo -e "${YELLOW}Database:${NC}"
        grep "DATABASE_URL=" .env | head -1
        echo ""
        echo -e "${YELLOW}Environment:${NC}"
        grep "ENVIRONMENT=" .env || echo "ENVIRONMENT=not set"
        echo ""
        echo -e "${YELLOW}PostgreSQL Enabled:${NC}"
        grep "USE_POSTGRESQL=" .env
        echo ""
        echo -e "${YELLOW}Supabase URL:${NC}"
        grep "SUPABASE_URL=" .env
    else
        echo -e "${RED}No .env file found!${NC}"
        echo "Please run: $0 local  or  $0 production"
    fi
}

function switch_to_local() {
    echo -e "${BLUE}Switching to local development environment...${NC}"
    
    if [ ! -f .env.local ]; then
        echo -e "${RED}Error: .env.local file not found!${NC}"
        exit 1
    fi
    
    # Backup current .env if exists
    if [ -f .env ]; then
        cp .env .env.backup
        echo -e "${YELLOW}Backed up current .env to .env.backup${NC}"
    fi
    
    # Copy local environment
    cp .env.local .env
    echo -e "${GREEN}✅ Switched to local environment${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Start Supabase: cd backend && supabase start"
    echo "2. Run application: cd backend && uv run uvicorn kalshiflow.app:app --reload"
    echo ""
    show_current_env
}

function switch_to_production() {
    echo -e "${BLUE}Switching to production environment...${NC}"
    
    if [ ! -f .env.production ]; then
        echo -e "${RED}Error: .env.production file not found!${NC}"
        exit 1
    fi
    
    # Backup current .env if exists
    if [ -f .env ]; then
        cp .env .env.backup
        echo -e "${YELLOW}Backed up current .env to .env.backup${NC}"
    fi
    
    # Copy production environment
    cp .env.production .env
    echo -e "${GREEN}✅ Switched to production environment${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Verify Supabase connection: cd backend && uv run python test_db_connection.py"
    echo "2. Deploy to Render or run locally: cd backend && uv run uvicorn kalshiflow.app:app --reload"
    echo ""
    show_current_env
}

# Main script logic
case "$1" in
    "local")
        switch_to_local
        ;;
    "production") 
        switch_to_production
        ;;
    "current")
        show_current_env
        ;;
    *)
        print_usage
        exit 1
        ;;
esac