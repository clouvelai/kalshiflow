#!/bin/bash

# Validation Script for Kalshi Flowboard Supabase Setup
# Tests both local and production environments to ensure everything works

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function test_database_connection() {
    local env_name="$1"
    print_section "Testing $env_name Database Connection"
    
    cd backend
    
    result=$(uv run python -c "
import asyncio
import sys
sys.path.append('src')
from kalshiflow.database_factory import get_current_database
from dotenv import load_dotenv
load_dotenv()

async def test():
    try:
        db = get_current_database()
        await db.initialize()
        stats = await db.get_db_stats()
        if hasattr(db, 'close'):
            await db.close()
        return True
    except Exception as e:
        print(f'Error: {e}')
        return False

result = asyncio.run(test())
print('PASSED' if result else 'FAILED')
    " 2>/dev/null)
    
    if [[ "$result" == *"PASSED"* ]]; then
        print_success "$env_name database connection working"
        return 0
    else
        print_error "$env_name database connection failed"
        echo "$result"
        return 1
    fi
    
    cd ..
}

function test_environment_files() {
    print_section "Checking Environment Files"
    
    if [ -f .env.local ]; then
        print_success ".env.local exists"
    else
        print_error ".env.local missing"
        return 1
    fi
    
    if [ -f .env.production ]; then
        print_success ".env.production exists"
    else
        print_error ".env.production missing"
        return 1
    fi
    
    if [ -f .env.example ]; then
        print_success ".env.example exists"
    else
        print_warning ".env.example missing (not critical)"
    fi
    
    return 0
}

function test_supabase_cli() {
    print_section "Checking Supabase CLI"
    
    if command -v supabase &> /dev/null; then
        version=$(supabase --version 2>/dev/null | head -1)
        print_success "Supabase CLI installed: $version"
    else
        print_error "Supabase CLI not installed"
        return 1
    fi
    
    return 0
}

function test_local_environment() {
    print_section "Testing Local Environment"
    
    # Switch to local
    ./scripts/switch-env.sh local > /dev/null 2>&1
    
    # Check if local Supabase is running
    if docker ps | grep -q "supabase"; then
        print_success "Local Supabase containers are running"
        test_database_connection "Local"
    else
        print_warning "Local Supabase not running - starting it..."
        cd backend
        supabase start > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            print_success "Local Supabase started successfully"
            test_database_connection "Local"
            cd ..
        else
            print_error "Failed to start local Supabase"
            cd ..
            return 1
        fi
    fi
}

function test_production_environment() {
    print_section "Testing Production Environment"
    
    # Switch to production
    ./scripts/switch-env.sh production > /dev/null 2>&1
    
    test_database_connection "Production"
}

function main() {
    echo -e "${BLUE}Kalshi Flowboard Supabase Setup Validation${NC}"
    echo "This script will test your complete Supabase configuration"
    echo ""
    
    # Store original environment
    if [ -f .env ]; then
        cp .env .env.validation.backup
    fi
    
    total_tests=0
    passed_tests=0
    
    # Test 1: Environment files
    if test_environment_files; then
        ((passed_tests++))
    fi
    ((total_tests++))
    
    # Test 2: Supabase CLI
    if test_supabase_cli; then
        ((passed_tests++))
    fi
    ((total_tests++))
    
    # Test 3: Local environment
    if test_local_environment; then
        ((passed_tests++))
    fi
    ((total_tests++))
    
    # Test 4: Production environment
    if test_production_environment; then
        ((passed_tests++))
    fi
    ((total_tests++))
    
    # Restore original environment
    if [ -f .env.validation.backup ]; then
        mv .env.validation.backup .env
        print_success "Restored original environment configuration"
    fi
    
    # Results
    print_section "Validation Results"
    echo -e "Tests passed: ${GREEN}$passed_tests${NC}/$total_tests"
    
    if [ $passed_tests -eq $total_tests ]; then
        print_success "All tests passed! Your Supabase setup is working correctly."
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Use ./scripts/switch-env.sh local for development"
        echo "2. Use ./scripts/switch-env.sh production for deployment"
        echo "3. See SUPABASE_SETUP.md for detailed documentation"
        return 0
    else
        print_error "Some tests failed. Please check the output above."
        echo ""
        echo -e "${BLUE}Troubleshooting:${NC}"
        echo "1. Ensure Docker is running for local Supabase"
        echo "2. Check your production Supabase credentials"
        echo "3. See SUPABASE_SETUP.md for troubleshooting guide"
        return 1
    fi
}

# Run main function
main "$@"