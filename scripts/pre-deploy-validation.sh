#!/bin/bash
set -e

echo "🔍 Pre-deployment Validation"
echo "============================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

VALIDATION_FAILED=false

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        VALIDATION_FAILED=true
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "1. Checking project structure..."
if [ ! -f "backend/pyproject.toml" ]; then
    print_status 1 "Backend pyproject.toml not found"
else
    print_status 0 "Backend structure verified"
fi

if [ ! -f "frontend/package.json" ]; then
    print_status 1 "Frontend package.json not found"
else
    print_status 0 "Frontend structure verified"
fi

echo ""
echo "2. Running backend E2E regression test..."
cd backend
if uv run pytest tests/test_backend_e2e_regression.py -v --tb=short; then
    print_status 0 "Backend E2E test passed"
else
    print_status 1 "Backend E2E test failed"
fi
cd ..

echo ""
echo "3. Checking frontend build..."
cd frontend
if npm run build > /dev/null 2>&1; then
    print_status 0 "Frontend build successful"
else
    print_status 1 "Frontend build failed"
fi
cd ..

echo ""
echo "4. Validating environment configuration..."
if [ ! -f ".env" ] && [ ! -f ".env.production" ]; then
    print_warning "No environment files found - ensure Railway variables are set"
else
    print_status 0 "Environment configuration files present"
fi

echo ""
echo "5. Checking Railway configuration..."
if [ ! -f "railway.toml" ]; then
    print_status 1 "railway.toml not found"
else
    print_status 0 "railway.toml configuration found"
fi

if [ ! -f "nixpacks.toml" ]; then
    print_status 1 "nixpacks.toml not found"
else
    print_status 0 "nixpacks.toml configuration found"
fi

echo ""
echo "=============================="
if [ "$VALIDATION_FAILED" = true ]; then
    echo -e "${RED}❌ Pre-deployment validation FAILED${NC}"
    echo "   Fix the above issues before deploying"
    exit 1
else
    echo -e "${GREEN}✅ Pre-deployment validation PASSED${NC}"
    echo "   Ready for deployment to Railway"
    exit 0
fi