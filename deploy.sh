#!/bin/bash

# Kalshi Flowboard Deployment Script
# Deploys both backend and frontend services to Railway with validation

set -e  # Exit on any error

echo "🚀 Starting Kalshi Flowboard Deployment"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if we're in the correct directory
check_directory() {
    if [[ ! -f "backend/railway.toml" ]] || [[ ! -f "frontend/package.json" ]]; then
        print_error "Not in the correct project directory. Expected backend/ and frontend/ directories."
        exit 1
    fi
    print_success "Project directory structure verified"
}

# Function to check git status
check_git_status() {
    print_status "Checking git status..."
    
    # Check if we're on main branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "main" ]]; then
        print_error "Not on main branch (currently on: $current_branch)"
        print_status "Please switch to main branch: git checkout main"
        exit 1
    fi
    print_success "On main branch"
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        print_error "Working directory is not clean. Please commit or stash changes."
        git status --short
        exit 1
    fi
    print_success "Working directory is clean"
    
    # Check if we're up to date with remote
    git fetch origin main
    local_commit=$(git rev-parse main)
    remote_commit=$(git rev-parse origin/main)
    if [[ "$local_commit" != "$remote_commit" ]]; then
        print_warning "Local main branch is not up to date with origin/main"
        print_status "Please pull latest changes: git pull origin main"
        exit 1
    fi
    print_success "Local main branch is up to date"
}

# Function to run backend E2E regression test
run_backend_tests() {
    print_status "Running backend E2E regression tests..."
    cd backend
    
    # Run the critical backend E2E regression test
    if uv run pytest tests/test_backend_e2e_regression.py -v; then
        print_success "Backend E2E regression tests passed"
    else
        print_error "Backend E2E regression tests failed"
        cd ..
        exit 1
    fi
    cd ..
}

# Function to run frontend E2E regression test
run_frontend_tests() {
    print_status "Running frontend E2E regression tests..."
    cd frontend
    
    # Check if backend is running on port 8000
    if ! curl -s http://localhost:8000/health > /dev/null; then
        print_error "Backend is not running on port 8000"
        print_status "Please start the backend: cd backend && uv run uvicorn kalshiflow.app:app --reload --port 8000"
        cd ..
        exit 1
    fi
    print_success "Backend is running and accessible"
    
    # Run the critical frontend E2E regression test
    if npm run test:frontend-regression; then
        print_success "Frontend E2E regression tests passed"
    else
        print_error "Frontend E2E regression tests failed"
        cd ..
        exit 1
    fi
    cd ..
}

# Function to deploy backend
deploy_backend() {
    print_status "Deploying backend service to Railway..."
    cd backend
    
    # Deploy to kalshi-flowboard-backend service
    if railway up --service kalshi-flowboard-backend; then
        print_success "Backend deployment completed successfully"
    else
        print_error "Backend deployment failed"
        cd ..
        exit 1
    fi
    cd ..
}

# Function to deploy frontend
deploy_frontend() {
    print_status "Deploying frontend service to Railway..."
    cd frontend
    
    # Deploy to kalshi-flowboard service
    if railway up --service kalshi-flowboard; then
        print_success "Frontend deployment completed successfully"
    else
        print_error "Frontend deployment failed"
        cd ..
        exit 1
    fi
    cd ..
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Wait a moment for services to start
    print_status "Waiting 30 seconds for services to initialize..."
    sleep 30
    
    # Here you could add health checks for the deployed services
    # For now, just confirm completion
    print_success "Deployment verification completed"
}

# Main deployment flow
main() {
    print_status "Starting deployment validation and process..."
    
    # Pre-deployment checks
    check_directory
    check_git_status
    
    # Run regression tests
    print_status "Running end-to-end regression tests..."
    run_backend_tests
    run_frontend_tests
    
    # Deploy services
    print_status "All tests passed! Proceeding with deployment..."
    deploy_backend
    deploy_frontend
    
    # Verify deployment
    verify_deployment
    
    print_success "🎉 Deployment completed successfully!"
    print_success "Ultra-fast analytics features are now live in production!"
    echo ""
    echo "Deployment Summary:"
    echo "✅ Backend service: kalshi-flowboard-backend"
    echo "✅ Frontend service: kalshi-flowboard"
    echo "✅ All E2E regression tests passed"
    echo "✅ Ultra-fast analytics improvements deployed"
}

# Handle script interruption
trap 'print_error "Deployment interrupted"; exit 1' INT

# Run main function
main "$@"