#!/bin/bash

# Kalshi Flowboard - Initialize and Start Script
# This script sets up the entire development environment and starts both backend and frontend

set -e  # Exit on error

echo "====================================="
echo "   Kalshi Flowboard - Setup Script"
echo "====================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking for required tools..."

if ! command_exists uv; then
    echo "Error: 'uv' is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command_exists npm; then
    echo "Error: 'npm' is not installed. Please install Node.js first."
    exit 1
fi

echo "✓ All required tools found"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠ Warning: .env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  Created .env file. Please update it with your Kalshi API credentials."
        echo ""
    else
        echo "  Error: .env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
uv sync
echo "✓ Backend dependencies installed"
echo ""

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd ../frontend
npm install
echo "✓ Frontend dependencies installed"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start services
echo "Starting services..."
echo "====================================="
echo ""

# Start backend
echo "Starting backend server on http://localhost:8000..."
cd ../backend
uv run uvicorn kalshiflow.app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is running"
else
    echo "⚠ Backend may not have started correctly"
fi
echo ""

# Start frontend
echo "Starting frontend development server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"
echo ""

echo "====================================="
echo "   Services are starting..."
echo "====================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Health:   http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID