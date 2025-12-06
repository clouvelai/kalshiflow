#!/bin/bash
set -e

echo "🚀 Railway Deployment Setup Script"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Railway CLI
if ! command_exists railway; then
    echo "❌ Railway CLI not found. Please install: npm install -g @railway/cli"
    exit 1
fi

echo "✅ Railway CLI found"

# Check if logged in to Railway
if ! railway whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Railway. Please run: railway login"
    exit 1
fi

echo "✅ Railway authentication verified"

# Enable auto-deployment (when ready)
echo ""
echo "📋 To enable auto-deployment later, run:"
echo "   railway settings --auto-deploy=main"
echo ""
echo "🔧 To configure environment variables, run:"
echo "   railway variables set PYTHONPATH=\"/app/backend/src\""
echo "   railway variables set UVICORN_HOST=\"0.0.0.0\""
echo "   railway variables set UVICORN_PORT=\"\$PORT\""
echo "   railway variables set NODE_ENV=\"production\""
echo ""
echo "✅ Setup script complete. Configuration files are ready."