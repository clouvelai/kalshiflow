#!/bin/bash

# Frontend E2E Test Runner
# This script runs the comprehensive frontend E2E regression test

echo "🚀 Starting Frontend E2E Regression Test"
echo "========================================"

# Check if backend is running
echo "🔍 Checking if backend is running on port 8000..."
if ! nc -z localhost 8000 2>/dev/null; then
    echo "❌ Backend not detected on port 8000"
    echo "💡 Please start the backend server first:"
    echo "   cd backend && uv run uvicorn kalshiflow.app:app --reload --port 8000"
    exit 1
else
    echo "✅ Backend is running"
fi

# Run the test
echo ""
echo "🎯 Running comprehensive E2E test..."
npm run test:frontend-regression

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Frontend E2E Test PASSED!"
    echo "========================================"
    echo "✅ Application loads correctly"
    echo "✅ WebSocket connects successfully" 
    echo "✅ Live data flows and updates"
    echo "✅ All components function properly"
    echo "✅ Responsive design works"
    echo "✅ Real-time validation completed"
    echo ""
    echo "📸 Screenshots saved in: test-results/"
    ls -la test-results/*.png | wc -l | xargs echo "   Total screenshots:"
    echo ""
    echo "🏆 Frontend is ready for production!"
else
    echo ""
    echo "❌ Frontend E2E Test FAILED"
    echo "========================================"
    echo "📸 Check screenshots and videos in: test-results/"
    echo "📊 Run with --ui for interactive debugging:"
    echo "   npm run test:e2e-ui"
    exit 1
fi