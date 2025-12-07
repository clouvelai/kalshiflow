#!/bin/bash
# Railway Environment Variables for WebSocket Optimization
# Run this script to configure Railway with production WebSocket settings

echo "Setting Railway environment variables for WebSocket optimization..."

# WebSocket timeout and keepalive settings
railway variables set UVICORN_TIMEOUT_KEEP_ALIVE=300
railway variables set UVICORN_WS_PING_INTERVAL=30  
railway variables set UVICORN_WS_PING_TIMEOUT=10

# Application server configuration
railway variables set PYTHONPATH="/app/backend/src"
railway variables set UVICORN_HOST="0.0.0.0"
railway variables set UVICORN_PORT='$PORT'
railway variables set UVICORN_WORKERS=1

# Railway-specific optimizations
railway variables set NODE_ENV="production"
railway variables set RAILWAY_WEBSOCKET_TIMEOUT=300
railway variables set STARLETTE_WS_TIMEOUT=300

# Additional WebSocket stability settings
railway variables set WS_HEARTBEAT_INTERVAL=30
railway variables set CONNECTION_TIMEOUT=300

echo "Railway environment variables configured for WebSocket optimization!"
echo ""
echo "Next steps:"
echo "1. Deploy with: railway up"
echo "2. Monitor logs: railway logs --tail 50"
echo "3. Test WebSocket stability"