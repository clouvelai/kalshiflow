# Render Deployment Instructions - Phase 2.1 Service Creation

## Overview

These instructions will create the Kalshi Flowboard services on Render using the authenticated CLI. Execute these commands immediately after billing is configured.

**Current Status**: Render CLI authenticated as clouvel.ai@gmail.com (v2.5.0)  
**Task**: Complete Phase 2.1 - Create backend web service and frontend static site  
**Duration**: ~30-45 minutes  
**Prerequisites**: Billing must be set up on Render account

## Prerequisites Check

Before starting, verify these requirements:

```bash
# 1. Confirm CLI is authenticated and working
export PATH="$HOME/.local/bin:$PATH" && render auth whoami

# Expected output: clouvel.ai@gmail.com

# 2. Verify we're in the correct directory
pwd
# Expected: /Users/samuelclark/Desktop/kalshiflow

# 3. Confirm git repository is clean and up to date
git status
git pull origin main

# 4. Verify billing is configured (check Render dashboard)
echo "✅ Billing configured in Render dashboard? (manual check required)"
```

## Service Creation Commands

### Step 1: Create Backend Web Service

Create the Python backend service with exact configuration:

```bash
# Create backend web service
export PATH="$HOME/.local/bin:$PATH" && render create web \
  --name "kalshiflow-backend" \
  --env "python" \
  --build-command "cd backend && uv sync" \
  --start-command "cd backend && uv run uvicorn kalshiflow.app:app --host 0.0.0.0 --port \$PORT" \
  --root-directory "./" \
  --auto-deploy true \
  --branch main \
  -o json
```

**Expected Output**:
- JSON response with service details
- Service ID and URL will be provided
- Status should show "created" or "building"

**Service URL Format**: `https://kalshiflow-backend-XXXXX.onrender.com`

### Step 2: Create Frontend Static Site

Create the React frontend static site:

```bash
# Create frontend static site
export PATH="$HOME/.local/bin:$PATH" && render create static \
  --name "kalshiflow-frontend" \
  --build-command "cd frontend && npm install && npm run build" \
  --publish-directory "frontend/dist" \
  --root-directory "./" \
  --auto-deploy true \
  --branch main \
  -o json
```

**Expected Output**:
- JSON response with service details
- Service ID and URL will be provided
- Status should show "created" or "building"

**Service URL Format**: `https://kalshiflow-frontend-XXXXX.onrender.com`

### Step 3: Verify Service Creation

List all services to confirm creation:

```bash
# List all services
export PATH="$HOME/.local/bin:$PATH" && render services list -o json

# Check specific service details
export PATH="$HOME/.local/bin:$PATH" && render services get kalshiflow-backend -o json
export PATH="$HOME/.local/bin:$PATH" && render services get kalshiflow-frontend -o json
```

## Validation Steps

### 1. Backend Service Health Check

Wait for backend service to deploy (5-10 minutes), then test:

```bash
# Get backend URL from previous command output
BACKEND_URL="https://kalshiflow-backend-XXXXX.onrender.com"

# Test health endpoint (may initially fail until environment variables are set)
curl -f "$BACKEND_URL/health" || echo "Expected to fail - environment variables not yet configured"
```

### 2. Frontend Service Validation

Wait for frontend service to deploy (3-5 minutes), then test:

```bash
# Get frontend URL from previous command output
FRONTEND_URL="https://kalshiflow-frontend-XXXXX.onrender.com"

# Test frontend accessibility
curl -f "$FRONTEND_URL" -I || echo "Check if site is still building"
```

### 3. Service Status Monitoring

Check deployment progress:

```bash
# Monitor backend deployment
export PATH="$HOME/.local/bin:$PATH" && render services logs kalshiflow-backend

# Monitor frontend deployment
export PATH="$HOME/.local/bin:$PATH" && render services logs kalshiflow-frontend
```

## Expected Service URLs

After successful creation, you should have:

- **Backend Service**: `https://kalshiflow-backend-XXXXX.onrender.com`
  - Health Check: `https://kalshiflow-backend-XXXXX.onrender.com/health`
  - WebSocket: `wss://kalshiflow-backend-XXXXX.onrender.com/ws`

- **Frontend Service**: `https://kalshiflow-frontend-XXXXX.onrender.com`
  - Main Application: Direct access to React app

## Troubleshooting

### Common Issues and Solutions

**1. CLI Authentication Error**
```bash
# Re-authenticate if needed
export PATH="$HOME/.local/bin:$PATH" && render auth login
```

**2. Service Creation Fails**
```bash
# Check account status and billing
export PATH="$HOME/.local/bin:$PATH" && render account info -o json

# Verify repository connection
git remote -v
```

**3. Build Failures**
```bash
# Check build logs for specific errors
export PATH="$HOME/.local/bin:$PATH" && render services logs kalshiflow-backend --tail 100
export PATH="$HOME/.local/bin:$PATH" && render services logs kalshiflow-frontend --tail 100
```

**4. TTY/Interactive Mode Issues**
- All commands include `-o json` flag to avoid TTY issues
- If commands hang, use Ctrl+C and retry with explicit non-interactive flags

### Health Check Debugging

Backend service will initially fail health checks because environment variables are not configured. This is expected behavior.

**Normal progression**:
1. Service creates successfully ✅
2. Build completes ✅  
3. Service starts but health check fails ❌ (expected - no env vars)
4. After Phase 2.2 (environment variables), health check passes ✅

## Next Steps - Phase 2.2 Preview

After services are created, the next phase will configure:

1. **Environment Variables** (Phase 2.2):
   - Kalshi API credentials
   - Supabase database connection strings
   - Application configuration settings

2. **Service Configuration** (Phase 2.3):
   - Health check endpoints
   - Auto-scaling rules
   - Deployment triggers

3. **Frontend API Configuration** (Phase 2.4):
   - Update frontend to use production backend URLs
   - Configure CORS settings
   - SSL/domain configuration

## Validation Checklist

Mark each item as completed:

- [ ] CLI authentication confirmed (`render auth whoami`)
- [ ] Backend web service created successfully
- [ ] Frontend static site created successfully
- [ ] Both services appear in `render services list`
- [ ] Backend service URL accessible (even if health check fails)
- [ ] Frontend service URL accessible (may show connection errors initially)
- [ ] Service logs showing build activity
- [ ] Ready to proceed to Phase 2.2 (Environment Variables Configuration)

## Service IDs and URLs

Record your actual service details here for reference:

```
Backend Service:
- Name: kalshiflow-backend
- URL: https://kalshiflow-backend-XXXXX.onrender.com
- Service ID: [record from CLI output]

Frontend Service:  
- Name: kalshiflow-frontend
- URL: https://kalshiflow-frontend-XXXXX.onrender.com
- Service ID: [record from CLI output]
```

## Support and Contact

If issues occur during service creation:

1. Check Render dashboard for additional details
2. Review service logs via CLI commands above
3. Ensure billing is properly configured
4. Verify GitHub repository permissions

**Status**: Ready to execute Phase 2.2 (Environment Variables Configuration) after successful service creation.