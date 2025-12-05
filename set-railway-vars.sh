#!/bin/bash

# Script to set Railway environment variables from .env.production
# Usage: ./set-railway-vars.sh

echo "Setting Railway environment variables from .env.production..."

cd backend

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    echo "Error: .env.production file not found in backend directory"
    exit 1
fi

# Read variables from .env.production and set them in Railway
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    if [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove any leading/trailing whitespace and quotes
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    # Remove quotes if present
    if [[ $value =~ ^\".*\"$ ]]; then
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/')
    fi
    
    # Skip empty values
    if [[ -z "$value" ]]; then
        continue
    fi
    
    echo "Setting $key..."
    railway variables set "$key=$value"
    
done < .env.production

echo "✅ All environment variables set successfully!"
echo ""
echo "Next steps:"
echo "1. Deploy backend: railway up"
echo "2. Deploy frontend: cd ../frontend && railway up"