#!/usr/bin/env python
"""Run the RL E2E test but keep the data and query it."""

import asyncio
import subprocess
import time
import os
import sys

# Add the backend src to path
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend/src')
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend')

# Import the query script
from scripts.query_rl_orderbook import show_orderbook_data

async def main():
    """Run E2E test and query the data."""
    print("🚀 Starting RL E2E test (without cleanup)...")
    
    # Modify the test to skip cleanup
    test_code = """
import sys
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend/src')
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend')

# Import the test but override cleanup
from tests.test_rl_backend_e2e_regression import test_rl_backend_e2e_regression

# Override the cleanup function to do nothing
async def no_cleanup():
    print("⚠️  Skipping cleanup to preserve data for inspection")
    pass

# Monkey patch the cleanup
import tests.test_rl_backend_e2e_regression as test_module
test_module.cleanup_e2e_test = no_cleanup

# Run the test
import asyncio
asyncio.run(test_rl_backend_e2e_regression())
"""
    
    # Write temporary test file
    with open('/tmp/run_rl_test_no_cleanup.py', 'w') as f:
        f.write(test_code)
    
    # Run the modified test
    result = subprocess.run(
        ['uv', 'run', 'python', '/tmp/run_rl_test_no_cleanup.py'],
        cwd='/Users/samuelclark/Desktop/kalshiflow/backend',
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Test failed:")
        print(result.stdout)
        print(result.stderr)
    else:
        print("✅ Test completed successfully!")
        print("\n⏳ Waiting 2 seconds for data to settle...")
        await asyncio.sleep(2)
        
        print("\n📊 Querying orderbook data from database...\n")
        await show_orderbook_data()
    
    # Clean up temp file
    os.remove('/tmp/run_rl_test_no_cleanup.py')

if __name__ == "__main__":
    asyncio.run(main())