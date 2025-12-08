"""
Pytest configuration and shared fixtures for RL tests.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
from pathlib import Path

# Set up test environment variables BEFORE any imports
# This ensures the config module loads properly during test collection
os.environ.update({
    "KALSHI_API_KEY_ID": "test-api-key",
    "KALSHI_PRIVATE_KEY_PATH": "/tmp/test_private_key.pem",
    "DATABASE_URL": "postgresql://test:test@localhost/test_db",
    "RL_MARKET_TICKER": "TEST-MARKET",
    "RL_ORDERBOOK_BATCH_SIZE": "5",
    "RL_ORDERBOOK_FLUSH_INTERVAL": "0.1",
    "RL_ORDERBOOK_SAMPLE_RATE": "1",
    "ENVIRONMENT": "test",
    "DEBUG": "true"
})


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="session")
def setup_test_private_key():
    """Set up test private key file."""
    # Create test private key file
    test_private_key = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKB
wQNfFHDzj0v5AKKzBp3EBH2H3tJ2/r6TwJ5d4jZ1pKj0E8SQQj3CmF+Hv5gOqO7
rK5mF8FGn3HxX6mOx1OJXQ3V2RtKWZxKl6h7/jkQaL8UElLF2F+P5F2oVR1TzJs
vMw5p8Xj3I6i4V9vG8OP6Wl4K5Q7xQ==
-----END PRIVATE KEY-----"""
    
    with open("/tmp/test_private_key.pem", "w") as f:
        f.write(test_private_key)
    
    yield
    
    # Clean up test files
    try:
        os.unlink("/tmp/test_private_key.pem")
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture(scope="function", autouse=True)
async def clean_orderbook_registry():
    """Clean up global orderbook state registry before and after tests."""
    # Clean up before test
    try:
        from kalshiflow_rl.data.orderbook_state import _orderbook_states, _states_lock
        async with _states_lock:
            _orderbook_states.clear()
    except Exception:
        pass  # Ignore cleanup errors during setup
    
    yield
    
    # Clean up after test
    try:
        from kalshiflow_rl.data.orderbook_state import _orderbook_states, _states_lock
        async with _states_lock:
            _orderbook_states.clear()
    except Exception:
        pass  # Ignore cleanup errors during teardown