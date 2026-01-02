#!/usr/bin/env python3
"""
Test script to verify state metadata fix.
Ensures each state has only its appropriate metadata without pollution.
"""

import asyncio
import json
import logging
from typing import Dict, Any
import websockets
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StateMetadataValidator:
    """Validates that state transitions have correct metadata."""
    
    def __init__(self):
        self.state_transitions = []
        self.validation_errors = []
    
    def validate_orderbook_connect(self, metadata: Dict[str, Any]) -> bool:
        """Validate ORDERBOOK_CONNECT metadata - should NOT have trading data."""
        forbidden_fields = ["balance", "portfolio_value", "positions", "orders"]
        required_fields = ["ws_url", "markets", "market_count", "environment"]
        
        # Check for forbidden trading fields
        for field in forbidden_fields:
            if field in metadata:
                error = f"❌ ORDERBOOK_CONNECT has forbidden field '{field}' with value: {metadata[field]}"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        # Check for required fields
        for field in required_fields:
            if field not in metadata:
                error = f"❌ ORDERBOOK_CONNECT missing required field '{field}'"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        logger.info("✅ ORDERBOOK_CONNECT metadata is clean (no trading data)")
        return True
    
    def validate_trading_client_connect(self, metadata: Dict[str, Any]) -> bool:
        """Validate TRADING_CLIENT_CONNECT metadata."""
        forbidden_fields = ["balance", "portfolio_value", "positions", "orders"]
        required_fields = ["mode", "environment", "api_url"]
        
        # Check for forbidden fields
        for field in forbidden_fields:
            if field in metadata:
                error = f"❌ TRADING_CLIENT_CONNECT has forbidden field '{field}'"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        # Check for required fields
        for field in required_fields:
            if field not in metadata:
                error = f"❌ TRADING_CLIENT_CONNECT missing required field '{field}'"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        logger.info("✅ TRADING_CLIENT_CONNECT metadata is clean")
        return True
    
    def validate_kalshi_data_sync(self, metadata: Dict[str, Any]) -> bool:
        """Validate KALSHI_DATA_SYNC metadata - should NOT have actual data yet."""
        # Initial transition should only have sync_type
        forbidden_fields = ["balance", "portfolio_value", "positions", "orders"]
        required_fields = ["mode", "sync_type"]
        
        # This is the INITIAL transition to sync state, not the result
        for field in forbidden_fields:
            if field in metadata:
                error = f"❌ KALSHI_DATA_SYNC initial transition has data field '{field}' (should only appear in READY)"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        for field in required_fields:
            if field not in metadata:
                error = f"❌ KALSHI_DATA_SYNC missing required field '{field}'"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        logger.info("✅ KALSHI_DATA_SYNC metadata is clean (sync not complete yet)")
        return True
    
    def validate_ready(self, metadata: Dict[str, Any]) -> bool:
        """Validate READY metadata - should have all data."""
        required_orderbook_fields = ["markets_connected", "snapshots_received", "deltas_received"]
        
        for field in required_orderbook_fields:
            if field not in metadata:
                error = f"❌ READY missing orderbook field '{field}'"
                logger.error(error)
                self.validation_errors.append(error)
                return False
        
        # If trading is enabled, check for trading_client sub-object
        if "trading_client" in metadata:
            tc = metadata["trading_client"]
            required_trading_fields = ["balance", "portfolio_value", "positions", "orders"]
            for field in required_trading_fields:
                if field not in tc:
                    error = f"❌ READY trading_client missing field '{field}'"
                    logger.error(error)
                    self.validation_errors.append(error)
                    return False
            logger.info(f"✅ READY has trading data: balance={tc['balance']}, positions={tc['positions']}")
        else:
            logger.info("✅ READY metadata clean (orderbook only mode)")
        
        return True
    
    def validate_transition(self, transition: Dict[str, Any]) -> bool:
        """Validate a state transition event."""
        to_state = transition.get("to_state")
        metadata = transition.get("metadata", {})
        
        if not metadata:
            logger.warning(f"State {to_state} has no metadata")
            return True
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating {to_state} metadata:")
        logger.info(f"Metadata keys: {list(metadata.keys())}")
        
        if to_state == "orderbook_connect":
            return self.validate_orderbook_connect(metadata)
        elif to_state == "trading_client_connect":
            return self.validate_trading_client_connect(metadata)
        elif to_state == "kalshi_data_sync":
            return self.validate_kalshi_data_sync(metadata)
        elif to_state == "ready":
            return self.validate_ready(metadata)
        
        return True
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if not self.validation_errors:
            print("✅ ALL STATE TRANSITIONS HAVE CLEAN METADATA")
            print("\nKey validations passed:")
            print("  • ORDERBOOK_CONNECT has NO trading data (balance, positions, etc)")
            print("  • TRADING_CLIENT_CONNECT has API URL but NO account data")
            print("  • KALSHI_DATA_SYNC shows sync starting, not results")
            print("  • READY state contains all final data properly organized")
        else:
            print(f"❌ FOUND {len(self.validation_errors)} VALIDATION ERRORS:")
            for error in self.validation_errors:
                print(f"  {error}")
        
        print("="*80)


@pytest.mark.asyncio
async def test_state_metadata():
    """Connect to V3 WebSocket and validate state transitions."""
    validator = StateMetadataValidator()
    
    uri = "ws://localhost:8005/v3/ws"
    
    try:
        logger.info(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            logger.info("Connected! Monitoring state transitions...")
            
            # Listen for ~30 seconds to catch startup transitions
            timeout = 30
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    # Look for state transition events
                    if data.get("type") == "state_transition":
                        transition = data.get("data", {})
                        validator.state_transitions.append(transition)
                        
                        logger.info(f"\n📍 State Transition: {transition.get('from_state')} → {transition.get('to_state')}")
                        logger.info(f"   Context: {transition.get('context')}")
                        
                        # Validate the transition metadata
                        validator.validate_transition(transition)
                    
                    elif data.get("type") == "trader_status":
                        # Also check status updates
                        status = data.get("data", {})
                        state = status.get("state")
                        metrics = status.get("metrics", {})
                        logger.debug(f"Status update in state {state}: {len(metrics)} metric fields")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            logger.info(f"\nMonitoring complete. Received {len(validator.state_transitions)} state transitions.")
            
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        logger.info("\n⚠️  Make sure V3 trader is running on port 8005")
        logger.info("   Run: cd backend && uv run python src/kalshiflow_rl/traderv3/app.py")
        return False
    
    # Print summary
    validator.print_summary()
    
    return len(validator.validation_errors) == 0


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_state_metadata())
    
    if success:
        logger.info("\n✅ State metadata validation PASSED!")
        sys.exit(0)
    else:
        logger.error("\n❌ State metadata validation FAILED!")
        sys.exit(1)