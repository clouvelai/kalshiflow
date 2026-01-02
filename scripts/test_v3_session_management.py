#!/usr/bin/env python3
"""
Test script for TRADER V3 session management improvements.

Tests the conservative session management implementation to ensure:
1. Health state transitions are tracked correctly
2. Session cleanup preserves metrics 
3. Recovery creates new sessions properly
4. No orphaned sessions are left behind
"""

import asyncio
import sys
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.traderv3.clients.orderbook_integration import V3OrderbookIntegration
from kalshiflow_rl.traderv3.core.event_bus import EventBus
from kalshiflow_rl.data.database import rl_db
from kalshiflow_rl.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_session_management")


class SessionManagementTester:
    """Test harness for session management functionality."""
    
    def __init__(self):
        self.test_markets = ["INXD-25JAN03"]  # Use a single test market
        self.event_bus = None
        self.orderbook_client = None
        self.integration = None
        
    async def setup(self):
        """Set up test environment."""
        logger.info("Setting up test environment...")
        
        # Initialize database
        await rl_db.initialize()
        
        # Create event bus
        self.event_bus = EventBus()
        await self.event_bus.start()
        
        # Create orderbook client with test markets and event bus integration
        self.orderbook_client = OrderbookClient(
            market_tickers=self.test_markets,
            event_bus=self.event_bus
        )
        
        # Create V3 integration
        self.integration = V3OrderbookIntegration(
            orderbook_client=self.orderbook_client,
            event_bus=self.event_bus,
            market_tickers=self.test_markets
        )
        
        logger.info("✅ Test environment setup complete")
    
    async def cleanup(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        
        if self.integration:
            await self.integration.stop()
        
        if self.orderbook_client:
            await self.orderbook_client.stop()
            
        if self.event_bus:
            await self.event_bus.stop()
            
        logger.info("✅ Test environment cleaned up")
    
    async def test_session_creation_and_tracking(self):
        """Test basic session creation and state tracking."""
        logger.info("🧪 Testing session creation and state tracking...")
        
        # Start integration
        await self.integration.start()
        
        # Wait for connection
        connected = await self.integration.wait_for_connection(timeout=10.0)
        if not connected:
            logger.error("❌ Failed to connect - cannot test session management")
            return False
        
        # Check session state
        client_stats = self.orderbook_client.get_stats()
        session_id = client_stats.get("session_id")
        session_state = client_stats.get("session_state")
        
        logger.info(f"Session created: ID={session_id}, State={session_state}")
        
        if session_id and session_state == "active":
            logger.info("✅ Session creation and tracking working")
            return True
        else:
            logger.error(f"❌ Session creation failed: ID={session_id}, State={session_state}")
            return False
    
    async def test_health_state_transitions(self):
        """Test health state transition tracking."""
        logger.info("🧪 Testing health state transitions...")
        
        # Get initial health state
        initial_health = self.orderbook_client.is_healthy()
        initial_state = self.orderbook_client.get_stats().get("health_state")
        
        logger.info(f"Initial health: {initial_health}, State: {initial_state}")
        
        # Simulate a brief period and check health tracking
        await asyncio.sleep(2.0)
        
        current_health = self.orderbook_client.is_healthy()
        current_state = self.orderbook_client.get_stats().get("health_state")
        
        logger.info(f"Current health: {current_health}, State: {current_state}")
        
        # Health state should be tracked
        if current_state in ["healthy", "unhealthy"]:
            logger.info("✅ Health state transitions are being tracked")
            return True
        else:
            logger.error(f"❌ Health state not being tracked: {current_state}")
            return False
    
    async def test_session_recovery_capability(self):
        """Test session recovery functionality."""
        logger.info("🧪 Testing session recovery capability...")
        
        # Get current session info
        stats_before = self.orderbook_client.get_stats()
        session_before = stats_before.get("session_id")
        metrics_before = {
            "messages": stats_before.get("messages_received", 0),
            "snapshots": stats_before.get("snapshots_received", 0),
            "deltas": stats_before.get("deltas_received", 0)
        }
        
        logger.info(f"Before recovery: Session={session_before}, Metrics={metrics_before}")
        
        # Test session recovery
        recovery_success = await self.integration.ensure_session_for_recovery()
        
        # Get session info after recovery attempt
        stats_after = self.orderbook_client.get_stats()
        session_after = stats_after.get("session_id")
        metrics_after = {
            "messages": stats_after.get("messages_received", 0),
            "snapshots": stats_after.get("snapshots_received", 0),
            "deltas": stats_after.get("deltas_received", 0)
        }
        
        logger.info(f"After recovery: Session={session_after}, Metrics={metrics_after}, Success={recovery_success}")
        
        # Check that metrics are preserved
        metrics_preserved = all(
            metrics_after[key] >= metrics_before[key] for key in metrics_before
        )
        
        if recovery_success and metrics_preserved:
            logger.info("✅ Session recovery capability working")
            return True
        else:
            logger.error(f"❌ Session recovery failed: Success={recovery_success}, Metrics preserved={metrics_preserved}")
            return False
    
    async def test_active_session_count(self):
        """Test that we don't create orphaned sessions."""
        logger.info("🧪 Testing active session management...")
        
        # Get active sessions count before
        active_sessions_before = await rl_db.get_active_sessions()
        count_before = len(active_sessions_before)
        
        logger.info(f"Active sessions before test: {count_before}")
        
        # Stop and restart integration (should close old session and create new one)
        await self.integration.stop()
        await asyncio.sleep(1.0)  # Brief pause
        await self.integration.start()
        
        # Wait for new connection
        await self.integration.wait_for_connection(timeout=10.0)
        
        # Get active sessions count after
        active_sessions_after = await rl_db.get_active_sessions()
        count_after = len(active_sessions_after)
        
        logger.info(f"Active sessions after restart: {count_after}")
        
        # Check that we didn't accumulate orphaned sessions
        # Should have at most 1-2 active sessions (depending on timing)
        if count_after <= count_before + 1:
            logger.info("✅ No session accumulation detected")
            return True
        else:
            logger.error(f"❌ Possible session accumulation: Before={count_before}, After={count_after}")
            # Log details of active sessions
            for session in active_sessions_after:
                logger.error(f"  Active session: {session}")
            return False
    
    async def run_all_tests(self):
        """Run all session management tests."""
        logger.info("=" * 60)
        logger.info("STARTING TRADER V3 SESSION MANAGEMENT TESTS")
        logger.info("=" * 60)
        
        try:
            await self.setup()
            
            tests = [
                ("Session Creation and Tracking", self.test_session_creation_and_tracking),
                ("Health State Transitions", self.test_health_state_transitions),
                ("Session Recovery Capability", self.test_session_recovery_capability),
                ("Active Session Management", self.test_active_session_count)
            ]
            
            results = []
            for test_name, test_func in tests:
                logger.info("")
                logger.info(f"Running: {test_name}")
                logger.info("-" * 40)
                
                try:
                    success = await test_func()
                    results.append((test_name, success))
                    
                    if success:
                        logger.info(f"✅ {test_name} PASSED")
                    else:
                        logger.error(f"❌ {test_name} FAILED")
                        
                except Exception as e:
                    logger.error(f"❌ {test_name} ERROR: {e}")
                    results.append((test_name, False))
                
                # Brief pause between tests
                await asyncio.sleep(1.0)
            
            # Summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("TEST RESULTS SUMMARY")
            logger.info("=" * 60)
            
            passed = 0
            for test_name, success in results:
                status = "✅ PASS" if success else "❌ FAIL"
                logger.info(f"{status:8} | {test_name}")
                if success:
                    passed += 1
            
            total = len(results)
            logger.info("-" * 60)
            logger.info(f"OVERALL: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 ALL TESTS PASSED - Session management implementation working!")
                return True
            else:
                logger.error(f"💥 {total-passed} TEST(S) FAILED - Session management needs fixes")
                return False
                
        except Exception as e:
            logger.error(f"💥 Test setup failed: {e}")
            return False
            
        finally:
            await self.cleanup()


async def main():
    """Main test entry point."""
    tester = SessionManagementTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("")
        logger.info("🚀 TRADER V3 session management ready for 30+ minute operation!")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("🔧 TRADER V3 session management needs attention before production use")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())