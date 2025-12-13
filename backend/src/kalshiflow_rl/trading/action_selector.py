"""
ActionSelector for Kalshi Trading Actor - Minimal M1-M2 Stub Implementation.

This is a TEMPORARY stub for M1-M2 milestone work to enable end-to-end pipeline testing.
The full ActionSelector will be implemented in M3 with proper RL model integration.

Design:
- Always returns HOLD action (LimitOrderActions.HOLD = 0)
- Provides proper async interface for ActorService integration
- Logs all action requests for debugging M1-M2 pipeline
- Will be replaced with full RL implementation in M3

Note: This is NOT intended for production trading - it's purely for M1-M2 integration work.
"""

import logging
import time
from typing import Optional
import numpy as np

from ..environments.limit_order_action_space import LimitOrderActions, ActionType

logger = logging.getLogger("kalshiflow_rl.trading.action_selector")


class ActionSelectorStub:
    """
    TEMPORARY ActionSelector stub for M1-M2 pipeline integration.
    
    This minimal implementation always returns HOLD action to enable:
    - ActorService pipeline testing
    - Integration validation between components
    - M1-M2 milestone completion
    
    IMPORTANT: This will be replaced with full RL implementation in M3.
    """
    
    def __init__(self, debug: bool = True):
        """
        Initialize ActionSelector stub.
        
        Args:
            debug: Whether to log all action requests for debugging
        """
        self.debug = debug
        self.call_count = 0
        self.last_called_at: Optional[float] = None
        
        logger.info(f"🔧 ActionSelector STUB initialized for M1-M2 work")
        logger.info(f"   This is TEMPORARY - will be replaced with full RL in M3")
        logger.info(f"   Debug logging: {debug}")
    
    async def __call__(
        self, 
        observation: np.ndarray, 
        market_ticker: str
    ) -> int:
        """
        Select action for given observation and market.
        
        STUB BEHAVIOR: Always returns HOLD action for M1-M2 testing.
        
        Args:
            observation: Market observation array (ignored in stub)
            market_ticker: Market ticker (logged for debugging)
            
        Returns:
            int: Action ID (always 0 = HOLD for M1-M2)
        """
        self.call_count += 1
        self.last_called_at = time.time()
        
        if self.debug:
            logger.debug(
                f"ActionSelector STUB called #{self.call_count}: "
                f"market={market_ticker}, obs_shape={observation.shape} "
                f"-> returning HOLD (action=0)"
            )
        
        # TEMPORARY: Always return HOLD action for M1-M2 
        # This ensures pipeline works without actual RL logic
        return LimitOrderActions.HOLD.value  # Returns 0
    
    def get_stats(self) -> dict:
        """Get stub usage statistics for debugging."""
        return {
            "call_count": self.call_count,
            "last_called_at": self.last_called_at,
            "is_stub": True,
            "stub_action": "HOLD",
            "action_value": LimitOrderActions.HOLD.value
        }


# Factory function for ActorService integration
def create_action_selector_stub(debug: bool = True) -> ActionSelectorStub:
    """
    Create ActionSelector stub for M1-M2 integration.
    
    Args:
        debug: Enable debug logging
        
    Returns:
        ActionSelectorStub instance ready for ActorService.set_action_selector()
    """
    logger.info("Creating ActionSelector stub for M1-M2 pipeline testing")
    return ActionSelectorStub(debug=debug)


# Async function wrapper for direct ActorService integration
async def select_action_stub(observation: np.ndarray, market_ticker: str) -> int:
    """
    Simple async function for direct ActorService integration.
    
    TEMPORARY M1-M2 STUB: Always returns HOLD action.
    
    Args:
        observation: Market observation (ignored)
        market_ticker: Market ticker (logged)
        
    Returns:
        int: HOLD action (0)
    """
    logger.debug(f"select_action_stub called for {market_ticker} -> HOLD")
    return LimitOrderActions.HOLD.value


# ===============================================================================
# M1-M2 Integration Notes
# ===============================================================================
#
# This ActionSelector stub enables M1-M2 milestone completion by:
#
# 1. Providing proper async interface matching ActorService expectations
# 2. Always returning safe HOLD action (no actual trading in M1-M2)
# 3. Logging all calls for debugging pipeline integration
# 4. Supporting both class-based and function-based integration patterns
#
# Usage in ActorService:
#   # Option 1: Class-based 
#   selector = create_action_selector_stub()
#   actor_service.set_action_selector(selector)
#
#   # Option 2: Function-based
#   actor_service.set_action_selector(select_action_stub)
#
# M3 Replacement:
#   The full ActionSelector will:
#   - Load cached RL models from ActorService
#   - Use proper observation processing 
#   - Return intelligent trading actions based on market state
#   - Support multiple strategy selection (RL vs rule-based)
#   - Include action filtering and safety checks
#
# This stub will be DELETED when M3 implementation is complete.