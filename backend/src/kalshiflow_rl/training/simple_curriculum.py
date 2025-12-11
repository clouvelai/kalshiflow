#!/usr/bin/env python3
"""
Simple market-by-market curriculum implementation.

This module provides a basic curriculum that:
1. Loads a session 
2. Finds all viable markets (≥50 timesteps)
3. Trains on each market exactly once
4. NO adaptive behavior, NO mastery detection, NO complex logic

Just simple iteration: market1 -> train -> market2 -> train -> etc.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader

logger = logging.getLogger(__name__)


class SimpleMarketCurriculum:
    """
    Simple curriculum that trains on each viable market in a session exactly once.
    
    No complex logic - just finds markets with enough data and iterates through them.
    """
    
    def __init__(self, database_url: str, session_id: int, min_timesteps: int = 50):
        """
        Initialize simple curriculum.
        
        Args:
            database_url: PostgreSQL connection string
            session_id: Session to train on
            min_timesteps: Minimum timesteps required for a market to be viable
        """
        self.database_url = database_url
        self.session_id = session_id
        self.min_timesteps = min_timesteps
        self.session_data = None
        self.viable_markets = []
        self.current_market_index = 0
        
    async def initialize(self):
        """Load session data and find viable markets."""
        logger.info(f"Loading session {self.session_id}...")
        
        loader = SessionDataLoader(database_url=self.database_url)
        self.session_data = await loader.load_session(self.session_id)
        
        if not self.session_data:
            raise ValueError(f"Failed to load session {self.session_id}")
        
        logger.info(f"Session loaded: {self.session_data.get_episode_length()} timesteps, "
                   f"{len(self.session_data.markets_involved)} total markets")
        
        # Find viable markets
        self._find_viable_markets()
        
        if not self.viable_markets:
            raise ValueError(f"No viable markets found in session {self.session_id} "
                           f"with min {self.min_timesteps} timesteps")
        
        logger.info(f"Found {len(self.viable_markets)} viable markets:")
        for i, (market, count) in enumerate(self.viable_markets):
            logger.info(f"  {i+1}. {market}: {count} timesteps")
    
    def _find_viable_markets(self):
        """Find all markets with enough timesteps."""
        market_counts = {}
        
        # Count timesteps per market
        for data_point in self.session_data.data_points:
            for market_ticker in data_point.markets_data:
                market_counts[market_ticker] = market_counts.get(market_ticker, 0) + 1
        
        # Filter by minimum timesteps and sort by count (most data first)
        viable = [(market, count) for market, count in market_counts.items() 
                 if count >= self.min_timesteps]
        viable.sort(key=lambda x: x[1], reverse=True)
        
        self.viable_markets = viable
        logger.info(f"Found {len(viable)} viable markets (>={self.min_timesteps} timesteps)")
    
    def get_next_market(self) -> Optional[str]:
        """
        Get the next market to train on.
        
        Returns:
            Market ticker or None if all markets have been trained on
        """
        if self.current_market_index >= len(self.viable_markets):
            logger.info("All viable markets have been trained on")
            return None
        
        market_ticker = self.viable_markets[self.current_market_index][0]
        market_timesteps = self.viable_markets[self.current_market_index][1]
        
        logger.info(f"Next market: {market_ticker} ({market_timesteps} timesteps) "
                   f"[{self.current_market_index + 1}/{len(self.viable_markets)}]")
        
        return market_ticker
    
    def advance_to_next_market(self):
        """Move to the next market in the curriculum."""
        self.current_market_index += 1
    
    def create_market_view(self, market_ticker: str):
        """Create a MarketSessionView for the specified market."""
        if not self.session_data:
            raise ValueError("Session data not loaded. Call initialize() first.")
        
        return self.session_data.create_market_view(market_ticker)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum training progress."""
        return {
            'session_id': self.session_id,
            'total_viable_markets': len(self.viable_markets),
            'current_market_index': self.current_market_index,
            'markets_completed': self.current_market_index,
            'markets_remaining': len(self.viable_markets) - self.current_market_index,
            'min_timesteps_required': self.min_timesteps,
            'viable_markets': [
                {'market': market, 'timesteps': count} 
                for market, count in self.viable_markets
            ],
            'next_market': (self.viable_markets[self.current_market_index][0] 
                           if self.current_market_index < len(self.viable_markets) else None)
        }


async def train_simple_curriculum(session_id: int, 
                                training_func,
                                min_timesteps: int = 50,
                                database_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute simple curriculum training.
    
    Args:
        session_id: Session to train on
        training_func: Function that trains on a single market view
                      Should accept (market_view, market_ticker) and return training results
        min_timesteps: Minimum timesteps required for viable markets
        database_url: Database URL (uses environment if not provided)
    
    Returns:
        Dictionary with training results for all markets
    """
    if database_url is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not set in environment")
    
    # Initialize curriculum
    curriculum = SimpleMarketCurriculum(database_url, session_id, min_timesteps)
    await curriculum.initialize()
    
    logger.info(f"Starting simple curriculum training on session {session_id}")
    logger.info(f"Will train on {len(curriculum.viable_markets)} markets")
    
    training_results = {
        'session_id': session_id,
        'markets_trained': [],
        'training_summary': curriculum.get_training_summary(),
        'market_results': {}
    }
    
    # Train on each market exactly once
    market_ticker = curriculum.get_next_market()
    while market_ticker is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ON MARKET: {market_ticker}")
        logger.info(f"{'='*60}")
        
        # Create market view
        market_view = curriculum.create_market_view(market_ticker)
        if market_view is None:
            logger.error(f"Failed to create market view for {market_ticker}")
            curriculum.advance_to_next_market()
            market_ticker = curriculum.get_next_market()
            continue
        
        logger.info(f"Market view created: {market_view.get_episode_length()} timesteps")
        
        try:
            # Train on this market
            market_results = await training_func(market_view, market_ticker)
            
            training_results['markets_trained'].append(market_ticker)
            training_results['market_results'][market_ticker] = market_results
            
            logger.info(f"✅ Training completed for {market_ticker}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for {market_ticker}: {e}")
            training_results['market_results'][market_ticker] = {'error': str(e)}
        
        # Move to next market
        curriculum.advance_to_next_market()
        market_ticker = curriculum.get_next_market()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SIMPLE CURRICULUM COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Session {session_id}: Trained on {len(training_results['markets_trained'])} markets")
    logger.info(f"Markets: {training_results['markets_trained']}")
    
    return training_results


if __name__ == "__main__":
    """Simple test of curriculum functionality."""
    import asyncio
    
    async def dummy_training_func(market_view, market_ticker):
        """Dummy training function for testing."""
        await asyncio.sleep(0.1)  # Simulate training
        return {
            'market': market_ticker,
            'timesteps': market_view.get_episode_length(),
            'trained': True
        }
    
    async def main():
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL not set")
            return
        
        session_id = 9  # Default test session
        
        try:
            results = await train_simple_curriculum(
                session_id=session_id,
                training_func=dummy_training_func,
                min_timesteps=50
            )
            
            print("\n" + "="*60)
            print("CURRICULUM TEST RESULTS")
            print("="*60)
            print(f"Session: {results['session_id']}")
            print(f"Markets trained: {len(results['markets_trained'])}")
            print(f"Markets: {results['markets_trained']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())