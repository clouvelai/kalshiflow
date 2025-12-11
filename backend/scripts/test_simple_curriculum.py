#!/usr/bin/env python3
"""
Test script for simple market-by-market curriculum.

This script demonstrates the simple curriculum working on session 9:
1. Load session 9
2. Find all viable markets (≥50 timesteps)
3. For each market, create a MarketSessionView
4. Show that we can iterate through each market
5. Simulate training on each market (without actual SB3 training)

This validates that the simple curriculum infrastructure works before
adding real training.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.training.simple_curriculum import (
    SimpleMarketCurriculum, train_simple_curriculum
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def simulate_training_on_market(market_view, market_ticker: str):
    """
    Simulate training on a single market.
    
    This is a dummy training function that shows what data is available
    and simulates training without actually running SB3.
    """
    logger.info(f"📊 Starting simulated training on {market_ticker}")
    
    # Simulate some analysis time
    await asyncio.sleep(0.1)
    
    # Analyze the market view
    episode_length = market_view.get_episode_length()
    
    # Sample a few data points to show what's available
    sample_indices = [0, episode_length // 2, -1] if episode_length > 0 else []
    sample_data = []
    
    for idx in sample_indices:
        actual_idx = idx if idx >= 0 else episode_length + idx
        if 0 <= actual_idx < episode_length:
            dp = market_view.get_timestep_data(actual_idx)
            if dp and market_ticker in dp.markets_data:
                market_data = dp.markets_data[market_ticker]
                
                # Calculate basic stats
                volume = 0
                for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                    if side in market_data:
                        volume += sum(market_data[side].values())
                
                sample_data.append({
                    'step': actual_idx,
                    'timestamp': dp.timestamp.strftime('%H:%M:%S'),
                    'time_gap': dp.time_gap,
                    'activity': dp.activity_score,
                    'volume': volume
                })
    
    # Log analysis results
    logger.info(f"  Market: {market_ticker}")
    logger.info(f"  Episode length: {episode_length} timesteps")
    logger.info(f"  Duration: {market_view.total_duration}")
    logger.info(f"  Coverage: {market_view.market_coverage:.1%}")
    logger.info(f"  Quality score: {market_view.data_quality_score:.2%}")
    
    if sample_data:
        logger.info(f"  Sample data points:")
        for sample in sample_data:
            logger.info(f"    Step {sample['step']:4d}: {sample['timestamp']} "
                       f"(gap={sample['time_gap']:5.1f}s, activity={sample['activity']:.2f}, "
                       f"volume={sample['volume']:8.0f})")
    
    # Simulate training metrics
    simulated_results = {
        'market_ticker': market_ticker,
        'episode_length': episode_length,
        'duration_seconds': market_view.total_duration.total_seconds(),
        'quality_score': market_view.data_quality_score,
        'coverage': market_view.market_coverage,
        'sample_data': sample_data,
        'simulated_reward': episode_length * 0.01,  # Dummy reward
        'training_completed': True
    }
    
    logger.info(f"✅ Simulated training completed for {market_ticker}")
    return simulated_results


async def test_simple_curriculum():
    """Test the simple curriculum with session 9."""
    logger.info("="*80)
    logger.info("TESTING SIMPLE MARKET-BY-MARKET CURRICULUM")
    logger.info("="*80)
    
    # Check environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("❌ DATABASE_URL not set in environment")
        return
    
    session_id = 9
    min_timesteps = 50
    
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Min timesteps required: {min_timesteps}")
    logger.info(f"Database URL: {database_url[:50]}...")
    
    try:
        # Test the simple curriculum
        results = await train_simple_curriculum(
            session_id=session_id,
            training_func=simulate_training_on_market,
            min_timesteps=min_timesteps,
            database_url=database_url
        )
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("CURRICULUM TEST RESULTS")
        logger.info("="*80)
        
        logger.info(f"Session ID: {results['session_id']}")
        logger.info(f"Markets found and trained: {len(results['markets_trained'])}")
        
        if results['markets_trained']:
            logger.info(f"\nMarkets trained on:")
            for i, market in enumerate(results['markets_trained'], 1):
                market_result = results['market_results'][market]
                if 'error' not in market_result:
                    logger.info(f"  {i:2d}. {market:<40} "
                               f"({market_result['episode_length']:4d} timesteps, "
                               f"{market_result['duration_seconds']:6.0f}s, "
                               f"quality={market_result['quality_score']:.1%})")
                else:
                    logger.info(f"  {i:2d}. {market:<40} ❌ ERROR: {market_result['error']}")
        
        # Summary statistics
        successful_markets = [m for m in results['markets_trained'] 
                            if 'error' not in results['market_results'][m]]
        
        if successful_markets:
            total_timesteps = sum(results['market_results'][m]['episode_length'] 
                                for m in successful_markets)
            total_duration = sum(results['market_results'][m]['duration_seconds'] 
                               for m in successful_markets)
            avg_quality = sum(results['market_results'][m]['quality_score'] 
                            for m in successful_markets) / len(successful_markets)
            
            logger.info(f"\nSUMMARY STATISTICS:")
            logger.info(f"  Successful markets: {len(successful_markets)}")
            logger.info(f"  Total training timesteps: {total_timesteps:,}")
            logger.info(f"  Total session duration: {total_duration:.0f} seconds ({total_duration/3600:.1f} hours)")
            logger.info(f"  Average data quality: {avg_quality:.1%}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ SIMPLE CURRICULUM TEST PASSED!")
        logger.info("="*80)
        logger.info("The infrastructure is ready for real SB3 training.")
        logger.info("Each market can be trained on independently using MarketSessionView.")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Curriculum test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_curriculum_initialization():
    """Test just the curriculum initialization to debug any issues."""
    logger.info("Testing curriculum initialization...")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("❌ DATABASE_URL not set")
        return False
    
    try:
        curriculum = SimpleMarketCurriculum(database_url, session_id=9, min_timesteps=50)
        await curriculum.initialize()
        
        summary = curriculum.get_training_summary()
        logger.info(f"✅ Curriculum initialized successfully:")
        logger.info(f"  Session: {summary['session_id']}")
        logger.info(f"  Viable markets: {summary['total_viable_markets']}")
        logger.info(f"  Markets: {[m['market'] for m in summary['viable_markets'][:5]]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Curriculum initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Simple Curriculum Test Script")
    print("="*40)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test simple curriculum')
    parser.add_argument('--init-only', action='store_true',
                       help='Test only curriculum initialization')
    args = parser.parse_args()
    
    if args.init_only:
        success = asyncio.run(test_curriculum_initialization())
        sys.exit(0 if success else 1)
    else:
        try:
            results = asyncio.run(test_simple_curriculum())
            sys.exit(0)
        except Exception:
            sys.exit(1)