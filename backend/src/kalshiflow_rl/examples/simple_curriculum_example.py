#!/usr/bin/env python3
"""
Simple curriculum learning example using SimpleSessionCurriculum.

This script demonstrates how to use the SimpleSessionCurriculum to train
on all valid markets in a session, tracking performance metadata.

Usage:
    python simple_curriculum_example.py [session_id]
    
Examples:
    python simple_curriculum_example.py 9        # Train on session 9
    python simple_curriculum_example.py          # Train on latest session
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kalshiflow_rl.training.curriculum import SimpleSessionCurriculum, train_single_session
from kalshiflow_rl.environments.market_agnostic_env import EnvConfig


async def run_simple_curriculum_example(session_id: int = None):
    """
    Run simple curriculum learning on a session.
    
    Args:
        session_id: Session to train on, or None for latest session
    """
    print("=" * 80)
    print("SIMPLE CURRICULUM LEARNING EXAMPLE")
    print("=" * 80)
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return
    
    # Create environment configuration
    env_config = EnvConfig(
        max_markets=1,           # Single market training  
        temporal_features=True,  # Include temporal features
        cash_start=10000        # Starting cash in cents ($100)
    )
    
    try:
        print(f"\n1. Initializing SimpleSessionCurriculum...")
        print(f"   Environment config: {env_config}")
        
        # Use convenience function for single session training
        print(f"\n2. Training session {session_id or 'latest'}...")
        
        if session_id is None:
            # Find latest session
            curriculum = SimpleSessionCurriculum(database_url, env_config)
            from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
            loader = SessionDataLoader(database_url)
            sessions = await loader.get_available_sessions()
            if not sessions:
                print("❌ No sessions found in database")
                return
            session_id = max(s['session_id'] for s in sessions)
            print(f"   Using latest session: {session_id}")
        
        # Train on the session
        results = await train_single_session(
            session_id=session_id,
            database_url=database_url, 
            env_config=env_config,
            min_snapshots=1,
            min_deltas=1
        )
        
        print(f"\n" + "=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        
        # Display session summary
        summary = results.get_summary()
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   Session ID:        {summary['session_id']}")
        print(f"   Total Markets:     {summary['total_markets']}")
        print(f"   Success Rate:      {summary['success_rate']:.1%}")
        print(f"   Successful:        {summary['successful_markets']}")
        print(f"   Failed:            {summary['failed_markets']}")
        print(f"   Total Episodes:    {summary['total_episodes']}")
        print(f"   Total Timesteps:   {summary['total_timesteps']}")
        print(f"   Duration:          {summary['total_duration']}")
        
        if summary['total_episodes'] > 0:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   Average Reward:    ${summary['avg_reward']:.2f}")
            print(f"   Best Reward:       ${summary['best_reward']:.2f}")
            print(f"   Worst Reward:      ${summary['worst_reward']:.2f}")
        
        # Display per-market details
        print(f"\n🎯 PER-MARKET RESULTS:")
        print(f"   {'Market':<40} {'Success':<8} {'Reward':<10} {'Length':<8} {'Coverage':<10}")
        print("   " + "-" * 80)
        
        for result in results.market_results:
            status = "✅ Yes" if result.success else "❌ No"
            reward = f"${result.total_reward:.2f}" if result.success else "N/A"
            length = str(result.episode_length) if result.success else "N/A"
            coverage = f"{result.market_coverage:.1%}" if result.success else "N/A"
            
            print(f"   {result.market_ticker[:38]:<40} {status:<8} {reward:<10} {length:<8} {coverage:<10}")
            
            if not result.success and result.error_message:
                print(f"      Error: {result.error_message}")
        
        print(f"\n" + "=" * 80)
        print("✅ SIMPLE CURRICULUM EXAMPLE COMPLETE")
        print("=" * 80)
        
        # Additional insights
        if results.successful_markets > 0:
            successful_results = [r for r in results.market_results if r.success]
            
            # Find best and worst performing markets
            best_market = max(successful_results, key=lambda r: r.total_reward)
            worst_market = min(successful_results, key=lambda r: r.total_reward)
            
            print(f"\n💡 INSIGHTS:")
            print(f"   Best performer:    {best_market.market_ticker} (${best_market.total_reward:.2f})")
            print(f"   Worst performer:   {worst_market.market_ticker} (${worst_market.total_reward:.2f})")
            
            # Coverage analysis
            high_coverage = [r for r in successful_results if r.market_coverage > 0.5]
            print(f"   High coverage (>50%): {len(high_coverage)} markets")
            
            # Episode length analysis  
            avg_length = sum(r.episode_length for r in successful_results) / len(successful_results)
            print(f"   Avg episode length: {avg_length:.1f} timesteps")
            
        return results
        
    except Exception as e:
        print(f"\n❌ Error during curriculum training: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_multi_session_example():
    """Example of training on multiple sessions sequentially."""
    print("\n" + "=" * 80) 
    print("MULTI-SESSION CURRICULUM EXAMPLE")
    print("=" * 80)
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return
    
    # Get available sessions
    from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
    loader = SessionDataLoader(database_url)
    sessions = await loader.get_available_sessions()
    
    if len(sessions) < 2:
        print(f"❌ Need at least 2 sessions for multi-session example (found {len(sessions)})")
        return
    
    # Use the 2 most recent sessions
    session_ids = sorted([s['session_id'] for s in sessions])[-2:]
    print(f"Training on sessions: {session_ids}")
    
    # Initialize curriculum
    curriculum = SimpleSessionCurriculum(
        database_url=database_url,
        env_config=EnvConfig(temporal_features=True, cash_start=10000)
    )
    
    # Train on each session
    for session_id in session_ids:
        print(f"\nTraining session {session_id}...")
        result = await curriculum.train_session(session_id)
        
        print(f"   Session {session_id}: "
              f"success_rate={result.get_success_rate():.1%}, "
              f"markets={result.total_markets}, "
              f"avg_reward=${result.avg_reward:.2f}")
    
    # Get overall summary
    overall = curriculum.get_overall_summary()
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   Total Sessions:     {overall['total_sessions']}")
    print(f"   Total Markets:      {overall['total_markets']}")
    print(f"   Overall Success:    {overall['overall_success_rate']:.1%}")
    print(f"   Avg Reward:         ${overall['avg_reward_across_sessions']:.2f}")
    
    return curriculum


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simple curriculum learning example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_curriculum_example.py 9      # Train on session 9  
  python simple_curriculum_example.py        # Train on latest session
  python simple_curriculum_example.py --multi # Train on multiple sessions
        """
    )
    
    parser.add_argument('session_id', type=int, nargs='?',
                       help='Session ID to train on (uses latest if not provided)')
    parser.add_argument('--multi', action='store_true',
                       help='Run multi-session training example')
    
    args = parser.parse_args()
    
    try:
        if args.multi:
            await run_multi_session_example()
        else:
            await run_simple_curriculum_example(args.session_id)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())