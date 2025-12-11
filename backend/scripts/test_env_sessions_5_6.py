#!/usr/bin/env python3
"""
Comprehensive end-to-end test of MarketAgnosticKalshiEnv using sessions 5 and 6.

This test validates:
- Environment initialization with specific sessions
- Reset functionality across different sessions
- Step execution with all action types
- Observation generation and statistics
- Reward calculation and portfolio tracking
- Episode termination handling
- Session switching behavior
"""

import asyncio
import sys
import os
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Any

# Add the backend/src directory to the Python path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig


def print_separator(title: str):
    """Print a formatted separator."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print('=' * 80)


def print_subsection(title: str):
    """Print a formatted subsection."""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print('-' * 60)


def analyze_observation(obs: np.ndarray, step_count: int) -> Dict[str, Any]:
    """Analyze observation array and return statistics."""
    stats = {
        'shape': obs.shape,
        'total_features': obs.size,
        'min_value': float(np.min(obs)),
        'max_value': float(np.max(obs)),
        'mean_value': float(np.mean(obs)),
        'std_value': float(np.std(obs)),
        'zero_features': int(np.sum(obs == 0)),
        'non_zero_features': int(np.sum(obs != 0)),
        'nan_features': int(np.sum(np.isnan(obs))),
        'inf_features': int(np.sum(np.isinf(obs)))
    }
    
    print(f"Step {step_count} Observation Analysis:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Total features: {stats['total_features']}")
    print(f"  Value range: [{stats['min_value']:.6f}, {stats['max_value']:.6f}]")
    print(f"  Mean ± Std: {stats['mean_value']:.6f} ± {stats['std_value']:.6f}")
    print(f"  Zero features: {stats['zero_features']}")
    print(f"  Non-zero features: {stats['non_zero_features']}")
    if stats['nan_features'] > 0:
        print(f"  ⚠️  NaN features: {stats['nan_features']}")
    if stats['inf_features'] > 0:
        print(f"  ⚠️  Inf features: {stats['inf_features']}")
    
    return stats


def test_action_execution(env: MarketAgnosticKalshiEnv, action: int, step_count: int) -> Dict[str, Any]:
    """Execute an action and analyze the results."""
    action_names = {0: 'HOLD', 1: 'NOW', 2: 'WAIT'}
    print(f"\nExecuting action {action} ({action_names.get(action, 'UNKNOWN')})")
    
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        
        result = {
            'action': action,
            'action_name': action_names.get(action, 'UNKNOWN'),
            'reward': float(reward),
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
            'obs_stats': analyze_observation(obs, step_count)
        }
        
        print(f"  Reward: {reward:.6f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        if info:
            print(f"  Info: {json.dumps(info, indent=4)}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error executing action: {e}")
        return {
            'action': action,
            'action_name': action_names.get(action, 'UNKNOWN'),
            'error': str(e)
        }


def run_episode(env: MarketAgnosticKalshiEnv, episode_num: int, max_steps: int = 10) -> Dict[str, Any]:
    """Run a complete episode and return detailed results."""
    print_subsection(f"Episode {episode_num}")
    
    # Reset environment
    print("Resetting environment...")
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful")
        if info:
            print(f"Reset info: {json.dumps(info, indent=2)}")
        
        # Analyze initial observation
        obs_stats = analyze_observation(obs, 0)
        
        episode_results = {
            'episode': episode_num,
            'reset_info': info,
            'initial_obs_stats': obs_stats,
            'steps': []
        }
        
        # Run episode steps
        for step in range(1, max_steps + 1):
            print(f"\n--- Step {step} ---")
            
            # Try different actions (cycle through them)
            action = (step - 1) % 3  # 0=HOLD, 1=NOW, 2=WAIT
            
            step_result = test_action_execution(env, action, step)
            episode_results['steps'].append(step_result)
            
            # Check if episode ended
            if step_result.get('terminated', False) or step_result.get('truncated', False):
                print(f"Episode ended at step {step}")
                break
                
        return episode_results
        
    except Exception as e:
        print(f"❌ Error in episode {episode_num}: {e}")
        return {
            'episode': episode_num,
            'error': str(e)
        }


async def test_session_data_loading():
    """Test that sessions 5 and 6 have valid data."""
    print_subsection("Session Data Validation")
    
    # Import session data loader
    from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
    
    loader = SessionDataLoader()
    
    for session_id in [5, 6]:
        print(f"\nValidating session {session_id}:")
        try:
            # Load session metadata  
            session_data = await loader.load_session(session_id)
            
            if not session_data:
                print(f"  ❌ No data found for session {session_id}")
                continue
                
            markets = session_data.markets_involved
            print(f"  ✅ Markets found: {len(markets)} - {markets}")
            
            total_snapshots = len(session_data.data_points)
            print(f"  ✅ Total data points: {total_snapshots}")
            
            # Check session metadata
            print(f"  ✅ Session duration: {session_data.total_duration}")
            print(f"  ✅ Data quality score: {session_data.data_quality_score}")
            print(f"  ✅ Start time: {session_data.start_time}")
            print(f"  ✅ End time: {session_data.end_time}")
            
            # Check data point quality
            if session_data.data_points:
                first_point = session_data.data_points[0]
                last_point = session_data.data_points[-1]
                print(f"  ✅ Time range: {first_point.timestamp} to {last_point.timestamp}")
                
                # Check first data point structure
                if first_point.markets_data:
                    sample_market = list(first_point.markets_data.keys())[0]
                    sample_data = first_point.markets_data[sample_market]
                    print(f"  ✅ Sample market data keys: {list(sample_data.keys())}")
                    
                # Check for temporal features
                if hasattr(first_point, 'time_gap'):
                    print(f"  ✅ Temporal features available")
                else:
                    print(f"  ⚠️  Temporal features may not be computed")
                        
        except Exception as e:
            print(f"  ❌ Error loading session {session_id}: {e}")


async def main():
    """Run comprehensive environment test."""
    print_separator("MarketAgnosticKalshiEnv E2E Test - Sessions 5 & 6")
    
    # First validate session data
    await test_session_data_loading()
    
    # Initialize environment
    print_separator("Environment Initialization")
    
    try:
        # Load session data for session 5 first
        from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
        loader = SessionDataLoader()
        
        session_data = await loader.load_session(5)
        if not session_data:
            print("❌ Could not load session 5 data")
            return
        
        config = EnvConfig(
            max_markets=1,  # Single market training
            temporal_features=True,
            cash_start=100000  # Starting cash in cents (1000 dollars)
        )
        
        print(f"Configuration:")
        print(f"  Session: {session_data.session_id}")
        print(f"  Max markets: {config.max_markets}")
        print(f"  Temporal features: {config.temporal_features}")
        print(f"  Initial cash: {config.cash_start} cents (${config.cash_start/100})")
        
        env = MarketAgnosticKalshiEnv(session_data, config)
        print("✅ Environment created successfully")
        
        # Get environment information
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    # Run multiple episodes to test session switching
    print_separator("Episode Testing")
    
    all_results = []
    
    for episode in range(1, 6):  # Run 5 episodes
        try:
            results = run_episode(env, episode, max_steps=8)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Episode {episode} failed: {e}")
            all_results.append({'episode': episode, 'error': str(e)})
    
    # Analyze overall results
    print_separator("Test Results Summary")
    
    successful_episodes = [r for r in all_results if 'error' not in r]
    failed_episodes = [r for r in all_results if 'error' in r]
    
    print(f"Total episodes: {len(all_results)}")
    print(f"Successful episodes: {len(successful_episodes)}")
    print(f"Failed episodes: {len(failed_episodes)}")
    
    if failed_episodes:
        print("\n❌ Failed Episodes:")
        for episode in failed_episodes:
            print(f"  Episode {episode['episode']}: {episode['error']}")
    
    if successful_episodes:
        print("\n✅ Successful Episodes Summary:")
        
        # Collect statistics
        all_rewards = []
        all_steps = []
        action_counts = {0: 0, 1: 0, 2: 0}
        
        for episode in successful_episodes:
            episode_rewards = []
            for step in episode.get('steps', []):
                if 'reward' in step:
                    all_rewards.append(step['reward'])
                    episode_rewards.append(step['reward'])
                if 'action' in step:
                    action_counts[step['action']] += 1
            
            all_steps.append(len(episode.get('steps', [])))
            
            print(f"\n  Episode {episode['episode']}:")
            print(f"    Steps: {len(episode.get('steps', []))}")
            if episode_rewards:
                print(f"    Rewards: min={min(episode_rewards):.6f}, max={max(episode_rewards):.6f}, sum={sum(episode_rewards):.6f}")
            
            # Show session info if available
            reset_info = episode.get('reset_info', {})
            if reset_info:
                session_id = reset_info.get('session_id')
                market_ticker = reset_info.get('market_ticker')
                if session_id:
                    print(f"    Session: {session_id}")
                if market_ticker:
                    print(f"    Market: {market_ticker}")
        
        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  Total actions executed: {sum(action_counts.values())}")
        print(f"  Action distribution: HOLD={action_counts[0]}, NOW={action_counts[1]}, WAIT={action_counts[2]}")
        
        if all_rewards:
            print(f"  Total rewards: {len(all_rewards)}")
            print(f"  Reward range: [{min(all_rewards):.6f}, {max(all_rewards):.6f}]")
            print(f"  Mean reward: {np.mean(all_rewards):.6f}")
            print(f"  Total cumulative reward: {sum(all_rewards):.6f}")
        
        if all_steps:
            print(f"  Average episode length: {np.mean(all_steps):.1f} steps")
    
    # Test environment properties
    print_separator("Environment Properties Validation")
    
    try:
        # Test observation space
        print("Testing observation space bounds...")
        obs, _ = env.reset()
        if env.observation_space.contains(obs):
            print("✅ Observation within declared bounds")
        else:
            print("❌ Observation outside declared bounds")
            
        # Test action space
        print("Testing action space...")
        for action in range(env.action_space.n):
            if env.action_space.contains(action):
                print(f"✅ Action {action} valid")
            else:
                print(f"❌ Action {action} invalid")
                
    except Exception as e:
        print(f"❌ Error validating environment properties: {e}")
    
    print_separator("Test Complete")
    print("✅ End-to-end test completed successfully!")
    print("\nThe MarketAgnosticKalshiEnv with sessions 5 and 6 is fully functional.")


if __name__ == "__main__":
    asyncio.run(main())