#!/usr/bin/env python3
"""
Validate MarketAgnosticKalshiEnv compatibility with Stable Baselines3.

This script validates that the MarketAgnosticKalshiEnv environment passes all
gymnasium and SB3 validation checks, confirming the 52-feature observation space
and 5-action Discrete space work correctly with SB3 algorithms.

Usage:
    python validate_sb3_environment.py [session_id]
    
    If no session_id provided, will use the most recent available session.

Examples:
    $ python validate_sb3_environment.py
    Uses most recent session for validation
    
    $ python validate_sb3_environment.py 9
    Uses session 9 for validation
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


async def get_test_environment(session_id: Optional[int] = None) -> MarketAgnosticKalshiEnv:
    """
    Create a test environment for validation using real session data.
    
    Args:
        session_id: Optional session ID to use. If None, uses most recent session.
        
    Returns:
        MarketAgnosticKalshiEnv instance ready for validation
    """
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set in environment")
    
    # Load session data
    loader = SessionDataLoader(database_url=database_url)
    
    # If no session_id provided, find the most recent one
    if session_id is None:
        sessions = await loader.get_available_sessions()
        if not sessions:
            raise ValueError("No closed sessions found in database")
        session_id = max(s['session_id'] for s in sessions)
        print(f"Using most recent session: {session_id}")
    
    print(f"Loading session {session_id}...")
    session_data = await loader.load_session(session_id)
    
    if not session_data:
        raise ValueError(f"Failed to load session {session_id}")
    
    print(f"Session loaded: {len(session_data.markets_involved)} markets, {session_data.get_episode_length()} timesteps")
    
    # Create market view for the first market with sufficient data
    for market_ticker in session_data.markets_involved:
        try:
            market_view = session_data.create_market_view(market_ticker)
            if market_view and market_view.get_episode_length() >= 10:
                print(f"Using market view: {market_ticker} ({market_view.get_episode_length()} timesteps)")
                
                # Create environment
                env_config = EnvConfig(
                    max_markets=1,
                    temporal_features=True,
                    cash_start=10000  # $100 in cents
                )
                
                env = MarketAgnosticKalshiEnv(
                    market_view=market_view,
                    config=env_config
                )
                
                return env
        except Exception as e:
            print(f"Failed to create view for {market_ticker}: {e}")
            continue
    
    raise ValueError("No markets found with sufficient data for validation")


def validate_observation_space(env: MarketAgnosticKalshiEnv) -> Dict[str, Any]:
    """
    Validate the observation space matches expected 52-feature structure.
    
    Args:
        env: Environment to validate
        
    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("OBSERVATION SPACE VALIDATION")
    print("="*60)
    
    results = {
        'observation_space_type': str(type(env.observation_space)),
        'observation_shape': env.observation_space.shape,
        'observation_dtype': str(env.observation_space.dtype),
        'expected_features': 52,
        'actual_features': env.observation_space.shape[0] if env.observation_space.shape else 0,
        'dimension_match': False,
        'sample_observation_valid': False,
        'reset_observation_valid': False
    }
    
    # Check observation space dimensions
    print(f"Observation space: {env.observation_space}")
    print(f"Expected features: 52")
    print(f"Actual features: {results['actual_features']}")
    
    results['dimension_match'] = (results['actual_features'] == 52)
    print(f"Dimension match: {'✅' if results['dimension_match'] else '❌'}")
    
    # Test observation space sampling
    try:
        sample_obs = env.observation_space.sample()
        print(f"Sample observation shape: {sample_obs.shape}")
        print(f"Sample observation dtype: {sample_obs.dtype}")
        results['sample_observation_valid'] = (sample_obs.shape == (52,))
        print(f"Sample valid: {'✅' if results['sample_observation_valid'] else '❌'}")
    except Exception as e:
        print(f"Sample observation failed: {e}")
        results['sample_observation_valid'] = False
    
    # Test environment reset observation
    try:
        reset_obs, reset_info = env.reset()
        print(f"Reset observation shape: {reset_obs.shape}")
        print(f"Reset observation dtype: {reset_obs.dtype}")
        print(f"Reset info keys: {list(reset_info.keys())}")
        results['reset_observation_valid'] = (reset_obs.shape == (52,))
        print(f"Reset observation valid: {'✅' if results['reset_observation_valid'] else '❌'}")
        
        # Check for NaN/inf values
        has_nan = np.isnan(reset_obs).any()
        has_inf = np.isinf(reset_obs).any()
        print(f"Contains NaN: {'❌' if has_nan else '✅'}")
        print(f"Contains Inf: {'❌' if has_inf else '✅'}")
        
        results['observation_stats'] = {
            'min': float(np.min(reset_obs)),
            'max': float(np.max(reset_obs)),
            'mean': float(np.mean(reset_obs)),
            'std': float(np.std(reset_obs)),
            'has_nan': has_nan,
            'has_inf': has_inf
        }
        
    except Exception as e:
        print(f"Reset observation failed: {e}")
        results['reset_observation_valid'] = False
    
    return results


def validate_action_space(env: MarketAgnosticKalshiEnv) -> Dict[str, Any]:
    """
    Validate the action space is properly configured for SB3.
    
    Args:
        env: Environment to validate
        
    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("ACTION SPACE VALIDATION")
    print("="*60)
    
    results = {
        'action_space_type': str(type(env.action_space)),
        'action_space_n': env.action_space.n if hasattr(env.action_space, 'n') else None,
        'expected_actions': 5,
        'action_count_match': False,
        'sample_action_valid': False,
        'action_execution_test': False
    }
    
    # Check action space structure
    print(f"Action space: {env.action_space}")
    print(f"Expected actions: 5 (HOLD, BUY_YES_LIMIT, SELL_YES_LIMIT, BUY_NO_LIMIT, SELL_NO_LIMIT)")
    print(f"Actual actions: {results['action_space_n']}")
    
    results['action_count_match'] = (results['action_space_n'] == 5)
    print(f"Action count match: {'✅' if results['action_count_match'] else '❌'}")
    
    # Test action space sampling
    try:
        sample_action = env.action_space.sample()
        print(f"Sample action: {sample_action} (type: {type(sample_action)})")
        results['sample_action_valid'] = (0 <= sample_action < 5)
        print(f"Sample action valid: {'✅' if results['sample_action_valid'] else '❌'}")
    except Exception as e:
        print(f"Sample action failed: {e}")
        results['sample_action_valid'] = False
    
    # Test action execution
    try:
        env.reset()
        
        # Test all actions
        action_results = []
        for action in range(5):
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                action_results.append({
                    'action': action,
                    'success': True,
                    'reward': float(reward),
                    'terminated': terminated,
                    'truncated': truncated,
                    'info_keys': list(info.keys())
                })
            except Exception as action_e:
                action_results.append({
                    'action': action,
                    'success': False,
                    'error': str(action_e)
                })
                print(f"Action {action} failed: {action_e}")
            
            # Reset for next action test
            env.reset()
        
        successful_actions = sum(1 for r in action_results if r['success'])
        print(f"Successful actions: {successful_actions}/5")
        results['action_execution_test'] = (successful_actions >= 4)  # At least 4/5 should work
        print(f"Action execution test: {'✅' if results['action_execution_test'] else '❌'}")
        
        results['action_test_details'] = action_results
        
    except Exception as e:
        print(f"Action execution test failed: {e}")
        results['action_execution_test'] = False
    
    return results


def run_gymnasium_validation(env: MarketAgnosticKalshiEnv) -> Dict[str, Any]:
    """
    Run gymnasium's built-in environment validation.
    
    Args:
        env: Environment to validate
        
    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("GYMNASIUM VALIDATION")
    print("="*60)
    
    results = {
        'gymnasium_check_passed': False,
        'gymnasium_error': None
    }
    
    try:
        print("Running gymnasium.utils.env_checker.check_env()...")
        check_env(env, warn=True, skip_render_check=True)
        results['gymnasium_check_passed'] = True
        print("✅ Gymnasium validation passed!")
        
    except Exception as e:
        results['gymnasium_check_passed'] = False
        results['gymnasium_error'] = str(e)
        print(f"❌ Gymnasium validation failed: {e}")
    
    return results


def run_sb3_validation(env: MarketAgnosticKalshiEnv) -> Dict[str, Any]:
    """
    Run Stable Baselines3's environment validation.
    
    Args:
        env: Environment to validate
        
    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("STABLE BASELINES3 VALIDATION")
    print("="*60)
    
    results = {
        'sb3_check_passed': False,
        'sb3_error': None
    }
    
    try:
        print("Running stable_baselines3.common.env_checker.check_env()...")
        sb3_check_env(env, warn=True, skip_render_check=True)
        results['sb3_check_passed'] = True
        print("✅ Stable Baselines3 validation passed!")
        
    except Exception as e:
        results['sb3_check_passed'] = False
        results['sb3_error'] = str(e)
        print(f"❌ Stable Baselines3 validation failed: {e}")
    
    return results


def run_episode_simulation(env: MarketAgnosticKalshiEnv, max_steps: int = 50) -> Dict[str, Any]:
    """
    Run a simulated episode to test environment dynamics.
    
    Args:
        env: Environment to test
        max_steps: Maximum steps to simulate
        
    Returns:
        Dict with simulation results
    """
    print("\n" + "="*60)
    print("EPISODE SIMULATION")
    print("="*60)
    
    results = {
        'episode_completed': False,
        'steps_taken': 0,
        'total_reward': 0.0,
        'final_observation_valid': False,
        'portfolio_tracking': {},
        'error': None
    }
    
    try:
        obs, info = env.reset()
        print(f"Episode started with observation shape: {obs.shape}")
        
        total_reward = 0.0
        step_count = 0
        
        # Track initial portfolio state
        initial_cash = env.order_manager.get_cash_balance_cents()
        initial_portfolio = env.order_manager.get_portfolio_value_cents(env._get_current_market_prices())
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Check observation validity
            if np.isnan(obs).any() or np.isinf(obs).any():
                print(f"❌ Invalid observation at step {step}")
                break
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode terminated naturally at step {step}")
                results['episode_completed'] = True
                break
        
        # Final portfolio state
        final_cash = env.order_manager.get_cash_balance_cents()
        final_portfolio = env.order_manager.get_portfolio_value_cents(env._get_current_market_prices())
        
        results['steps_taken'] = step_count
        results['total_reward'] = total_reward
        results['final_observation_valid'] = not (np.isnan(obs).any() or np.isinf(obs).any())
        results['portfolio_tracking'] = {
            'initial_cash': initial_cash,
            'final_cash': final_cash,
            'initial_portfolio_value': initial_portfolio,
            'final_portfolio_value': final_portfolio,
            'portfolio_change': final_portfolio - initial_portfolio
        }
        
        print(f"Episode summary:")
        print(f"  Steps taken: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Portfolio change: {final_portfolio - initial_portfolio:.2f} cents")
        print(f"  Final observation valid: {'✅' if results['final_observation_valid'] else '❌'}")
        
        if not results['episode_completed'] and step_count < max_steps:
            print("⚠️  Episode ended early due to invalid observations")
        elif step_count == max_steps:
            print("⚠️  Episode reached maximum steps without natural termination")
        else:
            print("✅ Episode completed successfully")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Episode simulation failed: {e}")
    
    return results


async def main():
    """Main validation function."""
    print("="*80)
    print("STABLE BASELINES3 ENVIRONMENT VALIDATION")
    print("="*80)
    
    # Get session_id from command line args
    session_id = None
    if len(sys.argv) > 1:
        try:
            session_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid session_id: {sys.argv[1]}")
            sys.exit(1)
    
    try:
        # Create test environment
        print("Creating test environment...")
        env = await get_test_environment(session_id)
        
        # Run all validations
        validation_results = {}
        
        # 1. Observation space validation
        validation_results['observation_space'] = validate_observation_space(env)
        
        # 2. Action space validation
        validation_results['action_space'] = validate_action_space(env)
        
        # 3. Gymnasium validation
        validation_results['gymnasium'] = run_gymnasium_validation(env)
        
        # 4. Stable Baselines3 validation
        validation_results['sb3'] = run_sb3_validation(env)
        
        # 5. Episode simulation
        validation_results['episode_simulation'] = run_episode_simulation(env)
        
        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        # Check overall validation status
        critical_checks = [
            validation_results['observation_space']['dimension_match'],
            validation_results['observation_space']['reset_observation_valid'],
            validation_results['action_space']['action_count_match'],
            validation_results['action_space']['action_execution_test'],
            validation_results['gymnasium']['gymnasium_check_passed'],
            validation_results['sb3']['sb3_check_passed']
        ]
        
        overall_success = all(critical_checks)
        
        print(f"✅ Observation space (52 features): {'PASS' if validation_results['observation_space']['dimension_match'] else 'FAIL'}")
        print(f"✅ Action space (5 actions): {'PASS' if validation_results['action_space']['action_count_match'] else 'FAIL'}")
        print(f"✅ Gymnasium validation: {'PASS' if validation_results['gymnasium']['gymnasium_check_passed'] else 'FAIL'}")
        print(f"✅ Stable Baselines3 validation: {'PASS' if validation_results['sb3']['sb3_check_passed'] else 'FAIL'}")
        print(f"✅ Episode simulation: {'PASS' if validation_results['episode_simulation']['final_observation_valid'] else 'FAIL'}")
        
        print(f"\n{'='*80}")
        if overall_success:
            print("🎉 ALL VALIDATIONS PASSED - ENVIRONMENT IS SB3 READY!")
            print("The MarketAgnosticKalshiEnv is fully compatible with Stable Baselines3.")
        else:
            print("❌ VALIDATION FAILURES DETECTED")
            print("The environment needs fixes before SB3 integration.")
            
            # Print specific failures
            if not validation_results['gymnasium']['gymnasium_check_passed']:
                print(f"Gymnasium error: {validation_results['gymnasium']['gymnasium_error']}")
            if not validation_results['sb3']['sb3_check_passed']:
                print(f"SB3 error: {validation_results['sb3']['sb3_error']}")
        
        print("="*80)
        
        # Exit with appropriate code
        return 0 if overall_success else 1
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)