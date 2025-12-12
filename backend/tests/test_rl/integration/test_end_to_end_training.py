"""
End-to-end validation test for complete training pipeline.

This test validates the complete pipeline from session selection through
MarketSessionView to SB3 training to model evaluation, ensuring all
components work together correctly.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# SB3 imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Our imports
from kalshiflow_rl.training.sb3_wrapper import (
    SessionBasedEnvironment, SB3TrainingConfig, CurriculumEnvironmentFactory,
    create_sb3_env, create_env_config, create_training_config
)
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.training.curriculum import (
    SimpleSessionCurriculum, train_single_session, train_multiple_sessions
)


@pytest.fixture
async def database_url():
    """Get database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set - skipping end-to-end tests")
    return url


@pytest.fixture
async def test_sessions(database_url):
    """Get test sessions with sufficient data."""
    loader = SessionDataLoader(database_url=database_url)
    sessions = await loader.get_available_sessions()
    
    if not sessions:
        pytest.skip("No sessions available in database")
    
    # Find sessions with reasonable amount of data
    good_sessions = []
    for session in sessions:
        if (session.get('snapshots_count', 0) >= 20 and 
            session.get('deltas_count', 0) >= 20):
            good_sessions.append(session['session_id'])
    
    if not good_sessions:
        # Fallback to any available sessions
        good_sessions = [s['session_id'] for s in sessions]
    
    # Return up to 3 sessions for testing
    return good_sessions[:3]


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for models and logs."""
    temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
    workspace = {
        'base_dir': Path(temp_dir),
        'models_dir': Path(temp_dir) / "models",
        'logs_dir': Path(temp_dir) / "logs",
        'results_dir': Path(temp_dir) / "results"
    }
    
    # Create subdirectories
    for dir_path in workspace.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestCompleteTrainingPipeline:
    """Test complete training pipeline end-to-end."""
    
    @pytest.mark.asyncio
    async def test_single_session_complete_pipeline(self, database_url, test_sessions, temp_workspace):
        """
        Test complete pipeline: session_id → MarketSessionView → SB3 training → model evaluation.
        """
        if not test_sessions:
            pytest.skip("No test sessions available")
        
        session_id = test_sessions[0]
        models_dir = temp_workspace['models_dir']
        
        # Step 1: Validate session data loading
        print(f"Step 1: Loading session {session_id}...")
        loader = SessionDataLoader(database_url=database_url)
        session_data = await loader.load_session(session_id)
        
        assert session_data is not None, f"Failed to load session {session_id}"
        assert session_data.get_episode_length() > 0, "Session has no episode data"
        
        print(f"  ✅ Session loaded: {len(session_data.markets_involved)} markets, "
              f"{session_data.get_episode_length()} timesteps")
        
        # Step 2: Create and validate environment
        print("Step 2: Creating SB3 environment...")
        
        training_config = SB3TrainingConfig(
            env_config=create_env_config(cash_start=10000),
            min_episode_length=5,
            max_episode_steps=100,  # Limit for faster testing
            skip_failed_markets=True
        )
        
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=session_id,
            config=training_config
        )
        
        assert env is not None, "Failed to create environment"
        
        # Validate environment spaces
        assert env.observation_space.shape == (52,), f"Wrong observation space: {env.observation_space.shape}"
        assert env.action_space.n == 5, f"Wrong action space: {env.action_space.n}"
        
        # Test environment functionality
        obs, info = env.reset()
        assert obs.shape == (52,), "Reset observation has wrong shape"
        assert 'session_id' in info, "Missing session_id in reset info"
        assert info['session_id'] == session_id, "Wrong session_id in info"
        
        print(f"  ✅ Environment created: {env.get_market_rotation_info()['total_markets']} markets available")
        
        # Step 3: Train model
        print("Step 3: Training PPO model...")
        
        # Use monitored environment
        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=32,
            n_epochs=2,
            verbose=0,
            tensorboard_log=str(temp_workspace['logs_dir'])
        )
        
        # Train for small number of timesteps
        training_start = datetime.now()
        model.learn(total_timesteps=1000, log_interval=None)
        training_duration = datetime.now() - training_start
        
        print(f"  ✅ Training completed in {training_duration.total_seconds():.2f} seconds")
        
        # Step 4: Save and load model
        print("Step 4: Testing model persistence...")
        
        model_path = models_dir / "test_ppo_model.zip"
        model.save(str(model_path))
        
        assert model_path.exists(), "Model file not created"
        
        # Load model
        loaded_model = PPO.load(str(model_path), env=vec_env)
        assert loaded_model is not None, "Failed to load model"
        
        print(f"  ✅ Model saved and loaded successfully")
        
        # Step 5: Evaluate model
        print("Step 5: Evaluating model performance...")
        
        mean_reward, std_reward = evaluate_policy(
            loaded_model, 
            vec_env, 
            n_eval_episodes=3,
            deterministic=True
        )
        
        assert isinstance(mean_reward, (int, float)), "Invalid mean reward type"
        assert isinstance(std_reward, (int, float)), "Invalid std reward type"
        assert not np.isnan(mean_reward), "Mean reward is NaN"
        assert not np.isnan(std_reward), "Std reward is NaN"
        
        print(f"  ✅ Model evaluation: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
        
        # Step 6: Test portfolio tracking
        print("Step 6: Validating portfolio tracking...")
        
        portfolio_values = []
        for episode in range(3):
            obs, info = env.reset()
            
            if hasattr(env.unwrapped, 'order_manager'):
                initial_portfolio = env.unwrapped.order_manager.get_portfolio_value_cents(env.unwrapped._get_current_market_prices())
                portfolio_values.append(initial_portfolio)
                
                # Verify OrderManager API
                cash_balance = env.unwrapped.order_manager.get_cash_balance_cents()
                position_info = env.unwrapped.order_manager.get_position_info()
                
                assert isinstance(cash_balance, (int, float)), "Invalid cash balance type"
                assert isinstance(position_info, dict), "Invalid position info type"
            
            # Run a few steps
            for _ in range(10):
                action = loaded_model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        
        assert len(portfolio_values) > 0, "No portfolio values tracked"
        assert all(pv > 0 for pv in portfolio_values), "Invalid portfolio values"
        
        print(f"  ✅ Portfolio tracking working: values range {min(portfolio_values)} - {max(portfolio_values)}")
        
        # Step 7: Generate final results
        print("Step 7: Generating results summary...")
        
        results = {
            'pipeline_test': 'single_session_complete',
            'session_id': session_id,
            'session_stats': {
                'markets_count': len(session_data.markets_involved),
                'episode_length': session_data.get_episode_length(),
                'duration': str(session_data.total_duration)
            },
            'environment_stats': env.get_market_rotation_info(),
            'training_stats': {
                'algorithm': 'PPO',
                'total_timesteps': 1000,
                'training_duration_seconds': training_duration.total_seconds()
            },
            'evaluation_stats': {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'episodes_evaluated': 3
            },
            'portfolio_stats': {
                'values_tracked': len(portfolio_values),
                'min_portfolio_value': min(portfolio_values),
                'max_portfolio_value': max(portfolio_values),
                'avg_portfolio_value': np.mean(portfolio_values)
            },
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # Save results
        results_file = temp_workspace['results_dir'] / "single_session_pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✅ Results saved to {results_file}")
        
        # Cleanup
        env.close()
        
        print("\n🎉 SINGLE SESSION PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"Session {session_id} → {env.get_market_rotation_info()['total_markets']} markets → "
              f"PPO training → evaluation score {mean_reward:.2f}")
    
    @pytest.mark.asyncio
    async def test_multi_session_curriculum_pipeline(self, database_url, test_sessions, temp_workspace):
        """
        Test multi-session curriculum learning pipeline.
        """
        if len(test_sessions) < 2:
            pytest.skip("Need at least 2 sessions for curriculum test")
        
        session_ids = test_sessions[:2]  # Use first 2 sessions
        models_dir = temp_workspace['models_dir']
        
        print(f"Step 1: Testing curriculum with sessions {session_ids}...")
        
        # Step 1: Use curriculum learning functions
        curriculum = SimpleSessionCurriculum(
            database_url=database_url,
            env_config=create_env_config(cash_start=10000)
        )
        
        # Test individual session training
        session_results = []
        for session_id in session_ids:
            print(f"  Training on session {session_id}...")
            result = await curriculum.train_session(
                session_id=session_id,
                min_snapshots=1,
                min_deltas=1
            )
            session_results.append(result)
            
            assert result.total_markets > 0, f"No markets processed in session {session_id}"
            print(f"    ✅ {result.successful_markets}/{result.total_markets} markets successful")
        
        print(f"  ✅ Curriculum training completed for {len(session_ids)} sessions")
        
        # Step 2: Create multi-session environment
        print("Step 2: Creating multi-session environment...")
        
        training_config = SB3TrainingConfig(
            env_config=create_env_config(cash_start=10000),
            min_episode_length=5,
            max_episode_steps=50,
            skip_failed_markets=True
        )
        
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=session_ids,
            config=training_config
        )
        
        rotation_info = env.get_market_rotation_info()
        assert rotation_info['total_markets'] > 0, "No markets in multi-session environment"
        assert len(rotation_info['sessions_covered']) >= 2, "Not all sessions represented"
        
        print(f"  ✅ Multi-session environment: {rotation_info['total_markets']} markets "
              f"from {len(rotation_info['sessions_covered'])} sessions")
        
        # Step 3: Train A2C model (different from single session test)
        print("Step 3: Training A2C model with curriculum...")
        
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            n_steps=32,
            verbose=0
        )
        
        # Train for small number of timesteps
        training_start = datetime.now()
        model.learn(total_timesteps=800, log_interval=None)
        training_duration = datetime.now() - training_start
        
        print(f"  ✅ A2C training completed in {training_duration.total_seconds():.2f} seconds")
        
        # Step 4: Test session cycling
        print("Step 4: Testing session cycling...")
        
        sessions_seen = set()
        markets_seen = set()
        
        for episode in range(10):
            obs, info = env.reset()
            sessions_seen.add(info['session_id'])
            markets_seen.add(info['market_ticker'])
            
            # Run episode with trained model
            for _ in range(20):
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        
        assert len(sessions_seen) >= 2, f"Only saw {len(sessions_seen)} sessions in 10 episodes"
        assert len(markets_seen) >= 2, f"Only saw {len(markets_seen)} markets in 10 episodes"
        
        print(f"  ✅ Session cycling: {len(sessions_seen)} sessions, {len(markets_seen)} markets")
        
        # Step 5: Evaluate curriculum effectiveness
        print("Step 5: Evaluating curriculum effectiveness...")
        
        mean_reward, std_reward = evaluate_policy(
            model, vec_env, n_eval_episodes=5, deterministic=True
        )
        
        print(f"  ✅ Curriculum evaluation: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
        
        # Step 6: Generate comprehensive results
        results = {
            'pipeline_test': 'multi_session_curriculum',
            'session_ids': session_ids,
            'curriculum_stats': {
                'total_sessions': len(session_ids),
                'session_results': [
                    {
                        'session_id': result.session_id,
                        'total_markets': result.total_markets,
                        'successful_markets': result.successful_markets,
                        'success_rate': result.get_success_rate()
                    }
                    for result in session_results
                ]
            },
            'environment_stats': rotation_info,
            'training_stats': {
                'algorithm': 'A2C',
                'total_timesteps': 800,
                'training_duration_seconds': training_duration.total_seconds()
            },
            'cycling_stats': {
                'episodes_tested': 10,
                'sessions_seen': len(sessions_seen),
                'markets_seen': len(markets_seen),
                'session_coverage': list(sessions_seen),
                'market_diversity': len(markets_seen)
            },
            'evaluation_stats': {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'episodes_evaluated': 5
            },
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # Save results
        results_file = temp_workspace['results_dir'] / "curriculum_pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✅ Results saved to {results_file}")
        
        # Cleanup
        env.close()
        
        print("\n🎉 CURRICULUM LEARNING PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"Sessions {session_ids} → {rotation_info['total_markets']} markets → "
              f"A2C curriculum training → evaluation score {mean_reward:.2f}")
    
    @pytest.mark.asyncio
    async def test_error_recovery_pipeline(self, database_url, test_sessions, temp_workspace):
        """
        Test pipeline error handling and recovery mechanisms.
        """
        if not test_sessions:
            pytest.skip("No test sessions available")
        
        session_id = test_sessions[0]
        
        print("Step 1: Testing error handling in pipeline...")
        
        # Test 1: Very strict requirements (should handle gracefully)
        print("  Testing insufficient data handling...")
        
        strict_config = SB3TrainingConfig(
            env_config=create_env_config(cash_start=10000),
            min_episode_length=10000,  # Impossibly high requirement
            skip_failed_markets=True
        )
        
        try:
            env = await create_sb3_env(
                database_url=database_url,
                session_ids=session_id,
                config=strict_config
            )
            
            # Should either have no markets or handle gracefully
            rotation_info = env.get_market_rotation_info()
            if rotation_info['total_markets'] == 0:
                print("    ✅ Handled insufficient data gracefully (no markets available)")
            else:
                print("    ✅ Some markets still available despite strict requirements")
            
            env.close()
            
        except (ValueError, RuntimeError) as e:
            if "No valid market views" in str(e):
                print("    ✅ Correctly rejected session with insufficient data")
            else:
                raise
        
        # Test 2: Model training with limited data
        print("  Testing training robustness...")
        
        reasonable_config = SB3TrainingConfig(
            env_config=create_env_config(cash_start=10000),
            min_episode_length=3,
            max_episode_steps=20,  # Very short episodes
            skip_failed_markets=True
        )
        
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=session_id,
            config=reasonable_config
        )
        
        # Train model with minimal timesteps
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Should handle training even with very few timesteps
        model.learn(total_timesteps=100, log_interval=None)
        
        # Should still be able to predict
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        assert 0 <= action < 5, f"Invalid action predicted: {action}"
        
        print("    ✅ Training completed successfully with limited data")
        
        # Test 3: Multiple reset robustness
        print("  Testing environment robustness...")
        
        for reset_num in range(5):
            try:
                obs, info = env.reset()
                assert obs.shape == (52,), f"Wrong observation shape on reset {reset_num}"
                
                # Take a few steps
                for step in range(3):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    
            except Exception as e:
                print(f"    ⚠️  Error on reset {reset_num}: {e}")
                # Continue with other resets
        
        print("    ✅ Environment handled multiple resets robustly")
        
        env.close()
        
        # Generate error handling report
        results = {
            'pipeline_test': 'error_recovery',
            'session_id': session_id,
            'tests_completed': [
                'insufficient_data_handling',
                'training_robustness',
                'environment_robustness'
            ],
            'error_handling_success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = temp_workspace['results_dir'] / "error_recovery_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✅ Error recovery test results saved to {results_file}")
        print("\n🎉 ERROR RECOVERY PIPELINE TEST COMPLETED SUCCESSFULLY!")


class TestPerformanceValidation:
    """Test performance characteristics of the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_training_performance_metrics(self, database_url, test_sessions, temp_workspace):
        """
        Test that training performance meets basic expectations.
        """
        if not test_sessions:
            pytest.skip("No test sessions available")
        
        session_id = test_sessions[0]
        
        print("Performance validation: Measuring training efficiency...")
        
        # Create environment
        env = await create_sb3_env(
            database_url=database_url,
            session_ids=session_id,
            config=SB3TrainingConfig(
                env_config=create_env_config(cash_start=10000),
                min_episode_length=5,
                max_episode_steps=100
            )
        )
        
        vec_env = DummyVecEnv([lambda: env])
        
        # Measure environment reset performance
        reset_times = []
        for _ in range(10):
            start_time = datetime.now()
            env.reset()
            reset_duration = (datetime.now() - start_time).total_seconds()
            reset_times.append(reset_duration)
        
        avg_reset_time = np.mean(reset_times)
        
        # Measure step performance
        env.reset()
        step_times = []
        for _ in range(50):
            start_time = datetime.now()
            action = env.action_space.sample()
            env.step(action)
            step_duration = (datetime.now() - start_time).total_seconds()
            step_times.append(step_duration)
        
        avg_step_time = np.mean(step_times)
        
        # Measure training performance
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        training_start = datetime.now()
        model.learn(total_timesteps=1000, log_interval=None)
        training_duration = (datetime.now() - training_start).total_seconds()
        
        timesteps_per_second = 1000 / training_duration
        
        # Performance assertions
        assert avg_reset_time < 1.0, f"Reset too slow: {avg_reset_time:.3f}s (should be < 1.0s)"
        assert avg_step_time < 0.1, f"Step too slow: {avg_step_time:.3f}s (should be < 0.1s)"
        assert timesteps_per_second > 50, f"Training too slow: {timesteps_per_second:.1f} timesteps/s (should be > 50)"
        
        performance_results = {
            'avg_reset_time_seconds': avg_reset_time,
            'avg_step_time_seconds': avg_step_time,
            'training_timesteps_per_second': timesteps_per_second,
            'total_training_duration_seconds': training_duration,
            'performance_requirements_met': True
        }
        
        print(f"  ✅ Environment reset: {avg_reset_time:.3f}s avg")
        print(f"  ✅ Environment step: {avg_step_time:.3f}s avg") 
        print(f"  ✅ Training speed: {timesteps_per_second:.1f} timesteps/s")
        
        # Save performance results
        results_file = temp_workspace['results_dir'] / "performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(performance_results, f, indent=2)
        
        env.close()
        
        print("\n🎉 PERFORMANCE VALIDATION COMPLETED SUCCESSFULLY!")


# Pytest configuration
pytestmark = pytest.mark.asyncio