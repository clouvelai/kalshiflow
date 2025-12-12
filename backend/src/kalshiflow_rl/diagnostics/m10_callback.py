"""
M10 Diagnostics Callback for Stable Baselines3.

Integrates action tracking, reward analysis, and observation validation
into SB3 training pipeline with comprehensive diagnostics output.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .action_tracker import ActionTracker
from .reward_analyzer import RewardAnalyzer  
from .observation_validator import ObservationValidator
from .diagnostics_logger import DiagnosticsLogger, DiagnosticsConfig


class M10DiagnosticsCallback(BaseCallback):
    """
    Comprehensive M10 diagnostics callback for SB3 training.
    
    This callback provides the core M10 instrumentation to diagnose
    why agents exhibit HOLD-only behavior by tracking:
    
    - Action distribution patterns and exploration
    - Reward signal quality and sparsity
    - Observation space validation and health
    - Episode-level summaries and trends
    - Training progress diagnostics
    
    All diagnostics are logged to both console (for SB3) and organized
    files for detailed analysis.
    """
    
    def __init__(
        self,
        output_dir: str,
        session_id: Optional[int] = None,
        algorithm: str = "unknown",
        
        # Diagnostic component configuration
        action_tracking: bool = True,
        reward_analysis: bool = True,
        observation_validation: bool = True,
        
        # Logging configuration
        console_output: bool = True,
        detailed_logging: bool = True,
        
        # Frequency controls
        step_log_freq: int = 1000,     # Log step summaries every N steps
        episode_log_freq: int = 100,    # Log episode summaries every N episodes
        validation_freq: int = 100,     # Detailed observation validation frequency
        
        # Console summary frequency (for SB3 integration)
        console_summary_freq: int = 500,
        
        verbose: int = 1
    ):
        """
        Initialize M10 diagnostics callback.
        
        Args:
            output_dir: Directory for organized diagnostic output
            session_id: Session ID for organized file naming
            algorithm: Algorithm name for file organization
            action_tracking: Enable action distribution tracking
            reward_analysis: Enable reward signal analysis
            observation_validation: Enable observation validation
            console_output: Enable console output (for SB3)
            detailed_logging: Enable detailed file logging
            step_log_freq: Frequency of step-level logging
            episode_log_freq: Frequency of episode summary logging
            validation_freq: Frequency of detailed observation validation
            console_summary_freq: Frequency of console diagnostic summaries
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        # Configuration
        self.session_id = session_id
        self.algorithm = algorithm
        self.step_log_freq = step_log_freq
        self.episode_log_freq = episode_log_freq
        self.validation_freq = validation_freq
        self.console_summary_freq = console_summary_freq
        
        # Diagnostic components
        self.action_tracker: Optional[ActionTracker] = None
        self.reward_analyzer: Optional[RewardAnalyzer] = None
        self.observation_validator: Optional[ObservationValidator] = None
        
        if action_tracking:
            self.action_tracker = ActionTracker(max_events=50000)
        
        if reward_analysis:
            self.reward_analyzer = RewardAnalyzer(max_events=50000)
        
        if observation_validation:
            self.observation_validator = ObservationValidator(
                expected_obs_dim=52,  # MarketAgnosticKalshiEnv observation dimension
                validation_freq=validation_freq
            )
        
        # Consolidated logging
        diagnostics_config = DiagnosticsConfig(
            session_id=session_id,
            algorithm=algorithm,
            console_output=console_output,
            file_output=detailed_logging,
            json_output=detailed_logging
        )
        
        self.diagnostics_logger = DiagnosticsLogger(
            output_dir=output_dir,
            config=diagnostics_config,
            create_subdirs=True
        )
        
        # Training state tracking
        self.episode_count = 0
        self.current_episode_start_step = 0
        self.last_console_summary = 0
        self.last_episode_summary = 0
        
        # Performance tracking
        self.callback_start_time = time.time()
        self.total_callback_time = 0.0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.diagnostics_logger.log_training_milestone(
            {
                'event': 'training_started',
                'total_timesteps': getattr(self.model, 'total_timesteps', 'unknown'),
                'algorithm': self.algorithm,
                'session_id': self.session_id,
                'callback_config': {
                    'action_tracking': self.action_tracker is not None,
                    'reward_analysis': self.reward_analyzer is not None, 
                    'observation_validation': self.observation_validator is not None,
                    'step_log_freq': self.step_log_freq,
                    'episode_log_freq': self.episode_log_freq
                }
            },
            milestone_name="Training Started"
        )
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        callback_step_start = time.time()
        
        try:
            # Get environment and current state
            env = self.training_env.envs[0]
            if not hasattr(env, 'unwrapped'):
                return True
                
            unwrapped_env = env.unwrapped
            
            # Skip if environment not properly initialized
            if not hasattr(unwrapped_env, 'order_manager') or unwrapped_env.order_manager is None:
                return True
            
            # Get current state information
            current_step = getattr(unwrapped_env, 'current_step', 0)
            global_step = self.num_timesteps
            
            # Get action information
            action = self.locals.get('actions', [None])[0]
            if action is not None:
                action_int = int(action) if hasattr(action, 'item') else int(action)
            else:
                action_int = 0  # Default to HOLD
            
            # Get current observation
            obs = self.locals.get('new_obs', self.locals.get('observations', None))
            if obs is not None and len(obs) > 0:
                current_obs = obs[0] if hasattr(obs[0], 'shape') else obs
            else:
                current_obs = None
            
            # Get reward information  
            rewards = self.locals.get('rewards', [0.0])
            current_reward = float(rewards[0]) if len(rewards) > 0 else 0.0
            
            # Get market state for context
            market_state = self._extract_market_state(unwrapped_env)
            
            # Get portfolio information
            portfolio_info = self._extract_portfolio_info(unwrapped_env)
            
            # Action tracking
            if self.action_tracker:
                self.action_tracker.track_action(
                    action=action_int,
                    step=current_step,
                    global_step=global_step,
                    market_state=market_state,
                    action_probs=None,  # TODO: Extract from policy if available
                    exploration=False   # TODO: Detect exploration vs exploitation
                )
            
            # Reward analysis
            if self.reward_analyzer:
                self.reward_analyzer.track_reward(
                    reward=current_reward,
                    step=current_step,
                    global_step=global_step,
                    portfolio_value_change=portfolio_info.get('portfolio_change', 0.0),
                    previous_portfolio_value=portfolio_info.get('previous_portfolio_value', 0.0),
                    current_portfolio_value=portfolio_info.get('current_portfolio_value', 0.0),
                    position_info=portfolio_info.get('position_info', {}),
                    cash_balance=portfolio_info.get('cash_balance', 0.0),
                    market_state=market_state,
                    action=action_int,
                    action_name=ActionTracker.ACTION_NAMES.get(action_int, f"UNKNOWN_{action_int}")
                )
            
            # Observation validation
            if self.observation_validator and current_obs is not None:
                validation_result = self.observation_validator.validate_observation(
                    observation=current_obs,
                    step=current_step,
                    global_step=global_step,
                    detailed_validation=None  # Use frequency-based validation
                )
                
                # Log observation issues if detected
                if not validation_result.get('observation_valid', True):
                    self.diagnostics_logger.log_observation_event(
                        validation_result,
                        console_summary=f"Obs validation failed: {', '.join(validation_result.get('issues', []))}"
                    )
            
            # Periodic step logging
            if global_step % self.step_log_freq == 0:
                step_summary = {
                    'global_step': global_step,
                    'episode_step': current_step,
                    'action': action_int,
                    'reward': current_reward,
                    'portfolio_value': portfolio_info.get('current_portfolio_value', 0.0),
                    'market_state': market_state
                }
                
                self.diagnostics_logger.log_diagnostic(
                    category="training",
                    event_type="step_summary",
                    data=step_summary,
                    console_summary=f"Step {global_step}: A={action_int}, R={current_reward:.2f}"
                )
            
            # Console summary output (less frequent)
            if global_step > 0 and global_step % self.console_summary_freq == 0:
                self._log_console_summary(global_step)
            
            # Check for episode end
            dones = self.locals.get('dones', [False])
            if len(dones) > 0 and dones[0]:
                self._on_episode_end()
            
        except Exception as e:
            # Don't fail training due to diagnostics issues
            if self.verbose > 0:
                print(f"M10 Diagnostics error: {e}")
        
        finally:
            # Track callback overhead
            callback_duration = time.time() - callback_step_start
            self.total_callback_time += callback_duration
        
        return True
    
    def _on_episode_end(self) -> None:
        """Process episode completion and generate summaries."""
        self.episode_count += 1
        
        try:
            # Generate component summaries
            episode_summaries = {}
            
            if self.action_tracker:
                action_summary = self.action_tracker.end_episode()
                episode_summaries['action_analysis'] = action_summary
                
                # Log action summary
                self.diagnostics_logger.log_episode_summary({
                    'category': 'action_summary',
                    **action_summary
                })
            
            if self.reward_analyzer:
                reward_summary = self.reward_analyzer.end_episode()
                episode_summaries['reward_analysis'] = reward_summary
                
                # Log reward summary
                self.diagnostics_logger.log_episode_summary({
                    'category': 'reward_summary', 
                    **reward_summary
                })
            
            if self.observation_validator:
                obs_summary = self.observation_validator.end_episode()
                episode_summaries['observation_analysis'] = obs_summary
                
                # Log observation summary if issues detected
                if obs_summary.get('issues_detected', 0) > 0:
                    self.diagnostics_logger.log_episode_summary({
                        'category': 'observation_summary',
                        **obs_summary
                    })
            
            # Comprehensive episode summary
            if self.episode_count % self.episode_log_freq == 0:
                self._log_comprehensive_episode_summary(episode_summaries)
        
        except Exception as e:
            if self.verbose > 0:
                print(f"M10 episode end processing error: {e}")
    
    def _on_training_end(self) -> None:
        """Called when training ends - generate final summary."""
        try:
            # Compile final statistics
            final_stats = {
                'training_metadata': {
                    'total_timesteps': self.num_timesteps,
                    'total_episodes': self.episode_count,
                    'algorithm': self.algorithm,
                    'session_id': self.session_id,
                    'total_training_time': time.time() - self.callback_start_time,
                    'diagnostics_overhead': self.total_callback_time,
                    'diagnostics_overhead_pct': (self.total_callback_time / (time.time() - self.callback_start_time)) * 100
                },
                'action_statistics': self.action_tracker.get_overall_statistics() if self.action_tracker else None,
                'reward_statistics': self.reward_analyzer.get_overall_statistics() if self.reward_analyzer else None,
                'observation_statistics': self.observation_validator.get_overall_statistics() if self.observation_validator else None
            }
            
            # Save final summary
            summary_path = self.diagnostics_logger.save_final_summary(final_stats)
            
            # Generate final console summary
            self._log_final_console_summary(final_stats)
            
            # Close logger
            self.diagnostics_logger.close()
        
        except Exception as e:
            if self.verbose > 0:
                print(f"M10 training end processing error: {e}")
    
    def get_diagnostics_directory(self) -> str:
        """Get the organized diagnostics output directory."""
        return self.diagnostics_logger.get_output_directory()
    
    def get_diagnostics_files(self) -> Dict[str, str]:
        """Get paths to all diagnostic files."""
        return self.diagnostics_logger.get_diagnostics_files()
    
    def _extract_market_state(self, env) -> Dict[str, Any]:
        """Extract current market state information from environment."""
        try:
            market_state = {}
            
            # Get current market prices
            current_prices = env._get_current_market_prices() if hasattr(env, '_get_current_market_prices') else {}
            if current_prices and env.current_market in current_prices:
                yes_price, no_price = current_prices[env.current_market]
                market_state.update({
                    'yes_price': yes_price,
                    'no_price': no_price,
                    'spread': abs(yes_price - no_price) if yes_price and no_price else None
                })
            
            # Get market ticker
            market_state['market_ticker'] = getattr(env, 'current_market', 'unknown')
            
            # Get session info
            market_state['session_id'] = getattr(env.market_view, 'session_id', None) if hasattr(env, 'market_view') else None
            
            # Calculate total liquidity (if possible)
            # This would require access to current orderbook state
            market_state['total_liquidity'] = None  # TODO: Implement if needed
            
            return market_state
            
        except Exception:
            return {'error': 'failed_to_extract_market_state'}
    
    def _extract_portfolio_info(self, env) -> Dict[str, Any]:
        """Extract portfolio and position information from environment."""
        try:
            portfolio_info = {}
            
            if hasattr(env, 'order_manager') and env.order_manager:
                order_manager = env.order_manager
                
                # Get current market prices for portfolio valuation
                current_prices = env._get_current_market_prices() if hasattr(env, '_get_current_market_prices') else {}
                
                # Get portfolio values
                current_portfolio_value = order_manager.get_portfolio_value_cents(current_prices)
                cash_balance = order_manager.get_cash_balance_cents()
                position_info = order_manager.get_position_info()
                
                # Calculate portfolio change (requires previous value tracking)
                # For now, we'll track this within the reward analyzer
                portfolio_info = {
                    'current_portfolio_value': current_portfolio_value,
                    'cash_balance': cash_balance,
                    'position_info': position_info,
                    'portfolio_change': 0.0,  # Will be calculated in reward analyzer
                    'previous_portfolio_value': current_portfolio_value  # Placeholder
                }
            
            return portfolio_info
            
        except Exception:
            return {'error': 'failed_to_extract_portfolio_info'}
    
    def _log_console_summary(self, global_step: int) -> None:
        """Log concise console summary of diagnostic status."""
        try:
            # Collect component summaries
            summaries = []
            
            if self.action_tracker:
                summaries.append(self.action_tracker.get_action_console_summary())
            
            if self.reward_analyzer:
                summaries.append(self.reward_analyzer.get_reward_console_summary())
            
            if self.observation_validator:
                summaries.append(self.observation_validator.get_observation_console_summary())
            
            # Combine summaries
            if summaries:
                combined_summary = f"\n{'='*80}\n" + "M10 DIAGNOSTICS STATUS" + f"\nStep: {global_step:,}\n" + "\n".join(summaries) + f"\n{'='*80}"
                print(combined_summary)  # Direct console output
                
                # Also log to file
                self.diagnostics_logger.log_training_milestone(
                    {
                        'global_step': global_step,
                        'console_summaries': summaries,
                        'episodes_completed': self.episode_count
                    },
                    milestone_name=f"Console Summary - Step {global_step:,}"
                )
            
            self.last_console_summary = global_step
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Console summary error: {e}")
    
    def _log_comprehensive_episode_summary(self, episode_summaries: Dict[str, Any]) -> None:
        """Log comprehensive episode summary combining all diagnostic components."""
        
        # Create comprehensive summary
        comprehensive_summary = {
            'episode_milestone': self.episode_count,
            'global_step': self.num_timesteps,
            'diagnostic_summaries': episode_summaries,
            'training_progress': {
                'episodes_per_timestep': self.episode_count / max(self.num_timesteps, 1),
                'avg_episode_length': self.num_timesteps / max(self.episode_count, 1)
            }
        }
        
        self.diagnostics_logger.log_training_milestone(
            comprehensive_summary,
            milestone_name=f"Episode {self.episode_count} Summary"
        )
    
    def _log_final_console_summary(self, final_stats: Dict[str, Any]) -> None:
        """Log final comprehensive console summary."""
        print(f"\n{'='*80}")
        print("M10 DIAGNOSTICS - FINAL SUMMARY")
        print(f"{'='*80}")
        
        metadata = final_stats['training_metadata']
        print(f"Training completed: {metadata['total_timesteps']:,} timesteps, {metadata['total_episodes']} episodes")
        print(f"Algorithm: {metadata['algorithm']}, Session: {metadata['session_id']}")
        print(f"Diagnostics overhead: {metadata['diagnostics_overhead_pct']:.2f}% ({metadata['diagnostics_overhead']:.2f}s)")
        
        # Component summaries
        if final_stats.get('action_statistics'):
            action_stats = final_stats['action_statistics']
            if 'hold_dominance' in action_stats:
                hold_pct = action_stats['hold_dominance']['hold_percentage']
                activity = action_stats['hold_dominance']['trading_activity_level']
                print(f"\n🎯 ACTION ANALYSIS:")
                print(f"   HOLD dominance: {hold_pct:.1f}% | Trading activity: {activity.upper()}")
        
        if final_stats.get('reward_statistics'):
            reward_stats = final_stats['reward_statistics']
            if 'learning_signal_analysis' in reward_stats:
                signal_strength = reward_stats['learning_signal_analysis']['signal_strength']
                sparsity = reward_stats['reward_quality']['overall_sparsity_pct']
                print(f"\n💰 REWARD ANALYSIS:")
                print(f"   Learning signal: {signal_strength.upper()} | Sparsity: {sparsity:.1f}% zero rewards")
        
        if final_stats.get('observation_statistics'):
            obs_stats = final_stats['observation_statistics']
            if 'quality_assessment' in obs_stats:
                quality = obs_stats['quality_assessment']['overall_quality']
                error_rate = obs_stats['validation_rates']['overall_error_rate_pct']
                print(f"\n👁 OBSERVATION ANALYSIS:")
                print(f"   Quality: {quality.upper()} | Error rate: {error_rate:.2f}%")
        
        print(f"\n📁 Diagnostics saved to: {self.diagnostics_logger.get_output_directory()}")
        print(f"{'='*80}")
    
    def __del__(self):
        """Cleanup when callback is destroyed."""
        try:
            if hasattr(self, 'diagnostics_logger') and self.diagnostics_logger:
                self.diagnostics_logger.close()
        except:
            pass