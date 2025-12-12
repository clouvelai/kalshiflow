"""
M10 Reward Signal Analysis.

Analyzes reward signal quality, sparsity, and components to understand
if the reward structure is providing sufficient learning gradient.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class RewardEvent:
    """Single reward event with detailed breakdown."""
    step: int
    global_step: int
    total_reward: float
    
    # Portfolio components
    portfolio_value_change: float
    previous_portfolio_value: float
    current_portfolio_value: float
    
    # Position context
    position_info: Dict[str, Any]
    cash_balance: float
    
    # Market context
    market_state: Dict[str, Any]
    timestamp: float
    
    # Action that led to this reward
    action: int
    action_name: str


class RewardAnalyzer:
    """
    Analyzes reward signal quality and provides insights into learning gradient.
    
    Key insights tracked:
    - Reward sparsity (% of zero rewards)
    - Reward magnitude distribution
    - Portfolio value progression
    - Reward-to-action correlation
    - Learning signal strength
    """
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize reward analyzer.
        
        Args:
            max_events: Maximum reward events to store
        """
        self.max_events = max_events
        
        # Event storage
        self.reward_events: List[RewardEvent] = []
        self.episode_rewards: List[float] = []
        
        # Running statistics
        self.total_rewards = 0.0
        self.episode_count = 0
        self.total_steps = 0
        
        # Episode-level tracking
        self.episode_reward_sums: List[float] = []
        self.episode_portfolio_changes: List[float] = []
        self.episode_zero_reward_counts: List[int] = []
        
        # Reward magnitude analysis
        self.reward_magnitude_bins = defaultdict(int)
        self.rewards_by_action = defaultdict(list)
        
        # Portfolio progression
        self.portfolio_values: List[float] = []
        self.cash_balances: List[float] = []
        
    def track_reward(
        self,
        reward: float,
        step: int, 
        global_step: int,
        portfolio_value_change: float,
        previous_portfolio_value: float,
        current_portfolio_value: float,
        position_info: Dict[str, Any],
        cash_balance: float,
        market_state: Dict[str, Any],
        action: int,
        action_name: str
    ) -> None:
        """
        Track a single reward event with full context.
        
        Args:
            reward: Total reward received
            step: Episode step number
            global_step: Global training step
            portfolio_value_change: Change in portfolio value
            previous_portfolio_value: Portfolio value before action
            current_portfolio_value: Portfolio value after action
            position_info: Current position information
            cash_balance: Current cash balance
            market_state: Current market conditions
            action: Action that led to this reward
            action_name: Human-readable action name
        """
        # Create reward event
        event = RewardEvent(
            step=step,
            global_step=global_step,
            total_reward=reward,
            portfolio_value_change=portfolio_value_change,
            previous_portfolio_value=previous_portfolio_value,
            current_portfolio_value=current_portfolio_value,
            position_info=position_info.copy(),
            cash_balance=cash_balance,
            market_state=market_state.copy(),
            action=action,
            action_name=action_name,
            timestamp=time.time()
        )
        
        # Store event (with memory management)
        self.reward_events.append(event)
        if len(self.reward_events) > self.max_events:
            self.reward_events.pop(0)
        
        # Update running statistics
        self.total_rewards += reward
        self.episode_rewards.append(reward)
        self.total_steps += 1
        
        # Track by action
        self.rewards_by_action[action].append(reward)
        
        # Magnitude binning for analysis
        self._bin_reward_magnitude(reward)
        
        # Portfolio progression
        self.portfolio_values.append(current_portfolio_value)
        self.cash_balances.append(cash_balance)
    
    def end_episode(self) -> Dict[str, Any]:
        """
        Mark episode end and return episode reward summary.
        
        Returns:
            Episode reward summary statistics
        """
        self.episode_count += 1
        
        # Calculate episode statistics
        total_episode_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)
        zero_rewards = sum(1 for r in self.episode_rewards if r == 0.0)
        
        # Portfolio progression for this episode
        if self.portfolio_values:
            episode_portfolio_start = self.portfolio_values[-episode_length] if len(self.portfolio_values) >= episode_length else None
            episode_portfolio_end = self.portfolio_values[-1] if self.portfolio_values else None
            episode_portfolio_change = (episode_portfolio_end - episode_portfolio_start) if (episode_portfolio_start is not None and episode_portfolio_end is not None) else 0.0
        else:
            episode_portfolio_change = 0.0
            episode_portfolio_start = None
            episode_portfolio_end = None
        
        # Store episode statistics
        self.episode_reward_sums.append(total_episode_reward)
        self.episode_portfolio_changes.append(episode_portfolio_change)
        self.episode_zero_reward_counts.append(zero_rewards)
        
        # Reward quality analysis
        non_zero_rewards = [r for r in self.episode_rewards if r != 0.0]
        reward_sparsity = (zero_rewards / episode_length * 100) if episode_length > 0 else 100.0
        
        # Episode summary
        summary = {
            'episode': self.episode_count,
            'episode_length': episode_length,
            'total_reward': total_episode_reward,
            'average_reward': total_episode_reward / episode_length if episode_length > 0 else 0.0,
            'portfolio_change_cents': episode_portfolio_change,
            'portfolio_start_cents': episode_portfolio_start,
            'portfolio_end_cents': episode_portfolio_end,
            
            # Reward quality metrics
            'reward_sparsity_pct': reward_sparsity,
            'zero_rewards': zero_rewards,
            'non_zero_rewards': len(non_zero_rewards),
            'reward_magnitude': {
                'min': min(self.episode_rewards) if self.episode_rewards else 0.0,
                'max': max(self.episode_rewards) if self.episode_rewards else 0.0,
                'std': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
                'mean_non_zero': np.mean(non_zero_rewards) if non_zero_rewards else 0.0
            },
            
            # Learning signal analysis
            'learning_signal': {
                'reward_variance': np.var(self.episode_rewards) if self.episode_rewards else 0.0,
                'signal_strength': 'strong' if reward_sparsity < 50 else 
                                'moderate' if reward_sparsity < 80 else 'weak',
                'gradient_quality': 'good' if len(non_zero_rewards) > episode_length * 0.2 else 'poor'
            }
        }
        
        # Reset episode tracking
        self.episode_rewards = []
        
        return summary
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics across all episodes."""
        if not self.reward_events:
            return {'warning': 'No rewards tracked yet'}
        
        # Overall reward distribution
        all_rewards = [event.total_reward for event in self.reward_events]
        portfolio_changes = [event.portfolio_value_change for event in self.reward_events]
        
        # Sparsity analysis
        zero_rewards = sum(1 for r in all_rewards if r == 0.0)
        total_rewards_tracked = len(all_rewards)
        reward_sparsity = (zero_rewards / total_rewards_tracked * 100) if total_rewards_tracked > 0 else 100.0
        
        # Magnitude analysis
        non_zero_rewards = [r for r in all_rewards if r != 0.0]
        
        # Action-reward correlation
        action_reward_analysis = {}
        for action, rewards in self.rewards_by_action.items():
            if rewards:
                action_reward_analysis[action] = {
                    'count': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'total_reward': sum(rewards),
                    'non_zero_rewards': sum(1 for r in rewards if r != 0.0),
                    'sparsity_pct': (sum(1 for r in rewards if r == 0.0) / len(rewards) * 100)
                }
        
        # Episode trends
        recent_episodes = self.episode_reward_sums[-10:] if self.episode_reward_sums else []
        recent_portfolio_changes = self.episode_portfolio_changes[-10:] if self.episode_portfolio_changes else []
        recent_sparsity = self.episode_zero_reward_counts[-10:] if self.episode_zero_reward_counts else []
        
        # Portfolio progression analysis
        portfolio_analysis = self._analyze_portfolio_progression()
        
        return {
            'total_rewards_tracked': total_rewards_tracked,
            'episodes_completed': self.episode_count,
            
            # Overall reward quality
            'reward_quality': {
                'overall_sparsity_pct': reward_sparsity,
                'zero_rewards': zero_rewards,
                'non_zero_rewards': len(non_zero_rewards),
                'avg_reward': np.mean(all_rewards),
                'total_reward': sum(all_rewards),
                'reward_range': (min(all_rewards), max(all_rewards)) if all_rewards else (0, 0),
                'reward_std': np.std(all_rewards) if all_rewards else 0.0
            },
            
            # Critical diagnostic: Learning signal strength
            'learning_signal_analysis': {
                'signal_strength': 'strong' if reward_sparsity < 30 else
                                 'moderate' if reward_sparsity < 70 else
                                 'weak' if reward_sparsity < 95 else 'critical',
                'gradient_availability': len(non_zero_rewards) / total_rewards_tracked if total_rewards_tracked > 0 else 0.0,
                'reward_variance': np.var(all_rewards) if all_rewards else 0.0,
                'non_zero_reward_magnitude': {
                    'mean': np.mean(non_zero_rewards) if non_zero_rewards else 0.0,
                    'std': np.std(non_zero_rewards) if non_zero_rewards else 0.0,
                    'min': min(non_zero_rewards) if non_zero_rewards else 0.0,
                    'max': max(non_zero_rewards) if non_zero_rewards else 0.0
                }
            },
            
            # Action-reward correlation
            'action_reward_correlation': action_reward_analysis,
            
            # Portfolio progression
            'portfolio_analysis': portfolio_analysis,
            
            # Recent trends
            'recent_trends': {
                'recent_episode_rewards': recent_episodes,
                'recent_portfolio_changes': recent_portfolio_changes,
                'avg_recent_reward': np.mean(recent_episodes) if recent_episodes else 0.0,
                'avg_recent_portfolio_change': np.mean(recent_portfolio_changes) if recent_portfolio_changes else 0.0
            },
            
            # Magnitude distribution
            'magnitude_distribution': dict(self.reward_magnitude_bins)
        }
    
    def get_reward_console_summary(self) -> str:
        """Get concise console summary of reward signal quality."""
        stats = self.get_overall_statistics()
        
        if 'warning' in stats:
            return "⚠️  No rewards tracked yet"
        
        sparsity = stats['reward_quality']['overall_sparsity_pct']
        signal_strength = stats['learning_signal_analysis']['signal_strength']
        avg_reward = stats['reward_quality']['avg_reward']
        total_reward = stats['reward_quality']['total_reward']
        non_zero_count = stats['reward_quality']['non_zero_rewards']
        
        # Portfolio progression
        portfolio_stats = stats['portfolio_analysis']
        portfolio_trend = portfolio_stats.get('progression_trend', 'unknown')
        portfolio_volatility = portfolio_stats.get('volatility', 0.0)
        
        # Determine diagnostic status
        if sparsity > 95:
            status_emoji = "🛑"
            status_text = "CRITICAL: >95% zero rewards - No learning signal"
        elif sparsity > 80:
            status_emoji = "⚠️"
            status_text = "WARNING: >80% zero rewards - Weak learning signal"
        elif sparsity > 50:
            status_emoji = "⚡"
            status_text = "MODERATE: >50% zero rewards - Limited learning signal"
        else:
            status_emoji = "✅"
            status_text = "GOOD: <50% zero rewards - Strong learning signal"
        
        summary = f"""
{status_emoji} REWARD SIGNAL ANALYSIS
{status_text}

Reward Quality:
  Sparsity: {sparsity:.1f}% zero rewards | Non-zero: {non_zero_count:,}
  Signal Strength: {signal_strength.upper()}
  Avg Reward: {avg_reward:.2f} | Total: {total_reward:.2f}

Portfolio Progression:
  Trend: {portfolio_trend.upper()}
  Volatility: {portfolio_volatility:.2f} cents
  Episodes: {stats['episodes_completed']}
""".strip()
        
        return summary
    
    def _bin_reward_magnitude(self, reward: float) -> None:
        """Bin reward magnitudes for distribution analysis."""
        if reward == 0.0:
            self.reward_magnitude_bins['zero'] += 1
        elif abs(reward) < 1.0:
            self.reward_magnitude_bins['tiny_<1cent'] += 1
        elif abs(reward) < 10.0:
            self.reward_magnitude_bins['small_1-10cents'] += 1
        elif abs(reward) < 100.0:
            self.reward_magnitude_bins['medium_10-100cents'] += 1
        elif abs(reward) < 1000.0:
            self.reward_magnitude_bins['large_100-1000cents'] += 1
        else:
            self.reward_magnitude_bins['huge_>1000cents'] += 1
    
    def _analyze_portfolio_progression(self) -> Dict[str, Any]:
        """Analyze portfolio value progression over time."""
        if len(self.portfolio_values) < 2:
            return {'insufficient_data': True}
        
        # Calculate portfolio changes
        portfolio_changes = np.diff(self.portfolio_values)
        
        # Trend analysis
        if len(self.portfolio_values) > 10:
            early_values = self.portfolio_values[:len(self.portfolio_values)//3]
            late_values = self.portfolio_values[-len(self.portfolio_values)//3:]
            
            early_mean = np.mean(early_values)
            late_mean = np.mean(late_values)
            
            if late_mean > early_mean * 1.05:
                trend = 'increasing'
            elif late_mean < early_mean * 0.95:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Volatility analysis
        portfolio_volatility = np.std(portfolio_changes) if len(portfolio_changes) > 0 else 0.0
        
        return {
            'progression_trend': trend,
            'volatility': portfolio_volatility,
            'total_portfolio_change': self.portfolio_values[-1] - self.portfolio_values[0] if self.portfolio_values else 0.0,
            'portfolio_range': (min(self.portfolio_values), max(self.portfolio_values)) if self.portfolio_values else (0, 0),
            'positive_changes': sum(1 for change in portfolio_changes if change > 0),
            'negative_changes': sum(1 for change in portfolio_changes if change < 0),
            'zero_changes': sum(1 for change in portfolio_changes if change == 0)
        }