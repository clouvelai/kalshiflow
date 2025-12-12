"""
M10 Action Distribution Tracking.

Tracks action selections, exploration patterns, and agent behavior to diagnose
why the agent exhibits HOLD-only behavior.
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class ActionEvent:
    """Single action event with context."""
    step: int
    global_step: int
    action: int
    action_name: str
    market_state: Dict[str, Any]
    timestamp: float
    
    # Optional action probabilities if available
    action_probs: Optional[List[float]] = None
    exploration: bool = False  # Was this exploratory vs greedy


class ActionTracker:
    """
    Tracks action distribution patterns and exploration behavior.
    
    Key insights tracked:
    - Action distribution over time (are we stuck in HOLD?)
    - Exploration vs exploitation patterns  
    - Market conditions when non-HOLD actions are chosen
    - Action entropy and diversity metrics
    """
    
    # Action mapping for Kalshi limit order space
    ACTION_NAMES = {
        0: "HOLD",
        1: "BUY_YES_LIMIT", 
        2: "SELL_YES_LIMIT",
        3: "BUY_NO_LIMIT",
        4: "SELL_NO_LIMIT"
    }
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize action tracker.
        
        Args:
            max_events: Maximum action events to store (prevents memory issues)
        """
        self.max_events = max_events
        
        # Event storage
        self.action_events: List[ActionEvent] = []
        self.episode_actions: List[int] = []
        
        # Running statistics  
        self.action_counts = Counter()
        self.episode_count = 0
        self.total_steps = 0
        
        # Episode-level tracking
        self.episode_action_distributions: List[Dict[int, int]] = []
        self.episode_entropies: List[float] = []
        
        # Market condition analysis
        self.actions_by_spread: defaultdict = defaultdict(list)
        self.actions_by_liquidity: defaultdict = defaultdict(list)
        
    def track_action(
        self, 
        action: int,
        step: int,
        global_step: int,
        market_state: Dict[str, Any],
        action_probs: Optional[List[float]] = None,
        exploration: bool = False
    ) -> None:
        """
        Track a single action event.
        
        Args:
            action: Integer action (0-4)
            step: Episode step number
            global_step: Global training step
            market_state: Current market conditions (spread, liquidity, etc.)
            action_probs: Action probabilities from policy if available
            exploration: Whether this was an exploratory action
        """
        # Create action event
        event = ActionEvent(
            step=step,
            global_step=global_step,
            action=action,
            action_name=self.ACTION_NAMES.get(action, f"UNKNOWN_{action}"),
            market_state=market_state.copy(),
            timestamp=time.time(),
            action_probs=action_probs.copy() if action_probs else None,
            exploration=exploration
        )
        
        # Store event (with memory management)
        self.action_events.append(event)
        if len(self.action_events) > self.max_events:
            self.action_events.pop(0)
        
        # Update running statistics
        self.action_counts[action] += 1
        self.episode_actions.append(action)
        self.total_steps += 1
        
        # Market condition analysis
        spread = market_state.get('spread', None)
        if spread is not None:
            # Bin spreads for analysis
            spread_bin = self._bin_spread(spread)
            self.actions_by_spread[spread_bin].append(action)
        
        liquidity = market_state.get('total_liquidity', None)
        if liquidity is not None:
            # Bin liquidity for analysis
            liquidity_bin = self._bin_liquidity(liquidity) 
            self.actions_by_liquidity[liquidity_bin].append(action)
    
    def end_episode(self) -> Dict[str, Any]:
        """
        Mark episode end and return episode summary.
        
        Returns:
            Episode action summary statistics
        """
        self.episode_count += 1
        
        # Calculate episode action distribution
        episode_dist = Counter(self.episode_actions)
        total_episode_actions = len(self.episode_actions)
        
        # Calculate action entropy for this episode
        if total_episode_actions > 0:
            action_probs = [episode_dist[i] / total_episode_actions for i in range(5)]
            entropy = self._calculate_entropy(action_probs)
        else:
            entropy = 0.0
        
        # Store episode statistics
        self.episode_action_distributions.append(dict(episode_dist))
        self.episode_entropies.append(entropy)
        
        # Episode summary
        summary = {
            'episode': self.episode_count,
            'episode_length': total_episode_actions,
            'action_distribution': dict(episode_dist),
            'action_percentages': {
                self.ACTION_NAMES[i]: (episode_dist[i] / total_episode_actions * 100) if total_episode_actions > 0 else 0
                for i in range(5)
            },
            'action_entropy': entropy,
            'exploration_actions': sum(1 for event in self.action_events[-total_episode_actions:] if event.exploration),
            'non_hold_actions': sum(episode_dist[i] for i in range(1, 5)),
            'hold_percentage': (episode_dist[0] / total_episode_actions * 100) if total_episode_actions > 0 else 0
        }
        
        # Reset episode tracking
        self.episode_actions = []
        
        return summary
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get comprehensive action statistics across all episodes."""
        total_actions = sum(self.action_counts.values())
        
        if total_actions == 0:
            return {'warning': 'No actions tracked yet'}
        
        # Overall distribution
        overall_distribution = {
            self.ACTION_NAMES[i]: self.action_counts[i]
            for i in range(5)
        }
        
        overall_percentages = {
            self.ACTION_NAMES[i]: (self.action_counts[i] / total_actions * 100)
            for i in range(5) 
        }
        
        # Calculate overall entropy
        action_probs = [self.action_counts[i] / total_actions for i in range(5)]
        overall_entropy = self._calculate_entropy(action_probs)
        
        # Episode-level analysis
        recent_episodes = self.episode_action_distributions[-10:] if self.episode_action_distributions else []
        recent_entropies = self.episode_entropies[-10:] if self.episode_entropies else []
        
        # Trend analysis: Are we getting more diverse over time?
        if len(self.episode_entropies) >= 5:
            early_entropy = np.mean(self.episode_entropies[:5])
            recent_entropy = np.mean(self.episode_entropies[-5:])
            entropy_trend = 'increasing' if recent_entropy > early_entropy else 'decreasing'
        else:
            entropy_trend = 'insufficient_data'
        
        # Market condition analysis
        market_analysis = self._analyze_market_conditions()
        
        return {
            'total_actions_tracked': total_actions,
            'episodes_completed': self.episode_count,
            'overall_distribution': overall_distribution,
            'overall_percentages': overall_percentages,
            'overall_entropy': overall_entropy,
            
            # Critical diagnostic: HOLD dominance
            'hold_dominance': {
                'hold_percentage': overall_percentages['HOLD'],
                'is_hold_dominant': overall_percentages['HOLD'] > 90.0,
                'non_hold_actions': sum(self.action_counts[i] for i in range(1, 5)),
                'trading_activity_level': 'none' if overall_percentages['HOLD'] > 99 else 
                                        'minimal' if overall_percentages['HOLD'] > 95 else
                                        'low' if overall_percentages['HOLD'] > 80 else 'normal'
            },
            
            # Exploration analysis
            'exploration_analysis': {
                'avg_episode_entropy': np.mean(recent_entropies) if recent_entropies else 0.0,
                'entropy_trend': entropy_trend,
                'max_entropy_possible': np.log(5),  # log(5 actions)
                'exploration_ratio': (np.mean(recent_entropies) / np.log(5)) if recent_entropies else 0.0
            },
            
            # Market condition insights
            'market_condition_analysis': market_analysis,
            
            # Recent episode trends
            'recent_episodes': {
                'last_10_episodes': recent_episodes,
                'avg_hold_pct_recent': np.mean([ep.get(0, 0) for ep in recent_episodes]) / 
                                     np.mean([sum(ep.values()) for ep in recent_episodes]) * 100 
                                     if recent_episodes else 100.0
            }
        }
    
    def get_action_console_summary(self) -> str:
        """Get concise console summary of action patterns."""
        stats = self.get_overall_statistics()
        
        if 'warning' in stats:
            return "⚠️  No actions tracked yet"
        
        hold_pct = stats['overall_percentages']['HOLD']
        activity_level = stats['hold_dominance']['trading_activity_level']
        recent_entropy = stats['exploration_analysis']['avg_episode_entropy']
        max_entropy = stats['exploration_analysis']['max_entropy_possible']
        
        # Determine diagnostic status
        if hold_pct > 99:
            status_emoji = "🛑"
            status_text = "CRITICAL: 99%+ HOLD - No trading activity"
        elif hold_pct > 95:
            status_emoji = "⚠️"
            status_text = "WARNING: 95%+ HOLD - Minimal trading"
        elif hold_pct > 80:
            status_emoji = "⚡"
            status_text = "LOW: 80%+ HOLD - Limited trading"
        else:
            status_emoji = "✅"
            status_text = "NORMAL: Balanced action distribution"
        
        summary = f"""
{status_emoji} ACTION DISTRIBUTION SUMMARY
{status_text}

Overall Distribution:
  HOLD: {hold_pct:.1f}% | Trading: {100-hold_pct:.1f}%
  BUY_YES: {stats['overall_percentages']['BUY_YES_LIMIT']:.1f}% | SELL_YES: {stats['overall_percentages']['SELL_YES_LIMIT']:.1f}%
  BUY_NO: {stats['overall_percentages']['BUY_NO_LIMIT']:.1f}% | SELL_NO: {stats['overall_percentages']['SELL_NO_LIMIT']:.1f}%

Exploration Metrics:
  Action Entropy: {recent_entropy:.3f}/{max_entropy:.3f} ({recent_entropy/max_entropy*100:.1f}% of max)
  Activity Level: {activity_level.upper()}
  Episodes Tracked: {stats['episodes_completed']}
""".strip()
        
        return summary
    
    def _bin_spread(self, spread: float) -> str:
        """Bin spread values for analysis."""
        if spread < 0.01:
            return "tight_<1cent"
        elif spread < 0.05:
            return "narrow_1-5cents"
        elif spread < 0.10:
            return "medium_5-10cents"
        elif spread < 0.20:
            return "wide_10-20cents"
        else:
            return "very_wide_>20cents"
    
    def _bin_liquidity(self, liquidity: float) -> str:
        """Bin liquidity values for analysis."""
        if liquidity < 100:
            return "low_<100"
        elif liquidity < 1000:
            return "medium_100-1000"
        elif liquidity < 5000:
            return "high_1k-5k"
        else:
            return "very_high_>5k"
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy of action probabilities."""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p)
        return entropy
    
    def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze how market conditions affect action selection."""
        analysis = {
            'spread_analysis': {},
            'liquidity_analysis': {},
            'insights': []
        }
        
        # Spread analysis
        for spread_bin, actions in self.actions_by_spread.items():
            if len(actions) > 0:
                action_dist = Counter(actions)
                total = len(actions)
                hold_pct = (action_dist[0] / total * 100) if total > 0 else 0
                
                analysis['spread_analysis'][spread_bin] = {
                    'total_actions': total,
                    'hold_percentage': hold_pct,
                    'trading_actions': sum(action_dist[i] for i in range(1, 5))
                }
        
        # Liquidity analysis  
        for liquidity_bin, actions in self.actions_by_liquidity.items():
            if len(actions) > 0:
                action_dist = Counter(actions)
                total = len(actions)
                hold_pct = (action_dist[0] / total * 100) if total > 0 else 0
                
                analysis['liquidity_analysis'][liquidity_bin] = {
                    'total_actions': total,
                    'hold_percentage': hold_pct,
                    'trading_actions': sum(action_dist[i] for i in range(1, 5))
                }
        
        # Generate insights
        if self.actions_by_spread:
            min_hold_spread = min(
                self.actions_by_spread.keys(),
                key=lambda x: analysis['spread_analysis'].get(x, {}).get('hold_percentage', 100)
            )
            if analysis['spread_analysis'].get(min_hold_spread, {}).get('hold_percentage', 100) < 90:
                analysis['insights'].append(f"Most trading occurs in {min_hold_spread} spread conditions")
            else:
                analysis['insights'].append("High HOLD percentage across all spread conditions")
        
        return analysis