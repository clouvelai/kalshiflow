"""
M10 Observation Space Monitoring and Validation.

Validates observation quality, detects numerical issues, and monitors
feature distributions to ensure proper normalization.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class ObservationSnapshot:
    """Snapshot of observation state with validation metrics."""
    step: int
    global_step: int
    observation: np.ndarray
    
    # Validation metrics
    has_nan: bool
    has_inf: bool
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    
    # Feature-level analysis
    feature_stats: Dict[str, float]
    timestamp: float


class ObservationValidator:
    """
    Validates observation space quality and detects issues.
    
    Key validations:
    - NaN/Inf detection and tracking
    - Feature normalization verification
    - Feature distribution analysis
    - Observation stability over time
    - Numerical range validation
    """
    
    def __init__(self, 
                 expected_obs_dim: int = 52,
                 validation_freq: int = 100,
                 max_snapshots: int = 1000):
        """
        Initialize observation validator.
        
        Args:
            expected_obs_dim: Expected observation dimension
            validation_freq: Frequency of detailed validation (every N steps)
            max_snapshots: Maximum observation snapshots to store
        """
        self.expected_obs_dim = expected_obs_dim
        self.validation_freq = validation_freq
        self.max_snapshots = max_snapshots
        
        # Snapshot storage
        self.observation_snapshots: List[ObservationSnapshot] = []
        
        # Running validation statistics
        self.total_observations = 0
        self.nan_count = 0
        self.inf_count = 0
        self.dimension_errors = 0
        self.episode_count = 0
        
        # Feature distribution tracking
        self.feature_min_values = np.full(expected_obs_dim, np.inf)
        self.feature_max_values = np.full(expected_obs_dim, -np.inf)
        self.feature_mean_running = np.zeros(expected_obs_dim)
        self.feature_variance_running = np.zeros(expected_obs_dim)
        
        # Episode-level tracking
        self.episode_observation_counts: List[int] = []
        self.episode_issues: List[Dict[str, Any]] = []
        
        # Issue detection thresholds
        self.validation_thresholds = {
            'extreme_values': (-1000.0, 1000.0),
            'nan_tolerance': 0.01,  # 1% NaN tolerance
            'inf_tolerance': 0.001,  # 0.1% Inf tolerance
            'feature_range_warning': 100.0  # Warn if feature range > 100
        }
        
    def validate_observation(
        self,
        observation: np.ndarray,
        step: int,
        global_step: int,
        detailed_validation: bool = None
    ) -> Dict[str, Any]:
        """
        Validate a single observation and return validation results.
        
        Args:
            observation: Observation array to validate
            step: Episode step number
            global_step: Global training step
            detailed_validation: Force detailed validation (otherwise uses frequency)
            
        Returns:
            Validation results with any issues detected
        """
        self.total_observations += 1
        
        # Basic validation
        issues = []
        validation_result = {
            'step': step,
            'global_step': global_step,
            'observation_valid': True,
            'issues': issues
        }
        
        # Dimension check
        if observation.shape[0] != self.expected_obs_dim:
            issues.append(f"Wrong observation dimension: {observation.shape[0]}, expected {self.expected_obs_dim}")
            self.dimension_errors += 1
            validation_result['observation_valid'] = False
        
        # NaN check
        has_nan = np.any(np.isnan(observation))
        if has_nan:
            nan_count = np.sum(np.isnan(observation))
            issues.append(f"Contains {nan_count} NaN values")
            self.nan_count += 1
            validation_result['observation_valid'] = False
        
        # Infinity check
        has_inf = np.any(np.isinf(observation))
        if has_inf:
            inf_count = np.sum(np.isinf(observation))
            issues.append(f"Contains {inf_count} infinite values")
            self.inf_count += 1
            validation_result['observation_valid'] = False
        
        # Extreme value check
        min_val = float(np.min(observation)) if not (has_nan or has_inf) else np.nan
        max_val = float(np.max(observation)) if not (has_nan or has_inf) else np.nan
        
        extreme_min, extreme_max = self.validation_thresholds['extreme_values']
        if not np.isnan(min_val) and (min_val < extreme_min or max_val > extreme_max):
            issues.append(f"Extreme values detected: range [{min_val:.3f}, {max_val:.3f}]")
        
        # Update running statistics (if observation is valid)
        if not (has_nan or has_inf) and observation.shape[0] == self.expected_obs_dim:
            self._update_feature_statistics(observation)
        
        # Detailed validation (periodic or forced)
        if detailed_validation or (global_step % self.validation_freq == 0):
            validation_result.update(self._detailed_validation(observation, step, global_step))
        
        return validation_result
    
    def end_episode(self) -> Dict[str, Any]:
        """
        Mark episode end and return episode observation summary.
        
        Returns:
            Episode observation quality summary
        """
        self.episode_count += 1
        
        # Count observations this episode
        episode_obs_count = len([s for s in self.observation_snapshots 
                               if s.step >= 0])  # All recent snapshots are from current episode
        
        # Collect episode issues
        recent_snapshots = self.observation_snapshots[-episode_obs_count:] if episode_obs_count > 0 else []
        episode_issues = {
            'nan_observations': sum(1 for s in recent_snapshots if s.has_nan),
            'inf_observations': sum(1 for s in recent_snapshots if s.has_inf),
            'total_observations': episode_obs_count,
            'observation_quality_score': self._calculate_quality_score(recent_snapshots)
        }
        
        # Store episode statistics
        self.episode_observation_counts.append(episode_obs_count)
        self.episode_issues.append(episode_issues)
        
        # Episode summary
        summary = {
            'episode': self.episode_count,
            'observations_processed': episode_obs_count,
            'observation_quality': episode_issues,
            'issues_detected': episode_issues['nan_observations'] + episode_issues['inf_observations'],
            'quality_score': episode_issues['observation_quality_score'],
            'feature_distribution_health': self._assess_feature_distribution_health()
        }
        
        return summary
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get comprehensive observation validation statistics."""
        if self.total_observations == 0:
            return {'warning': 'No observations validated yet'}
        
        # Overall validation rates
        nan_rate = (self.nan_count / self.total_observations * 100)
        inf_rate = (self.inf_count / self.total_observations * 100)
        dimension_error_rate = (self.dimension_errors / self.total_observations * 100)
        
        # Feature distribution analysis
        feature_analysis = self._analyze_feature_distributions()
        
        # Recent episode trends
        recent_episodes = self.episode_issues[-10:] if self.episode_issues else []
        recent_quality_scores = [ep['observation_quality_score'] for ep in recent_episodes]
        
        # Overall quality assessment
        overall_quality = self._assess_overall_quality(nan_rate, inf_rate, dimension_error_rate)
        
        return {
            'total_observations_validated': self.total_observations,
            'episodes_completed': self.episode_count,
            
            # Validation rates
            'validation_rates': {
                'nan_rate_pct': nan_rate,
                'inf_rate_pct': inf_rate,
                'dimension_error_rate_pct': dimension_error_rate,
                'overall_error_rate_pct': nan_rate + inf_rate + dimension_error_rate
            },
            
            # Quality assessment
            'quality_assessment': overall_quality,
            
            # Feature distribution health
            'feature_distribution_analysis': feature_analysis,
            
            # Recent trends
            'recent_trends': {
                'recent_episode_quality_scores': recent_quality_scores,
                'avg_recent_quality': np.mean(recent_quality_scores) if recent_quality_scores else 0.0,
                'quality_trend': self._assess_quality_trend(recent_quality_scores)
            },
            
            # Detailed diagnostics
            'diagnostics': {
                'expected_observation_dim': self.expected_obs_dim,
                'validation_frequency': self.validation_freq,
                'validation_thresholds': self.validation_thresholds,
                'snapshots_stored': len(self.observation_snapshots)
            }
        }
    
    def get_observation_console_summary(self) -> str:
        """Get concise console summary of observation validation status."""
        stats = self.get_overall_statistics()
        
        if 'warning' in stats:
            return "⚠️  No observations validated yet"
        
        error_rate = stats['validation_rates']['overall_error_rate_pct']
        nan_rate = stats['validation_rates']['nan_rate_pct']
        inf_rate = stats['validation_rates']['inf_rate_pct']
        quality = stats['quality_assessment']['overall_quality']
        feature_health = stats['feature_distribution_analysis']['distribution_health']
        
        # Determine diagnostic status
        if error_rate > 5.0:
            status_emoji = "🛑"
            status_text = "CRITICAL: >5% observation errors"
        elif error_rate > 1.0:
            status_emoji = "⚠️"
            status_text = "WARNING: >1% observation errors"
        elif error_rate > 0.1:
            status_emoji = "⚡"
            status_text = "CAUTION: Some observation issues detected"
        else:
            status_emoji = "✅"
            status_text = "GOOD: Clean observation space"
        
        summary = f"""
{status_emoji} OBSERVATION VALIDATION
{status_text}

Validation Status:
  Error Rate: {error_rate:.2f}% | NaN: {nan_rate:.2f}% | Inf: {inf_rate:.2f}%
  Quality: {quality.upper()} | Feature Health: {feature_health.upper()}
  Observations: {stats['total_observations_validated']:,} | Episodes: {stats['episodes_completed']}

Feature Distribution:
  Dimension: {stats['diagnostics']['expected_observation_dim']} features
  Range Health: {feature_health.upper()}
""".strip()
        
        return summary
    
    def _detailed_validation(self, observation: np.ndarray, step: int, global_step: int) -> Dict[str, Any]:
        """Perform detailed validation and create observation snapshot."""
        
        # Calculate comprehensive statistics
        has_nan = np.any(np.isnan(observation))
        has_inf = np.any(np.isinf(observation))
        
        if not (has_nan or has_inf):
            min_val = float(np.min(observation))
            max_val = float(np.max(observation))
            mean_val = float(np.mean(observation))
            std_val = float(np.std(observation))
            
            # Feature-level analysis
            feature_stats = {
                'feature_ranges': [(float(observation[i] if not np.isnan(observation[i]) else 0)) 
                                 for i in range(min(len(observation), 10))],  # First 10 features
                'zero_features': int(np.sum(observation == 0.0)),
                'constant_features': int(np.sum(np.abs(observation - np.mean(observation)) < 1e-10)),
                'outlier_features': int(np.sum(np.abs(observation) > 3 * np.std(observation))) if np.std(observation) > 0 else 0
            }
        else:
            min_val = max_val = mean_val = std_val = np.nan
            feature_stats = {'validation_failed': True}
        
        # Create snapshot
        snapshot = ObservationSnapshot(
            step=step,
            global_step=global_step,
            observation=observation.copy(),
            has_nan=has_nan,
            has_inf=has_inf,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            feature_stats=feature_stats,
            timestamp=time.time()
        )
        
        # Store snapshot (with memory management)
        self.observation_snapshots.append(snapshot)
        if len(self.observation_snapshots) > self.max_snapshots:
            self.observation_snapshots.pop(0)
        
        return {
            'detailed_validation': True,
            'snapshot_created': True,
            'feature_analysis': feature_stats,
            'value_range': (min_val, max_val) if not np.isnan(min_val) else None
        }
    
    def _update_feature_statistics(self, observation: np.ndarray) -> None:
        """Update running feature-level statistics."""
        if observation.shape[0] != self.expected_obs_dim:
            return
        
        # Update min/max tracking
        self.feature_min_values = np.minimum(self.feature_min_values, observation)
        self.feature_max_values = np.maximum(self.feature_max_values, observation)
        
        # Update running mean and variance (Welford's online algorithm)
        n = self.total_observations
        delta = observation - self.feature_mean_running
        self.feature_mean_running += delta / n
        delta2 = observation - self.feature_mean_running
        self.feature_variance_running += delta * delta2
    
    def _analyze_feature_distributions(self) -> Dict[str, Any]:
        """Analyze feature distribution health across all features."""
        if self.total_observations < 2:
            return {'insufficient_data': True}
        
        # Calculate feature standard deviations
        feature_stds = np.sqrt(self.feature_variance_running / (self.total_observations - 1))
        feature_ranges = self.feature_max_values - self.feature_min_values
        
        # Identify problematic features
        constant_features = np.sum(feature_stds < 1e-10)
        extreme_range_features = np.sum(feature_ranges > self.validation_thresholds['feature_range_warning'])
        zero_variance_features = np.sum(feature_stds == 0.0)
        
        # Overall distribution health
        if constant_features > self.expected_obs_dim * 0.1:  # >10% constant features
            distribution_health = 'poor'
        elif extreme_range_features > self.expected_obs_dim * 0.2:  # >20% extreme ranges
            distribution_health = 'moderate'
        else:
            distribution_health = 'good'
        
        return {
            'distribution_health': distribution_health,
            'feature_statistics': {
                'constant_features': int(constant_features),
                'zero_variance_features': int(zero_variance_features),
                'extreme_range_features': int(extreme_range_features),
                'avg_feature_std': float(np.mean(feature_stds)),
                'avg_feature_range': float(np.mean(feature_ranges)),
                'feature_range_distribution': {
                    'min_range': float(np.min(feature_ranges)),
                    'max_range': float(np.max(feature_ranges)),
                    'std_range': float(np.std(feature_ranges))
                }
            }
        }
    
    def _calculate_quality_score(self, snapshots: List[ObservationSnapshot]) -> float:
        """Calculate observation quality score for a set of snapshots."""
        if not snapshots:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Penalize for NaN/Inf
        nan_penalty = sum(1 for s in snapshots if s.has_nan) / len(snapshots)
        inf_penalty = sum(1 for s in snapshots if s.has_inf) / len(snapshots)
        
        score -= (nan_penalty + inf_penalty)
        
        # Penalize for extreme values
        extreme_penalty = 0.0
        for snapshot in snapshots:
            if not (snapshot.has_nan or snapshot.has_inf):
                extreme_min, extreme_max = self.validation_thresholds['extreme_values']
                if snapshot.min_value < extreme_min or snapshot.max_value > extreme_max:
                    extreme_penalty += 0.1
        
        score -= extreme_penalty / len(snapshots)
        
        return max(0.0, score)
    
    def _assess_feature_distribution_health(self) -> str:
        """Assess current feature distribution health."""
        analysis = self._analyze_feature_distributions()
        
        if 'insufficient_data' in analysis:
            return 'unknown'
        
        return analysis['distribution_health']
    
    def _assess_overall_quality(self, nan_rate: float, inf_rate: float, dimension_error_rate: float) -> Dict[str, Any]:
        """Assess overall observation quality."""
        total_error_rate = nan_rate + inf_rate + dimension_error_rate
        
        if total_error_rate > 10.0:
            quality = 'critical'
        elif total_error_rate > 5.0:
            quality = 'poor'
        elif total_error_rate > 1.0:
            quality = 'moderate'
        elif total_error_rate > 0.1:
            quality = 'good'
        else:
            quality = 'excellent'
        
        return {
            'overall_quality': quality,
            'total_error_rate': total_error_rate,
            'quality_factors': {
                'nan_contribution': nan_rate,
                'inf_contribution': inf_rate,
                'dimension_contribution': dimension_error_rate
            }
        }
    
    def _assess_quality_trend(self, quality_scores: List[float]) -> str:
        """Assess quality trend from recent episodes."""
        if len(quality_scores) < 3:
            return 'insufficient_data'
        
        # Simple trend analysis
        recent_avg = np.mean(quality_scores[-3:])
        earlier_avg = np.mean(quality_scores[:-3]) if len(quality_scores) > 3 else np.mean(quality_scores[:3])
        
        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'degrading'
        else:
            return 'stable'