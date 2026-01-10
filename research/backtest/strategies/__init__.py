"""
Strategy implementations for point-in-time backtesting.

All strategies must implement the Strategy protocol defined in protocol.py.

Available Strategies:
    - RLMNoStrategy: Reverse Line Movement - bet NO when YES dominates but price drops
    - RLMFilteredStrategy: RLM_NO with stricter filters (min_price_drop=10, price_bucket=30-70)
    - SLateTimingStrategy: Late-Arriving Large Money - bet NO when large trades arrive late
    - RLMVolumeWeightedStrategy: RLM using volume-weighted flow instead of trade count
    - MomentumFlipStrategy: Buy YES when price drops but selling pressure eases
    - LateWhaleStrategy: Follow large trades that go against crowd consensus
    - create_rlm_strategy: Factory function for custom RLM parameters
    - create_rlm_filtered_strategy: Factory function for custom RLM FILTERED parameters
    - create_slate_strategy: Factory function for custom S-LATE parameters

Usage:
    from research.backtest.strategies import RLMNoStrategy, RLMFilteredStrategy, SLateTimingStrategy

    # Default parameters (matches production)
    rlm = RLMNoStrategy()
    rlm_filtered = RLMFilteredStrategy()
    slate = SLateTimingStrategy()

    # Custom parameters for sensitivity analysis
    rlm = create_rlm_strategy(yes_threshold=0.80, min_price_drop=10)
    rlm_filtered = create_rlm_filtered_strategy(min_price_drop=15, max_no_price=60)
    slate = create_slate_strategy(min_trades=100, large_threshold_dollars=75)
"""

from .protocol import Strategy
from .rlm_no import RLMNoStrategy, create_rlm_strategy
from .rlm_filtered import RLMFilteredStrategy, create_rlm_filtered_strategy
from .s_late_timing import SLateTimingStrategy, create_slate_strategy
from .rlm_volume_weighted import RLMVolumeWeightedStrategy, create_rlm_volume_weighted_strategy
from .momentum_flip import MomentumFlipStrategy, create_momentum_flip_strategy
from .late_whale import LateWhaleStrategy, create_late_whale_strategy
from .fade_contrarian_whale import FadeContrarianWhaleStrategy, create_fade_contrarian_whale_strategy
from .aligned_whale import AlignedWhaleStrategy, create_aligned_whale_strategy
from .whale_cluster import WhaleClusterStrategy, create_whale_cluster_strategy
from .whale_cluster_no import WhaleClusterNoStrategy, create_whale_cluster_no_strategy
from .dip_buyer import DipBuyerStrategy, create_dip_buyer_strategy
from .spike_fader import SpikeFaderStrategy, create_spike_fader_strategy

__all__ = [
    'Strategy',
    'RLMNoStrategy',
    'create_rlm_strategy',
    'RLMFilteredStrategy',
    'create_rlm_filtered_strategy',
    'SLateTimingStrategy',
    'create_slate_strategy',
    'RLMVolumeWeightedStrategy',
    'create_rlm_volume_weighted_strategy',
    'MomentumFlipStrategy',
    'create_momentum_flip_strategy',
    'LateWhaleStrategy',
    'create_late_whale_strategy',
    'FadeContrarianWhaleStrategy',
    'create_fade_contrarian_whale_strategy',
    'AlignedWhaleStrategy',
    'create_aligned_whale_strategy',
    'WhaleClusterStrategy',
    'create_whale_cluster_strategy',
    'WhaleClusterNoStrategy',
    'create_whale_cluster_no_strategy',
    'DipBuyerStrategy',
    'create_dip_buyer_strategy',
    'SpikeFaderStrategy',
    'create_spike_fader_strategy',
]
