# Strategy plugins directory
# Each plugin file should register itself with StrategyRegistry

# Import plugins to trigger registration
from .hold import HoldStrategy
from .rlm_no import RLMNoStrategy
from .odmr import ODMRStrategy
from .agentic_research import AgenticResearchStrategy

__all__ = ['HoldStrategy', 'RLMNoStrategy', 'ODMRStrategy', 'AgenticResearchStrategy']
