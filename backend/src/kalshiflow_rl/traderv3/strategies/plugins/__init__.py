# Strategy plugins directory
# Each plugin file should register itself with StrategyRegistry

# Strategy plugins
from .deep_agent import DeepAgentStrategy

__all__ = [
    'DeepAgentStrategy',
]
