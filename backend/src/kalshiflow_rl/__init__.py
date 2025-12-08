"""
Kalshi Flow RL Trading Subsystem

A reinforcement learning trading subsystem that provides:
- Non-blocking orderbook data ingestion from Kalshi WebSocket
- Historical data preprocessing for RL training environments  
- Paper trading simulation for inference-only actor loops
- Model registry and hot-reload capabilities

This package works alongside the existing Kalshi Flowboard application,
sharing database and authentication infrastructure while maintaining
strict pipeline isolation between training and inference.
"""

__version__ = "0.1.0"
__author__ = "Kalshi Flow RL Team"

from .config import config, logger

__all__ = [
    "config",
    "logger",
]