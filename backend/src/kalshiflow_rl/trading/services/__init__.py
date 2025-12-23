"""
Trading Services Package for TRADER 2.0

Extracted services from the monolithic OrderManager:
- OrderService: Order lifecycle management  
- PositionTracker: Position tracking and P&L calculation
- StateSync: API state synchronization
- StatusLogger: State machine and service status tracking
- FillProcessor: Async fill queue processing
- WebSocketManager: WebSocket connection management
- ApiClient: Enhanced API client wrapper
"""

from .order_service import OrderService, OrderInfo, OrderStatus, OrderSide, ContractSide
from .position_tracker import PositionTracker, Position, FillInfo
from .state_sync import StateSync
from .status_logger import StatusLogger, StatusEntry, ServiceStatus, TraderState
from .fill_processor import FillProcessor, FillEvent
from .websocket_manager import WebSocketManager, ConnectionState, WebSocketConfig
from .api_client import ApiClient, ApiCall

__all__ = [
    # Order management
    "OrderService",
    "OrderInfo", 
    "OrderStatus",
    "OrderSide",
    "ContractSide",
    
    # Position tracking
    "PositionTracker",
    "Position",
    "FillInfo",
    
    # State synchronization
    "StateSync",
    
    # Status logging
    "StatusLogger",
    "StatusEntry",
    "ServiceStatus", 
    "TraderState",
    
    # Fill processing
    "FillProcessor",
    "FillEvent",
    
    # WebSocket management
    "WebSocketManager",
    "ConnectionState",
    "WebSocketConfig",
    
    # API client
    "ApiClient",
    "ApiCall"
]