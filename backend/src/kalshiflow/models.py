"""
Pydantic models for Kalshi WebSocket trade messages and internal data structures.
"""

from typing import Optional, List, Literal, Any, Dict
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator


class Trade(BaseModel):
    """A single trade event from Kalshi public trades stream."""
    market_ticker: str = Field(..., description="Market ticker symbol")
    yes_price: int = Field(..., description="YES price in cents (0-100)")
    no_price: int = Field(..., description="NO price in cents (0-100)")
    yes_price_dollars: float = Field(..., description="YES price in dollars (0.00-1.00)")
    no_price_dollars: float = Field(..., description="NO price in dollars (0.00-1.00)")
    count: int = Field(..., description="Number of shares traded")
    taker_side: Literal["yes", "no"] = Field(..., description="Side taken by the taker")
    ts: int = Field(..., description="Trade timestamp in milliseconds")
    
    @field_validator('yes_price', 'no_price')
    @classmethod
    def validate_price_range(cls, v):
        """Ensure prices are in valid range (0-100 cents)."""
        if not 0 <= v <= 100:
            raise ValueError(f"Price must be between 0 and 100 cents, got {v}")
        return v
    
    @field_validator('yes_price_dollars', 'no_price_dollars')
    @classmethod
    def validate_dollar_price_range(cls, v):
        """Ensure dollar prices are in valid range (0.00-1.00) and convert Decimal to float."""
        # Convert Decimal to float for JSON serialization
        if isinstance(v, Decimal):
            v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Dollar price must be between 0.00 and 1.00, got {v}")
        return v
        
    @field_validator('count')
    @classmethod
    def validate_positive_count(cls, v):
        """Ensure trade count is positive."""
        if v <= 0:
            raise ValueError(f"Trade count must be positive, got {v}")
        return v
    
    @property
    def timestamp(self) -> datetime:
        """Convert ts to datetime object."""
        return datetime.fromtimestamp(self.ts / 1000.0)
    
    @property
    def price_display(self) -> str:
        """Human-readable price for the taker side."""
        if self.taker_side == "yes":
            return f"${self.yes_price_dollars:.2f}"
        else:
            return f"${self.no_price_dollars:.2f}"
    
    def __lt__(self, other) -> bool:
        """Less than comparison based on timestamp."""
        if not isinstance(other, Trade):
            return NotImplemented
        return self.ts < other.ts
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison based on timestamp."""
        if not isinstance(other, Trade):
            return NotImplemented
        return self.ts <= other.ts
    
    def __gt__(self, other) -> bool:
        """Greater than comparison based on timestamp."""
        if not isinstance(other, Trade):
            return NotImplemented
        return self.ts > other.ts
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison based on timestamp."""
        if not isinstance(other, Trade):
            return NotImplemented
        return self.ts >= other.ts
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on all fields."""
        if not isinstance(other, Trade):
            return NotImplemented
        return (
            self.market_ticker == other.market_ticker and
            self.yes_price == other.yes_price and
            self.no_price == other.no_price and
            self.count == other.count and
            self.taker_side == other.taker_side and
            self.ts == other.ts
        )
    
    def __hash__(self) -> int:
        """Hash based on unique trade characteristics."""
        return hash((
            self.market_ticker,
            self.yes_price,
            self.no_price,
            self.count,
            self.taker_side,
            self.ts
        ))


class TradeMessage(BaseModel):
    """WebSocket message containing trade data from Kalshi."""
    msg: Dict[str, Any] = Field(..., description="Raw message data from Kalshi")
    
    def to_trade(self) -> Trade:
        """Convert raw Kalshi message to Trade object."""
        # Convert timestamp from seconds to milliseconds for consistent internal processing
        raw_timestamp = self.msg["ts"]
        timestamp_ms = raw_timestamp * 1000 if raw_timestamp < 2000000000 else raw_timestamp
        
        return Trade(
            market_ticker=self.msg["market_ticker"],
            yes_price=self.msg["yes_price"],
            no_price=self.msg["no_price"],
            yes_price_dollars=self.msg["yes_price"] / 100.0,
            no_price_dollars=self.msg["no_price"] / 100.0,
            count=self.msg["count"],
            taker_side=self.msg["taker_side"],
            ts=timestamp_ms
        )


class TickerState(BaseModel):
    """Aggregated state for a market ticker."""
    ticker: str = Field(..., description="Market ticker symbol")
    last_yes_price: int = Field(..., description="Last YES price in cents")
    last_no_price: int = Field(..., description="Last NO price in cents")
    last_trade_time: int = Field(..., description="Last trade timestamp in milliseconds")
    volume_window: int = Field(default=0, description="Trade volume in current window")
    trade_count_window: int = Field(default=0, description="Number of trades in current window")
    yes_flow: int = Field(default=0, description="YES-side volume in window")
    no_flow: int = Field(default=0, description="NO-side volume in window")
    price_points: List[Any] = Field(default_factory=list, description="Enhanced price history with volume and timestamp data for advanced visualizations")
    
    # Optional metadata fields (populated when available)
    title: Optional[str] = Field(None, description="Human-readable market title")
    category: Optional[str] = Field(None, description="Market category (Elections, Sports, etc)")
    liquidity_dollars: Optional[float] = Field(None, description="Total market liquidity in dollars")
    open_interest: Optional[int] = Field(None, description="Current open interest")
    latest_expiration_time: Optional[str] = Field(None, description="Market expiration time")
    
    @property
    def net_flow(self) -> int:
        """Net flow (yes_flow - no_flow)."""
        return self.yes_flow - self.no_flow
    
    @property
    def last_yes_price_dollars(self) -> float:
        """Last YES price in dollars."""
        return self.last_yes_price / 100.0
    
    @property
    def last_no_price_dollars(self) -> float:
        """Last NO price in dollars."""
        return self.last_no_price / 100.0
    
    @property
    def flow_direction(self) -> Literal["bullish", "bearish", "neutral"]:
        """Overall flow direction based on net flow."""
        if self.net_flow > 0:
            return "bullish"
        elif self.net_flow < 0:
            return "bearish"
        else:
            return "neutral"
    
    @property
    def last_price(self) -> float:
        """Last trade price in dollars (uses YES price as primary market price)."""
        return self.last_yes_price_dollars
    
    def dict(self, **kwargs) -> dict:
        """Override dict() to include computed properties."""
        data = super().model_dump(**kwargs)
        # Add computed properties
        data["net_flow"] = self.net_flow
        data["last_yes_price_dollars"] = self.last_yes_price_dollars
        data["last_no_price_dollars"] = self.last_no_price_dollars
        data["last_price"] = self.last_price
        data["flow_direction"] = self.flow_direction
        return data


class WebSocketMessage(BaseModel):
    """Base class for WebSocket messages sent to frontend."""
    type: str = Field(..., description="Message type identifier")
    data: Dict[str, Any] = Field(..., description="Message payload")


class SnapshotMessage(WebSocketMessage):
    """Initial snapshot message sent when frontend connects."""
    type: Literal["snapshot"] = "snapshot"
    data: Dict[str, Any] = Field(..., description="Snapshot data containing recent_trades and hot_markets")
    
    @field_validator('data')
    @classmethod
    def validate_snapshot_data(cls, v):
        """Ensure snapshot has required fields."""
        required_fields = ["recent_trades", "hot_markets"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Snapshot data must contain '{field}'")
        # analytics_data is optional for backward compatibility
        return v


class TradeUpdateMessage(WebSocketMessage):
    """Incremental trade update message sent to frontend."""
    type: Literal["trade"] = "trade"
    data: Dict[str, Any] = Field(..., description="Trade update data containing trade, ticker_state, and optionally current_minute_stats")
    
    @field_validator('data')
    @classmethod
    def validate_trade_data(cls, v):
        """Ensure trade update has required fields."""
        if "trade" not in v or "ticker_state" not in v:
            raise ValueError("Trade update data must contain 'trade' and 'ticker_state'")
        # current_minute_stats is optional
        return v


class AnalyticsDataMessage(WebSocketMessage):
    """Analytics time series data message sent to frontend with dual time modes."""
    type: Literal["analytics_data"] = "analytics_data"
    data: Dict[str, Any] = Field(..., description="Analytics data containing hour_minute_mode and day_hour_mode")
    
    @field_validator('data')
    @classmethod
    def validate_analytics_data(cls, v):
        """Ensure analytics data has required dual time mode fields."""
        if "hour_minute_mode" not in v:
            raise ValueError("Analytics data must contain 'hour_minute_mode'")
        if "day_hour_mode" not in v:
            raise ValueError("Analytics data must contain 'day_hour_mode'")
        
        # Validate structure of each mode
        for mode_name in ["hour_minute_mode", "day_hour_mode"]:
            mode_data = v[mode_name]
            if not isinstance(mode_data, dict):
                raise ValueError(f"{mode_name} must be a dictionary")
            if "time_series" not in mode_data:
                raise ValueError(f"{mode_name} must contain 'time_series'")
            if "summary_stats" not in mode_data:
                raise ValueError(f"{mode_name} must contain 'summary_stats'")
        
        return v


class AnalyticsIncrementalMessage(WebSocketMessage):
    """Lightweight incremental analytics message for real-time broadcasting.
    
    This message type sends only current period data and summary statistics,
    reducing bandwidth by ~95% compared to full analytics data.
    
    Data structure:
    - current_minute_data: Current minute bucket (65 bytes)
    - current_hour_data: Current hour bucket (65 bytes) 
    - hour_minute_mode_summary: Summary stats for hour/minute mode (~100 bytes)
    - day_hour_mode_summary: Summary stats for day/hour mode (~100 bytes)
    
    Total size: ~330 bytes vs 6KB for full analytics (95% reduction)
    """
    type: Literal["analytics_incremental"] = "analytics_incremental"
    data: Dict[str, Any] = Field(..., description="Incremental analytics data with current period buckets and summary stats")
    
    @field_validator('data')
    @classmethod
    def validate_incremental_data(cls, v):
        """Ensure incremental analytics data has required fields."""
        required_fields = [
            "current_minute_data",
            "current_hour_data", 
            "hour_minute_mode_summary",
            "day_hour_mode_summary"
        ]
        
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Incremental analytics data must contain '{field}'")
        
        # Validate current period data structure
        for period_field in ["current_minute_data", "current_hour_data"]:
            period_data = v[period_field]
            if not isinstance(period_data, dict):
                raise ValueError(f"{period_field} must be a dictionary")
            required_period_fields = ["timestamp", "volume_usd", "trade_count"]
            for req_field in required_period_fields:
                if req_field not in period_data:
                    raise ValueError(f"{period_field} must contain '{req_field}'")
        
        # Validate summary stats structure
        for summary_field in ["hour_minute_mode_summary", "day_hour_mode_summary"]:
            summary_data = v[summary_field]
            if not isinstance(summary_data, dict):
                raise ValueError(f"{summary_field} must be a dictionary")
            # Summary stats should have peak/total volume/trades and current period stats
        
        return v


class ConnectionStatus(BaseModel):
    """WebSocket connection status information."""
    connected: bool = Field(..., description="Whether connection is active")
    last_connected: Optional[datetime] = Field(None, description="Last successful connection time")
    reconnect_attempts: int = Field(default=0, description="Number of reconnection attempts")
    error_message: Optional[str] = Field(None, description="Last error message if any")