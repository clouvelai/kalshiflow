"""Pydantic v2 response models for Kalshi API.

Typed models for REST responses and WebSocket messages. Uses
model_dump() to produce dict output compatible with existing code.
All fields use snake_case matching Kalshi's API convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# REST response models
# ---------------------------------------------------------------------------

class Balance(BaseModel):
    """GET /portfolio/balance."""
    balance: int  # cents
    portfolio_value: int  # cents


class OrderbookLevel(BaseModel):
    """Single price level: [price_cents, quantity]."""
    price: int
    quantity: int


class Orderbook(BaseModel):
    """GET /markets/{ticker}/orderbook."""
    yes: List[List[int]] = Field(default_factory=list)  # [[price, qty], ...]
    no: List[List[int]] = Field(default_factory=list)


class Market(BaseModel):
    """Market within an event."""
    ticker: str
    event_ticker: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    status: str = ""
    result: str = ""
    rules_primary: str = ""
    yes_bid: int = 0
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    last_price: int = 0
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    close_time: Optional[str] = None
    expiration_time: Optional[str] = None

    model_config = {"extra": "allow"}


class Event(BaseModel):
    """GET /events/{event_ticker}."""
    event_ticker: str
    title: str = ""
    subtitle: str = ""
    category: str = ""
    mutually_exclusive: bool = True
    series_ticker: str = ""
    markets: List[Market] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class Order(BaseModel):
    """Order from GET /portfolio/orders or POST /portfolio/orders."""
    order_id: str = ""
    ticker: str = ""
    action: str = ""  # buy / sell
    side: str = ""    # yes / no
    type: str = "limit"
    status: str = ""
    price: int = 0
    yes_price: int = 0
    no_price: int = 0
    count: int = 0
    remaining_count: int = 0
    created_time: Optional[str] = None
    expiration_time: Optional[str] = None
    order_group_id: str = ""

    model_config = {"extra": "allow"}


class OrderResponse(BaseModel):
    """POST /portfolio/orders response wrapper."""
    order: Order = Field(default_factory=Order)


class Position(BaseModel):
    """Single position from GET /portfolio/positions."""
    ticker: str = ""
    event_ticker: str = ""
    position: int = 0  # positive = YES, negative = NO
    total_traded: int = 0  # cost in cents
    realized_pnl: int = 0
    fees_paid: int = 0
    market_exposure: int = 0
    resting_orders_count: int = 0

    model_config = {"extra": "allow"}


class Fill(BaseModel):
    """Single fill from GET /portfolio/fills."""
    trade_id: str = ""
    order_id: str = ""
    ticker: str = ""
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    no_price: int = 0
    is_taker: bool = False
    created_time: Optional[str] = None

    model_config = {"extra": "allow"}


class Settlement(BaseModel):
    """Single settlement from GET /portfolio/settlements."""
    ticker: str = ""
    market_result: str = ""
    revenue: int = 0
    payout: int = 0
    settled_time: Optional[str] = None

    model_config = {"extra": "allow"}


class QueuePosition(BaseModel):
    """Queue position for a resting order."""
    order_id: str = ""
    queue_position: int = 0

    model_config = {"extra": "allow"}


class OrderGroup(BaseModel):
    """POST /portfolio/order_groups response."""
    order_group_id: str = ""

    model_config = {"extra": "allow"}


class ExchangeStatus(BaseModel):
    """GET /exchange/status."""
    exchange_active: bool = False
    trading_active: bool = False
    exchange_estimated_resume_time: Optional[str] = None


# ---------------------------------------------------------------------------
# WebSocket message models
# ---------------------------------------------------------------------------

class WSTrade(BaseModel):
    """Public trade from 'trade' WS channel."""
    market_ticker: str = ""
    yes_price: int = 0
    no_price: int = 0
    count: int = 0
    taker_side: str = ""  # yes / no
    ts: int = 0  # unix timestamp (seconds)
    trade_id: str = ""

    model_config = {"extra": "allow"}


class WSTickerUpdate(BaseModel):
    """Price update from 'ticker' WS channel."""
    market_ticker: str = ""
    price: int = 0
    yes_bid: int = 0
    yes_ask: int = 0
    volume: int = 0
    open_interest: int = 0

    model_config = {"extra": "allow"}


class WSFillNotification(BaseModel):
    """Fill notification from 'fill' WS channel."""
    trade_id: str = ""
    order_id: str = ""
    market_ticker: str = ""
    is_taker: bool = False
    side: str = ""
    action: str = ""
    count: int = 0
    yes_price: int = 0
    no_price: int = 0
    created_time: Optional[str] = None

    model_config = {"extra": "allow"}


class WSPositionUpdate(BaseModel):
    """Position update from 'market_positions' WS channel."""
    market_ticker: str = ""
    position: int = 0
    market_exposure: int = 0
    realized_pnl: int = 0
    total_traded: int = 0
    resting_orders_count: int = 0

    model_config = {"extra": "allow"}


class WSOrderbookSnapshot(BaseModel):
    """Orderbook snapshot from 'orderbook_delta' WS channel (type=snapshot)."""
    market_ticker: str = ""
    yes: List[List[int]] = Field(default_factory=list)
    no: List[List[int]] = Field(default_factory=list)
    seq: int = 0
    ts: int = 0

    model_config = {"extra": "allow"}


class WSOrderbookDelta(BaseModel):
    """Orderbook delta from 'orderbook_delta' WS channel (type=delta)."""
    market_ticker: str = ""
    price: int = 0
    delta: int = 0
    side: str = ""  # yes / no
    seq: int = 0
    ts: int = 0

    model_config = {"extra": "allow"}
