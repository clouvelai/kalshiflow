"""Shared fixtures and factory functions for V3 Trader unit tests.

Factory functions (not fixtures) for parameterization flexibility.
Context manager for tool dependency injection to prevent cross-test contamination.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from kalshiflow_rl.traderv3.single_arb.index import (
    ArbLeg,
    EventArbIndex,
    EventMeta,
    MarketMeta,
    MicrostructureSignals,
)
from kalshiflow_rl.traderv3.config.environment import V3Config


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_market_meta(
    ticker: str = "MKT-A",
    event_ticker: str = "EVENT-1",
    yes_bid: Optional[int] = 40,
    yes_ask: Optional[int] = 45,
    bid_size: int = 10,
    ask_size: int = 10,
    **overrides,
) -> MarketMeta:
    """Create a MarketMeta with sensible defaults and live BBO."""
    m = MarketMeta(
        ticker=ticker,
        event_ticker=event_ticker,
        title=overrides.pop("title", ticker),
        status=overrides.pop("status", "open"),
    )
    # Apply any remaining overrides
    for k, v in overrides.items():
        if hasattr(m, k):
            setattr(m, k, v)
    # Set BBO
    if yes_bid is not None or yes_ask is not None:
        m.update_bbo(yes_bid, yes_ask, bid_size, ask_size, source="test")
    return m


def make_event_meta(
    event_ticker: str = "EVENT-1",
    n_markets: int = 3,
    mutually_exclusive: bool = True,
    market_prices: Optional[List[Dict]] = None,
    **overrides,
) -> EventMeta:
    """Create an EventMeta with N markets that have BBO data.

    Args:
        market_prices: List of dicts with yes_bid, yes_ask per market.
            Defaults to evenly splitting 100 across N markets.
    """
    event = EventMeta(
        event_ticker=event_ticker,
        title=overrides.pop("title", f"Test Event {event_ticker}"),
        mutually_exclusive=mutually_exclusive,
        loaded_at=time.time(),
    )
    for k, v in overrides.items():
        if hasattr(event, k):
            setattr(event, k, v)

    if market_prices is None:
        # Spread prices evenly: e.g. 3 markets -> ~33c each
        base = 100 // n_markets
        market_prices = []
        for i in range(n_markets):
            price = base if i < n_markets - 1 else 100 - base * (n_markets - 1)
            market_prices.append({
                "yes_bid": price - 2,
                "yes_ask": price + 2,
            })

    for i, mp in enumerate(market_prices):
        ticker = f"{event_ticker}-MKT-{chr(65 + i)}"
        m = make_market_meta(
            ticker=ticker,
            event_ticker=event_ticker,
            yes_bid=mp.get("yes_bid"),
            yes_ask=mp.get("yes_ask"),
            bid_size=mp.get("bid_size", 10),
            ask_size=mp.get("ask_size", 10),
        )
        event.markets[ticker] = m

    return event


def make_index(
    events: Optional[List[EventMeta]] = None,
    fee: int = 1,
    min_edge: float = 3.0,
) -> EventArbIndex:
    """Create a pre-populated EventArbIndex."""
    index = EventArbIndex(fee_per_contract_cents=fee, min_edge_cents=min_edge)
    if events:
        for event in events:
            index._events[event.event_ticker] = event
            for ticker in event.markets:
                index._ticker_to_event[ticker] = event.event_ticker
    return index


def make_mock_trading_client(**overrides) -> MagicMock:
    """Create a MagicMock trading client with preset async returns."""
    client = MagicMock()

    # Default async methods
    place_response = overrides.pop("place_order_response", {
        "order": {"order_id": "test-order-001", "status": "resting"}
    })
    client.place_order = AsyncMock(return_value=place_response)

    balance_response = overrides.pop("balance_response", {
        "balance": 50000,  # $500 in cents
    })
    client.get_balance = AsyncMock(return_value=balance_response)

    positions_response = overrides.pop("positions_response", {
        "market_positions": [],
    })
    client.get_positions = AsyncMock(return_value=positions_response)

    fills_response = overrides.pop("fills_response", {
        "fills": [],
    })
    client.get_fills = AsyncMock(return_value=fills_response)

    orders_response = overrides.pop("orders_response", {
        "orders": [],
    })
    client.get_orders = AsyncMock(return_value=orders_response)

    cancel_response = overrides.pop("cancel_response", {
        "order": {"order_id": "test-order-001", "status": "canceled"},
    })
    client.cancel_order = AsyncMock(return_value=cancel_response)

    settlements_response = overrides.pop("settlements_response", {
        "settlements": [],
    })
    client.get_settlements = AsyncMock(return_value=settlements_response)

    client.order_group_id = overrides.pop("order_group_id", "test-group-001")

    return client


def make_config(**overrides) -> V3Config:
    """Create a V3Config with test defaults, without reading env vars."""
    defaults = {
        "api_url": "https://demo-api.kalshi.co/trade-api/v2",
        "ws_url": "wss://demo-api.kalshi.co/trade-api/ws/v2",
        "api_key_id": "test-key-id",
        "private_key_content": "test-private-key",
        "market_tickers": [],
    }
    defaults.update(overrides)
    return V3Config(**defaults)


@contextmanager
def inject_tool_context(
    index=None,
    gateway=None,
    memory=None,
    search=None,
    sniper=None,
    sniper_config=None,
    session=None,
    context_builder=None,
    broadcast=None,
    health_service=None,
    auto_actions=None,
):
    """Context manager to inject and reset V2 ToolContext.

    Usage:
        with inject_tool_context(index=mock_index, gateway=mock_gw):
            result = await some_tool.ainvoke({...})
        # ToolContext is auto-reset after the block
    """
    from kalshiflow_rl.traderv3.single_arb.tools import ToolContext, set_context, get_context

    saved = get_context()

    ctx = ToolContext(
        gateway=gateway or MagicMock(),
        index=index or make_index(),
        memory=memory or MagicMock(),
        search=search,
        sniper=sniper,
        sniper_config=sniper_config,
        session=session or MagicMock(),
        context_builder=context_builder or MagicMock(),
        broadcast=broadcast,
        health_service=health_service,
        auto_actions=auto_actions,
    )
    set_context(ctx)

    try:
        yield ctx
    finally:
        set_context(saved)
