"""In-memory top-trades ranking for Flowboard."""

from datetime import datetime, timedelta

import pytest

from kalshiflow.aggregator import TradeAggregator
from kalshiflow.models import Trade


def _trade(ticker: str, count: int, yes_price: int, ts: int, side: str = "yes") -> Trade:
    return Trade(
        market_ticker=ticker,
        yes_price=yes_price,
        no_price=100 - yes_price,
        yes_price_dollars=yes_price / 100.0,
        no_price_dollars=(100 - yes_price) / 100.0,
        count=count,
        taker_side=side,
        ts=ts,
    )


@pytest.mark.asyncio
async def test_top_trades_ranked_by_in_memory_volume():
    aggregator = TradeAggregator(window_minutes=10)
    now_ms = int(datetime.now().timestamp() * 1000)

    aggregator.process_trade(_trade("SMALL", 10, 50, now_ms - 1000))
    aggregator.process_trade(_trade("BIG", 500, 80, now_ms - 500))
    aggregator.process_trade(_trade("MID", 100, 40, now_ms - 200))

    top = await aggregator.update_top_trades(force=True)

    assert [trade["market_ticker"] for trade in top] == ["BIG", "MID", "SMALL"]
    assert top[0]["volume_dollars"] == pytest.approx(400.0)
    assert top[1]["volume_dollars"] == pytest.approx(40.0)
    assert top[2]["volume_dollars"] == pytest.approx(5.0)
    assert "time_ago" in top[0]


@pytest.mark.asyncio
async def test_top_trades_ignore_trades_outside_window():
    aggregator = TradeAggregator(window_minutes=10)
    aggregator._top_trades_window_minutes = 10
    now_ms = int(datetime.now().timestamp() * 1000)
    stale_ms = int((datetime.now() - timedelta(minutes=30)).timestamp() * 1000)

    aggregator.process_trade(_trade("STALE", 1000, 90, stale_ms))
    aggregator.process_trade(_trade("FRESH", 20, 50, now_ms))

    top = await aggregator.update_top_trades(force=True)

    assert len(top) == 1
    assert top[0]["market_ticker"] == "FRESH"
