"""
EventArbIndex - Core stateful index for single-event arbitrage.

Rich data structures backed by full Kalshi API responses, with real-time
orderbook depth, trade feed, and fast arb signal operators.
"""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.index")

MAX_RECENT_TRADES = 50
MAX_ORDERBOOK_LEVELS = 5

# Microstructure constants
WHALE_THRESHOLD = 100       # Contracts per trade to qualify as whale
RAPID_TRADE_MS = 100        # Max inter-trade gap for rapid sequence
VOLUME_WINDOW_SECONDS = 300 # 5-minute rolling window for volume stats


@dataclass
class ArbLeg:
    """One leg of a multi-market arb."""
    ticker: str
    title: str
    side: str       # "yes" or "no"
    action: str     # "buy" or "sell"
    price_cents: int
    size_available: int


@dataclass
class ArbOpportunity:
    """Detected opportunity emitted to Captain."""
    event_ticker: str
    direction: str       # "long" or "short"
    edge_cents: float
    edge_after_fees: float
    legs: List[ArbLeg]
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "event_ticker": self.event_ticker,
            "direction": self.direction,
            "edge_cents": self.edge_cents,
            "edge_after_fees": self.edge_after_fees,
            "legs": [
                {
                    "ticker": leg.ticker,
                    "title": leg.title,
                    "side": leg.side,
                    "action": leg.action,
                    "price_cents": leg.price_cents,
                    "size_available": leg.size_available,
                }
                for leg in self.legs
            ],
            "detected_at": self.detected_at,
        }


@dataclass
class MicrostructureSignals:
    """Incrementally computed microstructure signals per market."""
    # Whale detection
    whale_trade_count: int = 0
    last_whale_ts: float = 0.0
    last_whale_size: int = 0

    # Trade cadence
    avg_inter_trade_ms: float = 0.0
    min_inter_trade_ms: float = 9999.0
    rapid_sequence_count: int = 0

    # Size consistency (over recent trades)
    consistent_size_ratio: float = 0.0   # 1.0 = all same size, 0.0 = high variance
    modal_trade_size: int = 0

    # Orderbook imbalance
    book_imbalance: float = 0.0          # (bid_depth - ask_depth) / total_depth
    total_bid_depth: int = 0
    total_ask_depth: int = 0

    # Volume (5-minute window)
    volume_5m: int = 0
    buy_volume_5m: int = 0
    sell_volume_5m: int = 0
    buy_sell_ratio: float = 0.5          # 0.0 = all sells, 1.0 = all buys

    def to_dict(self) -> Dict:
        return {
            "whale_trade_count": self.whale_trade_count,
            "last_whale_ts": self.last_whale_ts,
            "last_whale_size": self.last_whale_size,
            "avg_inter_trade_ms": round(self.avg_inter_trade_ms, 1),
            "min_inter_trade_ms": round(self.min_inter_trade_ms, 1) if self.min_inter_trade_ms < 9999.0 else None,
            "rapid_sequence_count": self.rapid_sequence_count,
            "consistent_size_ratio": round(self.consistent_size_ratio, 2),
            "modal_trade_size": self.modal_trade_size,
            "book_imbalance": round(self.book_imbalance, 3),
            "total_bid_depth": self.total_bid_depth,
            "total_ask_depth": self.total_ask_depth,
            "volume_5m": self.volume_5m,
            "buy_volume_5m": self.buy_volume_5m,
            "sell_volume_5m": self.sell_volume_5m,
            "buy_sell_ratio": round(self.buy_sell_ratio, 2),
        }


@dataclass
class MarketMeta:
    """
    Wraps one market from the Kalshi API response plus live state.

    Stores full API response fields, live orderbook depth, trade buffer,
    and ticker_v2 data (price, volume, OI).
    """
    # Full API fields
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    ticker: str = ""
    event_ticker: str = ""
    title: str = ""
    subtitle: str = ""
    status: str = "open"
    close_time: Optional[str] = None
    volume_24h: int = 0
    open_interest: int = 0

    # Live orderbook BBO
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    yes_bid_size: int = 0
    yes_ask_size: int = 0

    # Full orderbook depth: [[price, size], ...]
    yes_levels: List[List[int]] = field(default_factory=list)
    no_levels: List[List[int]] = field(default_factory=list)

    # Live trades (ring buffer)
    recent_trades: List[Dict] = field(default_factory=list)
    trade_count: int = 0
    last_trade_price: Optional[int] = None
    last_trade_side: Optional[str] = None

    # Ticker_v2 data
    last_price: Optional[int] = None
    volume_delta_total: int = 0
    oi_delta_total: int = 0

    # Derived / freshness
    yes_mid: Optional[float] = None
    spread: Optional[int] = None
    source: str = "none"
    ws_updated_at: float = 0.0
    api_updated_at: float = 0.0

    # Microstructure signals (computed incrementally)
    micro: MicrostructureSignals = field(default_factory=MicrostructureSignals)
    _last_trade_ts: float = 0.0  # For inter-trade delta computation

    @classmethod
    def from_api(cls, market_data: Dict, event_ticker: str) -> "MarketMeta":
        """Create MarketMeta from a single market dict in the event API response."""
        ticker = market_data.get("ticker", "")
        return cls(
            raw=market_data,
            ticker=ticker,
            event_ticker=event_ticker,
            title=market_data.get("yes_sub_title", market_data.get("title", ticker)),
            subtitle=market_data.get("no_sub_title", ""),
            status=market_data.get("status", "open"),
            close_time=market_data.get("close_time") or market_data.get("expected_expiration_time"),
            volume_24h=market_data.get("volume_24h", 0),
            open_interest=market_data.get("open_interest", 0),
            last_price=market_data.get("last_price") or market_data.get("previous_price"),
        )

    def update_bbo(
        self,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
        bid_size: int = 0,
        ask_size: int = 0,
        source: str = "ws",
    ) -> None:
        """Update BBO and recompute derived fields."""
        self.yes_bid = yes_bid
        self.yes_ask = yes_ask
        self.yes_bid_size = bid_size
        self.yes_ask_size = ask_size
        self.source = source

        if source == "ws":
            self.ws_updated_at = time.time()
        elif source == "api":
            self.api_updated_at = time.time()

        # Derived
        if yes_bid is not None and yes_ask is not None:
            self.yes_mid = (yes_bid + yes_ask) / 2.0
            self.spread = yes_ask - yes_bid
        elif yes_bid is not None:
            self.yes_mid = float(yes_bid)
            self.spread = None
        elif yes_ask is not None:
            self.yes_mid = float(yes_ask)
            self.spread = None
        else:
            self.yes_mid = None
            self.spread = None

    def update_orderbook(
        self,
        yes_levels: List[List[int]],
        no_levels: List[List[int]],
        source: str = "ws",
    ) -> None:
        """Update full orderbook depth, extract BBO, and compute book imbalance."""
        self.yes_levels = yes_levels[:MAX_ORDERBOOK_LEVELS]
        self.no_levels = no_levels[:MAX_ORDERBOOK_LEVELS]

        # Extract BBO from levels
        yes_bid = yes_levels[0][0] if yes_levels else None
        yes_bid_size = yes_levels[0][1] if yes_levels else 0
        # YES ask = 100 - best NO bid price
        yes_ask = 100 - no_levels[0][0] if no_levels else None
        yes_ask_size = no_levels[0][1] if no_levels else 0

        self.update_bbo(yes_bid, yes_ask, yes_bid_size, yes_ask_size, source)

        # Book imbalance from depth
        bid_depth = sum(level[1] for level in self.yes_levels) if self.yes_levels else 0
        ask_depth = sum(level[1] for level in self.no_levels) if self.no_levels else 0
        total_depth = bid_depth + ask_depth
        self.micro.total_bid_depth = bid_depth
        self.micro.total_ask_depth = ask_depth
        self.micro.book_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

    def update_ticker(self, price: Optional[int], volume_delta: int = 0, oi_delta: int = 0) -> None:
        """Update from ticker_v2 channel data."""
        if price is not None:
            self.last_price = price
        self.volume_delta_total += volume_delta
        self.oi_delta_total += oi_delta

    def add_trade(self, trade: Dict) -> None:
        """Add a public trade to the ring buffer and update microstructure signals."""
        self.recent_trades.insert(0, trade)
        if len(self.recent_trades) > MAX_RECENT_TRADES:
            self.recent_trades = self.recent_trades[:MAX_RECENT_TRADES]
        self.trade_count += 1
        self.last_trade_price = trade.get("yes_price")
        self.last_trade_side = trade.get("taker_side")

        # --- Microstructure signal updates ---
        trade_ts = trade.get("ts", time.time())
        trade_count = trade.get("count", 1)
        taker_side = trade.get("taker_side", "")

        # Whale detection
        if trade_count >= WHALE_THRESHOLD:
            self.micro.whale_trade_count += 1
            self.micro.last_whale_ts = trade_ts
            self.micro.last_whale_size = trade_count

        # Trade cadence (EMA of inter-trade delta)
        if self._last_trade_ts > 0:
            delta_ms = (trade_ts - self._last_trade_ts) * 1000
            if delta_ms > 0:
                if self.micro.avg_inter_trade_ms == 0:
                    self.micro.avg_inter_trade_ms = delta_ms
                else:
                    # EMA with alpha=0.2
                    self.micro.avg_inter_trade_ms = (
                        0.2 * delta_ms + 0.8 * self.micro.avg_inter_trade_ms
                    )
                if delta_ms < self.micro.min_inter_trade_ms:
                    self.micro.min_inter_trade_ms = delta_ms
                # Rapid sequence counting
                if delta_ms < RAPID_TRADE_MS:
                    self.micro.rapid_sequence_count += 1
        self._last_trade_ts = trade_ts

        # Volume and buy/sell ratio (recompute from ring buffer, 5-min window)
        self._recompute_volume_window()

        # Size consistency (recompute from recent 20 trades)
        self._recompute_size_consistency()

    def _recompute_volume_window(self) -> None:
        """Recompute 5-minute volume stats from ring buffer."""
        cutoff = time.time() - VOLUME_WINDOW_SECONDS
        total = 0
        buy = 0
        sell = 0
        for t in self.recent_trades:
            ts = t.get("ts", 0)
            if ts < cutoff:
                break  # Ring buffer is newest-first, stop at old trades
            count = t.get("count", 1)
            total += count
            side = t.get("taker_side", "")
            if side == "yes":
                buy += count
            elif side == "no":
                sell += count
        self.micro.volume_5m = total
        self.micro.buy_volume_5m = buy
        self.micro.sell_volume_5m = sell
        self.micro.buy_sell_ratio = buy / total if total > 0 else 0.5

    def _recompute_size_consistency(self) -> None:
        """Compute size consistency from recent 20 trades."""
        recent = self.recent_trades[:20]
        if len(recent) < 3:
            return
        sizes = [t.get("count", 1) for t in recent]
        avg = sum(sizes) / len(sizes)
        if avg == 0:
            return
        variance = sum((s - avg) ** 2 for s in sizes) / len(sizes)
        cv = (variance ** 0.5) / avg  # Coefficient of variation
        self.micro.consistent_size_ratio = max(0.0, 1.0 - cv)
        # Modal size
        counter = Counter(sizes)
        self.micro.modal_trade_size = counter.most_common(1)[0][0]

    @property
    def has_data(self) -> bool:
        return self.yes_bid is not None or self.yes_ask is not None

    @property
    def freshness_seconds(self) -> float:
        latest = max(self.ws_updated_at, self.api_updated_at)
        if latest == 0:
            return 9999.0  # No data yet, use large number (JSON doesn't support Infinity)
        return time.time() - latest

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "status": self.status,
            "close_time": self.close_time,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "yes_bid_size": self.yes_bid_size,
            "yes_ask_size": self.yes_ask_size,
            "yes_levels": self.yes_levels,
            "no_levels": self.no_levels,
            "yes_mid": self.yes_mid,
            "spread": self.spread,
            "source": self.source,
            "ws_updated_at": self.ws_updated_at,
            "api_updated_at": self.api_updated_at,
            "freshness_seconds": round(self.freshness_seconds, 1),
            "volume_24h": self.volume_24h,
            "open_interest": self.open_interest,
            "last_price": self.last_price,
            "last_trade_price": self.last_trade_price,
            "last_trade_side": self.last_trade_side,
            "trade_count": self.trade_count,
            "volume_delta_total": self.volume_delta_total,
            "oi_delta_total": self.oi_delta_total,
            "micro": self.micro.to_dict(),
        }


@dataclass
class EventMeta:
    """
    Wraps one event from the Kalshi API plus live data and computed signals.
    """
    # Full API fields
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    event_ticker: str = ""
    series_ticker: str = ""
    title: str = ""
    category: str = ""
    mutually_exclusive: bool = False
    subtitle: str = ""

    markets: Dict[str, MarketMeta] = field(default_factory=dict)
    loaded_at: float = 0.0
    updated_at: float = 0.0

    # Event metadata (image, links)
    image_url: Optional[str] = None
    kalshi_url: Optional[str] = None

    # Candlestick data (fetched at startup, refreshed periodically)
    # Raw response from get_event_candlesticks: {market_tickers, market_candlesticks}
    candlesticks: Dict[str, Any] = field(default_factory=dict)
    candlesticks_fetched_at: float = 0.0

    # Structured event understanding (populated by UnderstandingBuilder)
    # Contains: participants, key_factors, trading_summary, timeline, extensions, etc.
    understanding: Optional[Dict[str, Any]] = None

    # Mentions-specific data (CUMULATIVE STATE across Captain sessions)
    # Contains:
    # - lexeme_pack: LexemePackLite (parsed rules) - Dict version
    # - current_count: int (CUMULATIVE mention count across all sources)
    # - evidence: List[Dict] (all grounded hits with full provenance)
    # - sources_scanned: List[str] (URLs already processed to prevent double-counting)
    # - last_scan_ts: float (timestamp of last scan)
    mentions_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        """Explicit event type for UI and Captain awareness.

        Returns:
            "mutually_exclusive" - Outcomes are mutually exclusive, sum should = 100%
            "independent" - Independent binary outcomes, sum > 100% is normal
        """
        if self.mutually_exclusive:
            return "mutually_exclusive"
        else:
            return "independent"

    @classmethod
    async def from_event_ticker(
        cls, event_ticker: str, trading_client
    ) -> Optional["EventMeta"]:
        """Fetch event via REST, construct EventMeta with MarketMeta per market."""
        try:
            event_data = await trading_client.get_event(event_ticker)
            if not event_data:
                logger.warning(f"Empty response for event {event_ticker}")
                return None

            # get_event returns the event dict directly (with markets attached)
            event = event_data

            markets_data = event.get("markets", [])
            if not markets_data:
                logger.warning(f"Event {event_ticker} has no markets")
                return None

            meta = cls(
                raw=event,
                event_ticker=event_ticker,
                series_ticker=event.get("series_ticker", ""),
                title=event.get("title", event_ticker),
                category=event.get("category", ""),
                mutually_exclusive=event.get("mutually_exclusive", False),
                subtitle=event.get("sub_title", ""),
                image_url=event.get("image_url") or event.get("featured_image_url"),
                kalshi_url=f"https://kalshi.com/markets/{event_ticker.lower()}",
                loaded_at=time.time(),
            )

            for market in markets_data:
                ticker = market.get("ticker", "")
                if not ticker:
                    continue
                market_status = market.get("status", "open")
                if market_status not in ("open", "active"):
                    continue
                meta.markets[ticker] = MarketMeta.from_api(market, event_ticker)

            logger.info(
                f"Loaded EventMeta {event_ticker}: {meta.title} "
                f"({len(meta.markets)} markets, mutually_exclusive={meta.mutually_exclusive})"
            )
            return meta

        except Exception as e:
            logger.error(f"Failed to load EventMeta {event_ticker}: {e}")
            return None

    async def sync(self, trading_client) -> None:
        """Re-fetch from API and update metadata (not live data)."""
        try:
            event_data = await trading_client.get_event(self.event_ticker)
            if not event_data:
                return
            event = event_data
            self.raw = event
            self.title = event.get("title", self.title)
            self.category = event.get("category", self.category)
            self.mutually_exclusive = event.get("mutually_exclusive", self.mutually_exclusive)

            # Update market metadata from API (preserve live data)
            for market_data in event.get("markets", []):
                ticker = market_data.get("ticker", "")
                if ticker in self.markets:
                    m = self.markets[ticker]
                    m.raw = market_data
                    m.volume_24h = market_data.get("volume_24h", m.volume_24h)
                    m.open_interest = market_data.get("open_interest", m.open_interest)
                    m.close_time = market_data.get("close_time") or market_data.get("expected_expiration_time") or m.close_time
                    m.status = market_data.get("status", m.status)

        except Exception as e:
            logger.error(f"Failed to sync EventMeta {self.event_ticker}: {e}")

    # ---- Candlestick Helpers ----

    def candlestick_summary(self) -> Dict[str, Any]:
        """Token-efficient candlestick summary per market.

        Returns compact trend info: price_trend, price_range, volume_trend, current_vs_7d_avg.
        """
        if not self.candlesticks:
            return {}

        market_tickers = self.candlesticks.get("market_tickers", [])
        market_candlesticks = self.candlesticks.get("market_candlesticks", [])

        if not market_tickers or not market_candlesticks:
            return {}

        summaries = {}
        for i, ticker in enumerate(market_tickers):
            if i >= len(market_candlesticks) or not market_candlesticks[i]:
                continue

            candles = market_candlesticks[i]
            if len(candles) < 2:
                continue

            # Extract price OHLC from candles
            prices = []
            volumes = []
            for c in candles:
                price_data = c.get("price", {})
                if price_data:
                    close = price_data.get("close")
                    if close is not None:
                        prices.append(close)
                vol = c.get("volume", 0)
                if vol is not None:
                    volumes.append(vol)

            if len(prices) < 2:
                continue

            # Trend: compare first third vs last third
            third = max(1, len(prices) // 3)
            early_avg = sum(prices[:third]) / third
            late_avg = sum(prices[-third:]) / third
            diff = late_avg - early_avg

            if diff > 3:
                price_trend = "up"
            elif diff < -3:
                price_trend = "down"
            else:
                price_trend = "flat"

            # Volume trend
            volume_trend = "flat"
            if len(volumes) >= 4:
                early_vol = sum(volumes[:len(volumes)//2])
                late_vol = sum(volumes[len(volumes)//2:])
                if late_vol > early_vol * 1.5:
                    volume_trend = "increasing"
                elif late_vol < early_vol * 0.5:
                    volume_trend = "decreasing"

            # Current vs 7d avg
            avg_price = sum(prices) / len(prices)
            current_price = prices[-1]

            summaries[ticker] = {
                "price_trend": price_trend,
                "price_high": max(prices),
                "price_low": min(prices),
                "price_current": current_price,
                "price_7d_avg": round(avg_price, 1),
                "volume_trend": volume_trend,
                "total_volume": sum(volumes),
                "candle_count": len(candles),
            }

        return summaries

    def candlestick_series(self) -> Dict[str, list]:
        """Return compact time-series per market for frontend charting.

        Returns dict keyed by market ticker, each value is a list of
        {ts, c, v} dicts (epoch seconds, close price in cents, volume).
        """
        if not self.candlesticks:
            return {}

        market_tickers = self.candlesticks.get("market_tickers", [])
        market_candlesticks = self.candlesticks.get("market_candlesticks", [])
        if not market_tickers or not market_candlesticks:
            return {}

        series = {}
        for i, ticker in enumerate(market_tickers):
            if i >= len(market_candlesticks) or not market_candlesticks[i]:
                continue

            points = []
            for c in market_candlesticks[i]:
                price_data = c.get("price", {})
                close = price_data.get("close") if price_data else None
                if close is None:
                    continue
                points.append({
                    "ts": c.get("end_period_ts", c.get("start_period_ts", 0)),
                    "c": close,
                    "v": c.get("volume", 0),
                })

            if points:
                series[ticker] = points

        return series

    # ---- Signal Operators ----

    def market_sum(self) -> Optional[float]:
        """Sum of YES mids across all markets. Should be ~100 in balanced market."""
        total = 0.0
        for m in self.markets.values():
            if m.yes_mid is None:
                return None
            total += m.yes_mid
        return total if self.markets else None

    def market_sum_bid(self) -> Optional[int]:
        """Sum of YES bids. >100 = short arb opportunity."""
        total = 0
        for m in self.markets.values():
            if m.yes_bid is None:
                return None
            total += m.yes_bid
        return total if self.markets else None

    def market_sum_ask(self) -> Optional[int]:
        """Sum of YES asks. <100 = long arb opportunity."""
        total = 0
        for m in self.markets.values():
            if m.yes_ask is None:
                return None
            total += m.yes_ask
        return total if self.markets else None

    def long_edge(self, fee_per_contract: int = 1) -> Optional[float]:
        """100 - sum_ask - total_fees. Positive = profitable long arb."""
        sum_ask = self.market_sum_ask()
        if sum_ask is None:
            return None
        total_fees = fee_per_contract * len(self.markets)
        return 100 - sum_ask - total_fees

    def short_edge(self, fee_per_contract: int = 1) -> Optional[float]:
        """sum_bid - 100 - total_fees. Positive = profitable short arb."""
        sum_bid = self.market_sum_bid()
        if sum_bid is None:
            return None
        total_fees = fee_per_contract * len(self.markets)
        return sum_bid - 100 - total_fees

    def deviation(self) -> Optional[float]:
        """abs(market_sum - 100). How far from equilibrium."""
        ms = self.market_sum()
        if ms is None:
            return None
        return abs(ms - 100)

    def widest_spread_market(self) -> Optional[MarketMeta]:
        """Market with largest bid-ask spread (most illiquid)."""
        widest = None
        widest_spread = -1
        for m in self.markets.values():
            if m.spread is not None and m.spread > widest_spread:
                widest_spread = m.spread
                widest = m
        return widest

    def most_active_market(self) -> Optional[MarketMeta]:
        """Market with most recent trades."""
        most = None
        most_trades = -1
        for m in self.markets.values():
            if m.trade_count > most_trades:
                most_trades = m.trade_count
                most = m
        return most

    @property
    def all_markets_have_data(self) -> bool:
        return (
            len(self.markets) > 0
            and all(m.has_data for m in self.markets.values())
        )

    @property
    def markets_with_data(self) -> int:
        return sum(1 for m in self.markets.values() if m.has_data)

    @property
    def markets_total(self) -> int:
        return len(self.markets)

    def to_dict(self) -> Dict:
        sum_bid = self.market_sum_bid()
        sum_ask = self.market_sum_ask()
        sum_mid = self.market_sum()
        result = {
            "event_ticker": self.event_ticker,
            "series_ticker": self.series_ticker,
            "title": self.title,
            "category": self.category,
            "mutually_exclusive": self.mutually_exclusive,
            "event_type": self.event_type,  # "mutually_exclusive" | "independent"
            "subtitle": self.subtitle,
            "markets": {t: m.to_dict() for t, m in self.markets.items()},
            "sum_yes_bid": sum_bid,
            "sum_yes_ask": sum_ask,
            "sum_yes_mid": round(sum_mid, 1) if sum_mid is not None else None,
            "long_edge": None,   # Computed dynamically by caller with fee config
            "short_edge": None,  # Computed dynamically by caller with fee config
            "markets_with_data": self.markets_with_data,
            "markets_total": self.markets_total,
            "market_count": self.markets_total,
            "all_markets_have_data": self.all_markets_have_data,
            "loaded_at": self.loaded_at,
            "updated_at": self.updated_at,
            "image_url": self.image_url,
            "kalshi_url": self.kalshi_url,
            "candlestick_summary": self.candlestick_summary() if self.candlesticks else {},
            "candlestick_series": self.candlestick_series() if self.candlesticks else {},
        }

        # Add micro_summary aggregated across markets
        total_whale = sum(m.micro.whale_trade_count for m in self.markets.values())
        total_vol5m = sum(m.micro.volume_5m for m in self.markets.values())
        total_rapid = sum(m.micro.rapid_sequence_count for m in self.markets.values())
        total_buy5m = sum(m.micro.buy_volume_5m for m in self.markets.values())
        total_sell5m = sum(m.micro.sell_volume_5m for m in self.markets.values())
        result["micro_summary"] = {
            "total_whale_trades": total_whale,
            "total_volume_5m": total_vol5m,
            "total_rapid_sequences": total_rapid,
            "buy_sell_ratio": round(total_buy5m / (total_buy5m + total_sell5m), 2) if (total_buy5m + total_sell5m) > 0 else 0.5,
        }

        # Add structured understanding if available
        if self.understanding:
            result["understanding"] = self.understanding

        # Add mentions info if available (for Captain visibility)
        if self.mentions_data:
            lexeme_pack = self.mentions_data.get("lexeme_pack", {})
            result["mentions_speaker"] = lexeme_pack.get("speaker")
            result["mentions_entity"] = lexeme_pack.get("entity")
            result["mentions_current_count"] = self.mentions_data.get("current_count", 0)
            result["mentions_evidence_count"] = len(self.mentions_data.get("evidence", []))
            # Include simulation status
            result["mentions_has_baseline"] = bool(self.mentions_data.get("baseline_estimates"))
            result["mentions_last_simulation_ts"] = self.mentions_data.get("last_simulation_ts")

        return result


class EventArbIndex:
    """
    Core stateful index for single-event arb detection.

    Holds N concurrent EventMeta objects. On every orderbook update,
    recomputes probability sums and detects arb opportunities.
    """

    def __init__(self, fee_per_contract_cents: int = 1, min_edge_cents: float = 3.0):
        self._events: Dict[str, EventMeta] = {}
        self._ticker_to_event: Dict[str, str] = {}  # market_ticker -> event_ticker
        self._fee_per_contract = fee_per_contract_cents
        self._min_edge_cents = min_edge_cents

    @property
    def events(self) -> Dict[str, EventMeta]:
        return self._events

    @property
    def market_tickers(self) -> List[str]:
        """All market tickers across all events."""
        return list(self._ticker_to_event.keys())

    @property
    def is_ready(self) -> bool:
        """True when all events have been loaded and all markets have orderbook data."""
        if not self._events:
            return False
        return all(event.all_markets_have_data for event in self._events.values())

    @property
    def readiness_summary(self) -> str:
        """Human-readable readiness status."""
        if not self._events:
            return "No events loaded"
        ready_events = sum(1 for e in self._events.values() if e.all_markets_have_data)
        total_markets = sum(e.markets_total for e in self._events.values())
        markets_with_data = sum(e.markets_with_data for e in self._events.values())
        return f"{ready_events}/{len(self._events)} events ready, {markets_with_data}/{total_markets} markets with data"

    def get_event_for_ticker(self, market_ticker: str) -> Optional[str]:
        """Get event_ticker for a given market_ticker."""
        return self._ticker_to_event.get(market_ticker)

    async def load_event(self, event_ticker: str, trading_client) -> Optional[EventMeta]:
        """Fetch event + markets from REST, create EventMeta."""
        meta = await EventMeta.from_event_ticker(event_ticker, trading_client)
        if meta is None:
            return None

        self._events[event_ticker] = meta
        for ticker in meta.markets:
            self._ticker_to_event[ticker] = event_ticker

        logger.info(
            f"Loaded event {event_ticker}: {meta.title} "
            f"({meta.markets_total} markets, mutually_exclusive={meta.mutually_exclusive})"
        )
        return meta

    def on_orderbook_update(
        self,
        market_ticker: str,
        yes_levels: List[List[int]],
        no_levels: List[List[int]],
        source: str = "ws",
    ) -> Optional[ArbOpportunity]:
        """Update full orderbook depth, recompute signals, detect arb."""
        event_ticker = self._ticker_to_event.get(market_ticker)
        if not event_ticker:
            return None

        event = self._events.get(event_ticker)
        if not event:
            return None

        market = event.markets.get(market_ticker)
        if not market:
            return None

        market.update_orderbook(yes_levels, no_levels, source)
        event.updated_at = time.time()

        return self._detect_arb(event_ticker)

    def on_bbo_update(
        self,
        market_ticker: str,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
        bid_size: int = 0,
        ask_size: int = 0,
        source: str = "ws",
    ) -> Optional[ArbOpportunity]:
        """Update BBO only (backward compat), recompute signals, detect arb."""
        event_ticker = self._ticker_to_event.get(market_ticker)
        if not event_ticker:
            return None

        event = self._events.get(event_ticker)
        if not event:
            return None

        market = event.markets.get(market_ticker)
        if not market:
            return None

        market.update_bbo(yes_bid, yes_ask, bid_size, ask_size, source)
        event.updated_at = time.time()

        return self._detect_arb(event_ticker)

    def on_ticker_update(
        self,
        market_ticker: str,
        price: Optional[int] = None,
        volume_delta: int = 0,
        oi_delta: int = 0,
    ) -> None:
        """Update from ticker_v2 channel."""
        event_ticker = self._ticker_to_event.get(market_ticker)
        if not event_ticker:
            return
        event = self._events.get(event_ticker)
        if not event:
            return
        market = event.markets.get(market_ticker)
        if not market:
            return
        market.update_ticker(price, volume_delta, oi_delta)
        event.updated_at = time.time()

    def on_trade(self, market_ticker: str, trade_data: Dict) -> None:
        """Record trade in the market's trade buffer."""
        event_ticker = self._ticker_to_event.get(market_ticker)
        if not event_ticker:
            return
        event = self._events.get(event_ticker)
        if not event:
            return
        market = event.markets.get(market_ticker)
        if not market:
            return
        market.add_trade(trade_data)

    def _detect_arb(self, event_ticker: str) -> Optional[ArbOpportunity]:
        """Check if edge exceeds threshold, return opportunity if so."""
        event = self._events.get(event_ticker)
        if not event or not event.all_markets_have_data:
            return None

        fee = self._fee_per_contract
        long_e = event.long_edge(fee)
        short_e = event.short_edge(fee)

        if long_e is not None and long_e > self._min_edge_cents:
            legs = self._build_long_legs(event)
            if legs:
                return ArbOpportunity(
                    event_ticker=event_ticker,
                    direction="long",
                    edge_cents=100 - (event.market_sum_ask() or 0),
                    edge_after_fees=long_e,
                    legs=legs,
                )

        if short_e is not None and short_e > self._min_edge_cents:
            legs = self._build_short_legs(event)
            if legs:
                return ArbOpportunity(
                    event_ticker=event_ticker,
                    direction="short",
                    edge_cents=(event.market_sum_bid() or 0) - 100,
                    edge_after_fees=short_e,
                    legs=legs,
                )

        return None

    def _build_long_legs(self, event: EventMeta) -> List[ArbLeg]:
        """Build buy-YES legs for long arb."""
        legs = []
        for m in event.markets.values():
            if m.yes_ask is not None:
                legs.append(ArbLeg(
                    ticker=m.ticker,
                    title=m.title,
                    side="yes",
                    action="buy",
                    price_cents=m.yes_ask,
                    size_available=m.yes_ask_size,
                ))
        return legs

    def _build_short_legs(self, event: EventMeta) -> List[ArbLeg]:
        """Build sell-YES (buy NO) legs for short arb."""
        legs = []
        for m in event.markets.values():
            if m.yes_bid is not None:
                legs.append(ArbLeg(
                    ticker=m.ticker,
                    title=m.title,
                    side="no",
                    action="buy",
                    price_cents=100 - m.yes_bid,
                    size_available=m.yes_bid_size,
                ))
        return legs

    def get_snapshot(self) -> Dict:
        """Full state for WebSocket broadcast."""
        events_dict = {}
        for et, event in self._events.items():
            d = event.to_dict()
            # Inject edge with fee config
            d["long_edge"] = round(event.long_edge(self._fee_per_contract), 1) if event.long_edge(self._fee_per_contract) is not None else None
            d["short_edge"] = round(event.short_edge(self._fee_per_contract), 1) if event.short_edge(self._fee_per_contract) is not None else None
            events_dict[et] = d
        return {
            "events": events_dict,
            "total_events": len(self._events),
            "total_markets": len(self._ticker_to_event),
            "timestamp": time.time(),
        }

    def get_event_snapshot(self, event_ticker: str) -> Optional[Dict]:
        """Single event snapshot for Captain tools."""
        event = self._events.get(event_ticker)
        if not event:
            return None
        d = event.to_dict()
        d["long_edge"] = round(event.long_edge(self._fee_per_contract), 1) if event.long_edge(self._fee_per_contract) is not None else None
        d["short_edge"] = round(event.short_edge(self._fee_per_contract), 1) if event.short_edge(self._fee_per_contract) is not None else None
        # Add signal summary for Captain
        d["signals"] = {
            "market_sum": round(event.market_sum(), 2) if event.market_sum() is not None else None,
            "market_sum_bid": event.market_sum_bid(),
            "market_sum_ask": event.market_sum_ask(),
            "deviation": round(event.deviation(), 2) if event.deviation() is not None else None,
            "widest_spread_ticker": event.widest_spread_market().ticker if event.widest_spread_market() else None,
            "most_active_ticker": event.most_active_market().ticker if event.most_active_market() else None,
            # Microstructure aggregates
            "total_whale_trades": sum(m.micro.whale_trade_count for m in event.markets.values()),
            "total_volume_5m": sum(m.micro.volume_5m for m in event.markets.values()),
            "total_rapid_sequences": sum(m.micro.rapid_sequence_count for m in event.markets.values()),
        }
        return d
