"""SwingDetector - Pure-computation module for detecting significant price moves.

Detects price swings from two complementary sources:

A) Historical candlestick scan — Walks EventMeta.candlestick_series() output
   (Dict[str, list] keyed by market_ticker, each value a list of {ts, c, v} dicts)
   and identifies candles where |close_n - close_{n-1}| >= threshold. Volume
   confirmation requires the swing candle volume exceeds multiplier * average
   event hourly volume. Runs at startup + on each hourly candlestick refresh.

B) Live BBO tracking — Maintains per-market reference price (yes_mid). When the
   mid deviates >= threshold from the reference within a sliding time window,
   emits a PriceSwing. Reference resets after detection or when the window
   expires.

Architecture position:
  - Sits alongside AttentionRouter as a signal source in the single-arb system
  - Feeds detected swings into SwingNewsService for news correlation
  - Pure Python, no I/O, no async — all state maintained in-memory

Design principles:
  - Deterministic: same inputs always produce same outputs
  - Deduplication prevents redundant swing emissions
  - Cooldown on live swings prevents flooding from volatile markets
  - Minimal memory footprint via seen-key sets and cooldown dicts

Key data structures:
  - PriceSwing: immutable record of a detected price move
  - _RefPrice: internal mutable reference for live BBO tracking
  - SwingDetector: stateful detector with scan + live update methods
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .index import EventMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.swing_detector")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PriceSwing:
    """A detected significant price move from either candlestick or live data."""

    event_ticker: str
    market_ticker: str
    market_title: str
    direction: str           # "up" / "down"
    change_cents: float
    price_before: float
    price_after: float
    swing_start_ts: float
    swing_end_ts: float
    volume_during: int
    source: str              # "candlestick" / "live"
    news_searched: bool = False
    news_found: bool = False


@dataclass
class _RefPrice:
    """Internal reference price for live BBO tracking."""

    price: float
    ts: float
    event_ticker: str
    market_title: str


# ---------------------------------------------------------------------------
# SwingDetector
# ---------------------------------------------------------------------------

class SwingDetector:
    """Detects significant price swings from candlestick history and live BBO updates.

    Parameters:
        min_change_cents: Minimum absolute price change (in cents) to qualify as a swing.
        volume_multiplier: Swing candle volume must exceed this multiple of
            the average hourly volume across the event's candlestick data.
        live_window_seconds: Sliding window for live BBO reference prices and
            cooldown between live swing emissions on the same market.
    """

    def __init__(
        self,
        min_change_cents: float = 5.0,
        volume_multiplier: float = 2.0,
        live_window_seconds: float = 900.0,
    ) -> None:
        self.min_change_cents = min_change_cents
        self.volume_multiplier = volume_multiplier
        self.live_window_seconds = live_window_seconds

        # All detected swings (append-only)
        self._swings: List[PriceSwing] = []

        # Candlestick dedup: (market_ticker, swing_start_ts)
        self._seen_candle_keys: Set[Tuple[str, float]] = set()

        # Live BBO reference prices per market_ticker
        self._references: Dict[str, _RefPrice] = {}

        # Live swing cooldown: market_ticker -> last emission timestamp
        self._recent_live: Dict[str, float] = {}

        # Telemetry counters
        self._candle_scans: int = 0
        self._bbo_updates: int = 0
        self._candle_swings_detected: int = 0
        self._live_swings_detected: int = 0

    # ------------------------------------------------------------------
    # A) Candlestick scanning
    # ------------------------------------------------------------------

    def scan_candlesticks(
        self,
        event_ticker: str,
        event_title: str,
        candlestick_series: Dict[str, list],
        market_titles: Optional[Dict[str, str]] = None,
    ) -> List[PriceSwing]:
        """Scan candlestick series for price swings.

        Args:
            event_ticker: The event ticker (e.g. "KXBTC-25020").
            event_title: Human-readable event title.
            candlestick_series: Output of EventMeta.candlestick_series().
                Dict keyed by market_ticker, each value is a list of
                {"ts": float, "c": int, "v": int} dicts sorted by time.
            market_titles: Optional mapping of market_ticker -> human title.
                Falls back to market_ticker if not provided.

        Returns:
            List of newly detected PriceSwings (deduped against previous scans).
        """
        self._candle_scans += 1
        if not candlestick_series:
            return []

        market_titles = market_titles or {}

        # Compute average hourly volume across all markets in this event
        avg_hourly_volume = self._compute_avg_hourly_volume(candlestick_series)

        # Volume threshold: swing candle must exceed this
        volume_threshold = avg_hourly_volume * self.volume_multiplier

        new_swings: List[PriceSwing] = []

        for market_ticker, candles in candlestick_series.items():
            if len(candles) < 2:
                continue

            title = market_titles.get(market_ticker, market_ticker)

            for i in range(1, len(candles)):
                prev = candles[i - 1]
                curr = candles[i]

                prev_close = prev.get("c")
                curr_close = curr.get("c")
                if prev_close is None or curr_close is None:
                    continue

                change = curr_close - prev_close
                abs_change = abs(change)

                if abs_change < self.min_change_cents:
                    continue

                # Volume confirmation
                candle_volume = curr.get("v", 0)
                if volume_threshold > 0 and candle_volume < volume_threshold:
                    continue

                # Dedup check
                swing_start_ts = prev.get("ts", 0.0)
                dedup_key = (market_ticker, swing_start_ts)
                if dedup_key in self._seen_candle_keys:
                    continue

                self._seen_candle_keys.add(dedup_key)

                swing = PriceSwing(
                    event_ticker=event_ticker,
                    market_ticker=market_ticker,
                    market_title=title,
                    direction="up" if change > 0 else "down",
                    change_cents=abs_change,
                    price_before=float(prev_close),
                    price_after=float(curr_close),
                    swing_start_ts=swing_start_ts,
                    swing_end_ts=curr.get("ts", 0.0),
                    volume_during=candle_volume,
                    source="candlestick",
                )
                new_swings.append(swing)
                self._swings.append(swing)
                self._candle_swings_detected += 1

                logger.info(
                    "Candlestick swing: %s %s %.1fc (%s→%s) vol=%d | %s",
                    market_ticker,
                    swing.direction,
                    abs_change,
                    prev_close,
                    curr_close,
                    candle_volume,
                    title,
                )

        return new_swings

    def scan_all_events(self, events: Dict[str, "EventMeta"]) -> List[PriceSwing]:
        """Scan all events in the index for candlestick swings.

        Args:
            events: Dict of event_ticker -> EventMeta from EventArbIndex.events.

        Returns:
            Aggregated list of newly detected PriceSwings across all events.
        """
        all_new: List[PriceSwing] = []

        for event_ticker, event_meta in events.items():
            series = event_meta.candlestick_series()
            if not series:
                continue

            # Build market_titles from EventMeta.markets
            market_titles = {
                ticker: m.title
                for ticker, m in event_meta.markets.items()
                if m.title
            }

            new_swings = self.scan_candlesticks(
                event_ticker=event_ticker,
                event_title=event_meta.title,
                candlestick_series=series,
                market_titles=market_titles,
            )
            all_new.extend(new_swings)

        if all_new:
            logger.info(
                "scan_all_events: %d new swings from %d events",
                len(all_new),
                len(events),
            )

        return all_new

    # ------------------------------------------------------------------
    # B) Live BBO tracking
    # ------------------------------------------------------------------

    def on_bbo_update(
        self,
        event_ticker: str,
        market_ticker: str,
        market_title: str,
        yes_mid: float,
        timestamp: Optional[float] = None,
    ) -> Optional[PriceSwing]:
        """Process a live BBO update. Returns a PriceSwing if threshold crossed.

        Maintains per-market reference prices. When yes_mid deviates from the
        reference by >= min_change_cents within live_window_seconds, emits a
        PriceSwing and resets the reference. If the reference is older than the
        window, it silently updates to the current price.

        Args:
            event_ticker: Event ticker for this market.
            market_ticker: The market ticker.
            market_title: Human-readable market title.
            yes_mid: Current YES midpoint price in cents.
            timestamp: Optional epoch timestamp. Defaults to time.time().

        Returns:
            A PriceSwing if a swing was detected, None otherwise.
        """
        self._bbo_updates += 1
        now = timestamp if timestamp is not None else time.time()

        ref = self._references.get(market_ticker)

        # First observation for this market — set reference
        if ref is None:
            self._references[market_ticker] = _RefPrice(
                price=yes_mid,
                ts=now,
                event_ticker=event_ticker,
                market_title=market_title,
            )
            return None

        elapsed = now - ref.ts

        # Reference expired — slide forward without emitting
        if elapsed > self.live_window_seconds:
            self._references[market_ticker] = _RefPrice(
                price=yes_mid,
                ts=now,
                event_ticker=event_ticker,
                market_title=market_title,
            )
            return None

        change = yes_mid - ref.price
        abs_change = abs(change)

        if abs_change < self.min_change_cents:
            return None

        # Cooldown check — prevent flooding from volatile markets
        last_emission = self._recent_live.get(market_ticker, 0.0)
        if (now - last_emission) < self.live_window_seconds:
            return None

        # Emit swing
        swing = PriceSwing(
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            market_title=market_title,
            direction="up" if change > 0 else "down",
            change_cents=abs_change,
            price_before=ref.price,
            price_after=yes_mid,
            swing_start_ts=ref.ts,
            swing_end_ts=now,
            volume_during=0,  # No volume info from BBO alone
            source="live",
        )

        self._swings.append(swing)
        self._live_swings_detected += 1

        # Reset reference and record cooldown
        self._references[market_ticker] = _RefPrice(
            price=yes_mid,
            ts=now,
            event_ticker=event_ticker,
            market_title=market_title,
        )
        self._recent_live[market_ticker] = now

        logger.info(
            "Live swing: %s %s %.1fc (%.1f→%.1f) window=%.0fs | %s",
            market_ticker,
            swing.direction,
            abs_change,
            ref.price,
            yes_mid,
            elapsed,
            market_title,
        )

        return swing

    # ------------------------------------------------------------------
    # Query / mutation for news pipeline
    # ------------------------------------------------------------------

    def get_unsearched_swings(self, limit: int = 10) -> List[PriceSwing]:
        """Get swings that haven't been searched for news yet.

        Returns swings sorted by change_cents descending (largest moves first),
        limited to `limit` results.

        Args:
            limit: Maximum number of swings to return.

        Returns:
            List of PriceSwing instances where news_searched is False.
        """
        unsearched = [s for s in self._swings if not s.news_searched]
        unsearched.sort(key=lambda s: s.change_cents, reverse=True)
        return unsearched[:limit]

    def mark_searched(self, swing: PriceSwing, news_found: bool = False) -> None:
        """Mark a swing as searched (called by SwingNewsService after news lookup).

        Args:
            swing: The PriceSwing to mark.
            news_found: Whether relevant news was found for this swing.
        """
        swing.news_searched = True
        swing.news_found = news_found

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_swings(self) -> List[PriceSwing]:
        """All detected swings (both candlestick and live)."""
        return list(self._swings)

    @property
    def stats(self) -> Dict:
        """Telemetry stats for status reporting and debugging.

        Returns:
            Dict with scan counts, swing counts, and breakdown by source.
        """
        candle_count = sum(1 for s in self._swings if s.source == "candlestick")
        live_count = sum(1 for s in self._swings if s.source == "live")
        searched = sum(1 for s in self._swings if s.news_searched)
        news_found = sum(1 for s in self._swings if s.news_found)

        return {
            "total_swings": len(self._swings),
            "candlestick_swings": candle_count,
            "live_swings": live_count,
            "news_searched": searched,
            "news_found": news_found,
            "unsearched": len(self._swings) - searched,
            "candle_scans": self._candle_scans,
            "bbo_updates": self._bbo_updates,
            "tracked_markets": len(self._references),
            "seen_candle_keys": len(self._seen_candle_keys),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_avg_hourly_volume(candlestick_series: Dict[str, list]) -> float:
        """Compute average hourly volume across all markets in the series.

        Estimates the hourly rate from the total volume and time span of each
        market's candle data, then averages across markets. Returns 0.0 if
        no valid data.

        Args:
            candlestick_series: Dict keyed by market_ticker, values are lists
                of {"ts": float, "c": int, "v": int} dicts.

        Returns:
            Average hourly volume as a float.
        """
        hourly_rates: List[float] = []

        for _ticker, candles in candlestick_series.items():
            if len(candles) < 2:
                continue

            total_volume = sum(c.get("v", 0) for c in candles)
            if total_volume == 0:
                continue

            first_ts = candles[0].get("ts", 0)
            last_ts = candles[-1].get("ts", 0)
            span_hours = (last_ts - first_ts) / 3600.0

            if span_hours > 0:
                hourly_rates.append(total_volume / span_hours)

        if not hourly_rates:
            return 0.0

        return sum(hourly_rates) / len(hourly_rates)
