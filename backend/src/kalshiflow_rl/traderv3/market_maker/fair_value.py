"""Fair value estimation for market making.

Combines multiple signals into a single fair value estimate per market:
  1. Microprice (depth-weighted) - already computed on MarketMeta
  2. VWAP from recent trades
  3. Complement constraint for mutually-exclusive events (sum ≈ 100)

The FairValueEstimator is called by QuoteEngine each refresh cycle.
"""

import logging
import time
from typing import Dict, List, Optional

from ..single_arb.index import EventMeta, MarketMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.fair_value")

# Weights for combining signals (must sum to 1.0)
W_MICROPRICE = 0.5
W_VWAP = 0.2
W_COMPLEMENT = 0.3

# VWAP window
VWAP_WINDOW_SECONDS = 300  # 5 minutes


def estimate_fair_value(
    market: MarketMeta,
    event: EventMeta,
    complement_target: float = 100.0,
) -> Optional[float]:
    """Estimate fair value for a single market.

    Combines microprice, VWAP, and complement signals. Falls back gracefully
    when signals are unavailable.

    Args:
        market: MarketMeta with live orderbook data.
        event: EventMeta (needed for complement constraint on ME events).
        complement_target: Sum target for ME events (default 100).

    Returns:
        Fair value in cents, or None if no data.
    """
    signals: List[tuple] = []  # [(value, weight), ...]

    # Signal 1: Microprice (depth-weighted mid)
    if market.microprice is not None:
        signals.append((market.microprice, W_MICROPRICE))

    # Signal 2: VWAP from recent trades
    vwap = _compute_vwap(market)
    if vwap is not None:
        signals.append((vwap, W_VWAP))

    # Signal 3: Complement constraint (ME events only)
    if event.mutually_exclusive and len(event.markets) > 1:
        comp = _compute_complement_fv(market, event, complement_target)
        if comp is not None:
            signals.append((comp, W_COMPLEMENT))

    if not signals:
        # Absolute fallback: midpoint
        return market.yes_mid

    # Weighted average (renormalize weights)
    total_weight = sum(w for _, w in signals)
    if total_weight == 0:
        return market.yes_mid

    fv = sum(v * w for v, w in signals) / total_weight
    return round(fv, 2)


def estimate_all_fair_values(event: EventMeta) -> Dict[str, float]:
    """Estimate fair values for all markets in an event.

    Args:
        event: EventMeta with populated markets.

    Returns:
        Dict of market_ticker -> fair_value_cents.
    """
    fvs = {}
    for ticker, market in event.markets.items():
        fv = estimate_fair_value(market, event)
        if fv is not None:
            fvs[ticker] = fv
    return fvs


def _compute_vwap(market: MarketMeta) -> Optional[float]:
    """Volume-weighted average price from recent trades.

    Uses the market's recent_trades ring buffer, filtered to VWAP_WINDOW.
    """
    if not market.recent_trades:
        return None

    cutoff = time.time() - VWAP_WINDOW_SECONDS
    total_volume = 0
    total_value = 0.0

    for trade in market.recent_trades:
        ts = trade.get("ts", 0)
        if ts < cutoff:
            break  # Ring buffer is newest-first
        price = trade.get("yes_price")
        count = trade.get("count", 1)
        if price is not None and count > 0:
            total_value += price * count
            total_volume += count

    if total_volume == 0:
        return None

    return total_value / total_volume


def _compute_complement_fv(
    market: MarketMeta,
    event: EventMeta,
    target: float = 100.0,
) -> Optional[float]:
    """Complement-implied fair value for mutually exclusive events.

    For ME events, probabilities should sum to ~100. If all other markets
    have fair values, this market's complement FV = target - sum(others).

    Args:
        market: The market we're computing FV for.
        event: The parent event.
        target: Sum target (default 100 for ME).

    Returns:
        Complement-implied FV in cents, or None.
    """
    other_sum = 0.0
    count = 0

    for ticker, other in event.markets.items():
        if ticker == market.ticker:
            continue
        # Use microprice as the best available signal for other markets
        fv = other.microprice or other.yes_mid
        if fv is not None:
            other_sum += fv
            count += 1

    # Need data from all other markets for complement to be useful
    if count < len(event.markets) - 1:
        return None

    complement_fv = target - other_sum
    # Clamp to valid price range
    return max(1.0, min(99.0, complement_fv))
