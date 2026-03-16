"""Kalshi fee calculator for market making.

Computes maker/taker fees, break-even spreads, and profitability thresholds.
All prices/fees are in cents (1-99 scale for Kalshi binary contracts).

Fee formulas (from Kalshi docs):
  maker_fee = 0.0175 * P * (1 - P)  per contract
  taker_fee = 0.07   * P * (1 - P)  per contract

where P = price / 100 (probability).

Key Properties:
  - Maker fee maxes at 0.4375c at P=50c (vs taker max 1.75c)
  - Maker is ~75% cheaper than taker
  - Break-even spread = 2 * maker_fee (both legs are maker)
"""

from typing import Dict


def maker_fee(price_cents: int) -> float:
    """Maker fee per contract at a given price.

    Args:
        price_cents: Price in cents (1-99).

    Returns:
        Fee in cents (float, may be fractional).
    """
    p = price_cents / 100.0
    return 0.0175 * p * (1.0 - p) * 100.0  # Convert back to cents


def taker_fee(price_cents: int) -> float:
    """Taker fee per contract at a given price.

    Args:
        price_cents: Price in cents (1-99).

    Returns:
        Fee in cents (float).
    """
    p = price_cents / 100.0
    return 0.07 * p * (1.0 - p) * 100.0


def break_even_spread(mid_cents: float) -> float:
    """Minimum spread (in cents) to break even on a round-trip as maker.

    Both the bid and the ask are maker orders, so total fee = 2 * maker_fee.
    We approximate using maker_fee at mid for both sides.

    Args:
        mid_cents: Midpoint price in cents.

    Returns:
        Break-even spread in cents (float).
    """
    fee = maker_fee(int(round(mid_cents)))
    return 2.0 * fee


def spread_pnl_per_contract(bid_cents: int, ask_cents: int) -> float:
    """Net P&L per contract from capturing the spread (both sides fill as maker).

    Args:
        bid_cents: Our bid price.
        ask_cents: Our ask price.

    Returns:
        Net P&L in cents per contract (spread - fees).
    """
    spread = ask_cents - bid_cents
    bid_fee = maker_fee(bid_cents)
    ask_fee = maker_fee(ask_cents)
    return spread - bid_fee - ask_fee


def fee_schedule(price_cents: int) -> Dict[str, float]:
    """Full fee breakdown at a given price.

    Returns:
        Dict with maker_fee, taker_fee, break_even_spread, maker_savings_pct.
    """
    mf = maker_fee(price_cents)
    tf = taker_fee(price_cents)
    bes = break_even_spread(float(price_cents))
    savings = (1.0 - mf / tf) * 100.0 if tf > 0 else 0.0
    return {
        "price_cents": price_cents,
        "maker_fee": round(mf, 4),
        "taker_fee": round(tf, 4),
        "break_even_spread": round(bes, 4),
        "maker_savings_pct": round(savings, 1),
    }
