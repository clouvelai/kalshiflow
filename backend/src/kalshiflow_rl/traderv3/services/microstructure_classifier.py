"""
Microstructure Signal Classifier - Purely Descriptive Signal Categorization.

This service provides clean, factual classifications of microstructure signals
without making trading recommendations. The LLM decides how to use these signals.

Design Principles:
    1. PURELY DESCRIPTIVE - No trading recommendations
    2. FACTUAL CATEGORIES - Based on objective thresholds
    3. LLM DECIDES ACTION - Classification informs, doesn't dictate
    4. NO HARDCODED STRATEGIES - No "fade" or "follow" language

Usage:
    classifier = MicrostructureClassifier()
    classification = classifier.classify(microstructure_context)
    prompt_text = classification.to_prompt_string()

Version: v1
Created: 2026-01
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class FlowCategory(str, Enum):
    """Trade flow categories - purely descriptive."""
    HEAVY_YES = "HEAVY_YES"      # >65% YES trades
    HEAVY_NO = "HEAVY_NO"        # <35% YES trades
    BALANCED = "BALANCED"        # 35-65% YES trades
    LIGHT_ACTIVITY = "LIGHT"     # <10 total trades


class LiquidityCategory(str, Enum):
    """Liquidity categories based on spread."""
    TIGHT = "TIGHT"         # avg spread <= 2c
    NORMAL = "NORMAL"       # avg spread 3-4c
    WIDE = "WIDE"           # avg spread 5-7c
    VERY_WIDE = "VERY_WIDE" # avg spread > 7c


class VolatilityCategory(str, Enum):
    """Spread volatility categories."""
    STABLE = "STABLE"       # volatility < 0.5
    MODERATE = "MODERATE"   # volatility 0.5-1.0
    HIGH = "HIGH"           # volatility > 1.0


class ImbalanceCategory(str, Enum):
    """Orderbook volume imbalance categories."""
    STRONG_BUY = "STRONG_BUY"     # imbalance > 0.3
    MILD_BUY = "MILD_BUY"         # imbalance 0.1-0.3
    BALANCED = "BALANCED"         # imbalance -0.1 to 0.1
    MILD_SELL = "MILD_SELL"       # imbalance -0.3 to -0.1
    STRONG_SELL = "STRONG_SELL"   # imbalance < -0.3


@dataclass
class MicrostructureClassification:
    """
    Purely descriptive microstructure classification.

    NO trading recommendations - just factual categorizations.
    The LLM decides what to do with this information.
    """

    # === Flow Classification ===
    yes_trade_ratio: float
    flow_category: FlowCategory
    total_trades: int

    # === Liquidity Classification ===
    avg_spread: float
    liquidity_category: LiquidityCategory
    no_spread: int
    yes_spread: int

    # === Volatility Classification ===
    spread_volatility: float
    volatility_category: VolatilityCategory

    # === Orderbook Imbalance ===
    volume_imbalance: float
    imbalance_category: ImbalanceCategory

    # === Price Movement ===
    price_change_from_open: int  # Positive = price fell
    last_yes_price: int

    # === Data Quality Flags ===
    data_is_stale: bool = False
    insufficient_trades: bool = False

    def to_prompt_string(self) -> str:
        """
        Format as clean, factual summary for LLM consumption.

        Returns a compact multi-line string with categorized signals.
        """
        lines = []

        # Trade flow
        lines.append(
            f"- Trade Flow: {self.flow_category.value} "
            f"({self.yes_trade_ratio:.0%} YES, {self.total_trades} trades)"
        )

        # Price movement
        if self.price_change_from_open != 0:
            direction = "down" if self.price_change_from_open > 0 else "up"
            lines.append(
                f"- Price: {abs(self.price_change_from_open)}c {direction} from open "
                f"(current: {self.last_yes_price}c)"
            )

        # Liquidity
        lines.append(
            f"- Liquidity: {self.liquidity_category.value} "
            f"(spread: NO={self.no_spread}c, YES={self.yes_spread}c)"
        )

        # Orderbook imbalance
        lines.append(
            f"- Order Flow: {self.imbalance_category.value} "
            f"(imbalance: {self.volume_imbalance:+.2f})"
        )

        # Volatility (only if notable)
        if self.volatility_category != VolatilityCategory.STABLE:
            lines.append(
                f"- Spread Volatility: {self.volatility_category.value} "
                f"(σ={self.spread_volatility:.2f})"
            )

        # Data quality warnings
        if self.data_is_stale:
            lines.append("- WARNING: Data may be stale")
        if self.insufficient_trades:
            lines.append("- WARNING: Limited trade data (<10 trades)")

        return "\n".join(lines)


class MicrostructureClassifier:
    """
    Classify microstructure signals WITHOUT recommending actions.

    This classifier provides factual categorizations based on objective
    thresholds. It does NOT tell the LLM what to do with the signals.
    """

    # === Flow Thresholds ===
    HEAVY_YES_THRESHOLD = 0.65
    HEAVY_NO_THRESHOLD = 0.35
    MIN_TRADES_FOR_FLOW = 10

    # === Liquidity Thresholds (avg spread in cents) ===
    TIGHT_SPREAD_MAX = 2
    NORMAL_SPREAD_MAX = 4
    WIDE_SPREAD_MAX = 7

    # === Volatility Thresholds ===
    STABLE_VOLATILITY_MAX = 0.5
    MODERATE_VOLATILITY_MAX = 1.0

    # === Imbalance Thresholds ===
    STRONG_IMBALANCE_THRESHOLD = 0.3
    MILD_IMBALANCE_THRESHOLD = 0.1

    def classify(self, ctx: 'MicrostructureContext') -> MicrostructureClassification:
        """
        Classify microstructure signals from context.

        Args:
            ctx: MicrostructureContext from state module

        Returns:
            MicrostructureClassification with factual categories
        """
        # === FLOW CLASSIFICATION ===
        # Handle None values early for use in multiple places
        total_trades = ctx.total_trades if ctx.total_trades is not None else 0
        yes_ratio = ctx.yes_ratio if ctx.yes_ratio is not None else 0.5

        if total_trades < self.MIN_TRADES_FOR_FLOW:
            flow_category = FlowCategory.LIGHT_ACTIVITY
        elif yes_ratio > self.HEAVY_YES_THRESHOLD:
            flow_category = FlowCategory.HEAVY_YES
        elif yes_ratio < self.HEAVY_NO_THRESHOLD:
            flow_category = FlowCategory.HEAVY_NO
        else:
            flow_category = FlowCategory.BALANCED

        # === LIQUIDITY CLASSIFICATION ===
        # Handle None values gracefully (markets without orderbook data)
        no_spread = ctx.no_spread if ctx.no_spread is not None else 50  # Default to wide
        yes_spread = ctx.yes_spread if ctx.yes_spread is not None else 50
        avg_spread = (no_spread + yes_spread) / 2
        if avg_spread <= self.TIGHT_SPREAD_MAX:
            liquidity_category = LiquidityCategory.TIGHT
        elif avg_spread <= self.NORMAL_SPREAD_MAX:
            liquidity_category = LiquidityCategory.NORMAL
        elif avg_spread <= self.WIDE_SPREAD_MAX:
            liquidity_category = LiquidityCategory.WIDE
        else:
            liquidity_category = LiquidityCategory.VERY_WIDE

        # === VOLATILITY CLASSIFICATION ===
        spread_volatility = ctx.spread_volatility if ctx.spread_volatility is not None else 0.0
        if spread_volatility < self.STABLE_VOLATILITY_MAX:
            volatility_category = VolatilityCategory.STABLE
        elif spread_volatility < self.MODERATE_VOLATILITY_MAX:
            volatility_category = VolatilityCategory.MODERATE
        else:
            volatility_category = VolatilityCategory.HIGH

        # === IMBALANCE CLASSIFICATION ===
        volume_imbalance = ctx.volume_imbalance if ctx.volume_imbalance is not None else 0.0
        if volume_imbalance > self.STRONG_IMBALANCE_THRESHOLD:
            imbalance_category = ImbalanceCategory.STRONG_BUY
        elif volume_imbalance > self.MILD_IMBALANCE_THRESHOLD:
            imbalance_category = ImbalanceCategory.MILD_BUY
        elif volume_imbalance < -self.STRONG_IMBALANCE_THRESHOLD:
            imbalance_category = ImbalanceCategory.STRONG_SELL
        elif volume_imbalance < -self.MILD_IMBALANCE_THRESHOLD:
            imbalance_category = ImbalanceCategory.MILD_SELL
        else:
            imbalance_category = ImbalanceCategory.BALANCED

        # === DATA QUALITY FLAGS ===
        # Handle None values for age comparisons
        trade_flow_age = ctx.trade_flow_age_seconds if ctx.trade_flow_age_seconds is not None else 999
        orderbook_age = ctx.orderbook_age_seconds if ctx.orderbook_age_seconds is not None else 999
        data_is_stale = (trade_flow_age > 120 or orderbook_age > 30)
        insufficient_trades = total_trades < self.MIN_TRADES_FOR_FLOW

        # Handle other potential None values for return
        price_drop = ctx.price_drop_from_open if ctx.price_drop_from_open is not None else 0
        last_yes_price = ctx.last_yes_price if ctx.last_yes_price is not None else 50

        return MicrostructureClassification(
            # Flow
            yes_trade_ratio=yes_ratio,
            flow_category=flow_category,
            total_trades=total_trades,
            # Liquidity
            avg_spread=avg_spread,
            liquidity_category=liquidity_category,
            no_spread=no_spread,
            yes_spread=yes_spread,
            # Volatility
            spread_volatility=spread_volatility,
            volatility_category=volatility_category,
            # Imbalance
            volume_imbalance=volume_imbalance,
            imbalance_category=imbalance_category,
            # Price
            price_change_from_open=price_drop,
            last_yes_price=last_yes_price,
            # Quality
            data_is_stale=data_is_stale,
            insufficient_trades=insufficient_trades,
        )

    def classify_to_prompt_string(self, ctx: 'MicrostructureContext') -> str:
        """
        Convenience method: classify and format in one call.

        Args:
            ctx: MicrostructureContext from state module

        Returns:
            Formatted string ready for LLM prompt
        """
        classification = self.classify(ctx)
        return classification.to_prompt_string()


# Type hint import (for runtime)
try:
    from ..state.microstructure_context import MicrostructureContext
except ImportError:
    # Allow module to load without the import for testing
    MicrostructureContext = None
