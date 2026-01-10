"""
Core data structures for point-in-time backtesting.

These dataclasses maintain state that reflects ONLY what is known at a given
point in time, avoiding look-ahead bias.

Key Principle:
    When a strategy evaluates whether to enter a position, it should only
    have access to information that would have been available at that moment
    in real-time trading. This means:

    - Trade counts reflect only trades BEFORE the current moment
    - Prices reflect the current market state, not final settlement
    - Settlement results are loaded separately (for P&L calculation only)

Example Flow:
    1. Trade arrives at time T
    2. MarketState.update(trade) is called - state now reflects post-trade reality
    3. Strategy.on_trade(trade, state) evaluates if signal fires
    4. If signal fires, SignalEntry captures the entry_price at time T
    5. Later, settlement data determines if the signal was profitable
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Trade:
    """
    Normalized trade record from Kalshi public trade feed.

    This is the atomic unit of market activity. Each trade represents
    a matched order between a buyer and seller.

    Attributes:
        market_ticker: Unique market identifier (e.g., 'INXD-25JAN03-B4980')
        timestamp: When the trade occurred (timezone-aware UTC)
        yes_price: Price of YES contract in cents (1-99)
        taker_side: Which side the taker (aggressor) took: 'yes' or 'no'
        count: Number of contracts traded
        trade_id: Optional unique trade identifier

    Note on taker_side:
        - 'yes' means someone bought YES (hit the ask) - bullish signal
        - 'no' means someone bought NO (hit the bid) - bearish signal
        The taker pays the spread, so their direction indicates conviction.
    """
    market_ticker: str
    timestamp: datetime
    yes_price: int          # Price in cents (1-99)
    taker_side: str         # 'yes' or 'no'
    count: int              # Number of contracts
    trade_id: Optional[str] = None

    def __post_init__(self):
        """Validate trade data."""
        if self.yes_price < 1 or self.yes_price > 99:
            raise ValueError(f"yes_price must be 1-99, got {self.yes_price}")
        if self.taker_side not in ('yes', 'no'):
            raise ValueError(f"taker_side must be 'yes' or 'no', got {self.taker_side}")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")


@dataclass
class SignalEntry:
    """
    Captured at the moment a signal fires.

    This is the key anti-look-ahead structure: it records the state of
    the market AT THE TIME the signal fired, not what happened later.

    Attributes:
        market_ticker: Which market the signal is for
        signal_time: When the signal fired (timestamp of triggering trade)
        entry_price_cents: YES price at signal moment (what we'd pay to enter)
        side: Which side we're betting on: 'yes' or 'no'
        signal_strength: Confidence metric from strategy (0.0 to 1.0)
        metadata: Strategy-specific data (for analysis/debugging)

    P&L Calculation (done later with settlement data):
        If side == 'yes' and market settles YES:
            profit = 100 - entry_price_cents (we paid X, received 100)
        If side == 'yes' and market settles NO:
            profit = -entry_price_cents (we paid X, received 0)
        If side == 'no' and market settles NO:
            profit = 100 - (100 - entry_price_cents) = entry_price_cents
            (we paid 100-X for NO, received 100)
        If side == 'no' and market settles YES:
            profit = -(100 - entry_price_cents) (we paid 100-X, received 0)
    """
    market_ticker: str
    signal_time: datetime
    entry_price_cents: int      # Price at signal moment (NOT final price)
    side: str                   # 'yes' or 'no'
    signal_strength: float      # Confidence metric (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if self.entry_price_cents < 1 or self.entry_price_cents > 99:
            raise ValueError(f"entry_price_cents must be 1-99, got {self.entry_price_cents}")
        if self.side not in ('yes', 'no'):
            raise ValueError(f"side must be 'yes' or 'no', got {self.side}")
        if self.signal_strength < 0.0 or self.signal_strength > 1.0:
            raise ValueError(f"signal_strength must be 0.0-1.0, got {self.signal_strength}")

    @property
    def effective_entry_price(self) -> int:
        """
        Price we actually pay to enter the position.

        - For YES bets: we pay yes_price
        - For NO bets: we pay (100 - yes_price)
        """
        if self.side == 'yes':
            return self.entry_price_cents
        else:
            return 100 - self.entry_price_cents


@dataclass
class MarketState:
    """
    Per-market state, updated on each trade.

    This tracks the cumulative state of a market as we process trades
    chronologically. At any point in time, this represents ONLY what
    is known from trades processed so far.

    Attributes:
        market_ticker: Unique market identifier

        Trade Tracking:
            yes_trades: Count of trades where taker bought YES
            no_trades: Count of trades where taker bought NO
            total_contracts: Sum of all contracts traded
            yes_contracts: Contracts where taker bought YES
            no_contracts: Contracts where taker bought NO

        Price Tracking (point-in-time):
            first_yes_price: First YES price we saw
            last_yes_price: Most recent YES price (current market price)
            open_price: Opening price (first trade or TMO price if available)

        Time Tracking:
            first_trade_time: When first trade occurred
            last_trade_time: Most recent trade time

        Settlement (loaded separately, NOT from trades):
            settlement_result: 'yes' or 'no' after market closes
            close_time: When market closed

        metadata: Extensible dict for strategy-specific tracking

    Key Properties:
        total_trades: yes_trades + no_trades
        yes_ratio: Fraction of trades that were YES buys (sentiment indicator)
        price_drop: How much price fell from open (positive = bearish)
    """
    market_ticker: str

    # Core trade tracking
    yes_trades: int = 0
    no_trades: int = 0
    total_contracts: int = 0
    yes_contracts: int = 0
    no_contracts: int = 0

    # Price tracking (point-in-time)
    first_yes_price: Optional[int] = None
    last_yes_price: Optional[int] = None
    open_price: Optional[int] = None  # Can be first trade or TMO

    # Time tracking
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    # Settlement data (loaded separately, not from trades)
    settlement_result: Optional[str] = None  # 'yes' or 'no'
    close_time: Optional[datetime] = None

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        """Total number of trades processed."""
        return self.yes_trades + self.no_trades

    @property
    def yes_ratio(self) -> float:
        """
        Fraction of trades where taker bought YES.

        This is a sentiment indicator:
        - High yes_ratio (>0.5): More aggressive buying of YES (bullish flow)
        - Low yes_ratio (<0.5): More aggressive buying of NO (bearish flow)

        Returns 0.0 if no trades yet.
        """
        if self.total_trades == 0:
            return 0.0
        return self.yes_trades / self.total_trades

    @property
    def no_ratio(self) -> float:
        """Fraction of trades where taker bought NO."""
        return 1.0 - self.yes_ratio

    @property
    def price_drop(self) -> int:
        """
        Price drop from open to current (positive = price went down).

        This measures bearish pressure:
        - Positive value: Price has fallen (bearish)
        - Negative value: Price has risen (bullish)
        - Zero: No change or insufficient data

        Returns 0 if open_price or last_yes_price not available.
        """
        if self.open_price is None or self.last_yes_price is None:
            return 0
        return self.open_price - self.last_yes_price

    @property
    def price_change_pct(self) -> float:
        """
        Percentage price change from open.

        Returns:
            Percentage change (e.g., -10.5 means price fell 10.5%)
            Returns 0.0 if insufficient data.
        """
        if self.open_price is None or self.open_price == 0 or self.last_yes_price is None:
            return 0.0
        return ((self.last_yes_price - self.open_price) / self.open_price) * 100

    @property
    def volume_imbalance(self) -> float:
        """
        Contract volume imbalance (positive = more YES volume).

        Returns value from -1 (all NO) to +1 (all YES).
        Returns 0.0 if no contracts traded.
        """
        if self.total_contracts == 0:
            return 0.0
        return (self.yes_contracts - self.no_contracts) / self.total_contracts

    def update(self, trade: Trade) -> None:
        """
        Update state with a new trade.

        This is called BEFORE the strategy evaluates the trade, so the
        state reflects the market AFTER this trade occurred.

        Args:
            trade: The trade to process

        Note:
            We track prices from YES trades because they directly indicate
            the market's YES price. NO trades are at (100 - yes_price) but
            we store the yes_price for consistency.
        """
        # Track trade counts by taker side
        if trade.taker_side == 'yes':
            self.yes_trades += 1
            self.yes_contracts += trade.count
        else:
            self.no_trades += 1
            self.no_contracts += trade.count
        self.total_contracts += trade.count

        # Track YES price (all trades have yes_price, regardless of taker_side)
        if self.first_yes_price is None:
            self.first_yes_price = trade.yes_price
        self.last_yes_price = trade.yes_price

        # Set open price from first trade if not already set (e.g., from TMO)
        if self.open_price is None:
            self.open_price = trade.yes_price

        # Track times
        if self.first_trade_time is None:
            self.first_trade_time = trade.timestamp
        self.last_trade_time = trade.timestamp

    def set_open_price(self, price: int) -> None:
        """
        Set open price explicitly (e.g., from TMO data).

        This should be called BEFORE processing trades if you have
        TMO (theoretical market open) data available.

        Args:
            price: Opening price in cents (1-99)
        """
        if price < 1 or price > 99:
            raise ValueError(f"price must be 1-99, got {price}")
        self.open_price = price

    def set_settlement(self, result: str, close_time: Optional[datetime] = None) -> None:
        """
        Set settlement result (called after market closes).

        This is NOT used for signal generation - only for P&L calculation
        after the backtest is complete.

        Args:
            result: 'yes' or 'no'
            close_time: Optional timestamp when market closed
        """
        if result not in ('yes', 'no'):
            raise ValueError(f"result must be 'yes' or 'no', got {result}")
        self.settlement_result = result
        self.close_time = close_time

    def copy(self) -> 'MarketState':
        """Create a copy of the current state (useful for snapshots)."""
        return MarketState(
            market_ticker=self.market_ticker,
            yes_trades=self.yes_trades,
            no_trades=self.no_trades,
            total_contracts=self.total_contracts,
            yes_contracts=self.yes_contracts,
            no_contracts=self.no_contracts,
            first_yes_price=self.first_yes_price,
            last_yes_price=self.last_yes_price,
            open_price=self.open_price,
            first_trade_time=self.first_trade_time,
            last_trade_time=self.last_trade_time,
            settlement_result=self.settlement_result,
            close_time=self.close_time,
            metadata=self.metadata.copy()
        )


@dataclass
class JourneySignalEntry:
    """
    Signal entry for journey-style strategies (profit from price movement, not settlement).

    Unlike SignalEntry which calculates P&L from settlement, JourneySignalEntry
    tracks entry and exit within the market's lifetime based on price movement.

    Journey P&L = exit_price - entry_price (for YES positions)
    Journey P&L = entry_price - exit_price (for NO positions, since NO price moves inversely)

    Key Differences from SignalEntry:
        - P&L is calculated from price movement, NOT settlement outcome
        - Exit is triggered by target, timeout, or stop loss
        - Must track position state through subsequent trades

    Attributes:
        market_ticker: Which market the signal is for
        signal_time: When the signal fired (entry time)
        entry_price_cents: YES price at entry (what we'd pay to enter YES)
        side: Which side we're betting on: 'yes' or 'no'
        signal_strength: Confidence metric from strategy (0.0 to 1.0)

        # Exit tracking (populated when position closes)
        exit_time: When the position was closed
        exit_price_cents: YES price at exit
        exit_reason: Why we exited ('target', 'timeout', 'stop_loss')
        trades_held: Number of trades we held the position through
        pnl_cents: Profit/loss in cents (exit_price - entry_price for YES)

        metadata: Strategy-specific data (for analysis/debugging)
    """
    market_ticker: str
    signal_time: datetime
    entry_price_cents: int      # YES price at entry
    side: str                   # 'yes' or 'no'
    signal_strength: float      # Confidence metric (0.0 to 1.0)

    # Exit tracking (None until position is closed)
    exit_time: Optional[datetime] = None
    exit_price_cents: Optional[int] = None
    exit_reason: Optional[str] = None  # 'target', 'timeout', 'stop_loss', 'market_end'
    trades_held: int = 0
    pnl_cents: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if self.entry_price_cents < 1 or self.entry_price_cents > 99:
            raise ValueError(f"entry_price_cents must be 1-99, got {self.entry_price_cents}")
        if self.side not in ('yes', 'no'):
            raise ValueError(f"side must be 'yes' or 'no', got {self.side}")
        if self.signal_strength < 0.0 or self.signal_strength > 1.0:
            raise ValueError(f"signal_strength must be 0.0-1.0, got {self.signal_strength}")

    @property
    def is_closed(self) -> bool:
        """Check if the position has been closed."""
        return self.exit_time is not None

    @property
    def is_winner(self) -> bool:
        """Check if the position was profitable."""
        if self.pnl_cents is None:
            return False
        return self.pnl_cents > 0

    def close(
        self,
        exit_time: datetime,
        exit_price_cents: int,
        exit_reason: str,
        trades_held: int
    ) -> None:
        """
        Close the position and calculate P&L.

        For YES positions: P&L = exit_price - entry_price
            (We bought at entry, sell at exit)

        For NO positions: P&L = (100 - exit_price) - (100 - entry_price) = entry_price - exit_price
            (NO price is inverse of YES price)
        """
        self.exit_time = exit_time
        self.exit_price_cents = exit_price_cents
        self.exit_reason = exit_reason
        self.trades_held = trades_held

        # Calculate P&L based on position side
        if self.side == 'yes':
            # Bought YES at entry_price, sold at exit_price
            self.pnl_cents = exit_price_cents - self.entry_price_cents
        else:
            # Bought NO at (100 - entry_price), sold at (100 - exit_price)
            # P&L = (100 - exit_price) - (100 - entry_price) = entry_price - exit_price
            self.pnl_cents = self.entry_price_cents - exit_price_cents
