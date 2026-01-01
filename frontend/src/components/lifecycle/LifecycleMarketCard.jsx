import React, { memo, useRef, useEffect, useState } from 'react';

// ============================================================================
// HOOKS
// ============================================================================

/**
 * useAnimatedValue - Hook for tracking value changes and providing animation classes
 *
 * Returns animation class and key when value changes:
 * - Green glow + slight scale-up for increases
 * - Red glow + slight scale-down for decreases
 * - Uses key prop to re-trigger CSS animation
 *
 * @param {number} value - Current value to track
 * @param {boolean} enabled - Whether animations are enabled (default true)
 * @param {string} increaseClass - Custom class for increase animation
 * @param {string} decreaseClass - Custom class for decrease animation
 * @returns {{ animClass: string, animKey: number }} Animation class and unique key
 */
const useAnimatedValue = (value, enabled = true, increaseClass = 'animate-value-increase', decreaseClass = 'animate-value-decrease') => {
  const prevValueRef = useRef(value);
  const [animState, setAnimState] = useState({ class: '', key: 0 });

  useEffect(() => {
    if (!enabled) return;

    const prevValue = prevValueRef.current;

    // Only animate if we have a valid previous value and value actually changed
    if (prevValue !== null && prevValue !== undefined && value !== prevValue) {
      const isIncrease = value > prevValue;
      const animClass = isIncrease ? increaseClass : decreaseClass;

      // Update animation state with new key to force re-render
      setAnimState(prev => ({
        class: animClass,
        key: prev.key + 1
      }));

      // Clear animation class after animation completes
      const timer = setTimeout(() => {
        setAnimState(prev => ({ ...prev, class: '' }));
      }, 500);

      prevValueRef.current = value;
      return () => clearTimeout(timer);
    }

    prevValueRef.current = value;
  }, [value, enabled, increaseClass, decreaseClass]);

  return { animClass: animState.class, animKey: animState.key };
};

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

function formatTimeSince(ms) {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h`;
}

function formatTimeUntil(seconds) {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

function formatVolume(cents) {
  const dollars = Math.abs(cents) / 100;
  if (dollars >= 1000) return `$${(dollars / 1000).toFixed(1)}k`;
  return `$${dollars.toFixed(0)}`;
}

// ============================================================================
// CATEGORY STYLE SYSTEM
// ============================================================================

const categoryStyles = {
  sports: {
    bg: 'bg-gradient-to-r from-blue-600/30 to-blue-500/20',
    border: 'border-blue-500/40',
    text: 'text-blue-400',
    glow: 'shadow-blue-500/20'
  },
  crypto: {
    bg: 'bg-gradient-to-r from-orange-600/30 to-orange-500/20',
    border: 'border-orange-500/40',
    text: 'text-orange-400',
    glow: 'shadow-orange-500/20'
  },
  entertainment: {
    bg: 'bg-gradient-to-r from-purple-600/30 to-purple-500/20',
    border: 'border-purple-500/40',
    text: 'text-purple-400',
    glow: 'shadow-purple-500/20'
  },
  media_mentions: {
    bg: 'bg-gradient-to-r from-pink-600/30 to-pink-500/20',
    border: 'border-pink-500/40',
    text: 'text-pink-400',
    glow: 'shadow-pink-500/20'
  },
  politics: {
    bg: 'bg-gradient-to-r from-red-600/30 to-red-500/20',
    border: 'border-red-500/40',
    text: 'text-red-400',
    glow: 'shadow-red-500/20'
  },
  economics: {
    bg: 'bg-gradient-to-r from-green-600/30 to-green-500/20',
    border: 'border-green-500/40',
    text: 'text-green-400',
    glow: 'shadow-green-500/20'
  },
  climate: {
    bg: 'bg-gradient-to-r from-cyan-600/30 to-cyan-500/20',
    border: 'border-cyan-500/40',
    text: 'text-cyan-400',
    glow: 'shadow-cyan-500/20'
  },
  financials: {
    bg: 'bg-gradient-to-r from-emerald-600/30 to-emerald-500/20',
    border: 'border-emerald-500/40',
    text: 'text-emerald-400',
    glow: 'shadow-emerald-500/20'
  },
  science: {
    bg: 'bg-gradient-to-r from-indigo-600/30 to-indigo-500/20',
    border: 'border-indigo-500/40',
    text: 'text-indigo-400',
    glow: 'shadow-indigo-500/20'
  },
  world: {
    bg: 'bg-gradient-to-r from-teal-600/30 to-teal-500/20',
    border: 'border-teal-500/40',
    text: 'text-teal-400',
    glow: 'shadow-teal-500/20'
  },
  tech: {
    bg: 'bg-gradient-to-r from-violet-600/30 to-violet-500/20',
    border: 'border-violet-500/40',
    text: 'text-violet-400',
    glow: 'shadow-violet-500/20'
  },
  culture: {
    bg: 'bg-gradient-to-r from-fuchsia-600/30 to-fuchsia-500/20',
    border: 'border-fuchsia-500/40',
    text: 'text-fuchsia-400',
    glow: 'shadow-fuchsia-500/20'
  },
  default: {
    bg: 'bg-gradient-to-r from-gray-600/30 to-gray-500/20',
    border: 'border-gray-500/40',
    text: 'text-gray-400',
    glow: 'shadow-gray-500/20'
  }
};

const getCategoryStyle = (category) => {
  const key = (category || '').toLowerCase();
  return categoryStyles[key] || categoryStyles.default;
};

// ============================================================================
// MEMOIZED SUB-COMPONENTS
// ============================================================================

/**
 * CategoryBadge - Gradient pill with category-specific styling
 */
const CategoryBadge = memo(({ category }) => {
  const style = getCategoryStyle(category);

  return (
    <span className={`
      text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-md
      border backdrop-blur-sm
      font-medium
      ${style.bg} ${style.border} ${style.text}
    `}>
      {category || 'other'}
    </span>
  );
});

CategoryBadge.displayName = 'CategoryBadge';

/**
 * StatusBadges - Collection of NEW, HOT, SOON, DORMANT badges
 */
const StatusBadges = memo(({ isNew, isHot, closingSoon, signalReady, isDormant }) => (
  <div className="flex items-center gap-1.5">
    {isDormant && (
      <span className="
        text-[10px] px-2 py-0.5 rounded-md font-medium
        bg-gradient-to-r from-gray-600/30 to-gray-500/20
        border border-gray-500/40 text-gray-400
      ">
        DORMANT
      </span>
    )}
    {isNew && (
      <span className="
        text-[10px] px-2 py-0.5 rounded-md font-medium
        bg-gradient-to-r from-blue-600/30 to-blue-500/20
        border border-blue-500/40 text-blue-400
        shadow-sm shadow-blue-500/20
      ">
        NEW
      </span>
    )}
    {isHot && (
      <span className="
        text-[10px] px-2 py-0.5 rounded-md font-medium
        bg-gradient-to-r from-orange-600/30 to-orange-500/20
        border border-orange-500/40 text-orange-400
        shadow-sm shadow-orange-500/20
      ">
        HOT
      </span>
    )}
    {closingSoon && (
      <span className="
        text-[10px] px-2 py-0.5 rounded-md font-medium
        bg-gradient-to-r from-amber-600/30 to-amber-500/20
        border border-amber-500/40 text-amber-400
        shadow-sm shadow-amber-500/20
      ">
        SOON
      </span>
    )}
    {signalReady && (
      <span className="
        text-[10px] px-2 py-0.5 rounded-md font-bold
        bg-gradient-to-r from-amber-500/40 to-amber-400/30
        border border-amber-400/60 text-amber-300
        animate-signal-pulse
        shadow-md shadow-amber-500/30
      ">
        SIGNAL READY
      </span>
    )}
  </div>
));

StatusBadges.displayName = 'StatusBadges';

/**
 * PriceDisplay - Price, spread, and delta with animations
 */
const PriceDisplay = memo(({ midPrice, spread, priceDelta, animClass }) => (
  <div className="flex items-baseline justify-between gap-4">
    <div className="flex items-baseline gap-3">
      <span className={`
        text-xl font-mono font-bold text-white
        transition-all duration-300
        ${animClass}
      `}>
        {midPrice}<span className="text-gray-500 text-base">c</span>
      </span>
      {spread !== null && (
        <span className="text-xs text-gray-500 font-mono">
          {spread}c spread
        </span>
      )}
    </div>

    {priceDelta !== 0 && (
      <span className={`
        text-sm font-mono font-medium px-2 py-0.5 rounded-md
        ${priceDelta > 0
          ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
          : 'text-red-400 bg-red-500/10 border border-red-500/20'
        }
      `}>
        {priceDelta > 0 ? '+' : ''}{priceDelta}c
      </span>
    )}
  </div>
));

PriceDisplay.displayName = 'PriceDisplay';

/**
 * StatsRow - Volume and time displays
 */
const StatsRow = memo(({ volume, volume24h, volumeDelta, timeSinceTrack, timeUntilClose, closingSoon }) => (
  <div className="flex items-center justify-between text-xs text-gray-500">
    <div className="flex items-center gap-3">
      <span className="font-mono">
        Vol: {formatVolume(volume || 0)}
        {volume24h > 0 && (
          <span className="text-blue-400 ml-1.5">
            ({formatVolume(volume24h)})
          </span>
        )}
        {volumeDelta !== 0 && (
          <span className={`ml-1 ${volumeDelta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            ({volumeDelta > 0 ? '+' : ''}{formatVolume(volumeDelta)})
          </span>
        )}
      </span>
    </div>

    <div className="flex items-center gap-3">
      {timeSinceTrack && (
        <span className="flex items-center gap-1.5" title="Time since tracking started">
          <ClockIcon />
          <span className="font-mono">{timeSinceTrack}</span>
        </span>
      )}
      {timeUntilClose && (
        <span
          className={`flex items-center gap-1.5 ${closingSoon ? 'text-amber-400' : ''}`}
          title="Time until market closes"
        >
          <HourglassIcon />
          <span className="font-mono">{timeUntilClose}</span>
        </span>
      )}
    </div>
  </div>
));

StatsRow.displayName = 'StatsRow';

/**
 * TradeCountsDisplay - YES/NO trade counts with animations
 */
const TradeCountsDisplay = memo(({ yesTrades, noTrades, yesRatio, totalTrades, yesAnimClass, noAnimClass }) => (
  <div className="flex items-center justify-between text-xs">
    <span className="text-gray-400 font-medium">Trades:</span>
    <span className="font-mono flex items-center gap-2">
      <span className={`
        text-emerald-400 px-1.5 py-0.5 rounded
        bg-emerald-500/10 border border-emerald-500/20
        transition-all duration-300
        ${yesAnimClass}
      `}>
        {yesTrades} YES
      </span>
      <span className="text-gray-600">/</span>
      <span className={`
        text-red-400 px-1.5 py-0.5 rounded
        bg-red-500/10 border border-red-500/20
        transition-all duration-300
        ${noAnimClass}
      `}>
        {noTrades} NO
      </span>
      {totalTrades > 0 && (
        <span className="text-gray-500 font-medium ml-1">
          ({(yesRatio * 100).toFixed(0)}%)
        </span>
      )}
    </span>
  </div>
));

TradeCountsDisplay.displayName = 'TradeCountsDisplay';

/**
 * PriceMovementDisplay - First/last price tracking
 */
const PriceMovementDisplay = memo(({ firstYesPrice, lastYesPrice, priceDrop }) => {
  if (firstYesPrice === null || firstYesPrice === undefined) return null;

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-gray-400 font-medium">Price:</span>
      <span className="font-mono text-white flex items-center gap-1">
        <span className="text-gray-300">{firstYesPrice}c</span>
        {lastYesPrice !== null && lastYesPrice !== undefined && lastYesPrice !== firstYesPrice && (
          <>
            <span className="text-gray-600 mx-1">{'\u2192'}</span>
            <span className="text-white">{lastYesPrice}c</span>
            {priceDrop !== 0 && (
              <span className={`
                ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium
                ${priceDrop > 0
                  ? 'text-red-400 bg-red-500/10 border border-red-500/20'
                  : 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                }
              `}>
                {priceDrop > 0 ? '\u2193' : '\u2191'}{Math.abs(priceDrop)}c
              </span>
            )}
          </>
        )}
      </span>
    </div>
  );
});

PriceMovementDisplay.displayName = 'PriceMovementDisplay';

/**
 * ProgressBar - Animated progress toward signal threshold
 */
const ProgressBar = memo(({ totalTrades, signalReady, progressPercent, threshold, signalTriggerCount = 0 }) => {
  // Gradient and glow based on progress state
  const progressGradient = signalReady
    ? 'bg-gradient-to-r from-amber-500 to-amber-400'
    : progressPercent >= 80
      ? 'bg-gradient-to-r from-blue-500 to-cyan-400'
      : progressPercent >= 50
        ? 'bg-gradient-to-r from-blue-600 to-blue-500'
        : 'bg-gradient-to-r from-gray-600 to-gray-500';

  const progressGlow = signalReady
    ? 'shadow-[0_0_12px_rgba(245,158,11,0.5)]'
    : progressPercent >= 80
      ? 'shadow-[0_0_8px_rgba(59,130,246,0.4)]'
      : '';

  const tradesRemaining = threshold - totalTrades;

  return (
    <div className="relative">
      {/* Track */}
      <div className="h-2 bg-gray-800/80 rounded-full overflow-hidden border border-gray-700/50">
        {/* Fill */}
        <div
          className={`
            h-full rounded-full transition-all duration-500 ease-out
            ${progressGradient} ${progressGlow}
          `}
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between text-[10px] mt-1.5">
        <div className="flex items-center gap-2">
          <span className="text-gray-500 font-mono">{totalTrades} trades</span>
          {signalTriggerCount > 0 && (
            <span className="text-amber-400 font-mono">
              {signalTriggerCount}× triggered
            </span>
          )}
        </div>
        {signalReady ? (
          <span className="text-amber-400 font-bold animate-signal-text">
            SIGNAL READY
          </span>
        ) : tradesRemaining > 0 ? (
          <span className="text-gray-500 font-mono">
            {tradesRemaining} to signal
          </span>
        ) : (
          <span className="text-blue-400 font-mono">
            Threshold met
          </span>
        )}
      </div>
    </div>
  );
});

ProgressBar.displayName = 'ProgressBar';

/**
 * ExpandedDetails - Additional details shown when card is expanded
 */
const ExpandedDetails = memo(({ market, midPrice, priceDelta }) => (
  <div className="
    mt-3 pt-3
    border-t border-gray-700/50
    text-xs space-y-2
    animate-expand-in
  ">
    {/* Bid/Ask */}
    <div className="flex justify-between text-gray-400">
      <span>Bid / Ask:</span>
      <span className="font-mono text-white">
        {market.yes_bid || '-'}c / {market.yes_ask || '-'}c
      </span>
    </div>

    {/* Open Interest */}
    {market.open_interest !== undefined && (
      <div className="flex justify-between text-gray-400">
        <span>Open Interest:</span>
        <span className="font-mono text-white">
          {market.open_interest?.toLocaleString() || '-'}
        </span>
      </div>
    )}

    {/* Price at discovery */}
    {market.price_at_track !== undefined && (
      <div className="flex justify-between text-gray-400">
        <span>Price at discovery:</span>
        <span className="font-mono text-white">
          {market.price_at_track}c {'\u2192'} {midPrice}c
          {priceDelta !== 0 && (
            <span className={`ml-1 ${priceDelta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              ({priceDelta > 0 ? '+' : ''}{priceDelta}c)
            </span>
          )}
        </span>
      </div>
    )}

    {/* Status */}
    <div className="flex justify-between text-gray-400">
      <span>Status:</span>
      <span className={`
        font-medium px-2 py-0.5 rounded
        ${market.status === 'determined'
          ? 'text-amber-400 bg-amber-500/10 border border-amber-500/20'
          : 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
        }
      `}>
        {market.status || 'active'}
      </span>
    </div>
  </div>
));

ExpandedDetails.displayName = 'ExpandedDetails';

/**
 * TradingStateDisplay - Shows our trading activity in this market
 *
 * Displays:
 * - Active/recent orders with status badges
 * - Current position with P&L
 * - Settlement with final P&L
 */
const TradingStateDisplay = memo(({ trading, market }) => {
  if (!trading) return null;

  const { orders = [], position, settlement, trading_state } = trading;

  // Filter to active orders
  const activeOrders = orders.filter(o => ['pending', 'resting', 'partial'].includes(o.status));

  // Format cents to dollars
  const formatPnL = (cents) => {
    const dollars = cents / 100;
    const sign = dollars >= 0 ? '+' : '';
    return `${sign}$${Math.abs(dollars).toFixed(2)}`;
  };

  // Calculate unrealized P&L using live market prices
  const calcUnrealizedPnL = (pos, mkt) => {
    if (!pos || pos.count === 0) return pos?.unrealized_pnl || 0;

    const qty = pos.count;
    const totalCost = pos.total_cost || (qty * pos.avg_entry_price);

    // Get current YES price from market (prefer bid/ask midpoint, then price)
    const currentYesPrice = mkt?.yes_bid && mkt?.yes_ask
      ? Math.round((mkt.yes_bid + mkt.yes_ask) / 2)
      : mkt?.price || 0;

    // If no price data, fall back to backend's calculation
    if (!currentYesPrice) return pos.unrealized_pnl || 0;

    // Calculate value based on side (NO = inverse of YES price)
    const valuePerContract = pos.side === 'no'
      ? (100 - currentYesPrice)
      : currentYesPrice;

    const totalValue = valuePerContract * qty;
    return totalValue - totalCost;  // Returns cents
  };

  // Nothing to show
  if (!position && !settlement && activeOrders.length === 0) {
    return null;
  }

  return (
    <div className="
      mt-3 pt-2.5
      border-t border-gray-700/40
      space-y-2
    ">
      {/* Section header */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[10px] uppercase tracking-wide text-cyan-400 font-medium">
          Trading
        </span>
        <span className={`
          text-[9px] uppercase px-1.5 py-0.5 rounded font-medium
          ${trading_state === 'position_open'
            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
            : trading_state === 'settled'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              : trading_state === 'order_resting'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                : trading_state === 'order_pending'
                  ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40 animate-pulse'
                  : 'bg-gray-500/20 text-gray-400 border border-gray-500/40'
          }
        `}>
          {trading_state?.replace('_', ' ') || 'monitoring'}
        </span>
      </div>

      {/* Active Orders */}
      {activeOrders.length > 0 && (
        <div className="space-y-1.5">
          {activeOrders.map((order) => (
            <div
              key={order.order_id}
              className="
                flex items-center justify-between
                text-xs px-2 py-1 rounded
                bg-gray-800/50 border border-gray-700/40
              "
            >
              <div className="flex items-center gap-2">
                <span className={`
                  px-1.5 py-0.5 rounded text-[10px] font-bold uppercase
                  ${order.action === 'buy'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }
                `}>
                  {order.action}
                </span>
                <span className="text-gray-300 font-mono">
                  {order.count} {order.side?.toUpperCase()}
                </span>
                <span className="text-gray-500">@</span>
                <span className="text-white font-mono">{order.price}c</span>
              </div>
              <span className={`
                text-[10px] px-1.5 py-0.5 rounded font-medium
                ${order.status === 'pending'
                  ? 'bg-yellow-500/20 text-yellow-400 animate-pulse'
                  : order.status === 'resting'
                    ? 'bg-blue-500/20 text-blue-400'
                    : order.status === 'partial'
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : order.status === 'filled'
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : 'bg-gray-500/20 text-gray-400'
                }
              `}>
                {order.status === 'partial' ? `${order.fill_count}/${order.count}` : order.status}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Current Position */}
      {position && position.count > 0 && (
        <div className={`
          flex items-center justify-between
          text-xs px-2 py-1.5 rounded-md
          ${calcUnrealizedPnL(position, market) >= 0
            ? 'bg-emerald-900/20 border border-emerald-500/25'
            : 'bg-red-900/20 border border-red-500/25'
          }
        `}>
          <div className="flex items-center gap-1.5">
            <span className="text-gray-400 font-medium">POS:</span>
            <span className={`
              font-mono font-medium
              ${position.side === 'yes' ? 'text-emerald-400' : 'text-red-400'}
            `}>
              {position.count} {position.side?.toUpperCase()}
            </span>
            <span className="text-gray-600">@</span>
            <span className="text-gray-400 font-mono">{position.avg_entry_price}c</span>
          </div>
          <span className={`
            font-mono font-semibold
            ${calcUnrealizedPnL(position, market) >= 0 ? 'text-emerald-400' : 'text-red-400'}
          `}>
            {formatPnL(calcUnrealizedPnL(position, market))}
          </span>
        </div>
      )}

      {/* Settlement */}
      {settlement && (
        <div className={`
          flex items-center justify-between
          text-xs px-2 py-1.5 rounded-md
          ${settlement.final_pnl >= 0
            ? 'bg-emerald-900/25 border border-emerald-500/30'
            : 'bg-red-900/25 border border-red-500/30'
          }
        `}>
          <div className="flex items-center gap-1.5">
            <span className={`
              px-1.5 py-0.5 rounded font-bold text-[10px] uppercase
              ${settlement.final_pnl >= 0
                ? 'bg-emerald-500/25 text-emerald-300'
                : 'bg-red-500/25 text-red-300'
              }
            `}>
              {settlement.final_pnl >= 0 ? 'WIN' : 'LOSS'}
            </span>
            <span className="text-gray-500">
              Settled <span className="text-gray-300 font-medium uppercase">{settlement.result}</span>
            </span>
          </div>
          <span className={`
            font-mono font-bold
            ${settlement.final_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}
          `}>
            {formatPnL(settlement.final_pnl)}
          </span>
        </div>
      )}
    </div>
  );
});

TradingStateDisplay.displayName = 'TradingStateDisplay';

// ============================================================================
// ICON COMPONENTS
// ============================================================================

const ClockIcon = memo(() => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
));

ClockIcon.displayName = 'ClockIcon';

const HourglassIcon = memo(() => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M12 8V4m0 16v-4m-6-4h12M7 4h10v2a4 4 0 01-4 4h-2a4 4 0 01-4-4V4zm0 16h10v-2a4 4 0 00-4-4h-2a4 4 0 00-4 4v2z" />
  </svg>
));

HourglassIcon.displayName = 'HourglassIcon';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

/**
 * LifecycleMarketCard - Card displaying market with deltas since tracking
 *
 * Features:
 * - Gradient backgrounds with backdrop blur
 * - Value-change animations (green/red glows on trade count changes)
 * - Hover micro-interactions (scale, shadow)
 * - Signal-ready amber glow state
 * - Progress bar with animated fill
 *
 * Props:
 *   - market: Market data object
 *   - rlmState: RLM state { yes_trades, no_trades, yes_ratio, price_drop, etc. }
 *   - tradePulse: Trade pulse { side: 'yes'|'no', ts: timestamp } for animation
 *   - rlmConfig: RLM strategy config from backend { min_trades, yes_threshold, min_price_drop }
 */
const LifecycleMarketCard = ({ market, rlmState, tradePulse, rlmConfig }) => {
  const [expanded, setExpanded] = useState(false);

  // RLM stats from prop (or defaults if not available)
  const yesTrades = rlmState?.yes_trades || 0;
  const noTrades = rlmState?.no_trades || 0;
  const totalTrades = rlmState?.total_trades || (yesTrades + noTrades);
  const yesRatio = rlmState?.yes_ratio || (totalTrades > 0 ? yesTrades / totalTrades : 0);
  const firstYesPrice = rlmState?.first_yes_price;
  const lastYesPrice = rlmState?.last_yes_price;
  const priceDrop = rlmState?.price_drop || 0;
  const signalTriggerCount = rlmState?.signal_trigger_count || 0;

  // RLM strategy thresholds from backend config (with sensible fallbacks)
  const minTrades = rlmConfig?.min_trades || 25;
  const yesThreshold = rlmConfig?.yes_threshold || 0.70;
  const minPriceDrop = rlmConfig?.min_price_drop || 2;

  // Progress bar toward signal threshold
  const progressPercent = Math.min(100, (totalTrades / minTrades) * 100);
  const signalReady = totalTrades >= minTrades && yesRatio >= yesThreshold && priceDrop >= minPriceDrop;

  // Animated values for trade counts
  const { animClass: yesAnimClass } = useAnimatedValue(
    yesTrades,
    true,
    'animate-trade-yes-pulse',
    'animate-trade-yes-pulse'
  );
  const { animClass: noAnimClass } = useAnimatedValue(
    noTrades,
    true,
    'animate-trade-no-pulse',
    'animate-trade-no-pulse'
  );
  const { animClass: priceAnimClass } = useAnimatedValue(
    market.price || 0,
    true,
    'animate-price-change',
    'animate-price-change'
  );

  // Calculate derived values
  const midPrice = market.yes_bid && market.yes_ask
    ? Math.round((market.yes_bid + market.yes_ask) / 2)
    : market.price || 0;

  const spread = market.yes_bid && market.yes_ask
    ? market.yes_ask - market.yes_bid
    : null;

  const priceDelta = market.price_delta || 0;
  const volumeDelta = market.volume_delta || 0;

  // Time since tracking
  const trackedAt = market.tracked_at;
  const timeSinceTrack = trackedAt
    ? formatTimeSince(Date.now() - trackedAt * 1000)
    : null;

  // Is this a "new" market (tracked < 5 minutes)?
  const isNew = trackedAt && (Date.now() - trackedAt * 1000) < 5 * 60 * 1000;

  // Is this "hot" (high volume delta)?
  const isHot = Math.abs(volumeDelta) > 50000;

  // Time until close
  const timeToClose = market.time_to_close_seconds
    ?? (market.close_ts ? market.close_ts - Math.floor(Date.now() / 1000) : null);
  const timeUntilClose = timeToClose && timeToClose > 0
    ? formatTimeUntil(timeToClose)
    : null;

  // Is this closing soon (< 1 hour)?
  const closingSoon = timeToClose && timeToClose < 3600;

  // Is this market dormant (no 24h volume)?
  const isDormant = (market.volume_24h || 0) === 0;

  // Determine card border/glow state
  const isDetermined = market.status === 'determined';

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={`
        bg-gradient-to-br from-gray-800/70 via-gray-850/60 to-gray-900/70
        backdrop-blur-sm rounded-xl
        border
        shadow-lg shadow-black/10
        transition-all duration-300 ease-out
        cursor-pointer
        p-4
        ${isDormant
          ? 'opacity-50 saturate-50 border-gray-700/40'
          : signalReady
            ? 'border-amber-500/60 shadow-xl shadow-amber-500/20 animate-signal-glow'
            : isDetermined
              ? 'border-amber-500/40 bg-amber-900/10'
              : isNew
                ? 'border-blue-500/40 ring-1 ring-blue-500/30'
                : 'border-gray-700/60'
        }
        hover:scale-[1.02] hover:shadow-xl hover:border-gray-600/80
      `}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          {/* Category badge + ticker */}
          <div className="flex items-center gap-2 mb-1.5">
            <CategoryBadge category={market.category} />
            <span className="text-xs text-gray-500 font-mono truncate">
              {market.ticker}
            </span>
          </div>

          {/* Event title */}
          <h3 className="text-sm text-white font-medium truncate leading-tight">
            {market.event_title || market.title || market.ticker}
          </h3>
        </div>

        {/* Status badges */}
        <StatusBadges
          isDormant={isDormant}
          isNew={isNew}
          isHot={isHot}
          closingSoon={closingSoon}
          signalReady={signalReady}
        />
      </div>

      {/* Main stats row */}
      <PriceDisplay
        midPrice={midPrice}
        spread={spread}
        priceDelta={priceDelta}
        animClass={priceAnimClass}
      />

      {/* Secondary stats row */}
      <div className="mt-3">
        <StatsRow
          volume={market.volume}
          volume24h={market.volume_24h}
          volumeDelta={volumeDelta}
          timeSinceTrack={timeSinceTrack}
          timeUntilClose={timeUntilClose}
          closingSoon={closingSoon}
        />
      </div>

      {/* RLM Stats Section */}
      <div className="
        mt-4 pt-3
        border-t border-gray-700/40
        space-y-3
      ">
        {/* Trade counts and ratio */}
        <TradeCountsDisplay
          yesTrades={yesTrades}
          noTrades={noTrades}
          yesRatio={yesRatio}
          totalTrades={totalTrades}
          yesAnimClass={yesAnimClass}
          noAnimClass={noAnimClass}
        />

        {/* Price movement */}
        <PriceMovementDisplay
          firstYesPrice={firstYesPrice}
          lastYesPrice={lastYesPrice}
          priceDrop={priceDrop}
        />

        {/* Progress bar */}
        <ProgressBar
          totalTrades={totalTrades}
          signalReady={signalReady}
          progressPercent={progressPercent}
          threshold={minTrades}
          signalTriggerCount={signalTriggerCount}
        />
      </div>

      {/* Trading State - Orders, Positions, P&L */}
      <TradingStateDisplay trading={market.trading} market={market} />

      {/* Expanded details */}
      {expanded && (
        <ExpandedDetails
          market={market}
          midPrice={midPrice}
          priceDelta={priceDelta}
        />
      )}

      {/* Animation Keyframes */}
      <style>{`
        @keyframes trade-yes-pulse {
          0% { box-shadow: none; }
          50% { box-shadow: 0 0 12px rgba(34, 197, 94, 0.5); }
          100% { box-shadow: none; }
        }

        @keyframes trade-no-pulse {
          0% { box-shadow: none; }
          50% { box-shadow: 0 0 12px rgba(239, 68, 68, 0.5); }
          100% { box-shadow: none; }
        }

        @keyframes price-change {
          0% { text-shadow: none; }
          50% { text-shadow: 0 0 8px rgba(255, 255, 255, 0.5); }
          100% { text-shadow: none; }
        }

        @keyframes signal-glow {
          0%, 100% { box-shadow: 0 0 15px rgba(245, 158, 11, 0.3), 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
          50% { box-shadow: 0 0 25px rgba(245, 158, 11, 0.5), 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
        }

        @keyframes signal-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.9; transform: scale(1.02); }
        }

        @keyframes signal-text {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }

        @keyframes value-increase {
          0% { transform: scale(1); box-shadow: none; }
          50% { transform: scale(1.02); box-shadow: 0 0 12px rgba(34, 197, 94, 0.4); }
          100% { transform: scale(1); box-shadow: none; }
        }

        @keyframes value-decrease {
          0% { transform: scale(1); box-shadow: none; }
          50% { transform: scale(0.98); box-shadow: 0 0 12px rgba(239, 68, 68, 0.4); }
          100% { transform: scale(1); box-shadow: none; }
        }

        @keyframes expand-in {
          0% { opacity: 0; transform: translateY(-8px); }
          100% { opacity: 1; transform: translateY(0); }
        }

        .animate-trade-yes-pulse {
          animation: trade-yes-pulse 450ms ease-out;
        }

        .animate-trade-no-pulse {
          animation: trade-no-pulse 450ms ease-out;
        }

        .animate-price-change {
          animation: price-change 400ms ease-out;
        }

        .animate-signal-glow {
          animation: signal-glow 2s ease-in-out infinite;
        }

        .animate-signal-pulse {
          animation: signal-pulse 1.5s ease-in-out infinite;
        }

        .animate-signal-text {
          animation: signal-text 1.5s ease-in-out infinite;
        }

        .animate-value-increase {
          animation: value-increase 450ms ease-out;
        }

        .animate-value-decrease {
          animation: value-decrease 450ms ease-out;
        }

        .animate-expand-in {
          animation: expand-in 200ms ease-out;
        }
      `}</style>
    </div>
  );
};

export default memo(LifecycleMarketCard);
