import React, { useState } from 'react';

/**
 * LifecycleMarketCard - Card displaying market with deltas since tracking
 *
 * Key insight: Everything anchored to "since you started tracking"
 *
 * Props:
 *   - market: Market data object
 *   - rlmState: RLM state { yes_trades, no_trades, yes_ratio, price_drop, etc. }
 *   - tradePulse: Trade pulse { side: 'yes'|'no', ts: timestamp } for animation
 */
const LifecycleMarketCard = ({ market, rlmState, tradePulse }) => {
  const [expanded, setExpanded] = useState(false);

  // RLM stats from prop (or defaults if not available)
  const yesTrades = rlmState?.yes_trades || 0;
  const noTrades = rlmState?.no_trades || 0;
  const totalTrades = rlmState?.total_trades || (yesTrades + noTrades);
  const yesRatio = rlmState?.yes_ratio || (totalTrades > 0 ? yesTrades / totalTrades : 0);
  const firstYesPrice = rlmState?.first_yes_price;
  const lastYesPrice = rlmState?.last_yes_price;
  const priceDrop = rlmState?.price_drop || 0;

  // Progress bar: 15 trades is the RLM signal threshold
  const SIGNAL_THRESHOLD = 15;
  const progressPercent = Math.min(100, (totalTrades / SIGNAL_THRESHOLD) * 100);
  const signalReady = totalTrades >= SIGNAL_THRESHOLD && yesRatio > 0.65 && priceDrop > 0;

  // Pulse animation class based on trade side
  const pulseClass = tradePulse
    ? tradePulse.side === 'yes'
      ? 'animate-pulse-green'
      : 'animate-pulse-red'
    : '';

  // Calculate derived values
  const midPrice = market.yes_bid && market.yes_ask
    ? Math.round((market.yes_bid + market.yes_ask) / 2)
    : market.price || 0;

  const spread = market.yes_bid && market.yes_ask
    ? market.yes_ask - market.yes_bid
    : null;

  const priceDelta = market.price_delta || 0;
  const volumeDelta = market.volume_delta || 0;

  // Time since tracking (tracked_at is Unix seconds, Date.now() is milliseconds)
  const trackedAt = market.tracked_at;
  const timeSinceTrack = trackedAt
    ? formatTimeSince(Date.now() - trackedAt * 1000)
    : null;

  // Is this a "new" market (tracked < 5 minutes)?
  const isNew = trackedAt && (Date.now() - trackedAt * 1000) < 5 * 60 * 1000;

  // Is this "hot" (high volume delta)?
  const isHot = Math.abs(volumeDelta) > 50000; // $500+ volume delta

  // Time until close (from backend time_to_close_seconds or calculated from close_ts)
  const timeToClose = market.time_to_close_seconds
    ?? (market.close_ts ? market.close_ts - Math.floor(Date.now() / 1000) : null);
  const timeUntilClose = timeToClose && timeToClose > 0
    ? formatTimeUntil(timeToClose)
    : null;

  // Is this closing soon (< 1 hour)?
  const closingSoon = timeToClose && timeToClose < 3600;

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={`
        bg-gray-800/60 rounded-lg p-3 cursor-pointer
        border transition-all duration-200
        ${market.status === 'determined' ? 'border-amber-500/50 bg-amber-900/10' : 'border-gray-700'}
        ${isNew ? 'ring-1 ring-blue-500/50 animate-pulse-subtle' : ''}
        ${pulseClass}
        hover:border-gray-600 hover:bg-gray-800/80
      `}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex-1 min-w-0">
          {/* Category badge + ticker */}
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded ${
              getCategoryStyle(market.category)
            }`}>
              {market.category || 'other'}
            </span>
            <span className="text-xs text-gray-500 font-mono truncate">
              {market.ticker}
            </span>
          </div>

          {/* Event title */}
          <h3 className="text-sm text-white font-medium truncate">
            {market.event_title || market.title || market.ticker}
          </h3>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-1">
          {isNew && (
            <span className="text-[10px] bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded font-medium">
              NEW
            </span>
          )}
          {isHot && (
            <span className="text-[10px] bg-orange-500/20 text-orange-400 px-1.5 py-0.5 rounded">
              HOT
            </span>
          )}
          {closingSoon && (
            <span className="text-[10px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded">
              SOON
            </span>
          )}
        </div>
      </div>

      {/* Main stats row */}
      <div className="flex items-baseline justify-between gap-4">
        {/* Price + spread */}
        <div className="flex items-baseline gap-3">
          <span className="text-lg font-mono text-white">
            {midPrice}c
          </span>
          {spread !== null && (
            <span className="text-xs text-gray-500">
              {spread}c spread
            </span>
          )}
        </div>

        {/* Price delta */}
        {priceDelta !== 0 && (
          <span className={`text-sm font-mono ${
            priceDelta > 0 ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {priceDelta > 0 ? '+' : ''}{priceDelta}c
          </span>
        )}
      </div>

      {/* Secondary stats row */}
      <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
        <div className="flex items-center gap-3">
          {/* Volume + delta */}
          <span>
            Vol: {formatVolume(market.volume || 0)}
            {volumeDelta !== 0 && (
              <span className={volumeDelta > 0 ? 'text-emerald-400' : 'text-red-400'}>
                {' '}({volumeDelta > 0 ? '+' : ''}{formatVolume(volumeDelta)})
              </span>
            )}
          </span>
        </div>

        {/* Time displays: tracked time + close time */}
        <div className="flex items-center gap-2">
          {timeSinceTrack && (
            <span className="flex items-center gap-1" title="Time since tracking started">
              <ClockIcon />
              {timeSinceTrack}
            </span>
          )}
          {timeUntilClose && (
            <span className={`flex items-center gap-1 ${closingSoon ? 'text-amber-400' : ''}`} title="Time until market closes">
              <HourglassIcon />
              {timeUntilClose}
            </span>
          )}
        </div>
      </div>

      {/* RLM Stats Section - Always shown (even 0 trades) */}
      <div className="mt-3 pt-2 border-t border-gray-700/50 space-y-2">
        {/* Trade counts and ratio */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Trades:</span>
          <span className="font-mono">
            <span className="text-emerald-400">{yesTrades} YES</span>
            {' / '}
            <span className="text-red-400">{noTrades} NO</span>
            {totalTrades > 0 && (
              <span className="text-gray-400 ml-2">
                ({(yesRatio * 100).toFixed(1)}%)
              </span>
            )}
          </span>
        </div>

        {/* Price movement */}
        {(firstYesPrice !== null && firstYesPrice !== undefined) && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">Price:</span>
            <span className="font-mono text-white">
              {firstYesPrice}c
              {lastYesPrice !== null && lastYesPrice !== undefined && lastYesPrice !== firstYesPrice && (
                <>
                  {' -> '}{lastYesPrice}c
                  <span className={priceDrop > 0 ? 'text-red-400 ml-1' : priceDrop < 0 ? 'text-emerald-400 ml-1' : ''}>
                    {priceDrop > 0 ? `(${String.fromCharCode(8595)}${priceDrop}c)` : priceDrop < 0 ? `(${String.fromCharCode(8593)}${Math.abs(priceDrop)}c)` : ''}
                  </span>
                </>
              )}
            </span>
          </div>
        )}

        {/* Progress bar toward 15-trade threshold */}
        <div className="relative">
          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                signalReady ? 'bg-amber-500' : progressPercent >= 100 ? 'bg-blue-500' : 'bg-gray-500'
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
            <span>{totalTrades} trades</span>
            {signalReady ? (
              <span className="text-amber-400 font-medium">SIGNAL READY</span>
            ) : (
              <span>{SIGNAL_THRESHOLD - totalTrades > 0 ? `${SIGNAL_THRESHOLD - totalTrades} to signal` : ''}</span>
            )}
          </div>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-700 text-xs space-y-2">
          {/* Price details */}
          <div className="flex justify-between text-gray-400">
            <span>Bid / Ask:</span>
            <span className="font-mono text-white">
              {market.yes_bid || '-'}c / {market.yes_ask || '-'}c
            </span>
          </div>

          {/* Open interest */}
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
                {market.price_at_track}c → {midPrice}c
                {priceDelta !== 0 && (
                  <span className={priceDelta > 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {' '}({priceDelta > 0 ? '+' : ''}{priceDelta}c)
                  </span>
                )}
              </span>
            </div>
          )}

          {/* Status */}
          <div className="flex justify-between text-gray-400">
            <span>Status:</span>
            <span className={`font-medium ${
              market.status === 'determined' ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {market.status || 'active'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper functions
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

function getCategoryStyle(category) {
  const styles = {
    sports: 'bg-blue-500/20 text-blue-400',
    crypto: 'bg-orange-500/20 text-orange-400',
    entertainment: 'bg-purple-500/20 text-purple-400',
    media_mentions: 'bg-pink-500/20 text-pink-400',
    politics: 'bg-red-500/20 text-red-400',
    economics: 'bg-green-500/20 text-green-400',
    climate: 'bg-cyan-500/20 text-cyan-400',
    financials: 'bg-emerald-500/20 text-emerald-400',
    science: 'bg-indigo-500/20 text-indigo-400',
    world: 'bg-teal-500/20 text-teal-400',
    tech: 'bg-violet-500/20 text-violet-400',
    culture: 'bg-fuchsia-500/20 text-fuchsia-400',
  };
  // Case-insensitive lookup
  const key = (category || '').toLowerCase();
  return styles[key] || 'bg-gray-500/20 text-gray-400';
}

// Simple clock icon (time since tracked)
const ClockIcon = () => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

// Hourglass icon (time until close)
const HourglassIcon = () => (
  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M12 8V4m0 16v-4m-6-4h12M7 4h10v2a4 4 0 01-4 4h-2a4 4 0 01-4-4V4zm0 16h10v-2a4 4 0 00-4-4h-2a4 4 0 00-4 4v2z" />
  </svg>
);

export default LifecycleMarketCard;
