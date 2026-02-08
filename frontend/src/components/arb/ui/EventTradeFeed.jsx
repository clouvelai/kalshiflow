import React, { memo, useMemo } from 'react';

/**
 * EventTradeFeed - Scrolling list of trades across all markets in an event.
 *
 * Supports both public trades (from `event_arb_trade` WS) and Captain trades.
 * Color-coded by taker side (yes=green, no=red) with optional source badges.
 */

/**
 * SourceBadge - Shows trade source (Captain vs Public)
 */
const SourceBadge = memo(({ source }) => {
  if (source === 'captain') {
    return (
      <span className="text-[8px] bg-cyan-500/15 text-cyan-400/80 px-1.5 py-0.5 rounded font-medium">
        CAPT
      </span>
    );
  }
  return (
    <span className="text-[8px] bg-gray-700/30 text-gray-500 px-1.5 py-0.5 rounded">
      PUB
    </span>
  );
});
SourceBadge.displayName = 'SourceBadge';

/**
 * StatusBadge - Shows trade status for Captain trades
 */
const StatusBadge = memo(({ status }) => {
  if (!status) return null;

  const statusConfig = {
    filled: { bg: 'bg-emerald-500/15', text: 'text-emerald-400/80', label: 'FILLED' },
    executed: { bg: 'bg-emerald-500/15', text: 'text-emerald-400/80', label: 'FILLED' },
    partial: { bg: 'bg-amber-500/15', text: 'text-amber-400/80', label: 'PARTIAL' },
    cancelled: { bg: 'bg-red-500/15', text: 'text-red-400/80', label: 'CANCEL' },
    expired: { bg: 'bg-red-500/15', text: 'text-red-400/80', label: 'EXPRD' },
    pending: { bg: 'bg-gray-500/15', text: 'text-gray-400/80', label: 'PEND' },
    placed: { bg: 'bg-gray-500/15', text: 'text-gray-400/80', label: 'PLACED' },
    resting: { bg: 'bg-blue-500/15', text: 'text-blue-400/80', label: 'REST' },
  };

  const config = statusConfig[status.toLowerCase()] || statusConfig.pending;

  return (
    <span className={`text-[8px] ${config.bg} ${config.text} px-1.5 py-0.5 rounded font-medium`}>
      {config.label}
    </span>
  );
});
StatusBadge.displayName = 'StatusBadge';

const TradeLine = memo(({ trade, showSource = false }) => {
  const {
    market_ticker,
    yes_price,
    no_price,
    count,
    taker_side,
    ts,
    source,
    status,
    reasoning,
  } = trade;

  const sideColor = taker_side === 'yes' ? 'text-emerald-400' : taker_side === 'no' ? 'text-red-400' : 'text-gray-400';
  const price = taker_side === 'no' ? no_price : yes_price;

  const timeStr = useMemo(() => {
    if (!ts) return '';
    const d = new Date(ts > 1e12 ? ts : ts * 1000);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }, [ts]);

  // Short ticker: last segment after last dash
  const shortTicker = market_ticker?.split('-').pop() || market_ticker;

  // Truncate reasoning to 50 chars
  const truncatedReason = reasoning && reasoning.length > 50
    ? reasoning.substring(0, 47) + '...'
    : reasoning;

  return (
    <div className="flex items-center gap-2 text-[10px] font-mono py-0.5 px-1 hover:bg-gray-800/20 rounded group">
      <span className="text-gray-600 w-14 flex-shrink-0 tabular-nums">{timeStr}</span>
      {showSource && (
        <span className="w-10 flex-shrink-0">
          <SourceBadge source={source} />
        </span>
      )}
      <span className="text-gray-500 w-12 flex-shrink-0 truncate">{shortTicker}</span>
      <span className={`font-semibold w-8 flex-shrink-0 uppercase ${sideColor}`}>
        {taker_side || '?'}
      </span>
      <span className={`w-8 flex-shrink-0 text-right tabular-nums ${sideColor}`}>
        {price != null ? `${price}c` : '--'}
      </span>
      <span className="text-gray-600 w-6 flex-shrink-0 text-right tabular-nums">
        x{count || 1}
      </span>
      {source === 'captain' && (
        <>
          <span className="w-12 flex-shrink-0">
            <StatusBadge status={status} />
          </span>
          {truncatedReason && (
            <span
              className="text-gray-600 truncate flex-1 text-[9px]"
              title={reasoning}
            >
              {truncatedReason}
            </span>
          )}
        </>
      )}
    </div>
  );
});
TradeLine.displayName = 'TradeLine';

const EventTradeFeed = ({ trades = [], maxHeight = 280, showSource = false }) => {
  if (trades.length === 0) {
    return (
      <div className="text-center py-4">
        <p className="text-[11px] text-gray-600">
          No trades yet
        </p>
      </div>
    );
  }

  // Count captain vs public trades
  const captainCount = trades.filter(t => t.source === 'captain').length;
  const publicCount = trades.length - captainCount;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          {showSource && captainCount > 0 && (
            <span className="text-[10px] text-cyan-400/80 font-mono tabular-nums">
              {captainCount} captain
            </span>
          )}
          {showSource && publicCount > 0 && (
            <span className="text-[10px] text-gray-500 font-mono tabular-nums">
              {publicCount} public
            </span>
          )}
        </div>
      </div>
      <div
        className="space-y-0 overflow-y-auto"
        style={{ maxHeight }}
      >
        {trades.map((trade, idx) => (
          <TradeLine key={trade.id || idx} trade={trade} showSource={showSource} />
        ))}
      </div>
    </div>
  );
};

export { StatusBadge };
export default memo(EventTradeFeed);
