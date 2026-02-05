import React, { memo, useMemo } from 'react';

/**
 * EventTradeFeed - Scrolling list of public trades across all markets in an event.
 *
 * Receives trades from `event_arb_trade` WS messages.
 * Color-coded by taker side (yes=green, no=red).
 */

const TradeLine = memo(({ trade }) => {
  const {
    market_ticker,
    yes_price,
    no_price,
    count,
    taker_side,
    ts,
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

  return (
    <div className="flex items-center gap-2 text-[10px] font-mono py-0.5 px-1 hover:bg-gray-800/30 rounded">
      <span className="text-gray-600 w-14 flex-shrink-0">{timeStr}</span>
      <span className="text-gray-500 w-12 flex-shrink-0 truncate">{shortTicker}</span>
      <span className={`font-bold w-8 flex-shrink-0 uppercase ${sideColor}`}>
        {taker_side || '?'}
      </span>
      <span className={`w-8 flex-shrink-0 text-right ${sideColor}`}>
        {price != null ? `${price}c` : '--'}
      </span>
      <span className="text-gray-600 w-6 flex-shrink-0 text-right">
        x{count || 1}
      </span>
    </div>
  );
});
TradeLine.displayName = 'TradeLine';

const EventTradeFeed = ({ trades = [], maxHeight = 280 }) => {
  if (trades.length === 0) {
    return (
      <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Trade Feed
        </div>
        <p className="text-[11px] text-gray-600 text-center py-2">
          No trades yet
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
          Trade Feed
        </span>
        <span className="text-[10px] text-gray-600 font-mono">
          {trades.length} trades
        </span>
      </div>
      <div
        className="space-y-0 overflow-y-auto"
        style={{ maxHeight }}
      >
        {trades.map(trade => (
          <TradeLine key={trade.id} trade={trade} />
        ))}
      </div>
    </div>
  );
};

export default memo(EventTradeFeed);
