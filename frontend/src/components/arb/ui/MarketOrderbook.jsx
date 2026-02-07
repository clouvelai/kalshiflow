import React, { memo, useMemo } from 'react';

/**
 * MarketOrderbook - Compact inline orderbook depth display.
 *
 * Shows bid/ask levels with size bars, centered on spread.
 * YES bids on left (cyan), YES asks on right (derived from NO bids, red).
 */

const MarketOrderbook = ({ market, maxLevels = 5 }) => {
  const {
    ticker,
    title,
    yes_levels = [],
    no_levels = [],
    yes_bid,
    yes_ask,
    spread,
  } = market;

  const bids = useMemo(() => yes_levels.slice(0, maxLevels), [yes_levels, maxLevels]);
  const asks = useMemo(() =>
    no_levels.slice(0, maxLevels).map(([price, size]) => [100 - price, size]),
    [no_levels, maxLevels]
  );

  const maxSize = useMemo(() => {
    const allSizes = [...bids.map(l => l[1]), ...asks.map(l => l[1])];
    return Math.max(...allSizes, 1);
  }, [bids, asks]);

  if (bids.length === 0 && asks.length === 0) {
    return (
      <div className="bg-gray-800/15 rounded-lg p-3 border border-gray-800/20">
        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">
          Orderbook: {title || ticker}
        </div>
        <p className="text-[11px] text-gray-600 text-center py-2">
          No depth data
        </p>
      </div>
    );
  }

  // Pad shorter side to match longer side length
  const numLevels = Math.max(bids.length, asks.length);

  return (
    <div className="bg-gray-800/15 rounded-lg p-3 border border-gray-800/20">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
          Orderbook: {title || ticker}
        </span>
        {spread != null && (
          <span className="text-[10px] font-mono text-gray-500 tabular-nums">
            spread: {spread}c
          </span>
        )}
      </div>

      {/* Column headers */}
      <div className="flex items-center text-[8px] text-gray-600 uppercase tracking-wider font-semibold mb-1">
        <span className="w-8 text-right">Size</span>
        <span className="w-8 text-right mr-1">Bid</span>
        <span className="flex-1" />
        <span className="w-8 text-left ml-1">Ask</span>
        <span className="w-8 text-left">Size</span>
      </div>

      {/* Levels */}
      <div className="space-y-px">
        {Array.from({ length: numLevels }).map((_, i) => {
          const bid = bids[i];
          const ask = asks[i];
          const bidPct = bid ? (bid[1] / maxSize) * 100 : 0;
          const askPct = ask ? (ask[1] / maxSize) * 100 : 0;

          return (
            <div key={i} className="flex items-center h-4">
              {/* Bid size */}
              <span className="w-8 text-right text-[9px] font-mono text-gray-500 tabular-nums">
                {bid ? bid[1] : ''}
              </span>
              {/* Bid price */}
              <span className="w-8 text-right text-[10px] font-mono text-cyan-400/80 mr-1 tabular-nums">
                {bid ? bid[0] : ''}
              </span>
              {/* Bid bar | Ask bar */}
              <div className="flex-1 flex h-3">
                <div className="w-1/2 flex justify-end">
                  <div
                    className="h-full bg-cyan-500/15 rounded-l-sm"
                    style={{ width: `${bidPct}%` }}
                  />
                </div>
                <div className="w-px bg-gray-700/30" />
                <div className="w-1/2">
                  <div
                    className="h-full bg-red-500/15 rounded-r-sm"
                    style={{ width: `${askPct}%` }}
                  />
                </div>
              </div>
              {/* Ask price */}
              <span className="w-8 text-left text-[10px] font-mono text-red-400/70 ml-1 tabular-nums">
                {ask ? ask[0] : ''}
              </span>
              {/* Ask size */}
              <span className="w-8 text-left text-[9px] font-mono text-gray-500 tabular-nums">
                {ask ? ask[1] : ''}
              </span>
            </div>
          );
        })}
      </div>

      {/* BBO summary */}
      <div className="flex items-center justify-between text-[10px] font-mono tabular-nums border-t border-gray-700/20 pt-1 mt-1">
        <span className="text-cyan-400/60">
          {yes_bid != null ? `Best bid: ${yes_bid}c` : ''}
        </span>
        <span className="text-red-400/60">
          {yes_ask != null ? `Best ask: ${yes_ask}c` : ''}
        </span>
      </div>
    </div>
  );
};

export default memo(MarketOrderbook);
