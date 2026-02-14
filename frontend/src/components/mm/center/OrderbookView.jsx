import React, { memo, useMemo } from 'react';

/**
 * OrderbookView - Full-depth orderbook visualization (centerpiece).
 *
 * Shows bid/ask levels with horizontal size bars, our quotes highlighted,
 * fair value line, spread indicator, and queue position markers.
 */

const LEVELS = 8;

const OrderbookView = ({ market, ourQuotes, fairValue }) => {
  const {
    yes_levels = [],
    no_levels = [],
    yes_bid,
    yes_ask,
    spread,
    ticker,
    title,
  } = market || {};

  // Our bid/ask info
  const ourBid = ourQuotes?.bid;
  const ourAsk = ourQuotes?.ask;

  // YES bids (from yes_levels), YES asks (derived from no_levels)
  const bids = useMemo(() => yes_levels.slice(0, LEVELS), [yes_levels]);
  const asks = useMemo(
    () => no_levels.slice(0, LEVELS).map(([price, size]) => [100 - price, size]),
    [no_levels]
  );

  const maxSize = useMemo(() => {
    const allSizes = [...bids.map(l => l[1]), ...asks.map(l => l[1])];
    return Math.max(...allSizes, 1);
  }, [bids, asks]);

  const numLevels = Math.max(bids.length, asks.length, 3);

  // Check if a price matches our quote
  const isOurBid = (price) => ourBid && ourBid.price_cents === price;
  const isOurAsk = (price) => ourAsk && ourAsk.price_cents === price;

  if (!market || (bids.length === 0 && asks.length === 0)) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="text-[11px] text-gray-600 mb-1">No orderbook data</div>
          <div className="text-[10px] text-gray-700 font-mono">{ticker || 'Select a market'}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 px-4 py-3">
      {/* Title bar */}
      <div className="flex items-center justify-between mb-3 shrink-0">
        <div>
          <div className="text-[11px] font-semibold text-gray-300 truncate max-w-[300px]" title={title}>
            {title || ticker}
          </div>
          <div className="text-[9px] font-mono text-gray-600">{ticker}</div>
        </div>
        <div className="flex items-center gap-3">
          {fairValue != null && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-gray-500 uppercase">FV</span>
              <span className="text-[11px] font-mono text-violet-400 tabular-nums">{fairValue.toFixed(1)}c</span>
            </div>
          )}
          {spread != null && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-gray-500 uppercase">Spread</span>
              <span className={`text-[11px] font-mono tabular-nums ${spread <= 2 ? 'text-emerald-400' : spread <= 5 ? 'text-amber-400' : 'text-red-400'}`}>
                {spread}c
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center text-[8px] text-gray-600 uppercase tracking-wider font-semibold mb-1 shrink-0">
        <span className="w-10 text-right">Size</span>
        <span className="w-8 text-right mr-2">Bid</span>
        <span className="flex-1" />
        <span className="w-8 text-left ml-2">Ask</span>
        <span className="w-10 text-left">Size</span>
      </div>

      {/* Orderbook levels */}
      <div className="flex-1 flex flex-col justify-center space-y-px">
        {Array.from({ length: numLevels }).map((_, i) => {
          const bid = bids[i];
          const ask = asks[i];
          const bidPct = bid ? (bid[1] / maxSize) * 100 : 0;
          const askPct = ask ? (ask[1] / maxSize) * 100 : 0;
          const bidIsOurs = bid && isOurBid(bid[0]);
          const askIsOurs = ask && isOurAsk(ask[0]);

          return (
            <div key={i} className="flex items-center h-6">
              {/* Bid size */}
              <span className={`w-10 text-right text-[10px] font-mono tabular-nums ${bidIsOurs ? 'text-cyan-300 font-semibold' : 'text-gray-500'}`}>
                {bid ? bid[1] : ''}
              </span>
              {/* Bid price */}
              <span className={`w-8 text-right text-[11px] font-mono mr-2 tabular-nums ${bidIsOurs ? 'text-cyan-300 font-bold' : 'text-cyan-400/80'}`}>
                {bid ? bid[0] : ''}
              </span>

              {/* Bars */}
              <div className="flex-1 flex h-5 relative">
                {/* Fair value line */}
                {fairValue != null && yes_bid != null && yes_ask != null && (
                  <div
                    className="absolute top-0 bottom-0 w-px bg-violet-500/40 z-10"
                    style={{
                      left: `${Math.max(0, Math.min(100, ((fairValue - (yes_bid || 0)) / ((yes_ask || 100) - (yes_bid || 0))) * 100))}%`,
                    }}
                  />
                )}

                {/* Bid bar (right-aligned) */}
                <div className="w-1/2 flex justify-end">
                  <div
                    className={`h-full rounded-l-sm transition-all duration-300 ${bidIsOurs ? 'bg-cyan-400/30 border-r-2 border-cyan-400' : 'bg-cyan-500/12'}`}
                    style={{ width: `${bidPct}%` }}
                  />
                </div>
                {/* Center divider */}
                <div className="w-px bg-gray-700/40" />
                {/* Ask bar (left-aligned) */}
                <div className="w-1/2">
                  <div
                    className={`h-full rounded-r-sm transition-all duration-300 ${askIsOurs ? 'bg-red-400/30 border-l-2 border-red-400' : 'bg-red-500/12'}`}
                    style={{ width: `${askPct}%` }}
                  />
                </div>
              </div>

              {/* Ask price */}
              <span className={`w-8 text-left text-[11px] font-mono ml-2 tabular-nums ${askIsOurs ? 'text-red-300 font-bold' : 'text-red-400/70'}`}>
                {ask ? ask[0] : ''}
              </span>
              {/* Ask size */}
              <span className={`w-10 text-left text-[10px] font-mono tabular-nums ${askIsOurs ? 'text-red-300 font-semibold' : 'text-gray-500'}`}>
                {ask ? ask[1] : ''}
              </span>
            </div>
          );
        })}
      </div>

      {/* BBO summary + our quotes */}
      <div className="mt-2 pt-2 border-t border-gray-700/20 shrink-0">
        <div className="flex items-center justify-between text-[10px] font-mono tabular-nums">
          <span className="text-cyan-400/60">
            {yes_bid != null ? `Best bid: ${yes_bid}c` : ''}
          </span>
          <span className="text-red-400/60">
            {yes_ask != null ? `Best ask: ${yes_ask}c` : ''}
          </span>
        </div>
        {(ourBid || ourAsk) && (
          <div className="flex items-center justify-between text-[9px] font-mono mt-1">
            {ourBid ? (
              <span className="text-cyan-300/80">
                Our bid: {ourBid.price_cents}c x{ourBid.size}
                {ourBid.queue_position != null && <span className="text-gray-600 ml-1">Q:{ourBid.queue_position}</span>}
              </span>
            ) : <span />}
            {ourAsk ? (
              <span className="text-red-300/80">
                Our ask: {ourAsk.price_cents}c x{ourAsk.size}
                {ourAsk.queue_position != null && <span className="text-gray-600 ml-1">Q:{ourAsk.queue_position}</span>}
              </span>
            ) : <span />}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(OrderbookView);
