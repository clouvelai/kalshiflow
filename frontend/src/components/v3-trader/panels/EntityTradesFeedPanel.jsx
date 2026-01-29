import React, { useMemo } from 'react';

/**
 * EntityTradesFeedPanel - Live trade feed filtered to entity-tracked markets
 *
 * Shows real-time trades for markets that are tracked in the entity index.
 * Each trade shows: ticker, side (YES/NO), price, time ago, and entity name.
 */
const EntityTradesFeedPanel = ({ entityIndex, eventTrades, tradePulses }) => {
  // Build set of market tickers from entity index
  const trackedMarkets = useMemo(() => {
    const markets = new Set();
    const marketToEntity = new Map();

    entityIndex?.entities?.forEach(entity => {
      entity.markets?.forEach(market => {
        markets.add(market.market_ticker);
        marketToEntity.set(market.market_ticker, entity.canonical_name);
      });
    });

    return { markets, marketToEntity };
  }, [entityIndex]);

  // Flatten all event trades and filter to tracked markets
  const filteredTrades = useMemo(() => {
    const allTrades = [];

    Object.values(eventTrades || {}).forEach(trades => {
      trades.forEach(trade => {
        if (trackedMarkets.markets.has(trade.ticker)) {
          allTrades.push({
            ...trade,
            entityName: trackedMarkets.marketToEntity.get(trade.ticker),
          });
        }
      });
    });

    // Sort by timestamp (most recent first) and limit to 30
    return allTrades
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, 30);
  }, [eventTrades, trackedMarkets]);

  // Format time ago
  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now - then;
    const diffSecs = Math.floor(diffMs / 1000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
    return `${Math.floor(diffSecs / 3600)}h ago`;
  };

  // Check if a trade has a recent pulse (for animation)
  const hasPulse = (ticker) => {
    const pulse = tradePulses?.[ticker];
    if (!pulse) return false;
    return Date.now() - pulse.ts < 1000; // Pulse within last 1 second
  };

  if (filteredTrades.length === 0) {
    return (
      <div className="mb-6 bg-gray-900/50 rounded-lg border border-gray-800 p-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
          <h3 className="text-sm font-medium text-gray-200">Entity Market Trades</h3>
          <span className="text-xs text-gray-500">
            Watching {trackedMarkets.markets.size} markets
          </span>
        </div>
        <div className="text-xs text-gray-500 text-center py-4">
          Waiting for trades in entity-tracked markets...
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6 bg-gray-900/50 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <h3 className="text-sm font-medium text-gray-200">Entity Market Trades</h3>
          <span className="text-xs text-gray-500">
            {filteredTrades.length} trades | {trackedMarkets.markets.size} markets
          </span>
        </div>
      </div>

      <div className="space-y-1 max-h-64 overflow-y-auto">
        {filteredTrades.map((trade, idx) => {
          const isPulsing = hasPulse(trade.ticker);
          const isYes = trade.side === 'yes';

          return (
            <div
              key={`${trade.trade_id || idx}-${trade.timestamp}`}
              className={`
                flex items-center justify-between py-1.5 px-2 rounded text-xs
                ${isPulsing
                  ? isYes
                    ? 'bg-green-900/40 border border-green-700/50'
                    : 'bg-red-900/40 border border-red-700/50'
                  : 'bg-gray-800/30 border border-transparent'
                }
                transition-all duration-300
              `}
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                {/* Side indicator */}
                <span className={`
                  font-medium w-8 text-center rounded px-1
                  ${isYes ? 'text-green-400 bg-green-900/30' : 'text-red-400 bg-red-900/30'}
                `}>
                  {trade.side.toUpperCase()}
                </span>

                {/* Ticker */}
                <span className="text-gray-300 font-mono truncate max-w-32" title={trade.ticker}>
                  {trade.ticker}
                </span>

                {/* Entity name */}
                {trade.entityName && (
                  <span className="text-gray-500 truncate max-w-24" title={trade.entityName}>
                    ({trade.entityName})
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3">
                {/* Count */}
                {trade.count && (
                  <span className="text-gray-400">
                    x{trade.count}
                  </span>
                )}

                {/* Price */}
                <span className={`font-mono w-10 text-right ${isYes ? 'text-green-400' : 'text-red-400'}`}>
                  {trade.yes_price}c
                </span>

                {/* Time ago */}
                <span className="text-gray-500 w-14 text-right">
                  {formatTimeAgo(trade.timestamp)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default EntityTradesFeedPanel;
