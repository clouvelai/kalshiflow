import React, { memo } from 'react';

/**
 * MarketSelector - Horizontal tab bar for switching between markets in an event.
 */
const MarketSelector = ({ markets = [], selectedTicker, onSelect }) => {
  if (markets.length <= 1) return null;

  return (
    <div className="flex items-center gap-1 px-3 py-2 border-b border-gray-800/20 overflow-x-auto shrink-0">
      {markets.map(m => {
        const isActive = m.ticker === selectedTicker;
        return (
          <button
            key={m.ticker}
            onClick={() => onSelect(m.ticker)}
            className={`px-2.5 py-1 rounded-md text-[10px] font-mono whitespace-nowrap transition-colors ${
              isActive
                ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30 border border-transparent'
            }`}
            title={m.title || m.ticker}
          >
            {m.subtitle || m.ticker}
          </button>
        );
      })}
    </div>
  );
};

export default memo(MarketSelector);
