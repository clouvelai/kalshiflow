import React, { useState, useEffect, useRef } from 'react';
import TradeRow from './TradeRow';

const TradeTape = ({ trades = [], selectedTicker, onTickerSelect }) => {
  const [newTradeIds, setNewTradeIds] = useState(new Set());
  const previousTradesRef = useRef([]);

  // Track new trades for highlighting
  useEffect(() => {
    const previousIds = new Set(previousTradesRef.current.map(trade => trade.ts + trade.market_ticker));
    const currentIds = new Set(trades.map(trade => trade.ts + trade.market_ticker));
    
    const newIds = new Set();
    currentIds.forEach(id => {
      if (!previousIds.has(id)) {
        newIds.add(id);
      }
    });

    if (newIds.size > 0) {
      setNewTradeIds(prev => new Set([...prev, ...newIds]));
      
      // Remove highlighting after 2 seconds
      const idsArray = Array.from(newIds);
      const timeoutId = setTimeout(() => {
        setNewTradeIds(prevIds => {
          const updated = new Set(prevIds);
          idsArray.forEach(id => updated.delete(id));
          return updated;
        });
      }, 2000);
      
      // Return cleanup function for this specific timeout
      return () => {
        clearTimeout(timeoutId);
      };
    }

    previousTradesRef.current = trades;
  }, [trades]);

  if (trades.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-100" data-testid="trade-tape">
        <div className="p-3 border-b border-gray-100">
          <h3 className="text-sm font-medium text-gray-700">Live Trade Feed</h3>
          <p className="text-xs text-gray-500">Real-time market activity</p>
        </div>
        <div className="p-6 text-center">
          <div className="text-gray-400 text-sm">
            <div className="animate-spin w-4 h-4 border-2 border-gray-300 border-t-blue-400 rounded-full mx-auto mb-2"></div>
            Waiting for trades...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-100" data-testid="trade-tape">
      {/* Header - Reduced prominence */}
      <div className="p-3 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-gray-700">Live Trade Feed</h3>
            <p className="text-xs text-gray-500">
              {trades.length} recent trades
            </p>
          </div>
          <div className="flex items-center text-xs text-gray-500">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full mr-1 animate-pulse"></div>
            Live
          </div>
        </div>
      </div>

      {/* Column Headers - Reduced size */}
      <div className="px-3 py-1 bg-gray-50 border-b border-gray-100">
        <div className="flex items-center justify-between space-x-2 text-xs font-medium text-gray-500 uppercase">
          <div className="w-14">Time</div>
          <div className="flex-1">Market</div>
          <div className="flex-shrink-0">Side</div>
          <div className="w-14 text-right">Price</div>
          <div className="w-10 text-right">Size</div>
        </div>
      </div>

      {/* Trade List - Reduced height with scrollbar styling */}
      <div className="max-h-64 overflow-y-auto scrollbar-thin">
        {trades.map((trade, index) => {
          const tradeId = trade.ts + trade.market_ticker;
          const isNew = newTradeIds.has(tradeId);
          const isSelected = selectedTicker === trade.market_ticker;

          return (
            <div 
              key={`${trade.ts}-${trade.market_ticker}-${index}`}
              className={isNew ? 'animate-fade-in' : ''}
            >
              <TradeRow
                trade={trade}
                onClick={onTickerSelect}
                isSelected={isSelected}
                isNew={isNew}
              />
            </div>
          );
        })}
      </div>

      {/* Footer - Minimal */}
      <div className="px-3 py-1 bg-gray-50 border-t border-gray-100">
        <p className="text-xs text-gray-400 text-center">
          Click trade to view details
        </p>
      </div>
    </div>
  );
};

export default TradeTape;