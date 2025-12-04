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
      setNewTradeIds(newIds);
      
      // Remove highlighting after 2 seconds
      setTimeout(() => {
        setNewTradeIds(prev => {
          const updated = new Set(prev);
          newIds.forEach(id => updated.delete(id));
          return updated;
        });
      }, 2000);
    }

    previousTradesRef.current = trades;
  }, [trades]);

  if (trades.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Live Trade Tape</h2>
          <p className="text-sm text-gray-500">Real-time market activity</p>
        </div>
        <div className="p-8 text-center">
          <div className="text-gray-400 text-sm">
            <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mx-auto mb-2"></div>
            Waiting for trades...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">Live Trade Tape</h2>
        <p className="text-sm text-gray-500">
          {trades.length} recent trades • Newest first
        </p>
      </div>

      {/* Column Headers */}
      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between space-x-3 text-xs font-medium text-gray-500 uppercase">
          <div className="w-16">Time</div>
          <div className="flex-1">Market</div>
          <div className="flex-shrink-0">Side</div>
          <div className="w-16 text-right">Price</div>
          <div className="w-12 text-right">Size</div>
        </div>
      </div>

      {/* Trade List */}
      <div className="max-h-96 overflow-y-auto">
        {trades.map((trade, index) => {
          const tradeId = trade.ts + trade.market_ticker;
          const isNew = newTradeIds.has(tradeId);
          const isSelected = selectedTicker === trade.market_ticker;

          return (
            <TradeRow
              key={`${trade.ts}-${trade.market_ticker}-${index}`}
              trade={trade}
              onClick={onTickerSelect}
              isSelected={isSelected}
              isNew={isNew}
            />
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          Click on any trade to view market details
        </p>
      </div>
    </div>
  );
};

export default TradeTape;