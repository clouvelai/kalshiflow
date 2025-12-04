import React from 'react';
import TickerCard from './TickerCard';

const HotMarkets = ({ markets = [], selectedTicker, onTickerSelect }) => {
  if (markets.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">
            Hot Markets - Loading top markets ranked by trading volume in the last 10 minutes
          </h2>
        </div>
        <div className="p-8 text-center">
          <div className="text-gray-400 text-sm">
            <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full mx-auto mb-2"></div>
            Loading hot markets...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">
          Hot Markets - Top {markets.length} markets ranked by trading volume in the last 10 minutes
        </h2>
      </div>

      {/* Markets Grid */}
      <div className="p-4">
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {markets.map((market, index) => (
            <div key={market.ticker} className="relative">
              {/* Rank indicator */}
              <div className="absolute -left-1 top-1 z-10">
                <div className={`
                  w-6 h-6 rounded-full text-xs font-bold flex items-center justify-center text-white
                  ${index === 0 ? 'bg-yellow-500' : index === 1 ? 'bg-gray-400' : index === 2 ? 'bg-orange-600' : 'bg-gray-300'}
                `}>
                  {index + 1}
                </div>
              </div>
              
              <div className="ml-4">
                <TickerCard
                  market={market}
                  onClick={onTickerSelect}
                  isSelected={selectedTicker === market.ticker}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">
          Click on any market to view detailed information
        </p>
      </div>
    </div>
  );
};

export default HotMarkets;