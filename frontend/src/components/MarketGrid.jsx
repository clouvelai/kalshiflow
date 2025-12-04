import React from 'react';
import MarketCard from './MarketCard';

const MarketGrid = ({ markets = [], selectedTicker, onTickerSelect }) => {
  if (markets.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8" data-testid="market-grid">
        <div className="text-center">
          <div className="text-gray-400 mb-4">
            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
              <span className="text-2xl">📈</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Loading Markets</h3>
            <p className="text-sm text-gray-500 mb-4">
              Discovering the hottest trading opportunities...
            </p>
            <div className="flex justify-center">
              <div className="animate-spin w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="market-grid">
      {/* Section Header */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              🔥 Hot Markets
            </h2>
            <p className="text-gray-600">
              Top markets ranked by trading volume in the last 10 minutes
            </p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">
              {markets.length}
            </div>
            <div className="text-sm text-gray-600">Active Markets</div>
          </div>
        </div>
      </div>

      {/* Markets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {markets.map((market, index) => (
          <div 
            key={market.ticker}
            className="animate-slide-up"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <MarketCard
              market={market}
              onClick={onTickerSelect}
              isSelected={selectedTicker === market.ticker}
              rank={index + 1}
            />
          </div>
        ))}
      </div>

      {/* Grid Footer */}
      <div className="bg-gray-50 rounded-xl p-4">
        <div className="flex items-center justify-center space-x-6 text-sm text-gray-600">
          <div className="flex items-center">
            <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
            <span>YES Flow</span>
          </div>
          <div className="flex items-center">
            <span className="w-3 h-3 bg-red-500 rounded-full mr-2"></span>
            <span>NO Flow</span>
          </div>
          <div className="flex items-center">
            <span className="text-lg mr-2">🔥</span>
            <span>High Volume</span>
          </div>
        </div>
        <div className="text-center mt-2">
          <p className="text-xs text-gray-500">
            Click on any market card to view detailed information and trading history
          </p>
        </div>
      </div>
    </div>
  );
};

export default MarketGrid;