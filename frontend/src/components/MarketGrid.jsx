import React from 'react';
import MarketCard from './MarketCard';

const MarketGrid = ({ markets = [], ...props }) => {
  if (markets.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8" {...props} data-testid={props['data-testid'] || "market-grid"}>
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
    <div className="space-y-6" {...props} data-testid={props['data-testid'] || "market-grid"}>
      {/* Section Header */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            {/* Enhanced Hot Markets Header with Live Data */}
            <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 via-red-500 to-pink-600 rounded-2xl blur opacity-20"></div>
              <div className="relative bg-white/90 backdrop-blur-sm rounded-xl p-5 border border-white/40">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
                      <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                      <div className="w-1 h-1 bg-pink-500 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
                    </div>
                    <div>
                      <h2 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-gray-900 via-orange-800 to-red-900 bg-clip-text text-transparent">
                        Hot Markets
                      </h2>
                      <p className="text-gray-600 text-sm mt-1">
                        Top markets ranked by trading volume in the last 10 minutes
                      </p>
                    </div>
                  </div>
                  
                  {/* Kalshi Link Button */}
                  <a
                    href="https://kalshi.com/sign-up/?referral=3f328bbb-7b1b-479a-93f9-9f8197e92a70"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="
                      group flex items-center gap-2 px-4 py-2.5
                      bg-gradient-to-r from-emerald-500 to-green-600 
                      hover:from-emerald-600 hover:to-green-700
                      rounded-xl shadow-lg hover:shadow-xl
                      transition-all duration-200 ease-out
                      hover:scale-105 hover:-translate-y-0.5
                      border border-emerald-400/30
                    "
                    title="Visit Kalshi to trade these markets"
                    data-testid="kalshi-header-link"
                  >
                    <span className="w-6 h-6 bg-white/20 rounded-lg flex items-center justify-center">
                      <span className="text-white font-bold text-sm">K</span>
                    </span>
                    <span className="text-white font-semibold text-sm hidden sm:block">
                      Trade on Kalshi
                    </span>
                  </a>
                </div>
              </div>
            </div>
          </div>
          
          <div className="text-center ml-6">
            <div className="relative">
              {/* Gradient background glow */}
              <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 via-red-500 to-pink-500 rounded-xl blur opacity-25"></div>
              <div className="relative bg-white/90 backdrop-blur-sm rounded-xl p-4 border border-white/50">
                <div className="text-3xl font-bold text-transparent bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text">
                  {markets.length}
                </div>
                <div className="text-sm text-gray-600 font-medium">Active Markets</div>
              </div>
            </div>
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
              rank={index + 1}
            />
          </div>
        ))}
      </div>

    </div>
  );
};

export default React.memo(MarketGrid, (prevProps, nextProps) => {
  // Only re-render if markets array actually changed
  // Compare by length and ticker values to avoid unnecessary re-renders
  if (prevProps.markets?.length !== nextProps.markets?.length) {
    return false; // Re-render needed
  }
  
  // If same length, check if any market tickers changed
  // This is a fast comparison that catches most real updates
  for (let i = 0; i < (prevProps.markets?.length || 0); i++) {
    if (prevProps.markets[i]?.ticker !== nextProps.markets[i]?.ticker ||
        prevProps.markets[i]?.volume_window !== nextProps.markets[i]?.volume_window) {
      return false; // Re-render needed
    }
  }
  
  return true; // No re-render needed
});