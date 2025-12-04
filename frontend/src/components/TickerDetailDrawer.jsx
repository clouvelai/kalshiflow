import React from 'react';
import VolumeWeightedSparkline from './VolumeWeightedSparkline';

const TickerDetailDrawer = ({ ticker, tickerData, onClose, isOpen }) => {
  if (!isOpen || !ticker || !tickerData) {
    return null;
  }

  const { marketData, recentTrades = [] } = tickerData;

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatPrice = (price) => `${price}%`;
  const formatVolume = (volume) => {
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(2)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume?.toString() || '0';
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" onClick={onClose}>
      <div className="absolute inset-0 bg-black bg-opacity-50" />
      
      <div className="absolute right-0 top-0 h-full w-full max-w-2xl bg-white shadow-xl">
        <div 
          className="h-full overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 z-10">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <h2 className="text-xl font-bold text-gray-900 truncate" title={ticker}>
                  {ticker}
                </h2>
                <p className="text-sm text-gray-500">Market Details</p>
              </div>
              
              <button
                onClick={onClose}
                className="ml-4 text-gray-400 hover:text-gray-600 transition-colors p-2"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {marketData && (
              <>
                {/* Price Chart Section */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Live Price Momentum</h3>
                  <div className="bg-white rounded-lg p-4">
                    <VolumeWeightedSparkline 
                      data={marketData.price_points || []} 
                      width={420} 
                      height={100}
                      className="w-full"
                      showVolumeIndicators={true}
                      animationDuration={400}
                    />
                  </div>
                </div>

                {/* Key Statistics */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white rounded p-3 text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {formatPrice(marketData.last_yes_price)}
                      </div>
                      <div className="text-sm text-gray-500">YES Price</div>
                    </div>
                    
                    <div className="bg-white rounded p-3 text-center">
                      <div className="text-2xl font-bold text-red-600">
                        {formatPrice(marketData.last_no_price)}
                      </div>
                      <div className="text-sm text-gray-500">NO Price</div>
                    </div>
                    
                    <div className="bg-white rounded p-3 text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {formatVolume(marketData.volume_window)}
                      </div>
                      <div className="text-sm text-gray-500">Volume (10m)</div>
                    </div>
                    
                    <div className="bg-white rounded p-3 text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {marketData.trade_count_window || 0}
                      </div>
                      <div className="text-sm text-gray-500">Trades (10m)</div>
                    </div>
                  </div>

                  {/* Flow Direction */}
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div className="bg-white rounded p-3">
                      <div className="text-lg font-bold text-green-600">
                        {formatVolume(marketData.yes_flow || 0)}
                      </div>
                      <div className="text-sm text-gray-500">YES Flow</div>
                    </div>
                    
                    <div className="bg-white rounded p-3">
                      <div className="text-lg font-bold text-red-600">
                        {formatVolume(marketData.no_flow || 0)}
                      </div>
                      <div className="text-sm text-gray-500">NO Flow</div>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Recent Trades */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Recent Trades ({recentTrades.length})
              </h3>
              
              {recentTrades.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  No recent trades for this market
                </div>
              ) : (
                <div className="bg-white rounded max-h-64 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Side</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Price</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Size</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {recentTrades.slice(0, 20).map((trade, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-mono text-xs">
                            {formatTime(trade.ts)}
                          </td>
                          <td className="px-4 py-2">
                            <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full
                              ${trade.taker_side?.toLowerCase() === 'yes' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                              }`}>
                              {trade.taker_side?.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right font-mono">
                            {formatPrice(
                              trade.taker_side?.toLowerCase() === 'yes' 
                                ? trade.yes_price 
                                : trade.no_price
                            )}
                          </td>
                          <td className="px-4 py-2 text-right font-mono">
                            {trade.count}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TickerDetailDrawer;