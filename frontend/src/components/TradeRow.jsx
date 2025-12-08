import React from 'react';

const TradeRow = ({ trade, onClick, isSelected, isNew = false }) => {
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatTicker = (ticker) => {
    // Handle missing ticker gracefully
    if (!ticker) return 'N/A';
    // Truncate long ticker names for display
    return ticker.length > 20 ? `${ticker.substring(0, 17)}...` : ticker;
  };

  const formatPrice = (price) => {
    return `${price}¢`;
  };

  const getSideColor = (side) => {
    return side?.toLowerCase() === 'yes' ? 'text-green-600' : 'text-red-600';
  };

  const getSideBg = (side) => {
    return side?.toLowerCase() === 'yes' ? 'bg-green-50' : 'bg-red-50';
  };

  return (
    <div
      className={`
        group cursor-pointer border-l-4 transition-all duration-200
        ${isSelected 
          ? 'border-l-blue-500 bg-blue-50' 
          : 'border-l-transparent hover:border-l-gray-300 hover:bg-gray-50'
        }
        ${isNew ? 'animate-pulse bg-yellow-50' : ''}
        p-3 border-b border-gray-100
      `}
      onClick={() => onClick?.(trade.market_ticker)}
    >
      <div className="flex items-center justify-between space-x-3">
        {/* Time */}
        <div className="flex-shrink-0 w-16">
          <span className="text-xs font-mono text-gray-500">
            {formatTime(trade.ts)}
          </span>
        </div>

        {/* Market Ticker */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-gray-900 truncate">
            {formatTicker(trade.market_ticker)}
          </div>
        </div>

        {/* Direction */}
        <div className="flex-shrink-0">
          <span className={`
            inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
            ${getSideBg(trade.taker_side)} ${getSideColor(trade.taker_side)}
          `}>
            {trade.taker_side?.toUpperCase() || 'N/A'}
          </span>
        </div>

        {/* Price */}
        <div className="flex-shrink-0 w-16 text-right">
          <span className="text-sm font-mono font-medium text-gray-900">
            {formatPrice(
              trade.taker_side?.toLowerCase() === 'yes' 
                ? trade.yes_price 
                : trade.no_price
            )}
          </span>
        </div>

        {/* Size */}
        <div className="flex-shrink-0 w-12 text-right">
          <span className="text-sm font-mono text-gray-600">
            {trade.count}
          </span>
        </div>
      </div>
    </div>
  );
};

export default TradeRow;