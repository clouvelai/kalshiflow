import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

const TopTradesList = ({ trades = [], windowMinutes = 10 }) => {
  // Format volume in thousands or millions
  const formatVolume = (volume) => {
    if (volume >= 1000000) {
      return `$${(volume / 1000000).toFixed(2)}M`;
    }
    if (volume >= 1000) {
      return `$${(volume / 1000).toFixed(1)}k`;
    }
    return `$${volume.toFixed(0)}`;
  };

  // Format price with cents symbol
  const formatPrice = (price) => {
    return `${price}¢`;
  };

  // Split trades into two columns
  const { leftColumn, rightColumn } = useMemo(() => {
    const midpoint = Math.ceil(trades.length / 2);
    return {
      leftColumn: trades.slice(0, midpoint),
      rightColumn: trades.slice(midpoint)
    };
  }, [trades]);

  const TradeCard = ({ trade, index, isEntering }) => {
    const isYes = trade.taker_side === 'yes';
    
    return (
      <div 
        className={`
          bg-gray-900 rounded-lg p-3 border border-gray-800 
          transition-all duration-500 transform
          ${isEntering ? 'animate-slide-in opacity-0' : 'opacity-100'}
          hover:border-gray-700 hover:bg-gray-850
        `}
        style={{
          animationDelay: isEntering ? `${index * 50}ms` : '0ms',
          animationFillMode: 'forwards'
        }}
      >
        <div className="flex justify-between items-start mb-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-gray-400">
                #{index + 1}
              </span>
              <span className="text-xs font-semibold text-white">
                {trade.market_ticker}
              </span>
            </div>
            {trade.title && (
              <p className="text-xs text-gray-400 line-clamp-2">
                {trade.title}
              </p>
            )}
          </div>
          <span className="text-xs text-gray-500 ml-2">
            {trade.time_ago}
          </span>
        </div>
        
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center gap-2">
            {isYes ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
            <span className={`text-sm font-medium ${
              isYes ? 'text-green-400' : 'text-red-400'
            }`}>
              {isYes ? 'YES' : 'NO'}
            </span>
            <span className="text-xs text-gray-400">
              @ {formatPrice(isYes ? trade.yes_price : trade.no_price)}
            </span>
          </div>
          
          <div className="text-right">
            <div className="text-sm font-bold text-white">
              {formatVolume(trade.volume_dollars)}
            </div>
            <div className="text-xs text-gray-500">
              {trade.count.toLocaleString()} shares
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (!trades || trades.length === 0) {
    return (
      <div className="bg-gray-950 rounded-xl p-6 border border-gray-800">
        <h2 className="text-lg font-bold text-white mb-4">
          Top Trades by Volume
          <span className="ml-2 text-xs text-gray-500">
            (Last {windowMinutes} minutes)
          </span>
        </h2>
        <div className="text-center py-8 text-gray-500">
          Waiting for trade data...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-950 rounded-xl p-6 border border-gray-800">
      <h2 className="text-lg font-bold text-white mb-4">
        Top Trades by Volume
        <span className="ml-2 text-xs text-gray-500">
          (Last {windowMinutes} minutes)
        </span>
      </h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-3">
          {leftColumn.map((trade, idx) => (
            <TradeCard
              key={`${trade.market_ticker}-${trade.ts}-${idx}`}
              trade={trade}
              index={idx}
              isEntering={false}
            />
          ))}
        </div>
        
        <div className="space-y-3">
          {rightColumn.map((trade, idx) => (
            <TradeCard
              key={`${trade.market_ticker}-${trade.ts}-${idx}`}
              trade={trade}
              index={idx + leftColumn.length}
              isEntering={false}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default TopTradesList;