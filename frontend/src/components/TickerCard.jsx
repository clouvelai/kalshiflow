import React from 'react';
import Sparkline from './Sparkline';

const TickerCard = ({ market, onClick, isSelected }) => {
  const formatTicker = (ticker) => {
    // Truncate long ticker names for display
    return ticker.length > 25 ? `${ticker.substring(0, 22)}...` : ticker;
  };

  const formatPrice = (price) => {
    return `${price}%`;
  };

  const formatVolume = (volume) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume?.toString() || '0';
  };

  const getFlowDirection = (yesFlow, noFlow) => {
    if (yesFlow > noFlow) return 'yes';
    if (noFlow > yesFlow) return 'no';
    return 'neutral';
  };

  const getFlowColor = (direction) => {
    switch (direction) {
      case 'yes': return 'text-green-600 bg-green-50';
      case 'no': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getFlowIcon = (direction) => {
    switch (direction) {
      case 'yes': return '↗';
      case 'no': return '↘';
      default: return '→';
    }
  };

  const flowDirection = getFlowDirection(market.yes_flow, market.no_flow);
  const netFlow = Math.abs((market.yes_flow || 0) - (market.no_flow || 0));

  return (
    <div
      className={`
        group cursor-pointer transition-all duration-200 rounded-lg border
        ${isSelected 
          ? 'border-blue-500 bg-blue-50 shadow-md' 
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
        }
      `}
      onClick={() => onClick?.(market.ticker)}
    >
      <div className="p-4">
        {/* Header with ticker and price */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="font-medium text-gray-900 text-sm truncate" title={market.ticker}>
              {formatTicker(market.ticker)}
            </h3>
          </div>
          <div className="ml-2 text-right">
            <div className="text-lg font-bold text-gray-900">
              {formatPrice(market.last_yes_price)}
            </div>
            <div className="text-xs text-gray-500">YES</div>
          </div>
        </div>

        {/* Sparkline */}
        <div className="mb-3">
          <Sparkline 
            data={market.price_points || []} 
            width={120} 
            height={24}
            className="w-full"
          />
        </div>

        {/* Stats row */}
        <div className="flex items-center justify-between text-sm">
          {/* Volume */}
          <div>
            <div className="font-medium text-gray-900">
              {formatVolume(market.volume_window)}
            </div>
            <div className="text-xs text-gray-500">Volume</div>
          </div>

          {/* Flow direction and net flow */}
          <div className="text-right">
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getFlowColor(flowDirection)}`}>
              <span className="mr-1">{getFlowIcon(flowDirection)}</span>
              {formatVolume(netFlow)}
            </div>
            <div className="text-xs text-gray-500 mt-1">Net Flow</div>
          </div>
        </div>

        {/* Additional stats on hover */}
        <div className="mt-3 pt-3 border-t border-gray-100 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Trades:</span>
              <span className="ml-1 font-medium">{market.trade_count_window || 0}</span>
            </div>
            <div>
              <span className="text-gray-500">NO:</span>
              <span className="ml-1 font-medium">{market.last_no_price}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TickerCard;