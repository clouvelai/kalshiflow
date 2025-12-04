import React from 'react';
import Sparkline from './Sparkline';

const MarketCard = ({ market, onClick, isSelected = false, rank }) => {
  if (!market) return null;

  const handleClick = () => {
    if (onClick && market.ticker) {
      onClick(market.ticker);
    }
  };

  // Calculate flow direction and intensity
  const flowDirection = market.yes_flow > market.no_flow ? 'YES' : 'NO';
  const flowIntensity = Math.abs(market.yes_flow - market.no_flow);
  const maxFlow = Math.max(market.yes_flow, market.no_flow);
  const flowPercentage = maxFlow > 0 ? (flowIntensity / maxFlow) * 100 : 0;

  // Determine hotness level based on volume
  const getHotnessLevel = (volume) => {
    if (volume >= 1000) return 3; // Very hot
    if (volume >= 500) return 2;  // Hot  
    if (volume >= 100) return 1;  // Warm
    return 0; // Cool
  };

  const hotnessLevel = getHotnessLevel(market.volume_window || 0);
  const hotnessColors = [
    'border-gray-200 bg-white', // Cool
    'border-yellow-200 bg-yellow-50', // Warm
    'border-orange-200 bg-orange-50', // Hot
    'border-red-200 bg-red-50' // Very hot
  ];

  const hotnessIndicators = ['🟢', '🟡', '🟠', '🔥'];

  return (
    <div 
      className={`
        relative cursor-pointer rounded-xl border-2 p-4 transition-all duration-300 
        hover:shadow-lg hover:-translate-y-1 hover:border-blue-300
        ${isSelected ? 'border-blue-500 bg-blue-50 shadow-md' : hotnessColors[hotnessLevel]}
        ${isSelected ? 'ring-2 ring-blue-200' : ''}
      `}
      onClick={handleClick}
      data-testid={`market-card-${market.ticker}`}
    >
      {/* Rank Badge */}
      {rank && (
        <div className={`
          absolute -top-2 -left-2 w-6 h-6 rounded-full text-xs font-bold 
          flex items-center justify-center text-white z-10
          ${rank === 1 ? 'bg-yellow-500' : 
            rank === 2 ? 'bg-gray-400' : 
            rank === 3 ? 'bg-orange-600' : 'bg-gray-300'}
        `}>
          {rank}
        </div>
      )}

      {/* Hotness Indicator */}
      <div className="absolute top-2 right-2 text-lg">
        {hotnessIndicators[hotnessLevel]}
      </div>

      {/* Market Ticker */}
      <div className="mb-3">
        <h3 className="font-semibold text-gray-900 text-sm truncate" title={market.ticker}>
          {market.ticker}
        </h3>
      </div>

      {/* Price Section */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Last Price</span>
          <span className="text-lg font-bold text-gray-900">
            {market.last_price ? `${Math.round(market.last_price * 100)}¢` : '--'}
          </span>
        </div>
        {market.last_price && (
          <div className="text-xs text-gray-600">
            {Math.round(market.last_price * 100)}% YES
          </div>
        )}
      </div>

      {/* Volume */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Volume (10m)</span>
          <span className="text-sm font-semibold text-orange-600">
            {market.volume_window || 0}
          </span>
        </div>
      </div>

      {/* Flow Direction */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Flow</span>
          <span className={`
            text-sm font-semibold px-2 py-1 rounded-full
            ${flowDirection === 'YES' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}
          `}>
            {flowDirection}
          </span>
        </div>
        <div className="mt-1">
          <div className="w-full bg-gray-200 rounded-full h-1">
            <div 
              className={`h-1 rounded-full transition-all duration-500 ${
                flowDirection === 'YES' ? 'bg-green-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(flowPercentage, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Sparkline */}
      {market.price_points && market.price_points.length > 1 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Price Trend</div>
          <div className="h-8">
            <Sparkline 
              data={market.price_points}
              width={180}
              height={32}
              color={flowDirection === 'YES' ? '#10b981' : '#ef4444'}
              strokeWidth={1.5}
            />
          </div>
        </div>
      )}

      {/* Footer Stats */}
      <div className="pt-2 border-t border-gray-100">
        <div className="flex justify-between text-xs text-gray-500">
          <span>YES: {market.yes_flow || 0}</span>
          <span>NO: {market.no_flow || 0}</span>
        </div>
      </div>

      {/* Hover Overlay */}
      <div className="absolute inset-0 bg-blue-50 opacity-0 hover:opacity-10 rounded-xl transition-opacity duration-200 pointer-events-none"></div>
    </div>
  );
};

export default MarketCard;