import React from 'react';
import VolumeWeightedSparkline from './VolumeWeightedSparkline';

// Utility function to format volume numbers
const formatVolume = (volume) => {
  if (!volume || volume === 0) return '$0';
  
  if (volume >= 1000000) {
    return `$${(volume / 1000000).toFixed(1)}M`;
  } else if (volume >= 1000) {
    return `$${(volume / 1000).toFixed(1)}k`;
  } else {
    return `$${volume}`;
  }
};

// Utility function to format time to expiration
const formatTimeToExpiration = (expirationTime) => {
  if (!expirationTime) return null;
  
  try {
    const now = new Date();
    const expiration = new Date(expirationTime);
    const timeDiff = expiration - now;
    
    if (timeDiff <= 0) return 'Expired';
    
    const minutes = Math.floor(timeDiff / (1000 * 60));
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    // If less than 60 minutes, show minutes/seconds
    if (minutes < 60) {
      if (minutes < 1) {
        const seconds = Math.floor(timeDiff / 1000);
        return `${seconds} second${seconds !== 1 ? 's' : ''}`;
      }
      return `${minutes} minute${minutes !== 1 ? 's' : ''}`;
    }
    
    // If less than 1 day, show hours
    if (hours < 24) {
      return `${hours} hour${hours !== 1 ? 's' : ''}`;
    }
    
    // If 1 day or more, show days and hours
    const remainingHours = hours % 24;
    if (remainingHours === 0) {
      return `${days} day${days !== 1 ? 's' : ''}`;
    }
    
    return `${days} day${days !== 1 ? 's' : ''} ${remainingHours} hour${remainingHours !== 1 ? 's' : ''}`;
  } catch {
    return null;
  }
};

// Utility function to format liquidity
const formatLiquidity = (liquidity) => {
  if (!liquidity || liquidity === 0) return null;
  
  if (liquidity >= 1000000000) {
    return `$${(liquidity / 1000000000).toFixed(1)}B`;
  } else if (liquidity >= 1000000) {
    return `$${(liquidity / 1000000).toFixed(1)}M`;
  } else if (liquidity >= 1000) {
    return `$${(liquidity / 1000).toFixed(0)}K`;
  } else {
    return `$${Math.round(liquidity)}`;
  }
};

// Utility function to format open interest
const formatOpenInterest = (openInterest) => {
  if (!openInterest || openInterest === 0) return null;
  
  if (openInterest >= 1000000) {
    return `${(openInterest / 1000000).toFixed(1)}M`;
  } else if (openInterest >= 1000) {
    return `${(openInterest / 1000).toFixed(1)}K`;
  } else {
    return `${openInterest}`;
  }
};

const MarketCard = ({ market, onClick, isSelected = false, rank }) => {
  if (!market) return null;

  const handleClick = () => {
    if (onClick && market.ticker) {
      onClick(market.ticker);
    }
  };

  // Calculate net flow percentage for progress bar
  const netFlow = market.net_flow || 0;
  const totalFlow = (market.yes_flow || 0) + (market.no_flow || 0);
  const flowPercentage = totalFlow > 0 ? (Math.abs(netFlow) / totalFlow) * 100 : 0;

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



      {/* Market Title */}
      <div className="mb-3">
        {/* Primary Title - Always ticker */}
        <h3 className="font-semibold text-gray-900 text-lg leading-tight" title={market.ticker}>
          {market.ticker}
        </h3>
      </div>

      {/* Price Section */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Last Yes Price</span>
          <span className="text-lg font-bold text-gray-900">
            {market.last_yes_price ? `${market.last_yes_price}¢` : '--'}
          </span>
        </div>
      </div>

      {/* Volume */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Volume (10m)</span>
          <span className="text-sm font-semibold text-purple-400">
            {formatVolume(market.volume_window)}
          </span>
        </div>
      </div>

      {/* Trade Count */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Trades (10m)</span>
          <span className="text-sm font-semibold text-gray-700">
            {market.trade_count_window || 0}
          </span>
        </div>
      </div>


      {/* Net Flow */}
      <div className="mb-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">Net Flow (10m)</span>
          <span className={`
            text-sm font-semibold px-2 py-1 rounded-full
            ${market.net_flow > 0 ? 'bg-green-100 text-green-700' : 
              market.net_flow < 0 ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'}
          `}>
            {formatVolume(market.net_flow || 0)}
          </span>
        </div>
        {market.net_flow !== 0 && (
          <div className="mt-1">
            <div className="w-full bg-gray-200 rounded-full h-1">
              <div 
                className={`h-1 rounded-full transition-all duration-500 ${
                  market.net_flow > 0 ? 'bg-green-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(flowPercentage, 100)}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Volume-Weighted Sparkline */}
      {market.price_points && market.price_points.length > 1 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Live Price Momentum</div>
          <div className="h-10">
            <VolumeWeightedSparkline 
              data={market.price_points}
              width={180}
              height={40}
              showVolumeIndicators={true}
              animationDuration={400}
              className="rounded-sm"
            />
          </div>
        </div>
      )}

      {/* Yes/No Flow Stats */}
      <div className="pt-2 border-t border-gray-100 mb-3">
        <div className="flex justify-between text-xs">
          <span className={`
            ${netFlow > 0 ? 'text-green-600 font-semibold' : 'text-gray-500'}
          `}>
            YES: {formatVolume(market.yes_flow || 0)}
          </span>
          <span className={`
            ${netFlow < 0 ? 'text-red-600 font-semibold' : 'text-gray-500'}
          `}>
            NO: {formatVolume(market.no_flow || 0)}
          </span>
        </div>
      </div>

      {/* Metadata Footer */}
      {(market.title || market.liquidity_dollars || market.open_interest || market.latest_expiration_time) && (
        <div className="mt-3 pt-3 border-t border-gray-200 bg-gray-50 -mx-4 -mb-4 px-4 pb-4 rounded-b-xl">
          {/* Market Title */}
          {market.title && (
            <div className="text-xs text-gray-700 mb-2 line-clamp-2 font-medium" title={market.title}>
              {market.title}
            </div>
          )}
          
          {/* Metadata Row */}
          <div className="flex items-center justify-between gap-2 text-xs text-gray-500">
            <div className="flex items-center gap-3 flex-wrap">
              {/* Liquidity */}
              {formatLiquidity(market.liquidity_dollars) && (
                <span className="inline-flex items-center gap-1">
                  <span className="text-blue-500">💧</span>
                  <span>{formatLiquidity(market.liquidity_dollars)}</span>
                </span>
              )}
              
              {/* Open Interest */}
              {formatOpenInterest(market.open_interest) && (
                <span className="inline-flex items-center gap-1">
                  <span className="text-purple-500">👥</span>
                  <span>{formatOpenInterest(market.open_interest)}</span>
                </span>
              )}
            </div>
            
            {/* Time to Expiration */}
            {formatTimeToExpiration(market.latest_expiration_time) && (
              <span className="inline-flex items-center gap-1 text-gray-400">
                <span>⏰</span>
                <span>{formatTimeToExpiration(market.latest_expiration_time)}</span>
              </span>
            )}
          </div>
        </div>
      )}

      {/* Hover Overlay */}
      <div className="absolute inset-0 bg-blue-50 opacity-0 hover:opacity-10 rounded-xl transition-opacity duration-200 pointer-events-none"></div>
    </div>
  );
};

export default MarketCard;