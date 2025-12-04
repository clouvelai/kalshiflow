import React, { useState, useEffect } from 'react';

const CounterAnimation = ({ value, label, icon, comingSoon = false }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (comingSoon || typeof value !== 'number') return;

    const duration = 500; // Animation duration in ms
    const startValue = displayValue;
    const endValue = value;
    const startTime = Date.now();

    const animate = () => {
      const now = Date.now();
      const progress = Math.min((now - startTime) / duration, 1);
      
      // Easing function for smooth animation
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);
      const currentValue = Math.round(startValue + (endValue - startValue) * easeOutCubic);
      
      setDisplayValue(currentValue);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, comingSoon, displayValue]);

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:border-blue-200 animate-slide-up">
      {/* Icon */}
      <div className="flex items-center justify-between mb-4">
        <div className="text-3xl transform hover:scale-110 transition-transform duration-200">{icon}</div>
        {comingSoon && (
          <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full animate-pulse">
            Coming Soon
          </span>
        )}
      </div>

      {/* Value */}
      <div className="mb-2">
        {comingSoon ? (
          <div className="text-3xl font-bold text-gray-400">--</div>
        ) : (
          <div className="text-3xl font-bold text-gray-900 font-mono animate-counter">
            {displayValue.toLocaleString()}
          </div>
        )}
      </div>

      {/* Label */}
      <div className="text-sm font-medium text-gray-600">{label}</div>
      
      {/* Pulse effect for live data */}
      {!comingSoon && (
        <div className="mt-3">
          <div className="flex items-center text-xs text-green-600">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-glow-pulse"></div>
            <span className="font-medium">Live</span>
          </div>
        </div>
      )}
    </div>
  );
};

const HeroStats = ({ 
  tradesCount = 0, 
  totalVolume = null, 
  netFlow = null 
}) => {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 mb-8" data-testid="hero-stats">
      {/* Title Section */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Market Overview
        </h1>
        <p className="text-gray-600 text-lg">
          Real-time Kalshi trading activity and market statistics
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Total Trades */}
        <CounterAnimation
          value={tradesCount}
          label="Total Trades"
          icon="📊"
        />

        {/* Total Volume - Placeholder */}
        <CounterAnimation
          value={totalVolume}
          label="Total Volume"
          icon="💰"
          comingSoon={true}
        />

        {/* Net Flow - Placeholder */}
        <CounterAnimation
          value={netFlow}
          label="Net Flow"
          icon="🌊"
          comingSoon={true}
        />
      </div>

      {/* Additional Info */}
      <div className="mt-6 text-center">
        <div className="inline-flex items-center px-4 py-2 bg-white rounded-full shadow-sm border border-gray-200">
          <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
          <span className="text-sm font-medium text-gray-700">
            Connected to live Kalshi data feed
          </span>
        </div>
      </div>
    </div>
  );
};

export default HeroStats;