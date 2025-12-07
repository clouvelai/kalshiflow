import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

// Utility function to format volume numbers
const formatVolume = (volume) => {
  if (!volume || volume === 0) return '$0';
  
  const absVolume = Math.abs(volume);
  if (absVolume >= 1000000) {
    return `$${(volume / 1000000).toFixed(1)}M`;
  } else if (absVolume >= 1000) {
    return `$${(volume / 1000).toFixed(1)}k`;
  } else {
    return `$${Math.round(volume)}`;
  }
};

// Utility function to format volume for axis (without $ sign)
const formatVolumeAxis = (volume) => {
  if (!volume || volume === 0) return '0';
  
  const absVolume = Math.abs(volume);
  if (absVolume >= 1000000) {
    return `${(volume / 1000000).toFixed(1)}M`;
  } else if (absVolume >= 1000) {
    return `${(volume / 1000).toFixed(1)}k`;
  } else {
    return `${Math.round(volume)}`;
  }
};

// Utility function to format trade count (integers only)
const formatTradeCount = (count) => {
  return Math.round(count || 0);
};

/**
 * Animated counter component that smoothly transitions between numeric values.
 * 
 * Provides smooth, visually appealing number transitions using requestAnimationFrame
 * and cubic easing. Optimized for real-time data streams with debouncing and
 * intelligent animation management to prevent restart loops.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {number} props.value - Target numeric value to animate towards
 * @param {Function} [props.formatter] - Function to format the display value (default: toLocaleString)
 * @param {number} [props.duration=1200] - Animation duration in milliseconds
 * @param {string} [props.className=""] - Additional CSS classes to apply
 * @param {...Object} props.props - Additional props passed to the span element
 * 
 * @example
 * // Basic usage with default formatting
 * <AnimatedCounter value={1234567} />
 * 
 * @example
 * // Custom formatting for currency
 * <AnimatedCounter 
 *   value={1234567} 
 *   formatter={(val) => `$${(val/1000).toFixed(1)}k`}
 *   duration={800}
 * />
 * 
 * @example
 * // Integer-only formatting for trade counts
 * <AnimatedCounter 
 *   value={42} 
 *   formatter={(val) => Math.round(val).toString()}
 * />
 * 
 * Performance & Animation Features:
 * - Uses requestAnimationFrame for optimal performance
 * - Debounced updates prevent rapid animation restarts
 * - Skips animation for minimal value changes (< 1% difference)
 * - Smooth ease-out transitions with longer duration for stability
 * - Prevents animation loops with intelligent state management
 * - Automatically cleans up animation frames
 * 
 * @returns {JSX.Element} Span element displaying the animated value
 */
const AnimatedCounter = ({ 
  value, 
  formatter = (val) => val.toLocaleString(), 
  duration = 1200, // Increased from 750ms to prevent overlap with 1s backend updates
  className = "",
  ...props 
}) => {
  const [displayValue, setDisplayValue] = useState(value || 0);
  const [isAnimating, setIsAnimating] = useState(false);
  const debounceTimeoutRef = useRef(null);
  const lastTargetValueRef = useRef(value || 0);

  useEffect(() => {
    // Validate that we received a numeric value to animate to
    if (typeof value !== 'number') return;

    // Skip if value hasn't actually changed
    if (value === lastTargetValueRef.current) return;
    
    // Calculate percentage change to avoid animating tiny fluctuations
    const percentChange = lastTargetValueRef.current === 0 ? 1 : 
      Math.abs((value - lastTargetValueRef.current) / lastTargetValueRef.current);
    
    // Skip animation for changes smaller than 0.1% to reduce visual noise while still animating meaningful changes
    if (percentChange < 0.001 && lastTargetValueRef.current !== 0) {
      setDisplayValue(value);
      lastTargetValueRef.current = value;
      return;
    }

    // Clear any pending debounced animation
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Debounce rapid updates to prevent animation restart loops
    debounceTimeoutRef.current = setTimeout(() => {
      // If already animating, let current animation finish before starting new one
      if (isAnimating) {
        // Re-schedule this update after current animation completes
        debounceTimeoutRef.current = setTimeout(() => {
          startAnimation(value);
        }, 100);
        return;
      }

      startAnimation(value);
    }, 50); // 50ms debounce for rapid updates

    function startAnimation(targetValue) {
      // Update the target value reference
      lastTargetValueRef.current = targetValue;
      
      // Capture current state for animation calculation
      const startValue = displayValue;  // Where we're animating from
      const endValue = targetValue;     // Where we're animating to
      const startTime = Date.now();     // Animation start timestamp

      // Skip animation if values are the same to avoid unnecessary work
      if (startValue === endValue) return;

      // Mark animation as active
      setIsAnimating(true);

      // Track animation frame ID for cleanup
      let animationFrameId = null;
      let isActive = true;  // Flag to check if animation should continue

      // Animation loop using requestAnimationFrame for optimal performance
      const animate = () => {
        // Early exit if component unmounted or animation cancelled
        if (!isActive) return;

        const now = Date.now();
        
        // Calculate animation progress (0 to 1) clamped to avoid overrun
        const progress = Math.min((now - startTime) / duration, 1);
        
        // Apply ease-out cubic easing for natural, smooth animation
        // This creates a fast start that gradually slows down
        // Formula: y = 1 - (1 - x)³
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        
        // Interpolate between start and end values using eased progress
        const currentValue = startValue + (endValue - startValue) * easeOutCubic;
        
        // Update the displayed value
        setDisplayValue(currentValue);

        // Continue animation if not yet complete
        if (progress < 1 && isActive) {
          // Schedule next frame - browser will optimize this for 60fps
          animationFrameId = requestAnimationFrame(animate);
        } else {
          // Animation complete
          setIsAnimating(false);
        }
      };

      // Start the animation loop
      animationFrameId = requestAnimationFrame(animate);

      // Cleanup function to prevent memory leaks
      return () => {
        isActive = false;  // Stop animation loop
        setIsAnimating(false);
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);  // Cancel pending frame
        }
      };
    }

    // Cleanup function for the effect
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [value, duration]); // Removed displayValue from dependencies to prevent restart loops

  return (
    <span className={className} {...props}>
      {formatter(displayValue)}
    </span>
  );
};

/**
 * Animated counter specialized for volume/currency display.
 * 
 * Pre-configured AnimatedCounter with currency formatting that automatically
 * converts large numbers to readable format (e.g., $1.2M, $500k).
 * 
 * @component
 * @param {Object} props - Component props
 * @param {number} props.value - Volume value in USD to display
 * @param {string} [props.className=""] - Additional CSS classes
 * @param {...Object} props.props - Additional props passed to AnimatedCounter
 * 
 * @example
 * <AnimatedVolumeCounter value={1234567} />  // Displays: $1.2M
 * <AnimatedVolumeCounter value={500000} />   // Displays: $500k
 * <AnimatedVolumeCounter value={42} />       // Displays: $42
 * 
 * @returns {JSX.Element} AnimatedCounter with volume formatting
 */
const AnimatedVolumeCounter = ({ value, className = "", ...props }) => (
  <AnimatedCounter 
    value={value || 0}
    formatter={formatVolume}
    className={className}
    {...props}
  />
);

/**
 * Animated counter specialized for trade count display.
 * 
 * Pre-configured AnimatedCounter with integer formatting and thousand
 * separators for displaying trade counts and similar discrete values.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {number} props.value - Trade count or integer value to display
 * @param {string} [props.className=""] - Additional CSS classes
 * @param {...Object} props.props - Additional props passed to AnimatedCounter
 * 
 * @example
 * <AnimatedTradeCounter value={1234} />    // Displays: 1,234
 * <AnimatedTradeCounter value={42.7} />    // Displays: 43 (rounded)
 * <AnimatedTradeCounter value={0} />       // Displays: 0
 * 
 * @returns {JSX.Element} AnimatedCounter with integer formatting
 */
const AnimatedTradeCounter = ({ value, className = "", ...props }) => (
  <AnimatedCounter 
    value={value || 0}
    formatter={(val) => Math.round(val).toLocaleString()}
    className={className}
    {...props}
  />
);

const UnifiedAnalytics = ({ 
  hourAnalyticsData = {
    current_period: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    summary_stats: { total_volume_usd: 0, total_trades: 0, peak_volume_usd: 0, peak_trades: 0 },
    time_series: []
  },
  dayAnalyticsData = {
    current_period: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    summary_stats: { total_volume_usd: 0, total_trades: 0, peak_volume_usd: 0, peak_trades: 0 },
    time_series: []
  },
  ...props
}) => {
  const [timeMode, setTimeMode] = useState('hour'); // 'hour' or 'day'
  
  // Get current mode analytics data
  const currentModeAnalytics = useMemo(() => {
    return timeMode === 'hour' ? hourAnalyticsData : dayAnalyticsData;
  }, [hourAnalyticsData, dayAnalyticsData, timeMode]);
  
  // Get current period data and timestamp
  const currentPeriodData = useMemo(() => {
    return currentModeAnalytics.current_period;
  }, [currentModeAnalytics]);
  
  const currentTimestamp = useMemo(() => {
    return currentPeriodData.timestamp;
  }, [currentPeriodData]);
  
  // Get summary stats
  const summaryStats = useMemo(() => {
    return currentModeAnalytics.summary_stats;
  }, [currentModeAnalytics]);
  
  // Optimized date formatter functions
  const formatTimeString = useCallback((timestamp) => {
    const date = new Date(timestamp);
    return timeMode === 'hour' 
      ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
      : date.toLocaleDateString([], { month: 'short', day: '2-digit', hour: '2-digit', hour12: false });
  }, [timeMode]);

  // Prepare chart data - optimized for performance
  const chartData = useMemo(() => {
    const timeSeries = currentModeAnalytics.time_series || [];
    
    // Early return if no data
    if (timeSeries.length === 0 && !currentPeriodData?.timestamp) {
      return [];
    }
    
    // Process time series data with optimized operations
    const processedSeries = timeSeries.map(point => ({
      ...point,
      timeString: formatTimeString(point.timestamp),
      isCurrentPeriod: point.timestamp === currentTimestamp
    }));
    
    // Add current period if missing (optimized check)
    if (currentPeriodData && currentTimestamp && 
        !processedSeries.some(point => point.timestamp === currentTimestamp)) {
      processedSeries.push({
        timestamp: currentTimestamp,
        volume_usd: currentPeriodData.volume_usd,
        trade_count: currentPeriodData.trade_count,
        timeString: formatTimeString(currentTimestamp),
        isCurrentPeriod: true
      });
    }
    
    // Sort by timestamp (only if needed)
    return processedSeries.length > 1 
      ? processedSeries.sort((a, b) => a.timestamp - b.timestamp)
      : processedSeries;
  }, [currentModeAnalytics.time_series, currentTimestamp, currentPeriodData, formatTimeString]);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const volumeData = payload.find(p => p.dataKey === 'volume_usd');
      const tradeData = payload.find(p => p.dataKey === 'trade_count');
      
      return (
        <div className="relative">
          {/* Gradient background glow */}
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-2xl blur opacity-30"></div>
          
          {/* Main tooltip container */}
          <div className="relative bg-white/95 backdrop-blur-xl border border-white/50 rounded-2xl shadow-2xl ring-1 ring-white/20 overflow-hidden">
            {/* Header gradient bar */}
            <div className="h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500"></div>
            
            {/* Content */}
            <div className="p-5">
              {/* Time label */}
              <div className="mb-4 pb-3 border-b border-gray-100/50">
                <p className="text-sm font-bold bg-gradient-to-r from-gray-900 via-blue-800 to-indigo-900 bg-clip-text text-transparent">
                  {label}
                </p>
                <p className="text-xs text-gray-500 font-medium mt-1">
                  {timeMode === 'hour' ? 'Current Minute' : 'Current Hour'}
                </p>
              </div>
              
              {/* Volume metric */}
              {volumeData && (
                <div className="mb-3">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-md shadow-sm ring-1 ring-blue-200"></div>
                      <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Volume</span>
                    </div>
                  </div>
                  <div className="mt-1 text-lg font-bold text-blue-600">
                    $<AnimatedCounter 
                      value={volumeData.value || 0}
                      formatter={(val) => Math.round(val).toLocaleString()}
                      duration={800}
                    />
                  </div>
                </div>
              )}
              
              {/* Trade count metric */}
              {tradeData && (
                <div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-1 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full shadow-sm ring-1 ring-emerald-200"></div>
                      <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Trades</span>
                    </div>
                  </div>
                  <div className="mt-1 text-lg font-bold text-emerald-600">
                    <AnimatedCounter 
                      value={tradeData.value || 0}
                      formatter={(val) => Math.round(val).toLocaleString()}
                      duration={800}
                    />
                  </div>
                </div>
              )}
            </div>
            
            {/* Bottom accent */}
            <div className="h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent"></div>
          </div>
        </div>
      );
    }
    return null;
  };

  // Calculate chart scaling - optimized
  const { maxVolume, maxTrades } = useMemo(() => {
    if (chartData.length === 0) {
      return { maxVolume: 100, maxTrades: 10 }; // Default minimums for empty charts
    }
    
    let maxVol = 0;
    let maxTrd = 0;
    
    // Single pass through data instead of multiple map calls
    for (const point of chartData) {
      const vol = point.volume_usd || 0;
      const trd = point.trade_count || 0;
      if (vol > maxVol) maxVol = vol;
      if (trd > maxTrd) maxTrd = trd;
    }
    
    return {
      maxVolume: Math.max(maxVol, 1), // Ensure minimum of 1
      maxTrades: Math.max(maxTrd, 1)  // Ensure minimum of 1
    };
  }, [chartData]);

  // Optimize time mode toggle functions
  const handleHourModeToggle = useCallback(() => {
    setTimeMode('hour');
  }, []);

  const handleDayModeToggle = useCallback(() => {
    setTimeMode('day');
  }, []);

  return (
    <div className="bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 rounded-2xl p-8 mb-8 shadow-sm border border-white/20" {...props} data-testid={props['data-testid'] || "unified-analytics"}>
      {/* Title Section with Toggle */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-3" data-testid="analytics-title">
          Market Analytics
        </h1>
        <p className="text-gray-600 text-lg mb-6 font-medium" data-testid="analytics-subtitle">
          Real-time trading activity and market insights
        </p>
        
        {/* Time Mode Toggle */}
        <div className="inline-flex bg-white/80 backdrop-blur-sm rounded-xl p-1.5 shadow-lg border border-gray-200/50" data-testid="time-mode-toggle">
          <button
            onClick={handleHourModeToggle}
            className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all duration-300 ${
              timeMode === 'hour'
                ? 'bg-blue-600 text-white shadow-lg transform scale-[1.02]'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50/50'
            }`}
            data-testid="hour-view-button"
          >
            Hour View
          </button>
          <button
            onClick={handleDayModeToggle}
            className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all duration-300 ${
              timeMode === 'day'
                ? 'bg-blue-600 text-white shadow-lg transform scale-[1.02]'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50/50'
            }`}
            data-testid="day-view-button"
          >
            Day View
          </button>
        </div>
      </div>

      {/* Summary Stats Grid - Using simplified analytics data with animated counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8" data-testid="summary-stats-grid">
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-volume-stat">
          <div className="text-3xl font-bold text-blue-600 mb-1" data-testid="peak-volume-value">
            <AnimatedVolumeCounter value={summaryStats.peak_volume_usd} />
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Volume</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'minute' : 'hour'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-trades-stat">
          <div className="text-3xl font-bold text-emerald-600 mb-1" data-testid="peak-trades-value">
            <AnimatedTradeCounter value={summaryStats.peak_trades} />
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Trades</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'minute' : 'hour'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-volume-stat">
          <div className="text-3xl font-bold text-purple-600 mb-1" data-testid="total-volume-value">
            <AnimatedVolumeCounter value={summaryStats.total_volume_usd} />
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Volume</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'hourly' : 'daily'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-trades-stat">
          <div className="text-3xl font-bold text-indigo-600 mb-1" data-testid="total-trades-value">
            <AnimatedTradeCounter value={summaryStats.total_trades} />
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Trades</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'hourly' : 'daily'}</div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 p-8" data-testid="chart-section">
        {/* Header with Current Stats */}
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-6">
          {/* Left Side - Trading Activity Header */}
          <div className="flex-1">
            <div className="relative mb-4">
              <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl blur opacity-20"></div>
              <div className="relative bg-white/90 backdrop-blur-sm rounded-xl p-5 border border-white/40">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                    <div className="w-1 h-1 bg-purple-500 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
                  </div>
                  <h2 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-gray-900 via-blue-800 to-indigo-900 bg-clip-text text-transparent" data-testid="chart-title">
                    Trading Activity
                  </h2>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right Side - Current Stats - Using simplified analytics data */}
          {chartData.length > 0 && (
            <div className="flex-shrink-0">
              <div className="relative">
                {/* Gradient background glow */}
                <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-xl blur opacity-25"></div>
                
                {/* Main stats container */}
                <div className="relative bg-white/95 backdrop-blur-xl border border-white/60 rounded-xl shadow-xl ring-1 ring-white/30 overflow-hidden" data-testid="current-stats">
                  {/* Header gradient bar */}
                  <div className="h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500"></div>
                  
                  {/* Content */}
                  <div className="p-4">
                    {/* Live indicator and period label */}
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse shadow-sm" data-testid="live-indicator"></div>
                        <span className="text-green-700 text-xs font-bold uppercase tracking-wider">Live</span>
                      </div>
                      <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider" data-testid="current-period-label">
                        Current {timeMode === 'hour' ? 'Minute' : 'Hour'}
                      </div>
                    </div>
                    
                    {/* Current stats - Combined on one line with animated counters */}
                    <div className="flex items-center justify-between gap-4">
                      {/* Volume */}
                      <div className="flex-1 min-w-0" data-testid="current-volume">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-2.5 bg-gradient-to-r from-blue-500 to-blue-600 rounded-sm shadow-sm ring-1 ring-blue-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Volume</span>
                        </div>
                        <div className="text-lg font-bold text-blue-700" data-testid="current-volume-value">
                          <AnimatedVolumeCounter value={currentPeriodData.volume_usd} />
                        </div>
                      </div>
                      
                      {/* Trades */}
                      <div className="flex-1 min-w-0" data-testid="current-trades">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-0.5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full shadow-sm ring-1 ring-emerald-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Trades</span>
                        </div>
                        <div className="text-lg font-bold text-emerald-700" data-testid="current-trades-value">
                          <AnimatedTradeCounter value={currentPeriodData.trade_count} />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Bottom accent */}
                  <div className="h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent"></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {chartData.length === 0 ? (
          <div className="h-80 flex items-center justify-center text-gray-500" data-testid="chart-loading">
            <div className="text-center">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-36 mx-auto mb-3"></div>
                <div className="h-3 bg-gray-200 rounded w-28 mx-auto"></div>
              </div>
              <p className="mt-3 font-medium">Loading analytics data...</p>
            </div>
          </div>
        ) : (
          <div className="h-80 relative" data-testid="analytics-chart">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartData}
                margin={{
                  top: 20,
                  right: 40,
                  left: 40,
                  bottom: 20,
                }}
                animationDuration={400}
                animationEasing="ease-out"
                isAnimationActive={true}
                key={`chart-${timeMode}-${chartData.length}`}
              >
                <CartesianGrid strokeDasharray="2 4" stroke="#e2e8f0" strokeOpacity={0.6} />
                
                {/* X Axis - Time */}
                <XAxis 
                  dataKey="timeString"
                  tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }}
                  tickLine={{ stroke: '#cbd5e1' }}
                  axisLine={{ stroke: '#cbd5e1' }}
                  interval="preserveStartEnd"
                  height={40}
                />
                
                {/* Left Y Axis - Volume */}
                <YAxis 
                  yAxisId="volume"
                  orientation="left"
                  tick={{ fontSize: 11, fill: '#3b82f6', fontWeight: 500 }}
                  tickLine={{ stroke: '#3b82f6', strokeWidth: 1 }}
                  axisLine={{ stroke: '#3b82f6', strokeWidth: 2 }}
                  tickFormatter={formatVolumeAxis}
                  domain={[0, maxVolume * 1.1]}
                  width={65}
                  label={{ value: 'Volume (USD)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: '11px', fontWeight: 600, fill: '#3b82f6' } }}
                />
                
                {/* Right Y Axis - Trade Count */}
                <YAxis 
                  yAxisId="trades"
                  orientation="right"
                  tick={{ fontSize: 11, fill: '#10b981', fontWeight: 500 }}
                  tickLine={{ stroke: '#10b981', strokeWidth: 1 }}
                  axisLine={{ stroke: '#10b981', strokeWidth: 2 }}
                  tickFormatter={formatTradeCount}
                  domain={[0, maxTrades * 1.1]}
                  width={50}
                  label={{ value: 'Trades', angle: 90, position: 'insideRight', style: { textAnchor: 'middle', fontSize: '11px', fontWeight: 600, fill: '#10b981' } }}
                />
                
                <Tooltip content={<CustomTooltip />} />
                
                {/* Volume bars (primary) */}
                <Bar
                  yAxisId="volume"
                  dataKey="volume_usd"
                  fill="#3b82f6"
                  fillOpacity={0.8}
                  radius={[3, 3, 0, 0]}
                  name="Volume (USD)"
                  isAnimationActive={true}
                  animationDuration={400}
                  animationEasing="ease-out"
                  animationBegin={0}
                />
                
                {/* Trade count line (secondary) */}
                <Line
                  yAxisId="trades"
                  type="monotone"
                  dataKey="trade_count"
                  stroke="#10b981"
                  strokeWidth={3}
                  dot={{ r: 3, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }}
                  activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 3, fill: '#fff' }}
                  name="Trade Count"
                  isAnimationActive={true}
                  animationDuration={400}
                  animationEasing="ease-out"
                  animationBegin={100}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default React.memo(UnifiedAnalytics);