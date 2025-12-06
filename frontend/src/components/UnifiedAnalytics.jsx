import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
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

const UnifiedAnalytics = ({ 
  analyticsData = { 
    hour_minute_mode: { time_series: [], summary_stats: {} },
    day_hour_mode: { time_series: [], summary_stats: {} }
  },
  ...props
}) => {
  const [timeMode, setTimeMode] = useState('hour'); // 'hour' or 'day'
  
  // Memoize current mode data to prevent unnecessary recalculations
  const currentModeData = useMemo(() => {
    return timeMode === 'hour' 
      ? analyticsData.hour_minute_mode 
      : analyticsData.day_hour_mode;
  }, [analyticsData, timeMode]);
  
  const timeSeriesData = useMemo(() => currentModeData.time_series || [], [currentModeData.time_series]);
  const summaryStats = useMemo(() => currentModeData.summary_stats || {}, [currentModeData.summary_stats]);
  
  // Use a timer to update currentTimestamp every 30 seconds to keep current period highlighting accurate
  const [currentTimestamp, setCurrentTimestamp] = useState(() => {
    const now = new Date();
    if (timeMode === 'hour') {
      const minuteStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                    now.getHours(), now.getMinutes(), 0, 0);
      return minuteStart.getTime();
    } else {
      const hourStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                  now.getHours(), 0, 0, 0);
      return hourStart.getTime();
    }
  });

  // Update currentTimestamp every 30 seconds to keep current period highlighting accurate
  useEffect(() => {
    const updateTimestamp = () => {
      const now = new Date();
      if (timeMode === 'hour') {
        const minuteStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                      now.getHours(), now.getMinutes(), 0, 0);
        setCurrentTimestamp(minuteStart.getTime());
      } else {
        const hourStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                    now.getHours(), 0, 0, 0);
        setCurrentTimestamp(hourStart.getTime());
      }
    };

    // Update immediately when timeMode changes
    updateTimestamp();

    // Then update every 30 seconds
    const interval = setInterval(updateTimestamp, 30000);
    
    return () => clearInterval(interval);
  }, [timeMode]);
  
  // Extract only the specific data we need to avoid dependency on entire analyticsData object
  const currentMinuteData = analyticsData.current_minute_data;
  const currentHourData = analyticsData.current_hour_data;
  const ultraFastTotals = analyticsData.ultra_fast_totals;
  
  // Memoize expensive chart data formatting to prevent unnecessary recalculations on every render
  const chartData = useMemo(() => {
    return timeSeriesData.map(point => {
      let enhancedPoint = { ...point };
      
      // CRITICAL: Update current period data with ultra-fast values for real-time chart updates
      if (point.timestamp === currentTimestamp) {
        if (timeMode === 'hour' && currentMinuteData && 
            (currentMinuteData.volume_usd > 0 || currentMinuteData.trade_count > 0)) {
          // Use ultra-fast current minute data for hour mode
          enhancedPoint.volume_usd = currentMinuteData.volume_usd;
          enhancedPoint.trade_count = currentMinuteData.trade_count;
        } else if (timeMode === 'day' && currentHourData && 
                   (currentHourData.volume_usd > 0 || currentHourData.trade_count > 0)) {
          // Use current hour data for day mode
          enhancedPoint.volume_usd = currentHourData.volume_usd;
          enhancedPoint.trade_count = currentHourData.trade_count;
        }
      }
      
      return {
        ...enhancedPoint,
        // Convert timestamp to time string for display
        timeString: timeMode === 'hour' 
          ? new Date(point.timestamp).toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit',
              hour12: false 
            })
          : new Date(point.timestamp).toLocaleDateString([], { 
              month: 'short',
              day: '2-digit',
              hour: '2-digit',
              hour12: false
            }),
        // Format volume for display
        volumeDisplayed: enhancedPoint.volume_usd?.toFixed(0) || 0,
        // Mark if this is the current period
        isCurrentPeriod: point.timestamp === currentTimestamp
      };
    });
  }, [timeSeriesData, timeMode, currentTimestamp, currentMinuteData, currentHourData]);

  // Custom tooltip component with premium styling
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
              {/* Time label with enhanced styling */}
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
                    ${volumeData.value?.toLocaleString()}
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
                    {tradeData.value?.toLocaleString()}
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

  // Memoize expensive calculations for chart scaling
  const { maxVolume, maxTrades } = useMemo(() => {
    return {
      maxVolume: Math.max(...chartData.map(d => d.volume_usd || 0)),
      maxTrades: Math.max(...chartData.map(d => d.trade_count || 0))
    };
  }, [chartData]);

  // Note: Removed currentPeriodData lookup to eliminate race conditions
  // that were causing oscillation between real values and 0/0

  // State persistence for last known good values to prevent 0-fallbacks during oscillation
  const [lastKnownValues, setLastKnownValues] = useState({
    minute: { volume: 0, trades: 0 },
    hour: { volume: 0, trades: 0 }
  });

  // Debouncing ref to prevent excessive state updates
  const updateTimeoutRef = useRef(null);

  // Optimized state update function with debouncing
  const updateLastKnownValues = useCallback((timeMode, volume, trades) => {
    // Clear any pending updates
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }

    // Debounce updates to prevent excessive re-renders during rapid data flow
    updateTimeoutRef.current = setTimeout(() => {
      setLastKnownValues(prev => ({
        ...prev,
        [timeMode]: { volume, trades }
      }));
    }, 50); // 50ms debounce
  }, []);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

  // Memoize current period statistics with prioritized ultra-fast data source
  const { currentVolume, currentTrades, currentVolumeKey, currentTradesKey, summaryStatsWithUltraFast } = useMemo(() => {
    const volumeKey = timeMode === 'hour' ? 'current_minute_volume_usd' : 'current_hour_volume_usd';
    const tradesKey = timeMode === 'hour' ? 'current_minute_trades' : 'current_hour_trades';
    
    let currentVolume = 0;
    let currentTrades = 0;
    
    // Create enhanced summary stats that merge ultra-fast data with regular summary stats
    const enhancedSummaryStats = { ...summaryStats };
    
    // CRITICAL FIX: Use mode-specific totals as base values with ultra-fast updates as authoritative
    // This gives us different Hour vs Day base totals AND real-time responsiveness working together
    
    // Priority 1: Use ultra-fast mode-specific totals if available (most current)
    if (ultraFastTotals && ultraFastTotals.last_update) {
      const ultraFastAge = Date.now() - ultraFastTotals.last_update;
      
      // Use ultra-fast totals if they're recent (< 5 seconds old)
      if (ultraFastAge < 5000) {
        if (timeMode === 'hour') {
          // Hour mode: Use 60-minute window totals from ultra-fast data
          enhancedSummaryStats.total_volume_usd = ultraFastTotals.hour_mode_total_volume_usd;
          enhancedSummaryStats.total_trades = ultraFastTotals.hour_mode_total_trades;
        } else {
          // Day mode: Use 24-hour window totals from ultra-fast data
          enhancedSummaryStats.total_volume_usd = ultraFastTotals.day_mode_total_volume_usd;
          enhancedSummaryStats.total_trades = ultraFastTotals.day_mode_total_trades;
        }
      } else {
        // Ultra-fast data is stale, fall back to mode-specific base values from summaryStats
        // summaryStats already contains mode-specific values from hour_minute_mode vs day_hour_mode
        enhancedSummaryStats.total_volume_usd = summaryStats.total_volume_usd || 0;
        enhancedSummaryStats.total_trades = summaryStats.total_trades || 0;
      }
    } else {
      // Priority 2: Use mode-specific base values from summaryStats if no ultra-fast data
      // summaryStats already contains mode-specific values from hour_minute_mode vs day_hour_mode
      enhancedSummaryStats.total_volume_usd = summaryStats.total_volume_usd || 0;
      enhancedSummaryStats.total_trades = summaryStats.total_trades || 0;
    }
    
    if (timeMode === 'hour') {
      // PRIORITY 1: Ultra-fast current minute data (most authoritative)
      if (currentMinuteData && (currentMinuteData.volume_usd > 0 || currentMinuteData.trade_count > 0)) {
        currentVolume = currentMinuteData.volume_usd;
        currentTrades = currentMinuteData.trade_count;
        
        // CRITICAL: Update summary stats with ultra-fast current minute data
        enhancedSummaryStats.current_minute_volume_usd = currentVolume;
        enhancedSummaryStats.current_minute_trades = currentTrades;
        
        // Update last known good values with debouncing
        updateLastKnownValues('minute', currentVolume, currentTrades);
      }
      // PRIORITY 2: Summary stats (if ultra-fast data not available)
      else if (summaryStats[volumeKey] > 0 || summaryStats[tradesKey] > 0) {
        currentVolume = summaryStats[volumeKey] || 0;
        currentTrades = summaryStats[tradesKey] || 0;
        
        // Update last known good values if this is new data
        if (currentVolume > lastKnownValues.minute.volume || currentTrades > lastKnownValues.minute.trades) {
          updateLastKnownValues('minute', currentVolume, currentTrades);
        }
      }
      // PRIORITY 3: Last known good values (prevent 0-fallback)
      else {
        currentVolume = lastKnownValues.minute.volume;
        currentTrades = lastKnownValues.minute.trades;
        
        // Use last known values in enhanced summary stats too
        enhancedSummaryStats.current_minute_volume_usd = currentVolume;
        enhancedSummaryStats.current_minute_trades = currentTrades;
      }
    } else {
      // Day mode: prioritize current hour data
      if (currentHourData && (currentHourData.volume_usd > 0 || currentHourData.trade_count > 0)) {
        currentVolume = currentHourData.volume_usd;
        currentTrades = currentHourData.trade_count;
        
        // CRITICAL: Update summary stats with current hour data for day mode
        enhancedSummaryStats.current_hour_volume_usd = currentVolume;
        enhancedSummaryStats.current_hour_trades = currentTrades;
        
        // Update last known good values with debouncing
        updateLastKnownValues('hour', currentVolume, currentTrades);
      }
      // Fallback to summary stats for hour mode
      else if (summaryStats[volumeKey] > 0 || summaryStats[tradesKey] > 0) {
        currentVolume = summaryStats[volumeKey] || 0;
        currentTrades = summaryStats[tradesKey] || 0;
        
        // Update last known good values if this is new data
        if (currentVolume > lastKnownValues.hour.volume || currentTrades > lastKnownValues.hour.trades) {
          updateLastKnownValues('hour', currentVolume, currentTrades);
        }
      }
      // Last known good values fallback
      else {
        currentVolume = lastKnownValues.hour.volume;
        currentTrades = lastKnownValues.hour.trades;
        
        // Use last known values in enhanced summary stats too
        enhancedSummaryStats.current_hour_volume_usd = currentVolume;
        enhancedSummaryStats.current_hour_trades = currentTrades;
      }
    }
    
    return {
      currentVolumeKey: volumeKey,
      currentTradesKey: tradesKey,
      currentVolume,
      currentTrades,
      summaryStatsWithUltraFast: enhancedSummaryStats
    };
  }, [timeMode, summaryStats, currentMinuteData, currentHourData, ultraFastTotals, lastKnownValues, updateLastKnownValues]);

  // Memoize label descriptions to prevent recreation on every render
  const labelDescriptions = useMemo(() => {
    return {
      title: timeMode === 'hour' ? "Trading Activity (Last Hour)" : "Trading Activity (Last 24 Hours)",
      periodLabel: timeMode === 'hour' ? "minute" : "hour",
      peakPeriodLabel: timeMode === 'hour' ? "minute" : "hour",
      totalPeriodLabel: timeMode === 'hour' ? "hourly" : "daily"
    };
  }, [timeMode]);

  // Optimize time mode toggle functions with useCallback to prevent unnecessary re-renders
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

      {/* Summary Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8" data-testid="summary-stats-grid">
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-volume-stat">
          <div className="text-3xl font-bold text-blue-600 mb-1" data-testid="peak-volume-value">
            {formatVolume(summaryStatsWithUltraFast.peak_volume_usd || 0)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Volume</div>
          <div className="text-xs text-gray-500 mt-1">{labelDescriptions.peakPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-trades-stat">
          <div className="text-3xl font-bold text-emerald-600 mb-1" data-testid="peak-trades-value">
            {(summaryStatsWithUltraFast.peak_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Trades</div>
          <div className="text-xs text-gray-500 mt-1">{labelDescriptions.peakPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-volume-stat">
          <div className="text-3xl font-bold text-purple-600 mb-1" data-testid="total-volume-value">
            {formatVolume(summaryStatsWithUltraFast.total_volume_usd || 0)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Volume</div>
          <div className="text-xs text-gray-500 mt-1">{labelDescriptions.totalPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-trades-stat">
          <div className="text-3xl font-bold text-indigo-600 mb-1" data-testid="total-trades-value">
            {(summaryStatsWithUltraFast.total_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Trades</div>
          <div className="text-xs text-gray-500 mt-1">{labelDescriptions.totalPeriodLabel}</div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 p-8" data-testid="chart-section">
        {/* Header with Current Stats - Horizontally Aligned */}
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-6">
          {/* Left Side - Trading Activity Header */}
          <div className="flex-1">
            {/* Enhanced Chart Header with Live Data */}
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
          
          {/* Right Side - Current Stats */}
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
                    
                    {/* Current stats - Combined on one line */}
                    <div className="flex items-center justify-between gap-4">
                      {/* Volume */}
                      <div className="flex-1 min-w-0" data-testid="current-volume">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-2.5 bg-gradient-to-r from-blue-500 to-blue-600 rounded-sm shadow-sm ring-1 ring-blue-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Volume</span>
                        </div>
                        <div className="text-lg font-bold text-blue-700" data-testid="current-volume-value">
                          {formatVolume(currentVolume)}
                        </div>
                      </div>
                      
                      {/* Trades */}
                      <div className="flex-1 min-w-0" data-testid="current-trades">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-0.5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full shadow-sm ring-1 ring-emerald-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Trades</span>
                        </div>
                        <div className="text-lg font-bold text-emerald-700" data-testid="current-trades-value">
                          {currentTrades}
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
                animationDuration={0}
                isAnimationActive={false}
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
                  isAnimationActive={false}
                  animationDuration={0}
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
                  isAnimationActive={false}
                  animationDuration={0}
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