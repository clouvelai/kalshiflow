import React, { useState, useMemo, useCallback } from 'react';
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
  realtimeData = {
    current_minute: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    current_hour: { timestamp: 0, volume_usd: 0, trade_count: 0 },
    mode_totals: {
      hour_mode_total_volume_usd: 0,
      hour_mode_total_trades: 0,
      day_mode_total_volume_usd: 0,
      day_mode_total_trades: 0
    },
    peaks: { peak_volume_usd: 0, peak_trades: 0 }
  },
  ...props
}) => {
  const [timeMode, setTimeMode] = useState('hour'); // 'hour' or 'day'
  
  // Get current mode data for historical chart
  const currentModeData = useMemo(() => {
    return timeMode === 'hour' 
      ? analyticsData.hour_minute_mode 
      : analyticsData.day_hour_mode;
  }, [analyticsData, timeMode]);
  
  // Get current period timestamp for highlighting current bar
  const currentTimestamp = useMemo(() => {
    const now = new Date();
    if (timeMode === 'hour') {
      // Current minute timestamp
      return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                     now.getHours(), now.getMinutes(), 0, 0).getTime();
    } else {
      // Current hour timestamp  
      return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                     now.getHours(), 0, 0, 0).getTime();
    }
  }, [timeMode]);
  
  // Get current period data from realtimeData
  const currentPeriodData = useMemo(() => {
    return timeMode === 'hour' 
      ? realtimeData.current_minute 
      : realtimeData.current_hour;
  }, [realtimeData, timeMode]);
  
  // Get mode-specific totals from realtimeData
  const modeTotals = useMemo(() => {
    const totals = realtimeData.mode_totals;
    return timeMode === 'hour' 
      ? {
          total_volume_usd: totals.hour_mode_total_volume_usd,
          total_trades: totals.hour_mode_total_trades
        }
      : {
          total_volume_usd: totals.day_mode_total_volume_usd,
          total_trades: totals.day_mode_total_trades
        };
  }, [realtimeData.mode_totals, timeMode]);
  
  // Prepare chart data with current period enhancement
  const chartData = useMemo(() => {
    const timeSeries = currentModeData.time_series || [];
    
    return timeSeries.map(point => {
      let enhancedPoint = { ...point };
      
      // Enhance current period with real-time data
      if (point.timestamp === currentTimestamp && currentPeriodData) {
        enhancedPoint.volume_usd = currentPeriodData.volume_usd;
        enhancedPoint.trade_count = currentPeriodData.trade_count;
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
        // Mark if this is the current period
        isCurrentPeriod: point.timestamp === currentTimestamp
      };
    });
  }, [currentModeData.time_series, timeMode, currentTimestamp, currentPeriodData]);

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

  // Calculate chart scaling
  const { maxVolume, maxTrades } = useMemo(() => {
    return {
      maxVolume: Math.max(...chartData.map(d => d.volume_usd || 0)),
      maxTrades: Math.max(...chartData.map(d => d.trade_count || 0))
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

      {/* Summary Stats Grid - Using realtimeData directly */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8" data-testid="summary-stats-grid">
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-volume-stat">
          <div className="text-3xl font-bold text-blue-600 mb-1" data-testid="peak-volume-value">
            {formatVolume(realtimeData.peaks.peak_volume_usd)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Volume</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'minute' : 'hour'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="peak-trades-stat">
          <div className="text-3xl font-bold text-emerald-600 mb-1" data-testid="peak-trades-value">
            {(realtimeData.peaks.peak_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Trades</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'minute' : 'hour'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-volume-stat">
          <div className="text-3xl font-bold text-purple-600 mb-1" data-testid="total-volume-value">
            {formatVolume(modeTotals.total_volume_usd)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Volume</div>
          <div className="text-xs text-gray-500 mt-1">{timeMode === 'hour' ? 'hourly' : 'daily'}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300" data-testid="total-trades-stat">
          <div className="text-3xl font-bold text-indigo-600 mb-1" data-testid="total-trades-value">
            {(modeTotals.total_trades || 0).toLocaleString()}
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
          
          {/* Right Side - Current Stats - Using realtimeData directly */}
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
                          {formatVolume(currentPeriodData.volume_usd)}
                        </div>
                      </div>
                      
                      {/* Trades */}
                      <div className="flex-1 min-w-0" data-testid="current-trades">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-0.5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full shadow-sm ring-1 ring-emerald-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Trades</span>
                        </div>
                        <div className="text-lg font-bold text-emerald-700" data-testid="current-trades-value">
                          {currentPeriodData.trade_count}
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