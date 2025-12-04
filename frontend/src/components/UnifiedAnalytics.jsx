import React, { useState } from 'react';
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
  } 
}) => {
  const [timeMode, setTimeMode] = useState('hour'); // 'hour' or 'day'
  
  // Get current mode data
  const currentModeData = timeMode === 'hour' 
    ? analyticsData.hour_minute_mode 
    : analyticsData.day_hour_mode;
  
  const timeSeriesData = currentModeData.time_series || [];
  const summaryStats = currentModeData.summary_stats || {};
  
  // Get current timestamp for highlighting
  const getCurrentTimestamp = () => {
    const now = new Date();
    if (timeMode === 'hour') {
      // Current minute
      const minuteStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                    now.getHours(), now.getMinutes(), 0, 0);
      return minuteStart.getTime();
    } else {
      // Current hour
      const hourStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                  now.getHours(), 0, 0, 0);
      return hourStart.getTime();
    }
  };
  
  const currentTimestamp = getCurrentTimestamp();
  
  // Format data for recharts
  const formatChartData = (data) => {
    return data.map(point => ({
      ...point,
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
      volumeDisplayed: point.volume_usd?.toFixed(0) || 0,
      // Mark if this is the current period
      isCurrentPeriod: point.timestamp === currentTimestamp
    }));
  };

  const chartData = formatChartData(timeSeriesData);

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

  // Calculate max values for scaling
  const maxVolume = Math.max(...chartData.map(d => d.volume_usd || 0));
  const maxTrades = Math.max(...chartData.map(d => d.trade_count || 0));

  // Find current period data
  const currentPeriodData = chartData.find(d => d.isCurrentPeriod);
  
  // Get current period stats from summary
  const currentVolumeKey = timeMode === 'hour' ? 'current_minute_volume_usd' : 'current_hour_volume_usd';
  const currentTradesKey = timeMode === 'hour' ? 'current_minute_trades' : 'current_hour_trades';
  const currentVolume = summaryStats[currentVolumeKey] || currentPeriodData?.volume_usd || 0;
  const currentTrades = summaryStats[currentTradesKey] || currentPeriodData?.trade_count || 0;

  // Title and period description
  const title = timeMode === 'hour' ? "Trading Activity (Last Hour)" : "Trading Activity (Last 24 Hours)";
  const periodLabel = timeMode === 'hour' ? "minute" : "hour";
  
  // Time period descriptions for metrics
  const peakPeriodLabel = timeMode === 'hour' ? "minute" : "hour";
  const totalPeriodLabel = timeMode === 'hour' ? "hourly" : "daily";

  // Current time period for tooltip
  const getCurrentTimePeriodText = () => {
    const now = new Date();
    if (timeMode === 'hour') {
      // Show current minute (e.g., "14:23")
      return now.toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      });
    } else {
      // Show current hour (e.g., "Dec 04, 14:00")
      return now.toLocaleDateString([], { 
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        hour12: false
      });
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 rounded-2xl p-8 mb-8 shadow-sm border border-white/20">
      {/* Title Section with Toggle */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-3">
          Market Analytics
        </h1>
        <p className="text-gray-600 text-lg mb-6 font-medium">
          Real-time trading activity and market insights
        </p>
        
        {/* Time Mode Toggle */}
        <div className="inline-flex bg-white/80 backdrop-blur-sm rounded-xl p-1.5 shadow-lg border border-gray-200/50">
          <button
            onClick={() => setTimeMode('hour')}
            className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all duration-300 ${
              timeMode === 'hour'
                ? 'bg-blue-600 text-white shadow-lg transform scale-[1.02]'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50/50'
            }`}
          >
            Hour View
          </button>
          <button
            onClick={() => setTimeMode('day')}
            className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all duration-300 ${
              timeMode === 'day'
                ? 'bg-blue-600 text-white shadow-lg transform scale-[1.02]'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50/50'
            }`}
          >
            Day View
          </button>
        </div>
      </div>

      {/* Summary Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300">
          <div className="text-3xl font-bold text-blue-600 mb-1">
            {formatVolume(summaryStats.peak_volume_usd || 0)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Volume</div>
          <div className="text-xs text-gray-500 mt-1">{peakPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300">
          <div className="text-3xl font-bold text-emerald-600 mb-1">
            {(summaryStats.peak_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Peak Trades</div>
          <div className="text-xs text-gray-500 mt-1">{peakPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300">
          <div className="text-3xl font-bold text-purple-600 mb-1">
            {formatVolume(summaryStats.total_volume_usd || 0)}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Volume</div>
          <div className="text-xs text-gray-500 mt-1">{totalPeriodLabel}</div>
        </div>
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 text-center hover:shadow-xl transition-all duration-300">
          <div className="text-3xl font-bold text-indigo-600 mb-1">
            {(summaryStats.total_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">Total Trades</div>
          <div className="text-xs text-gray-500 mt-1">{totalPeriodLabel}</div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 p-8">
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              {/* Enhanced Chart Header with Live Data */}
              <div className="relative mb-4">
                <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl blur opacity-20"></div>
                <div className="relative bg-white/90 backdrop-blur-sm rounded-xl p-5 border border-white/40">
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                        <div className="w-1 h-1 bg-purple-500 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
                      </div>
                      <h2 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-gray-900 via-blue-800 to-indigo-900 bg-clip-text text-transparent">
                        Trading Activity
                      </h2>
                    </div>
                    
                    {/* Live Status Only */}
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-green-50 to-emerald-50 rounded-full border border-green-200/50 shadow-sm">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-sm"></div>
                      <span className="text-green-700 text-xs font-semibold uppercase tracking-wider">Live</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-3 bg-gradient-to-r from-blue-500 to-blue-600 rounded-sm shadow-sm"></div>
                  <span className="font-medium text-gray-700">Volume (USD)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-sm shadow-sm"></div>
                  <span className="font-medium text-gray-700">Trade Count</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {chartData.length === 0 ? (
          <div className="h-80 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-36 mx-auto mb-3"></div>
                <div className="h-3 bg-gray-200 rounded w-28 mx-auto"></div>
              </div>
              <p className="mt-3 font-medium">Loading analytics data...</p>
            </div>
          </div>
        ) : (
          <div className="h-80 relative">
            {/* Floating Current Stats in Upper Right */}
            <div className="absolute top-4 right-4 z-10">
              <div className="relative">
                {/* Gradient background glow */}
                <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-xl blur opacity-25"></div>
                
                {/* Main stats container */}
                <div className="relative bg-white/95 backdrop-blur-xl border border-white/60 rounded-xl shadow-xl ring-1 ring-white/30 overflow-hidden">
                  {/* Header gradient bar */}
                  <div className="h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500"></div>
                  
                  {/* Content */}
                  <div className="p-4">
                    {/* Live indicator and period label */}
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse shadow-sm"></div>
                        <span className="text-green-700 text-xs font-bold uppercase tracking-wider">Live</span>
                      </div>
                      <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider">
                        Current {timeMode === 'hour' ? 'Minute' : 'Hour'}
                      </div>
                    </div>
                    
                    {/* Current stats */}
                    <div className="space-y-3">
                      {/* Volume */}
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-2.5 bg-gradient-to-r from-blue-500 to-blue-600 rounded-sm shadow-sm ring-1 ring-blue-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Volume</span>
                        </div>
                        <div className="text-lg font-bold text-blue-700">
                          {formatVolume(currentVolume)}
                        </div>
                      </div>
                      
                      {/* Trades */}
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-2.5 h-0.5 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full shadow-sm ring-1 ring-emerald-200"></div>
                          <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Trades</span>
                        </div>
                        <div className="text-lg font-bold text-emerald-700">
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
            
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartData}
                margin={{
                  top: 20,
                  right: 40,
                  left: 40,
                  bottom: 20,
                }}
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
                  shape={(props) => {
                    const { fill, x, y, width, height, payload } = props;
                    const isCurrentPeriod = payload.isCurrentPeriod;
                    
                    return (
                      <g>
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill={isCurrentPeriod ? "#10b981" : fill}
                          fillOpacity={isCurrentPeriod ? 0.95 : 0.8}
                          rx={3}
                          ry={3}
                          filter={isCurrentPeriod ? "url(#glow)" : "none"}
                        />
                        {isCurrentPeriod && (
                          <rect
                            x={x}
                            y={y}
                            width={width}
                            height={height}
                            fill="#10b981"
                            fillOpacity={0.4}
                            rx={3}
                            ry={3}
                          >
                            <animate
                              attributeName="fillOpacity"
                              values="0.4;0.7;0.4"
                              dur="2s"
                              repeatCount="indefinite"
                            />
                          </rect>
                        )}
                        <defs>
                          <filter id="glow">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                            <feMerge> 
                              <feMergeNode in="coloredBlur"/>
                              <feMergeNode in="SourceGraphic"/> 
                            </feMerge>
                          </filter>
                        </defs>
                      </g>
                    );
                  }}
                />
                
                {/* Trade count line (secondary) */}
                <Line
                  yAxisId="trades"
                  type="monotone"
                  dataKey="trade_count"
                  stroke="#10b981"
                  strokeWidth={3}
                  dot={{ r: 3, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }}
                  activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 3, fill: '#fff', filter: 'drop-shadow(0 2px 4px rgba(16, 185, 129, 0.3))' }}
                  name="Trade Count"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className="mt-8 text-center">
        <div className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50">
          <span className="text-sm font-semibold text-gray-700">
            Connected to Kalshi data feed
          </span>
        </div>
      </div>
    </div>
  );
};

export default UnifiedAnalytics;