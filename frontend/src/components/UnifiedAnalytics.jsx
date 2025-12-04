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
    return `$${(volume / 1000000).toFixed(2)}M`;
  } else if (absVolume >= 1000) {
    return `$${(volume / 1000).toFixed(2)}k`;
  } else {
    return `$${volume}`;
  }
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

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const volumeData = payload.find(p => p.dataKey === 'volume_usd');
      const tradeData = payload.find(p => p.dataKey === 'trade_count');
      
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-gray-900">{`Time: ${label}`}</p>
          {volumeData && (
            <p className="text-blue-600">
              {`Volume: $${volumeData.value?.toLocaleString()}`}
            </p>
          )}
          {tradeData && (
            <p className="text-green-600">
              {`Trades: ${tradeData.value}`}
            </p>
          )}
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

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 mb-8">
      {/* Title Section with Toggle */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Market Analytics
        </h1>
        <p className="text-gray-600 text-lg mb-4">
          Real-time trading activity and market statistics
        </p>
        
        {/* Time Mode Toggle */}
        <div className="inline-flex bg-white rounded-lg p-1 shadow-sm border border-gray-200">
          <button
            onClick={() => setTimeMode('hour')}
            className={`px-4 py-2 rounded-md font-medium text-sm transition-all duration-200 ${
              timeMode === 'hour'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Hour View
          </button>
          <button
            onClick={() => setTimeMode('day')}
            className={`px-4 py-2 rounded-md font-medium text-sm transition-all duration-200 ${
              timeMode === 'day'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Day View
          </button>
        </div>
      </div>

      {/* Summary Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">
            {formatVolume(summaryStats.peak_volume_usd || 0)}
          </div>
          <div className="text-sm text-gray-600">Peak Volume</div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            {(summaryStats.peak_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Peak Trades</div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">
            {formatVolume(summaryStats.total_volume_usd || 0)}
          </div>
          <div className="text-sm text-gray-600">Total Volume</div>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 text-center">
          <div className="text-2xl font-bold text-indigo-600">
            {(summaryStats.total_trades || 0).toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Total Trades</div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-gray-900">{title}</h2>
            {(currentVolume > 0 || currentTrades > 0) && (
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-600">Live</span>
                </div>
                <div className="bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                  <span className="text-green-700 font-medium">
                    Current: ${currentVolume.toLocaleString()} • {currentTrades} trades
                  </span>
                </div>
              </div>
            )}
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-600 mt-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span>Historical {periodLabel}s</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>Current {periodLabel} (Live)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>Trade Count</span>
            </div>
          </div>
        </div>
        
        {chartData.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-32 mx-auto mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-24 mx-auto"></div>
              </div>
              <p className="mt-2">Loading analytics data...</p>
            </div>
          </div>
        ) : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartData}
                margin={{
                  top: 10,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                
                {/* X Axis - Time */}
                <XAxis 
                  dataKey="timeString"
                  tick={{ fontSize: 12, fill: '#666' }}
                  tickLine={{ stroke: '#ccc' }}
                  axisLine={{ stroke: '#ccc' }}
                  interval="preserveStartEnd"
                />
                
                {/* Left Y Axis - Volume */}
                <YAxis 
                  yAxisId="volume"
                  orientation="left"
                  tick={{ fontSize: 12, fill: '#2563eb' }}
                  tickLine={{ stroke: '#2563eb' }}
                  axisLine={{ stroke: '#2563eb' }}
                  tickFormatter={(value) => `$${value.toLocaleString()}`}
                  domain={[0, maxVolume * 1.1]}
                />
                
                {/* Right Y Axis - Trade Count */}
                <YAxis 
                  yAxisId="trades"
                  orientation="right"
                  tick={{ fontSize: 12, fill: '#16a34a' }}
                  tickLine={{ stroke: '#16a34a' }}
                  axisLine={{ stroke: '#16a34a' }}
                  domain={[0, maxTrades * 1.1]}
                />
                
                <Tooltip content={<CustomTooltip />} />
                
                {/* Volume bars (primary) */}
                <Bar
                  yAxisId="volume"
                  dataKey="volume_usd"
                  fill="#3b82f6"
                  fillOpacity={0.7}
                  radius={[2, 2, 0, 0]}
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
                          fillOpacity={isCurrentPeriod ? 0.9 : 0.7}
                          rx={2}
                          ry={2}
                        />
                        {isCurrentPeriod && (
                          <rect
                            x={x}
                            y={y}
                            width={width}
                            height={height}
                            fill="#10b981"
                            fillOpacity={0.3}
                            rx={2}
                            ry={2}
                          >
                            <animate
                              attributeName="fillOpacity"
                              values="0.3;0.6;0.3"
                              dur="2s"
                              repeatCount="indefinite"
                            />
                          </rect>
                        )}
                      </g>
                    );
                  }}
                />
                
                {/* Trade count line (secondary) */}
                <Line
                  yAxisId="trades"
                  type="monotone"
                  dataKey="trade_count"
                  stroke="#16a34a"
                  strokeWidth={3}
                  dot={{ r: 4, fill: '#16a34a' }}
                  activeDot={{ r: 6, stroke: '#16a34a', strokeWidth: 2, fill: '#fff' }}
                  name="Trade Count"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Connection Status */}
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

export default UnifiedAnalytics;