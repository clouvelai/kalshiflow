import React from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const AnalyticsChart = ({ analyticsData = [], analyticsSummary = {}, title = "Trading Activity (Last Hour)" }) => {
  // Get current minute timestamp for highlighting
  const getCurrentMinuteTimestamp = () => {
    const now = new Date();
    const minuteStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 
                                  now.getHours(), now.getMinutes(), 0, 0);
    return minuteStart.getTime();
  };
  
  const currentMinuteTimestamp = getCurrentMinuteTimestamp();
  // Format data for recharts
  const formatChartData = (data) => {
    return data.map(point => ({
      ...point,
      // Convert timestamp to time string for display
      timeString: new Date(point.timestamp).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      }),
      // Format volume for display
      volumeDisplayed: point.volume_usd?.toFixed(0) || 0,
      // Mark if this is the current minute
      isCurrentMinute: point.timestamp === currentMinuteTimestamp
    }));
  };

  const chartData = formatChartData(analyticsData);

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

  // Find current minute data
  const currentMinuteData = chartData.find(d => d.isCurrentMinute);
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-900">{title}</h2>
          {(currentMinuteData || analyticsSummary.current_minute_volume_usd > 0) && (
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-gray-600">Live</span>
              </div>
              <div className="bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                <span className="text-green-700 font-medium">
                  Current: ${(analyticsSummary.current_minute_volume_usd || currentMinuteData?.volume_usd || 0).toLocaleString()} • {(analyticsSummary.current_minute_trades || currentMinuteData?.trade_count || 0)} trades
                </span>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-6 text-sm text-gray-600 mt-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span>Historical Minutes</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>Current Minute (Live)</span>
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
                fill={(data) => data.isCurrentMinute ? "#10b981" : "#3b82f6"}
                fillOpacity={(data) => data.isCurrentMinute ? 0.9 : 0.7}
                radius={[2, 2, 0, 0]}
                name="Volume (USD)"
                shape={(props) => {
                  const { fill, x, y, width, height, payload } = props;
                  const isCurrentMinute = payload.isCurrentMinute;
                  
                  return (
                    <g>
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={isCurrentMinute ? "#10b981" : fill}
                        fillOpacity={isCurrentMinute ? 0.9 : 0.7}
                        rx={2}
                        ry={2}
                      />
                      {isCurrentMinute && (
                        <>
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
                        </>
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
              
              {/* Remove static reference line since current minute is now highlighted */}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {/* Summary stats below chart - use backend calculations when available, fall back to frontend calculations */}
      {(chartData.length > 0 || Object.keys(analyticsSummary).length > 0) && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="font-semibold text-blue-600">
                ${(analyticsSummary.peak_volume_usd !== undefined 
                  ? analyticsSummary.peak_volume_usd 
                  : Math.max(...chartData.map(d => d.volume_usd || 0))
                ).toLocaleString()}
              </div>
              <div className="text-gray-500">Peak Volume</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-green-600">
                {analyticsSummary.peak_trades !== undefined 
                  ? analyticsSummary.peak_trades 
                  : Math.max(...chartData.map(d => d.trade_count || 0))}
              </div>
              <div className="text-gray-500">Peak Trades</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-purple-600">
                ${(analyticsSummary.total_volume_usd !== undefined 
                  ? analyticsSummary.total_volume_usd 
                  : chartData.reduce((sum, d) => sum + (d.volume_usd || 0), 0)
                ).toLocaleString()}
              </div>
              <div className="text-gray-500">Total Volume</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-indigo-600">
                {analyticsSummary.total_trades !== undefined 
                  ? analyticsSummary.total_trades 
                  : chartData.reduce((sum, d) => sum + (d.trade_count || 0), 0)}
              </div>
              <div className="text-gray-500">Total Trades</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyticsChart;