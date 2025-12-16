import React from 'react';

const ExecutionStats = ({ stats }) => {
  // Helper function to format percentage
  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '--';
    return `${(value * 100).toFixed(1)}%`;
  };

  // Helper function to format time
  const formatTime = (seconds) => {
    if (seconds === null || seconds === undefined) return '--';
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${(seconds / 60).toFixed(1)}m`;
  };

  // Helper function to get color for win rate
  const getWinRateColor = (rate) => {
    if (rate === null || rate === undefined) return 'text-gray-400';
    if (rate >= 0.6) return 'text-green-400';
    if (rate >= 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  // Default stats if none provided
  const defaultStats = {
    total_fills: 0,
    maker_fills: 0,
    taker_fills: 0,
    avg_fill_time: null,
    win_rate: null,
    total_volume: 0,
    total_pnl: 0,
    success_rate: null,
    cancelled_orders: 0,
    rejected_orders: 0
  };

  const displayStats = stats || defaultStats;

  return (
    <div className="space-y-3">
      {/* Primary Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Total Fills"
          value={displayStats.total_fills || 0}
          format="number"
          highlight={displayStats.total_fills > 0}
        />
        
        <StatCard
          label="Win Rate"
          value={displayStats.win_rate}
          format="percentage"
          color={getWinRateColor(displayStats.win_rate)}
        />
      </div>

      {/* Fill Types */}
      <div className="bg-gray-700/30 rounded p-3 space-y-2">
        <h4 className="text-xs font-semibold text-gray-400 uppercase">Fill Breakdown</h4>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">MAKER</span>
            <span className="text-sm font-mono text-gray-300">
              {displayStats.maker_fills || 0}
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-xs px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded">TAKER</span>
            <span className="text-sm font-mono text-gray-300">
              {displayStats.taker_fills || 0}
            </span>
          </div>
        </div>

        {/* Fill ratio bar */}
        {displayStats.total_fills > 0 && (
          <div className="w-full h-2 bg-gray-600 rounded overflow-hidden">
            <div className="h-full flex">
              <div 
                className="bg-blue-400 transition-all duration-500"
                style={{ 
                  width: `${(displayStats.maker_fills / displayStats.total_fills) * 100}%` 
                }}
              />
              <div 
                className="bg-purple-400 transition-all duration-500"
                style={{ 
                  width: `${(displayStats.taker_fills / displayStats.total_fills) * 100}%` 
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Avg Fill Time"
          value={displayStats.avg_fill_time}
          format="time"
        />
        
        <StatCard
          label="Success Rate"
          value={displayStats.success_rate}
          format="percentage"
          color={displayStats.success_rate >= 0.8 ? 'text-green-400' : 
                 displayStats.success_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400'}
        />
      </div>

      {/* Volume and P&L */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Total Volume"
          value={displayStats.total_volume}
          format="currency"
        />
        
        <StatCard
          label="Total P&L"
          value={displayStats.total_pnl}
          format="currency"
          color={displayStats.total_pnl > 0 ? 'text-green-400' : 
                 displayStats.total_pnl < 0 ? 'text-red-400' : 'text-gray-400'}
          showSign={true}
        />
      </div>

      {/* Order Issues */}
      {(displayStats.cancelled_orders > 0 || displayStats.rejected_orders > 0) && (
        <div className="bg-red-900/20 rounded p-2 space-y-1 border border-red-800/50">
          <div className="flex justify-between items-center">
            <span className="text-xs text-red-400">Cancelled</span>
            <span className="text-xs font-mono text-red-300">{displayStats.cancelled_orders}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-red-400">Rejected</span>
            <span className="text-xs font-mono text-red-300">{displayStats.rejected_orders}</span>
          </div>
        </div>
      )}

      {/* Additional Metrics (if available) */}
      {displayStats.sharpe_ratio !== undefined && (
        <div className="pt-2 border-t border-gray-700">
          <StatCard
            label="Sharpe Ratio"
            value={displayStats.sharpe_ratio}
            format="decimal"
            color={displayStats.sharpe_ratio > 1 ? 'text-green-400' : 
                   displayStats.sharpe_ratio > 0 ? 'text-yellow-400' : 'text-red-400'}
          />
        </div>
      )}
    </div>
  );
};

// Stat Card Component
const StatCard = ({ label, value, format = 'number', color = 'text-gray-300', highlight = false, showSign = false }) => {
  // Format the displayed value
  const formatValue = (val) => {
    if (val === null || val === undefined) return '--';
    
    switch (format) {
      case 'currency':
        const sign = showSign && val > 0 ? '+' : '';
        return `${sign}$${Math.abs(val).toFixed(2)}`;
      case 'percentage':
        return `${(val * 100).toFixed(1)}%`;
      case 'time':
        if (val < 1) return `${(val * 1000).toFixed(0)}ms`;
        if (val < 60) return `${val.toFixed(1)}s`;
        return `${(val / 60).toFixed(1)}m`;
      case 'decimal':
        return val.toFixed(2);
      case 'number':
      default:
        return val.toString();
    }
  };

  return (
    <div className={`bg-gray-700/30 rounded p-2 ${highlight ? 'ring-1 ring-blue-500/50' : ''}`}>
      <p className="text-xs text-gray-400">{label}</p>
      <p className={`text-lg font-mono ${color}`}>
        {formatValue(value)}
      </p>
    </div>
  );
};

export default ExecutionStats;