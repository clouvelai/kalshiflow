import React from 'react';

const TraderStatePanel = ({ state, executionStats }) => {
  // Helper function to format currency
  const formatCurrency = (amount) => {
    if (!amount && amount !== 0) return '$--';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  // Helper function to format percentage
  const formatPercent = (value) => {
    if (!value && value !== 0) return '--';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  // Helper function to format numbers
  const formatNumber = (num) => {
    if (!num && num !== 0) return '--';
    return num.toLocaleString();
  };

  // Default empty state
  const emptyState = {
    environment: 'paper',
    portfolio_value: 0,
    cash_balance: 0,
    positions: {},
    open_orders: [],
    recent_fills: [],
    metrics: {
      orders_placed: 0,
      fill_rate: 0,
      win_rate: 0,
      daily_pnl: 0
    }
  };

  const displayState = state || emptyState;

  return (
    <div className="space-y-4">
      {/* Portfolio Summary */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Portfolio Summary</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-500">Total Value</p>
            <p className="text-lg font-bold text-white">
              {formatCurrency(displayState.portfolio_value)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Cash Balance</p>
            <p className="text-lg font-bold text-white">
              {formatCurrency(displayState.cash_balance)}
            </p>
          </div>
        </div>
        
        {/* Zero Balance Warning */}
        {displayState.cash_balance === 0 && displayState.portfolio_value === 0 && (
          <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded">
            <p className="text-xs text-yellow-400">
              ⚠️ <span className="font-semibold">Zero Balance Detected</span>
            </p>
            <p className="text-xs text-yellow-400/80 mt-1">
              Trading disabled - Demo account needs funding. Add balance through Kalshi demo account UI.
            </p>
          </div>
        )}
      </div>

      {/* Positions */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Open Positions</h3>
        {displayState.positions && Object.keys(displayState.positions).length > 0 ? (
          <div className="space-y-2">
            {Object.entries(displayState.positions).map(([ticker, position]) => (
              <div key={ticker} className="flex justify-between items-start text-xs">
                <div className="flex-1">
                  <p className="font-medium text-white">{ticker}</p>
                  <p className="text-gray-400">
                    {position.contracts} {position.side} @ {formatCurrency(position.avg_price/100)}
                  </p>
                </div>
                <div className="text-right">
                  <p className={`font-medium ${position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatCurrency(position.unrealized_pnl)}
                  </p>
                  <p className="text-gray-400">
                    {formatPercent((position.current_price - position.avg_price) / position.avg_price * 100)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-500">No open positions</p>
        )}
      </div>

      {/* Open Orders */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Open Orders</h3>
        {displayState.open_orders && displayState.open_orders.length > 0 ? (
          <div className="space-y-1">
            {displayState.open_orders.slice(0, 5).map((order, idx) => (
              <div key={idx} className="text-xs text-gray-300">
                {order.side} {order.quantity} @ {formatCurrency(order.price/100)}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-500">No open orders</p>
        )}
      </div>

      {/* Execution Statistics */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Execution Statistics</h3>
        <div className="space-y-3">
          {/* Primary Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-700 border border-gray-600 rounded p-2">
              <p className="text-xs text-gray-400">Total Fills</p>
              <p className={`text-lg font-mono ${
                (executionStats?.total_fills || 0) > 0 ? 'text-green-400' : 'text-gray-300'
              }`}>
                {executionStats?.total_fills || 0}
              </p>
            </div>
            
            <div className="bg-gray-700 border border-gray-600 rounded p-2">
              <p className="text-xs text-gray-400">Win Rate</p>
              <p className={`text-lg font-mono ${
                executionStats?.win_rate ? 
                  (executionStats.win_rate >= 0.6 ? 'text-green-400' : 
                   executionStats.win_rate >= 0.4 ? 'text-yellow-400' : 'text-red-400') : 
                  'text-gray-300'
              }`}>
                {executionStats?.win_rate ? `${(executionStats.win_rate * 100).toFixed(1)}%` : '--'}
              </p>
            </div>
          </div>

          {/* Fill Types */}
          <div className="bg-gray-700 border border-gray-600 rounded p-3 space-y-2">
            <h4 className="text-xs font-semibold text-gray-400 uppercase">Fill Breakdown</h4>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">MAKER</span>
                <span className="text-sm font-mono text-gray-300">
                  {executionStats?.maker_fills || 0}
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className="text-xs px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded">TAKER</span>
                <span className="text-sm font-mono text-gray-300">
                  {executionStats?.taker_fills || 0}
                </span>
              </div>
            </div>

            {/* Fill ratio bar */}
            {(executionStats?.total_fills || 0) > 0 && (
              <div className="w-full h-2 bg-gray-600 rounded overflow-hidden">
                <div className="h-full flex">
                  <div 
                    className="bg-blue-400 transition-all duration-500"
                    style={{ 
                      width: `${((executionStats?.maker_fills || 0) / executionStats.total_fills) * 100}%` 
                    }}
                  />
                  <div 
                    className="bg-purple-400 transition-all duration-500"
                    style={{ 
                      width: `${((executionStats?.taker_fills || 0) / executionStats.total_fills) * 100}%` 
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-700 border border-gray-600 rounded p-2">
              <p className="text-xs text-gray-400">Success Rate</p>
              <p className={`text-sm font-mono ${
                executionStats?.success_rate ? 
                  (executionStats.success_rate >= 0.8 ? 'text-green-400' : 
                   executionStats.success_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400') : 
                  'text-gray-300'
              }`}>
                {executionStats?.success_rate ? `${(executionStats.success_rate * 100).toFixed(1)}%` : '--'}
              </p>
            </div>
            
            <div className="bg-gray-700 border border-gray-600 rounded p-2">
              <p className="text-xs text-gray-400">Total P&L</p>
              <p className={`text-sm font-mono ${
                executionStats?.total_pnl ? 
                  (executionStats.total_pnl > 0 ? 'text-green-400' : 
                   executionStats.total_pnl < 0 ? 'text-red-400' : 'text-gray-400') : 
                  'text-gray-300'
              }`}>
                {executionStats?.total_pnl ? 
                  `${executionStats.total_pnl > 0 ? '+' : ''}${formatCurrency(executionStats.total_pnl)}` : 
                  '--'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Breakdown */}
      {displayState.actor_metrics?.action_counts && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Action Breakdown</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-gray-500">Hold</p>
              <p className="text-sm font-medium text-amber-400">
                {formatNumber(displayState.actor_metrics.action_counts.hold || 0)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Buy YES</p>
              <p className="text-sm font-medium text-green-400">
                {formatNumber(displayState.actor_metrics.action_counts.buy_yes || 0)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Sell YES</p>
              <p className="text-sm font-medium text-red-400">
                {formatNumber(displayState.actor_metrics.action_counts.sell_yes || 0)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Buy NO</p>
              <p className="text-sm font-medium text-purple-400">
                {formatNumber(displayState.actor_metrics.action_counts.buy_no || 0)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Sell NO</p>
              <p className="text-sm font-medium text-blue-400">
                {formatNumber(displayState.actor_metrics.action_counts.sell_no || 0)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Total</p>
              <p className="text-sm font-medium text-white">
                {formatNumber(displayState.actor_metrics.total_actions || 0)}
              </p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default TraderStatePanel;