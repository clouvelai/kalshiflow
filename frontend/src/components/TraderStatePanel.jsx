import React, { useState } from 'react';

const TraderStatePanel = ({ 
  state, 
  executionStats, 
  showExecutionStats = true, 
  showOnlyExecutionStats = false,
  showPositions = true,
  showOrders = true,
  showActionBreakdown = true 
}) => {
  const [showCents, setShowCents] = useState(false);
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

  // Helper function to format value based on cents/dollars mode
  const formatValueFromKalshi = (position, field) => {
    // Use Kalshi's exact field names - prefer dollars format if available
    const dollarsField = `${field}_dollars`;
    const centsField = field;
    
    if (showCents) {
      // Show cents value
      const centsValue = position[centsField];
      if (centsValue !== undefined && centsValue !== null) {
        return `${centsValue.toLocaleString()}¢`;
      }
      // If no cents value, convert from dollars
      const dollarsValue = position[dollarsField];
      if (dollarsValue !== undefined && dollarsValue !== null) {
        return `${Math.round(parseFloat(dollarsValue) * 100).toLocaleString()}¢`;
      }
      return '--';
    } else {
      // Show dollars value
      const dollarsValue = position[dollarsField];
      if (dollarsValue !== undefined && dollarsValue !== null) {
        return formatCurrency(parseFloat(dollarsValue));
      }
      // If no dollars value, convert from cents
      const centsValue = position[centsField];
      if (centsValue !== undefined && centsValue !== null) {
        return formatCurrency(centsValue / 100);
      }
      return '--';
    }
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

  // If only showing execution stats, return just that section
  if (showOnlyExecutionStats) {
    return (
      <div className="space-y-3">
        {/* Primary Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-3 hover:bg-gray-700/70 transition-colors">
            <p className="text-xs text-gray-400 mb-1">Total Fills</p>
            <p className={`text-xl font-mono font-bold ${
              (executionStats?.total_fills || 0) > 0 ? 'text-green-400' : 'text-gray-300'
            }`}>
              {executionStats?.total_fills || 0}
            </p>
          </div>
          
          <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-3 hover:bg-gray-700/70 transition-colors">
            <p className="text-xs text-gray-400 mb-1">Win Rate</p>
            <p className={`text-xl font-mono font-bold ${
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
        <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-3 hover:bg-gray-700/70 transition-colors">
          <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">Fill Breakdown</h4>
          
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded font-medium">MAKER</span>
              <span className="text-sm font-mono text-gray-300">
                {executionStats?.maker_fills || 0}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-xs px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded font-medium">TAKER</span>
              <span className="text-sm font-mono text-gray-300">
                {executionStats?.taker_fills || 0}
              </span>
            </div>
          </div>

          {/* Fill ratio bar */}
          {(executionStats?.total_fills || 0) > 0 && (
            <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
              <div className="h-full flex">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                  style={{ 
                    width: `${((executionStats?.maker_fills || 0) / executionStats.total_fills) * 100}%` 
                  }}
                />
                <div 
                  className="bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
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
          <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-3 hover:bg-gray-700/70 transition-colors">
            <p className="text-xs text-gray-400 mb-1">Success Rate</p>
            <p className={`text-lg font-mono font-bold ${
              executionStats?.success_rate ? 
                (executionStats.success_rate >= 0.8 ? 'text-green-400' : 
                 executionStats.success_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400') : 
                'text-gray-300'
            }`}>
              {executionStats?.success_rate ? `${(executionStats.success_rate * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          
          <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-3 hover:bg-gray-700/70 transition-colors">
            <p className="text-xs text-gray-400 mb-1">Total P&L</p>
            <p className={`text-lg font-mono font-bold ${
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
    );
  }

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

      {/* Enhanced Positions Section */}
      {showPositions && (
      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        <div className="bg-gray-700/30 px-4 py-3 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-100">📊 Open Positions</h3>
            <div className="flex items-center space-x-3">
              {displayState.positions && Object.keys(displayState.positions).length > 0 && (
                <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full font-medium">
                  {Object.keys(displayState.positions).length} Active
                </span>
              )}
              <button
                onClick={() => setShowCents(!showCents)}
                className="text-xs px-2 py-1 bg-gray-600/50 hover:bg-gray-600/80 text-gray-300 rounded font-medium transition-colors"
              >
                {showCents ? '$' : '¢'}
              </button>
            </div>
          </div>
        </div>
        <div className="p-4">
          {displayState.positions && Object.keys(displayState.positions).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(displayState.positions).map(([key, position]) => {
                // Extract ticker - it might be in the position object or use the key
                const ticker = position.ticker || position.market_ticker || key;
                const percentChange = position.avg_price ? ((position.current_price - position.avg_price) / position.avg_price * 100) : 0;
                const isProfit = position.unrealized_pnl >= 0;
                
                return (
                  <div key={key} className="group bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 transition-all duration-200 border border-gray-700 hover:border-gray-600">
                    {/* Main Position Info */}
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <p className="font-semibold text-white text-sm">{ticker}</p>
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            position.side === 'YES' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {position.side}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400">
                          <span className="font-mono">{position.contracts}</span> contracts
                        </p>
                      </div>
                      <div className="text-right">
                        <p className={`font-bold text-sm ${
                          (position.realized_pnl_dollars ? parseFloat(position.realized_pnl_dollars) : (position.realized_pnl || 0) / 100) >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(position.realized_pnl_dollars ? parseFloat(position.realized_pnl_dollars) : (position.realized_pnl || 0) / 100) >= 0 ? '+' : ''}{formatValueFromKalshi(position, 'realized_pnl')}
                        </p>
                        <p className="text-xs text-gray-400">Realized P&L</p>
                      </div>
                    </div>
                    
                    {/* Kalshi Position Details */}
                    <div className="grid grid-cols-2 gap-2 text-xs mt-3 pt-2 border-t border-gray-700/50">
                      <div>
                        <p className="text-gray-500">Cost Basis</p>
                        <p className="text-gray-300 font-mono">{formatCurrency(position.cost_basis || 0)}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Market Exposure</p>
                        <p className="text-gray-300 font-mono">{formatValueFromKalshi(position, 'market_exposure')}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Total Traded</p>
                        <p className="text-gray-300 font-mono">{formatValueFromKalshi(position, 'total_traded')}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Fees Paid</p>
                        <p className="text-gray-300 font-mono">{formatValueFromKalshi(position, 'fees_paid')}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Realized P&L</p>
                        <p className="text-gray-300 font-mono">{formatValueFromKalshi(position, 'realized_pnl')}</p>
                      </div>
                    </div>
                    {position.last_updated_ts && (
                      <div className="mt-2 text-xs">
                        <p className="text-gray-500">Last Updated</p>
                        <p className="text-gray-400 font-mono">
                          {new Date(position.last_updated_ts).toLocaleString()}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-6">
              <p className="text-xs text-gray-500">No open positions</p>
              <p className="text-xs text-gray-600 mt-1">Waiting for trading signals...</p>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Enhanced Open Orders Section */}
      {showOrders && (
      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
        <div className="bg-gray-700/30 px-4 py-3 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-100">📋 Open Orders</h3>
            {displayState.open_orders && displayState.open_orders.length > 0 && (
              <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full font-medium">
                {displayState.open_orders.length} Pending
              </span>
            )}
          </div>
        </div>
        <div className="p-4">
          {displayState.open_orders && displayState.open_orders.length > 0 ? (
            <div className="space-y-2">
              {displayState.open_orders.slice(0, 5).map((order, idx) => {
                const orderTime = order.created_at ? new Date(order.created_at) : null;
                const timeElapsed = orderTime ? Date.now() - orderTime.getTime() : 0;
                const minutesElapsed = Math.floor(timeElapsed / 60000);
                const secondsElapsed = Math.floor((timeElapsed % 60000) / 1000);
                
                return (
                  <div key={idx} className="group bg-gray-700/30 hover:bg-gray-700/50 rounded-lg p-3 transition-all duration-200 border border-gray-700 hover:border-gray-600">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                            order.side === 'BUY' || order.side === 'YES' ? 
                              'bg-green-500/20 text-green-400' : 
                              'bg-red-500/20 text-red-400'
                          }`}>
                            {order.side}
                          </span>
                          <span className="text-xs px-1.5 py-0.5 bg-gray-600/50 text-gray-300 rounded">
                            {order.order_type || 'LIMIT'}
                          </span>
                          {order.ticker && (
                            <p className="text-xs text-gray-400 font-medium">{order.ticker}</p>
                          )}
                        </div>
                        <p className="text-xs text-gray-300">
                          <span className="font-mono">{order.quantity}</span> contracts @ 
                          <span className="font-mono text-white ml-1">{formatCurrency(order.price/100)}</span>
                        </p>
                      </div>
                      <div className="text-right">
                        {order.current_price !== undefined && (
                          <p className="text-xs text-gray-400 mb-1">
                            Market: <span className="font-mono">{formatCurrency(order.current_price/100)}</span>
                          </p>
                        )}
                        {orderTime && (
                          <p className="text-xs text-gray-500">
                            {minutesElapsed > 0 ? `${minutesElapsed}m ` : ''}{secondsElapsed}s ago
                          </p>
                        )}
                      </div>
                    </div>
                    {order.current_price !== undefined && (
                      <div className="mt-2 pt-2 border-t border-gray-700/50">
                        <div className="flex justify-between items-center">
                          <p className="text-xs text-gray-500">Distance from market:</p>
                          <p className={`text-xs font-mono ${
                            Math.abs(order.price - order.current_price) < 5 ? 'text-green-400' : 
                            Math.abs(order.price - order.current_price) < 10 ? 'text-yellow-400' : 
                            'text-gray-400'
                          }`}>
                            {formatCurrency(Math.abs(order.price - order.current_price)/100)} 
                            ({((Math.abs(order.price - order.current_price) / order.current_price) * 100).toFixed(1)}%)
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
              {displayState.open_orders.length > 5 && (
                <p className="text-xs text-gray-500 text-center mt-2">
                  +{displayState.open_orders.length - 5} more orders
                </p>
              )}
            </div>
          ) : (
            <div className="text-center py-6">
              <p className="text-xs text-gray-500">No open orders</p>
              <p className="text-xs text-gray-600 mt-1">Orders will appear here when placed</p>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Execution Statistics - Only show if showExecutionStats is true */}
      {showExecutionStats && (
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
      )}

      {/* Action Breakdown */}
      {showActionBreakdown && displayState.actor_metrics?.action_counts && (
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