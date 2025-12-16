import React from 'react';
import TradesFeed from './TradesFeed';

const TraderStatePanel = ({ state }) => {
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
      <div className="bg-gray-700 rounded-lg p-4">
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
      </div>

      {/* Positions */}
      <div className="bg-gray-700 rounded-lg p-4">
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
      <div className="bg-gray-700 rounded-lg p-4">
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

      {/* Trading Metrics */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Trading Metrics</h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-gray-500">Orders Placed</p>
            <p className="text-sm font-medium text-white">
              {formatNumber(displayState.metrics?.orders_placed)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Fill Rate</p>
            <p className="text-sm font-medium text-white">
              {displayState.metrics?.fill_rate ? `${(displayState.metrics.fill_rate * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Win Rate</p>
            <p className="text-sm font-medium text-white">
              {displayState.metrics?.win_rate ? `${(displayState.metrics.win_rate * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Daily P&L</p>
            <p className={`text-sm font-medium ${
              displayState.metrics?.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {formatCurrency(displayState.metrics?.daily_pnl)}
            </p>
          </div>
        </div>
      </div>

      {/* Recent Fills */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Recent Fills</h3>
        <TradesFeed fills={displayState.recent_fills || []} />
      </div>
    </div>
  );
};

export default TraderStatePanel;