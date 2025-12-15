import React from 'react';

const TraderStatePanel = ({ state }) => {
  if (!state) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-gray-400">
        <p>Waiting for trader state...</p>
      </div>
    );
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value || 0);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-4">
      {/* Actor Configuration */}
      {state.actor && (
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <h2 className="text-xl font-semibold text-white mb-3">Actor</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-400 text-sm">Strategy</p>
              <p className="text-lg font-mono text-blue-400">{state.actor.strategy}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Status</p>
              <p className={`text-lg font-mono ${state.actor.enabled ? 'text-green-400' : 'text-red-400'}`}>
                {state.actor.enabled ? 'ENABLED' : 'DISABLED'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-sm">Throttle</p>
              <p className="text-lg font-mono text-white">{state.actor.throttle_ms}ms</p>
            </div>
          </div>
        </div>
      )}

      {/* Markets Tracking */}
      {state.markets && (
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <h2 className="text-xl font-semibold text-white mb-3">Markets</h2>
          <div className="mb-3">
            <div className="flex justify-between items-center">
              <p className="text-gray-400 text-sm">Tracked Markets</p>
              <p className="text-lg font-mono text-white">{state.markets.market_count}</p>
            </div>
            <div className="mt-2">
              <div className="flex flex-wrap gap-1">
                {state.markets.tracked_tickers?.map((ticker) => (
                  <span key={ticker} className="px-2 py-1 bg-gray-700 text-white font-mono text-xs rounded">
                    {ticker}
                  </span>
                ))}
              </div>
            </div>
          </div>
          {state.markets.per_market_activity && Object.keys(state.markets.per_market_activity).length > 0 && (
            <div>
              <p className="text-gray-400 text-sm mb-2">Market Activity</p>
              <div className="space-y-1">
                {Object.entries(state.markets.per_market_activity).map(([market, activity]) => (
                  <div key={market} className="flex justify-between items-center">
                    <span className="font-mono text-xs text-white">{market}</span>
                    <span className="text-xs text-gray-300">
                      Orders: {activity.total_orders || 0} | Fills: {activity.total_fills || 0}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Portfolio Summary - matches MarketGrid card styling */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Portfolio</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-400 text-sm">Cash Available</p>
            <p className="text-2xl font-mono text-white">
              {formatCurrency(state.cash_balance)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Portfolio Value</p>
            <p className="text-2xl font-mono text-white">
              {formatCurrency(state.portfolio_value)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Promised Cash</p>
            <p className="text-xl font-mono text-yellow-400">
              {formatCurrency(state.promised_cash)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Daily P&L</p>
            <p className={`text-xl font-mono ${
              state.metrics?.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {formatCurrency(state.metrics?.daily_pnl || 0)}
            </p>
          </div>
        </div>
      </div>

      {/* Positions */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Positions</h2>
        {Object.keys(state.positions || {}).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 text-gray-400">Market</th>
                  <th className="text-right py-2 text-gray-400">Contracts</th>
                  <th className="text-right py-2 text-gray-400">Cost Basis</th>
                  <th className="text-right py-2 text-gray-400">Realized P&L</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(state.positions).map(([market, position]) => (
                  <tr key={market} className="border-b border-gray-700">
                    <td className="py-2 text-white font-mono text-xs">{market}</td>
                    <td className="text-right py-2 text-white">{position.contracts}</td>
                    <td className="text-right py-2 text-gray-300">
                      {formatCurrency(position.cost_basis)}
                    </td>
                    <td className={`text-right py-2 ${
                      position.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatCurrency(position.realized_pnl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-400 text-sm">No open positions</p>
        )}
      </div>

      {/* Open Orders */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Open Orders</h2>
        {(state.open_orders || []).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 text-gray-400">Market</th>
                  <th className="text-left py-2 text-gray-400">Side</th>
                  <th className="text-right py-2 text-gray-400">Qty</th>
                  <th className="text-right py-2 text-gray-400">Price</th>
                  <th className="text-left py-2 text-gray-400">Status</th>
                </tr>
              </thead>
              <tbody>
                {state.open_orders.map((order, idx) => (
                  <tr key={order.order_id || idx} className="border-b border-gray-700">
                    <td className="py-2 text-white font-mono text-xs">{order.ticker}</td>
                    <td className="py-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        order.side === 'BUY' 
                          ? 'bg-green-900 text-green-400' 
                          : 'bg-red-900 text-red-400'
                      }`}>
                        {order.side} {order.contract_side}
                      </span>
                    </td>
                    <td className="text-right py-2 text-white">{order.quantity}</td>
                    <td className="text-right py-2 text-white">{order.limit_price}¢</td>
                    <td className="py-2">
                      <span className="text-yellow-400 text-xs">{order.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-400 text-sm">No open orders</p>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Performance</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-400 text-sm">Orders Placed</p>
            <p className="text-xl font-mono text-white">{state.metrics?.orders_placed || 0}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Orders Filled</p>
            <p className="text-xl font-mono text-white">{state.metrics?.orders_filled || 0}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Fill Rate</p>
            <p className="text-xl font-mono text-white">
              {formatPercentage(state.metrics?.fill_rate || 0)}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Volume Traded</p>
            <p className="text-xl font-mono text-white">
              {formatCurrency(state.metrics?.total_volume_traded || 0)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TraderStatePanel;