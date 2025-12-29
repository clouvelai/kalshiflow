import React from 'react';
import {
  DollarSign,
  Briefcase,
  Database,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Activity,
  Shield,
  ShoppingCart,
  FileText,
  Clock,
  ArrowUpCircle,
  ArrowDownCircle,
  Hash
} from 'lucide-react';

/**
 * TradingSessionPanel - Unified trading session display
 * Shows balance, portfolio, P&L from Kalshi sync
 *
 * @param {Object} props
 * @param {Object} props.tradingState - Trading state from WebSocket
 * @param {number} props.lastUpdateTime - Last update timestamp
 */
const TradingSessionPanel = ({ tradingState, lastUpdateTime }) => {
  // Extract values with defaults - direct display, no animations
  const hasState = tradingState?.has_state ?? false;
  const balance = tradingState?.balance ?? 0;
  const portfolioValue = tradingState?.portfolio_value ?? 0;
  const totalValue = balance + portfolioValue;
  const positionCount = tradingState?.position_count ?? 0;
  const orderCount = tradingState?.order_count ?? 0;
  const pnl = tradingState?.pnl ?? null;
  const orderGroup = tradingState?.order_group ?? null;
  const syncTimestamp = tradingState?.sync_timestamp ?? null;

  // Format helpers
  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatPnLCurrency = (cents) => {
    const dollars = cents / 100;
    const prefix = cents >= 0 ? '+' : '';
    return prefix + new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(dollars);
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Early return for no state
  if (!hasState) {
    return (
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-gray-500" />
            <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Session</h3>
          </div>
          <span className="text-xs text-gray-500 font-mono">Waiting for data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Trading Session</h3>
        </div>
        <div className="flex items-center space-x-4">
          {pnl && pnl.session_start_time && (
            <div className="flex items-center space-x-2">
              <Clock className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-xs text-gray-500 font-mono">
                <span className="text-gray-600">Session:</span> {formatTime(pnl.session_start_time)}
              </span>
            </div>
          )}
          <div className="flex items-center space-x-2">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Sync:</span> {formatTime(syncTimestamp)}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Update:</span> {formatTime(lastUpdateTime)}
            </span>
          </div>
        </div>
      </div>

      {/* Main Value Cards - Row 1 */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        {/* Cash Available */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-500 uppercase">Cash Available</span>
          </div>
          <div className="text-2xl font-mono font-bold text-green-400">
            {formatCurrency(balance)}
          </div>
        </div>

        {/* In Positions */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <Briefcase className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-gray-500 uppercase">In Positions</span>
          </div>
          <div className="text-2xl font-mono font-bold text-blue-400">
            {formatCurrency(portfolioValue)}
          </div>
        </div>

        {/* Total Value */}
        <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/50">
          <div className="flex items-center space-x-2 mb-2">
            <Database className="w-4 h-4 text-white" />
            <span className="text-xs text-gray-500 uppercase">Total Value</span>
          </div>
          <div className="text-2xl font-mono font-bold text-white">
            {formatCurrency(totalValue)}
          </div>
        </div>

        {/* Session P&L */}
        <div className={`bg-gray-800/30 rounded-lg p-4 border ${
          pnl && pnl.session_pnl >= 0 ? 'border-green-700/30' : 'border-red-700/30'
        }`}>
          <div className="flex items-center space-x-2 mb-2">
            {pnl && pnl.session_pnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
            <span className="text-xs text-gray-500 uppercase">Session P&L</span>
            {pnl && pnl.session_pnl_percent !== undefined && (
              <span className={`text-xs font-mono ml-auto ${
                pnl.session_pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ({pnl.session_pnl_percent >= 0 ? '+' : ''}{pnl.session_pnl_percent.toFixed(1)}%)
              </span>
            )}
          </div>
          {pnl ? (
            <div className={`text-2xl font-mono font-bold ${
              pnl.session_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {formatPnLCurrency(pnl.session_pnl ?? 0)}
            </div>
          ) : (
            <div className="text-2xl font-mono font-bold text-gray-500">--</div>
          )}
        </div>
      </div>

      {/* Session Cash Flow - Row 2 */}
      {pnl && (
        <div className="grid grid-cols-4 gap-4 mb-4">
          {/* Cash Invested */}
          <div className="bg-gray-800/30 rounded-lg p-3 border border-amber-700/30">
            <div className="flex items-center space-x-2 mb-1">
              <ArrowUpCircle className="w-3 h-3 text-amber-400" />
              <span className="text-xs text-gray-500 uppercase" title="Cash spent on orders this session">
                Cash Invested
              </span>
            </div>
            <div className="text-lg font-mono font-bold text-amber-400">
              {formatCurrency(pnl.session_cash_invested ?? 0)}
            </div>
          </div>

          {/* Cash Received */}
          <div className="bg-gray-800/30 rounded-lg p-3 border border-cyan-700/30">
            <div className="flex items-center space-x-2 mb-1">
              <ArrowDownCircle className="w-3 h-3 text-cyan-400" />
              <span className="text-xs text-gray-500 uppercase" title="Cash received from settlements this session">
                Cash Received
              </span>
            </div>
            <div className="text-lg font-mono font-bold text-cyan-400">
              {formatCurrency(pnl.session_cash_received ?? 0)}
            </div>
          </div>

          {/* Orders Placed */}
          <div className="bg-gray-800/30 rounded-lg p-3 border border-purple-700/30">
            <div className="flex items-center space-x-2 mb-1">
              <Hash className="w-3 h-3 text-purple-400" />
              <span className="text-xs text-gray-500 uppercase" title="Orders placed this session">
                Orders Placed
              </span>
            </div>
            <div className="text-lg font-mono font-bold text-purple-400">
              {pnl.session_orders_count ?? 0}
            </div>
          </div>

          {/* Settled Positions */}
          <div className="bg-gray-800/30 rounded-lg p-3 border border-orange-700/30">
            <div className="flex items-center space-x-2 mb-1">
              <CheckCircle className="w-3 h-3 text-orange-400" />
              <span className="text-xs text-gray-500 uppercase" title="Positions settled this session">
                Settled
              </span>
            </div>
            <div className="text-lg font-mono font-bold text-orange-400">
              {pnl.session_settlements_count ?? 0}
            </div>
          </div>
        </div>
      )}

      {/* Positions, Orders, Order Group - Row 3 */}
      <div className="grid grid-cols-3 gap-4">
        {/* Positions */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ShoppingCart className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-gray-500 uppercase">Positions</span>
            </div>
            <span className="text-lg font-mono font-bold text-white">{positionCount}</span>
          </div>
        </div>

        {/* Orders */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-yellow-400" />
              <span className="text-xs text-gray-500 uppercase">Orders</span>
            </div>
            <span className="text-lg font-mono font-bold text-white">{orderCount}</span>
          </div>
        </div>

        {/* Order Group */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-indigo-400" />
              <span className="text-xs text-gray-500 uppercase">Order Group</span>
            </div>
            {orderGroup && orderGroup.id ? (
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-gray-400">
                  {orderGroup.id.substring(0, 8)}
                </span>
                <span className={`px-1.5 py-0.5 text-xs font-bold rounded ${
                  orderGroup.status === 'active'
                    ? 'bg-green-900/50 text-green-400'
                    : orderGroup.status === 'inactive'
                    ? 'bg-gray-900/50 text-gray-400'
                    : 'bg-yellow-900/50 text-yellow-400'
                }`}>
                  {(orderGroup.status || 'N/A').toUpperCase()}
                </span>
                <span className="text-sm font-mono font-bold text-white">
                  {orderGroup.order_count || 0}
                </span>
              </div>
            ) : (
              <span className="text-sm font-mono text-gray-500">--</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingSessionPanel;
