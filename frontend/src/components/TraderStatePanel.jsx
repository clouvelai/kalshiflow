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
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-700/70 transition-colors">
            <p className="text-xs text-slate-400 mb-1">Total Fills</p>
            <p className={`text-xl font-mono font-bold ${
              (executionStats?.total_fills || 0) > 0 ? 'text-emerald-400' : 'text-slate-300'
            }`}>
              {executionStats?.total_fills || 0}
            </p>
          </div>
          
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-700/70 transition-colors">
            <p className="text-xs text-slate-400 mb-1">Win Rate</p>
            <p className={`text-xl font-mono font-bold ${
              executionStats?.win_rate ? 
                (executionStats.win_rate >= 0.6 ? 'text-emerald-400' : 
                 executionStats.win_rate >= 0.4 ? 'text-amber-400' : 'text-red-400') : 
                'text-slate-300'
            }`}>
              {executionStats?.win_rate ? `${(executionStats.win_rate * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
        </div>

        {/* Fill Types */}
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-700/70 transition-colors">
          <h4 className="text-xs font-semibold text-slate-300 uppercase mb-2">Fill Breakdown</h4>
          
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded font-medium border border-blue-500/30">MAKER</span>
              <span className="text-sm font-mono font-semibold text-slate-200">
                {executionStats?.maker_fills || 0}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded font-medium border border-purple-500/30">TAKER</span>
              <span className="text-sm font-mono font-semibold text-slate-200">
                {executionStats?.taker_fills || 0}
              </span>
            </div>
          </div>

          {/* Fill ratio bar */}
          {(executionStats?.total_fills || 0) > 0 && (
            <div className="w-full h-2 bg-slate-600 rounded-full overflow-hidden">
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
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-700/70 transition-colors">
            <p className="text-xs text-slate-400 mb-1">Success Rate</p>
            <p className={`text-lg font-mono font-bold ${
              executionStats?.success_rate ? 
                (executionStats.success_rate >= 0.8 ? 'text-emerald-400' : 
                 executionStats.success_rate >= 0.5 ? 'text-amber-400' : 'text-red-400') : 
                'text-slate-300'
            }`}>
              {executionStats?.success_rate ? `${(executionStats.success_rate * 100).toFixed(1)}%` : '--'}
            </p>
          </div>
          
          <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3 hover:bg-slate-700/70 transition-colors">
            <p className="text-xs text-slate-400 mb-1">Total P&L</p>
            <p className={`text-lg font-mono font-bold ${
              executionStats?.total_pnl ? 
                (executionStats.total_pnl > 0 ? 'text-emerald-400' : 
                 executionStats.total_pnl < 0 ? 'text-red-400' : 'text-slate-400') : 
                'text-slate-300'
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
      {/* Portfolio Stats Grid - 4 Boxes */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {(() => {
          const totalValue = (displayState.portfolio_value || 0) + (displayState.cash_balance || 0);
          const totalStartValue = (displayState.session_start_portfolio_value || 0) + (displayState.session_start_cash || 0);
          const totalChange = totalValue - totalStartValue;
          const totalPnlPercent = totalStartValue > 0 ? (totalChange / totalStartValue) * 100 : 0;
          
          // Position Box
          const positionValue = displayState.portfolio_value || 0;
          const positionChange = displayState.portfolio_value_change !== undefined ? displayState.portfolio_value_change : (positionValue - (displayState.session_start_portfolio_value || 0));
          
          // Cash Box
          const cashValue = displayState.cash_balance || 0;
          const cashChange = displayState.cash_balance_change !== undefined ? displayState.cash_balance_change : (cashValue - (displayState.session_start_cash || 0));
          
          return (
            <>
              {/* Position Box */}
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-700/50 p-6 text-center hover:shadow-xl transition-all duration-300">
                <div className="text-3xl font-bold text-blue-400 mb-1">
                  {formatCurrency(positionValue)}
                </div>
                {positionChange !== undefined && (
                  <div className={`text-sm mb-2 ${
                    positionChange >= 0 ? 'text-emerald-400' : positionChange < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {positionChange >= 0 ? '+' : ''}{formatCurrency(positionChange)}
                  </div>
                )}
                <div className="text-sm font-medium text-slate-400 uppercase tracking-wide">Position</div>
              </div>
              
              {/* Cash Box */}
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-700/50 p-6 text-center hover:shadow-xl transition-all duration-300">
                <div className="text-3xl font-bold text-emerald-400 mb-1">
                  {formatCurrency(cashValue)}
                </div>
                {cashChange !== undefined && (
                  <div className={`text-sm mb-2 ${
                    cashChange >= 0 ? 'text-emerald-400' : cashChange < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {cashChange >= 0 ? '+' : ''}{formatCurrency(cashChange)}
                  </div>
                )}
                <div className="text-sm font-medium text-slate-400 uppercase tracking-wide">Cash</div>
              </div>
              
              {/* Total Box */}
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-700/50 p-6 text-center hover:shadow-xl transition-all duration-300">
                <div className="text-3xl font-bold text-purple-400 mb-1">
                  {formatCurrency(totalValue)}
                </div>
                {totalChange !== undefined && (
                  <div className={`text-sm mb-2 ${
                    totalChange >= 0 ? 'text-emerald-400' : totalChange < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {totalChange >= 0 ? '+' : ''}{formatCurrency(totalChange)}
                  </div>
                )}
                <div className="text-sm font-medium text-slate-400 uppercase tracking-wide">Total</div>
              </div>
              
              {/* Total P&L Box */}
              <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-700/50 p-6 text-center hover:shadow-xl transition-all duration-300">
                <div className={`text-3xl font-bold mb-1 ${
                  totalChange >= 0 ? 'text-emerald-400' : totalChange < 0 ? 'text-red-400' : 'text-slate-400'
                }`}>
                  {totalChange >= 0 ? '+' : ''}{formatCurrency(totalChange)}
                </div>
                {totalStartValue > 0 && (
                  <div className={`text-sm mb-2 ${
                    totalPnlPercent >= 0 ? 'text-emerald-400' : totalPnlPercent < 0 ? 'text-red-400' : 'text-slate-400'
                  }`}>
                    {totalPnlPercent >= 0 ? '+' : ''}{totalPnlPercent.toFixed(2)}%
                  </div>
                )}
                <div className="text-sm font-medium text-slate-400 uppercase tracking-wide">Total P&L</div>
              </div>
            </>
          );
        })()}
      </div>
      
      {/* Warning Indicator */}
      {displayState.cash_balance === 0 && displayState.portfolio_value === 0 && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          <div className="text-xs text-amber-400">
            ⚠️ Zero Balance - Add funds to continue trading
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-2 gap-3">

        {/* Action Breakdown - Redesigned with No Op and Trades sections */}
        {showActionBreakdown && displayState.actor_metrics?.action_counts && (
          <div className="bg-gradient-to-br from-slate-800 to-slate-800/80 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-all">
            {/* Header with Events Received count */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Actions</h3>
              <div className="flex items-center space-x-2">
                <span className="text-xs px-2.5 py-1 bg-slate-700/50 text-slate-300 rounded-full font-medium">
                  {formatNumber(displayState.actor_metrics.events_queued || 0)} events
                </span>
                <span className="text-xs px-2.5 py-1 bg-blue-500/20 text-blue-400 rounded-full font-medium border border-blue-500/30">
                  {formatNumber(displayState.actor_metrics.total_actions || 0)} actions
                </span>
              </div>
            </div>
            
            <div className="space-y-3">
              {/* No Op Section */}
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">No Op</h4>
                  <span className="text-xs px-2 py-0.5 bg-slate-700/50 text-slate-300 rounded-full">
                    {formatNumber(
                      (displayState.actor_metrics.action_counts.hold || 0) +
                      (displayState.actor_metrics.action_counts.failed || 0) +
                      (displayState.actor_metrics.action_counts.throttled || 0)
                    )}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-x-3 gap-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Hold</span>
                    <span className="text-sm font-semibold text-amber-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.hold || 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Failed</span>
                    <span className="text-sm font-semibold text-orange-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.failed || 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Throttled</span>
                    <span className="text-sm font-semibold text-yellow-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.throttled || 0)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Trades Section */}
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-medium text-slate-300 uppercase tracking-wider">Trades</h4>
                  <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full border border-blue-500/30">
                    {formatNumber(
                      (displayState.actor_metrics.action_counts.buy_yes || 0) +
                      (displayState.actor_metrics.action_counts.sell_yes || 0) +
                      (displayState.actor_metrics.action_counts.buy_no || 0) +
                      (displayState.actor_metrics.action_counts.sell_no || 0)
                    )}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Buy YES</span>
                    <span className="text-sm font-semibold text-emerald-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.buy_yes || 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Sell YES</span>
                    <span className="text-sm font-semibold text-red-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.sell_yes || 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Buy NO</span>
                    <span className="text-sm font-semibold text-purple-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.buy_no || 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">Sell NO</span>
                    <span className="text-sm font-semibold text-blue-400 font-mono">
                      {formatNumber(displayState.actor_metrics.action_counts.sell_no || 0)}
                    </span>
                  </div>
                </div>
                
                {/* Trade distribution bar - only show trades, exclude No Op */}
                {(() => {
                  const tradesTotal = 
                    (displayState.actor_metrics.action_counts.buy_yes || 0) +
                    (displayState.actor_metrics.action_counts.sell_yes || 0) +
                    (displayState.actor_metrics.action_counts.buy_no || 0) +
                    (displayState.actor_metrics.action_counts.sell_no || 0);
                  
                  return tradesTotal > 0 ? (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <div className="flex h-2 rounded-full overflow-hidden bg-slate-700">
                        {displayState.actor_metrics.action_counts.buy_yes > 0 && (
                          <div 
                            className="bg-emerald-400 transition-all duration-500"
                            style={{ 
                              width: `${(displayState.actor_metrics.action_counts.buy_yes / tradesTotal) * 100}%` 
                            }}
                          />
                        )}
                        {displayState.actor_metrics.action_counts.sell_yes > 0 && (
                          <div 
                            className="bg-red-400 transition-all duration-500"
                            style={{ 
                              width: `${(displayState.actor_metrics.action_counts.sell_yes / tradesTotal) * 100}%` 
                            }}
                          />
                        )}
                        {displayState.actor_metrics.action_counts.buy_no > 0 && (
                          <div 
                            className="bg-purple-400 transition-all duration-500"
                            style={{ 
                              width: `${(displayState.actor_metrics.action_counts.buy_no / tradesTotal) * 100}%` 
                            }}
                          />
                        )}
                        {displayState.actor_metrics.action_counts.sell_no > 0 && (
                          <div 
                            className="bg-blue-400 transition-all duration-500"
                            style={{ 
                              width: `${(displayState.actor_metrics.action_counts.sell_no / tradesTotal) * 100}%` 
                            }}
                          />
                        )}
                      </div>
                    </div>
                  ) : null;
                })()}
              </div>
            </div>
          </div>
        )}

        {/* Session Cashflow */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-800/80 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-all">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">Session Cashflow</h3>
          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="text-xs text-slate-500">Invested</span>
              <span className="text-xs font-mono font-semibold text-red-400">
                -{formatCurrency(displayState.session_cash_invested || 0)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-slate-500">Recouped</span>
              <span className="text-xs font-mono font-semibold text-emerald-400">
                +{formatCurrency(displayState.session_cash_recouped || 0)}
              </span>
            </div>
            {displayState.net_cashflow !== undefined && (
              <div className="flex justify-between items-center pt-1.5 border-t border-slate-700/30">
                <span className="text-xs font-medium text-slate-400">Net</span>
                <span className={`text-sm font-mono font-bold ${
                  displayState.net_cashflow >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {displayState.net_cashflow >= 0 ? '+' : ''}{formatCurrency(displayState.net_cashflow || 0)}
                </span>
              </div>
            )}
            {displayState.session_total_fees_paid !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">Fees</span>
                <span className="text-xs font-mono font-semibold text-amber-400">
                  -{formatCurrency(displayState.session_total_fees_paid || 0)}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Positions Section */}
      {showPositions && (
      <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
        <div className="bg-slate-700/30 px-4 py-3 border-b border-slate-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">📊 Open Positions</h3>
            <div className="flex items-center space-x-3">
              {displayState.positions && Object.keys(displayState.positions).length > 0 && (
                <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full font-medium border border-blue-500/30">
                  {Object.keys(displayState.positions).length} Active
                </span>
              )}
              <button
                onClick={() => setShowCents(!showCents)}
                className="text-xs px-2 py-1 bg-slate-700/50 hover:bg-slate-700/80 text-slate-300 rounded font-medium transition-colors"
              >
                {showCents ? '$' : '¢'}
              </button>
            </div>
          </div>
        </div>
        <div className="p-4">
          {displayState.positions && Object.keys(displayState.positions).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(displayState.positions).map(([key, position]) => {
                // Extract ticker - it might be in the position object or use the key
                const ticker = position.ticker || position.market_ticker || key;
                const percentChange = position.avg_price ? ((position.current_price - position.avg_price) / position.avg_price * 100) : 0;
                const isProfit = position.unrealized_pnl >= 0;
                
                return (
                  <div key={key} className="group bg-slate-700/30 hover:bg-slate-700/50 rounded-lg p-4 transition-all duration-200 border border-slate-700 hover:border-slate-600">
                    {/* Main Position Info */}
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <p className="font-semibold text-white text-sm">{ticker}</p>
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                            position.side === 'YES' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'
                          }`}>
                            {position.side}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400">
                          <span className="font-mono">{position.contracts}</span> contracts
                        </p>
                      </div>
                      <div className="text-right">
                        <p className={`font-bold text-sm ${
                          (position.realized_pnl_dollars ? parseFloat(position.realized_pnl_dollars) : (position.realized_pnl || 0) / 100) >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}>
                          {(position.realized_pnl_dollars ? parseFloat(position.realized_pnl_dollars) : (position.realized_pnl || 0) / 100) >= 0 ? '+' : ''}{formatValueFromKalshi(position, 'realized_pnl')}
                        </p>
                        <p className="text-xs text-slate-400">Realized P&L</p>
                      </div>
                    </div>
                    
                    {/* Kalshi Position Details */}
                    <div className="grid grid-cols-2 gap-3 text-xs mt-3 pt-3 border-t border-slate-700/50">
                      <div>
                        <p className="text-slate-500 mb-1">Cost Basis</p>
                        <p className="text-slate-200 font-mono font-medium">{formatCurrency(position.cost_basis || 0)}</p>
                      </div>
                      <div>
                        <p className="text-slate-500 mb-1">Market Exposure</p>
                        <p className="text-slate-200 font-mono font-medium">{formatValueFromKalshi(position, 'market_exposure')}</p>
                      </div>
                      <div>
                        <p className="text-slate-500 mb-1">Total Traded</p>
                        <p className="text-slate-200 font-mono font-medium">{formatValueFromKalshi(position, 'total_traded')}</p>
                      </div>
                      <div>
                        <p className="text-slate-500 mb-1">Fees Paid</p>
                        <p className="text-slate-200 font-mono font-medium">{formatValueFromKalshi(position, 'fees_paid')}</p>
                      </div>
                      <div>
                        <p className="text-slate-500 mb-1">Realized P&L</p>
                        <p className="text-slate-200 font-mono font-medium">{formatValueFromKalshi(position, 'realized_pnl')}</p>
                      </div>
                    </div>
                    {position.last_updated_ts && (
                      <div className="mt-3 text-xs pt-3 border-t border-slate-700/50">
                        <p className="text-slate-500">Last Updated</p>
                        <p className="text-slate-400 font-mono">
                          {new Date(position.last_updated_ts).toLocaleString()}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-xs text-slate-400">No open positions</p>
              <p className="text-xs text-slate-500 mt-1">Waiting for trading signals...</p>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Enhanced Open Orders Section */}
      {showOrders && (
      <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
        <div className="bg-slate-700/30 px-4 py-3 border-b border-slate-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100">📋 Open Orders</h3>
            {displayState.open_orders && displayState.open_orders.length > 0 && (
              <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full font-medium border border-amber-500/30">
                {displayState.open_orders.length} Pending
              </span>
            )}
          </div>
        </div>
        <div className="p-4">
          {displayState.open_orders && displayState.open_orders.length > 0 ? (
            <div className="space-y-3">
              {displayState.open_orders.slice(0, 5).map((order, idx) => {
                const orderTime = order.created_at ? new Date(order.created_at) : (order.placed_at ? new Date(order.placed_at * 1000) : null);
                const timeElapsed = orderTime ? Date.now() - orderTime.getTime() : 0;
                const minutesElapsed = Math.floor(timeElapsed / 60000);
                const secondsElapsed = Math.floor((timeElapsed % 60000) / 1000);
                
                return (
                  <div key={idx} className="group bg-slate-700/30 hover:bg-slate-700/50 rounded-lg p-4 transition-all duration-200 border border-slate-700 hover:border-slate-600">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          {/* Trade ID Badge */}
                          {order.trade_sequence_id && (
                            <span className="px-2 py-0.5 bg-slate-700/70 text-slate-300 rounded text-xs font-bold border border-slate-600">
                              #{order.trade_sequence_id}
                            </span>
                          )}
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                            order.side === 'BUY' || order.side === 'YES' || order.contract_side === 'YES' ? 
                              'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 
                              'bg-red-500/20 text-red-400 border border-red-500/30'
                          }`}>
                            {order.side || order.contract_side || 'BUY'}
                          </span>
                          <span className="text-xs px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded font-medium border border-amber-500/30">
                            PENDING
                          </span>
                          {order.ticker && (
                            <p className="text-xs text-slate-400 font-medium">{order.ticker}</p>
                          )}
                        </div>
                        <p className="text-xs text-slate-300">
                          <span className="font-mono font-medium">{order.quantity}</span> contracts @ 
                          <span className="font-mono text-white ml-1 font-semibold">{formatCurrency((order.limit_price || order.price)/100)}</span>
                        </p>
                      </div>
                      <div className="text-right">
                        {(order.current_price !== undefined || order.placed_at) && (
                          <p className="text-xs text-slate-400 mb-1">
                            Market: <span className="font-mono">{formatCurrency(order.current_price/100)}</span>
                          </p>
                        )}
                        {(orderTime || order.placed_at) && (
                          <p className="text-xs text-slate-500">
                            {minutesElapsed > 0 ? `${minutesElapsed}m ` : ''}{secondsElapsed}s ago
                          </p>
                        )}
                      </div>
                    </div>
                    {order.current_price !== undefined && (
                      <div className="mt-3 pt-3 border-t border-slate-700/50">
                        <div className="flex justify-between items-center">
                          <p className="text-xs text-slate-500">Distance from market:</p>
                          <p className={`text-xs font-mono font-medium ${
                            Math.abs(order.price - order.current_price) < 5 ? 'text-emerald-400' : 
                            Math.abs(order.price - order.current_price) < 10 ? 'text-amber-400' : 
                            'text-slate-400'
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
                <p className="text-xs text-slate-500 text-center mt-2">
                  +{displayState.open_orders.length - 5} more orders
                </p>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-xs text-slate-400">No open orders</p>
              <p className="text-xs text-slate-500 mt-1">Orders will appear here when placed</p>
            </div>
          )}
        </div>
      </div>
      )}

      {/* Execution Statistics - Only show if showExecutionStats is true */}
      {showExecutionStats && (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Execution Statistics</h3>
          <div className="space-y-4">
            {/* Primary Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3">
                <p className="text-xs text-slate-400 mb-1">Total Fills</p>
                <p className={`text-xl font-mono font-bold ${
                  (executionStats?.total_fills || 0) > 0 ? 'text-emerald-400' : 'text-slate-300'
                }`}>
                  {executionStats?.total_fills || 0}
                </p>
              </div>
              
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3">
                <p className="text-xs text-slate-400 mb-1">Win Rate</p>
                <p className={`text-xl font-mono font-bold ${
                  executionStats?.win_rate ? 
                    (executionStats.win_rate >= 0.6 ? 'text-emerald-400' : 
                     executionStats.win_rate >= 0.4 ? 'text-amber-400' : 'text-red-400') : 
                    'text-slate-300'
                }`}>
                  {executionStats?.win_rate ? `${(executionStats.win_rate * 100).toFixed(1)}%` : '--'}
                </p>
              </div>
            </div>

            {/* Fill Types */}
            <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-4 space-y-3">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Fill Breakdown</h4>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded font-medium border border-blue-500/30">MAKER</span>
                  <span className="text-sm font-mono font-semibold text-slate-200">
                    {executionStats?.maker_fills || 0}
                  </span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded font-medium border border-purple-500/30">TAKER</span>
                  <span className="text-sm font-mono font-semibold text-slate-200">
                    {executionStats?.taker_fills || 0}
                  </span>
                </div>
              </div>

              {/* Fill ratio bar */}
              {(executionStats?.total_fills || 0) > 0 && (
                <div className="w-full h-2 bg-slate-600 rounded-full overflow-hidden">
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
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3">
                <p className="text-xs text-slate-400 mb-1">Success Rate</p>
                <p className={`text-lg font-mono font-semibold ${
                  executionStats?.success_rate ? 
                    (executionStats.success_rate >= 0.8 ? 'text-emerald-400' : 
                     executionStats.success_rate >= 0.5 ? 'text-amber-400' : 'text-red-400') : 
                    'text-slate-300'
                }`}>
                  {executionStats?.success_rate ? `${(executionStats.success_rate * 100).toFixed(1)}%` : '--'}
                </p>
              </div>
              
              <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-3">
                <p className="text-xs text-slate-400 mb-1">Total P&L</p>
                <p className={`text-lg font-mono font-semibold ${
                  executionStats?.total_pnl ? 
                    (executionStats.total_pnl > 0 ? 'text-emerald-400' : 
                     executionStats.total_pnl < 0 ? 'text-red-400' : 'text-slate-400') : 
                    'text-slate-300'
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


    </div>
  );
};

export default TraderStatePanel;