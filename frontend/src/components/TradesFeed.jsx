import React, { useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

const TradesFeed = ({ fills }) => {
  const [expandedRows, setExpandedRows] = useState(new Set());

  // Helper function to toggle row expansion
  const toggleRowExpansion = (index) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  // Helper function to format timestamp - shows actual timestamp of action
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '--:--:--';
    // Ensure we're using the actual timestamp from the action
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      // If timestamp is relative seconds, convert it
      if (typeof timestamp === 'number' && timestamp < 86400) {
        // Likely seconds since session start - show as relative time
        const hours = Math.floor(timestamp / 3600);
        const minutes = Math.floor((timestamp % 3600) / 60);
        const seconds = Math.floor(timestamp % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
      }
      return '--:--:--';
    }
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Helper function to get action color and background
  const getActionStyle = (action, success, reason) => {
    const actionStr = typeof action === 'object' ? action.action_name : action;
    const isSuccess = success !== false;
    
    // Check if this is a closing action
    const isClosing = reason && reason.startsWith('close_position:');
    const closingReason = isClosing ? reason.replace('close_position:', '') : null;
    
    if (!isSuccess) {
      return {
        color: 'text-red-400',
        bg: 'bg-red-900/20',
        borderColor: 'border-red-600/30'
      };
    }
    
    // Special styling for closing actions
    if (isClosing) {
      return {
        color: 'text-amber-400',
        bg: 'bg-amber-900/30',
        borderColor: 'border-amber-600/40',
        icon: '🔒',
        isClosing: true,
        closingReason: closingReason
      };
    }
    
    const actionUpper = actionStr?.toUpperCase() || '';
    
    if (actionUpper.includes('HOLD')) {
      return {
        color: 'text-amber-400',
        bg: 'bg-amber-900/20',
        borderColor: 'border-amber-600/30',
        icon: '⏸'
      };
    } else if (actionUpper.includes('BUY_YES') || actionUpper === 'BUY' || actionUpper === 'BID') {
      return {
        color: 'text-green-400',
        bg: 'bg-green-900/20',
        borderColor: 'border-green-600/30',
        icon: '↑'
      };
    } else if (actionUpper.includes('SELL_YES') || actionUpper === 'SELL' || actionUpper === 'ASK') {
      return {
        color: 'text-red-400',
        bg: 'bg-red-900/20',
        borderColor: 'border-red-600/30',
        icon: '↓'
      };
    } else if (actionUpper.includes('BUY_NO')) {
      return {
        color: 'text-purple-400',
        bg: 'bg-purple-900/20',
        borderColor: 'border-purple-600/30',
        icon: '↓'
      };
    } else if (actionUpper.includes('SELL_NO')) {
      return {
        color: 'text-blue-400',
        bg: 'bg-blue-900/20',
        borderColor: 'border-blue-600/30',
        icon: '↑'
      };
    } else if (actionUpper.includes('CANCEL')) {
      return {
        color: 'text-gray-500',
        bg: 'bg-gray-800/20',
        borderColor: 'border-gray-600/30',
        icon: '✕'
      };
    }
    
    return {
      color: 'text-gray-400',
      bg: 'bg-gray-800/20',
      borderColor: 'border-gray-600/30'
    };
  };
  
  // Helper function to format closing reason
  const formatClosingReason = (reason) => {
    if (!reason) return '';
    const reasonMap = {
      'take_profit': 'Take Profit',
      'stop_loss': 'Stop Loss',
      'cash_recovery': 'Cash Recovery',
      'market_closing': 'Market Closing',
      'max_hold_time': 'Max Hold Time'
    };
    return reasonMap[reason] || reason.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Helper function to format action text
  const formatAction = (action) => {
    if (!action) return 'UNKNOWN';
    
    let actionStr = typeof action === 'object' ? 
      (action.action_name || action.name || JSON.stringify(action)) : 
      action.toString();
    
    actionStr = actionStr.toUpperCase();
    
    // Simplify the display names
    if (actionStr.includes('BUY_YES_LIMIT')) return 'BUY YES';
    if (actionStr.includes('SELL_YES_LIMIT')) return 'SELL YES';
    if (actionStr.includes('BUY_NO_LIMIT')) return 'BUY NO';
    if (actionStr.includes('SELL_NO_LIMIT')) return 'SELL NO';
    if (actionStr.includes('HOLD')) return 'HOLD';
    if (actionStr.includes('CANCEL')) return 'CANCEL';
    
    return actionStr.replace(/_/g, ' ');
  };

  // Helper function to format price
  const formatPrice = (price) => {
    if (price === null || price === undefined) return '--';
    if (price <= 1) {
      return `${Math.round(price * 100)}¢`;
    }
    return `$${price.toFixed(2)}`;
  };

  // Helper function to format observation data
  const formatObservationData = (observation) => {
    if (!observation) return null;
    
    const features = observation.features || {};
    const rawArray = observation.raw_array || [];
    
    return (
      <div className="mt-2 p-3 bg-gray-900/50 rounded-lg border border-gray-700/50 space-y-3">
        {/* Orderbook Features */}
        {features.orderbook && (
          <div>
            <h4 className="text-xs font-semibold text-gray-400 mb-2">Orderbook State</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">YES Bid/Ask:</span>
                <span className="text-green-400 font-mono">
                  {formatPrice(features.orderbook.yes_bid)} / {formatPrice(features.orderbook.yes_ask)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">NO Bid/Ask:</span>
                <span className="text-red-400 font-mono">
                  {formatPrice(features.orderbook.no_bid)} / {formatPrice(features.orderbook.no_ask)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">YES Size:</span>
                <span className="text-gray-300 font-mono">
                  {features.orderbook.yes_bid_size?.toFixed(0)} / {features.orderbook.yes_ask_size?.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">NO Size:</span>
                <span className="text-gray-300 font-mono">
                  {features.orderbook.no_bid_size?.toFixed(0)} / {features.orderbook.no_ask_size?.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Spread:</span>
                <span className="text-amber-400 font-mono">{formatPrice(features.orderbook.spread)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Mid Price:</span>
                <span className="text-blue-400 font-mono">{formatPrice(features.orderbook.mid_price)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Market Dynamics */}
        {features.market_dynamics && (
          <div>
            <h4 className="text-xs font-semibold text-gray-400 mb-2">Market Dynamics</h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="flex flex-col">
                <span className="text-gray-500">Imbalance</span>
                <span className={`font-mono ${features.market_dynamics.imbalance > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {(features.market_dynamics.imbalance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-500">Volume Ratio</span>
                <span className="text-purple-400 font-mono">
                  {features.market_dynamics.volume_ratio?.toFixed(2)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-500">Momentum</span>
                <span className={`font-mono ${features.market_dynamics.price_momentum > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {features.market_dynamics.price_momentum?.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Portfolio State */}
        {features.portfolio && (
          <div>
            <h4 className="text-xs font-semibold text-gray-400 mb-2">Portfolio State</h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="flex flex-col">
                <span className="text-gray-500">Position</span>
                <span className={`font-mono ${features.portfolio.position > 0 ? 'text-green-400' : features.portfolio.position < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                  {features.portfolio.position?.toFixed(0)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-500">Cash</span>
                <span className="text-blue-400 font-mono">
                  ${features.portfolio.cash_available?.toFixed(0)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-500">Unrealized P&L</span>
                <span className={`font-mono ${features.portfolio.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${features.portfolio.unrealized_pnl?.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Raw Observation Array */}
        {rawArray && rawArray.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-gray-400 mb-2">
              Raw Observation Vector ({rawArray.length} features)
            </h4>
            <div className="bg-gray-800/50 rounded p-2 max-h-32 overflow-y-auto">
              <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap">
                [{rawArray.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ')}]
              </pre>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Default empty state
  if (!fills || fills.length === 0) {
    return (
      <div className="space-y-2">
        <div className="text-gray-500 text-sm text-center py-8">
          <div className="mb-2">🤖</div>
          <div>Waiting for trading decisions...</div>
          <div className="text-xs mt-1 text-gray-600">Model will analyze markets and execute trades</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-[600px] overflow-y-auto custom-scrollbar">
      <style jsx>{`
        @keyframes slideInFromLeft {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        .animate-slide-in-left {
          animation: slideInFromLeft 0.3s ease-out;
        }
      `}</style>
      {fills.map((fill, index) => {
        const isExpanded = expandedRows.has(index);
        const action = fill.action?.action_name || fill.action;
        const positionSize = fill.action?.position_size;
        const isSuccess = fill.execution_result?.executed !== false && fill.success !== false;
        const reason = fill.action?.reason || fill.execution_result?.reason;
        const actionStyle = getActionStyle(action, isSuccess, reason);
        const hasObservation = fill.observation && (fill.observation.raw_array?.length > 0 || fill.observation.features);
        const tradeId = fill.trade_sequence_id || fill.execution_result?.trade_sequence_id;
        
        return (
          <div 
            key={`fill-${index}-${fill.timestamp}`}
            className={`group font-mono text-xs rounded-lg ${actionStyle.bg} border ${actionStyle.borderColor} 
                      transition-all duration-200 hover:shadow-lg hover:scale-[1.02] hover:border-opacity-80
                      ${index === 0 ? 'animate-slide-in-left' : ''}`}
          >
            {/* Main Row */}
            <div 
              className={`py-3 px-3 flex items-center justify-between rounded-lg
                        ${hasObservation ? 'cursor-pointer hover:bg-gray-700/30' : ''}
                        transition-colors duration-150`}
              onClick={() => hasObservation && toggleRowExpansion(index)}
            >
              <div className="flex items-center space-x-3 flex-1">
                {/* Expand/Collapse Icon */}
                {hasObservation && (
                  <div className="text-gray-500">
                    {isExpanded ? 
                      <ChevronDownIcon className="h-3 w-3" /> : 
                      <ChevronRightIcon className="h-3 w-3" />
                    }
                  </div>
                )}

                {/* Trade ID Badge */}
                {tradeId && (
                  <span className="px-2 py-0.5 bg-gray-700/70 text-gray-300 rounded text-xs font-bold border border-gray-600">
                    #{tradeId}
                  </span>
                )}

                {/* Timestamp */}
                <span className="text-gray-500 w-20">
                  {formatTimestamp(fill.timestamp)}
                </span>

                {/* Action with Icon */}
                <div className="flex items-center space-x-2">
                  {actionStyle.icon && (
                    <span className={`${actionStyle.color} text-lg group-hover:scale-110 transition-transform`}>
                      {actionStyle.icon}
                    </span>
                  )}
                  <span className={`${actionStyle.color} font-semibold w-24`}>
                    {formatAction(action)}
                  </span>
                  {/* Closing Badge */}
                  {actionStyle.isClosing && actionStyle.closingReason && (
                    <span className="px-2 py-0.5 bg-amber-900/40 text-amber-300 rounded text-xs font-medium border border-amber-600/30">
                      CLOSING: {formatClosingReason(actionStyle.closingReason)}
                    </span>
                  )}
                </div>

                {/* Position Size (for non-HOLD actions) */}
                {positionSize && positionSize > 0 && (
                  <span className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded-md text-xs font-medium
                                 group-hover:bg-gray-700/70 transition-colors">
                    {positionSize} contracts
                  </span>
                )}

                {/* Market Ticker - Show full ticker */}
                {fill.market_ticker && (
                  <span className="text-gray-400 text-xs font-medium px-2 py-1 bg-gray-700/30 rounded-md
                                 group-hover:bg-gray-700/50 transition-colors">
                    {fill.market_ticker}
                  </span>
                )}

                {/* Price (if applicable) */}
                {fill.action?.limit_price !== undefined && fill.action?.limit_price !== null && (
                  <span className="text-gray-300">
                    @{formatPrice(fill.action.limit_price)}
                  </span>
                )}
              </div>

              <div className="flex items-center space-x-2">
                {/* Execution Status */}
                {fill.execution_result && (
                  <span className={`text-xs px-2 py-1 rounded-md font-medium ${
                    fill.execution_result.executed ? 'bg-green-900/30 text-green-400' : 
                    fill.execution_result.status === 'hold' ? 'bg-amber-900/30 text-amber-400' :
                    'bg-red-900/30 text-red-400'
                  }`}>
                    {fill.execution_result.status || (fill.execution_result.executed ? 'executed' : 'failed')}
                  </span>
                )}

                {/* Success Indicator */}
                <span className={`${isSuccess ? 'text-green-400' : 'text-red-400'} text-sm`}>
                  {isSuccess ? '✓' : '✗'}
                </span>
              </div>
            </div>

            {/* Expanded Observation Details */}
            {isExpanded && hasObservation && (
              <div className="border-t border-gray-700/50">
                {formatObservationData(fill.observation)}
              </div>
            )}

            {/* Error Message (if any) */}
            {fill.execution_result?.error && (
              <div className="px-3 pb-2">
                <div className="text-xs text-red-400 bg-red-900/20 rounded p-1.5 flex items-start space-x-1">
                  <span className="text-red-500">⚠️</span>
                  <span className="flex-1">
                    {fill.execution_result.error.includes('average_cost_cents') 
                      ? 'Position tracking error - backend issue'
                      : fill.execution_result.error.length > 100 
                        ? fill.execution_result.error.substring(0, 100) + '...'
                        : fill.execution_result.error}
                  </span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default TradesFeed;