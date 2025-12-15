import React, { useState } from 'react';

const ActionItem = ({ action, onClick, isSelected }) => {
  const actionColors = {
    'BUY_YES_LIMIT': 'text-green-400',
    'SELL_YES_LIMIT': 'text-red-400',
    'BUY_NO_LIMIT': 'text-blue-400',
    'SELL_NO_LIMIT': 'text-orange-400',
    'HOLD': 'text-gray-400'
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div
      className={`p-3 rounded cursor-pointer transition-colors ${
        isSelected ? 'bg-gray-700' : 'bg-gray-900 hover:bg-gray-800'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className={`font-semibold ${
            actionColors[action.action?.action_name] || 'text-white'
          }`}>
            {action.action?.action_name || 'UNKNOWN'}
          </span>
          <span className="text-sm text-gray-500">
            {action.market_ticker}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {formatTime(action.timestamp)}
        </span>
      </div>
      {action.execution_result?.executed && action.execution_result?.order_id && (
        <div className="text-xs text-gray-400 mt-1">
          Order: {action.execution_result.order_id.slice(0, 8)}...
        </div>
      )}
      {action.execution_result?.error && (
        <div className="text-xs text-red-400 mt-1">
          Error: {action.execution_result.error}
        </div>
      )}
    </div>
  );
};

const ActionDetail = ({ action }) => {
  if (!action) return null;

  const formatValue = (value, decimals = 2) => {
    if (value === null || value === undefined) return 'N/A';
    return typeof value === 'number' ? value.toFixed(decimals) : value;
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <h3 className="text-lg font-semibold text-white">Action Details</h3>
      
      {/* Action Information */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-gray-400 text-sm">Market</span>
          <span className="text-white font-mono text-sm">{action.market_ticker}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400 text-sm">Action</span>
          <span className="text-white font-semibold">{action.action?.action_name}</span>
        </div>
        {action.action?.quantity && (
          <div className="flex justify-between">
            <span className="text-gray-400 text-sm">Quantity</span>
            <span className="text-white">{action.action.quantity}</span>
          </div>
        )}
        {action.action?.limit_price && (
          <div className="flex justify-between">
            <span className="text-gray-400 text-sm">Limit Price</span>
            <span className="text-white">{action.action.limit_price}¢</span>
          </div>
        )}
      </div>

      {/* Observation Data */}
      {action.observation && Object.keys(action.observation).length > 0 && (
        <>
          <div className="border-t border-gray-700 pt-3">
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Market State</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Yes Bid/Ask</span>
                <div className="text-white">
                  {formatValue(action.observation.yes_bid)}¢ / {formatValue(action.observation.yes_ask)}¢
                </div>
              </div>
              <div>
                <span className="text-gray-500">No Bid/Ask</span>
                <div className="text-white">
                  {formatValue(action.observation.no_bid)}¢ / {formatValue(action.observation.no_ask)}¢
                </div>
              </div>
              <div>
                <span className="text-gray-500">Spread</span>
                <div className="text-white">{formatValue(action.observation.spread)}¢</div>
              </div>
              <div>
                <span className="text-gray-500">Mid Price</span>
                <div className="text-white">{formatValue(action.observation.mid_price)}¢</div>
              </div>
              {action.observation.position !== undefined && (
                <div>
                  <span className="text-gray-500">Position</span>
                  <div className="text-white">{action.observation.position}</div>
                </div>
              )}
              {action.observation.cash_available !== undefined && (
                <div>
                  <span className="text-gray-500">Cash Available</span>
                  <div className="text-white">${formatValue(action.observation.cash_available)}</div>
                </div>
              )}
            </div>
          </div>

          {/* Orderbook Sizes */}
          <div className="border-t border-gray-700 pt-3">
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Orderbook Depth</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Yes Bid Size</span>
                <div className="text-white">{formatValue(action.observation.yes_bid_size, 0)}</div>
              </div>
              <div>
                <span className="text-gray-500">Yes Ask Size</span>
                <div className="text-white">{formatValue(action.observation.yes_ask_size, 0)}</div>
              </div>
              <div>
                <span className="text-gray-500">No Bid Size</span>
                <div className="text-white">{formatValue(action.observation.no_bid_size, 0)}</div>
              </div>
              <div>
                <span className="text-gray-500">No Ask Size</span>
                <div className="text-white">{formatValue(action.observation.no_ask_size, 0)}</div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Execution Result */}
      <div className="border-t border-gray-700 pt-3">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">Execution</h4>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Status</span>
            <span className={`font-semibold ${
              action.execution_result?.executed ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {action.execution_result?.status || 'Unknown'}
            </span>
          </div>
          {action.execution_result?.order_id && (
            <div className="flex justify-between">
              <span className="text-gray-500">Order ID</span>
              <span className="text-white font-mono text-xs">
                {action.execution_result.order_id.slice(0, 12)}...
              </span>
            </div>
          )}
          {action.execution_result?.error && (
            <div className="mt-2">
              <span className="text-red-400">Error: {action.execution_result.error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ActionFeed = ({ actions }) => {
  const [selectedAction, setSelectedAction] = useState(null);

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Trading Actions</h2>
        <div className="h-[400px] overflow-y-auto space-y-2">
          {actions && actions.length > 0 ? (
            actions.map((action, idx) => (
              <ActionItem
                key={`${action.timestamp}-${idx}`}
                action={action}
                onClick={() => setSelectedAction(action)}
                isSelected={selectedAction === action}
              />
            ))
          ) : (
            <p className="text-gray-400 text-sm">No trading actions yet...</p>
          )}
        </div>
      </div>

      {selectedAction && <ActionDetail action={selectedAction} />}
    </div>
  );
};

export default ActionFeed;