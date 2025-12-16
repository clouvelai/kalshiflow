import React from 'react';

const TradesFeed = ({ fills }) => {
  // Helper function to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '--:--:--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Helper function to get action color
  const getActionColor = (action, success) => {
    if (!success) return 'text-red-500';
    
    switch (action?.toLowerCase()) {
      case 'buy':
      case 'buy_yes':
      case 'bid':
        return 'text-green-400';
      case 'sell':
      case 'sell_yes':
      case 'buy_no':
      case 'ask':
        return 'text-red-400';
      case 'cancel':
      case 'cancelled':
        return 'text-gray-500';
      case 'hold':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  // Helper function to format action text
  const formatAction = (action) => {
    if (!action) return 'UNKNOWN';
    
    // Handle if action is an object (e.g., {action_name: "BUY_YES_LIMIT"})
    let actionStr;
    if (typeof action === 'object' && action !== null) {
      // Try to get action_name from object
      actionStr = (action.action_name || action.name || JSON.stringify(action)).toString().toUpperCase();
    } else {
      // Convert action to uppercase and make it more readable
      actionStr = action.toString().toUpperCase();
    }
    
    // Handle various action formats
    if (actionStr.includes('BUY_YES')) return 'BUY YES';
    if (actionStr.includes('SELL_YES')) return 'SELL YES';
    if (actionStr.includes('BUY_NO')) return 'BUY NO';
    if (actionStr.includes('SELL_NO')) return 'SELL NO';
    if (actionStr.includes('CANCEL')) return 'CANCEL';
    if (actionStr.includes('HOLD')) return 'HOLD';
    
    return actionStr;
  };

  // Helper function to format price
  const formatPrice = (price) => {
    if (price === null || price === undefined) return '--';
    // Convert to cents if needed (assuming price is between 0-1)
    if (price <= 1) {
      return `${Math.round(price * 100)}¢`;
    }
    return `$${price.toFixed(2)}`;
  };

  // Helper function to get success indicator
  const getSuccessIndicator = (fill) => {
    if (fill.success === false) return '✗';
    if (fill.filled || fill.success) return '✓';
    if (fill.status === 'pending') return '⏳';
    if (fill.status === 'cancelled') return '⊘';
    return '•';
  };

  // Helper function to get fill type badge
  const getFillTypeBadge = (fill) => {
    if (fill.is_maker) {
      return <span className="text-xs px-1 py-0.5 bg-blue-500/20 text-blue-400 rounded">MAKER</span>;
    }
    if (fill.is_taker) {
      return <span className="text-xs px-1 py-0.5 bg-purple-500/20 text-purple-400 rounded">TAKER</span>;
    }
    return null;
  };

  // Default empty state
  if (!fills || fills.length === 0) {
    return (
      <div className="space-y-2">
        <div className="text-gray-500 text-sm text-center py-4">
          No trades executed yet
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1 max-h-96 overflow-y-auto">
      {fills.map((fill, index) => {
        // Handle nested action structure if present
        const action = fill.action?.action_name || fill.action;
        const isSuccess = fill.success !== false && (fill.filled || fill.success);
        const actionColor = getActionColor(action, isSuccess);
        
        return (
          <div 
            key={`fill-${index}-${fill.timestamp}`}
            className={`font-mono text-xs py-1.5 px-2 rounded ${
              isSuccess ? 'bg-gray-700/30' : 'bg-red-900/20'
            } hover:bg-gray-700/50 transition-colors flex items-center justify-between`}
          >
            <div className="flex items-center space-x-2 flex-1">
              {/* Timestamp */}
              <span className="text-gray-500 w-20">
                {formatTimestamp(fill.timestamp)}
              </span>
              
              {/* Action */}
              <span className={`${actionColor} font-semibold w-20`}>
                {formatAction(action)}
              </span>
              
              {/* Market/Ticker (if available) */}
              {(fill.market || fill.ticker || fill.market_ticker) && (
                <span className="text-gray-400 text-xs truncate max-w-[80px]">
                  {fill.market || fill.ticker || fill.market_ticker}
                </span>
              )}
              
              {/* Price */}
              {(fill.price !== undefined || fill.fill_price !== undefined) && (
                <span className="text-gray-300">
                  @{formatPrice(fill.fill_price || fill.price)}
                </span>
              )}
              
              {/* Quantity (if available) */}
              {fill.quantity !== undefined && fill.quantity > 0 && (
                <span className="text-gray-500 text-xs">
                  x{fill.quantity}
                </span>
              )}
              
              {/* Fill type badge */}
              {getFillTypeBadge(fill)}
            </div>
            
            <div className="flex items-center space-x-2">
              {/* Model confidence or reason (if available) */}
              {fill.confidence !== undefined && (
                <span className="text-xs text-gray-500">
                  {(fill.confidence * 100).toFixed(0)}%
                </span>
              )}
              
              {/* Success indicator */}
              <span className={`${isSuccess ? 'text-green-400' : 'text-red-400'} text-sm`}>
                {getSuccessIndicator(fill)}
              </span>
            </div>
          </div>
        );
      })}
      
      {/* Show reasoning for latest trade if available */}
      {fills[0]?.reasoning && (
        <div className="mt-2 p-2 bg-gray-700/30 rounded border border-gray-600/50">
          <p className="text-xs text-gray-400">
            <span className="font-semibold">Decision:</span> {fills[0].reasoning}
          </p>
        </div>
      )}
    </div>
  );
};

export default TradesFeed;