import React, { memo } from 'react';
import { Clock, X } from 'lucide-react';

/**
 * OrderCancelledToast - Shows notification when orders are cancelled due to TTL expiry
 *
 * Displays the count and first few tickers of cancelled orders.
 * Amber-themed to indicate a warning/informational notification.
 * Positioned above SettlementToast and OrderFillToast.
 */
const OrderCancelledToast = ({ cancellation, onDismiss }) => {
  if (!cancellation) return null;

  const { count, tickers = [] } = cancellation;

  // Format tickers for display (show first 3, then "...")
  const displayTickers = tickers.slice(0, 3).join(', ');
  const hasMore = tickers.length > 3;

  return (
    <div className="fixed bottom-32 right-4 p-4 rounded-lg shadow-lg z-50 border backdrop-blur-sm
      bg-amber-900/90 border-amber-500 animate-pulse"
    >
      <div className="flex items-center space-x-3">
        <Clock className="w-5 h-5 text-amber-400" />
        <div>
          <div className="text-sm font-medium text-white">
            TTL Expired
          </div>
          <div className="text-base font-mono text-amber-300">
            {count} order{count !== 1 ? 's' : ''} cancelled
          </div>
          {tickers.length > 0 && (
            <div className="text-xs font-mono text-amber-400/80 mt-0.5">
              {displayTickers}{hasMore && ` +${tickers.length - 3} more`}
            </div>
          )}
        </div>
        <button
          onClick={onDismiss}
          className="text-gray-400 hover:text-white ml-2"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default memo(OrderCancelledToast);
