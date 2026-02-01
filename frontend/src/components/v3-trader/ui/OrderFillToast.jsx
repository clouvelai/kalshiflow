import React, { memo } from 'react';
import { CheckCircle, X } from 'lucide-react';

/**
 * OrderFillToast - Shows notification when an order fills
 *
 * Displays fill details: action, count, side, ticker, price, total
 * Positioned slightly above SettlementToast to avoid overlap.
 */
const OrderFillToast = ({ fill, onDismiss }) => {
  if (!fill) return null;

  const { action, count, side, ticker, price_cents, total_cents } = fill;

  // Format price and total
  const priceDisplay = `${price_cents}¢`;
  const totalDisplay = `$${(total_cents / 100).toFixed(2)}`;

  return (
    <div className="fixed bottom-20 right-4 p-4 rounded-lg shadow-lg z-50 border backdrop-blur-sm
      bg-green-900/90 border-green-700 animate-toast-enter"
    >
      <div className="flex items-center space-x-3">
        <CheckCircle className="w-5 h-5 text-green-400" />
        <div>
          <div className="text-sm font-medium text-white">
            Order Filled
          </div>
          <div className="text-base font-mono text-green-300">
            {action?.toUpperCase()} {count} {side?.toUpperCase()} {ticker}
          </div>
          <div className="text-sm font-mono text-green-400">
            @ {priceDisplay} = {totalDisplay}
          </div>
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

export default memo(OrderFillToast);
