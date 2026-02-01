import React, { memo } from 'react';
import { CheckCircle, X } from 'lucide-react';
import { formatSettlementCurrency } from '../../../utils/v3-trader';

/**
 * SettlementToast - Shows notification when a position closes
 */
const SettlementToast = ({ settlement, onDismiss }) => {
  if (!settlement) return null;

  const isProfit = (settlement.realized_pnl || 0) >= 0;

  return (
    <div className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg z-50 border backdrop-blur-sm
      ${isProfit ? 'bg-green-900/90 border-green-700' : 'bg-red-900/90 border-red-700'}
      animate-toast-enter`}
    >
      <div className="flex items-center space-x-3">
        <CheckCircle className={`w-5 h-5 ${isProfit ? 'text-green-400' : 'text-red-400'}`} />
        <div>
          <div className="text-sm font-medium text-white">
            Position Closed: {settlement.ticker}
          </div>
          <div className={`text-lg font-mono font-bold ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
            {formatSettlementCurrency(settlement.realized_pnl || 0)}
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

export default memo(SettlementToast);
