import React, { useState, memo } from 'react';
import { ChevronRight, ChevronDown, CheckCircle } from 'lucide-react';
import { formatSettlementCurrency, formatTime, formatCents, getPnLColor } from '../../../utils/v3-trader';

/**
 * SettlementRow - Memoized row component for settlements table
 */
const SettlementRow = memo(({ settlement, index }) => {
  const qty = Math.abs(settlement.position || 0);
  const totalCost = settlement.total_cost || settlement.total_traded || 0;
  const revenue = settlement.revenue || 0;
  const fees = settlement.fees || 0;
  const netPnl = settlement.net_pnl ?? (revenue - totalCost - fees);
  const isProfit = netPnl >= 0;

  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30">
      <td className="py-2 px-2 font-mono text-gray-300 text-xs">{settlement.ticker}</td>
      <td className="py-2 px-2 text-center">
        <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
          settlement.side === 'yes'
            ? 'bg-green-900/30 text-green-400'
            : 'bg-red-900/30 text-red-400'
        }`}>
          {settlement.side?.toUpperCase()}
        </span>
      </td>
      <td className="py-2 px-2 text-right font-mono text-gray-400">{qty}</td>
      <td className="py-2 px-2 text-right font-mono text-gray-400">{formatCents(totalCost)}</td>
      <td className="py-2 px-2 text-right font-mono text-gray-400">{formatCents(revenue)}</td>
      <td className="py-2 px-2 text-right font-mono text-gray-500 text-xs">
        {fees > 0 ? formatCents(fees) : '-'}
      </td>
      <td className={`py-2 px-2 text-right font-mono font-bold ${getPnLColor(netPnl)}`}>
        {formatSettlementCurrency(netPnl)}
      </td>
      <td className="py-2 px-2 text-right text-xs text-gray-500">
        {formatTime(settlement.closed_at)}
      </td>
    </tr>
  );
});

SettlementRow.displayName = 'SettlementRow';

/**
 * SettlementsPanel - Shows history of closed positions with economics
 */
const SettlementsPanel = ({ settlements }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Calculate total net P&L across all settlements
  const totalNetPnl = (settlements || []).reduce(
    (sum, s) => sum + (s.net_pnl || s.realized_pnl || 0), 0
  );

  const settlementsData = settlements || [];
  const hasSettlements = settlementsData.length > 0;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
          <CheckCircle className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">
            Recent Settlements
          </h3>
          <span className="text-xs text-gray-500">({settlementsData.length})</span>
        </div>
        <div className={`px-2 py-0.5 rounded text-xs font-bold font-mono ${getPnLColor(totalNetPnl)}`}>
          {formatSettlementCurrency(totalNetPnl)}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 max-h-[280px] overflow-y-auto bg-gray-800/30 rounded-lg border border-gray-700/50">
          {hasSettlements ? (
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="text-xs text-gray-500 uppercase border-b border-gray-800 bg-gray-900">
                  <th className="text-left py-2 px-2">Ticker</th>
                  <th className="text-center py-2 px-2">Side</th>
                  <th className="text-right py-2 px-2">Qty</th>
                  <th className="text-right py-2 px-2">Cost</th>
                  <th className="text-right py-2 px-2">Payout</th>
                  <th className="text-right py-2 px-2">Fees</th>
                  <th className="text-right py-2 px-2">Net P&L</th>
                  <th className="text-right py-2 px-2">Closed</th>
                </tr>
              </thead>
              <tbody>
                {settlementsData.map((s, idx) => (
                  <SettlementRow
                    key={`${s.ticker}-${s.closed_at}-${idx}`}
                    settlement={s}
                    index={idx}
                  />
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-gray-500">
              <CheckCircle className="w-8 h-8 mb-2 text-gray-600" />
              <span className="text-sm">No settlements yet</span>
              <span className="text-xs text-gray-600 mt-1">Closed positions will appear here</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default memo(SettlementsPanel);
