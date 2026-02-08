import React, { memo } from 'react';
import { fmtTime } from '../../utils/formatters';

const QuickTrades = memo(({ trades = [] }) => {
  const recent = trades.slice(0, 5);
  if (recent.length === 0) return null;

  return (
    <div className="border-t border-gray-800/30 px-3 py-2 shrink-0">
      <div className="text-[8px] font-semibold text-gray-600 uppercase tracking-wider mb-1">Recent Trades</div>
      <div className="space-y-0.5">
        {recent.map((trade) => {
          const isBuy = trade.action?.toLowerCase() !== 'sell';
          return (
            <div key={trade.id} className="flex items-center gap-1.5 text-[9px]">
              <span className={`font-semibold ${isBuy ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
                {isBuy ? 'B' : 'S'}
              </span>
              <span className="font-mono text-gray-400 truncate flex-1 min-w-0">
                {trade.kalshi_ticker || 'unknown'}
              </span>
              <span className="font-mono text-gray-500 tabular-nums shrink-0">
                {trade.contracts}x{trade.price_cents}c
              </span>
              <span className="font-mono text-gray-700 tabular-nums shrink-0">
                {fmtTime(trade.timestamp)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});

QuickTrades.displayName = 'QuickTrades';

export default QuickTrades;
