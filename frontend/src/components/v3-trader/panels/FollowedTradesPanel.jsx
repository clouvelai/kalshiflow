import React, { useState, memo } from 'react';
import { ChevronRight, ChevronDown, CheckCircle } from 'lucide-react';
import { formatAge, getSideClasses } from '../../../utils/v3-trader';

/**
 * FollowedTradeRow - Memoized row component for followed trades table
 */
const FollowedTradeRow = memo(({ trade, index }) => (
  <tr
    className="border-b border-gray-700/30 hover:bg-gray-800/50 transition-colors bg-green-900/5"
  >
    <td className="px-3 py-2 font-mono text-gray-300 text-xs">{trade.market_ticker}</td>
    <td className="px-3 py-2 text-center">
      <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
        trade.side === 'yes'
          ? 'bg-green-900/30 text-green-400 border border-green-700/50'
          : 'bg-red-900/30 text-red-400 border border-red-700/50'
      }`}>
        {trade.side}
      </span>
    </td>
    <td className="px-3 py-2 text-right font-mono text-gray-300">{trade.price_cents}c</td>
    <td className="px-3 py-2 text-right font-mono text-gray-300">{trade.our_count}</td>
    <td className="px-3 py-2 text-right font-mono text-gray-400">${trade.cost_dollars?.toFixed(2)}</td>
    <td className="px-3 py-2 text-right font-mono text-gray-400">${trade.payout_dollars?.toFixed(2)}</td>
    <td className="px-3 py-2 text-right font-mono font-bold text-green-400">
      ${((trade.payout_dollars || 0) - (trade.cost_dollars || 0)).toFixed(2)}
    </td>
    <td className="px-3 py-2 text-right font-mono text-gray-500 text-xs">{formatAge(trade.age_seconds)}</td>
  </tr>
));

FollowedTradeRow.displayName = 'FollowedTradeRow';

/**
 * FollowedTradesPanel - Shows trades we've followed (persists beyond whale queue window)
 */
const FollowedTradesPanel = ({ followedWhales }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  // Don't render if no followed trades
  if (!followedWhales || followedWhales.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-green-800/50 p-4 mt-4">
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
          <CheckCircle className="w-4 h-4 text-green-400" />
          <h3 className="text-sm font-bold text-green-400 uppercase tracking-wider">Followed Trades</h3>
          <span className="text-xs text-gray-500">({followedWhales.length})</span>
        </div>
      </div>

      {isExpanded && (
        <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden mt-4 max-h-[280px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10">
              <tr className="bg-gray-900 border-b border-gray-700/50">
                <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Market</th>
                <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Price</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Count</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Cost</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Payout</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Size</th>
                <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Age</th>
              </tr>
            </thead>
            <tbody>
              {followedWhales.map((trade, index) => (
                <FollowedTradeRow
                  key={trade.whale_id || index}
                  trade={trade}
                  index={index}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default memo(FollowedTradesPanel);
