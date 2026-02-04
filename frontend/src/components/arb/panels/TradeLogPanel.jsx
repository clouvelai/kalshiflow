import React, { memo } from 'react';
import { Zap, Clock } from 'lucide-react';

/**
 * Format an ISO timestamp to a short time string
 */
const formatTime = (ts) => {
  if (!ts) return '--';
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return ts;
  }
};

/**
 * TradeLogPanel - Table of arbitrage trade executions, most recent first
 */
const TradeLogPanel = ({ arbTrades = [] }) => {
  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-gray-800/80
      shadow-xl shadow-black/20
      p-5 flex flex-col h-full
    ">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-amber-900/30 to-amber-950/20 border border-amber-800/30">
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Trade Log</h3>
            <span className="text-[10px] text-gray-500 font-mono">{arbTrades.length} trades</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {arbTrades.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-sm">
            No arb trades yet.
          </div>
        ) : (
          <div className="
            bg-gradient-to-b from-gray-800/30 to-gray-900/30
            rounded-xl border border-gray-700/40
            overflow-hidden
          ">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-900/80 border-b border-gray-700/40">
                  <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">
                    <Clock className="w-3 h-3 inline mr-1" />Time
                  </th>
                  <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                  <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Price</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Spread</th>
                </tr>
              </thead>
              <tbody>
                {arbTrades.map((trade) => {
                  const sideColor = trade.side === 'yes'
                    ? 'bg-green-900/30 text-green-400 border-green-600/30'
                    : 'bg-red-900/30 text-red-400 border-red-600/30';

                  return (
                    <tr
                      key={trade.id}
                      className="border-b border-gray-700/20 hover:bg-gray-800/50 transition-colors"
                      title={trade.reasoning || ''}
                    >
                      <td className="px-3 py-2 font-mono text-[11px] text-gray-400">
                        {formatTime(trade.timestamp)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-gray-300">
                        {trade.kalshi_ticker}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${sideColor}`}>
                          {trade.side}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-300">
                        {trade.contracts}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-cyan-400">
                        {trade.price_cents}c
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">
                        {trade.spread_at_entry != null ? `${trade.spread_at_entry.toFixed(1)}c` : '--'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(TradeLogPanel);
