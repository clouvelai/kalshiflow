import React, { memo } from 'react';
import { Zap, Clock } from 'lucide-react';

const formatTime = (ts) => {
  if (!ts) return '--';
  try {
    const d = new Date(typeof ts === 'number' ? ts * 1000 : ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '--';
  }
};

const TYPE_COLORS = {
  quote_placed: 'bg-blue-500/10 text-blue-400 border-blue-600/20',
  quote_filled: 'bg-emerald-500/10 text-emerald-400 border-emerald-600/20',
  quote_cancelled: 'bg-gray-500/10 text-gray-400 border-gray-600/20',
  quotes_pulled: 'bg-red-500/10 text-red-400 border-red-600/20',
};

/**
 * TradeLogPanel - Fills, quote updates, cancels.
 */
const TradeLogPanel = ({ tradeLog = [] }) => {
  return (
    <div className="
      bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
      backdrop-blur-sm rounded-xl
      border border-gray-800/30
      shadow-lg shadow-black/10
      flex flex-col h-full
    ">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/20 shrink-0">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400/70" />
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Trade Log</span>
          <span className="text-[10px] text-gray-500 font-mono">{tradeLog.length}</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {tradeLog.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-sm">No activity yet.</div>
        ) : (
          <div className="bg-gradient-to-b from-gray-800/30 to-gray-900/30 rounded-xl border border-gray-700/40 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-900/80 border-b border-gray-700/40">
                  <th className="px-3 py-2 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">
                    <Clock className="w-3 h-3 inline mr-1" />Time
                  </th>
                  <th className="px-3 py-2 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Type</th>
                  <th className="px-3 py-2 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                  <th className="px-3 py-2 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                  <th className="px-3 py-2 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Price</th>
                  <th className="px-3 py-2 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                </tr>
              </thead>
              <tbody>
                {tradeLog.map((entry, i) => {
                  const typeStyle = TYPE_COLORS[entry.type] || 'bg-gray-500/10 text-gray-400 border-gray-600/20';
                  return (
                    <tr key={i} className="border-b border-gray-700/20 hover:bg-gray-800/50 transition-colors">
                      <td className="px-3 py-1.5 font-mono text-[11px] text-gray-400">
                        {formatTime(entry.timestamp)}
                      </td>
                      <td className="px-3 py-1.5">
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase border ${typeStyle}`}>
                          {(entry.type || '').replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 font-mono text-[10px] text-gray-300 truncate max-w-[120px]" title={entry.ticker || entry.market_ticker}>
                        {(() => { const t = entry.ticker || entry.market_ticker || '--'; const p = t.split('-'); return p.length >= 3 ? p[p.length - 1] : t; })()}
                      </td>
                      <td className="px-3 py-1.5 text-center">
                        {(entry.side || entry.quote_side) && (
                          <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                            (entry.side === 'yes' || entry.quote_side === 'bid')
                              ? 'bg-cyan-900/30 text-cyan-400'
                              : 'bg-red-900/30 text-red-400'
                          }`}>
                            {entry.quote_side || entry.side}
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-[10px] text-cyan-400 tabular-nums">
                        {entry.price_cents != null ? `${entry.price_cents}c` : '--'}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-[10px] text-gray-300 tabular-nums">
                        {entry.count || entry.size || entry.cancelled || '--'}
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
