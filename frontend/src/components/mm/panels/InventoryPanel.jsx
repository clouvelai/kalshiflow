import React, { memo } from 'react';
import { Briefcase } from 'lucide-react';

const formatPnL = (cents) => {
  const d = (cents || 0) / 100;
  const prefix = d >= 0 ? '+' : '';
  return `${prefix}$${Math.abs(d).toFixed(2)}`;
};

const pnlColor = (v) => v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-gray-500';

// Extract short suffix from ticker (e.g., KXNEWPOPE-70-PPIZ → PPIZ)
const shortName = (ticker) => {
  const parts = (ticker || '').split('-');
  return parts.length >= 3 ? parts[parts.length - 1] : ticker;
};

/**
 * InventoryPanel - Shows per-market quotes, positions, and P&L.
 */
const InventoryPanel = ({ inventory = [] }) => {
  const totalPnL = inventory.reduce((s, m) => s + (m.realized_pnl_cents || 0) + (m.unrealized_pnl_cents || 0), 0);
  const totalExposure = inventory.reduce((s, m) => s + Math.abs(m.position || 0), 0);
  const activeQuotes = inventory.filter(m => m.bid_quote || m.ask_quote).length;

  return (
    <div className="
      bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
      backdrop-blur-sm rounded-xl
      border border-gray-800/30
      shadow-lg shadow-black/10
      flex flex-col h-full
    ">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/20 shrink-0">
        <div className="flex items-center gap-2">
          <Briefcase className="w-4 h-4 text-violet-400/70" />
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Markets</span>
          <span className="text-[10px] text-gray-500 font-mono tabular-nums">
            {inventory.length}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {totalExposure > 0 && (
            <div className="text-[9px] text-gray-500">
              Exp: <span className="font-mono text-gray-400 tabular-nums">{totalExposure}</span>
            </div>
          )}
          <span className={`font-mono text-sm font-semibold tabular-nums ${pnlColor(totalPnL)}`}>
            {formatPnL(totalPnL)}
          </span>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {inventory.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <Briefcase className="w-6 h-6 mb-2 opacity-30" />
            <span className="text-[11px]">No markets loaded</span>
          </div>
        ) : (
          <div className="divide-y divide-gray-800/15">
            {inventory.map(m => {
              const unrealizedPnL = m.unrealized_pnl_cents || 0;
              const realizedPnL = m.realized_pnl_cents || 0;
              const totalMarketPnL = unrealizedPnL + realizedPnL;
              const pos = m.position || 0;
              const side = pos > 0 ? 'LONG' : pos < 0 ? 'SHORT' : null;
              const sideColor = side === 'LONG'
                ? 'bg-blue-500/10 text-blue-300/80'
                : side === 'SHORT'
                  ? 'bg-orange-500/10 text-orange-300/80'
                  : '';
              const bidPrice = m.bid_quote?.price;
              // Ask quote price is already in YES terms from backend
              const askYesPrice = m.ask_quote?.price ?? null;

              return (
                <div key={m.ticker} className="px-3 py-2 hover:bg-gray-800/15 transition-colors">
                  {/* Row 1: Name + position/side */}
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="text-[11px] font-semibold text-gray-300 truncate" title={m.title}>
                        {shortName(m.ticker)}
                      </span>
                      <span className="text-[9px] font-mono text-gray-600 truncate" title={m.ticker}>
                        {m.title?.slice(0, 25)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {side && (
                        <span className={`px-1.5 py-px rounded text-[8px] font-semibold uppercase ${sideColor}`}>
                          {side} {Math.abs(pos)}
                        </span>
                      )}
                      {totalMarketPnL !== 0 && (
                        <span className={`text-[10px] font-mono font-semibold tabular-nums ${pnlColor(totalMarketPnL)}`}>
                          {formatPnL(totalMarketPnL)}
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Row 2: Our quotes + mid */}
                  <div className="flex items-center justify-between text-[10px] font-mono tabular-nums">
                    <div className="flex items-center gap-2">
                      {bidPrice != null ? (
                        <span className="text-cyan-400/70">
                          <span className="text-[8px] text-gray-600 mr-0.5">B</span>{bidPrice}c
                        </span>
                      ) : (
                        <span className="text-gray-700">--</span>
                      )}
                      {m.mid_cents != null && (
                        <span className="text-gray-500">
                          <span className="text-[8px] text-gray-700 mr-0.5">FV</span>{typeof m.mid_cents === 'number' ? m.mid_cents.toFixed(1) : m.mid_cents}c
                        </span>
                      )}
                      {askYesPrice != null ? (
                        <span className="text-red-400/70">
                          <span className="text-[8px] text-gray-600 mr-0.5">A</span>{askYesPrice}c
                        </span>
                      ) : (
                        <span className="text-gray-700">--</span>
                      )}
                    </div>
                    {(m.total_buys > 0 || m.total_sells > 0) && (
                      <span className="text-[9px] text-gray-600">
                        {m.total_buys}B/{m.total_sells}S
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(InventoryPanel);
