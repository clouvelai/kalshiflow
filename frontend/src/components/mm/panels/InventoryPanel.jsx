import React, { memo } from 'react';
import { Briefcase } from 'lucide-react';

const formatPnL = (cents) => {
  const d = (cents || 0) / 100;
  const prefix = d >= 0 ? '+' : '';
  return `${prefix}$${Math.abs(d).toFixed(2)}`;
};

const pnlColor = (v) => v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-gray-500';

/**
 * InventoryPanel - Shows positions, exposure, P&L per market.
 */
const InventoryPanel = ({ inventory = [] }) => {
  const totalPnL = inventory.reduce((s, m) => s + (m.unrealized_pnl_cents || 0), 0);
  const totalExposure = inventory.reduce((s, m) => s + Math.abs(m.position || 0), 0);

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
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Inventory</span>
          <span className="text-[10px] text-gray-500 font-mono tabular-nums">
            {inventory.length} market{inventory.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-[9px] text-gray-500">
            Exposure: <span className="font-mono text-gray-400 tabular-nums">{totalExposure}</span>
          </div>
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
            <span className="text-[11px]">No inventory</span>
          </div>
        ) : (
          <div>
            {/* Column headers */}
            <div className="flex items-center gap-2 px-3 py-1.5 text-[8px] text-gray-600 uppercase tracking-wider border-b border-gray-800/20 sticky top-0 bg-gray-900/90 backdrop-blur-sm z-10">
              <span className="flex-1">Market</span>
              <span className="w-10 text-center shrink-0">Side</span>
              <span className="w-10 text-right shrink-0">Pos</span>
              <span className="w-12 text-right shrink-0">Avg</span>
              <span className="w-12 text-right shrink-0">Mid</span>
              <span className="w-16 text-right shrink-0">P&L</span>
            </div>
            {inventory.map(m => {
              const unrealizedPnL = m.unrealized_pnl_cents || 0;
              const side = (m.position || 0) > 0 ? 'LONG' : (m.position || 0) < 0 ? 'SHORT' : 'FLAT';
              const sideColor = side === 'LONG'
                ? 'bg-blue-500/10 text-blue-300/80'
                : side === 'SHORT'
                  ? 'bg-orange-500/10 text-orange-300/80'
                  : 'bg-gray-500/10 text-gray-500';

              return (
                <div
                  key={m.ticker}
                  className="flex items-center gap-2 px-3 py-1.5 hover:bg-gray-800/20 transition-colors border-b border-gray-700/10"
                >
                  <span className="flex-1 text-[10px] font-mono text-gray-400 truncate min-w-0" title={m.ticker}>
                    {m.ticker}
                  </span>
                  <span className={`w-10 text-center shrink-0 px-1.5 py-px rounded text-[8px] font-semibold uppercase ${sideColor}`}>
                    {side}
                  </span>
                  <span className="w-10 text-right shrink-0 text-[10px] font-mono text-gray-300 tabular-nums">
                    {Math.abs(m.position || 0)}
                  </span>
                  <span className="w-12 text-right shrink-0 text-[10px] font-mono text-gray-500 tabular-nums">
                    {m.avg_entry_cents != null ? `${m.avg_entry_cents}c` : '--'}
                  </span>
                  <span className="w-12 text-right shrink-0 text-[10px] font-mono text-cyan-400/70 tabular-nums">
                    {m.mid_cents != null ? `${m.mid_cents}c` : '--'}
                  </span>
                  <span className={`w-16 text-right shrink-0 text-[10px] font-mono font-semibold tabular-nums ${pnlColor(unrealizedPnL)}`}>
                    {formatPnL(unrealizedPnL)}
                  </span>
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
