import React, { memo, useMemo } from 'react';
import { Briefcase, TrendingUp, TrendingDown } from 'lucide-react';

/**
 * PositionPanel - Open positions table from tradingState
 *
 * Shows: ticker, side, quantity, avg_price, market_price, unrealized P&L
 */
const PositionPanel = ({ tradingState }) => {
  const positions = tradingState?.positions || [];

  // Calculate total unrealized P&L
  const totalPnL = useMemo(() => {
    return positions.reduce((sum, pos) => sum + (pos.unrealized_pnl ?? 0), 0);
  }, [positions]);

  const formatCents = (c) => `${c ?? 0}c`;
  const formatPnL = (cents) => {
    const d = (cents || 0) / 100;
    const prefix = d >= 0 ? '+' : '';
    return `${prefix}$${Math.abs(d).toFixed(2)}`;
  };

  return (
    <div className="
      bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
      backdrop-blur-sm rounded-xl
      border border-gray-800/30
      shadow-lg shadow-black/10
      p-4 flex flex-col h-full
    ">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Briefcase className="w-4 h-4 text-violet-400/70" />
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Positions</span>
          <span className="text-[10px] text-gray-500 font-mono tabular-nums">{positions.length} open</span>
        </div>
        {positions.length > 0 && (
          <span className={`font-mono text-sm font-semibold tabular-nums ${totalPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {formatPnL(totalPnL)}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {positions.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-[11px]">
            No open positions
          </div>
        ) : (
          <div className="
            bg-gray-800/15
            rounded-lg border border-gray-700/20
            overflow-hidden
          ">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-900/80 border-b border-gray-700/20">
                  <th className="px-3 py-2 text-left text-[9px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                  <th className="px-3 py-2 text-center text-[9px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                  <th className="px-3 py-2 text-right text-[9px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                  <th className="px-3 py-2 text-right text-[9px] text-gray-500 uppercase font-semibold tracking-wider">Avg</th>
                  <th className="px-3 py-2 text-right text-[9px] text-gray-500 uppercase font-semibold tracking-wider">Mkt</th>
                  <th className="px-3 py-2 text-right text-[9px] text-gray-500 uppercase font-semibold tracking-wider">P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => {
                  const qty = Math.abs(pos.position || pos.quantity || 0);
                  const avgPrice = pos.total_cost && qty > 0 ? Math.round(pos.total_cost / qty) : (pos.avg_price ?? 0);
                  const mktPrice = pos.current_value && qty > 0 ? Math.round(pos.current_value / qty) : (pos.market_price ?? 0);
                  const unrealizedPnL = pos.unrealized_pnl ?? 0;

                  const sideColor = pos.side === 'yes'
                    ? 'bg-blue-500/10 text-blue-300/80'
                    : 'bg-orange-500/10 text-orange-300/80';

                  const PnlIcon = unrealizedPnL >= 0 ? TrendingUp : TrendingDown;

                  return (
                    <tr key={pos.ticker || idx} className="border-b border-gray-700/15 hover:bg-gray-800/30 transition-colors">
                      <td className="px-3 py-1.5 font-mono text-gray-300 tabular-nums">{pos.ticker}</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className={`px-1.5 py-px rounded text-[9px] font-semibold uppercase ${sideColor}`}>
                          {pos.side}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-gray-300 tabular-nums">{qty}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-gray-400 tabular-nums">{formatCents(avgPrice)}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-cyan-400/80 tabular-nums">{formatCents(mktPrice)}</td>
                      <td className="px-3 py-1.5 text-right">
                        <div className="flex items-center justify-end gap-1 tabular-nums">
                          <PnlIcon className={`w-2.5 h-2.5 ${unrealizedPnL >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`} />
                          <span className={`font-mono font-semibold ${unrealizedPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {formatPnL(unrealizedPnL)}
                          </span>
                        </div>
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

export default memo(PositionPanel);
