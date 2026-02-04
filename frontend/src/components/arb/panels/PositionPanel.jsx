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
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-gray-800/80
      shadow-xl shadow-black/20
      p-5 flex flex-col h-full
    ">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-purple-900/30 to-purple-950/20 border border-purple-800/30">
            <Briefcase className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Positions</h3>
            <span className="text-[10px] text-gray-500 font-mono">{positions.length} open</span>
          </div>
        </div>
        {positions.length > 0 && (
          <span className={`font-mono text-sm font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatPnL(totalPnL)}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {positions.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-sm">
            No open positions.
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
                  <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                  <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Avg Price</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Mkt Price</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => {
                  const qty = Math.abs(pos.position || pos.quantity || 0);
                  const avgPrice = pos.total_cost && qty > 0 ? Math.round(pos.total_cost / qty) : (pos.avg_price ?? 0);
                  const mktPrice = pos.current_value && qty > 0 ? Math.round(pos.current_value / qty) : (pos.market_price ?? 0);
                  const unrealizedPnL = pos.unrealized_pnl ?? 0;

                  const sideColor = pos.side === 'yes'
                    ? 'bg-green-900/30 text-green-400 border-green-600/30'
                    : 'bg-red-900/30 text-red-400 border-red-600/30';

                  const PnlIcon = unrealizedPnL >= 0 ? TrendingUp : TrendingDown;

                  return (
                    <tr key={pos.ticker || idx} className="border-b border-gray-700/20 hover:bg-gray-800/50 transition-colors">
                      <td className="px-3 py-2 font-mono text-xs text-gray-300">{pos.ticker}</td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${sideColor}`}>
                          {pos.side}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-gray-300">{qty}</td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">{formatCents(avgPrice)}</td>
                      <td className="px-3 py-2 text-right font-mono text-cyan-400">{formatCents(mktPrice)}</td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex items-center justify-end gap-1">
                          <PnlIcon className={`w-3 h-3 ${unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`} />
                          <span className={`font-mono text-xs font-bold ${unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
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
