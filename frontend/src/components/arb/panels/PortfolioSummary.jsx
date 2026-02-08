import React, { memo } from 'react';
import { DollarSign, TrendingUp, Briefcase } from 'lucide-react';

const formatDollars = (cents) => {
  const d = (cents || 0) / 100;
  return `$${d.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const PortfolioSummary = memo(({ tradingState }) => {
  const balance = tradingState?.balance ?? 0;
  const pnl = tradingState?.pnl;
  const posCount = tradingState?.position_count ?? 0;

  const pnlCents = pnl?.realized_pnl_cents ?? pnl?.total_pnl_cents ?? 0;
  const pnlDollars = pnlCents / 100;
  const pnlColor = pnlDollars >= 0 ? 'text-emerald-400' : 'text-red-400';

  return (
    <div className="px-3 py-3 border-b border-gray-800/30">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <DollarSign className="w-3 h-3" />
            Balance
          </div>
          <span className="text-[13px] font-mono font-semibold text-cyan-400 tabular-nums">
            {formatDollars(balance)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <TrendingUp className="w-3 h-3" />
            P&L
          </div>
          <span className={`text-[13px] font-mono font-semibold tabular-nums ${pnlColor}`}>
            {pnlDollars >= 0 ? '+' : ''}{formatDollars(pnlCents)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <Briefcase className="w-3 h-3" />
            Positions
          </div>
          <span className="text-[12px] font-mono text-gray-300 tabular-nums">{posCount}</span>
        </div>
      </div>
    </div>
  );
});

PortfolioSummary.displayName = 'PortfolioSummary';

export default PortfolioSummary;
