import React, { memo } from 'react';
import { BarChart3 } from 'lucide-react';

const formatCents = (c) => {
  if (c == null) return '--';
  const d = c / 100;
  return `$${Math.abs(d).toFixed(2)}`;
};

const formatPct = (v) => {
  if (v == null) return '--';
  return `${(v * 100).toFixed(1)}%`;
};

/**
 * QuotePerformancePanel - Fill rates, spread capture, adverse selection.
 */
const QuotePerformancePanel = ({ performance }) => {
  if (!performance) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600">
        <div className="text-center">
          <BarChart3 className="w-6 h-6 mx-auto mb-2 opacity-30" />
          <span className="text-[11px]">No performance data yet</span>
        </div>
      </div>
    );
  }

  const metrics = [
    { label: 'Spread Captured', value: formatCents(performance.spread_captured_cents), color: 'text-emerald-400' },
    { label: 'Adverse Selection', value: formatCents(performance.adverse_selection_cents), color: 'text-red-400' },
    { label: 'Fees Paid', value: formatCents(performance.fees_paid_cents), color: 'text-amber-400' },
    { label: 'Realized P&L', value: formatCents(performance.realized_pnl_cents), color: performance?.realized_pnl_cents >= 0 ? 'text-emerald-400' : 'text-red-400' },
    { label: 'Bid Fill Rate', value: formatPct(performance.bid_fill_rate), color: 'text-cyan-400' },
    { label: 'Ask Fill Rate', value: formatPct(performance.ask_fill_rate), color: 'text-cyan-400' },
    { label: 'Quote Uptime', value: formatPct(performance.quote_uptime_pct), color: 'text-violet-400' },
    { label: 'Total Fills', value: String((performance.total_fills_bid || 0) + (performance.total_fills_ask || 0)), color: 'text-gray-300' },
  ];

  return (
    <div className="flex-1 flex flex-col min-h-0 p-4">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-4 h-4 text-violet-400/70" />
        <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Performance</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {metrics.map(m => (
          <div
            key={m.label}
            className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/20"
          >
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">{m.label}</div>
            <div className={`text-lg font-mono font-semibold tabular-nums ${m.color}`}>{m.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default memo(QuotePerformancePanel);
