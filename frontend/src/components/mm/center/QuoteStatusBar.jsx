import React, { memo } from 'react';

/**
 * QuoteStatusBar - Shows current quote state: active quotes, spread, requote cycle count.
 */
const QuoteStatusBar = ({ quoteState }) => {
  if (!quoteState) return null;

  const activeQuotes = quoteState.active_quotes ?? 0;
  const totalCycles = quoteState.total_requote_cycles ?? 0;
  const pulled = quoteState.quotes_pulled ?? false;
  const pullReason = quoteState.pull_reason || '';

  return (
    <div className={`flex items-center gap-3 px-3 py-1.5 border-b shrink-0 ${
      pulled ? 'border-red-500/20 bg-red-500/5' : 'border-gray-800/20'
    }`}>
      <div className="flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${pulled ? 'bg-red-500' : activeQuotes > 0 ? 'bg-emerald-500' : 'bg-gray-600'}`} />
        <span className={`text-[9px] font-semibold uppercase tracking-wider ${pulled ? 'text-red-400' : activeQuotes > 0 ? 'text-emerald-400' : 'text-gray-600'}`}>
          {pulled ? 'Quotes Pulled' : `${activeQuotes} Active`}
        </span>
      </div>

      {pulled && pullReason && (
        <span className="text-[9px] text-red-400/60 truncate max-w-[200px]">{pullReason}</span>
      )}

      {quoteState.spread_multiplier != null && quoteState.spread_multiplier !== 1 && (
        <span className="text-[9px] font-mono text-amber-400">
          spread {quoteState.spread_multiplier.toFixed(1)}x
        </span>
      )}

      {quoteState.fees_paid_cents > 0 && (
        <span className="text-[9px] font-mono text-gray-600">
          fees: ${(quoteState.fees_paid_cents / 100).toFixed(2)}
        </span>
      )}

      <span className="text-[9px] font-mono text-gray-600 ml-auto">
        cycle #{totalCycles}
      </span>
    </div>
  );
};

export default memo(QuoteStatusBar);
