import React, { memo } from 'react';
import { ArrowUpCircle, Check } from 'lucide-react';
import { fmtTime } from '../../utils/formatters';

/**
 * TradeCard - Single Captain trade.
 */
const TradeCard = memo(({ trade }) => {
  const isBuy = trade.action?.toLowerCase() !== 'sell';
  const isYes = trade.side?.toLowerCase() === 'yes';

  return (
    <div className={`rounded-lg border p-3 ${
      isBuy ? 'border-emerald-500/12 bg-emerald-950/8' : 'border-red-500/12 bg-red-950/8'
    }`}>
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`text-[9px] font-semibold px-1.5 py-px rounded uppercase tracking-wider ${
          isBuy ? 'bg-emerald-500/15 text-emerald-300/80' : 'bg-red-500/15 text-red-300/80'
        }`}>
          {trade.action?.toUpperCase() || 'BUY'}
        </span>
        <span className={`text-[9px] font-semibold px-1.5 py-px rounded uppercase ${
          isYes ? 'bg-blue-500/12 text-blue-300/80' : 'bg-orange-500/12 text-orange-300/80'
        }`}>
          {trade.side?.toUpperCase() || '?'}
        </span>
        <span className="text-[11px] font-mono text-gray-300">
          {trade.kalshi_ticker || 'unknown'}
        </span>
        {trade.contracts && (
          <span className="text-[10px] text-gray-500 font-mono">x{trade.contracts}</span>
        )}
        {trade.price_cents != null && (
          <span className="text-[10px] text-gray-400 font-mono">@{trade.price_cents}c</span>
        )}
        {trade.spread_at_entry != null && (
          <span className="text-[10px] text-cyan-500/70 font-mono">
            spread {trade.spread_at_entry.toFixed(1)}c
          </span>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <Check className="w-3 h-3 text-emerald-400/60" />
          <span className="text-[9px] text-gray-600 font-mono tabular-nums">
            {fmtTime(trade.timestamp)}
          </span>
        </div>
      </div>
      {trade.reasoning && (
        <div className="mt-1.5 pt-1.5 border-t border-gray-800/20">
          <span className="text-[10px] text-gray-500 leading-relaxed line-clamp-2" title={trade.reasoning}>
            {trade.reasoning}
          </span>
        </div>
      )}
    </div>
  );
});
TradeCard.displayName = 'TradeCard';

/**
 * TradesSection - List of Captain trades.
 */
const TradesSection = memo(({ trades }) => {
  if (trades.length === 0) return null;

  return (
    <div id="trades-section" data-testid="trades-section" data-count={trades.length} className="space-y-2">
      <div className="flex items-center gap-2 px-0.5">
        <ArrowUpCircle className="w-3.5 h-3.5 text-amber-500/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
          Trades
        </span>
        <span data-testid="trades-count" className="text-[10px] text-gray-600 font-mono tabular-nums ml-auto">
          {trades.length}
        </span>
      </div>
      <div className="max-h-[280px] overflow-y-auto space-y-1.5">
        {trades.map(trade => (
          <TradeCard key={trade.id} trade={trade} />
        ))}
      </div>
    </div>
  );
});
TradesSection.displayName = 'TradesSection';

export { TradeCard, TradesSection };
export default TradesSection;
