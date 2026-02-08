import React, { memo, useMemo } from 'react';
import { Sparkles, Loader2, Brain, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import EdgeBadge from '../../ui/EdgeBadge';

const MentionsTermRow = memo(function MentionsTermRow({ term, delta, markets }) {
  const prob = term.probability || 0;
  const termPrefix = term.term?.toLowerCase().slice(0, 4) || '';
  const market = markets?.find(m => m.ticker?.toLowerCase().includes(termPrefix));
  const marketYes = market?.yes_bid || market?.yes_mid || 50;
  const simProbCents = Math.round(prob * 100);
  const edge = simProbCents - marketYes;
  const trend = delta?.trend || '\u2192';
  const TrendIcon = trend === '\u2191' ? TrendingUp : trend === '\u2193' ? TrendingDown : Minus;
  const trendColor = trend === '\u2191' ? 'text-emerald-400' : trend === '\u2193' ? 'text-red-400' : 'text-gray-500';
  const edgeColor = Math.abs(edge) >= 10 ? (edge > 0 ? 'text-emerald-400' : 'text-red-400') : 'text-gray-400';
  const rowBg = Math.abs(edge) >= 15 ? (edge > 0 ? 'bg-emerald-950/15' : 'bg-red-950/15') : '';

  return (
    <div className={`flex items-center justify-between text-[10px] py-1 border-b border-gray-700/15 last:border-0 ${rowBg}`}>
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-300 w-24 truncate" title={term.term}>{term.term}</span>
        <TrendIcon className={`w-2.5 h-2.5 ${trendColor}`} />
      </div>
      <div className="flex items-center gap-3 tabular-nums">
        <span className="text-cyan-400/90 font-mono w-12 text-right">{(prob * 100).toFixed(0)}%</span>
        <span className="text-gray-600 w-8 text-center text-[9px]">vs</span>
        <span className="text-gray-300 font-mono w-10 text-right">{marketYes}c</span>
        <span className={`font-mono font-semibold w-12 text-right ${edgeColor}`}>{edge > 0 ? '+' : ''}{edge}c</span>
      </div>
    </div>
  );
});

const MentionsStrategySection = memo(function MentionsStrategySection({ mentions, markets }) {
  if (!mentions) return null;
  const hasTerms = mentions.terms?.length > 0;
  const marketList = markets ? Object.values(markets) : [];

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-amber-500/10">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-amber-400/80" />
          <span className="text-[9px] font-semibold text-amber-400/70 uppercase tracking-wider">Mentions Strategy</span>
        </div>
        <div className="flex items-center gap-2">
          {mentions.simulation_in_progress && (
            <span className="flex items-center gap-1 text-[9px] text-violet-400/80">
              <Loader2 className="w-2.5 h-2.5 animate-spin" /> Simulating
            </span>
          )}
          {mentions.history_count > 0 && (
            <span className="text-[9px] text-gray-500 font-mono tabular-nums bg-gray-800/40 rounded-full px-2 py-0.5">
              {mentions.history_count} sim{mentions.history_count !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>
      {hasTerms ? (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[9px] text-gray-500 uppercase tracking-wider pb-1 border-b border-gray-700/20">
            <span className="w-24">Term</span>
            <div className="flex items-center gap-3">
              <span className="w-12 text-right">Sim P</span>
              <span className="w-8 text-center">vs</span>
              <span className="w-10 text-right">Mkt</span>
              <span className="w-12 text-right">Edge</span>
            </div>
          </div>
          {mentions.terms.map(t => (
            <MentionsTermRow key={t.term} term={t} delta={mentions.deltas?.[t.term]} markets={marketList} />
          ))}
        </div>
      ) : (
        <div className="text-[10px] text-gray-600 italic">No simulation data yet</div>
      )}
      {mentions.news_context?.length > 0 && (
        <div className="text-[9px] text-gray-500 italic border-t border-gray-700/20 pt-2 mt-2">
          <Brain className="w-2.5 h-2.5 inline mr-1 text-violet-400/60" />
          {mentions.news_context[0]}
        </div>
      )}
    </div>
  );
});

const MarketsTab = memo(({ event, positionsByTicker, selectedMarket, onSelectMarket, mentionsData }) => {
  const {
    sum_yes_bid, sum_yes_ask, sum_yes_mid,
    long_edge, short_edge, signals,
    market_count, markets_with_data, all_markets_have_data,
    mutually_exclusive, event_type, series_ticker,
    markets: rawMarkets,
  } = event;

  const isIndependent = event_type === 'independent' || !mutually_exclusive;
  const markets = rawMarkets || {};
  const marketList = Object.values(markets);

  const formatVol = (v) => {
    if (v == null || v <= 0) return '--';
    if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
    if (v >= 1000) return `$${(v / 1000).toFixed(0)}k`;
    return `$${Math.round(v)}`;
  };

  const getFreshnessDisplay = (seconds) => {
    if (seconds == null || seconds >= 9999) return { text: '--', color: 'text-gray-600' };
    if (seconds < 5) return { text: `${Math.round(seconds)}s`, color: 'text-emerald-500' };
    if (seconds < 30) return { text: `${Math.round(seconds)}s`, color: 'text-amber-500' };
    if (seconds < 60) return { text: `${Math.round(seconds)}s`, color: 'text-red-400' };
    return { text: `${Math.floor(seconds / 60)}m`, color: 'text-red-400' };
  };

  const hasPositions = useMemo(() => marketList.some(m => positionsByTicker[m.ticker]), [marketList, positionsByTicker]);

  const positionSummary = useMemo(() => {
    if (!hasPositions) return null;
    let marketsWithPos = 0, totalUnrealized = 0, totalCost = 0, totalExposure = 0;
    for (const m of marketList) {
      const pos = positionsByTicker[m.ticker];
      if (pos) {
        marketsWithPos++;
        totalUnrealized += pos.unrealized_pnl || 0;
        totalCost += pos.total_cost || 0;
        totalExposure += pos.current_value || 0;
      }
    }
    return { marketsWithPos, totalUnrealized, totalCost, totalExposure };
  }, [marketList, positionsByTicker, hasPositions]);

  return (
    <div className="space-y-3">
      {/* Mentions Strategy */}
      {mentionsData && <MentionsStrategySection mentions={mentionsData} markets={rawMarkets} />}

      {/* Prob Sums & Signals */}
      <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/15">
        <div className="text-[9px] font-semibold text-cyan-400/70 uppercase tracking-wider mb-2">Probability Sums & Signals</div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-gray-300">
          <div className="flex justify-between"><span className="text-gray-500">Sum YES Bid</span><span className="font-mono tabular-nums">{sum_yes_bid != null ? `${sum_yes_bid}c` : '--'}</span></div>
          <div className="flex justify-between"><span className="text-gray-500">Sum YES Ask</span><span className="font-mono tabular-nums">{sum_yes_ask != null ? `${sum_yes_ask}c` : '--'}</span></div>
          <div className="flex justify-between"><span className="text-gray-500">Sum YES Mid</span><span className="font-mono font-semibold text-cyan-400 tabular-nums">{sum_yes_mid != null ? `${sum_yes_mid.toFixed(1)}c` : '--'}</span></div>
          {signals?.deviation != null && (
            <div className="flex justify-between"><span className="text-gray-500">Deviation</span><span className={`font-mono tabular-nums ${signals.deviation > 5 ? 'text-amber-400' : 'text-gray-400'}`}>{signals.deviation.toFixed(1)}c</span></div>
          )}
          <div className="flex justify-between items-center">
            <span className="text-gray-500">Long Edge</span>
            {isIndependent ? <span className="text-[10px] text-amber-500/60">n/a</span> : long_edge != null ? <EdgeBadge edgeCents={long_edge} direction="long" /> : <span className="font-mono text-gray-600">--</span>}
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-500">Short Edge</span>
            {isIndependent ? <span className="text-[10px] text-amber-500/60">n/a</span> : short_edge != null ? <EdgeBadge edgeCents={short_edge} direction="short" /> : <span className="font-mono text-gray-600">--</span>}
          </div>
          <div className="flex justify-between"><span className="text-gray-500">Coverage</span><span className="font-mono tabular-nums"><span className={all_markets_have_data ? 'text-emerald-400' : 'text-amber-400'}>{markets_with_data}</span><span className="text-gray-600">/{market_count}</span></span></div>
          {signals?.widest_spread_ticker && <div className="flex justify-between"><span className="text-gray-500">Widest Spread</span><span className="font-mono text-amber-400/60 text-[10px]">{signals.widest_spread_ticker}</span></div>}
          {signals?.most_active_ticker && <div className="flex justify-between"><span className="text-gray-500">Most Active</span><span className="font-mono text-cyan-400/60 text-[10px]">{signals.most_active_ticker}</span></div>}
          {series_ticker && <div className="flex justify-between"><span className="text-gray-500">Series</span><span className="font-mono">{series_ticker}</span></div>}
        </div>
      </div>

      {/* Markets Table */}
      {marketList.length > 0 && (
        <div className="bg-gray-800/15 rounded-lg p-3 border border-gray-800/20">
          <div className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Markets ({marketList.length})
            <span className="text-gray-600 font-normal ml-2">click for orderbook</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="text-[8px] text-gray-600 uppercase tracking-wider font-semibold">
                  <th className="text-left pb-1.5 pr-3 font-semibold">Outcome</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Bid</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Ask</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Mid</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Vol</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Age</th>
                  <th className="text-right pb-1.5 pl-4 px-2 font-semibold whitespace-nowrap border-l border-gray-700/10">Cost</th>
                  <th className="text-right pb-1.5 px-2 font-semibold whitespace-nowrap">Expo</th>
                  <th className="text-right pb-1.5 pl-2 font-semibold whitespace-nowrap">P&L</th>
                </tr>
              </thead>
              <tbody>
                {marketList.map((m) => {
                  const pos = positionsByTicker[m.ticker];
                  const freshness = getFreshnessDisplay(m.freshness_seconds);
                  return (
                    <tr
                      key={m.ticker}
                      className={`text-[11px] cursor-pointer transition-colors ${
                        selectedMarket === m.ticker ? 'bg-cyan-900/15' : 'hover:bg-gray-800/20'
                      }`}
                      onClick={() => onSelectMarket(selectedMarket === m.ticker ? null : m.ticker)}
                    >
                      <td className="py-1 pr-3">
                        <div className="text-gray-400 truncate max-w-[320px]">{m.title || m.ticker}</div>
                        <div className="text-[9px] text-gray-600 font-mono">{m.ticker}</div>
                      </td>
                      <td className="text-right py-1 px-2 font-mono text-gray-500 tabular-nums whitespace-nowrap">{m.yes_bid != null ? `${m.yes_bid}` : '--'}</td>
                      <td className="text-right py-1 px-2 font-mono text-gray-500 tabular-nums whitespace-nowrap">{m.yes_ask != null ? `${m.yes_ask}` : '--'}</td>
                      <td className="text-right py-1 px-2 font-mono text-cyan-400/60 tabular-nums whitespace-nowrap">{m.yes_mid != null ? `${m.yes_mid.toFixed(0)}` : '--'}</td>
                      <td className="text-right py-1 px-2 font-mono text-gray-600 text-[10px] tabular-nums whitespace-nowrap">{formatVol(m.volume_24h)}</td>
                      <td className={`text-right py-1 px-2 font-mono text-[10px] tabular-nums whitespace-nowrap ${freshness.color}`}>{freshness.text}</td>
                      <td className="text-right py-1 pl-4 px-2 font-mono text-gray-400 tabular-nums whitespace-nowrap border-l border-gray-700/10">{pos?.total_cost != null ? `$${(pos.total_cost / 100).toFixed(2)}` : '--'}</td>
                      <td className="text-right py-1 px-2 font-mono text-gray-500 tabular-nums whitespace-nowrap">{pos?.current_value != null ? `$${(pos.current_value / 100).toFixed(2)}` : '--'}</td>
                      <td className={`text-right py-1 pl-2 font-mono tabular-nums whitespace-nowrap ${
                        pos?.unrealized_pnl > 0 ? 'text-emerald-400' : pos?.unrealized_pnl < 0 ? 'text-red-400' : 'text-gray-600'
                      }`}>
                        {pos?.unrealized_pnl != null ? `${pos.unrealized_pnl >= 0 ? '+' : ''}$${(pos.unrealized_pnl / 100).toFixed(2)}` : '--'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr className="text-[11px] font-semibold border-t border-gray-700/20">
                  <td className="text-gray-400 pt-1.5 pr-3">TOTAL</td>
                  <td className="text-right pt-1.5 px-2 font-mono text-gray-400 tabular-nums">{sum_yes_bid != null ? `${sum_yes_bid}` : '--'}</td>
                  <td className="text-right pt-1.5 px-2 font-mono text-gray-400 tabular-nums">{sum_yes_ask != null ? `${sum_yes_ask}` : '--'}</td>
                  <td className="text-right pt-1.5 px-2 font-mono text-cyan-400 tabular-nums">{sum_yes_mid != null ? `${sum_yes_mid.toFixed(0)}` : '--'}</td>
                  <td className="pt-1.5 px-2" />
                  <td className="text-right pt-1.5 px-2 font-mono text-[10px] text-gray-600">/100</td>
                  <td className="text-right pt-1.5 pl-4 px-2 font-mono text-gray-400 tabular-nums border-l border-gray-700/10">{positionSummary ? `$${(positionSummary.totalCost / 100).toFixed(2)}` : ''}</td>
                  <td className="text-right pt-1.5 px-2 font-mono text-gray-400 tabular-nums">{positionSummary ? `$${(positionSummary.totalExposure / 100).toFixed(2)}` : ''}</td>
                  <td className={`text-right pt-1.5 pl-2 font-mono tabular-nums ${positionSummary?.totalUnrealized >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{positionSummary ? `${positionSummary.totalUnrealized >= 0 ? '+' : ''}$${(positionSummary.totalUnrealized / 100).toFixed(2)}` : ''}</td>
                </tr>
              </tfoot>
            </table>
            {positionSummary && (
              <div className="flex items-center justify-between text-[10px] border-t border-gray-700/20 pt-1.5 mt-0.5">
                <span className="text-gray-500">Positions: {positionSummary.marketsWithPos}/{marketList.length} markets</span>
                <span className={`font-mono font-semibold tabular-nums ${positionSummary.totalUnrealized >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {positionSummary.totalUnrealized >= 0 ? '+' : ''}${(positionSummary.totalUnrealized / 100).toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

MarketsTab.displayName = 'MarketsTab';

export default MarketsTab;
