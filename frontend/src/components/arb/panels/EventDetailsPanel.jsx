import React, { memo, useMemo, useState } from 'react';
import { List, BarChart2, ExternalLink, Sparkles, TrendingUp, TrendingDown, Minus, Brain, Loader2 } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';
import EventTradeFeed from '../ui/EventTradeFeed';
import MarketOrderbook from '../ui/MarketOrderbook';
import EventUnderstandingCard from './EventUnderstandingCard';
import HeatStripMatrix from '../ui/MarketCandlestickChart';

/**
 * TabButton - Orderbook/trades tab toggle.
 */
const TabButton = memo(({ active, onClick, icon: Icon, label }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${
      active
        ? 'bg-cyan-500/12 text-cyan-400 border border-cyan-500/20'
        : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
    }`}
  >
    <Icon className="w-3.5 h-3.5" />
    {label}
  </button>
));
TabButton.displayName = 'TabButton';

/**
 * SingleArbDetailsView - Full event detail with markets, orderbook, trades.
 */
const SingleArbDetailsView = memo(({ event, eventTrades = [], arbTrades = [], positionsByTicker = {}, mentionsData = null }) => {
  const [selectedMarket, setSelectedMarket] = useState(null);
  const [activeTab, setActiveTab] = useState('orderbook');

  const {
    event_ticker, title, category, series_ticker,
    mutually_exclusive, event_type,
    market_count, markets_with_data, all_markets_have_data,
    sum_yes_bid, sum_yes_ask, sum_yes_mid,
    long_edge, short_edge, signals,
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

  const hasPositions = useMemo(() =>
    marketList.some(m => positionsByTicker[m.ticker]),
    [marketList, positionsByTicker]
  );

  const positionSummary = useMemo(() => {
    if (!hasPositions) return null;
    let marketsWithPos = 0;
    let totalUnrealized = 0;
    let totalCost = 0;
    let totalExposure = 0;
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

  const getFreshnessDisplay = (seconds) => {
    if (seconds == null || seconds >= 9999) return { text: '--', color: 'text-gray-600' };
    if (seconds < 5) return { text: `${Math.round(seconds)}s`, color: 'text-emerald-500' };
    if (seconds < 30) return { text: `${Math.round(seconds)}s`, color: 'text-amber-500' };
    if (seconds < 60) return { text: `${Math.round(seconds)}s`, color: 'text-red-400' };
    return { text: `${Math.floor(seconds / 60)}m`, color: 'text-red-400' };
  };

  const mergedTrades = useMemo(() => {
    const eventMarkets = new Set(Object.keys(markets));
    const publicTrades = (eventTrades || [])
      .filter(t => t.event_ticker === event_ticker || eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'public' }));
    const captainTrades = (arbTrades || [])
      .filter(t => eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'captain' }));
    return [...publicTrades, ...captainTrades].sort((a, b) => (b.ts || 0) - (a.ts || 0));
  }, [event_ticker, markets, eventTrades, arbTrades]);

  const selectedMarketData = selectedMarket ? markets[selectedMarket] : null;

  return (
    <>
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-800/30 shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono text-gray-500">{event_ticker}</span>
          {category && (
            <span className="text-[9px] font-mono bg-gray-800/40 text-gray-400 rounded-full px-2 py-0.5">
              {category}
            </span>
          )}
          <span className={`text-[9px] font-mono rounded-full px-2 py-0.5 ${
            mutually_exclusive
              ? 'bg-emerald-500/10 text-emerald-400/70'
              : 'bg-amber-500/10 text-amber-400/70'
          }`}>
            {mutually_exclusive ? 'mut. excl.' : 'independent'}
          </span>
        </div>
        <div className="flex items-center gap-2.5">
          {event.image_url && (
            <img
              src={event.image_url}
              alt=""
              className="w-7 h-7 rounded-md object-cover opacity-80 flex-shrink-0"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          )}
          <h3 className="text-sm font-medium text-gray-200 leading-tight flex-1 min-w-0">
            {title || event_ticker}
          </h3>
          {(event.kalshi_url || event_ticker) && (
            <a
              href={event.kalshi_url || `https://kalshi.com/markets/${event_ticker.toLowerCase()}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-cyan-400 transition-colors flex-shrink-0"
              title="View on Kalshi"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-3">
        {/* Understanding - only show if it has meaningful content */}
        {event.understanding && (
          event.understanding.trading_summary || event.understanding.event_summary ||
          (event.understanding.participants && event.understanding.participants.length > 0) ||
          (event.understanding.key_factors && event.understanding.key_factors.length > 0)
        ) && (
          <EventUnderstandingCard
            understanding={event.understanding}
            lifecycle={event.lifecycle || null}
            causalModel={event.causal_model || null}
            markets={rawMarkets || null}
          />
        )}

        {/* Mentions Strategy - only for mentions events */}
        {mentionsData && (
          <MentionsStrategySection mentions={mentionsData} markets={rawMarkets} />
        )}

        {/* Prob Sums & Signals */}
        <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/15">
          <div className="text-[9px] font-semibold text-cyan-400/70 uppercase tracking-wider mb-2">
            Probability Sums &amp; Signals
          </div>
          <div className="space-y-1.5 text-[11px] text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Bid</span>
              <span className="font-mono tabular-nums">{sum_yes_bid != null ? `${sum_yes_bid}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Ask</span>
              <span className="font-mono tabular-nums">{sum_yes_ask != null ? `${sum_yes_ask}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Mid</span>
              <span className="font-mono font-semibold text-cyan-400 tabular-nums">
                {sum_yes_mid != null ? `${sum_yes_mid.toFixed(1)}c` : '--'}
              </span>
            </div>
            {signals?.deviation != null && (
              <div className="flex justify-between">
                <span className="text-gray-500">Deviation</span>
                <span className={`font-mono tabular-nums ${signals.deviation > 5 ? 'text-amber-400' : 'text-gray-400'}`}>
                  {signals.deviation.toFixed(1)}c
                </span>
              </div>
            )}
            {isIndependent ? (
              <div className="pt-1.5 border-t border-gray-700/15">
                <span className="text-[10px] text-amber-500/60">No spread arb - independent outcomes</span>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-center pt-1.5 border-t border-gray-700/15">
                  <span className="text-gray-500">Long Edge</span>
                  {long_edge != null ? <EdgeBadge edgeCents={long_edge} direction="long" /> : <span className="font-mono text-gray-600">--</span>}
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Short Edge</span>
                  {short_edge != null ? <EdgeBadge edgeCents={short_edge} direction="short" /> : <span className="font-mono text-gray-600">--</span>}
                </div>
              </>
            )}
            <div className="flex justify-between items-center pt-1.5 border-t border-gray-700/15">
              <span className="text-gray-500">Data Coverage</span>
              <span className="font-mono tabular-nums">
                <span className={all_markets_have_data ? 'text-emerald-400' : 'text-amber-400'}>{markets_with_data}</span>
                <span className="text-gray-600">/{market_count}</span>
                {all_markets_have_data && <span className="text-emerald-500/70 ml-1">&#10003;</span>}
              </span>
            </div>
            {signals?.widest_spread_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Widest Spread</span>
                <span className="font-mono text-amber-400/60 text-[10px]">{signals.widest_spread_ticker}</span>
              </div>
            )}
            {signals?.most_active_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Most Active</span>
                <span className="font-mono text-cyan-400/60 text-[10px]">{signals.most_active_ticker}</span>
              </div>
            )}
            {series_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Series</span>
                <span className="font-mono">{series_ticker}</span>
              </div>
            )}
          </div>
        </div>

        {/* Price History Heatmap */}
        {event.candlestick_series && Object.keys(event.candlestick_series).length > 0 && (
          <HeatStripMatrix candlestickSeries={event.candlestick_series} markets={markets} />
        )}

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
                        onClick={() => setSelectedMarket(selectedMarket === m.ticker ? null : m.ticker)}
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
                  <span className="text-gray-500">
                    Positions: {positionSummary.marketsWithPos}/{marketList.length} markets
                  </span>
                  <span className={`font-mono font-semibold tabular-nums ${positionSummary.totalUnrealized >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {positionSummary.totalUnrealized >= 0 ? '+' : ''}${(positionSummary.totalUnrealized / 100).toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Orderbook / Trades Tabs */}
        <div className="bg-gray-800/15 rounded-lg border border-gray-800/20 overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800/20 bg-gray-900/20">
            <TabButton active={activeTab === 'orderbook'} onClick={() => setActiveTab('orderbook')} icon={BarChart2} label="Orderbook" />
            <TabButton active={activeTab === 'trades'} onClick={() => setActiveTab('trades')} icon={List} label={`Trades (${mergedTrades.length})`} />
            {selectedMarket && (
              <span className="ml-auto text-[10px] text-gray-500 font-mono">{selectedMarket}</span>
            )}
          </div>
          <div className="p-3">
            {activeTab === 'orderbook' ? (
              selectedMarketData ? (
                <MarketOrderbook market={selectedMarketData} />
              ) : (
                <div className="text-center py-6 text-gray-600">
                  <BarChart2 className="w-5 h-5 mx-auto mb-1.5 opacity-40" />
                  <p className="text-[11px]">Select a market to view orderbook</p>
                </div>
              )
            ) : (
              <EventTradeFeed trades={mergedTrades} arbTrades={arbTrades} showSource={true} />
            )}
          </div>
        </div>
      </div>
    </>
  );
});

/**
 * MentionsTermRow - Single term probability row with edge calculation.
 */
const MentionsTermRow = memo(function MentionsTermRow({ term, delta, markets }) {
  const prob = term.probability || 0;

  // Try to find a matching market for this term
  const termPrefix = term.term?.toLowerCase().slice(0, 4) || '';
  const market = markets?.find(m =>
    m.ticker?.toLowerCase().includes(termPrefix)
  );

  const marketYes = market?.yes_bid || market?.yes_mid || 50;
  const simProbCents = Math.round(prob * 100);
  const edge = simProbCents - marketYes;

  const trend = delta?.trend || '\u2192';
  const TrendIcon = trend === '\u2191' ? TrendingUp : trend === '\u2193' ? TrendingDown : Minus;
  const trendColor = trend === '\u2191' ? 'text-emerald-400' : trend === '\u2193' ? 'text-red-400' : 'text-gray-500';

  const edgeColor = Math.abs(edge) >= 10
    ? (edge > 0 ? 'text-emerald-400' : 'text-red-400')
    : 'text-gray-400';

  const rowBg = Math.abs(edge) >= 15
    ? (edge > 0 ? 'bg-emerald-950/15' : 'bg-red-950/15')
    : '';

  return (
    <div className={`flex items-center justify-between text-[10px] py-1 border-b border-gray-700/15 last:border-0 ${rowBg}`}>
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-300 w-24 truncate" title={term.term}>
          {term.term}
        </span>
        <TrendIcon className={`w-2.5 h-2.5 ${trendColor}`} />
      </div>
      <div className="flex items-center gap-3 tabular-nums">
        <span className="text-cyan-400/90 font-mono w-12 text-right">
          {(prob * 100).toFixed(0)}%
        </span>
        <span className="text-gray-600 w-8 text-center text-[9px]">vs</span>
        <span className="text-gray-300 font-mono w-10 text-right">
          {marketYes}c
        </span>
        <span className={`font-mono font-semibold w-12 text-right ${edgeColor}`}>
          {edge > 0 ? '+' : ''}{edge}c
        </span>
      </div>
    </div>
  );
});

/**
 * MentionsStrategySection - Displays mentions simulation data within EventDetailsPanel.
 */
const MentionsStrategySection = memo(function MentionsStrategySection({ mentions, markets }) {
  if (!mentions) return null;

  const hasTerms = mentions.terms?.length > 0;
  const marketList = markets ? Object.values(markets) : [];

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 border border-amber-500/10">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-amber-400/80" />
          <span className="text-[9px] font-semibold text-amber-400/70 uppercase tracking-wider">
            Mentions Strategy
          </span>
        </div>
        <div className="flex items-center gap-2">
          {mentions.simulation_in_progress && (
            <span className="flex items-center gap-1 text-[9px] text-violet-400/80">
              <Loader2 className="w-2.5 h-2.5 animate-spin" />
              Simulating
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
            <MentionsTermRow
              key={t.term}
              term={t}
              delta={mentions.deltas?.[t.term]}
              markets={marketList}
            />
          ))}
        </div>
      ) : (
        <div className="text-[10px] text-gray-600 italic">
          No simulation data yet
        </div>
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

const EventDetailsPanel = ({ selectedEventTicker, events, eventTrades = [], arbTrades = [], tradingState, mentionsState }) => {
  const event = useMemo(() => {
    if (!selectedEventTicker || !events) return null;
    return events.get(selectedEventTicker) || null;
  }, [selectedEventTicker, events]);

  const positionsByTicker = useMemo(() => {
    const map = {};
    (tradingState?.positions || []).forEach(p => { map[p.ticker] = p; });
    return map;
  }, [tradingState?.positions]);

  if (!event) {
    return (
      <div className="bg-gradient-to-br from-gray-900/80 to-gray-950/80 rounded-2xl border border-cyan-500/8 shadow-lg shadow-cyan-500/3 flex flex-col items-center justify-center min-h-[200px]">
        <div className="animate-pulse space-y-3 w-3/4">
          <div className="h-3 bg-gray-800/60 rounded w-1/3" />
          <div className="h-5 bg-gray-800/40 rounded w-2/3" />
          <div className="h-20 bg-gray-800/20 rounded" />
        </div>
        <p className="text-[10px] text-gray-600 mt-4">Waiting for event data...</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-900/80 to-gray-950/80 rounded-2xl border border-cyan-500/8 shadow-lg shadow-cyan-500/3 overflow-hidden flex flex-col">
      <SingleArbDetailsView event={event} eventTrades={eventTrades} arbTrades={arbTrades} positionsByTicker={positionsByTicker} mentionsData={mentionsState?.[selectedEventTicker] || null} />
    </div>
  );
};

export default memo(EventDetailsPanel);
