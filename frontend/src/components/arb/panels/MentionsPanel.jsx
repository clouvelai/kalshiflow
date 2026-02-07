import { memo, useMemo } from 'react';
import { Sparkles, TrendingUp, TrendingDown, Minus, Brain, Loader2 } from 'lucide-react';

/**
 * MentionsPanel - Displays mentions market probability estimates, edge calculations,
 * and trading recommendations from the MentionsSpecialist subagent.
 */
const MentionsPanel = memo(function MentionsPanel({ mentionsState, events }) {
  // Get mentions events only (those with MENTION in ticker)
  const mentionsEvents = useMemo(() => {
    if (!events || events.size === 0) return [];
    return Array.from(events.values())
      .filter(e => e.event_ticker?.toUpperCase().includes('MENTION'))
      .map(e => ({
        ...e,
        mentions: mentionsState?.[e.event_ticker] || null
      }));
  }, [events, mentionsState]);

  if (mentionsEvents.length === 0) {
    return (
      <div className="bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
                      backdrop-blur-sm rounded-xl border border-amber-500/10
                      shadow-lg shadow-black/10 p-4">
        <div className="flex items-center gap-2 text-gray-500">
          <Sparkles className="w-3.5 h-3.5" />
          <span className="text-[11px]">No mentions markets active</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
                    backdrop-blur-sm rounded-xl border border-amber-500/10
                    shadow-lg shadow-black/10 p-4 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400/80" />
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">
            Mentions Strategy
          </span>
        </div>
        <span className="text-[10px] text-gray-500 font-mono tabular-nums">
          {mentionsEvents.length} event{mentionsEvents.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Events List */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {mentionsEvents.map(event => (
          <MentionsEventCard key={event.event_ticker} event={event} />
        ))}
      </div>
    </div>
  );
});

/**
 * MentionsEventCard - Card for a single mentions event showing terms and probabilities.
 */
const MentionsEventCard = memo(function MentionsEventCard({ event }) {
  const mentions = event.mentions;
  const hasData = mentions && mentions.terms?.length > 0;

  // Get markets from event for price matching
  const markets = event.markets ? Object.values(event.markets) : [];

  return (
    <div className="bg-gray-800/20 rounded-lg p-3 space-y-2 border border-gray-700/15">
      {/* Event Header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-mono text-amber-400/80 truncate tabular-nums" title={event.event_ticker}>
          {event.event_ticker}
        </span>
        {mentions?.simulation_in_progress && (
          <span className="flex items-center gap-1 text-[9px] text-violet-400/80">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            Simulating
          </span>
        )}
      </div>

      {/* Event Title (truncated) */}
      {event.title && (
        <div className="text-[10px] text-gray-400 truncate" title={event.title}>
          {event.title}
        </div>
      )}

      {/* Terms Table */}
      {hasData ? (
        <div className="space-y-1 mt-2">
          {/* Header Row */}
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
            <TermRow
              key={t.term}
              term={t}
              delta={mentions.deltas?.[t.term]}
              markets={markets}
            />
          ))}
        </div>
      ) : (
        <div className="text-[10px] text-gray-600 italic">
          No simulation data yet
        </div>
      )}

      {/* News Context */}
      {mentions?.news_context?.length > 0 && (
        <div className="text-[9px] text-gray-500 italic border-t border-gray-700/20 pt-2 mt-1">
          <Brain className="w-2.5 h-2.5 inline mr-1 text-violet-400/60" />
          {mentions.news_context[0]}
        </div>
      )}

      {/* History Count */}
      {mentions?.history_count > 0 && (
        <div className="text-[9px] text-gray-600 text-right font-mono tabular-nums">
          {mentions.history_count} sim{mentions.history_count !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
});

/**
 * TermRow - Single term probability row with edge calculation.
 */
const TermRow = memo(function TermRow({ term, delta, markets }) {
  const prob = term.probability || 0;

  // Try to find a matching market for this term
  // Match by first 4 chars of term (case-insensitive) in market ticker
  const termPrefix = term.term?.toLowerCase().slice(0, 4) || '';
  const market = markets?.find(m =>
    m.ticker?.toLowerCase().includes(termPrefix)
  );

  // Get market YES bid price (in cents)
  const marketYes = market?.yes_bid || market?.yes_mid || 50;

  // Calculate edge: simulation probability - market price
  const simProbCents = Math.round(prob * 100);
  const edge = simProbCents - marketYes;

  // Trend indicator from delta
  const trend = delta?.trend || '\u2192'; // → as default
  const TrendIcon = trend === '\u2191' ? TrendingUp : trend === '\u2193' ? TrendingDown : Minus;
  const trendColor = trend === '\u2191' ? 'text-emerald-400' : trend === '\u2193' ? 'text-red-400' : 'text-gray-500';

  // Edge color (green for positive, red for negative, gray for small)
  const edgeColor = Math.abs(edge) >= 10
    ? (edge > 0 ? 'text-emerald-400' : 'text-red-400')
    : 'text-gray-400';

  // Background highlight for large edges
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
        {/* Simulation Probability */}
        <span className="text-cyan-400/90 font-mono w-12 text-right">
          {(prob * 100).toFixed(0)}%
        </span>

        {/* vs Market */}
        <span className="text-gray-600 w-8 text-center text-[9px]">vs</span>
        <span className="text-gray-300 font-mono w-10 text-right">
          {marketYes}c
        </span>

        {/* Edge */}
        <span className={`font-mono font-semibold w-12 text-right ${edgeColor}`}>
          {edge > 0 ? '+' : ''}{edge}c
        </span>
      </div>
    </div>
  );
});

export default MentionsPanel;
