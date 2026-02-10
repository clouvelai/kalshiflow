import React, { memo, useState, useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus, Newspaper, ChevronDown, ChevronRight } from 'lucide-react';

const TIER_STYLES = {
  high_prob: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', label: 'HIGH' },
  mid_range: { bg: 'bg-amber-500/10', text: 'text-amber-400', label: 'MID' },
  low_prob: { bg: 'bg-gray-500/10', text: 'text-gray-400', label: 'LOW' },
  unknown: { bg: 'bg-gray-500/5', text: 'text-gray-600', label: '?' },
};

const DirectionIcon = ({ direction }) => {
  if (direction === 'up') return <TrendingUp className="w-3 h-3 text-emerald-400" />;
  if (direction === 'down') return <TrendingDown className="w-3 h-3 text-red-400" />;
  return <Minus className="w-3 h-3 text-gray-600" />;
};

const TierBadge = ({ tier }) => {
  const style = TIER_STYLES[tier] || TIER_STYLES.unknown;
  return (
    <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
};

const TermRow = memo(({ term, isExpanded, onToggle }) => {
  const velocity = term.price_velocity_1h || 0;
  const isMoving = Math.abs(velocity) >= 3;
  const rowBg = isMoving
    ? velocity > 0 ? 'bg-emerald-950/10' : 'bg-red-950/10'
    : '';

  return (
    <>
      <tr
        className={`text-[11px] cursor-pointer transition-colors hover:bg-gray-800/20 ${rowBg}`}
        onClick={onToggle}
      >
        <td className="py-1.5 pr-3">
          <div className="flex items-center gap-2">
            {isExpanded ? <ChevronDown className="w-3 h-3 text-gray-600" /> : <ChevronRight className="w-3 h-3 text-gray-600" />}
            <span className="font-medium text-gray-300 truncate max-w-[180px]" title={term.entity}>
              {term.entity}
            </span>
            {term.news_headline && <Newspaper className="w-3 h-3 text-violet-400" title={term.news_headline} />}
          </div>
        </td>
        <td className="text-right py-1.5 px-2 font-mono text-cyan-400/80 tabular-nums">
          {term.yes_mid != null ? `${Math.round(term.market_prob * 100)}c` : '--'}
        </td>
        <td className="text-right py-1.5 px-2">
          <div className="flex items-center justify-end gap-1">
            <DirectionIcon direction={term.price_direction} />
            <span className={`font-mono tabular-nums ${
              Math.abs(velocity) >= 3 ? (velocity > 0 ? 'text-emerald-400' : 'text-red-400') : 'text-gray-500'
            }`}>
              {velocity > 0 ? '+' : ''}{velocity.toFixed(1)}
            </span>
          </div>
        </td>
        <td className="text-right py-1.5 px-2 font-mono text-gray-500 tabular-nums">{term.spread || '--'}</td>
        <td className="text-right py-1.5 px-2 font-mono text-gray-600 text-[10px] tabular-nums">
          {term.volume_24h >= 1000 ? `${(term.volume_24h / 1000).toFixed(0)}k` : term.volume_24h || '--'}
        </td>
        <td className="text-center py-1.5 px-2"><TierBadge tier={term.tier} /></td>
      </tr>
      {isExpanded && (
        <tr className="bg-gray-900/30">
          <td colSpan={6} className="px-8 py-2 text-[10px] text-gray-500">
            <div className="space-y-1">
              <div className="font-mono text-gray-600">{term.market_ticker}</div>
              {term.accepted_forms?.length > 0 && (
                <div className="flex items-center gap-1 flex-wrap">
                  <span className="text-gray-600">Accepted:</span>
                  {term.accepted_forms.slice(0, 6).map((f, i) => (
                    <span key={i} className="bg-emerald-500/10 text-emerald-400/70 border border-emerald-500/15 px-1 py-0.5 rounded font-mono text-[9px]">{f}</span>
                  ))}
                </div>
              )}
              {term.prohibited_forms?.length > 0 && (
                <div className="flex items-center gap-1 flex-wrap">
                  <span className="text-gray-600">Prohibited:</span>
                  {term.prohibited_forms.slice(0, 4).map((f, i) => (
                    <span key={i} className="bg-red-500/10 text-red-400/60 border border-red-500/15 px-1 py-0.5 rounded font-mono line-through text-[9px]">{f}</span>
                  ))}
                </div>
              )}
              {term.news_headline && (
                <div className="flex items-center gap-1.5 text-violet-400/80">
                  <Newspaper className="w-3 h-3" />
                  <span>{term.news_headline}</span>
                </div>
              )}
              {term.rules_excerpt && (
                <div className="text-gray-600 italic truncate max-w-[500px]">{term.rules_excerpt}</div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
});
TermRow.displayName = 'TermRow';

const MentionsTab = memo(({ event }) => {
  const [sortKey, setSortKey] = useState('velocity');
  const [expandedTerm, setExpandedTerm] = useState(null);

  // Build term data from event snapshot data
  const terms = useMemo(() => {
    const allTerms = event?.mentions_all_terms || [];
    const termPrices = event?.mentions_term_prices || {};

    return allTerms.map(entity => {
      const price = termPrices[entity] || {};
      return {
        entity,
        market_ticker: price.ticker || '',
        market_prob: price.yes_mid != null ? price.yes_mid / 100 : 0,
        yes_bid: price.yes_bid,
        yes_ask: price.yes_ask,
        yes_mid: price.yes_mid,
        spread: price.spread || 0,
        volume_24h: price.volume_24h || 0,
        price_velocity_1h: 0,
        price_direction: 'stable',
        tier: price.yes_mid > 80 ? 'high_prob' : price.yes_mid < 30 ? 'low_prob' : 'mid_range',
        is_mover: false,
        accepted_forms: [],
        prohibited_forms: [],
        news_headline: null,
        rules_excerpt: '',
      };
    });
  }, [event]);

  const movers = useMemo(() => [], []);
  const tiers = useMemo(() => ({}), []);

  const sortedTerms = useMemo(() => {
    const copy = [...terms];
    switch (sortKey) {
      case 'velocity': copy.sort((a, b) => Math.abs(b.price_velocity_1h || 0) - Math.abs(a.price_velocity_1h || 0)); break;
      case 'price': copy.sort((a, b) => (b.market_prob || 0) - (a.market_prob || 0)); break;
      case 'spread': copy.sort((a, b) => (a.spread || 999) - (b.spread || 999)); break;
      case 'volume': copy.sort((a, b) => (b.volume_24h || 0) - (a.volume_24h || 0)); break;
      case 'entity': copy.sort((a, b) => (a.entity || '').localeCompare(b.entity || '')); break;
      default: break;
    }
    return copy;
  }, [terms, sortKey]);

  // Tier breakdown
  const tierBreakdown = useMemo(() => {
    const counts = tiers.high_prob != null ? tiers : { high_prob: 0, mid_range: 0, low_prob: 0 };
    if (tiers.high_prob == null) {
      for (const t of terms) {
        const tier = t.tier || 'unknown';
        counts[tier] = (counts[tier] || 0) + 1;
      }
    }
    return counts;
  }, [terms, tiers]);

  // Spread analysis
  const spreadAnalysis = useMemo(() => {
    const spreads = terms.filter(t => t.spread > 0).map(t => t.spread);
    if (!spreads.length) return null;
    return {
      avg: (spreads.reduce((a, b) => a + b, 0) / spreads.length).toFixed(1),
      min: Math.min(...spreads),
      max: Math.max(...spreads),
    };
  }, [terms]);

  if (!terms.length) {
    return <div className="text-[11px] text-gray-600 italic">No mentions data available</div>;
  }

  const SortHeader = ({ label, field, className = '' }) => (
    <th
      className={`text-right pb-1.5 px-2 font-semibold whitespace-nowrap cursor-pointer hover:text-gray-300 transition-colors ${
        sortKey === field ? 'text-cyan-400' : ''
      } ${className}`}
      onClick={() => setSortKey(field)}
    >
      {label} {sortKey === field && '↓'}
    </th>
  );

  return (
    <div className="space-y-3">
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-2">
        {/* Movers card */}
        <div className="bg-gray-800/20 rounded-lg p-2.5 border border-gray-700/15">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold mb-1.5">Movers</div>
          {movers.length > 0 ? (
            <div className="space-y-1">
              {movers.slice(0, 3).map((m, i) => (
                <div key={i} className="flex items-center justify-between text-[10px]">
                  <span className="text-gray-300 truncate max-w-[80px]">{m.entity}</span>
                  <span className={`font-mono ${m.price_velocity_1h > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {m.price_velocity_1h > 0 ? '+' : ''}{m.price_velocity_1h?.toFixed(1)}c/h
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[10px] text-gray-600 italic">No significant movers</div>
          )}
        </div>

        {/* Tier breakdown card */}
        <div className="bg-gray-800/20 rounded-lg p-2.5 border border-gray-700/15">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold mb-1.5">Tiers</div>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px]">
              <TierBadge tier="high_prob" />
              <span className="text-gray-300 font-mono">{tierBreakdown.high_prob || 0}</span>
            </div>
            <div className="flex items-center justify-between text-[10px]">
              <TierBadge tier="mid_range" />
              <span className="text-gray-300 font-mono">{tierBreakdown.mid_range || 0}</span>
            </div>
            <div className="flex items-center justify-between text-[10px]">
              <TierBadge tier="low_prob" />
              <span className="text-gray-300 font-mono">{tierBreakdown.low_prob || 0}</span>
            </div>
          </div>
        </div>

        {/* Spread analysis card */}
        <div className="bg-gray-800/20 rounded-lg p-2.5 border border-gray-700/15">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold mb-1.5">Spreads</div>
          {spreadAnalysis ? (
            <div className="space-y-1 text-[10px]">
              <div className="flex justify-between"><span className="text-gray-500">Avg</span><span className="text-gray-300 font-mono">{spreadAnalysis.avg}c</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Tightest</span><span className="text-emerald-400/70 font-mono">{spreadAnalysis.min}c</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Widest</span><span className="text-amber-400/70 font-mono">{spreadAnalysis.max}c</span></div>
            </div>
          ) : (
            <div className="text-[10px] text-gray-600 italic">No spread data</div>
          )}
        </div>
      </div>

      {/* Main terms table */}
      <div className="bg-gray-800/15 rounded-lg p-3 border border-gray-800/20">
        <div className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
          All Terms ({terms.length})
          <span className="text-gray-600 font-normal ml-2">click row to expand</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-[8px] text-gray-600 uppercase tracking-wider font-semibold">
                <th className="text-left pb-1.5 pr-3 font-semibold cursor-pointer hover:text-gray-300"
                    onClick={() => setSortKey('entity')}>
                  Term {sortKey === 'entity' && '↓'}
                </th>
                <SortHeader label="Price" field="price" />
                <SortHeader label="1h Chg" field="velocity" />
                <SortHeader label="Spread" field="spread" />
                <SortHeader label="Vol" field="volume" />
                <th className="text-center pb-1.5 px-2 font-semibold">Tier</th>
              </tr>
            </thead>
            <tbody>
              {sortedTerms.map(term => (
                <TermRow
                  key={term.entity || term.market_ticker}
                  term={term}
                  isExpanded={expandedTerm === term.entity}
                  onToggle={() => setExpandedTerm(expandedTerm === term.entity ? null : term.entity)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
});

MentionsTab.displayName = 'MentionsTab';

export default MentionsTab;
