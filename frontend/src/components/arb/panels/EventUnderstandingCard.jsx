import React, { memo, useState, useEffect } from 'react';
import {
  Clock,
  Users,
  Lightbulb,
  AlertTriangle,
  ExternalLink,
  Zap,
  Globe,
  Tv,
  MapPin,
  Tag,
  FileText,
  TrendingUp,
  ChevronDown,
  ChevronRight,
  Activity,
  Newspaper,
  Target,
  Calendar,
} from 'lucide-react';

// =============================================================================
// LIVE TIME HOOK - Subtle, ticks every 30s to keep "time to close" fresh
// =============================================================================

const useTimeToClose = (closeTimeISO) => {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!closeTimeISO) return;
    const id = setInterval(() => setNow(Date.now()), 30000); // tick every 30s
    return () => clearInterval(id);
  }, [closeTimeISO]);

  if (!closeTimeISO) return null;

  const diff = new Date(closeTimeISO).getTime() - now;
  if (diff <= 0) return { text: 'Closed', urgency: 'settled' };

  const hours = diff / 3600000;
  if (hours >= 48) return { text: `${Math.round(hours / 24)}d`, urgency: 'relaxed' };
  if (hours >= 2) return { text: `${Math.round(hours)}h`, urgency: 'relaxed' };
  if (hours >= 1) return { text: `${hours.toFixed(1)}h`, urgency: 'warm' };
  const mins = Math.round(diff / 60000);
  if (mins > 0) return { text: `${mins}m`, urgency: mins < 10 ? 'hot' : 'warm' };
  return { text: '<1m', urgency: 'hot' };
};

// =============================================================================
// DOMAIN CONFIG
// =============================================================================

const DOMAIN_CONFIG = {
  sports: { icon: Zap, accent: 'emerald' },
  politics: { icon: Globe, accent: 'violet' },
  corporate: { icon: TrendingUp, accent: 'blue' },
  entertainment: { icon: Tv, accent: 'pink' },
  crypto: { icon: Activity, accent: 'amber' },
  generic: { icon: Globe, accent: 'cyan' },
};

const ACCENT_CLASSES = {
  emerald: { badge: 'bg-emerald-500/10 text-emerald-400/80 border-emerald-500/15', dot: 'bg-emerald-400' },
  violet: { badge: 'bg-violet-500/10 text-violet-400/80 border-violet-500/15', dot: 'bg-violet-400' },
  blue: { badge: 'bg-blue-500/10 text-blue-400/80 border-blue-500/15', dot: 'bg-blue-400' },
  pink: { badge: 'bg-pink-500/10 text-pink-400/80 border-pink-500/15', dot: 'bg-pink-400' },
  amber: { badge: 'bg-amber-500/10 text-amber-400/80 border-amber-500/15', dot: 'bg-amber-400' },
  cyan: { badge: 'bg-cyan-500/10 text-cyan-400/80 border-cyan-500/15', dot: 'bg-cyan-400' },
};

const LIFECYCLE_COLORS = {
  dormant: 'bg-gray-500/10 text-gray-400/80 border-gray-500/15',
  discovery: 'bg-blue-500/10 text-blue-400/80 border-blue-500/15',
  building: 'bg-cyan-500/10 text-cyan-400/80 border-cyan-500/15',
  peak: 'bg-emerald-500/10 text-emerald-400/80 border-emerald-500/15',
  convergence: 'bg-amber-500/10 text-amber-400/80 border-amber-500/15',
  resolution: 'bg-red-500/10 text-red-400/80 border-red-500/15',
};

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

const Pill = memo(({ children, className = '', title }) => (
  <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border ${className}`} title={title}>
    {children}
  </span>
));
Pill.displayName = 'Pill';

const SectionLabel = memo(({ icon: Icon, children }) => (
  <div className="flex items-center gap-1 text-[9px] font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
    {Icon && <Icon className="w-2.5 h-2.5" />}
    {children}
  </div>
));
SectionLabel.displayName = 'SectionLabel';

// =============================================================================
// MAIN COMPONENT
// =============================================================================

const EventUnderstandingCard = ({ understanding, compact = false, lifecycle = null, causalModel = null, markets = null }) => {
  const [showMore, setShowMore] = useState(false);

  if (!understanding) return null;

  const {
    domain,
    category,
    status,
    close_time: closeTime,
    trading_summary,
    event_summary,
    key_factors = [],
    trading_considerations = [],
    participants = [],
    key_figures = [],
    timeline = [],
    venue,
    network,
    date,
    settlement_summary,
    resolution_source,
    wikipedia_urls = [],
    extensions = {},
    stale,
    mutually_exclusive,
    news_articles = [],
    news_fetched_at,
  } = understanding;

  const ttc = useTimeToClose(closeTime);
  const displaySummary = trading_summary || event_summary;
  const dc = DOMAIN_CONFIG[domain] || DOMAIN_CONFIG.generic;
  const ac = ACCENT_CLASSES[dc.accent] || ACCENT_CLASSES.cyan;
  const DomainIcon = dc.icon;

  // ─── COMPACT ───
  if (compact) {
    const compactText = displaySummary || settlement_summary;
    return (
      <div className="space-y-1.5">
        {/* Chips row */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <Pill className={ac.badge}>
            <DomainIcon className="w-2.5 h-2.5" />
            {category || domain}
          </Pill>
          {mutually_exclusive != null && (
            <Pill className={mutually_exclusive
              ? 'bg-emerald-500/8 text-emerald-400/70 border-emerald-500/12'
              : 'bg-amber-500/8 text-amber-400/70 border-amber-500/12'
            }>
              {mutually_exclusive ? 'Mut. Excl.' : 'Independent'}
            </Pill>
          )}
          {ttc && (
            <Pill className={
              ttc.urgency === 'hot' ? 'bg-red-500/10 text-red-400/80 border-red-500/15' :
              ttc.urgency === 'warm' ? 'bg-amber-500/10 text-amber-400/80 border-amber-500/15' :
              ttc.urgency === 'settled' ? 'bg-gray-800/40 text-gray-500 border-gray-700/20' :
              'bg-gray-800/30 text-gray-400 border-gray-700/20'
            }>
              {ttc.urgency === 'hot' && (
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
              )}
              <Clock className="w-2.5 h-2.5" />
              {ttc.text}
            </Pill>
          )}
          {network && <span className="text-[10px] text-gray-500">{network}</span>}
        </div>

        {compactText && (
          <p className="text-[11px] text-gray-400 leading-relaxed line-clamp-2">
            {compactText}
          </p>
        )}
      </div>
    );
  }

  // ─── FULL ───
  const hasExtras = key_figures.length > 0 || trading_considerations.length > 0 || settlement_summary;

  return (
    <div className="rounded-lg border border-gray-700/20 overflow-hidden">
      {/* Header: chips + summary */}
      <div className="px-3.5 py-3 bg-gray-800/15">
        {/* Row 1: chips */}
        <div className="flex items-center gap-1.5 flex-wrap mb-2">
          <Pill className={ac.badge}>
            <DomainIcon className="w-3 h-3" />
            {category || domain}
          </Pill>
          {mutually_exclusive != null && (
            <Pill className={mutually_exclusive
              ? 'bg-emerald-500/8 text-emerald-400/70 border-emerald-500/12'
              : 'bg-amber-500/8 text-amber-400/70 border-amber-500/12'
            }>
              {mutually_exclusive ? 'Mut. Excl.' : 'Independent'}
            </Pill>
          )}
          {lifecycle?.stage && (
            <Pill
              className={LIFECYCLE_COLORS[lifecycle.stage] || LIFECYCLE_COLORS.dormant}
              title={lifecycle.recommended_action || ''}
            >
              {lifecycle.stage}
            </Pill>
          )}

          {/* Right: time to close */}
          <div className="ml-auto flex items-center gap-1.5">
            {ttc && ttc.urgency !== 'settled' && (
              <span className={`inline-flex items-center gap-1 text-[10px] font-mono ${
                ttc.urgency === 'hot' ? 'text-red-400' :
                ttc.urgency === 'warm' ? 'text-amber-400' :
                'text-gray-500'
              }`}>
                {ttc.urgency === 'hot' && (
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                )}
                <Clock className="w-3 h-3" />
                {ttc.text}
              </span>
            )}
            {ttc?.urgency === 'settled' && (
              <span className="text-[10px] text-gray-500 font-mono">Closed</span>
            )}
            {stale && (
              <span className="text-[9px] text-amber-500/50">stale</span>
            )}
          </div>
        </div>

        {/* Summary text */}
        {displaySummary && (
          <p className="text-[12px] text-gray-300 leading-relaxed">{displaySummary}</p>
        )}

        {/* Meta row */}
        {(network || venue || date) && (
          <div className="flex flex-wrap items-center gap-3 mt-2 text-[10px] text-gray-500">
            {network && <span className="flex items-center gap-1"><Tv className="w-3 h-3" /> {network}</span>}
            {venue && <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> {venue.length > 45 ? venue.slice(0, 45) + '...' : venue}</span>}
            {date && <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {date}</span>}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="px-3.5 py-3 space-y-3">
        {/* Timeline segments */}
        {timeline.length > 0 && (
          <div>
            <SectionLabel icon={Clock}>Timeline</SectionLabel>
            <div className="flex gap-0.5 rounded overflow-hidden">
              {timeline.map((seg, i) => {
                const totalDur = timeline.reduce((s, t) => s + (t.duration_min || 30), 0);
                const pct = Math.max(((seg.duration_min || 30) / totalDur) * 100, 10);
                const segColors = [
                  'bg-cyan-500/15 text-cyan-400/80',
                  'bg-violet-500/10 text-violet-400/70',
                  'bg-emerald-500/10 text-emerald-400/70',
                  'bg-amber-500/10 text-amber-400/70',
                ];
                return (
                  <div
                    key={i}
                    className={`h-5 flex items-center justify-center rounded-sm ${segColors[i % segColors.length]}`}
                    style={{ width: `${pct}%`, minWidth: '36px' }}
                    title={seg.description || seg.name}
                  >
                    <span className="text-[8px] font-medium truncate px-1">{seg.name}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Causal Drivers (replaces Key Factors when available) */}
        {causalModel?.drivers?.length > 0 ? (
          <div>
            <SectionLabel icon={Target}>Causal Drivers</SectionLabel>
            <div className="space-y-1.5">
              {causalModel.drivers.map((d, i) => {
                const isStale = d.status === 'stale';
                const arrow = d.direction === 'bullish' ? '\u2191' : d.direction === 'bearish' ? '\u2193' : '\u2192';
                const arrowColor = d.direction === 'bullish' ? 'text-emerald-400' : d.direction === 'bearish' ? 'text-red-400' : 'text-gray-500';
                return (
                  <div key={i} className={`flex items-start gap-2 text-[11px] ${isStale ? 'opacity-40' : ''}`}>
                    <span className={`flex-shrink-0 font-mono font-bold ${arrowColor}`}>{arrow}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-gray-300 font-medium">{d.name}</span>
                        {isStale && <span className="text-[9px] text-gray-600">stale</span>}
                      </div>
                      {d.confidence != null && (
                        <div className="w-full h-[3px] bg-gray-800/40 rounded-full mt-1 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${d.direction === 'bullish' ? 'bg-emerald-500/60' : d.direction === 'bearish' ? 'bg-red-500/60' : 'bg-gray-500/60'}`}
                            style={{ width: `${Math.round(d.confidence * 100)}%` }}
                          />
                        </div>
                      )}
                      {d.market_links?.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {d.market_links.map((ml, j) => (
                            <span key={j} className="text-[8px] font-mono bg-gray-800/40 text-gray-500 px-1 py-0.5 rounded">
                              {ml.market_ticker}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            {causalModel.dominant_narrative && (
              <p className="text-[10px] text-gray-500 mt-2 leading-relaxed italic">{causalModel.dominant_narrative}</p>
            )}
            {causalModel.consensus_direction && (
              <Pill className={
                causalModel.consensus_direction === 'bullish' ? 'bg-emerald-500/8 text-emerald-400/70 border-emerald-500/12' :
                causalModel.consensus_direction === 'bearish' ? 'bg-red-500/8 text-red-400/70 border-red-500/12' :
                'bg-gray-500/8 text-gray-400/70 border-gray-500/12'
              }>
                {causalModel.consensus_direction}
              </Pill>
            )}
          </div>
        ) : key_factors.length > 0 ? (
          <div>
            <SectionLabel icon={Lightbulb}>Key Factors</SectionLabel>
            <div className="space-y-1">
              {key_factors.map((f, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px]">
                  <span className={`w-1 h-1 rounded-full flex-shrink-0 mt-[5px] ${ac.dot}`} style={{ opacity: 0.5 }} />
                  <span className="text-gray-400 leading-relaxed">{f}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {/* Catalyst Timeline */}
        {causalModel?.catalysts?.length > 0 && (
          <div>
            <SectionLabel icon={Calendar}>Catalysts</SectionLabel>
            <div className="space-y-1.5">
              {causalModel.catalysts.map((c, i) => {
                const now = Date.now();
                const catalystTime = c.expected_date ? new Date(c.expected_date).getTime() : null;
                const isImminent = catalystTime && !c.occurred && (catalystTime - now) < 4 * 3600000 && (catalystTime - now) > 0;
                const hasOccurred = c.occurred;
                const typeColor = c.type === 'scheduled' ? 'bg-blue-500/10 text-blue-400/70 border-blue-500/15' :
                  c.type === 'expected' ? 'bg-amber-500/10 text-amber-400/70 border-amber-500/15' :
                  'bg-gray-500/10 text-gray-400/70 border-gray-500/15';
                return (
                  <div key={i} className={`flex items-center gap-2 text-[11px] ${hasOccurred ? 'opacity-40' : ''}`}>
                    {isImminent && <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse flex-shrink-0" />}
                    {hasOccurred && <span className="text-emerald-500 flex-shrink-0 text-[10px]">&#10003;</span>}
                    {!isImminent && !hasOccurred && <span className="w-1.5 h-1.5 flex-shrink-0" />}
                    <span className="text-gray-300 flex-1 min-w-0 truncate">{c.name}</span>
                    {c.expected_date && (
                      <span className="text-[9px] text-gray-600 font-mono flex-shrink-0">
                        {new Date(c.expected_date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                      </span>
                    )}
                    {c.type && <Pill className={typeColor}>{c.type}</Pill>}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Microstructure Indicators */}
        {markets && (() => {
          const marketEntries = Object.values(markets).filter(m => m.micro &&
            (m.micro.whale_trade_count > 0 || m.micro.rapid_sequence_count > 0 ||
             Math.abs(m.micro.book_imbalance || 0) > 0.3 || m.micro.buy_sell_ratio != null));
          if (marketEntries.length === 0) return null;
          return (
            <div>
              <SectionLabel icon={Activity}>Microstructure</SectionLabel>
              <div className="space-y-1">
                {marketEntries.map((m, i) => {
                  const micro = m.micro;
                  const bsr = micro.buy_sell_ratio;
                  const buyPct = bsr != null ? Math.round(bsr * 100) : null;
                  return (
                    <div key={i} className="flex items-center gap-2 text-[10px]">
                      <span className="text-gray-400 truncate min-w-0 flex-1 max-w-[140px]" title={m.title || m.ticker}>
                        {(m.title || m.ticker || '').slice(0, 24)}
                      </span>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {micro.whale_trade_count > 0 && (
                          <span className="text-orange-400 font-mono text-[9px]" title="Whale trades">
                            {'\uD83D\uDC0B'} {micro.whale_trade_count}
                          </span>
                        )}
                        {micro.rapid_sequence_count > 0 && (
                          <span className="text-yellow-400 font-mono text-[9px]" title="Rapid sequences">
                            {'\u26A1'} {micro.rapid_sequence_count}
                          </span>
                        )}
                        {micro.book_imbalance != null && Math.abs(micro.book_imbalance) > 0.3 && (
                          <span className={`font-mono text-[9px] ${micro.book_imbalance > 0 ? 'text-emerald-400' : 'text-red-400'}`} title={`Book imbalance: ${micro.book_imbalance.toFixed(2)}`}>
                            {micro.book_imbalance > 0 ? '\u2191' : '\u2193'}
                          </span>
                        )}
                        {buyPct != null && (
                          <div className="flex items-center gap-0.5" title={`Buy: ${buyPct}% / Sell: ${100 - buyPct}%`}>
                            <div className="w-12 h-[4px] bg-gray-800/40 rounded-full overflow-hidden flex">
                              <div className="h-full bg-emerald-500/60 rounded-l-full" style={{ width: `${buyPct}%` }} />
                              <div className="h-full bg-red-500/60 rounded-r-full" style={{ width: `${100 - buyPct}%` }} />
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* Mentions extension */}
        {extensions.mentions && (
          <div className="bg-violet-500/5 rounded-lg px-3 py-2 border border-violet-500/10 space-y-2">
            <SectionLabel icon={Tag}>Mentions</SectionLabel>
            {/* Row 1: Core metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-[10px]">
              <div>
                <span className="text-gray-500">Entity</span>
                <div className="text-gray-200 font-mono">{extensions.mentions.entity}</div>
              </div>
              {extensions.mentions.speaker && (
                <div>
                  <span className="text-gray-500">Speaker</span>
                  <div className="text-gray-200">{extensions.mentions.speaker}</div>
                </div>
              )}
              {extensions.mentions.baseline_probability != null && (
                <div>
                  <span className="text-gray-500">Baseline P</span>
                  <div className="text-cyan-400 font-mono">{(extensions.mentions.baseline_probability * 100).toFixed(1)}%</div>
                </div>
              )}
              {extensions.mentions.current_probability != null && extensions.mentions.current_probability !== extensions.mentions.baseline_probability && (
                <div>
                  <span className="text-gray-500">Informed P</span>
                  <div className="text-violet-400 font-mono">{(extensions.mentions.current_probability * 100).toFixed(1)}%</div>
                </div>
              )}
              <div>
                <span className="text-gray-500">Count</span>
                <div className="text-gray-200 font-mono">{extensions.mentions.current_count ?? 0}</div>
              </div>
            </div>
            {/* Row 2: CI + Simulation info */}
            <div className="flex items-center gap-3 flex-wrap text-[10px]">
              {extensions.mentions.ci_lower != null && extensions.mentions.ci_upper != null && (
                <span className="text-gray-500">
                  CI: <span className="text-gray-300 font-mono">
                    {(extensions.mentions.ci_lower * 100).toFixed(0)}%-{(extensions.mentions.ci_upper * 100).toFixed(0)}%
                  </span>
                </span>
              )}
              {extensions.mentions.simulation_count > 0 && (
                <span className="text-gray-500">
                  Sims: <span className="text-gray-300 font-mono">{extensions.mentions.simulation_count}</span>
                </span>
              )}
            </div>
            {/* Row 3: Accepted/Prohibited forms */}
            {((extensions.mentions.accepted_forms?.length > 0) || (extensions.mentions.prohibited_forms?.length > 0)) && (
              <div className="flex items-center gap-2 flex-wrap">
                {extensions.mentions.accepted_forms?.slice(0, 5).map((form, i) => (
                  <span key={`a-${i}`} className="text-[9px] bg-emerald-500/10 text-emerald-400/70 border border-emerald-500/15 px-1.5 py-0.5 rounded font-mono">
                    {form}
                  </span>
                ))}
                {extensions.mentions.prohibited_forms?.slice(0, 3).map((form, i) => (
                  <span key={`p-${i}`} className="text-[9px] bg-red-500/10 text-red-400/60 border border-red-500/15 px-1.5 py-0.5 rounded font-mono line-through">
                    {form}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Recent News */}
        {news_articles.length > 0 && (
          <div>
            <SectionLabel icon={Newspaper}>Recent News</SectionLabel>
            {news_fetched_at && (() => {
              const ageHours = (Date.now() / 1000 - news_fetched_at) / 3600;
              if (ageHours > 4) {
                return (
                  <span className="text-[9px] text-amber-500/60 mb-1 block">
                    News {Math.round(ageHours)}h old - may be stale
                  </span>
                );
              }
              return null;
            })()}
            <div className="space-y-1.5">
              {news_articles.slice(0, 5).map((article, i) => {
                const publishedAgo = article.published_date
                  ? (() => {
                      const diff = Date.now() - new Date(article.published_date).getTime();
                      const hours = diff / 3600000;
                      if (hours < 1) return `${Math.round(diff / 60000)}m ago`;
                      if (hours < 24) return `${Math.round(hours)}h ago`;
                      return `${Math.round(hours / 24)}d ago`;
                    })()
                  : null;
                const snippet = (article.content || article.snippet || '').slice(0, 120);
                return (
                  <div key={i} className="bg-gray-800/20 rounded px-2.5 py-1.5 border border-gray-700/10">
                    <div className="flex items-start gap-2">
                      <div className="flex-1 min-w-0">
                        {article.url ? (
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[11px] text-gray-200 hover:text-cyan-400 transition-colors leading-tight line-clamp-2"
                          >
                            {article.title}
                            <ExternalLink className="w-2.5 h-2.5 inline ml-1 opacity-40" />
                          </a>
                        ) : (
                          <span className="text-[11px] text-gray-200 leading-tight line-clamp-2">
                            {article.title}
                          </span>
                        )}
                        {snippet && (
                          <p className="text-[10px] text-gray-500 mt-0.5 line-clamp-1">{snippet}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 mt-1 text-[9px] text-gray-600">
                      {article.source && (
                        <span className="bg-gray-700/30 px-1.5 py-0.5 rounded">{article.source}</span>
                      )}
                      {publishedAgo && <span>{publishedAgo}</span>}
                      {article.score != null && (
                        <span className="text-gray-600">rel: {(article.score * 100).toFixed(0)}%</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Participants */}
        {participants.length > 0 && (
          <div>
            <SectionLabel icon={Users}>Participants</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {participants.map((p, i) => {
                const entityLinks = causalModel?.entity_links?.filter(
                  el => el.entity_name?.toLowerCase() === p.name?.toLowerCase()
                ) || [];
                return (
                  <div key={i} className="bg-gray-800/20 rounded px-2.5 py-1.5 border border-gray-700/10">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-medium text-gray-200">{p.name}</span>
                      {p.role && (
                        <span className="text-[9px] bg-gray-700/40 text-gray-500 px-1.5 py-0.5 rounded-full">
                          {p.role}
                        </span>
                      )}
                      {p.wikipedia_url && (
                        <a href={p.wikipedia_url} target="_blank" rel="noopener noreferrer"
                          className="ml-auto text-gray-600 hover:text-cyan-400 transition-colors">
                          <ExternalLink className="w-2.5 h-2.5" />
                        </a>
                      )}
                    </div>
                    {p.summary && (
                      <p className="text-[10px] text-gray-500 line-clamp-2 mt-0.5 leading-relaxed">{p.summary}</p>
                    )}
                    {entityLinks.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {entityLinks.map((el, j) => (
                          <span key={j} className="text-[8px] font-mono bg-cyan-500/8 text-cyan-400/60 border border-cyan-500/10 px-1 py-0.5 rounded" title={el.relationship || ''}>
                            {el.market_ticker}{el.relationship ? ` (${el.relationship})` : ''}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* More details toggle */}
        {hasExtras && (
          <>
            <button
              type="button"
              aria-expanded={showMore}
              onClick={() => setShowMore(!showMore)}
              className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
            >
              {showMore ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <span className="uppercase tracking-wider font-semibold">{showMore ? 'Less' : 'More'}</span>
            </button>

            {showMore && (
              <div className="space-y-3 pt-1">
                {key_figures.length > 0 && (
                  <div>
                    <SectionLabel icon={Users}>Key Figures</SectionLabel>
                    {key_figures.map((f, i) => (
                      <div key={i} className="flex items-baseline gap-2 text-[10px] mb-0.5">
                        <span className="text-gray-300">{f.name}</span>
                        {f.role && <span className="text-gray-600">({f.role})</span>}
                        {f.relevance && <span className="text-gray-500">- {f.relevance}</span>}
                      </div>
                    ))}
                  </div>
                )}

                {trading_considerations.length > 0 && (
                  <div>
                    <SectionLabel icon={AlertTriangle}>Trading Notes</SectionLabel>
                    {trading_considerations.map((c, i) => (
                      <div key={i} className="flex items-start gap-2 text-[11px] mb-1">
                        <span className="w-1 h-1 rounded-full bg-amber-500/40 flex-shrink-0 mt-[5px]" />
                        <span className="text-gray-400">{c}</span>
                      </div>
                    ))}
                  </div>
                )}

                {settlement_summary && (
                  <div>
                    <SectionLabel icon={FileText}>Settlement</SectionLabel>
                    <p className="text-[10px] text-gray-400">{settlement_summary}</p>
                    {resolution_source && (
                      <a href={resolution_source} target="_blank" rel="noopener noreferrer"
                        className="text-[10px] text-cyan-500/50 hover:text-cyan-400 flex items-center gap-0.5 mt-0.5 transition-colors">
                        Source <ExternalLink className="w-2.5 h-2.5" />
                      </a>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* Sources footer */}
        {wikipedia_urls.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-gray-800/10 text-[9px] text-gray-600">
            {wikipedia_urls.slice(0, 3).map((url, i) => (
              <a key={i} href={url} target="_blank" rel="noopener noreferrer"
                className="text-cyan-500/40 hover:text-cyan-400 underline transition-colors">
                Wikipedia{wikipedia_urls.length > 1 ? ` [${i + 1}]` : ''}
              </a>
            ))}
            {trading_summary && <span className="italic">LLM synthesized</span>}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(EventUnderstandingCard);
