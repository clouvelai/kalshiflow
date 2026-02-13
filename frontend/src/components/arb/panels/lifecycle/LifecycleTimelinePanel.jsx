import React, { useMemo } from 'react';
import { Sunrise } from 'lucide-react';

/**
 * LifecycleTimelinePanel - Vertical timeline showing lifecycle milestones.
 *
 * Shows:
 * - Early bird opportunities (lime, pulsing) from attention signals
 * - Upcoming opens (green) with countdown
 * - Active events (blue) grouped by category
 * - Closing soon (amber) within 1 hour
 * - Determined (gray)
 */
const LifecycleTimelinePanel = ({ timeline = [], trackedEvents = new Map(), attentionItems = [], compact = false }) => {
  // Extract early bird attention items
  const earlyBirdItems = useMemo(() => {
    return (attentionItems || []).filter(item => item.category === 'early_bird');
  }, [attentionItems]);

  // Group timeline items by type
  const grouped = useMemo(() => {
    const groups = {
      upcoming_open: [],
      active: [],
      closing_soon: [],
      determined: [],
    };

    timeline.forEach(item => {
      const type = item.type || 'active';
      if (groups[type]) {
        groups[type].push(item);
      } else {
        groups.active.push(item);
      }
    });

    return groups;
  }, [timeline]);

  const formatCountdown = (seconds) => {
    if (!seconds || seconds <= 0) return 'now';
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d`;
  };

  const categoryBadge = (category) => {
    const colors = {
      sports: 'bg-green-500/20 text-green-400',
      crypto: 'bg-purple-500/20 text-purple-400',
      politics: 'bg-blue-500/20 text-blue-400',
      economics: 'bg-yellow-500/20 text-yellow-400',
      entertainment: 'bg-pink-500/20 text-pink-400',
    };
    const colorClass = Object.entries(colors).find(([key]) =>
      (category || '').toLowerCase().includes(key)
    )?.[1] || 'bg-gray-500/20 text-gray-400';

    return (
      <span className={`text-[10px] px-1.5 py-0.5 rounded ${colorClass}`}>
        {(category || 'Other').split(' ')[0]}
      </span>
    );
  };

  const colorClasses = {
    green: {
      dot: 'bg-green-400',
      border: 'hover:border-green-500/30',
      text: 'text-green-400',
    },
    amber: {
      dot: 'bg-amber-400',
      border: 'hover:border-amber-500/30',
      text: 'text-amber-400',
    },
    blue: {
      dot: 'bg-blue-400',
      border: 'hover:border-blue-500/30',
      text: 'text-blue-400',
    },
    gray: {
      dot: 'bg-gray-400',
      border: 'hover:border-gray-500/30',
      text: 'text-gray-400',
    },
  };

  const TimelineItem = ({ item, color }) => {
    const cls = colorClasses[color] || colorClasses.gray;
    return (
      <div className={`flex items-start gap-2 p-2 rounded-lg bg-gray-800/50 border border-gray-700/50 ${cls.border} transition-colors`}>
        <div className={`mt-0.5 w-2 h-2 rounded-full ${cls.dot} flex-shrink-0`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            {categoryBadge(item.category)}
            <span className="text-xs text-gray-300 truncate">{item.title || item.event_ticker}</span>
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-gray-500">{item.market_count || 0} markets</span>
            {item.countdown_seconds != null && (
              <span className={`text-[10px] ${cls.text} font-mono`}>
                {formatCountdown(item.countdown_seconds)}
              </span>
            )}
            {item.mutually_exclusive && (
              <span className="text-[10px] text-gray-600">ME</span>
            )}
          </div>
        </div>
      </div>
    );
  };

  const hasItems = timeline.length > 0 || earlyBirdItems.length > 0;

  if (!hasItems) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        <p>No lifecycle events yet</p>
        <p className="text-xs mt-1">Markets will appear as they are discovered</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 p-2">
      {/* Early Bird Opportunities */}
      {earlyBirdItems.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5 px-1">
            <Sunrise className="w-3 h-3 text-lime-400" />
            <span className="text-[10px] uppercase tracking-wider text-lime-400 font-semibold">
              Early Bird ({earlyBirdItems.length})
            </span>
          </div>
          <div className="space-y-1.5">
            {earlyBirdItems.map((item, i) => (
              <EarlyBirdCard key={`${item.event_ticker}:${item.market_ticker}:${i}`} item={item} />
            ))}
          </div>
        </div>
      )}

      {/* Upcoming Opens */}
      {grouped.upcoming_open.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-green-500 mb-1.5 px-1 font-medium">
            Opening Soon ({grouped.upcoming_open.length})
          </div>
          <div className="space-y-1">
            {grouped.upcoming_open.map(item => (
              <TimelineItem key={item.event_ticker} item={item} color="green" />
            ))}
          </div>
        </div>
      )}

      {/* Closing Soon */}
      {grouped.closing_soon.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-amber-500 mb-1.5 px-1 font-medium">
            Closing Soon ({grouped.closing_soon.length})
          </div>
          <div className="space-y-1">
            {grouped.closing_soon.map(item => (
              <TimelineItem key={item.event_ticker} item={item} color="amber" />
            ))}
          </div>
        </div>
      )}

      {/* Active */}
      {grouped.active.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-blue-500 mb-1.5 px-1 font-medium">
            Active ({grouped.active.length})
          </div>
          <div className="space-y-1">
            {grouped.active.map(item => (
              <TimelineItem key={item.event_ticker} item={item} color="blue" />
            ))}
          </div>
        </div>
      )}

      {/* Determined */}
      {grouped.determined.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 px-1 font-medium">
            Determined ({grouped.determined.length})
          </div>
          <div className="space-y-1">
            {grouped.determined.slice(0, 5).map(item => (
              <TimelineItem key={item.event_ticker} item={item} color="gray" />
            ))}
          </div>
        </div>
      )}

      {/* Stats footer */}
      <div className="text-[10px] text-gray-600 px-1 pt-1 border-t border-gray-800">
        {trackedEvents.size} events tracked
      </div>
    </div>
  );
};

/**
 * ScoreBar - Mini horizontal progress bar for a score component.
 */
const ScoreBar = ({ label, value, max, color }) => {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[8px] text-gray-500 uppercase tracking-wider w-[42px] shrink-0 text-right">{label}</span>
      <div className="flex-1 h-[4px] bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[8px] font-mono text-gray-500 w-[18px] shrink-0 tabular-nums text-right">{Math.round(value)}</span>
    </div>
  );
};

/**
 * EarlyBirdCard - Expanded card for an early bird opportunity in the lifecycle view.
 * Shows market ticker, strategy, fair value, and full score breakdown with bars.
 */
const EarlyBirdCard = ({ item }) => {
  const data = item.data || {};
  const strategy = data.strategy || 'unknown';
  const fairValue = data.fair_value;
  const totalScore = data.early_bird_score;
  const marketTicker = item.market_ticker || '';

  const strategyConfig = {
    complement: { label: 'Complement Pricing', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    series: { label: 'Series Pattern', color: 'text-blue-400', bg: 'bg-blue-500/10' },
    news: { label: 'News Catalyst', color: 'text-orange-400', bg: 'bg-orange-500/10' },
    captain_decide: { label: 'Captain Evaluate', color: 'text-violet-400', bg: 'bg-violet-500/10' },
    unknown: { label: 'Evaluating', color: 'text-gray-400', bg: 'bg-gray-500/10' },
  };

  const strat = strategyConfig[strategy] || strategyConfig.unknown;

  // Score color based on total
  const scoreColor = totalScore >= 60 ? 'text-emerald-400' : totalScore >= 40 ? 'text-amber-400' : 'text-gray-400';

  return (
    <div className="rounded-lg bg-gray-800/40 border border-lime-500/15 overflow-hidden">
      {/* Header row */}
      <div className="flex items-center gap-2 px-2.5 py-1.5">
        <div className="w-2 h-2 rounded-full bg-lime-400 animate-pulse shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-gray-200 font-medium truncate">
              {item.event_ticker || marketTicker}
            </span>
            {marketTicker && item.event_ticker && marketTicker !== item.event_ticker && (
              <span className="text-[9px] font-mono text-gray-500 truncate">
                {marketTicker}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`text-[9px] font-semibold px-1.5 py-px rounded ${strat.bg} ${strat.color}`}>
              {strat.label}
            </span>
            {fairValue != null && (
              <span className="text-[10px] font-mono text-emerald-400">
                FV: {Math.round(fairValue)}c
              </span>
            )}
          </div>
        </div>
        {totalScore != null && (
          <div className="text-right shrink-0">
            <div className={`text-sm font-bold font-mono tabular-nums ${scoreColor}`}>
              {Math.round(totalScore)}
            </div>
            <div className="text-[7px] text-gray-600 uppercase tracking-wider">score</div>
          </div>
        )}
      </div>

      {/* Score breakdown bars */}
      <div className="px-2.5 pb-2 pt-0.5 space-y-0.5">
        <ScoreBar label="Comp" value={data.complement_score || 0} max={25} color="bg-emerald-500" />
        <ScoreBar label="News" value={data.news_score || 0} max={20} color="bg-orange-500" />
        <ScoreBar label="Cat" value={data.category_score || 0} max={15} color="bg-blue-500" />
        <ScoreBar label="Time" value={data.timing_score || 0} max={10} color="bg-violet-500" />
        <ScoreBar label="Risk" value={data.risk_score || 0} max={10} color="bg-teal-500" />
      </div>

      {/* Summary */}
      {item.summary && (
        <div className="px-2.5 pb-2 border-t border-gray-800/50">
          <p className="text-[9px] text-gray-500 mt-1">{item.summary}</p>
        </div>
      )}
    </div>
  );
};

export default LifecycleTimelinePanel;
