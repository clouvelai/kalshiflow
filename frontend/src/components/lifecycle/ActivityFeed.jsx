import React, { useState, useMemo } from 'react';

/**
 * ActivityFeed - Professional log-style events feed
 *
 * Format: [timestamp] [event_type] message
 * Clean, minimal design for easy scanning.
 */
const ActivityFeed = ({ events, onClear, upcomingMarkets = [], rlmStates = {} }) => {
  // Collapse state for upcoming section
  const [upcomingExpanded, setUpcomingExpanded] = useState(false);

  // Calculate tracked trades summary from RLM states
  const { totalTrades, marketsWithTrades } = useMemo(() => {
    const states = Object.values(rlmStates || {});
    const total = states.reduce((sum, state) => sum + (state?.total_trades || 0), 0);
    const withTrades = states.filter(state => (state?.total_trades || 0) > 0).length;
    return { totalTrades: total, marketsWithTrades: withTrades };
  }, [rlmStates]);

  // Process events into display format (no complex aggregation - just simple list)
  const processedEvents = useMemo(() => {
    if (!events || events.length === 0) return [];

    // Sort by timestamp descending (newest first)
    return [...events].sort((a, b) => {
      const timeA = a.timestamp ? new Date(`1970-01-01 ${a.timestamp}`).getTime() : 0;
      const timeB = b.timestamp ? new Date(`1970-01-01 ${b.timestamp}`).getTime() : 0;
      return timeB - timeA;
    });
  }, [events]);

  // Deduplicate upcoming markets by ticker
  const uniqueUpcoming = useMemo(() => {
    const seen = new Set();
    return (upcomingMarkets || []).filter(m => {
      if (seen.has(m.ticker)) return false;
      seen.add(m.ticker);
      return true;
    });
  }, [upcomingMarkets]);

  if (events.length === 0) {
    return (
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 h-full min-h-96">
        <div className="p-3 border-b border-gray-800">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-300">Activity Feed</h3>
          </div>
          {/* Tracked Trades Summary */}
          {totalTrades > 0 && (
            <div className="mt-2 text-xs text-gray-400">
              <span className="text-emerald-400 font-medium">{totalTrades}</span> trades across{' '}
              <span className="text-blue-400 font-medium">{marketsWithTrades}</span> markets
            </div>
          )}
        </div>
        <div className="p-4 text-center text-gray-500 text-sm">
          <p>No activity yet</p>
          <p className="text-xs mt-1 text-gray-600">
            Events will appear as markets are tracked
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800 h-full min-h-96 flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-300">Activity Feed</h3>
          <button
            onClick={onClear}
            className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
          >
            Clear
          </button>
        </div>
        {/* Tracked Trades Summary */}
        {totalTrades > 0 && (
          <div className="mt-2 text-xs text-gray-400">
            <span className="text-emerald-400 font-medium">{totalTrades}</span> trades across{' '}
            <span className="text-blue-400 font-medium">{marketsWithTrades}</span> markets
          </div>
        )}
      </div>

      {/* Feed content */}
      <div className="flex-1 overflow-y-auto">
        {/* Activity events - log style (now at top) */}
        <div className="divide-y divide-gray-800/50">
          {processedEvents.map((event, idx) => (
            <EventRow key={event.id || `event-${idx}`} event={event} />
          ))}
        </div>

        {/* Upcoming markets section (moved to bottom, collapsible) */}
        {uniqueUpcoming.length > 0 && (
          <UpcomingSection
            markets={uniqueUpcoming}
            expanded={upcomingExpanded}
            onToggle={() => setUpcomingExpanded(!upcomingExpanded)}
          />
        )}
      </div>
    </div>
  );
};

/**
 * EventRow - Single event row in log format
 * [timestamp] [TYPE] message
 */
const EventRow = ({ event }) => {
  const { typeLabel, typeColor, bgColor } = getEventStyle(event.event_type, event);
  const message = formatEventMessage(event);

  return (
    <div className={`px-3 py-2 text-xs font-mono ${bgColor} hover:bg-gray-800/30 transition-colors`}>
      <div className="flex items-start gap-2">
        {/* Timestamp */}
        <span className="text-gray-500 flex-shrink-0 w-16">
          {event.timestamp || '--:--:--'}
        </span>

        {/* Event type badge */}
        <span className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${typeColor}`}>
          {typeLabel}
        </span>

        {/* Message */}
        <span className="text-gray-300 flex-1 break-words">
          {message}
        </span>
      </div>

      {/* Ticker on second line if present */}
      {event.market_ticker && (
        <div className="mt-1 ml-[72px] text-gray-500 text-[10px]">
          {event.market_ticker}
        </div>
      )}
    </div>
  );
};

/**
 * Get style config for event type
 * @param {string} eventType - The event type string
 * @param {object} event - The full event object (for metadata access)
 */
function getEventStyle(eventType, event = {}) {
  switch (eventType) {
    case 'startup':
      return {
        typeLabel: 'INIT',
        typeColor: 'bg-cyan-500/20 text-cyan-400',
        bgColor: 'bg-cyan-900/5'
      };
    case 'tracked':
    case 'created':
      return {
        typeLabel: 'TRACK',
        typeColor: 'bg-blue-500/20 text-blue-400',
        bgColor: ''
      };
    case 'determined':
      return {
        typeLabel: 'END',
        typeColor: 'bg-amber-500/20 text-amber-400',
        bgColor: 'bg-amber-900/5'
      };
    case 'settled':
    case 'finalized':
      return {
        typeLabel: 'SETTLE',
        typeColor: 'bg-emerald-500/20 text-emerald-400',
        bgColor: 'bg-emerald-900/5'
      };
    case 'status_change':
      return {
        typeLabel: 'STATUS',
        typeColor: 'bg-purple-500/20 text-purple-400',
        bgColor: ''
      };
    case 'closed':
      return {
        typeLabel: 'CLOSE',
        typeColor: 'bg-red-500/20 text-red-400',
        bgColor: 'bg-red-900/5'
      };
    // Trading activity events - RLM signal with spread awareness
    case 'rlm_signal': {
      const spreadPassed = event.metadata?.spread_passed ?? true;
      const status = event.metadata?.status;
      const isBlocked = status === 'skipped_spread' || spreadPassed === false;

      return {
        typeLabel: isBlocked ? 'BLOCKED' : 'SIGNAL',
        typeColor: isBlocked
          ? 'bg-red-500/25 text-red-400 border border-red-500/30'
          : 'bg-amber-500/20 text-amber-400',
        bgColor: isBlocked
          ? 'bg-red-900/10'
          : 'bg-amber-900/5'
      };
    }
    case 'order_fill':
      return {
        typeLabel: 'FILL',
        typeColor: 'bg-emerald-500/20 text-emerald-400',
        bgColor: 'bg-emerald-900/5'
      };
    case 'order_placed':
      return {
        typeLabel: 'ORDER',
        typeColor: 'bg-blue-500/20 text-blue-400',
        bgColor: 'bg-blue-900/5'
      };
    // Event exposure alerts (correlated positions across related markets)
    case 'event_alert': {
      const level = event.metadata?.level || 'info';
      if (level === 'error') {
        return {
          typeLabel: 'LOSS',
          typeColor: 'bg-red-500/25 text-red-400 border border-red-500/30',
          bgColor: 'bg-red-900/10'
        };
      } else if (level === 'warning') {
        return {
          typeLabel: 'RISK',
          typeColor: 'bg-amber-500/20 text-amber-400',
          bgColor: 'bg-amber-900/5'
        };
      } else {
        // info = arbitrage opportunity
        return {
          typeLabel: 'ARB',
          typeColor: 'bg-emerald-500/20 text-emerald-400',
          bgColor: 'bg-emerald-900/5'
        };
      }
    }
    default:
      return {
        typeLabel: eventType?.toUpperCase()?.slice(0, 6) || 'EVENT',
        typeColor: 'bg-gray-500/20 text-gray-400',
        bgColor: ''
      };
  }
}

/**
 * Format event message based on type
 */
function formatEventMessage(event) {
  switch (event.event_type) {
    case 'startup':
      const count = event.metadata?.count || 0;
      return `Loaded ${count} tracked markets`;
    case 'determined':
      return `Market determined: ${event.metadata?.result || 'result pending'}`;
    case 'settled':
      return 'Market settled';
    case 'finalized':
      return 'Market finalized';
    case 'status_change':
      return `${event.metadata?.old_status || '?'} -> ${event.metadata?.new_status || '?'}`;
    case 'tracked':
    case 'created':
      if (event.metadata?.title) {
        // Truncate long titles
        const title = event.metadata.title;
        return title.length > 50 ? title.slice(0, 47) + '...' : title;
      }
      return event.action || 'Market tracked';
    case 'closed':
      return event.metadata?.reason || 'Market closed';
    // Trading activity events - RLM signal with spread info
    case 'rlm_signal': {
      const meta = event.metadata || {};
      const yesRatio = meta.yes_ratio;
      const priceDrop = meta.price_drop;
      const currentSpread = meta.current_spread;
      const maxSpread = meta.max_spread;
      const spreadPassed = meta.spread_passed ?? true;
      const ticker = meta.market_ticker || '';

      // Build formatted message with spread tier info
      let msg = '';

      // Short ticker (last part after last dash)
      const shortTicker = ticker.split('-').slice(-1)[0] || ticker.slice(-12);

      if (yesRatio !== undefined && priceDrop !== undefined) {
        msg = `${shortTicker} ${Math.round(yesRatio * 100)}% YES ${priceDrop}c drop`;
      } else {
        msg = shortTicker || 'Signal';
      }

      // Add spread tier info if available
      if (currentSpread !== undefined && maxSpread !== undefined) {
        const spreadIcon = spreadPassed ? '\u2713' : '\u2717';  // checkmark or X
        msg += ` [${currentSpread}c/${maxSpread}c ${spreadIcon}]`;
      }

      return msg;
    }
    case 'order_fill':
      return event.reason || event.metadata?.message || 'Order filled';
    case 'order_placed':
      return event.reason || event.metadata?.message || 'Order placed';
    // Event exposure alerts (correlated positions across related markets)
    case 'event_alert': {
      // Use the reason/message field which contains the formatted alert message
      return event.reason || event.metadata?.message || 'Event exposure alert';
    }
    default:
      return event.action || event.metadata?.message || event.event_type || 'Event';
  }
}

/**
 * Format countdown from seconds to human-readable string
 */
function formatCountdown(seconds) {
  if (!seconds || seconds <= 0) return 'NOW';

  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);

  if (hours > 0) {
    return `${hours}h ${mins}m`;
  }
  return `${mins}m`;
}

/**
 * UpcomingSection - Collapsible upcoming markets section
 * Shows markets opening within 4 hours, collapsed by default
 */
const UpcomingSection = ({ markets, expanded, onToggle }) => {
  if (!markets || markets.length === 0) return null;

  const PREVIEW_COUNT = 5;
  const displayedMarkets = expanded ? markets : markets.slice(0, PREVIEW_COUNT);
  const hasMore = markets.length > PREVIEW_COUNT;

  return (
    <div className="border-t border-gray-800 bg-amber-900/5">
      {/* Collapsible header */}
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-amber-900/10 transition-colors"
      >
        <div className="flex items-center gap-2">
          {/* Chevron */}
          <span className={`text-amber-400 transition-transform ${expanded ? 'rotate-90' : ''}`}>
            {'\u25B6'}
          </span>
          <span className="text-[10px] text-amber-400 uppercase tracking-wide font-semibold">
            UPCOMING
          </span>
          <span className="text-[10px] text-gray-500">({markets.length})</span>
        </div>
        {!expanded && hasMore && (
          <span className="text-[10px] text-gray-500">Click to expand</span>
        )}
      </button>

      {/* Content (collapsed shows preview, expanded shows all) */}
      {(expanded || PREVIEW_COUNT > 0) && (
        <div className="px-3 pb-2 space-y-1">
          {displayedMarkets.map((market) => (
            <UpcomingMarketItem key={market.ticker} market={market} />
          ))}

          {/* Show more/less button */}
          {hasMore && (
            <button
              onClick={onToggle}
              className="text-[10px] text-amber-400 hover:text-amber-300 mt-1"
            >
              {expanded ? 'Show less' : `Show all ${markets.length}`}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * UpcomingMarketItem - Single upcoming market row
 */
const UpcomingMarketItem = ({ market }) => {
  const countdown = formatCountdown(market.countdown_seconds);

  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="text-amber-400 w-12 flex-shrink-0">{countdown}</span>
      <span className="text-gray-400 truncate flex-1">{market.title}</span>
      <span className="text-gray-600 flex-shrink-0">{market.category}</span>
    </div>
  );
};

export default ActivityFeed;
