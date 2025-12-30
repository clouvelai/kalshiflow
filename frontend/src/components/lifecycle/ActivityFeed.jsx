import React, { useMemo } from 'react';

/**
 * ActivityFeed - Smart aggregated events feed
 *
 * Rules:
 * - Aggregate same-type events within 30s window
 * - NEVER collapse determined/settled events (critical)
 * - Time-grouped sections (NOW, 2 MIN AGO, etc.)
 */
const ActivityFeed = ({ events, onClear }) => {
  // Aggregate and group events
  const groupedEvents = useMemo(() => {
    if (!events || events.length === 0) return [];

    const now = Date.now();
    const groups = [];
    let currentGroup = null;

    // Sort by timestamp descending (newest first)
    const sorted = [...events].sort((a, b) => {
      const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return timeB - timeA;
    });

    sorted.forEach((event, idx) => {
      const eventTime = event.timestamp ? new Date(event.timestamp).getTime() : now;
      const timeSince = now - eventTime;
      const timeLabel = getTimeLabel(timeSince);

      // Critical events never aggregate
      const isCritical = ['determined', 'settled', 'finalized'].includes(event.event_type);

      // Check if we should aggregate with previous event
      const prevEvent = sorted[idx - 1];
      const canAggregate = !isCritical &&
        prevEvent &&
        prevEvent.event_type === event.event_type &&
        prevEvent.action === event.action &&
        Math.abs(eventTime - new Date(prevEvent.timestamp).getTime()) < 30000;

      if (!canAggregate) {
        // Start new group or add as individual
        if (!currentGroup || currentGroup.timeLabel !== timeLabel) {
          if (currentGroup) groups.push(currentGroup);
          currentGroup = { timeLabel, events: [] };
        }

        if (isCritical) {
          // Critical events shown individually
          currentGroup.events.push({
            type: 'single',
            event,
            id: event.id
          });
        } else {
          // Start potential aggregation
          currentGroup.events.push({
            type: 'aggregate',
            eventType: event.event_type,
            action: event.action,
            tickers: [event.market_ticker],
            category: event.metadata?.category,
            id: event.id
          });
        }
      } else {
        // Aggregate with previous
        const lastItem = currentGroup.events[currentGroup.events.length - 1];
        if (lastItem && lastItem.type === 'aggregate') {
          lastItem.tickers.push(event.market_ticker);
        }
      }
    });

    if (currentGroup) groups.push(currentGroup);

    return groups;
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="bg-gray-900/50 rounded-lg border border-gray-800 h-full min-h-96">
        <div className="p-3 border-b border-gray-800 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-300">Activity Feed</h3>
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
      <div className="p-3 border-b border-gray-800 flex items-center justify-between flex-shrink-0">
        <h3 className="text-sm font-medium text-gray-300">Activity Feed</h3>
        <button
          onClick={onClear}
          className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Feed content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {groupedEvents.map((group, gIdx) => (
          <div key={gIdx}>
            {/* Time label */}
            <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">
              {group.timeLabel}
            </div>

            {/* Events in this time group */}
            <div className="space-y-2">
              {group.events.map((item, iIdx) => (
                <EventItem key={item.id || `${gIdx}-${iIdx}`} item={item} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * EventItem - Renders a single or aggregated event
 */
const EventItem = ({ item }) => {
  if (item.type === 'single') {
    const event = item.event;
    const isCritical = ['determined', 'settled', 'finalized'].includes(event.event_type);

    return (
      <div className={`text-sm p-2 rounded ${
        isCritical ? 'bg-amber-900/20 border border-amber-500/30' : 'bg-gray-800/50'
      }`}>
        <div className="flex items-center gap-2">
          {/* Event icon */}
          <span className={getEventIcon(event.event_type)} />

          {/* Event description */}
          <span className={isCritical ? 'text-amber-300' : 'text-gray-300'}>
            {formatSingleEvent(event)}
          </span>
        </div>

        {/* Ticker */}
        <div className="text-xs text-gray-500 mt-1 font-mono">
          {event.market_ticker}
        </div>
      </div>
    );
  }

  // Aggregated event
  const count = item.tickers.length;

  return (
    <div className="text-sm p-2 rounded bg-gray-800/50">
      <div className="flex items-center gap-2">
        <span className="text-blue-400">+{count}</span>
        <span className="text-gray-300">
          {formatAggregateEvent(item.action, count, item.category)}
        </span>
      </div>

      {/* Show first few tickers if small count */}
      {count <= 3 && (
        <div className="text-xs text-gray-500 mt-1 font-mono">
          {item.tickers.join(', ')}
        </div>
      )}
    </div>
  );
};

// Helper functions
function getTimeLabel(ms) {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 30) return 'NOW';
  if (seconds < 60) return '< 1 MIN AGO';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 2) return '1 MIN AGO';
  if (minutes < 5) return `${minutes} MIN AGO`;
  if (minutes < 10) return '5 MIN AGO';
  if (minutes < 30) return '10+ MIN AGO';
  return '30+ MIN AGO';
}

function getEventIcon(eventType) {
  switch (eventType) {
    case 'created':
    case 'tracked':
      return 'text-blue-400';
    case 'determined':
      return 'text-amber-400';
    case 'settled':
    case 'finalized':
      return 'text-emerald-400';
    case 'status_change':
      return 'text-purple-400';
    default:
      return 'text-gray-400';
  }
}

function formatSingleEvent(event) {
  switch (event.event_type) {
    case 'determined':
      return `Market determined: ${event.metadata?.result || 'result pending'}`;
    case 'settled':
      return `Market settled`;
    case 'finalized':
      return `Market finalized`;
    case 'status_change':
      return `Status: ${event.metadata?.old_status} → ${event.metadata?.new_status}`;
    default:
      return event.action || event.event_type;
  }
}

function formatAggregateEvent(action, count, category) {
  if (category) {
    return `${category} markets ${action || 'tracked'}`;
  }
  return `markets ${action || 'tracked'}`;
}

export default ActivityFeed;
