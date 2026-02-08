import React, { memo } from 'react';
import { Layers } from 'lucide-react';
import EventListItem from './EventListItem';

const EventList = memo(({ events, selectedEventTicker, onSelectEvent }) => {
  const totalEvents = events?.length || 0;

  return (
    <div className="flex-1 min-h-0 flex flex-col" data-testid="event-list">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800/30 shrink-0">
        <Layers className="w-3.5 h-3.5 text-cyan-400/70" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Events</span>
        <span className="text-[9px] font-mono text-gray-600 ml-auto tabular-nums">{totalEvents}</span>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0 py-1">
        {totalEvents === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-600">
            <Layers className="w-5 h-5 mb-1.5 opacity-30" />
            <span className="text-[10px]">No events</span>
          </div>
        ) : (
          events.map(event => (
            <EventListItem
              key={event.event_ticker}
              event={event}
              isSelected={event.event_ticker === selectedEventTicker}
              onSelect={onSelectEvent}
            />
          ))
        )}
      </div>
    </div>
  );
});

EventList.displayName = 'EventList';

export default EventList;
