import React, { memo, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import DashboardOverview from './DashboardOverview';
import EventView from './EventView';

const CenterContent = memo(({
  selectedEventTicker, events, eventTrades, arbTrades,
  tradingState, mentionsState, onDeselectEvent,
}) => {
  const event = useMemo(() => {
    if (!selectedEventTicker || !events) return null;
    return events.get(selectedEventTicker) || null;
  }, [selectedEventTicker, events]);

  const positionsByTicker = useMemo(() => {
    const map = {};
    (tradingState?.positions || []).forEach(p => { map[p.ticker] = p; });
    return map;
  }, [tradingState?.positions]);

  const mentionsData = selectedEventTicker
    ? mentionsState?.[selectedEventTicker] || null
    : null;

  return (
    <div className="flex-1 min-w-0 flex flex-col overflow-hidden bg-gray-950/30">
      <AnimatePresence mode="wait">
        {event ? (
          <motion.div
            key={selectedEventTicker}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="flex-1 min-h-0 flex flex-col"
          >
            <EventView
              event={event}
              eventTrades={eventTrades}
              arbTrades={arbTrades}
              positionsByTicker={positionsByTicker}
              mentionsData={mentionsData}
              onClose={onDeselectEvent}
            />
          </motion.div>
        ) : (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="flex-1 min-h-0 overflow-y-auto"
          >
            <DashboardOverview
              tradingState={tradingState}
              arbTrades={arbTrades}
              events={events}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

CenterContent.displayName = 'CenterContent';

export default CenterContent;
