import React, { memo, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { BarChart3, Brain } from 'lucide-react';
import DashboardOverview from './DashboardOverview';
import EventView from './EventView';
import AgentPanel from './AgentPanel';
import StartupProgress from './StartupProgress';

const MAIN_TABS = [
  { id: 'event', label: 'Event', icon: BarChart3 },
  { id: 'agent', label: 'Agent', icon: Brain },
];

const CenterContent = memo(({
  selectedEventTicker, events, eventTrades, arbTrades,
  tradingState, onDeselectEvent,
  // Agent props
  activeMainTab, setActiveMainTab,
  isRunning, cycleCount,
  thinking, activeToolCall, toolCalls,
  todos, memoryOps,
  sniperState, captainPaused, exchangeStatus, feedStats,
  // Startup props
  connectionStatus, systemState, startupMessages,
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

  const isStartingUp = (systemState === 'initializing' || systemState === 'startup' || events.size === 0)
    && connectionStatus === 'connected';

  return (
    <div className="flex-1 min-w-0 flex flex-col overflow-hidden bg-gray-950/30">
      {/* Top-level tab bar */}
      <div className="flex items-center gap-1 px-3 py-1.5 border-b border-gray-800/40 shrink-0 bg-gray-950/50">
        {MAIN_TABS.map(tab => {
          const isActive = activeMainTab === tab.id;
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveMainTab(tab.id)}
              className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-semibold tracking-wide transition-colors ${
                isActive
                  ? 'bg-gray-800/60 text-gray-100'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/25'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
              {/* Pulsing dot on Agent tab when Captain is running and user is on Event tab */}
              {tab.id === 'agent' && isRunning && !isActive && (
                <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-violet-500" />
                </span>
              )}
              {/* Static dot on Agent tab when active and running */}
              {tab.id === 'agent' && isRunning && isActive && (
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 ml-0.5" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        {activeMainTab === 'agent' ? (
          <motion.div
            key="agent"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="flex-1 min-h-0 flex flex-col"
          >
            <AgentPanel
              isRunning={isRunning}
              cycleCount={cycleCount}
              thinking={thinking}
              activeToolCall={activeToolCall}
              toolCalls={toolCalls}
              todos={todos}
              memoryOps={memoryOps}
              sniperState={sniperState}
              captainPaused={captainPaused}
              exchangeStatus={exchangeStatus}
              feedStats={feedStats}
            />
          </motion.div>
        ) : isStartingUp && !event ? (
          <motion.div
            key="startup"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="flex-1 min-h-0 flex flex-col"
          >
            <StartupProgress messages={startupMessages} />
          </motion.div>
        ) : event ? (
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
