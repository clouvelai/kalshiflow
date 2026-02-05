import React, { useState, useMemo } from 'react';
import { useArbWebSocket, useArbAgent } from '../../hooks/arb';
import ArbHeader from './layout/ArbHeader';
import ArbMetricsBar from './layout/ArbMetricsBar';
import EventIndexPanel from './panels/EventIndexPanel';
import EventDetailsPanel from './panels/EventDetailsPanel';
import AgentChatPanel from './panels/AgentChatPanel';
import PositionPanel from './panels/PositionPanel';

/**
 * ArbDashboard - Single-event arbitrage trading console.
 *
 * Monitors mutually exclusive events for probability completeness violations.
 */
const ArbDashboard = () => {
  const {
    connectionStatus,
    systemState,
    tradingState,
    arbTrades,
    agentMessages,
    events,
    eventTrades,
  } = useArbWebSocket();

  const {
    isRunning: agentIsRunning,
    currentSubagent,
    cycleCount,
    thinking,
    activeToolCall,
    toolCalls,
    todos,
    memoryOps,
    commandoSessions,
  } = useArbAgent(agentMessages);

  const [selectedEventTicker, setSelectedEventTicker] = useState(null);

  // Convert events Map to sorted array for EventIndexPanel
  const eventList = useMemo(() => {
    if (!events || events.size === 0) return [];
    return Array.from(events.values()).map(event => {
      const markets = event.markets ? Object.values(event.markets) : [];
      return {
        event_ticker: event.event_ticker,
        title: event.title,
        category: event.category,
        series_ticker: event.series_ticker,
        mutually_exclusive: event.mutually_exclusive,
        market_count: event.markets_total || markets.length,
        markets_with_data: event.markets_with_data || 0,
        all_markets_have_data: event.all_markets_have_data || false,
        sum_yes_bid: event.sum_yes_bid,
        sum_yes_ask: event.sum_yes_ask,
        sum_yes_mid: event.sum_yes_mid,
        long_edge: event.long_edge,
        short_edge: event.short_edge,
        loaded_at: event.loaded_at,
        updated_at: event.updated_at,
        markets,
      };
    });
  }, [events]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <ArbHeader
        connectionStatus={connectionStatus}
        systemState={systemState}
      />

      <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-6">
        <ArbMetricsBar
          tradingState={tradingState}
          arbTradeCount={arbTrades.length}
          events={events}
        />

        {/* Agent Chat - full width */}
        <AgentChatPanel
          thinking={thinking}
          activeToolCall={activeToolCall}
          toolCalls={toolCalls}
          arbTrades={arbTrades}
          isRunning={agentIsRunning}
          currentSubagent={currentSubagent}
          cycleCount={cycleCount}
          todos={todos}
          memoryOps={memoryOps}
          commandoSessions={commandoSessions}
        />

        {/* Event Index + Event Details side by side */}
        <div className="grid grid-cols-5 gap-6" style={{ minHeight: '420px' }}>
          <div className="col-span-3">
            <EventIndexPanel
              events={eventList}
              selectedEventTicker={selectedEventTicker}
              onSelectEvent={setSelectedEventTicker}
            />
          </div>
          <div className="col-span-2">
            <EventDetailsPanel
              selectedEventTicker={selectedEventTicker}
              events={events}
              eventTrades={eventTrades}
            />
          </div>
        </div>

        <PositionPanel tradingState={tradingState} />
      </div>
    </div>
  );
};

export default ArbDashboard;
