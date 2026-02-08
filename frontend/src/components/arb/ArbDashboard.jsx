import React, { useState, useMemo, useCallback } from 'react';
import { useArbWebSocket, useArbAgent } from '../../hooks/arb';
import ArbHeader from './layout/ArbHeader';
import LeftSidebar from './layout/LeftSidebar';
import RightSidebar from './layout/RightSidebar';
import CenterContent from './center/CenterContent';

/**
 * ArbDashboard - Three-panel command center for Captain agent trading.
 *
 * Layout: LeftSidebar (events + portfolio) | Center (dashboard/event detail) | RightSidebar (agent)
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
    feedStats,
    captainPaused,
    sendCaptainPauseToggle,
    exchangeStatus,
    mentionsState,
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
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);

  const handleDeselectEvent = useCallback(() => setSelectedEventTicker(null), []);

  // Convert events Map to sorted array for left sidebar
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
        event_type: event.event_type,
        understanding: event.understanding,
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
    }).sort((a, b) => {
      // Sort by best edge descending
      const edgeA = Math.max(a.long_edge || 0, a.short_edge || 0);
      const edgeB = Math.max(b.long_edge || 0, b.short_edge || 0);
      return edgeB - edgeA;
    });
  }, [events]);

  return (
    <div id="arb-dashboard" data-testid="arb-dashboard" className="h-screen flex flex-col bg-gray-950 text-white overflow-hidden">
      <ArbHeader
        connectionStatus={connectionStatus}
        systemState={systemState}
        feedStats={feedStats}
        captainPaused={captainPaused}
        onCaptainPauseToggle={sendCaptainPauseToggle}
        exchangeStatus={exchangeStatus}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        <LeftSidebar
          collapsed={leftCollapsed}
          onToggle={() => setLeftCollapsed(c => !c)}
          tradingState={tradingState}
          events={eventList}
          selectedEventTicker={selectedEventTicker}
          onSelectEvent={setSelectedEventTicker}
        />

        <CenterContent
          selectedEventTicker={selectedEventTicker}
          events={events}
          eventTrades={eventTrades}
          arbTrades={arbTrades}
          tradingState={tradingState}
          mentionsState={mentionsState}
          onDeselectEvent={handleDeselectEvent}
        />

        <RightSidebar
          collapsed={rightCollapsed}
          onToggle={() => setRightCollapsed(c => !c)}
          isRunning={agentIsRunning}
          currentSubagent={currentSubagent}
          cycleCount={cycleCount}
          thinking={thinking}
          activeToolCall={activeToolCall}
          toolCalls={toolCalls}
          todos={todos}
          memoryOps={memoryOps}
          commandoSessions={commandoSessions}
          arbTrades={arbTrades}
        />
      </div>
    </div>
  );
};

export default ArbDashboard;
