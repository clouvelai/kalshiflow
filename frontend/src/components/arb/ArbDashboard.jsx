import React, { useState } from 'react';
import { useArbWebSocket, useArbAgent, usePairIndex } from '../../hooks/arb';
import ArbHeader from './layout/ArbHeader';
import ArbMetricsBar from './layout/ArbMetricsBar';
import EventIndexPanel from './panels/EventIndexPanel';
import EventDetailsPanel from './panels/EventDetailsPanel';
import AgentChatPanel from './panels/AgentChatPanel';
import PositionPanel from './panels/PositionPanel';

/**
 * ArbDashboard - Main page for cross-venue arbitrage monitoring.
 */
const ArbDashboard = () => {
  const {
    connectionStatus,
    systemState,
    tradingState,
    spreads,
    arbTrades,
    agentMessages,
    pairIndex,
    eventCodex,
  } = useArbWebSocket();

  const {
    isRunning: agentIsRunning,
    currentSubagent,
    cycleCount,
    thinking,
    activeToolCall,
    toolCalls,
  } = useArbAgent(agentMessages);

  const mergedEvents = usePairIndex(pairIndex, spreads);

  const [selectedEventTicker, setSelectedEventTicker] = useState(null);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <ArbHeader
        connectionStatus={connectionStatus}
        systemState={systemState}
      />

      <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-6">
        <ArbMetricsBar
          tradingState={tradingState}
          spreads={spreads}
          arbTradeCount={arbTrades.length}
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
        />

        {/* Event Index + Event Details side by side */}
        <div className="grid grid-cols-5 gap-6" style={{ minHeight: '420px' }}>
          <div className="col-span-3">
            <EventIndexPanel
              events={mergedEvents}
              pairIndex={pairIndex}
              selectedEventTicker={selectedEventTicker}
              onSelectEvent={setSelectedEventTicker}
            />
          </div>
          <div className="col-span-2">
            <EventDetailsPanel
              selectedEventTicker={selectedEventTicker}
              eventCodex={eventCodex}
              pairIndex={pairIndex}
            />
          </div>
        </div>

        <PositionPanel tradingState={tradingState} />
      </div>
    </div>
  );
};

export default ArbDashboard;
