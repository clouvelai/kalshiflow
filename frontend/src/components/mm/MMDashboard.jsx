import React, { useState, useCallback } from 'react';
import { useMMWebSocket, useMMAgent } from '../../hooks/mm';
import MMHeader from './layout/MMHeader';
import MMCenterContent from './center/MMCenterContent';
import InventoryPanel from './panels/InventoryPanel';
import TradeLogPanel from './panels/TradeLogPanel';

/**
 * MMDashboard - Three-panel command center for Admiral market making.
 *
 * Layout: InventoryPanel (left) | Center (Orderbook/Agent/Performance tabs) | TradeLog (right)
 */
const MMDashboard = () => {
  const {
    connectionStatus,
    events,
    quoteState,
    inventory,
    agentMessages,
    tradingState,
    performance,
    tradeLog,
  } = useMMWebSocket();

  const {
    isRunning,
    cycleCount,
    cycleMode,
    thinking,
    toolCalls,
  } = useMMAgent(agentMessages);

  const [isPaused, setIsPaused] = useState(false);

  const handlePauseToggle = useCallback(() => {
    setIsPaused(p => !p);
    // TODO: Send pause/resume command via WS or REST
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-white overflow-hidden">
      <MMHeader
        connectionStatus={connectionStatus}
        quoteState={quoteState}
        performance={performance}
        onPauseToggle={handlePauseToggle}
        isPaused={isPaused}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left: Inventory */}
        <div className="w-72 shrink-0 border-r border-gray-800/30 overflow-hidden">
          <InventoryPanel inventory={inventory} />
        </div>

        {/* Center: Orderbook / Agent / Performance */}
        <MMCenterContent
          events={events}
          quoteState={quoteState}
          isRunning={isRunning}
          cycleCount={cycleCount}
          cycleMode={cycleMode}
          thinking={thinking}
          toolCalls={toolCalls}
          performance={performance}
        />

        {/* Right: Trade Log */}
        <div className="w-80 shrink-0 border-l border-gray-800/30 overflow-hidden">
          <TradeLogPanel tradeLog={tradeLog} />
        </div>
      </div>
    </div>
  );
};

export default MMDashboard;
