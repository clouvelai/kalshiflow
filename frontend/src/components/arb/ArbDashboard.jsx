import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useArbWebSocket, useArbAgent } from '../../hooks/arb';
import ArbHeader from './layout/ArbHeader';
import LeftSidebar from './layout/LeftSidebar';
import CenterContent from './center/CenterContent';
import OrdersSidebar from './panels/OrdersDrawer';

/**
 * ArbDashboard - Three-panel command center for Captain agent trading.
 *
 * Layout: LeftSidebar (events + portfolio) | Center (Event/Agent tabs) | OrdersSidebar (trades/fills)
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
    sniperState,
    startupMessages,
    accountHealth,
    attentionItems,
    attentionStats,
    autoActions,
    captainMode,
    captainTiming,
    discoveryState,
    gatewayConfig,
    tavilyBudget,
    lifecycleTimeline,
    trackedEvents,
    mmSnapshot,
    mmQuoteState,
    mmInventory,
    mmTradeLog,
    mmPerformance,
  } = useArbWebSocket();

  const {
    isRunning: agentIsRunning,
    cycleCount,
    cycleMode,
    thinking,
    activeToolCall,
    toolCalls,
    todos,
    memoryOps,
  } = useArbAgent(agentMessages);

  const [selectedEventTicker, setSelectedEventTicker] = useState(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [activeMainTab, setActiveMainTab] = useState('agent');

  const handleDeselectEvent = useCallback(() => setSelectedEventTicker(null), []);

  // Clear selection if selected event was evicted (no longer in events Map)
  useEffect(() => {
    if (selectedEventTicker && events && events.size > 0 && !events.has(selectedEventTicker)) {
      setSelectedEventTicker(null);
    }
  }, [selectedEventTicker, events]);

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
        gatewayConfig={gatewayConfig}
        tavilyBudget={tavilyBudget}
        mmQuoteState={mmQuoteState}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        <LeftSidebar
          collapsed={leftCollapsed}
          onToggle={() => setLeftCollapsed(c => !c)}
          tradingState={tradingState}
          events={eventList}
          selectedEventTicker={selectedEventTicker}
          onSelectEvent={setSelectedEventTicker}
          gatewayConfig={gatewayConfig}
        />

        <CenterContent
          selectedEventTicker={selectedEventTicker}
          events={events}
          eventTrades={eventTrades}
          arbTrades={arbTrades}
          tradingState={tradingState}
          onDeselectEvent={handleDeselectEvent}
          // Agent tab props
          activeMainTab={activeMainTab}
          setActiveMainTab={setActiveMainTab}
          isRunning={agentIsRunning}
          cycleCount={cycleCount}
          cycleMode={cycleMode}
          thinking={thinking}
          activeToolCall={activeToolCall}
          toolCalls={toolCalls}
          todos={todos}
          memoryOps={memoryOps}
          sniperState={sniperState}
          captainPaused={captainPaused}
          exchangeStatus={exchangeStatus}
          feedStats={feedStats}
          attentionItems={attentionItems}
          attentionStats={attentionStats}
          autoActions={autoActions}
          captainMode={captainMode}
          captainTiming={captainTiming}
          discoveryState={discoveryState}
          connectionStatus={connectionStatus}
          systemState={systemState}
          startupMessages={startupMessages}
          lifecycleTimeline={lifecycleTimeline}
          trackedEvents={trackedEvents}
          mmSnapshot={mmSnapshot}
          mmQuoteState={mmQuoteState}
          mmInventory={mmInventory}
          mmTradeLog={mmTradeLog}
          mmPerformance={mmPerformance}
        />

        <OrdersSidebar
          trades={arbTrades}
          tradingState={tradingState}
          collapsed={rightCollapsed}
          onToggle={() => setRightCollapsed(c => !c)}
          sniperOrderIds={sniperState?.recentActions?.flatMap(a => a.order_ids || []) || []}
          accountHealth={accountHealth}
        />
      </div>
    </div>
  );
};

export default ArbDashboard;
